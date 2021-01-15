# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the Neural Clustering Process (NCP) base model."""

import numpy as np
import tensorflow as tf


class NCPBase(tf.keras.Model):
  r"""Neural Clustering Process (NCP) base model implementing one NCP iteration.

  The model and notations are based on Algorithm 1 from the NCP paper
  https://arxiv.org/abs/1901.00409. In amortized probabilistic clustering, the
  goal is to learn a parameterized function $ q_{\theta}(c|x) $ to approximate
  the clustering posterior $ p(c|x) $ , where `x` represents the observations
  and `c` represents the cluster labels. NCP does this using a sequential
  point-wise expansion of $ p(c|x) $:
  $$
    p(c|x) = p(c_1|x_{1:N})p(c_2|c_1, x_{1:N})...p(c_N|c_{1:N-1}, x_{1:N})
  $$
  NCP handles a variable number of clusters by dynamically growing neural
  network representations and a variable-input softmax function.
  Permutation invariant representations are used to preserve the permutation
  symmetries within and between clusters.

  This class implements the forward pass for one iteration of the NCP model,
  $ q_{\theta}(c_i|c_{1:i-1}, x_{1:N}) $, which assigns a cluster label to
  the `i`-th point given the context of all points and the labels already
  assigned in previous iterations. Within this iteration, it is possible
  for the `i`-th point to join one of the `K` existing clusters, or create a new
  cluster (i.e. joining an empty cluster). Making this decision requires
  evaluating the `K+1` hypotheses, which has been parallelized as much as
  possible. Computing the cluster labels for `N` points requires `N` iterations.

  See `NCPWrapper` as an example of how to use the methods in this class.

  Attributes:
    assigned_point_layer: A `tf.keras.Model` or `tf.keras.layers.Layer` for the
      `h` function of NCP, which transforms each assigned point `x_i` to `h_i`,
      in order to build representations of existing clusters (`H_k`) from
      assigned points.
    unassigned_point_layer: A `tf.keras.Model` or `tf.keras.layers.Layer` for
      the `u` function of NCP, which transforms each unassigned point `x_i` to
      `u_i`, in order to build the global representation of unassigned points
      (`U`).
    cluster_layer: A `tf.keras.Model` or `tf.keras.layers.Layer` for the `g`
      function of NCP, which transforms the embedding of each cluster `H_k` to
      `g_k`, in order to build the global representation of already clustered
      points (`G`).
    logits_layer: A `tf.keras.Model` or `tf.keras.layers.Layer` for the `f`
      function of NCP (output dim must be 1), which computes the logits of
      possible cluster assignments.
  """

  def __init__(self, assigned_point_layer, unassigned_point_layer,
               cluster_layer, logits_layer):
    super(NCPBase, self).__init__()

    self.assigned_point_layer = assigned_point_layer
    self.unassigned_point_layer = unassigned_point_layer
    self.cluster_layer = cluster_layer
    self.logits_layer = logits_layer

  def preprocess_inputs(self, inputs):
    """Precomputes the assigned and unassigned representation for each point.

    Arguments:
      inputs: A `Tensor` of shape `[batch_size, num_points, x_dim]`. The input
        data containing all points to be clustered.

    Returns:
      h_x: A `Tensor` of shape `[batch_size, num_points, h_dim]`. The output of
        `assigned_point_layer(inputs)`, which will be used to build
        representations of existing clusters (`H_k`) from assigned points.
      u_x: A `Tensor` of shape `[batch_size, num_points, u_dim]`. The output of
        `unassigned_point_layer(inputs)`, which will be used to build the global
        representation of all unassigned points (`U`).
    """

    h_x = self.assigned_point_layer(inputs)
    u_x = self.unassigned_point_layer(inputs)

    preprocessed_inputs = [h_x, u_x]
    return preprocessed_inputs

  def initialize_states(self, preprocessed_inputs):
    """Initializes the global states.

    Arguments:
      preprocessed_inputs: a list of preprocessed inputs `[h_x, u_x]`, which is
        the output of `self.preprocess_inputs(inputs)`.

    Returns:
      A list containing the initialized global states:
      - u_aggr: A `Tensor` of shape `[batch_size, u_dim]`. The global
        representation of all unassigned points (`U`) by aggregating `u_x`.
      - h_aggr: A zero `Tensor` of shape `[batch_size, 1, h_dim]`. The initial
        representation of an empty cluster with no point assigned to it.
      - g_h: A zero `Tensor` of shape `[batch_size, 1, g_dim]`. This tensor will
        contain the output of `self.cluster_layer(h_aggr)`.
      - g_aggr: A zero `Tensor` of shape `[batch_size, g_dim]`. This tensor will
        contain the global representation of already clustered data (`G`) by
        aggregating `g_h`.
      - g_candidates: Initialized to `None`. This variable will updated to hold
        the hypothesized global representations after adding `x_i` to each of
        the possible clusters.
      - num_clusters: A `List[int]` (`length=batch_size`) initialized to ones
        indicating that there is only one possible cluster (i.e. the empty
        cluster) to join for each batch element at the first iteration.
    """

    h_x, u_x = preprocessed_inputs

    batch_size = u_x.shape[0]
    h_dim = h_x.shape[-1]
    g_dim = self.cluster_layer.compute_output_shape(h_x.shape)[-1]

    u_aggr = tf.reduce_sum(u_x, axis=1)
    h_aggr = tf.zeros([batch_size, 1, h_dim])
    g_h = tf.zeros([batch_size, 1, g_dim])
    g_aggr = tf.zeros([batch_size, g_dim])
    g_candidates = None

    num_clusters = [1] * batch_size

    states = [u_aggr, h_aggr, g_h, g_aggr, g_candidates, num_clusters]
    return states

  def call(self, next_preprocessed_input, states):
    """Computes the logits of adding the new point to all the possible clusters.

    Arguments:
      next_preprocessed_input: A list containing the preprocessed tensors
        of the next data point to be clustered: `[h_x_next, u_x_next]`, where
        `h_x_next = h_x[:, next_point]`, and `u_x_next = u_x[:, next_point]`.
          Here, `[h_x, u_x]` is the output of `self.preprocess_inputs(inputs)`,
          and `next_point` is the index of the next point to be clustered.
      states: A list containing the global states:
      - u_aggr: A `Tensor` of shape `[batch_size, u_dim]`. The global
        representation of currently unassigned points (`U`) by aggregating
        `u_x`.
      - h_aggr: A `Tensor` of shape `[batch_size, max(num_clusters), h_dim]`,
        Each slice along the second dimension is the representation of the k-th
        cluster (`H_k`) by aggregating `h_x` assigned to that cluster.
      - g_h: A `Tensor` of shape `[batch_size, max(num_clusters), g_dim]`,
        representing the output of `self.cluster_layer(h_aggr)`.
      - g_aggr: A `Tensor` of shape `[batch_size, g_dim]`. The global
        representation of already clustered data (`G`) by aggregating `g_h`.
      - g_candidates: `None` as part of the input states, but will be updated to
        a `Tensor` of shape `[batch_size, max(num_clusters), g_dim]` in the
        output states, holding the hypothesized global representations after
        adding `x_i` to each of the possible clusters.
      - num_clusters: A `List[int]` (`length=batch_size`) storing the number of
        possible clusters to join for each batch example at the current
        iteration. This includes the empty cluster. If the new point joins the
        empty cluster, it will effectively create a new cluster.

    Returns:
      logits: A `Tensor` of shape `[batch_size, max_num_clusters]`. The log
        likelihood of adding the next point to each of the possible clusters.
        For any row `k` in the batch where `num_clusters[k] < max_num_clusters`,
        `logits` will be padded with `-inf` for non-existing clusters as
        `logits[k, n_cluster[k]:] = -np.inf`.

      states: A list containing the updated global states.
    """

    h_x_next, u_x_next = next_preprocessed_input
    u_aggr, h_aggr, g_h, g_aggr, g_candidates, num_clusters = states

    # Take the current point out of the embedding of unassigned points (U).
    u_aggr -= u_x_next

    # Try adding h_x_next to every possible cluster.
    h_candidates = h_aggr + h_x_next[:, tf.newaxis]
    g_candidates = g_aggr[:,
                          tf.newaxis] + self.cluster_layer(h_candidates) - g_h

    u_aggr_expanded = tf.broadcast_to(
        u_aggr[:, tf.newaxis],
        [u_aggr.shape[0], g_candidates.shape[1], u_aggr.shape[-1]])

    logits = self.logits_layer(
        tf.concat([g_candidates, u_aggr_expanded], axis=-1))
    logits = tf.nn.log_softmax(tf.squeeze(logits, axis=-1), axis=-1)

    # Pads logits of non-existing clusters with -inf.
    logits_mask = np.zeros_like(logits)
    for i, k in enumerate(num_clusters):
      logits_mask[i, k:] = -np.inf
    logits_mask = tf.convert_to_tensor(logits_mask)
    logits += logits_mask

    states = [u_aggr, h_aggr, g_h, g_aggr, g_candidates, num_clusters]
    return logits, states

  def update_states_by_cluster_assignment(self, next_preprocessed_input, states,
                                          cluster_ids):
    """Updates global states based on the cluster assignments of the new points.

    The number of possible clusters and the shape of the tensors containing
    cluster repesentations are dynamically expanded if any row in the batch no
    longer has an empty cluster (due to the new point joining that cluster).

    Arguments:
      next_preprocessed_input: A list containing the preprocessed tensors
        of the next data point to be clustered: `[h_x_next, u_x_next]`, where
        `h_x_next = h_x[:, next_point]`, and `u_x_next = u_x[:, next_point]`.
          Here, `[h_x, u_x]` is the output of `self.preprocess_inputs(inputs)`,
          and `next_point` is the index of the next point to be clustered.
      states: A list containing the global states from the output of the
        `call()` method.
      cluster_ids: An int `Tensor` of shape `[batch_size]`. The cluster IDs
        assigned to the new points.

    Returns:
      A list of updated states.
    """
    h_x_next, _ = next_preprocessed_input
    batch_size = h_x_next.shape[0]
    u_aggr, h_aggr, g_h, g_aggr, g_candidates, num_clusters = states

    # The tensor indices used to update g_aggr, h_aggr, g_h.
    indices = tf.stack([tf.range(batch_size), cluster_ids], axis=1)

    # Non-batch equivalent: g_aggr = g_candidates[cluster_id]
    g_aggr = tf.gather_nd(g_candidates, indices)
    g_candidates = None

    # Non-batch equivalent: h_aggr[cluster_id] += h_x_next.
    h_aggr = tf.tensor_scatter_nd_add(h_aggr, indices, h_x_next)

    # Non-batch equivalent:
    # g_h[cluster_id] = self.cluster_layer(h_aggr[cluster_id])
    g_h_update = self.cluster_layer(tf.gather_nd(h_aggr, indices))
    g_h = tf.tensor_scatter_nd_update(g_h, indices, g_h_update)

    # Dynamically expand the number of clusters by growing the embeddings.
    max_num_clusters = h_aggr.shape[1]
    max_cluster_id = np.max(cluster_ids)
    if max_cluster_id == max_num_clusters - 1:
      h_aggr = tf.concat(
          [h_aggr, tf.zeros([h_aggr.shape[0], 1, h_aggr.shape[-1]])], axis=1)
      g_h = tf.concat(
          [g_h, tf.zeros(shape=[g_h.shape[0], 1, g_h.shape[-1]])], axis=1)

    for i, k in enumerate(cluster_ids):
      if k == num_clusters[i] - 1:
        num_clusters[i] += 1

    states = [u_aggr, h_aggr, g_h, g_aggr, g_candidates, num_clusters]
    return states
