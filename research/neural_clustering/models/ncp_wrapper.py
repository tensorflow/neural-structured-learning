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
"""NCP model wrapper and cluster samplers."""

from neural_clustering.utils.data_utils import batch_remap_label_ids
import tensorflow as tf


class GreedySampler():
  """Greedy sampler always choosing the class with the highest probability."""

  def sample(self, logits):
    """Selects the cluster IDs with with the highest probability.

    Arguments:
      logits: A float tensor of shape `[batch_size, num_clusters]` representing
        a batch of log likelihoods for all possible clusters.

    Returns:
      An int tensor of shape `[batch_size]`, the sampled cluster IDs.
    """
    return tf.argmax(logits, axis=-1, output_type=tf.int32)


class CategoricalSampler():
  """Random sampler using the categorical (multinomial) distribution."""

  def sample(self, logits):
    """Samples cluster IDs using the categorical (multinomial) distribution.

    Arguments:
      logits: A float tensor of shape `[batch_size, num_clusters]` representing
        a batch of log likelihoods for all possible clusters.

    Returns:
      An int tensor of shape `[batch_size]`, the sampled cluster IDs.
    """
    return tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0]


class NCPWrapper(tf.keras.Model):
  """Wrapper model implementing the full NCP clustering algorithm.

  This class wraps an instantiated NCP model and performs the full NCP
  clustering algorithm, which sequentially assigns cluster labels to N input
  data points by N forward passes through the NCP model.

  Attributes:
    ncp_model: A `NCPBase` model instance.
    sampler: A sampler class with a `sample(logits)` function for posterior
      sampling of cluster labels at inference time, or set to `None` for
      training. The `sample(logits)` function should take as input a float
      tensor of shape `[batch_size, num_clusters]` and return an int tensor of
      shape `[batch_size]` representing the sampled cluster IDs.
  """

  def __init__(self, ncp_model, sampler=None):
    super(NCPWrapper, self).__init__()
    self.ncp_model = ncp_model
    self.sampler = sampler

  def call(self, inputs, targets=None, training=False):
    """Computes the logits and samples the cluster labels for input data points.

    During training, the cluster labels are assigned as the targets to train the
    NCP model. The target labels in each batch example will be preprocessed
    through a remapping so that the first occurrence of each label is in
    ascending order. Please see `batch_remap_label_ids` for detailed
    explanation.

    At inference time, the cluster labels are assigned by the sampler based on
    the predicted logits.

    Arguments:
      inputs: A `Tensor` or `np.ndarray` of shape `[batch_size, num_points,
        x_dim]` for input data.
      targets: A `Tensor` or `np.ndarray` of shape `[batch_size, num_points]`
        for target labels at training time, or set to `None` for inference.
      training: Boolean, whether the model is run for training or inference.

    Returns:
      all_logits: A `Tensor` of shape `[batch_size, num_points]` representing
        the logits of the assigned clusters for each point.
      all_cluster_ids: A `Tensor` of shape `[batch_size, num_points]`
        representing the assigned cluster labels. At training time, this will be
        the target labels transformed by `batch_remap_label_ids`.
    """
    if training:
      if targets is None:
        raise ValueError(
            "targets need to be provided for training, cannot be `None`.")
      else:
        # Remaps target label IDs so that the first occurrence of each label ID
        # is in ascending order.
        targets, _ = batch_remap_label_ids(targets)

    batch_size, num_points, _ = inputs.shape
    # Preprocesses the inputs.
    preprocessed_inputs = self.ncp_model.preprocess_inputs(inputs)
    # Initializes the global states before iterating through all data points.
    states = self.ncp_model.initialize_states(preprocessed_inputs)

    all_logits = []
    all_cluster_ids = []
    # Assigns a cluster label to each point sequentially.
    for next_point in range(num_points):
      next_preprocessed_input = [x[:, next_point] for x in preprocessed_inputs]
      logits, states = self.ncp_model(next_preprocessed_input, states)
      if training:
        # Teacher forcing for training
        cluster_ids = targets[:, next_point]
      else:
        # Samples a cluster id from the variable-input softmax for inference.
        cluster_ids = self.sampler.sample(logits)

      # Only returns the logits of the assigned clusters due to variable size.
      indices = tf.stack([tf.range(batch_size), cluster_ids], axis=1)
      all_logits.append(tf.gather_nd(logits, indices))
      all_cluster_ids.append(cluster_ids)

      # Updates the global states after assigning the new point to a cluster.
      states = self.ncp_model.update_states_by_cluster_assignment(
          next_preprocessed_input, states, cluster_ids)

    all_logits = tf.stack(all_logits, axis=1)
    all_cluster_ids = tf.stack(all_cluster_ids, axis=1)
    return all_logits, all_cluster_ids

  def loss_function(self, logits):
    """The negative log likelihood (NLL) loss function.

    Arguments:
      logits: A `Tensor` of shape `[batch_size, num_points]` representing the
        log likelihoods of the assigned clusters for each point.

    Returns:
      A scalar `Tensor`. The NLL loss.
    """
    batch_avg_logits = tf.reduce_mean(logits, axis=0)
    nll_loss = -tf.reduce_sum(batch_avg_logits)
    return nll_loss
