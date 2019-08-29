# Copyright 2019 Google LLC
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
"""A layer to compute pairwise distances in Neural Structured Learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import enum

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import distances
from neural_structured_learning.lib import utils
import tensorflow as tf


class PairwiseDistance(tf.keras.layers.Layer):
  """A layer for computing a pairwise distance in Keras models.

  With `Model.add_loss`, this layer can be used to build a Keras model with
  graph regularization.

  Example:

  ```python
  def regularize_model(unregularized_model, inputs):
    features, neighbors, neighbor_weights = inputs
    # Standard logistic regression.
    logits = unregularized_model(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    # Add regularization.
    regularizer = layers.PairwiseDistance(
        configs.DistanceConfig(sum_over_axis=-1))
    graph_loss = regularizer(
        sources=logits,
        targets=unregularized_model(neighbors),
        weights=neighbor_weights)
    model.add_loss(graph_loss)
    model.add_metric(graph_loss, aggregation="mean", name="graph_loss")
    return model
  ```

  This layer makes some assumptions about how the input is shaped. Either (a)
  the first dimension of `sources` should divide the first dimension of
  `targets`, with the rest of dimensions being equal or (b) `targets` should
  have an additional neighborhood size dimension at axis -2, the last dimension
  of `sources` and `targets` should match, and all other dimensions of `sources`
  should also match with corresponding dimensions in `targets`. See
  `_replicate_sources` for details.
  """

  def __init__(self, distance_config=None, **kwargs):
    super(PairwiseDistance, self).__init__(**kwargs)
    self._distance_config = (
        configs.DistanceConfig()
        if distance_config is None else attr.evolve(distance_config))

  def _replicate_sources(self, sources, targets):
    """Replicates `sources` to match the shape of `targets`.

    `targets` should either have an additional neighborhood size dimension at
    axis -2 or be of the same rank as `sources`. If `targets` has an additional
    dimension and `sources` has rank k, the first k - 1 dimensions and last
    dimension of `sources` and `targets` should match. If `sources` and
    `targets` have the same rank, the last k - 1 dimensions should match and the
    first dimension of `targets` should be a multiple of the first dimension of
    `sources`. This multiple represents the fixed neighborhood size of each
    sample.

    Args:
      sources: Tensor with shape [..., feature_size] from which distance will be
        calculated.
      targets: Either a tensor with shape [..., neighborhood_size, feature_size]
        or [sources.shape[0] * neighborhood_size] + sources.shape[1:].

    Returns:
      `sources` replicated to be shape-compatible with `targets`.
    """
    # Depending on the rank of `sources` and `targets`, decide to broadcast
    # first, or replicate directly.
    if (sources.shape.ndims is not None and targets.shape.ndims is not None and
        sources.shape.ndims + 1 == targets.shape.ndims):
      return tf.broadcast_to(
          tf.expand_dims(sources, axis=-2), tf.shape(targets))

    return utils.replicate_embeddings(
        sources,
        tf.shape(targets)[0] // tf.shape(sources)[0])

  def call(self, inputs, weights=None):
    """Replicates sources and computes pairwise distance.

    Args:
      inputs: Symbolic inputs. Should be (sources, targets) if `weights` is
        non-symbolic. Otherwise, should be (sources, targets, weights).
      weights: If target weights are not symbolic, `weights` should be passed as
        a separate argument. In this case, `inputs` should have length 2.

    Returns:
      Pairwise distance tensor.
    """
    if weights is None:
      sources, targets, weights = inputs
    else:
      sources, targets = inputs

    return distances.pairwise_distance_wrapper(
        sources=self._replicate_sources(sources, targets),
        targets=targets,
        weights=weights,
        distance_config=self._distance_config)

  def __call__(self, sources, targets=None, weights=1., **kwargs):
    # __call__ is overridden so when constructing the model the user can pass
    # keyword arguments. Within the framework, Keras will always pass arguments
    # in a list.
    # If targets is None and len(sources) > 1, assume the function is being
    # called in a cloned context with all symbolic inputs.
    if targets is None and len(sources) == 3:
      return super(PairwiseDistance, self).__call__(sources, **kwargs)

    if targets is None and len(sources) == 2:
      return super(PairwiseDistance, self).__call__(
          sources, weights=weights, **kwargs)

    # Otherwise assume that the user is calling the function.
    if targets is None:
      raise ValueError("No targets provided.")

    if tf.get_static_value(weights) is None:
      return super(PairwiseDistance, self).__call__((sources, targets, weights),
                                                    **kwargs)

    return super(PairwiseDistance, self).__call__(
        (sources, targets), weights=tf.get_static_value(weights), **kwargs)

  def get_config(self):
    distance_config = attr.asdict(self._distance_config)
    distance_config.update({
        k: v.value
        for k, v in distance_config.items()
        if isinstance(v, enum.Enum)
    })
    config = super(PairwiseDistance, self).get_config()
    config["distance_config"] = distance_config
    return config

  @classmethod
  def from_config(cls, config):
    return cls(
        configs.DistanceConfig(**config["distance_config"]),
        name=config.get("name"))
