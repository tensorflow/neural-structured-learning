# Copyright 2020 Google LLC
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
"""Incorporates graph regularization with caching into a Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from neural_structured_learning.keras import graph_regularization
import tensorflow as tf


class GraphRegularizationWithCaching(graph_regularization.GraphRegularization):
  """Graph regularization class with support for caching neighbor state.

  It requires an instance of `NeighborCacheClient` to issue lookup and update
  operations to the neighbor cache.

  Example usage:

  ```python
  # Create a base model using the sequential, functional, or subclass API.
  base_model = tf.keras.Sequential(...)

  # Create a NeighborCacheClient object that implements the abstract methods.
  class NeighborCacheClientImpl(carls.NeighborCacheClient):
    def lookup(self, neighbor_ids):
      Send lookup request to cache server, and return the result.

    def update(self, neighbor_ids, neighbor_state):
      Send update request to the cache server.

  # Wrap the base model to include graph regularization using up to 1 neighbor
  # per sample.
  neighbor_cache_client = NeighborCacheClientImpl()
  graph_config = nsl.configs.make_graph_reg_config(max_neighbors=1)
  graph_model = carls.GraphRegularizationWithCaching(
      base_model, graph_config, neighbor_cache_client)

  # Compile, train, and evaluate the graph-regularized model as usual.
  graph_model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])
  graph_model.fit(train_dataset, epochs=5)
  graph_model.evaluate(test_dataset)
  ```
  """

  def __init__(self,
               base_model,
               graph_reg_config=None,
               neighbor_cache_client=None):
    """Class initializer.

    Args:
      base_model: Unregularized model to which the loss term resulting from
        graph regularization will be added.
      graph_reg_config: Instance of `nsl.configs.GraphRegConfig` that contains
        configuration for graph regularization. Use
        `nsl.configs.make_graph_reg_config` to construct one.
      neighbor_cache_client: Used to look up and update cached neighbor state.
    """

    super(GraphRegularizationWithCaching, self).__init__(
        base_model=base_model, graph_reg_config=graph_reg_config)
    self.neighbor_cache_client = neighbor_cache_client

  def _infer_and_update(self, features, training, **kwargs):
    """Invokes `base_model` on `features` and updates the neighbor cache."""
    if self.neighbor_cache_client:
      key = features.pop(self.neighbor_cache_client.key_feature_name)
    output = self.base_model(features, training=training, **kwargs)
    if self.neighbor_cache_client:
      # TODO(thunderfyc): Make update operation optional when we have config
      # for neighbor cache in graph config.
      self.neighbor_cache_client.update(key, output)
    return output

  def _get_neighbor_logits(self, nbr_features, training, **kwargs):
    """Gets the logits for neighbor examples."""
    nbr_logits = None
    if self.neighbor_cache_client:
      # Squeezes lookup keys from [(B*N), 1] to [B*N] to make the shape of
      # nbr_logits compatible with nbr_weights.
      nbr_lookup_keys = tf.squeeze(
          nbr_features[self.neighbor_cache_client.key_feature_name])
      nbr_logits = self.neighbor_cache_client.lookup(nbr_lookup_keys)
    if nbr_logits is None:
      nbr_logits = self._infer_and_update(
          nbr_features, training=training, **kwargs)
    return nbr_logits

  def call(self, inputs, training=False, **kwargs):
    """Incorporates graph regularization into the loss of `base_model`.

    Graph regularization is done on the logits layer and only during training.

    Args:
      inputs: Dictionary containing sample features, neighbor features, and
        neighbor weights in the same format as described in
        `utils.unpack_neighbor_features`.
      training: Boolean tensor that indicates if we are in training mode.
      **kwargs: Additional keyword arguments to be passed to `self.base_model`.

    Returns:
      The output tensors for the wrapped graph-regularized model.
    """
    # Invoke the call() function of the neighbor features layer directly instead
    # of invoking it as a callable to avoid Keras from wrapping placeholder
    # tensors with the tf.identity() op.
    sample_features, nbr_features, nbr_weights = self.nbr_features_layer.call(
        inputs)
    base_output = self._infer_and_update(
        sample_features, training=training, **kwargs)

    # For evaluation and prediction, we use the base model. So, this overridden
    # call function will get invoked only for training.
    has_nbr_inputs = nbr_weights is not None and nbr_features
    if (has_nbr_inputs and self.graph_reg_config.multiplier > 0):
      # Use logits for regularization.
      sample_logits = base_output
      nbr_logits = self._get_neighbor_logits(nbr_features, training, **kwargs)
      graph_loss = self.regularizer(
          sources=sample_logits, targets=nbr_logits, weights=nbr_weights)
    else:
      graph_loss = tf.constant(0, dtype=tf.float32)

    # Note that add_metric() cannot be invoked in a control flow branch.
    self.add_metric(graph_loss, name='graph_loss', aggregation='mean')
    self.add_loss(self.graph_reg_config.multiplier * graph_loss)

    return base_output
