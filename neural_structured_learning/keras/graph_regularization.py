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
"""Incorporates graph regularization into a Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as nsl_configs
from neural_structured_learning.keras import layers as nsl_layers
import tensorflow as tf


class GraphRegularization(tf.keras.Model):
  """Class that wraps a given `Keras` model to include graph regularization.

  Graph regularization is configured by an instance of
  `nsl.configs.GraphRegConfig` and the resulting loss is added as a
  regularization term to the model's training objective. The graph-regularized
  model reuses the layers and variables from the base model. So, training this
  model will also update the variables in the base model.

  Note: This class expects input data to include neighor features corresponding
  to the maximum number of neighbors used for graph regularization.

  Example usage:

  ```python
  # Create a base model using the sequential, functional, or subclass API.
  base_model = tf.keras.Sequential(...)

  # Wrap the base model to include graph regularization using up to 1 neighbor
  # per sample.
  graph_config = nsl.configs.make_graph_reg_config(max_neighbors=1)
  graph_model = nsl.keras.GraphRegularization(base_model, graph_config)

  # Compile, train, and evaluate the graph-regularized model as usual.
  graph_model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])
  graph_model.fit(train_dataset, epochs=5)
  graph_model.evaluate(test_dataset)
  ```
  """

  def __init__(self, base_model, graph_reg_config=None):
    """Class initializer.

    Args:
      base_model: Unregularized model to which the loss term resulting from
        graph regularization will be added.
      graph_reg_config: Instance of `nsl.configs.GraphRegConfig` that contains
        configuration for graph regularization. Use
        `nsl.configs.make_graph_reg_config` to construct one.
    """

    super(GraphRegularization, self).__init__(name='GraphRegularization')
    self.base_model = base_model
    self.graph_reg_config = (
        nsl_configs.GraphRegConfig()
        if graph_reg_config is None else graph_reg_config)
    self.nbr_features_layer = nsl_layers.NeighborFeatures(
        self.graph_reg_config.neighbor_config)
    self.regularizer = nsl_layers.PairwiseDistance(
        self.graph_reg_config.distance_config, name='graph_loss')

  # This override is required in case 'self.base_model' is a subclass model and
  # has overridden the compile function.
  def compile(self, *args, **kwargs):
    super(GraphRegularization, self).compile(*args, **kwargs)
    self.base_model.compile(*args, **kwargs)

  compile.__doc__ = tf.keras.Model.compile.__doc__

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
    sample_features, nbr_features, nbr_weights = self.nbr_features_layer(inputs)
    base_output = self.base_model(sample_features, training=training, **kwargs)

    has_nbr_inputs = nbr_weights is not None and nbr_features

    # 'training' is a boolean or boolean tensor. So, we have to use the tf.cond
    # op to be able to write conditional code based on its value.

    def graph_loss_with_regularization():
      if (has_nbr_inputs and self.graph_reg_config.multiplier > 0):
        # Use logits for regularization.
        sample_logits = base_output
        nbr_logits = self.base_model(nbr_features, training=training, **kwargs)
        return self.regularizer(
            sources=sample_logits, targets=nbr_logits, weights=nbr_weights)
      else:
        return tf.constant(0, dtype=tf.float32)

    def graph_loss_without_regularization():
      return tf.constant(0, dtype=tf.float32)

    graph_loss = tf.cond(
        tf.equal(training, tf.constant(True)), graph_loss_with_regularization,
        graph_loss_without_regularization)

    # Note that add_metric() cannot be invoked in a control flow branch.
    self.add_metric(graph_loss, name='graph_loss', aggregation='mean')
    self.add_loss(self.graph_reg_config.multiplier * graph_loss)

    return base_output
