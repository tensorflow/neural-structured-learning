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
"""Integration tests for neural_structured_learning.keras.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
from neural_structured_learning import configs
from neural_structured_learning.keras import layers
import numpy as np
import tensorflow as tf


_ERR_TOL = 3e-5  # Tolerance when comparing floats.


# TODO(ppham27): Update models to use NeighborFeatures
def _make_functional_regularized_model(distance_config):
  """Makes a model with `PairwiseDistance` and the functional API."""

  def _make_unregularized_model(inputs, num_classes):
    """Makes standard 1 layer MLP with logistic regression."""
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes)(x))
    return model

  # Each example has 4 features and 2 neighbors, each with an edge weight.
  inputs = (tf.keras.Input(shape=(4,), dtype=tf.float32, name='features'),
            tf.keras.Input(shape=(2, 4), dtype=tf.float32, name='neighbors'),
            tf.keras.Input(
                shape=(2, 1), dtype=tf.float32, name='neighbor_weights'))
  features, neighbors, neighbor_weights = inputs
  neighbors = tf.reshape(neighbors, (-1,) + tuple(features.shape[1:]))
  neighbor_weights = tf.reshape(neighbor_weights, [-1, 1])
  unregularized_model = _make_unregularized_model(features, 3)
  logits = unregularized_model(features)
  model = tf.keras.Model(inputs=inputs, outputs=logits)
  # Add regularization.
  regularizer = layers.PairwiseDistance(distance_config)
  graph_loss = regularizer(
      sources=logits,
      targets=unregularized_model(neighbors),
      weights=neighbor_weights)
  model.add_loss(graph_loss)
  model.add_metric(graph_loss, aggregation='mean', name='graph_loss')
  return model


class _PairwiseRegularizedModel(tf.keras.Model):
  """Example model for using `PairwiseDistance` by subclassing."""

  def __init__(self, distance_config, **kwargs):
    super(_PairwiseRegularizedModel, self).__init__(**kwargs)
    self._regularizer = layers.PairwiseDistance(
        distance_config, name='graph_loss')
    self._unregularized_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3),
    ])

  def call(self, inputs):
    features = inputs['features']
    neighbors = inputs['neighbors']
    neighbor_weights = inputs['neighbor_weights']
    # Forward pass.
    logits = self._unregularized_model(features)
    # Add regularization.
    graph_loss = self._regularizer(
        sources=logits,
        targets=self._unregularized_model(neighbors),
        weights=neighbor_weights)
    self.add_loss(graph_loss)
    self.add_metric(graph_loss, aggregation='mean', name='graph_loss')
    return logits


class LayersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for neural_structured_learning.keras.layers."""

  @parameterized.parameters(
      itertools.product(
          (_make_functional_regularized_model, _PairwiseRegularizedModel),
          configs.DistanceType.all()))
  def testModelFitAndEvaluate(self, model_fn, distance_type):
    """Fit and evaluate models with various distance configurations."""
    # Set up graph-regularized model.
    distance_config = configs.DistanceConfig(
        distance_type=distance_type,
        transform_fn=configs.TransformType.SOFTMAX,
        sum_over_axis=-1)
    model = model_fn(distance_config)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        ])
    # Fit and evaluate the model on dummy data that has 8 examples.
    features = {
        'features': np.random.normal(size=(8, 4)),
        'neighbors': np.random.normal(size=(8, 2, 4)),
        'neighbor_weights': np.random.uniform(size=(8, 2, 1)),
    }
    labels = np.random.randint(0, 3, size=8)
    train_history = model.fit(features, labels, batch_size=2, epochs=16).history
    evaluation_results = dict(
        zip(model.metrics_names, model.evaluate(features, labels,
                                                batch_size=4)))
    # Assert that losses and metrics were evaluated.
    self.assertAllGreater(train_history['graph_loss'], 0.)
    self.assertGreater(evaluation_results['graph_loss'], 0.)
    self.assertAllClose(
        train_history['loss'],
        np.add(train_history['graph_loss'],
               train_history['sparse_categorical_crossentropy']), _ERR_TOL)
    self.assertNear(
        evaluation_results['loss'], evaluation_results['graph_loss'] +
        evaluation_results['sparse_categorical_crossentropy'], _ERR_TOL)


if __name__ == '__main__':
  tf.test.main()
