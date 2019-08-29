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
"""Tests for neural_structured_learning.keras.layers.pairwise_distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs
from neural_structured_learning.keras.layers import pairwise_distance as pairwise_distance_lib
import numpy as np
from scipy import special
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

_ERR_TOL = 3e-5  # Tolerance when comparing floats.


@test_util.run_all_in_graph_and_eager_modes
class PairwiseDistanceTest(tf.test.TestCase):
  """Tests for `pairwise_distance_lib.PairwiseDistance`."""

  def testName(self):
    """Tests that the name is propagated to the base layer."""
    regularizer = pairwise_distance_lib.PairwiseDistance()
    self.assertEqual(regularizer.name, 'pairwise_distance')

    regularizer = pairwise_distance_lib.PairwiseDistance(name='regularizer')
    self.assertEqual(regularizer.name, 'regularizer')

  def testCall(self):
    """Makes a function from config and runs it."""
    regularizer = pairwise_distance_lib.PairwiseDistance(
        configs.DistanceConfig(
            distance_type=configs.DistanceType.KL_DIVERGENCE, sum_over_axis=-1),
        name='kl_loss')
    # Run a computation.
    example = np.array([0.3, 0.3, 0.4])
    neighbors = np.array([[0.9, 0.05, 0.05]])
    kl_loss = self.evaluate(regularizer(example, neighbors))
    # Assert correctness of KL divergence calculation.
    self.assertNear(kl_loss, np.sum(special.kl_div(example, neighbors)),
                    _ERR_TOL)

  def testWeights(self):
    """Tests that weights are propagated to the distance function."""
    regularizer = pairwise_distance_lib.PairwiseDistance(
        configs.DistanceConfig(
            distance_type=configs.DistanceType.KL_DIVERGENCE, sum_over_axis=-1),
        name='weighted_kl_loss')
    example = np.array([0.1, 0.4, 0.5])
    neighbors = np.array([[0.6, 0.2, 0.2], [0.9, 0.01, 0.09]])
    neighbor_weight = 0.5
    loss = self.evaluate(regularizer(example, neighbors, neighbor_weight))
    self.assertAllClose(
        loss,
        neighbor_weight *
        np.mean(np.sum(special.kl_div(example, neighbors), -1)), _ERR_TOL)

  def testAssertions(self):
    """Tests that assertions still work with Keras."""
    distance_config = configs.DistanceConfig(
        distance_type=configs.DistanceType.JENSEN_SHANNON_DIVERGENCE,
        sum_over_axis=-1)
    regularizer = pairwise_distance_lib.PairwiseDistance(distance_config)
    # Try Jennsen-Shannon divergence on an improper probability distribution.
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'x and/or y is not a proper probability distribution'):
      self.evaluate(regularizer(np.array([0.6, 0.5]), np.array([[0.25, 0.75]])))

  def testCallOverride(self):
    """Tests the overrides of Layer.__call__."""

    # Default distance configuration is mean squared error.
    def _distance_fn(x, y):
      return np.mean(np.square(x - y))

    # Common input.
    sources = np.array([1., 1., 1., 1.])
    targets = np.array([[4., 3., 2., 1.]])
    unweighted_distance = _distance_fn(sources, targets)

    def _make_symbolic_weights_model():
      """Makes a model where the weights are provided as input."""
      inputs = {
          'sources': tf.keras.Input(4),
          'targets': tf.keras.Input((1, 4)),
          'weights': tf.keras.Input((1, 1)),
      }
      pairwise_distance_fn = pairwise_distance_lib.PairwiseDistance()
      outputs = pairwise_distance_fn(**inputs)
      return tf.keras.Model(inputs=inputs, outputs=outputs)

    weights = np.array([[2.]])
    expected_distance = unweighted_distance * weights
    model = _make_symbolic_weights_model()
    self.assertNear(
        self.evaluate(
            model({
                'sources': sources,
                'targets': targets,
                'weights': weights,
            })), expected_distance, _ERR_TOL)

    def _make_fixed_weights_model(weights):
      """Makes a model where the weights are a static constant."""
      inputs = {
          'sources': tf.keras.Input(4),
          'targets': tf.keras.Input((1, 4)),
      }
      pairwise_distance_fn = pairwise_distance_lib.PairwiseDistance()
      outputs = pairwise_distance_fn(weights=weights, **inputs)
      return tf.keras.Model(inputs=inputs, outputs=outputs)

    model = _make_fixed_weights_model(0.25)
    expected_distance = 0.25 * unweighted_distance
    self.assertNear(
        self.evaluate(model({
            'sources': sources,
            'targets': targets,
        })), expected_distance, _ERR_TOL)
    # Considers invalid input.
    with self.assertRaisesRegex(ValueError, 'No targets provided'):
      pairwise_distance_lib.PairwiseDistance()(np.ones(5))

  def testReplicateSources(self):
    """Tests when sources and targets have the same rank."""

    def _make_model(sources_shape, targets_shape):
      """Makes a model where `sources` and `targets` have the same rank."""
      sources = tf.keras.Input(sources_shape, name='sources')
      targets = tf.keras.Input(targets_shape, name='targets')
      outputs = pairwise_distance_lib.PairwiseDistance(
          configs.DistanceConfig(
              distance_type=configs.DistanceType.KL_DIVERGENCE,
              reduction=tf.compat.v1.losses.Reduction.NONE,
              sum_over_axis=-1))(sources, targets)
      return tf.keras.Model(inputs=[sources, targets], outputs=outputs)

    model = _make_model(sources_shape=(4,), targets_shape=(4,))
    # Test when first dimension of targets is a multiple of the batch size.
    sources = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
    ])
    targets = np.array([
        [0.6, 0.2, 0.1, 0.1],
        [0.4, 0.5, 0.075, 0.025],
        [0.001, 0.333, 0.333, 0.333],
        [0.9, 0.05, 0.03, 0.02],
    ])
    kl_divergence = self.evaluate(model([sources, targets]))
    expected = np.sum(
        special.kl_div(np.repeat(sources, 2, axis=0), targets),
        -1,
        keepdims=True)
    self.assertAllClose(kl_divergence, expected, _ERR_TOL)
    # Test when that the shapes are not compatible.
    with self.assertRaisesRegex(ValueError, 'Shapes\\s.+\\sare incompatible'):
      self.evaluate(model([sources, targets[1:]]))
    # And also the case when targets has a neighborhood size dimension.
    model = _make_model(sources_shape=(4,), targets_shape=(2, 4))
    self.assertAllClose(
        self.evaluate(model([sources, targets.reshape((-1, 2, 4))])),
        expected.reshape((-1, 2, 1)), _ERR_TOL)


if __name__ == '__main__':
  tf.test.main()
