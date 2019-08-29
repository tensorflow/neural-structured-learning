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
"""Tests for neural_structured_learning.lib.distances."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import distances

import numpy as np
import tensorflow as tf


class DistancesTest(tf.test.TestCase):

  def _kl_func(self, x, y):
    eps = 1e-7
    return x * np.log(x + eps) - x * np.log(y + eps)

  def _jsd_func(self, x, y):
    m = 0.5 * (x + y)
    return 0.5 * self._kl_func(x, m) + 0.5 * self._kl_func(y, m)

  def _softmax_func(self, x, axis=-1):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

  def testKLDivergence(self):
    source_tensor = np.array([[1, 0, 0], [0.1, 0.2, 0.7]])
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]])

    expected_tensor = np.sum(self._kl_func(source_tensor, target_tensor), -1)
    expected_value = np.mean(expected_tensor)
    distance_tensor = distances.kl_divergence(
        source_tensor, target_tensor, axis=-1)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testJensenShannonDivergence(self):
    source_tensor = np.array([[1, 0, 0], [0.1, 0.2, 0.7]])
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]])

    expected_tensor = np.sum(self._jsd_func(source_tensor, target_tensor), -1)
    expected_value = np.mean(expected_tensor)
    distance_tensor = distances.jensen_shannon_divergence(
        source_tensor, target_tensor, axis=-1)

    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testInvalidMultinomialDistribution(self):
    source_tensor = np.array([[1, 0, 0], [0.1, 0.2, 0.8]])
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]])
    with self.cached_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        distance_tensor = distances.kl_divergence(
            source_tensor, target_tensor, axis=-1)
        distance_tensor.eval()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        distance_tensor = distances.jensen_shannon_divergence(
            source_tensor, target_tensor, axis=-1)
        distance_tensor.eval()

  def testInvalidNegativeMultinomialDistribution(self):
    source_tensor = np.array([[1, 0, 0], [-0.1, 0.3, 0.8]])
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]])
    with self.cached_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        distance_tensor = distances.kl_divergence(
            source_tensor, target_tensor, axis=-1)
        distance_tensor.eval()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        distance_tensor = distances.jensen_shannon_divergence(
            source_tensor, target_tensor, axis=-1)
        distance_tensor.eval()

  def testInvalidAxisForDivergence(self):
    source_tensor = np.array([[1, 0, 0], [1, 0, 0]])
    target_tensor = np.array([[1, 0, 0], [0, 1, 0]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        distance_tensor = distances.kl_divergence(
            source_tensor, target_tensor, axis=2)
        distance_tensor.eval()
      with self.assertRaises(ValueError):
        distance_tensor = distances.jensen_shannon_divergence(
            source_tensor, target_tensor, axis=2)
        distance_tensor.eval()

  def testL1Distance(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    distance_config = configs.DistanceConfig('l1', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, 3.25)

  def testL2Distance(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    distance_config = configs.DistanceConfig('l2', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, 10.25)

  def testCosineDistance(self):
    source_tensor = tf.constant([[1, 1], [1, 1], [3, 4], [-1, -1]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [5, 5], [4, 3], [1, 1]],
                                dtype='float32')
    distance_config = configs.DistanceConfig('cosine', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value,
                          0.51)  # sum([0.0, 1.0, 0.04, 2.0]) / 4

  def testJensenShannonDistance(self):
    source_tensor = np.array([[1, 0, 0], [0.1, 0.2, 0.7]], dtype='float32')
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]], dtype='float32')
    expected_tensor = np.sum(self._jsd_func(source_tensor, target_tensor), -1)
    expected_value = np.mean(expected_tensor)
    distance_config = configs.DistanceConfig(
        'jensen_shannon_divergence', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        tf.constant(source_tensor),
        tf.constant(target_tensor),
        distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testJensenShannonDistanceFromLogit(self):
    source = np.array([[1, 2, 3], [1, -1, 2]], dtype='float32')
    target = np.array([[1, 2, 3], [1, 0, -1]], dtype='float32')

    expected_value = np.mean(
        np.sum(
            self._jsd_func(
                self._softmax_func(source), self._softmax_func(target)), -1))

    distance_config = configs.DistanceConfig(
        'jensen_shannon_divergence', transform_fn='softmax', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        tf.constant(source),
        tf.constant(target),
        distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testKLDistance(self):
    source_tensor = np.array([[1, 0, 0], [0.1, 0.2, 0.7]], dtype='float32')
    target_tensor = np.array([[1, 0, 0], [0.1, 0.9, 0]], dtype='float32')

    expected_value = np.mean(
        np.sum(self._kl_func(source_tensor, target_tensor), -1))

    distance_config = configs.DistanceConfig('kl_divergence', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        tf.constant(source_tensor),
        tf.constant(target_tensor),
        distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testKLDistanceFromLogit(self):
    source = np.array([[1, 2, 3], [1, -1, 2]], dtype='float32')
    target = np.array([[1, 2, 3], [1, 0, -1]], dtype='float32')

    expected_value = np.mean(
        np.sum(
            self._kl_func(
                self._softmax_func(source), self._softmax_func(target)), -1))

    distance_config = configs.DistanceConfig(
        'kl_divergence', transform_fn='softmax', sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        tf.constant(source),
        tf.constant(target),
        distance_config=distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, expected_value)

  def testWeightedDistance(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    weights = tf.constant([[1], [0], [0.5], [0.5]], dtype='float32')

    l1_distance_config = configs.DistanceConfig('l1', sum_over_axis=-1)
    l1_distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, l1_distance_config)
    l2_distance_config = configs.DistanceConfig('l2', sum_over_axis=-1)
    l2_distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, l2_distance_config)
    with self.cached_session() as sess:
      l1_distance_value = sess.run(l1_distance_tensor)
      self.assertAllClose(l1_distance_value, 5.5 / 3)
      l2_distance_value = sess.run(l2_distance_tensor)
      self.assertAllClose(l2_distance_value, 18.5 / 3)

  def testDistanceReductionNone(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    weights = tf.constant([[1], [0], [0.5], [0.5]], dtype='float32')

    distance_config = configs.DistanceConfig(
        'l1', tf.compat.v1.losses.Reduction.NONE, sum_over_axis=-1)
    distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value,
                          [[0.0, 0.0], [0.0, 0.0], [2.0, 1.0], [2.0, 0.5]])

  def testDistanceReductionSum(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    weights = tf.constant([[1], [0], [0.5], [0.5]], dtype='float32')

    distance_sum_config = configs.DistanceConfig(
        'l1', tf.compat.v1.losses.Reduction.SUM, sum_over_axis=-1)
    distance_sum_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, distance_sum_config)
    with self.cached_session() as sess:
      distance_sum_value = sess.run(distance_sum_tensor)
      self.assertAllClose(distance_sum_value, 5.5)

  def testDistanceReductionMean(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    weights = tf.constant([[1], [0], [0.5], [0.5]], dtype='float32')

    distance_mean_config = configs.DistanceConfig(
        'l1', tf.compat.v1.losses.Reduction.MEAN, sum_over_axis=-1)
    distance_mean_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, distance_mean_config)
    with self.cached_session() as sess:
      distance_mean_value = sess.run(distance_mean_tensor)
      self.assertAllClose(distance_mean_value, 5.5 / 2.0)

  def testDistanceWithoutSumOverAxis(self):
    source_tensor = tf.constant([[1, 1], [2, 2], [0, 2], [5, 5]],
                                dtype='float32')
    target_tensor = tf.constant([[1, 1], [0, 2], [4, 4], [1, 4]],
                                dtype='float32')
    weights = tf.constant([[1], [0], [0.5], [0.5]], dtype='float32')

    distance_config = configs.DistanceConfig('l1')
    distance_tensor = distances.pairwise_distance_wrapper(
        source_tensor, target_tensor, weights, distance_config)
    with self.cached_session() as sess:
      distance_value = sess.run(distance_tensor)
      self.assertAllClose(distance_value, 5.5 / 6)

  def testDistanceWithTransformButNoSumOverAxis(self):
    source = np.array([[1, 1], [2, 2], [0, 2], [10, -10]], dtype='float32')
    target = np.array([[0, 0], [0, 2], [1, 3], [3, 3]], dtype='float32')

    distance_config = configs.DistanceConfig(
        distance_type='l1',
        reduction=tf.compat.v1.losses.Reduction.NONE,
        transform_fn='softmax')
    distance_tensor = distances.pairwise_distance_wrapper(
        tf.constant(source),
        tf.constant(target),
        distance_config=distance_config)

    expected_distance = np.abs(
        self._softmax_func(source) - self._softmax_func(target))
    with self.cached_session() as sess:
      distance = sess.run(distance_tensor)
      self.assertAllClose(distance, expected_distance)

  def testDistanceInvalidWeightShapeAlongAxis(self):
    source_tensor = tf.constant(1.0, dtype='float32', shape=[4, 2])
    target_tensor = tf.constant(1.0, dtype='float32', shape=[4, 2])
    weights = tf.constant(1.0, dtype='float32', shape=[4, 2])

    distance_config = configs.DistanceConfig(sum_over_axis=-1)
    with self.assertRaises(ValueError):
      distance_tensor = distances.pairwise_distance_wrapper(
          source_tensor, target_tensor, weights, distance_config)
      distance_tensor.eval()

  def testDistanceInvalidAxis(self):
    source_tensor = tf.constant(1.0, dtype='float32', shape=[4, 2])
    target_tensor = tf.constant(1.0, dtype='float32', shape=[4, 2])
    weights = tf.constant(1.0, dtype='float32', shape=[4, 2])

    distance_config = configs.DistanceConfig(sum_over_axis=2)
    with self.assertRaises(ValueError):
      distance_tensor = distances.pairwise_distance_wrapper(
          source_tensor, target_tensor, weights, distance_config)
      distance_tensor.eval()


if __name__ == '__main__':
  tf.test.main()
