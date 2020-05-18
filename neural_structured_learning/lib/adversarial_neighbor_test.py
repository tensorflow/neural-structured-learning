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
"""Tests for neural_structured_learning.lib.gen_adv_neighbor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import neural_structured_learning.configs as configs
from neural_structured_learning.lib import adversarial_neighbor as adv_lib
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def call_gen_adv_neighbor_with_gradient_tape(x, loss_fn, adv_config):
  with tf.GradientTape() as tape:
    tape.watch(x)
    loss = loss_fn(x)
  adv_neighbor, _ = adv_lib.gen_adv_neighbor(
      x, loss, adv_config, gradient_tape=tape)
  return adv_neighbor


def call_gen_adv_neighbor_with_tf_function(x, loss_fn, adv_config):

  @tf.function
  def gen_adv_neighbor(x):
    loss = loss_fn(x)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
    return adv_neighbor

  return gen_adv_neighbor(x)


def call_multi_iter_gen_adv_neighbor_with_gradient_tape(x, y, model_fn, loss_fn,
                                                        adv_config):
  with tf.GradientTape() as tape:
    tape.watch(x)
    loss = loss_fn(y, model_fn(x))
  adv_neighbor, _ = adv_lib.gen_adv_neighbor(
      x,
      loss,
      adv_config,
      gradient_tape=tape,
      pgd_labels=y,
      pgd_model_fn=model_fn,
      pgd_loss_fn=loss_fn)
  return adv_neighbor


def call_multi_iter_gen_adv_neighbor_with_tf_function(x, y, model_fn, loss_fn,
                                                      adv_config):

  @tf.function
  def gen_adv_neighbor(x):
    loss = loss_fn(y, model_fn(x))
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_labels=y,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn)
    return adv_neighbor

  return gen_adv_neighbor(x)


class GenAdvNeighborTest(tf.test.TestCase, parameterized.TestCase):

  def test_gen_adv_neighbor_for_single_tensor_feature(self):
    # Simple linear regression
    x = tf.constant([[-1.0, 1.0]])
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])
    with tf.GradientTape() as tape:
      tape.watch(x)
      y_hat = tf.matmul(x, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=tf.constant(1.0), adv_step_size=0.1, adv_grad_norm='l2')
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor = [[-1.0 + 0.1 * 0.6, 1.0 + 0.1 * 0.8]]
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def test_multi_iter_gen_adv_neighbor_for_single_tensor_feature(self):
    # Simple linear regression.
    x = tf.constant([[-1.0, 1.0]])
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])
    loss_fn = tf.math.squared_difference
    model_fn = lambda inp: tf.matmul(inp, w)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y_hat = model_fn(x)
      loss = loss_fn(y, y_hat)
    adv_config = configs.AdvNeighborConfig(
        feature_mask=tf.constant(1.0),
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    # Take two steps in the gradient direction.
    expected_neighbor = [[-1.0 + 0.1 * 0.6 * 2, 1.0 + 0.1 * 0.8 * 2]]
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def test_multi_iter_gen_adv_neighbor_proj_limits(self):
    # Simple linear regression.
    x = tf.constant([[-1.0, 1.0]])
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])
    loss_fn = tf.math.squared_difference
    model_fn = lambda input: tf.matmul(input, w)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y_hat = model_fn(x)
      loss = loss_fn(y, y_hat)
    adv_config = configs.AdvNeighborConfig(
        feature_mask=tf.constant(1.0),
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.15)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    # Take two steps in the gradient direction. Project back onto epsilon ball
    # after iteration 2.
    expected_neighbor = [[-1.0 + 0.1 * 0.6 * 1.5, 1.0 + 0.1 * 0.8 * 1.5]]
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def test_gen_adv_neighbor_for_tensor_list(self):
    x = [tf.constant([[-1.0]]), tf.constant([[1.0]])]
    y = tf.constant([0.0])
    w = [tf.constant([[3.0]]), tf.constant([[4.0]])]
    with tf.GradientTape() as tape:
      tape.watch(x)
      y_hat = tf.matmul(x[0], w[0]) + tf.matmul(x[1], w[1])
      loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=tf.constant(1.0), adv_step_size=0.1, adv_grad_norm='l2')
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor = [[[-1.0 + 0.1 * 0.6]], [[1.0 + 0.1 * 0.8]]]
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def test_multi_iter_gen_adv_neighbor_for_tensor_list(self):
    x = [tf.constant([[-1.0]]), tf.constant([[1.0]])]
    y = tf.constant([0.0])
    w = [tf.constant([[3.0]]), tf.constant([[4.0]])]
    loss_fn = tf.math.squared_difference
    model_fn = (lambda inp: tf.matmul(inp[0], w[0]) + tf.matmul(inp[1], w[1]))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y_hat = tf.matmul(x[0], w[0]) + tf.matmul(x[1], w[1])
      loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=tf.constant(1.0),
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor = [[[-1.0 + 0.1 * 0.6 * 2]], [[1.0 + 0.1 * 0.8 * 2]]]
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def _test_gen_adv_neighbor_for_feature_columns_setup(self):
    # For linear regression
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor_fc1 = [[-1.0 + 0.1 * 0.6]]
    expected_neighbor_fc2 = [[1.0 + 0.1 * 0.8]]
    adv_config = configs.AdvNeighborConfig(
        feature_mask={}, adv_step_size=0.1, adv_grad_norm='l2')
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_for_feature_columns(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_gen_adv_neighbor_for_feature_columns_setup())

    # Simple linear regression
    x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])

  def _test_multi_iter_gen_adv_neighbor_for_feature_columns_setup(self):
    # For linear regression
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    # Two iterations of 0.6 * 0.1, clipped.
    expected_neighbor_fc1 = [[-1.0 + 0.13 * 0.6]]
    # Two iterations of 0.8 * 0.1, clipped.
    expected_neighbor_fc2 = [[1.0 + 0.13 * 0.8]]
    adv_config = configs.AdvNeighborConfig(
        feature_mask={},
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.13)
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config

  @test_util.deprecated_graph_mode_only
  def test_multi_iter_gen_adv_neighbor_for_feature_columns(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_multi_iter_gen_adv_neighbor_for_feature_columns_setup())
    x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
    model_fn = (
        lambda inp: tf.matmul(tf.concat([inp['fc1'], inp['fc2']], axis=1), w))
    loss_fn = tf.math.squared_difference
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])

  def test_gen_adv_neighbor_for_feature_columns_v2(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_gen_adv_neighbor_for_feature_columns_setup())

    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression
      x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])

  def test_multi_iter_gen_adv_neighbor_for_feature_columns_v2(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_multi_iter_gen_adv_neighbor_for_feature_columns_setup())
    loss_fn = tf.math.squared_difference
    model_fn = (
        lambda inp: tf.matmul(tf.concat([inp['fc1'], inp['fc2']], axis=1), w))
    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression
      x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])

  def _test_gen_adv_neighbor_for_feature_columns_with_unused_setup(self):
    # For linear regression.
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
        'fc_unused': tf.constant([[0.0]]),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor_fc1 = [[-1.0 + 0.1 * 0.6]]
    expected_neighbor_fc2 = [[1.0 + 0.1 * 0.8]]
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_for_feature_columns_with_unused(self):
    # Simple linear regression
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_gen_adv_neighbor_for_feature_columns_with_unused_setup())
    x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[0.0]], actual_neighbor['fc_unused'])

  def test_gen_adv_neighbor_for_feature_columns_with_unused_v2(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self._test_gen_adv_neighbor_for_feature_columns_with_unused_setup())

    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression
      x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[0.0]], actual_neighbor['fc_unused'])

  def _test_multi_iter_gen_adv_neighbor_for_feature_columns_with_unused_setup(
      self):
    # For linear regression.
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
        'fc_unused': tf.constant([[0.0]]),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])

    # gradient = [[6, 8]], normalized gradient = [[0.6, 0.8]]
    # Two iterations of 0.6 * 0.1, clipped.
    expected_neighbor_fc1 = [[-1.0 + 0.13 * 0.6]]
    # Two iterations of 0.8 * 0.1, clipped.
    expected_neighbor_fc2 = [[1.0 + 0.13 * 0.8]]
    adv_config = configs.AdvNeighborConfig(
        feature_mask={},
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.13)
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config

  @test_util.deprecated_graph_mode_only
  def test_multi_iter_gen_adv_neighbor_for_feature_columns_with_unused(self):
    # Simple linear regression
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self.
        _test_multi_iter_gen_adv_neighbor_for_feature_columns_with_unused_setup(
        ))
    x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
    model_fn = (
        lambda inp: tf.matmul(tf.concat([inp['fc1'], inp['fc2']], axis=1), w))
    loss_fn = tf.math.squared_difference
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[0.0]], actual_neighbor['fc_unused'])

  def test_multi_iter_gen_adv_neighbor_for_feature_columns_with_unused_v2(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2, adv_config = (
        self.
        _test_multi_iter_gen_adv_neighbor_for_feature_columns_with_unused_setup(
        ))

    loss_fn = tf.math.squared_difference
    model_fn = (
        lambda inp: tf.matmul(tf.concat([inp['fc1'], inp['fc2']], axis=1), w))
    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression
      x_stacked = tf.concat([x['fc1'], x['fc2']], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[0.0]], actual_neighbor['fc_unused'])

  def _test_gen_adv_neighbor_for_feature_columns_with_int_feature(self):
    # For linear regression
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
        'fc_int': tf.constant([[2]], dtype=tf.dtypes.int32),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0], [1.0]])
    # gradient = [[18, 24]], normalized gradient = [[0.6, 0.8]]
    expected_neighbor_fc1 = [[-1.0 + 0.1 * 0.6]]
    expected_neighbor_fc2 = [[1.0 + 0.1 * 0.8]]
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_for_feature_columns_with_int_feature(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2 = (
        self._test_gen_adv_neighbor_for_feature_columns_with_int_feature())

    # Simple linear regression.
    x_stacked = tf.concat(
        [x['fc1'], x['fc2'],
         tf.cast(x['fc_int'], tf.dtypes.float32)], axis=1)
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[2]], actual_neighbor['fc_int'])

  def test_gen_adv_neighbor_for_feature_columns_with_int_feature_v2(self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2 = (
        self._test_gen_adv_neighbor_for_feature_columns_with_int_feature())

    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression.
      x_stacked = tf.concat(
          [x['fc1'], x['fc2'],
           tf.cast(x['fc_int'], tf.dtypes.float32)], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[2]], actual_neighbor['fc_int'])

  def _test_multi_iter_gen_adv_neighbor_for_feature_columns_with_int_feature(
      self):
    # For linear regression
    x = {
        'fc1': tf.constant([[-1.0]]),
        'fc2': tf.constant([[1.0]]),
        'fc_int': tf.constant([[2]], dtype=tf.dtypes.int32),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0], [1.0]])
    # Two iterations of 0.6 * 0.1, clipped.
    expected_neighbor_fc1 = [[-1.0 + 0.13 * 0.6]]
    # Two iterations of 0.8 * 0.1, clipped.
    expected_neighbor_fc2 = [[1.0 + 0.13 * 0.8]]
    return x, y, w, expected_neighbor_fc1, expected_neighbor_fc2

  @test_util.deprecated_graph_mode_only
  def test_multi_iter_gen_adv_neighbor_for_feature_columns_with_int_feature(
      self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2 = (
        self
        ._test_multi_iter_gen_adv_neighbor_for_feature_columns_with_int_feature(
        ))

    # Simple linear regression.
    x_stacked = tf.concat(
        [x['fc1'], x['fc2'],
         tf.cast(x['fc_int'], tf.dtypes.float32)], axis=1)
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)
    loss_fn = tf.math.squared_difference

    @tf.function
    def model_fn(inp):
      x_stacked = tf.concat(
          [inp['fc1'], inp['fc2'],
           tf.cast(inp['fc_int'], tf.dtypes.float32)],
          axis=1)
      return tf.matmul(x_stacked, w)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=None,
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.13)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[2]], actual_neighbor['fc_int'])

  def test_multi_iter_gen_adv_neighbor_for_feature_columns_with_int_feature_v2(
      self):
    x, y, w, expected_neighbor_fc1, expected_neighbor_fc2 = (
        self
        ._test_multi_iter_gen_adv_neighbor_for_feature_columns_with_int_feature(
        ))
    loss_fn = tf.math.squared_difference

    @tf.function
    def model_fn(inp):
      x_stacked = tf.concat(
          [inp['fc1'], inp['fc2'],
           tf.cast(inp['fc_int'], tf.dtypes.float32)],
          axis=1)
      return tf.matmul(x_stacked, w)

    with tf.GradientTape() as tape:
      tape.watch(x)
      # Simple linear regression.
      x_stacked = tf.concat(
          [x['fc1'], x['fc2'],
           tf.cast(x['fc_int'], tf.dtypes.float32)], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    adv_config = configs.AdvNeighborConfig(
        feature_mask=None,
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.13)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)

    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(expected_neighbor_fc1, actual_neighbor['fc1'])
    self.assertAllClose(expected_neighbor_fc2, actual_neighbor['fc2'])
    self.assertAllClose([[2]], actual_neighbor['fc_int'])

  def _test_gen_adv_neighbor_should_ignore_sparse_tensors_setup(self):
    # sparse_feature represents [[1, 0, 2], [0, 3, 0]].
    sparse_feature = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1]],
        values=[1.0, 2.0, 3.0],
        dense_shape=(2, 3))
    x = {'sparse': sparse_feature, 'dense': tf.constant([[-1.0], [1.0]])}
    w_sparse = tf.constant([[5.0], [4.0], [3.0]])
    w_dense = tf.constant([[6.0]])
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, w_sparse, w_dense, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_should_ignore_sparse_tensors(self):
    x, w_sparse, w_dense, adv_config = (
        self._test_gen_adv_neighbor_should_ignore_sparse_tensors_setup())
    prod_sparse = tf.sparse.reduce_sum(w_sparse * x['sparse'])
    prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
    loss = prod_dense + prod_sparse

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)

    # No perturbation on sparse features.
    actual_sparse = actual_neighbor['sparse']
    self.assertAllEqual(x['sparse'].dense_shape, actual_sparse.dense_shape)
    self.assertAllEqual(x['sparse'].indices, actual_sparse.indices)
    self.assertAllClose(x['sparse'].values, actual_sparse.values)
    # gradient = w_dense
    # perturbation = adv_step_size * sign(w_dense) = 0.1
    self.assertAllClose([[-0.9], [1.1]], actual_neighbor['dense'])

  def test_gen_adv_neighbor_should_ignore_sparse_tensors_v2(self):
    x, w_sparse, w_dense, adv_config = (
        self._test_gen_adv_neighbor_should_ignore_sparse_tensors_setup())

    with tf.GradientTape() as tape:
      tape.watch([x['sparse'].values, x['dense']])
      prod_sparse = tf.reduce_sum(
          tf.sparse.sparse_dense_matmul(x['sparse'], w_sparse))
      prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
      loss = prod_dense + prod_sparse

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)
    actual_neighbor = self.evaluate(adv_neighbor)

    # No perturbation on sparse features.
    actual_sparse = actual_neighbor['sparse']
    self.assertAllEqual(x['sparse'].dense_shape, actual_sparse.dense_shape)
    self.assertAllEqual(x['sparse'].indices, actual_sparse.indices)
    self.assertAllClose(x['sparse'].values, actual_sparse.values)
    # gradient = w_dense
    # perturbation = adv_step_size * sign(w_dense) = 0.1
    self.assertAllClose([[-0.9], [1.1]], actual_neighbor['dense'])

  def _test_multi_iter_gen_adv_neighbor_should_ignore_sparse_tensors_setup(
      self):

    @tf.function
    def model_fn(inp):
      prod_sparse = tf.reduce_sum(
          tf.sparse.sparse_dense_matmul(inp['sparse'], w_sparse))
      prod_dense = tf.reduce_sum(input_tensor=w_dense * inp['dense'])
      return prod_sparse + prod_dense

    @tf.function
    def loss_fn(label, pred):
      return tf.abs(label - pred)

    # sparse_feature represents [[1, 0, 2], [0, 3, 0]].
    sparse_feature = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1]],
        values=[1.0, 2.0, 3.0],
        dense_shape=(2, 3))
    x = {'sparse': sparse_feature, 'dense': tf.constant([[-1.0], [1.0]])}
    w_sparse = tf.constant([[5.0], [4.0], [3.0]])
    w_dense = tf.constant([[6.0]])
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None,
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.13)
    return x, w_sparse, w_dense, adv_config, model_fn, loss_fn

  @test_util.deprecated_graph_mode_only
  def test_multi_iter_gen_adv_neighbor_should_ignore_sparse_tensors(self):
    x, w_sparse, w_dense, adv_config, model_fn, loss_fn = (
        self
        ._test_multi_iter_gen_adv_neighbor_should_ignore_sparse_tensors_setup())
    y = tf.constant([0.0])

    prod_sparse = tf.reduce_sum(
        tf.sparse.sparse_dense_matmul(x['sparse'], w_sparse))
    prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
    loss = tf.abs(y - (prod_dense + prod_sparse))

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_labels=y,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn)
    actual_neighbor = self.evaluate(adv_neighbor)

    # No perturbation on sparse features.
    actual_sparse = actual_neighbor['sparse']
    self.assertAllEqual(x['sparse'].dense_shape, actual_sparse.dense_shape)
    self.assertAllEqual(x['sparse'].indices, actual_sparse.indices)
    self.assertAllClose(x['sparse'].values, actual_sparse.values)
    # gradient = w_dense
    # perturbation = adv_step_size * sign(w_dense) * 2 = 0.2, clipped to 0.13.
    self.assertAllClose([[-0.87], [1.13]], actual_neighbor['dense'])

  def test_multi_iter_gen_adv_neighbor_should_ignore_sparse_tensors_v2(self):
    x, w_sparse, w_dense, adv_config, model_fn, loss_fn = (
        self
        ._test_multi_iter_gen_adv_neighbor_should_ignore_sparse_tensors_setup())
    y = tf.constant([0.0])

    with tf.GradientTape() as tape:
      tape.watch([x['sparse'].values, x['dense']])
      prod_sparse = tf.reduce_sum(
          tf.sparse.sparse_dense_matmul(x['sparse'], w_sparse))
      prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
      loss = prod_dense + prod_sparse

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_labels=y,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        gradient_tape=tape)
    actual_neighbor = self.evaluate(adv_neighbor)

    # No perturbation on sparse features.
    actual_sparse = actual_neighbor['sparse']
    self.assertAllEqual(x['sparse'].dense_shape, actual_sparse.dense_shape)
    self.assertAllEqual(x['sparse'].indices, actual_sparse.indices)
    self.assertAllClose(x['sparse'].values, actual_sparse.values)
    # gradient = w_dense
    # perturbation = adv_step_size * sign(w_dense) * 2 = 0.2, clipped to 0.13.
    self.assertAllClose([[-0.87], [1.13]], actual_neighbor['dense'])

  def _test_gen_adv_neighbor_to_raise_for_sparse_tensors_setup(self):
    sparse_feature = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1]],
        values=[1.0, 2.0, 3.0],
        dense_shape=(2, 3))
    x = {'sparse': sparse_feature, 'dense': tf.constant([[-1.0], [1.0]])}
    w_sparse = tf.constant([[5.0], [4.0], [3.0]])
    w_dense = tf.constant([[6.0]])
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, w_sparse, w_dense, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_to_raise_for_sparse_tensors(self):
    x, w_sparse, w_dense, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_sparse_tensors_setup())
    prod_sparse = tf.sparse.reduce_sum(w_sparse * x['sparse'])
    prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
    loss = prod_dense + prod_sparse

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*sparse'):
      adv_lib.gen_adv_neighbor(x, loss, adv_config, raise_invalid_gradient=True)

  def test_gen_adv_neighbor_to_raise_for_sparse_tensors_v2(self):
    x, w_sparse, w_dense, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_sparse_tensors_setup())

    with tf.GradientTape() as tape:
      tape.watch([x['sparse'].values, x['dense']])
      prod_sparse = tf.reduce_sum(
          tf.sparse.sparse_dense_matmul(x['sparse'], w_sparse))
      prod_dense = tf.reduce_sum(input_tensor=w_dense * x['dense'])
      loss = prod_dense + prod_sparse

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*sparse'):
      adv_lib.gen_adv_neighbor(
          x, loss, adv_config, raise_invalid_gradient=True, gradient_tape=tape)

  def _test_gen_adv_neighbor_to_raise_for_nondifferentiable_input_setup(self):
    x = {
        'float': tf.constant([[-1.0]]),
        'int': tf.constant([[2]], dtype=tf.dtypes.int32),
    }
    y = tf.constant([0.0])
    w = tf.constant([[3.0], [4.0]])
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, y, w, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_to_raise_for_nondifferentiable_input(self):
    x, y, w, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_nondifferentiable_input_setup(
        ))
    x_stacked = tf.concat(
        [x['float'], tf.cast(x['int'], tf.dtypes.float32)], axis=1)
    y_hat = tf.matmul(x_stacked, w)
    loss = tf.math.squared_difference(y, y_hat)

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*int'):
      adv_lib.gen_adv_neighbor(x, loss, adv_config, raise_invalid_gradient=True)

  def test_gen_adv_neighbor_to_raise_for_nondifferentiable_input_v2(self):
    x, y, w, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_nondifferentiable_input_setup(
        ))
    with tf.GradientTape() as tape:
      tape.watch(x)
      x_stacked = tf.concat(
          [x['float'], tf.cast(x['int'], tf.dtypes.float32)], axis=1)
      y_hat = tf.matmul(x_stacked, w)
      loss = tf.math.squared_difference(y, y_hat)

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*int'):
      adv_lib.gen_adv_neighbor(
          x, loss, adv_config, raise_invalid_gradient=True, gradient_tape=tape)

  def _test_gen_adv_neighbor_to_raise_for_disconnected_input_setup(self):
    x = {
        'f1': tf.constant([[1.0]]),
        'f2': tf.constant([[2.0]]),
    }
    w = tf.constant([3.0])
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, w, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_to_raise_for_disconnected_input(self):
    x, w, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_disconnected_input_setup())
    loss = tf.reduce_sum(input_tensor=w * x['f1'])

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*f2'):
      adv_lib.gen_adv_neighbor(x, loss, adv_config, raise_invalid_gradient=True)

  def test_gen_adv_neighbor_to_raise_for_disconnected_input_v2(self):
    x, w, adv_config = (
        self._test_gen_adv_neighbor_to_raise_for_disconnected_input_setup())
    with tf.GradientTape() as tape:
      tape.watch(x)
      loss = tf.reduce_sum(input_tensor=w * x['f1'])

    with self.assertRaisesRegex(ValueError, 'Cannot perturb.*f2'):
      adv_lib.gen_adv_neighbor(
          x, loss, adv_config, raise_invalid_gradient=True, gradient_tape=tape)

  def test_gen_adv_neighbor_not_knowing_input_in_advance(self):

    @tf.function
    def _gen_adv_neighbor(x):
      w = tf.constant([[3.0], [4.0]])
      loss = tf.matmul(x, w)
      adv_config = configs.AdvNeighborConfig(
          feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
      adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
      return adv_neighbor

    x = tf.constant([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
    actual_neighbor = self.evaluate(_gen_adv_neighbor(x))

    # gradient = w
    # perturbation = adv_step_size * normalize(gradient) = [0.06, 0.08]
    expected_neighbor = [[1.06, 1.08], [0.06, 0.08], [-0.94, -0.92]]
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def test_multi_iter_gen_adv_neighbor_not_knowing_input_in_advance(self):

    @tf.function
    def _gen_adv_neighbor(x):
      w = tf.constant([[3.0], [4.0]])
      loss = tf.matmul(x, w)
      y = tf.constant([0.0])
      model_fn = lambda inp: tf.matmul(inp, w)

      @tf.function
      def loss_fn(_, pred):
        return pred

      adv_config = configs.AdvNeighborConfig(
          feature_mask=None,
          adv_step_size=0.1,
          adv_grad_norm='l2',
          epsilon=0.15,
          iterations=2)
      adv_neighbor, _ = adv_lib.gen_adv_neighbor(
          x,
          loss,
          adv_config,
          pgd_labels=y,
          pgd_model_fn=model_fn,
          pgd_loss_fn=loss_fn)
      return adv_neighbor

    x = tf.constant([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
    actual_neighbor = self.evaluate(_gen_adv_neighbor(x))

    # gradient = w
    # perturbation = 2 * adv_step_size * normalize(gradient) = [0.12, 0.16]
    # Clipping perturbation to 0.15 epsilon gives [0.09, 0.12].
    expected_neighbor = [[1.09, 1.12], [0.09, 0.12], [-0.91, -0.88]]
    self.assertAllClose(expected_neighbor, actual_neighbor)

  def _test_gen_adv_neighbor_for_all_input_disconnected_setup(self):
    x = tf.constant([[1.0]])
    loss = tf.constant(1.0)
    adv_config = configs.AdvNeighborConfig()
    return x, loss, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_for_all_input_disconnected(self):
    x, loss, adv_config = (
        self._test_gen_adv_neighbor_for_all_input_disconnected_setup())
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllEqual(x, actual_neighbor)

  def test_gen_adv_neighbor_for_all_input_disconnected_v2(self):
    x, loss, adv_config = (
        self._test_gen_adv_neighbor_for_all_input_disconnected_setup())
    with tf.GradientTape() as tape:
      tape.watch(x)
    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllEqual(x, actual_neighbor)

  def _test_gen_adv_neighbor_for_features_with_different_shapes_setup(self):
    w1 = tf.constant(1.0, shape=(2, 2, 3))
    w2 = tf.constant(1.0, shape=(2, 2))
    f1 = tf.constant([[[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                       [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]],
                      [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    f2 = tf.constant([[[1.2, 1.3], [1.4, 1.5]], [[0.0, 0.0], [0.0, 0.0]]])
    x = {'f1': f1, 'f2': f2}
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None, adv_step_size=0.1, adv_grad_norm='l2')
    return x, w1, w2, adv_config

  @test_util.deprecated_graph_mode_only
  def test_gen_adv_neighbor_for_features_with_different_shapes(self):
    x, w1, w2, adv_config = (
        self._test_gen_adv_neighbor_for_features_with_different_shapes_setup())
    loss = tf.reduce_sum(input_tensor=w1 *
                         x['f1']) + tf.reduce_sum(input_tensor=w2 * x['f2'])

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(x, loss, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w
    # perturbation = adv_step_size * normalize(gradient) = 0.025 in each dim
    perturbation = 0.025
    self.assertAllClose(x['f1'] + perturbation, actual_neighbor['f1'])
    self.assertAllClose(x['f2'] + perturbation, actual_neighbor['f2'])

  def test_gen_adv_neighbor_for_features_with_different_shapes_v2(self):
    x, w1, w2, adv_config = (
        self._test_gen_adv_neighbor_for_features_with_different_shapes_setup())
    with tf.GradientTape() as tape:
      tape.watch(list(x.values()))
      loss = tf.reduce_sum(input_tensor=w1 *
                           x['f1']) + tf.reduce_sum(input_tensor=w2 * x['f2'])

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x, loss, adv_config, gradient_tape=tape)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w
    # perturbation = adv_step_size * normalize(gradient) = 0.05 in each dim
    perturbation = 0.025
    self.assertAllClose(x['f1'] + perturbation, actual_neighbor['f1'])
    self.assertAllClose(x['f2'] + perturbation, actual_neighbor['f2'])

  def _test_multi_iter_gen_adv_neighbor_for_features_with_different_shapes_setup(
      self):
    w1 = tf.constant(1.0, shape=(2, 2, 3))
    w2 = tf.constant(1.0, shape=(2, 2))
    f1 = tf.constant([[[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                       [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]],
                      [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    f2 = tf.constant([[[1.2, 1.3], [1.4, 1.5]], [[0.0, 0.0], [0.0, 0.0]]])
    x = {'f1': f1, 'f2': f2}

    @tf.function
    def model_fn(inp):
      return tf.reduce_sum(input_tensor=w1 * inp['f1']) + tf.reduce_sum(
          input_tensor=w2 * inp['f2'])

    @tf.function
    def loss_fn(_, pred):
      return pred

    adv_config = configs.AdvNeighborConfig(
        feature_mask=None,
        adv_step_size=0.1,
        adv_grad_norm='l2',
        iterations=2,
        epsilon=0.15)
    return x, w1, w2, adv_config, model_fn, loss_fn

  @test_util.deprecated_graph_mode_only
  def test_multi_iter_gen_adv_neighbor_for_features_with_different_shapes(self):
    x, w1, w2, adv_config, model_fn, loss_fn = (
        self.
        _test_multi_iter_gen_adv_neighbor_for_features_with_different_shapes_setup(
        ))
    y = tf.constant([0.0])

    loss = tf.reduce_sum(input_tensor=w1 *
                         x['f1']) + tf.reduce_sum(input_tensor=w2 * x['f2'])

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w
    # perturbation = adv_step_size * normalize(gradient) = 0.025 in each dim
    # Epsilon 0.15, projects back to roughly 0.04330
    # f2 does not need to be clipped.
    perturbation = 0.025 * 2
    projected_perturb = perturbation * 0.15 / tf.sqrt(16 * tf.square(0.05))
    self.assertAllClose(x['f1'] + projected_perturb, actual_neighbor['f1'])
    self.assertAllClose(x['f2'] + projected_perturb, actual_neighbor['f2'])

  def test_multi_iter_gen_adv_neighbor_for_features_with_different_shapes_v2(
      self):
    x, w1, w2, adv_config, model_fn, loss_fn = (
        self.
        _test_multi_iter_gen_adv_neighbor_for_features_with_different_shapes_setup(
        ))
    y = tf.constant([0.0])

    with tf.GradientTape() as tape:
      tape.watch(list(x.values()))
      loss = tf.reduce_sum(input_tensor=w1 *
                           x['f1']) + tf.reduce_sum(input_tensor=w2 * x['f2'])

    adv_neighbor, _ = adv_lib.gen_adv_neighbor(
        x,
        loss,
        adv_config,
        gradient_tape=tape,
        pgd_model_fn=model_fn,
        pgd_loss_fn=loss_fn,
        pgd_labels=y)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w
    # perturbation = adv_step_size * normalize(gradient) = 0.025 in each dim
    # Epsilon 0.15, projects back to roughly 0.04330
    # f2 does not need to be clipped.
    perturbation = 0.025 * 2
    projected_perturb = perturbation * 0.15 / tf.sqrt(16 * tf.square(0.05))
    self.assertAllClose(x['f1'] + projected_perturb, actual_neighbor['f1'])
    self.assertAllClose(x['f2'] + projected_perturb, actual_neighbor['f2'])

  @parameterized.named_parameters([
      ('gradient_tape', call_gen_adv_neighbor_with_gradient_tape),
      ('tf_function', call_gen_adv_neighbor_with_tf_function),
  ])
  def test_gen_adv_neighbor_respects_feature_constraints(
      self, gen_adv_neighbor_fn):
    x = tf.constant([[0.0, 1.0]])
    w = tf.constant([[-1.0, 1.0]])

    loss_fn = lambda x: tf.linalg.matmul(x, w, transpose_b=True)
    adv_config = configs.AdvNeighborConfig(
        feature_mask=None,
        adv_step_size=0.1,
        adv_grad_norm='l2',
        clip_value_min=0.0,
        clip_value_max=1.0)
    adv_neighbor = gen_adv_neighbor_fn(x, loss_fn, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)
    self.assertAllClose(x, actual_neighbor)

  @parameterized.named_parameters([
      ('gradient_tape', call_gen_adv_neighbor_with_gradient_tape),
      ('tf_function', call_gen_adv_neighbor_with_tf_function),
  ])
  def test_gen_adv_neighbor_respects_per_feature_constraints(
      self, gen_adv_neighbor_fn):
    w1 = tf.constant(1.0, shape=(2, 2, 3))
    w2 = tf.constant(-1.0, shape=(2, 2))
    f1 = np.array([[[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                    [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]]],
                  dtype=np.float32)
    f2 = np.array([[[0.0, 0.33], [0.66, 1.0]]], dtype=np.float32)
    x = {'f1': tf.constant(f1), 'f2': tf.constant(f2)}

    def loss_fn(x):
      return tf.reduce_sum(w1 * x['f1']) + tf.reduce_sum(w2 * x['f2'])

    adv_step_size = 0.5
    adv_config = configs.AdvNeighborConfig(
        adv_step_size=adv_step_size,
        adv_grad_norm='infinity',
        clip_value_min={'f2': 0.0},
        clip_value_max={'f1': 1.0})
    adv_neighbor = gen_adv_neighbor_fn(x, loss_fn, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w, perturbation = adv_step_size * sign(w)
    expected_neighbor = {
        'f1': np.minimum(f1 + adv_step_size, 1.0),
        'f2': np.maximum(f2 - adv_step_size, 0.0),
    }
    self.assertAllClose(expected_neighbor, actual_neighbor)

  @parameterized.named_parameters([
      ('gradient_tape', call_multi_iter_gen_adv_neighbor_with_gradient_tape),
      ('tf_function', call_multi_iter_gen_adv_neighbor_with_tf_function),
  ])
  def test_gen_adv_neighbor_multi_iter_respects_feature_constraints(
      self, gen_adv_neighbor_fn):
    w1 = tf.constant(1.0, shape=(2, 2, 3))
    w2 = tf.constant(-1.0, shape=(2, 2))
    f1 = np.array([[[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                    [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]]],
                  dtype=np.float32)
    f2 = np.array([[[0.0, 0.33], [0.66, 1.0]]], dtype=np.float32)
    x = {'f1': tf.constant(f1), 'f2': tf.constant(f2)}
    y = tf.constant([0.0])

    def model_fn(x):
      return tf.reduce_sum(w1 * x['f1']) + tf.reduce_sum(w2 * x['f2'])

    def loss_fn(_, pred):
      return pred

    adv_step_size = 0.5
    adv_config = configs.AdvNeighborConfig(
        adv_step_size=adv_step_size,
        adv_grad_norm='infinity',
        epsilon=0.4,
        iterations=2,
        clip_value_min={'f2': 0.0},
        clip_value_max={'f1': 1.0})
    adv_neighbor = gen_adv_neighbor_fn(x, y, model_fn, loss_fn, adv_config)
    actual_neighbor = self.evaluate(adv_neighbor)

    # gradient = w, perturbation = min(adv_step_size * sign(w), 0.4)
    expected_neighbor = {
        'f1': np.minimum(f1 + 0.4, 1.0),
        'f2': np.maximum(f2 - 0.4, 0.0),
    }
    self.assertAllClose(expected_neighbor, actual_neighbor)


if __name__ == '__main__':
  tf.test.main()
