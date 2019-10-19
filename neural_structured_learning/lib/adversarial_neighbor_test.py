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

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import adversarial_neighbor as adv_lib
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class GenAdvNeighborTest(tf.test.TestCase):

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
    # perturbation = adv_step_size * normalize(gradient) = 0.025 in each dim
    perturbation = 0.025
    self.assertAllClose(x['f1'] + perturbation, actual_neighbor['f1'])
    self.assertAllClose(x['f2'] + perturbation, actual_neighbor['f2'])


if __name__ == '__main__':
  tf.test.main()
