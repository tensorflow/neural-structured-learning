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
"""Tests for neural_structured_learning.research.carls.dynamic_normalization."""

from research.carls import context
from research.carls import dynamic_memory_ops as dm_ops
from research.carls import dynamic_normalization as dn
from research.carls.testing import test_util
import numpy as np
import tensorflow as tf


class DynamicNormalizationTest(tf.test.TestCase):

  def setUp(self):
    super(DynamicNormalizationTest, self).setUp()
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(DynamicNormalizationTest, self).tearDown()

  def testDynamicndBatchNormalizationConsistency(self):
    """Checks that BatchNormalization falls into a special case of DN."""
    batch_size = 2
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=batch_size,
        distance_to_cluster_threshold=0.5,
        max_num_clusters=1,
        bootstrap_steps=0)
    inputs = np.array([[1, 2], [3, 4]])
    hidden_layer = tf.keras.layers.Dense(
        5, activation='relu', kernel_initializer='ones')(
            inputs)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=1, center=True, scale=True, momentum=0)(
            hidden_layer, training=True)
    dynamic_norm = dn.DynamicNormalization(
        dm_config,
        mode=dm_ops.LOOKUP_WITH_UPDATE,
        axis=1,
        epsilon=0.001,
        use_batch_normalization=True,
        service_address=self._kbs_address)(
            hidden_layer, training=True)

    self.assertAllClose(batch_norm.numpy(), dynamic_norm.numpy())

  def testKerasModel(self):
    """Tests the tf.keras interface."""
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=1,
        distance_to_cluster_threshold=0.5,
        max_num_clusters=1,
        bootstrap_steps=0)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,), name='inputs'),
        tf.keras.layers.Dense(4),
        dn.DynamicNormalization(
            dm_config,
            mode=dm_ops.LOOKUP_WITH_UPDATE,
            axis=1,
            epsilon=0.1,
            use_batch_normalization=True,
            service_address=self._kbs_address),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mse')

    x_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_train = np.array([1, 1, 0, 0])
    model.fit(x_train, y_train, epochs=10)

  def testTrainingLogistic(self):
    """Trains two logistic regression models with two normalizations."""
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=500,
        distance_to_cluster_threshold=0.9,
        max_num_clusters=1,
        bootstrap_steps=10)
    x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                  [1, 0, 1, 0], [0, 1, 0, 1]])
    y = np.array([[1], [1], [1], [1], [0], [0]])  # 0/1 label
    mode = tf.constant(dm_ops.LOOKUP_WITH_UPDATE, dtype=tf.int32)

    def _create_model(enable_dynamic_normalization: bool):
      hidden_layer = tf.keras.layers.Dense(5, activation='relu')
      if enable_dynamic_normalization:
        normalized_layer = dn.DynamicNormalization(
            dm_config,
            mode=mode,
            axis=1,
            epsilon=0.001,
            use_batch_normalization=False,
            service_address=self._kbs_address)
      else:
        normalized_layer = tf.keras.layers.BatchNormalization(
            axis=1, center=True, scale=True, momentum=0)
      output_layer = tf.keras.layers.Dense(1, kernel_initializer='ones')
      model = tf.keras.Sequential(
          [hidden_layer, normalized_layer, output_layer])
      return model

    def _loss(model, x, y):
      output = model(x)
      pred = 1 / (1 + tf.exp(-output))
      loss = y * tf.math.log(pred) + (1 - y) * tf.math.log(1 - pred)
      loss = tf.reduce_mean(-tf.math.reduce_sum(loss, axis=1))
      return loss

    def _grad(model, x, y):
      with tf.GradientTape() as tape:
        loss_value = _loss(model, x, y)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    bn_model = _create_model(False)  # Model with batch normalization.
    dn_model = _create_model(True)  # Model with dynamic normalization.
    optimizer_bn = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer_dn = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
      bn_loss_value, bn_grads = _grad(bn_model, x, y)
      dn_loss_value, dn_grads = _grad(dn_model, x, y)

      print('Step: {}, BN loss: {}, DN loss: {}'.format(i,
                                                        bn_loss_value.numpy(),
                                                        dn_loss_value.numpy()))
      # Update the trainable variables w.r.t. the logistic loss
      optimizer_bn.apply_gradients(zip(bn_grads, bn_model.trainable_variables))
      optimizer_dn.apply_gradients(zip(dn_grads, dn_model.trainable_variables))

    # Checks that DynamicNormalization consistently outperforms
    # BatchNormalization in terms of finding lower loss.
    self.assertGreater(bn_loss_value.numpy(), dn_loss_value.numpy())


if __name__ == '__main__':
  tf.test.main()
