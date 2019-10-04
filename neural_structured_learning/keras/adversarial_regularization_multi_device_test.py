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
"""Tests for neural_structured_learning.keras.adversarial_regularization.

The test cases here runs on multiple virtual devices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs
from neural_structured_learning.keras import adversarial_regularization
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

NUM_REPLICAS = 2


def _set_up_virtual_devices():
  """Sets up virtual CPU or GPU devices.

  This function has to be called before any TF ops has run, and can only be
  called at such moment. The setting is effective for all test classes and
  methods in this file.
  """
  if tf.test.is_gpu_available():
    device_type = 'GPU'
    kwargs = {'memory_limit': 1000}  # 1G ram on each virtual GPU
  else:
    device_type = 'CPU'
    kwargs = {}
  physical_devices = tf.config.experimental.list_physical_devices(device_type)
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0], [
          tf.config.experimental.VirtualDeviceConfiguration(**kwargs)
          for _ in range(NUM_REPLICAS)
      ])


def build_linear_keras_functional_model(input_shape, weights):
  inputs = keras.Input(shape=input_shape, name='feature')
  layer = keras.layers.Dense(
      1,
      use_bias=False,
      kernel_initializer=keras.initializers.Constant(weights))
  outputs = layer(inputs)
  return keras.Model(inputs=inputs, outputs=outputs)


class AdversarialRegularizationMultiDeviceTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AdversarialRegularizationMultiDeviceTest, cls).setUpClass()
    _set_up_virtual_devices()

  def _set_up_linear_regression(self, sample_weight=1.0):
    w = np.array([[4.0], [-3.0]])
    x0 = np.tile(np.array([[2.0, 3.0]]), (NUM_REPLICAS, 1))
    y0 = np.tile(np.array([[0.0]]), (NUM_REPLICAS, 1))
    adv_multiplier = 0.2
    adv_step_size = 0.01
    learning_rate = 0.01
    adv_config = configs.make_adv_reg_config(
        multiplier=adv_multiplier,
        adv_step_size=adv_step_size,
        adv_grad_norm='infinity')
    y_hat = np.dot(x0, w)
    loss = (y_hat - y0) / NUM_REPLICAS
    x_adv = x0 + adv_step_size * np.sign((y_hat - y0) * w.T)
    y_hat_adv = np.dot(x_adv, w)
    loss_adv = (y_hat_adv - y0) / NUM_REPLICAS
    grad_w_labeled_loss = sample_weight * 2. * np.matmul(x0.T, loss)
    grad_w_adv_loss = adv_multiplier * sample_weight * 2. * np.matmul(
        x_adv.T, loss_adv)
    w_new = w - learning_rate * (grad_w_labeled_loss + grad_w_adv_loss)
    return w, x0, y0, learning_rate, adv_config, w_new

  def _get_mirrored_strategy(self):
    device_type = 'GPU' if tf.test.is_gpu_available() else 'CPU'
    devices = ['{}:{}'.format(device_type, i) for i in range(NUM_REPLICAS)]
    return tf.distribute.MirroredStrategy(devices)

  def test_train_with_distribution_strategy(self):
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()
    inputs = tf.data.Dataset.from_tensor_slices({
        'feature': x0,
        'label': y0
    }).batch(NUM_REPLICAS)

    strategy = self._get_mirrored_strategy()
    with strategy.scope():
      # Makes sure we are running on multiple devices.
      self.assertEqual(NUM_REPLICAS, strategy.num_replicas_in_sync)
      model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
      adv_model = adversarial_regularization.AdversarialRegularization(
          model, label_keys=['label'], adv_config=adv_config)
      adv_model.compile(optimizer=keras.optimizers.SGD(lr), loss='MSE')

    adv_model.fit(x=inputs)

    # The updated weight should be the same regardless of the number of devices.
    self.assertAllClose(w_new, keras.backend.get_value(model.weights[0]))

  def test_train_with_loss_object(self):
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()
    inputs = tf.data.Dataset.from_tensor_slices({
        'feature': x0,
        'label': y0
    }).batch(NUM_REPLICAS)

    strategy = self._get_mirrored_strategy()
    with strategy.scope():
      model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
      adv_model = adversarial_regularization.AdversarialRegularization(
          model, label_keys=['label'], adv_config=adv_config)
      adv_model.compile(
          optimizer=keras.optimizers.SGD(lr),
          loss=tf.keras.losses.MeanSquaredError())
    adv_model.fit(x=inputs)

    self.assertAllClose(w_new, keras.backend.get_value(model.weights[0]))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
