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
"""Tests for neural_structured_learning.lib.multimodal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import multimodal_lib

import tensorflow as tf


class BimodalIntegrationTest(tf.test.TestCase):
  """Tests bimodal integration methods."""

  def _getInput(self, shape_tuple):
    return tf.keras.backend.placeholder(shape=shape_tuple, dtype='float32')

  def _checkTensorShape(self, tensor, shape):
    self.assertIsInstance(tensor, tf.Tensor)
    self.assertEqual(tensor.get_shape().as_list(), shape)

  def testApplyBimodalIntegration(self):
    """Tests applying bimodal integration."""
    # TODO(ct): Add more comprehensive UnitTests for testing the underlying
    # flow, instead of only testing the output shape.
    x = self._getInput([16, 256])
    y = self._getInput([16, 128])
    add_config = configs.IntegrationConfig('additive', 100)
    layer = multimodal_lib.bimodal_integration(x, y, 50, add_config)
    self._checkTensorShape(layer, [16, 50])
    mul_config = configs.IntegrationConfig('multiplicative', 100)
    layer = multimodal_lib.bimodal_integration(x, y, 20, mul_config)
    self._checkTensorShape(layer, [16, 20])
    tucker_config = configs.IntegrationConfig('tucker_decomp', [100, 100, 5])
    layer = multimodal_lib.bimodal_integration(x, y, 20, tucker_config)
    self._checkTensorShape(layer, [16, 20])

    add_none_config = configs.IntegrationConfig('additive', 100, None)
    layer = multimodal_lib.bimodal_integration(x, y, 50, add_none_config)
    self._checkTensorShape(layer, [16, 50])

    x = self._getInput((16, 1, 256))
    y = self._getInput((16, 1, 128))
    layer = multimodal_lib.bimodal_integration(x, y, 50, add_config)
    self._checkTensorShape(layer, [16, 1, 50])
    layer = multimodal_lib.bimodal_integration(x, y, 20, mul_config)
    self._checkTensorShape(layer, [16, 1, 20])
    layer = multimodal_lib.bimodal_integration(x, y, 20, tucker_config)
    self._checkTensorShape(layer, [16, 1, 20])


if __name__ == '__main__':
  tf.test.main()
