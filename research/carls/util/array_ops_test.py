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
"""Tests for neural_structured_learning.research.carls.util.array_ops."""

from research.carls.util import array_ops
import tensorflow as tf


class ArrayOpsTest(tf.test.TestCase):

  def test_increment_last_dim(self):
    # 1D case
    input_tensor = tf.constant([2])
    new_tensor = array_ops.increment_last_dim(input_tensor, 1)
    self.assertAllClose([2, 1], new_tensor.numpy())

    # 2D case
    input_tensor = tf.constant([[1, 2], [3, 4]])
    new_tensor = array_ops.increment_last_dim(input_tensor, 10)
    self.assertAllClose([[1, 2, 10], [3, 4, 10]], new_tensor.numpy())


if __name__ == '__main__':
  tf.test.main()
