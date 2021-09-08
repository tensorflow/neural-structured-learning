# Copyright 2020 Google LLC
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

"""Tests for the resnet library."""

from multi_representation_adversary import resnet
import tensorflow.compat.v2 as tf


class ResnetTest(tf.test.TestCase):

  def test_build_resnet(self):
    model = resnet.build_resnet_v1(input_shape=(32, 32, 3), depth=50,
                                   num_classes=10, return_logits=True)
    inputs = tf.random.uniform(shape=(4, 32, 32, 3))
    outputs = model(inputs)
    self.assertEqual(outputs.shape, [4, 10])


if __name__ == "__main__":
  tf.test.main()
