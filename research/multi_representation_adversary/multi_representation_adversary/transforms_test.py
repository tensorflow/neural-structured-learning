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

"""Tests for the transforms library."""

from multi_representation_adversary import transforms
import numpy as np
import tensorflow.compat.v2 as tf


class TransformsTest(tf.test.TestCase):

  def test_dct(self):
    x = tf.reshape(tf.constant([[.2, .5, .8], [.7, .4, .1], [.6, .9, .3]]),
                   shape=[1, 3, 3, 1])
    dct_x = transforms.dct(x)

    # Calculated by a reference implementation.
    expected = np.reshape(
        np.array([
            [1.5, 0.12247449, -0.21213203],
            [-0.12247449, -0.45, 0.25980762],
            [0.21213203, -0.4330127, -0.15],
        ]),
        [1, 3, 3, 1],
    )
    self.assertAllClose(dct_x, expected)

  def test_idct(self):
    x = tf.reshape(tf.constant([[.2, .5, .8], [.7, .4, .1], [.6, .9, .3]]),
                   shape=[1, 3, 3, 1])
    idct_x = transforms.idct(x)

    # Calculated by a reference implementation.
    expected = np.reshape(
        np.array([
            [1.42522291, -0.04099682, 0.09735938],
            [-0.44310533, -0.39329966, 0.18787686],
            [0.39594028, -0.49707437, -0.13192325],
        ]),
        [1, 3, 3, 1],
    )
    self.assertAllClose(idct_x, expected)

  def test_inverse(self):
    x = tf.reshape(tf.constant([[.2, .5, .8], [.7, .4, .1], [.6, .9, .3]]),
                   shape=[1, 3, 3, 1])
    self.assertAllClose(transforms.idct(transforms.dct(x)), x)


if __name__ == "__main__":
  tf.test.main()
