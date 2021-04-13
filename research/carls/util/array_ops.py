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
"""Array related ops."""

import tensorflow as tf


def increment_last_dim(input_tensor: tf.Tensor,
                       default_value: float) -> tf.Tensor:
  """Grows the size of last dimension of given `input_tensor` by one.

  Examples:
    - [[1, 2], [3, 4]] -> [[1, 2, 1], [3, 4, 1]] (default_value = 1).
    - [1, 2, 3] -> [1, 2, 3, 4] (default_value = 4).

  Args:
    input_tensor: a float tf.Tensor whose last dimension is to be incremented.
    default_value: a float value denoting the default value for the increased
      part.

  Returns:
    A new `tf.Tensor` with increased last dimension size.
  """
  input_tensor = tf.dtypes.cast(input_tensor, tf.float32)
  inc_tensor = tf.ones(tf.shape(input_tensor)[:-1])
  inc_tensor = tf.expand_dims(inc_tensor, -1) * default_value
  return tf.concat([input_tensor, inc_tensor], axis=-1)
