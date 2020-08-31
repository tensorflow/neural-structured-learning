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
"""Euclidean utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def euc_sq_distance(x, y, eval_mode=False):
  """Computes Euclidean squared distance.

  Args:
    x: Tensor of size B1 x d
    y: Tensor of size B2 x d
    eval_mode: boolean indicating whether to compute all pairwise distances or
      not. If eval_mode=False, must have B1=B2.

  Returns:
    Tensor of size B1 x B2 if eval_mode=True, otherwise Tensor of size B1 x 1.
  """
  x2 = tf.math.reduce_sum(x * x, axis=-1, keepdims=True)
  y2 = tf.math.reduce_sum(y * y, axis=-1, keepdims=True)
  if eval_mode:
    y2 = tf.transpose(y2)
    xy = tf.linalg.matmul(x, y, transpose_b=True)
  else:
    xy = tf.math.reduce_sum(x * y, axis=-1, keepdims=True)
  return x2 + y2 - 2 * xy


def givens_reflection(r, x):
  """Applies 2x2 reflections.

  Args:
    r: Tensor of size B x d representing relfection parameters per example.
    x: Tensor of size B x d representing points to reflect.

  Returns:
    Tensor of size B x s representing reflection of x by r.
  """
  batch_size = tf.shape(r)[0]
  givens = tf.reshape(r, (batch_size, -1, 2))
  givens = givens / tf.norm(givens, ord=2, axis=-1, keepdims=True)
  x = tf.reshape(x, (batch_size, -1, 2))
  x_ref = givens[:, :, 0:1] * tf.concat(
      (x[:, :, 0:1], -x[:, :, 1:]), axis=-1) + givens[:, :, 1:] * tf.concat(
          (x[:, :, 1:], x[:, :, 0:1]), axis=-1)
  return tf.reshape(x_ref, (batch_size, -1))


def givens_rotations(r, x):
  """Applies 2x2 rotations.

  Args:
    r: Tensor of size B x d representing rotation parameters per example.
    x: Tensor of size B x d representing points to rotate.

  Returns:
    Tensor of size B x s representing rotation of x by r.
  """
  batch_size = tf.shape(r)[0]
  givens = tf.reshape(r, (batch_size, -1, 2))
  givens = givens / tf.norm(givens, ord=2, axis=-1, keepdims=True)
  x = tf.reshape(x, (batch_size, -1, 2))
  x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * tf.concat(
      (-x[:, :, 1:], x[:, :, 0:1]), axis=-1)
  return tf.reshape(x_rot, (batch_size, -1))

