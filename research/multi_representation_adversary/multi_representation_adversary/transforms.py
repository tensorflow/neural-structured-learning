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

"""Transformations to apply to inputs before generating adversaries."""
import math

import gin
import numpy as np
import tensorflow.compat.v2 as tf


@gin.configurable
def dct(matrix):
  """Type-II discrete cosine transform (DCT).

  Args:
    matrix: A tensor of shape [batch, height, width, channel].

  Returns:
    A tensor of the same shape as input where DCT is applied on the height and
    width dimensions.
  """
  # matrix: [B, H, W, C]  (batch, height, width, channel)
  input_dim = matrix.shape[1]
  dct_mat = np.zeros((input_dim, input_dim), dtype=np.float32)
  # Constructs the DCT transform matrix.
  for i in range(input_dim):
    for j in range(input_dim):
      dct_mat[i, j] = (2.0 / math.sqrt(2.0 * input_dim)) * math.cos(
          i * (2.0 * j + 1.0) * math.pi * 1.0 / (2.0 * input_dim))
  dct_mat[0, :] = dct_mat[0, :] * 1.0 / math.sqrt(2)

  # matrix_t: [B, C, W, H]
  matrix_t = tf.transpose(matrix, perm=[0, 3, 2, 1])
  # Computes the transform. Multiplies dct_mat once for the rows, and once for
  # the columns.
  transformed_1d = tf.matmul(matrix_t, dct_mat.T)
  transformed_2d = tf.matmul(dct_mat, transformed_1d)

  # result: [B, H, W, C]
  return tf.transpose(transformed_2d, perm=[0, 3, 2, 1])


@gin.configurable
def idct(matrix):
  """Type-II inverse discrete cosine transform (IDCT).

  Args:
    matrix: A tensor of shape [batch, height, width, channel].

  Returns:
    A tensor of the same shape as input where IDCT is applied on the height
    and width dimensions.
  """
  # matrix: [B, H, W, C]  (batch, height, width, channel)
  input_dim = matrix.shape[1]
  idct_mat = np.zeros((input_dim, input_dim), dtype=np.float32)
  # Constructs the inverse-DCT transform matrix (same as Type-III DCT matrix).
  for i in range(input_dim):
    for j in range(1, input_dim):
      idct_mat[i, j] = math.sqrt(2.0 / input_dim) * math.cos(
          j * (2.0 * i + 1.0) * math.pi * 1.0 / (2.0 * input_dim))
  idct_mat[:, 0] = 1.0 / math.sqrt(input_dim)

  # matrix_t: [B, C, W, H]
  matrix_t = tf.transpose(matrix, perm=[0, 3, 2, 1])
  # Computes the transform. Multiplies idct_mat once for the rows, and once for
  # the columns.
  transformed_1d = tf.matmul(matrix_t, idct_mat.T)
  transformed_2d = tf.matmul(idct_mat, transformed_1d)

  # result: [B, H, W, C]
  return tf.transpose(transformed_2d, perm=[0, 3, 2, 1])


@gin.configurable
def identity(matrix):
  return matrix


@gin.configurable
def inverse_identity(matrix):
  return matrix


TRANSFORM_FUNCTIONS = {
    "identity": identity,
    "dct": dct,
}


INVERSE_TRANSFORM_FUNCTIONS = {
    "identity": inverse_identity,
    "dct": idct,
}
