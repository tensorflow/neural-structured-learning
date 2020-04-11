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
"""Hyperbolic utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MIN_NORM = 1e-15
MAX_TANH_ARG = 15.0
BALL_EPS = {tf.float32: 4e-3, tf.float64: 1e-5}

################## MATH FUNCTIONS #################


def artanh(x):
  eps = BALL_EPS[x.dtype]
  return tf.atanh(tf.minimum(tf.maximum(x, -1 + eps), 1 - eps))


def tanh(x):
  return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


################## HYP OPS ########################


def expmap0(u, c):
  """Hyperbolic exponential map at zero in the Poincare ball model.

  Args:
    u: Tensor of size B x dimension representing tangent vectors.
    c: Tensor of size 1 representing the absolute hyperbolic curvature.

  Returns:
    Tensor of shape B x dimension.
  """
  sqrt_c = tf.sqrt(c)
  u_norm = tf.maximum(tf.norm(u, axis=-1, keepdims=True), MIN_NORM)
  gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
  return project(gamma_1, c)


def logmap0(y, c):
  """Hyperbolic logarithmic map at zero in the Poincare ball model.

  Args:
    y: Tensor of size B x dimension representing hyperbolic points.
    c: Tensor of size 1 representing the absolute hyperbolic curvature.

  Returns:
    Tensor of shape B x dimension.
  """
  sqrt_c = tf.sqrt(c)
  y_norm = tf.maximum(tf.norm(y, axis=-1, keepdims=True), MIN_NORM)
  return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
  """Projects points to the Poincare ball.

  Args:
    x: Tensor of size B x dimension.
    c: Tensor of size 1 representing the absolute hyperbolic curvature.

  Returns:
    Tensor of shape B x dimension where each row is a point that lies within
    the Poincare ball.
  """
  eps = BALL_EPS[x.dtype]
  return tf.clip_by_norm(t=x, clip_norm=(1. - eps) / tf.sqrt(c), axes=[1])


def mobius_add(x, y, c):
  """Element-wise Mobius addition.

  Args:
    x: Tensor of size B x dimension representing hyperbolic points.
    y: Tensor of size B x dimension representing hyperbolic points.
    c: Tensor of size 1 representing the absolute hyperbolic curvature.

  Returns:
    Tensor of shape B x dimension representing the element-wise Mobius addition
    of x and y.
  """
  cx2 = c * tf.reduce_sum(x * x, axis=-1, keepdims=True)
  cy2 = c * tf.reduce_sum(y * y, axis=-1, keepdims=True)
  cxy = c * tf.reduce_sum(x * y, axis=-1, keepdims=True)
  num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
  denom = 1 + 2 * cxy + cx2 * cy2
  return project(num / tf.maximum(denom, MIN_NORM), c)


################## HYP DISTANCE ###################


def hyp_distance(x, y, c, eval_mode=False):
  """Hyperbolic distance on the Poincare ball.

  Args:
    x: Tensor of size B1 x d
    y: Tensor of size B2 x d
    c: Tensor of size 1 representing the absolute hyperbolic curvature.
    eval_mode: boolean indicating whether to compute all pairwise distances or
      not. If eval_mode=False, must have B1=B2.

  Returns:
    Tensor of size B1 x B2 if eval_mode=True, otherwise Tensor of size B1 x 1.
  """
  sqrt_c = tf.sqrt(c)
  x2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
  if eval_mode:
    y2 = tf.transpose(tf.reduce_sum(y * y, axis=-1, keepdims=True))
    xy = tf.matmul(x, y, transpose_b=True)
  else:
    y2 = tf.reduce_sum(y * y, axis=-1, keepdims=True)
    xy = tf.reduce_sum(x * y, axis=-1, keepdims=True)
  c1 = 1 - 2 * c * xy + c * y2
  c2 = 1 - c * x2
  num = tf.sqrt(tf.square(c1) * x2 + tf.square(c2) * y2 - (2 * c1 * c2) * xy)
  denom = 1 - 2 * c * xy + tf.square(c) * x2 * y2
  pairwise_norm = num / tf.maximum(denom, MIN_NORM)
  dist = artanh(sqrt_c * pairwise_norm)
  return 2 * dist / sqrt_c
