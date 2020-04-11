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
"""Complex embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgemb.models.base import KGModel
import tensorflow as tf


class BaseC(KGModel):
  """Base model class for complex embeddings."""

  def __init__(self, sizes, args):
    """Initialize complex KG embedding model.

    Args:
      sizes: Tuple of size 3 containing (n_entities, n_rels, n_entities).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    assert args.rank % 2 == 0, ("Complex models must have an even embedding "
                                "dimension.")
    super(BaseC, self).__init__(sizes, args)
    self.half_rank = self.rank // 2

  def get_rhs(self, input_tensor):
    return self.entity(input_tensor[:, 2])

  def get_candidates(self,):
    return self.entity.embeddings

  def similarity_score(self, lhs, rhs, eval_mode):
    lhs = lhs[:, :self.half_rank], lhs[:, self.half_rank:]
    rhs = rhs[:, :self.half_rank], rhs[:, self.half_rank:]
    if eval_mode:
      return tf.matmul(lhs[0], tf.transpose(rhs[0])) + tf.matmul(
          lhs[1], tf.transpose(rhs[1]))
    else:
      return tf.reduce_sum(
          lhs[0] * rhs[0] + lhs[1] * rhs[1], axis=-1, keepdims=True)

  def get_factors(self, input_tensor):
    lhs = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    rhs = self.entity(input_tensor[:, 2])

    lhs = lhs[:, :self.half_rank], lhs[:, self.half_rank:]
    rel = rel[:, :self.half_rank], rel[:, self.half_rank:]
    rhs = rhs[:, :self.half_rank], rhs[:, self.half_rank:]

    lhs = tf.sqrt(lhs[0]**2 + lhs[1]**2)
    rel = tf.sqrt(rel[0]**2 + rel[1]**2)
    rhs = tf.sqrt(rhs[0]**2 + rhs[1]**2)

    return lhs, rel, rhs


class Complex(BaseC):
  """Complex embeddings for simple link prediction."""

  def get_queries(self, input_tensor):
    lhs = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    lhs = lhs[:, :self.half_rank], lhs[:, self.half_rank:]
    rel = rel[:, :self.half_rank], rel[:, self.half_rank:]

    return tf.concat(
        [lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]],
        axis=1)


class RotatE(BaseC):
  """Complex embeddings with Euclidean rotations."""

  def get_queries(self, input_tensor):
    lhs = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    lhs = lhs[:, :self.half_rank], lhs[:, self.half_rank:]
    rel = rel[:, :self.half_rank], rel[:, self.half_rank:]

    rel_norm = tf.sqrt(rel[0]**2 + rel[1]**2)
    cos = tf.math.divide_no_nan(rel[0], rel_norm)
    sin = tf.math.divide_no_nan(rel[1], rel_norm)

    return tf.concat([lhs[0] * cos - lhs[1] * sin, lhs[0] * sin + lhs[1] * cos],
                     axis=1)
