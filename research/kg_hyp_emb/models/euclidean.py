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
"""Euclidean embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kg_hyp_emb.models.base import KGModel
from kg_hyp_emb.utils import euclidean as euc_utils
import numpy as np
import tensorflow as tf


class BaseE(KGModel):
  """Base model class for Euclidean embeddings."""

  def get_rhs(self, input_tensor):
    rhs = self.entity(input_tensor[:, 2])
    return rhs

  def get_candidates(self,):
    cands = self.entity.embeddings
    return cands

  def similarity_score(self, lhs, rhs, eval_mode):
    if self.sim == 'dot':
      if eval_mode:
        score = tf.matmul(lhs, tf.transpose(rhs))
      else:
        score = tf.reduce_sum(lhs * rhs, axis=-1, keepdims=True)
    elif self.sim == 'dist':
      score = -euc_utils.euc_sq_distance(lhs, rhs, eval_mode)
    else:
      raise AttributeError('Similarity function {} not recognized'.format(
          self.sim))
    return score


class CTDecomp(BaseE):
  """Canonical tensor decomposition."""

  def __init__(self, sizes, args):
    super(CTDecomp, self).__init__(sizes, args)
    self.sim = 'dot'

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    return tf.multiply(entity, rel)


class TransE(BaseE):
  """Euclidean translations."""

  def __init__(self, sizes, args):
    super(TransE, self).__init__(sizes, args)
    self.sim = 'dist'

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    return entity + rel


class RotE(BaseE):
  """2x2 Givens rotations."""

  def __init__(self, sizes, args):
    super(RotE, self).__init__(sizes, args)
    self.rel_diag = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='rotation_weights')
    self.sim = 'dist'

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    rel_diag = self.rel_diag(input_tensor[:, 1])
    return euc_utils.givens_rotations(rel_diag, entity) + rel


class RefE(BaseE):
  """2x2 Givens reflections."""

  def __init__(self, sizes, args):
    super(RefE, self).__init__(sizes, args)
    self.rel_diag = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='reflection_weights')
    self.sim = 'dist'

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    rel_diag = self.rel_diag(input_tensor[:, 1])
    return euc_utils.givens_reflection(rel_diag, entity) + rel


class MurE(BaseE):
  """Diagonal scaling."""

  def __init__(self, sizes, args):
    super(MurE, self).__init__(sizes, args)
    self.rel_diag = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='scaling_weights')
    self.sim = 'dist'

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    rel_diag = self.rel_diag(input_tensor[:, 1])
    return rel_diag * entity + rel


class AttE(BaseE):
  """Euclidean attention model that combines reflections and rotations."""

  def __init__(self, sizes, args):
    super(AttE, self).__init__(sizes, args)
    self.sim = 'dist'

    # reflection
    self.ref = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='reflection_weights')

    # rotation
    self.rot = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='rotation_weights')

    # attention
    self.context_vec = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='context_embeddings')
    self.scale = tf.keras.backend.ones(1) / np.sqrt(self.rank)

  def get_reflection_queries(self, entity, ref):
    queries = euc_utils.givens_reflection(ref, entity)
    return tf.reshape(queries, (-1, 1, self.rank))

  def get_rotation_queries(self, entity, rot):
    queries = euc_utils.givens_rotations(rot, entity)
    return tf.reshape(queries, (-1, 1, self.rank))

  def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    rot = self.rot(input_tensor[:, 1])
    ref = self.ref(input_tensor[:, 1])
    context_vec = self.context_vec(input_tensor[:, 1])
    ref_q = self.get_reflection_queries(entity, ref)
    rot_q = self.get_rotation_queries(entity, rot)

    # self-attention mechanism
    cands = tf.concat([ref_q, rot_q], axis=1)
    context_vec = tf.reshape(context_vec, (-1, 1, self.rank))
    att_weights = tf.reduce_sum(
        context_vec * cands * self.scale, axis=-1, keepdims=True)
    att_weights = tf.nn.softmax(att_weights, axis=-1)
    res = tf.reduce_sum(att_weights * cands, axis=1) + rel
    return res
