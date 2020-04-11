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
"""Hyperbolic embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgemb.models.base import KGModel
from kgemb.utils import euclidean as euc_utils
from kgemb.utils import hyperbolic as hyp_utils
import numpy as np
import tensorflow as tf


class BaseH(KGModel):
  """Base model class for hyperbolic embeddings."""

  def __init__(self, sizes, args):
    """Initialize Hyperbolic KG embedding model.

    Args:
      sizes: Tuple of size 3 containing (n_entities, n_rels, n_entities).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    super(BaseH, self).__init__(sizes, args)
    self.c = tf.Variable(
        initial_value=tf.keras.backend.ones(1), trainable=args.train_c)

  def get_rhs(self, input_tensor):
    c = tf.math.softplus(self.c)
    return hyp_utils.expmap0(self.entity(input_tensor[:, 2]), c)

  def get_candidates(self,):
    c = tf.math.softplus(self.c)
    return hyp_utils.expmap0(self.entity.embeddings, c)

  def similarity_score(self, lhs, rhs, eval_mode):
    c = tf.math.softplus(self.c)
    return -hyp_utils.hyp_distance(lhs, rhs, c, eval_mode)**2


class TransH(BaseH):
  """Hyperbolic translation with parameters defined in tangent space."""

  def get_queries(self, input_tensor):
    c = tf.math.softplus(self.c)
    lhs = hyp_utils.expmap0(self.entity(input_tensor[:, 0]), c)
    rel = hyp_utils.expmap0(self.rel(input_tensor[:, 1]), c)
    res = hyp_utils.mobius_add(lhs, rel, c)
    return res


class RotH(BaseH):
  """Hyperbolic rotation model using 2 x 2 givens rotations."""

  def __init__(self, sizes, args):
    super(RotH, self).__init__(sizes, args)
    self.rot = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='rotation_weights')
    self.rot_trans = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='translation_weights')

  def get_queries(self, input_tensor):
    c = tf.math.softplus(self.c)
    head = hyp_utils.expmap0(self.entity(input_tensor[:, 0]), c)
    trans = hyp_utils.expmap0(self.rot_trans(input_tensor[:, 1]), c)
    lhs = hyp_utils.mobius_add(head, trans, c)
    rot = euc_utils.givens_rotations(self.rot(input_tensor[:, 1]), lhs)
    rot = hyp_utils.project(rot, c)
    rel = hyp_utils.expmap0(self.rel(input_tensor[:, 1]), c)
    return hyp_utils.mobius_add(rot, rel, c)


class RefH(BaseH):
  """Hyperbolic reflection model with 2 x 2 reflections."""

  def __init__(self, sizes, args):
    super(RefH, self).__init__(sizes, args)
    self.ref = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='reflection_weights')

  def get_queries(self, input_tensor):
    c = tf.math.softplus(self.c)
    head = hyp_utils.expmap0(self.entity(input_tensor[:, 0]), c)
    ref = euc_utils.givens_reflection(self.ref(input_tensor[:, 1]), head)
    lhs = hyp_utils.project(ref, c)
    rel = hyp_utils.expmap0(self.rel(input_tensor[:, 1]), c)
    return hyp_utils.mobius_add(lhs, rel, c)


class AttH(BaseH):
  """Hyperbolic attention model that combines reflections and rotations."""

  def __init__(self, sizes, args):
    super(AttH, self).__init__(sizes, args)
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
    c = tf.math.softplus(self.c)
    entity = self.entity(input_tensor[:, 0])

    # candidates
    rot = self.rot(input_tensor[:, 1])
    ref = self.ref(input_tensor[:, 1])
    ref_q = self.get_reflection_queries(entity, ref)
    rot_q = self.get_rotation_queries(entity, rot)
    cands = tf.concat([ref_q, rot_q], axis=1)

    # self-attention mechanism
    context_vec = self.context_vec(input_tensor[:, 1])
    context_vec = tf.reshape(context_vec, (-1, 1, self.rank))
    att_weights = tf.reduce_sum(
        context_vec * cands * self.scale, axis=-1, keepdims=True)
    att_weights = tf.nn.softmax(att_weights, axis=-1)
    att_q = tf.reduce_sum(att_weights * cands, axis=1)
    lhs = hyp_utils.expmap0(att_q, c)

    # hyperbolic translation
    rel = hyp_utils.expmap0(self.rel(input_tensor[:, 1]), c)
    return hyp_utils.mobius_add(lhs, rel, c)
