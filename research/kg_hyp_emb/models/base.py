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
"""Abstract class for Knowledge Graph embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from kgemb.learning import regularizers
import numpy as np
import tensorflow as tf


class KGModel(tf.keras.Model, abc.ABC):
  """Abstract Knowledge Graph embedding model class.

  Module to define basic operations in KG embedding models, including embedding
  initialization, computing embeddings and triples' scores.
  """

  def __init__(self, sizes, args):
    """Initialize KG embedding model.

    Args:
      sizes: Tuple of size 3 containing (n_entities, n_rels, n_entities).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    super(KGModel, self).__init__()
    self.sizes = sizes
    self.rank = args.rank
    self.bias = args.bias
    self.initializer = getattr(tf.keras.initializers, args.initializer)
    self.entity_regularizer = getattr(regularizers, args.regularizer)(
        args.entity_reg)
    self.rel_regularizer = getattr(regularizers, args.regularizer)(args.rel_reg)
    self.entity = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.entity_regularizer,
        name='entity_embeddings')
    self.rel = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.rel_regularizer,
        name='relation_embeddings')
    train_biases = self.bias == 'learn'
    self.bh = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=1,
        embeddings_initializer='zeros',
        name='head_biases',
        trainable=train_biases)
    self.bt = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=1,
        embeddings_initializer='zeros',
        name='tail_biases',
        trainable=train_biases)
    self.gamma = tf.Variable(
        initial_value=args.gamma * tf.keras.backend.ones(1), trainable=False)

  @abc.abstractmethod
  def get_queries(self, input_tensor):
    """Get query embeddings using head and relationship for an index tensor.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing triples' indices.

    Returns:
      Tensor of size batch_size x embedding_dimension representing queries'
      embeddings.
    """
    pass

  @abc.abstractmethod
  def get_rhs(self, input_tensor):
    """Get right hand side (tail) embeddings for an index tensor.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing triples' indices.

    Returns:
      Tensor of size batch_size x embedding_dimension representing tail
      entities' embeddings.
    """
    pass

  @abc.abstractmethod
  def get_candidates(self):
    """Get all candidate tail embeddings in a knowledge graph dataset.

    Returns:
      Tensor of size n_entities x embedding_dimension representing embeddings
      for all enitities in the KG.
    """
    pass

  @abc.abstractmethod
  def similarity_score(self, lhs, rhs, eval_mode):
    """Computes a similarity score between queries and tail embeddings.

    Args:
      lhs: Tensor of size B1 x embedding_dimension containing queries'
        embeddings.
      rhs: Tensor of size B2 x embedding_dimension containing tail entities'
        embeddings.
      eval_mode: boolean to indicate whether to compute all pairs of scores or
        not. If False, B1 must be equal to B2.

    Returns:
      Tensor representing similarity scores. If eval_mode is False, this tensor
      has size B1 x 1, otherwise it has size B1 x B2.
    """
    pass

  def call(self, input_tensor, eval_mode=False):
    """Forward pass of KG embedding models.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing triples' indices.
      eval_mode: boolean to indicate whether to compute scores against all
        possible tail entities in the KG, or only individual triples' scores.

    Returns:
      Tensor containing triple scores. If eval_mode is False, this tensor
      has size batch_size x 1, otherwise it has size batch_size x n_entities
      where n_entities is the total number of entities in the KG.
    """
    lhs = self.get_queries(input_tensor)
    lhs_biases = self.bh(input_tensor[:, 0])
    if eval_mode:
      rhs = self.get_candidates()
      rhs_biases = self.bt.embeddings
    else:
      rhs = self.get_rhs(input_tensor)
      rhs_biases = self.bt(input_tensor[:, 2])
    predictions = self.score(lhs, lhs_biases, rhs, rhs_biases, eval_mode)
    return predictions

  def score(self, lhs, lhs_biases, rhs, rhs_biases, eval_mode):
    """Compute triple scores using embeddings and biases."""
    score = self.similarity_score(lhs, rhs, eval_mode)
    if self.bias == 'constant':
      return score + self.gamma
    elif self.bias == 'learn':
      if eval_mode:
        return score + lhs_biases + tf.transpose(rhs_biases)
      else:
        return score + lhs_biases + rhs_biases
    else:
      return score

  def get_scores_targets(self, input_tensor):
    """Computes triples' scores as well as scores againts all possible entities.

    Args:
      input_tensor: Tensor of size batch_size x 3 containing triples' indices.

    Returns:
      scores: Numpy array of size batch_size x n_entities containing queries'
              scores against all possible entities in the KG.
      targets: Numpy array of size batch_size x 1 containing triples' scores.
    """
    cand = self.get_candidates()
    cand_biases = self.bt.embeddings
    lhs = self.get_queries(input_tensor)
    lhs_biases = self.bh(input_tensor[:, 0])
    rhs = self.get_rhs(input_tensor)
    rhs_biases = self.bt(input_tensor[:, 2])
    scores = self.score(lhs, lhs_biases, cand, cand_biases, eval_mode=True)
    targets = self.score(lhs, lhs_biases, rhs, rhs_biases, eval_mode=False)
    return scores.numpy(), targets.numpy()

  def eval(self, examples, filters, batch_size=1000):
    """Compute ranking-based evaluation metrics.

    Args:
      examples: Tensor of size n_examples x 3 containing triples' indices.
      filters: Dict representing entities to skip per query for evaluation in
        the filtered setting.
      batch_size: batch size to use to compute scores.

    Returns:
      Evaluation metrics (mean rank, mean reciprocical rank and hits).
    """
    mean_rank = {}
    mean_reciprocal_rank = {}
    hits_at = {}
    total_examples = tf.data.experimental.cardinality(examples).numpy()
    batch_size = min(batch_size, total_examples)
    for missing in ['rhs', 'lhs']:
      ranks = np.ones(total_examples)
      for counter, input_tensor in enumerate(examples.batch(batch_size)):
        if batch_size * counter >= total_examples:
          break
        # reverse triple for head prediction
        if missing == 'lhs':
          input_tensor = tf.concat([
              input_tensor[:, 2:], input_tensor[:, 1:2] + self.sizes[1] // 2,
              input_tensor[:, 0:1]
          ],
                                   axis=1)
        scores, targets = self.get_scores_targets(input_tensor)
        for i, query in enumerate(input_tensor):
          query = query.numpy()
          filter_out = filters[missing][(query[0], query[1])]
          filter_out += [query[2]]
          scores[i, filter_out] = -1e6
        ranks[counter * batch_size:(counter + 1) * batch_size] += np.sum(
            (scores >= targets), axis=1)

      # compute ranking metrics
      mean_rank[missing] = np.mean(ranks)
      mean_reciprocal_rank[missing] = np.mean(1. / ranks)
      hits_at[missing] = {}
      for k in (1, 3, 10):
        hits_at[missing][k] = np.mean(ranks <= k)
    return mean_rank, mean_reciprocal_rank, hits_at
