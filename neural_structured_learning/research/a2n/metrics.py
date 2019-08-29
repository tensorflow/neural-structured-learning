# Copyright 2019 Google LLC
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
"""Evaluation Metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mrr(scores, candidates, labels):
  """Compute Mean Reciprocal Rank of labels in scores.

  Args:
    scores (tf.Tensor): batchsize, max_candidates tensor of scores
    candidates (tf.Tensor): batchsize, max_candidates tensor of candidate ids
    labels (tf.Tensor): batchsize tensor of ground truth labels
  Returns:
    rr (tf.Tensor): batchsize tensor of Reciprocal Rank values
  """
  _, top_score_ids = tf.nn.top_k(scores, k=tf.shape(scores)[-1])
  batch_indices = tf.cumsum(
      tf.ones_like(candidates, dtype=tf.int32), axis=0, exclusive=True
  )
  indices = tf.concat([tf.expand_dims(batch_indices, axis=-1),
                       tf.expand_dims(top_score_ids, -1)], -1)
  sorted_candidates = tf.gather_nd(candidates, indices)
  # label_ids = tf.expand_dims(tf.argmax(labels, axis=1), 1)
  label_rank_indices = tf.where(
      tf.equal(sorted_candidates, labels)
  )
  # +1 because top rank should be 1 not 0
  ranks = label_rank_indices[:, 1] + 1
  rr = 1.0 / tf.cast(ranks, tf.float32)
  return rr  # , ranks, label_rank_indices, sorted_candidates, top_score_ids


def hits_at_k(scores, candidates, labels, k=10):
  """Compute hits@k.

  Args:
    scores (tf.Tensor): batchsize, max_candidates tensor of scores
    candidates (tf.Tensor): batchsize, max_candidates tensor of candidate ids
    labels (tf.Tensor): batchsize tensor of ground truth labels
    k: values of k to evaluate hits@k
  Returns:
    rr (tf.Tensor): batchsize tensor of Reciprocal Rank values
  """
  _, top_score_ids = tf.nn.top_k(scores, k=k)
  batch_indices = tf.cumsum(
      tf.ones_like(top_score_ids, dtype=tf.int32), axis=0, exclusive=True
  )
  indices = tf.concat([tf.expand_dims(batch_indices, axis=-1),
                       tf.expand_dims(top_score_ids, -1)], -1)
  sorted_candidates = tf.gather_nd(candidates, indices)
  # label_ids = tf.expand_dims(tf.argmax(labels, axis=1), 1)
  hits = tf.reduce_max(
      tf.cast(tf.equal(sorted_candidates, labels), tf.float32), 1
  )
  return hits  # , sorted_candidates, top_score_ids
