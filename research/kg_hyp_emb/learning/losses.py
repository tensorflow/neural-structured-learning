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
"""Loss functions for KG with support for optional negative sampling."""

import abc
import tensorflow as tf


class LossFn(abc.ABC):
  """Abstract loss function for KG embeddings."""

  def __init__(self, sizes, neg_sample_size):
    """Initialize KG loss function.

    Args:
      sizes: Tuple of size 3 containing (n_entities, n_rels, n_entities).
      neg_sample_size: Integer indicating the number of negative samples to use.
    """
    self.n_entities = sizes[0]
    self.n_predicates = sizes[1]
    self.neg_sample_size = neg_sample_size
    self.use_neg_sampling = neg_sample_size > 0
    self.gamma = tf.Variable(
        self.neg_sample_size * tf.keras.backend.ones(1) / self.n_entities,
        trainable=False)

  @abc.abstractmethod
  def loss_from_logits(self, logits, full_labels, labels):
    """Computes KG loss.

    Args:
      logits: Tensor of size batch_size x n_entities containing predictions.
      full_labels: Tensor of size batch_size x n_entities containing one-hot
        labels.
      labels: Tensor of size batch_size x 1 containing sparse labels (index of
        correct tail entity).

    Returns:
      Average loss within batch.
    """
    pass

  def get_neg_sample_mask(self, logits, full_labels):
    """Generates negative sampling mask.

    Args:
      logits: Tensor of size batch_size x n_entities containing predictions.
      full_labels: Tensor of size batch_size x n_entities containing one-hot
        labels.

    Returns:
      neg_sample_mask: Tensor of size batch_size x n_entities with ones and
                       zeros (one indicates that the corresonding example
                       is masked).
    """
    neg_sample_mask = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    neg_sample_mask = tf.cast(neg_sample_mask > self.gamma, logits.dtype)
    neg_sample_mask = -1e6 * tf.maximum(neg_sample_mask - full_labels, 0)
    return neg_sample_mask

  def calculate_loss(self, model, input_batch):
    """Computes loss with or without negative sampling.

    Args:
      model: tf.keras.Model KG embedding model.
      input_batch: Tensor of size batch_size x 3 containing input triples.

    Returns:
      Average loss within the input_batch.
    """
    labels = input_batch[:, 2]
    logits = model(input_batch, eval_mode=True)
    full_labels = tf.one_hot(labels, depth=self.n_entities, dtype=logits.dtype)
    if self.use_neg_sampling:
      # mask some values for negative sampling
      neg_sample_mask = self.get_neg_sample_mask(logits, full_labels)
      # mask logits to only keep target and negative examples' scores
      logits = logits + neg_sample_mask
    return self.loss_from_logits(logits, full_labels, labels)


class SigmoidCrossEntropy(LossFn):
  """Sigmoid cross entropy loss."""

  def loss_from_logits(self, logits, full_labels, labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(full_labels, logits))


class SoftmaxCrossEntropy(LossFn):
  """Softmax cross entropy loss."""

  def loss_from_logits(self, logits, full_labels, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
