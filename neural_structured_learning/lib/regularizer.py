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

"""Provides functional interface(s) to generate regularizer(s)."""
from neural_structured_learning.lib import distances
from neural_structured_learning.lib import utils

import tensorflow as tf


def adv_regularizer(adv_neighbors, target_scores, model_fn, loss_fn):
  """Calculates adversarial loss from generated adversarial samples.

  Args:
    adv_neighbors: dense `float32` tensor, with two possible shapes: (a)
      `[batch_size, feat_len]` for pointwise samples, or (b)
      `[batch_size, seq_len, feat_len]` for sequence samples.
    target_scores: target tensor used to compute loss.
    model_fn: a method that has input tensor (same shape as `adv_neighbors`),
      `is_train` and `reuse` as inputs, and returns predicted logits.
    loss_fn: a loss function that has `target` and `prediction` as inputs, and
      returns a `float32` scalar.

  Returns:
    adv_loss: a `float32` denoting the adversarial loss.
  """
  adv_predictions = model_fn(
      adv_neighbors, is_train=tf.constant(False), reuse=True)
  adv_loss = loss_fn(target_scores, adv_predictions)
  tf.compat.v1.summary.scalar('adv_loss', adv_loss)
  return adv_loss


def _virtual_adv_regularizer(input_layer, embedding_fn, virtual_adv_config,
                             embedding, seed_perturbation):
  """Function to calculate virtual adversarial loss without randomness."""
  neighbor_config = virtual_adv_config.adv_neighbor_config

  def normalize_with_mask(perturbation):
    perturbation = utils.apply_feature_mask(perturbation,
                                            neighbor_config.feature_mask)
    return utils.normalize(perturbation, neighbor_config.adv_grad_norm)

  def loss_fn(embedding, perturbed_embedding):
    return distances.pairwise_distance_wrapper(
        sources=embedding,
        targets=perturbed_embedding,
        distance_config=virtual_adv_config.distance_config)

  perturbation = normalize_with_mask(seed_perturbation)

  # Uses the power iteration method and the finite difference method to
  # approximate the direction which increases virtual adversarial loss the most.
  for _ in range(virtual_adv_config.num_approx_steps):
    with tf.GradientTape() as tape:
      scaled_perturbation = virtual_adv_config.approx_difference * perturbation
      tape.watch(scaled_perturbation)
      virtual_adv_embedding = embedding_fn(input_layer + scaled_perturbation)
      virtual_adv_loss = loss_fn(embedding, virtual_adv_embedding)
    grad = tape.gradient(virtual_adv_loss, scaled_perturbation)
    perturbation = tf.stop_gradient(normalize_with_mask(grad))

  final_perturbation = neighbor_config.adv_step_size * perturbation
  virtual_adv_embedding = embedding_fn(input_layer + final_perturbation)
  # The gradient shouldn't be populated through the original embedding because
  # our goal is to drag the embedding of the virtual adversarial example to be
  # as close as that of the original example, but not the other way around.
  original_embedding = tf.stop_gradient(embedding)
  return loss_fn(original_embedding, virtual_adv_embedding)


def virtual_adv_regularizer(input_layer,
                            embedding_fn,
                            virtual_adv_config,
                            embedding=None):
  """Calculates virtual adversarial loss for the given input.

  Virtual adversarial loss is defined as the distance between the embedding of
  the input and that of a slightly perturbed input. Optimizing this loss helps
  smooth models locally.

  Reference paper: https://arxiv.org/pdf/1704.03976.pdf

  Args:
    input_layer: a dense tensor for input features whose first dimension is the
      training batch size.
    embedding_fn: a unary function that computes the embedding for the given
      `input_layer` input.
    virtual_adv_config: an `nsl.configs.VirtualAdvConfig` object that specifies
      parameters for generating adversarial examples and computing the the
      adversarial loss.
    embedding: (optional) a dense tensor representing the embedding of
      `input_layer`. If not provided, it will be calculated as
      `embedding_fn(input_layer)`.

  Returns:
    virtual_adv_loss: a `float32` denoting the virtural adversarial loss.
  """

  if embedding is None:
    embedding = embedding_fn(input_layer)

  seed_perturbation = tf.random.normal(tf.shape(input=input_layer))
  return _virtual_adv_regularizer(input_layer, embedding_fn, virtual_adv_config,
                                  embedding, seed_perturbation)
