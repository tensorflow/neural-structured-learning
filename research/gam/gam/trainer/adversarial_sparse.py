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
"""Functions used for Virtual Adversarial Training on sparse feature matrices."""

from .adversarial_dense import get_normalized_vector
from .adversarial_dense import get_normalizing_constant
import tensorflow as tf

epsilon = 5
num_power_iterations = 1
xi = 1e-6
scale_r = False


def kl_divergence_with_logit(q_logit, p_logit, mask):
  """Computes KL-divergence between to sets of logits for the masked samples."""
  q = tf.nn.softmax(q_logit)
  num_non_zero = tf.reduce_sum(mask)
  qlogq = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=q, logits=q_logit)
  qlogq = qlogq * mask / num_non_zero
  qlogp = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=q, logits=p_logit)
  qlogp = qlogp * mask / num_non_zero
  return qlogq - qlogp


def get_loss_vat(inputs, predictions, mask, is_train, model, placeholders,
                 predictions_var_scope):
  """Computes the virtual adversarial loss for the provided inputs.

  Args:
    inputs: A batch of input features, where the batch is the first dimension.
    predictions: The logits predicted by a model on the provided inputs.
    mask: A tensor of booleans specifying which samples to apply the virtual
      adversarial loss to.
    is_train: A boolean placeholder specifying if this is a training or testing
      setting.
    model: The model that generated the logits.
    placeholders: Placeholders for model encodings.
    predictions_var_scope: Variable scope for obtaining the predictions.

  Returns:
    A float value representing the virtual adversarial loss.
  """
  mask = tf.cast(mask, dtype=tf.float32)
  r_vadv = generate_virtual_adversarial_perturbation(
      inputs,
      predictions,
      model,
      placeholders,
      mask,
      predictions_var_scope,
      is_train=is_train)
  predictions = tf.stop_gradient(predictions)
  logit_p = predictions
  new_inputs = tf.sparse_add(inputs, r_vadv)
  with tf.variable_scope(
      predictions_var_scope, auxiliary_name_scope=False, reuse=True):
    encoding_m, _, _ = model.get_encoding_and_params(
        inputs=new_inputs,
        is_train=is_train,
        update_batch_stats=False,
        **placeholders)
    logit_m, _, _ = model.get_predictions_and_params(
        encoding=encoding_m, is_train=is_train, **placeholders)
  num_non_zero = tf.reduce_sum(mask)
  loss = kl_divergence_with_logit(logit_p, logit_m, mask)
  return tf.reduce_sum(loss) / num_non_zero


def generate_virtual_adversarial_perturbation(inputs,
                                              logits,
                                              model,
                                              placeholders,
                                              mask,
                                              predictions_var_scope,
                                              is_train=True):
  """Generates an adversarial perturbation for virtual adversarial training.

  Args:
    inputs: A batch of input features, where the batch is the first dimension.
    logits: The logits predicted by a model on the provided inputs.
    model: The model that generated the logits.
    placeholders: A dictionary mapping string names to Tensorflow placeholders
      that are passed to the models when generating the predictions.
    mask: A tensor of booleans specifying which samples to apply the virtual
      adversarial loss to.
    predictions_var_scope: Variable scope for obtaining the predictions.
    is_train: A boolean placeholder specifying if this is a training or testing
      setting.

  Returns:
    A Tensor of the same shape as the inputs containing the adversarial
    perturbation for these inputs.
  """
  # Generate random perturbations.
  d = tf.random_normal(shape=tf.shape(inputs))
  # Only apply perturbations on the masked samples.
  d = tf.multiply(d, mask[:, None])

  for _ in range(num_power_iterations):
    d = xi * get_normalized_vector(d)
    logit_p = logits
    new_inputs = tf.add(tf.sparse_tensor_to_dense(inputs), d)
    new_inputs = tf.contrib.layers.dense_to_sparse(new_inputs)
    with tf.variable_scope(
        predictions_var_scope, auxiliary_name_scope=False, reuse=True):
      encoding_m, _, _ = model.get_encoding_and_params(
          inputs=new_inputs,
          is_train=is_train,
          update_batch_stats=False,
          **placeholders)
      logit_m, _, _ = model.get_predictions_and_params(
          encoding=encoding_m, is_train=is_train, **placeholders)
    dist = kl_divergence_with_logit(logit_p, logit_m, mask)
    grad = tf.gradients(dist, [d], aggregation_method=2)[0]
    d = tf.stop_gradient(grad)

  r_vadv = get_normalized_vector(d)
  if scale_r:
    r_vadv *= get_normalizing_constant(inputs.values)
  r_vadv *= epsilon

  return tf.contrib.layers.dense_to_sparse(r_vadv)


def logsoftmax(x):
  """Softmax where the inputs are logits and the outputs remain logits."""
  xdev = x - tf.reduce_max(x, 1, keep_dims=True)
  lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
  return lsm


def entropy_y_x(logits, mask):
  """Entropy term to add to VAT with entropy minimization.

  Args:
    logits: A Tensor containing the predicted logits for a batch of samples.
    mask: A boolean Tensor specifying which samples to use in the calculation of
      the entropy.

  Returns:
    The entropy minimization loss.
  """
  mask = tf.cast(mask, dtype=tf.float32)
  p = tf.nn.softmax(logits)
  ent = tf.reduce_sum(p * logsoftmax(logits), 1)
  ent = tf.reduce_sum(tf.multiply(ent, mask)) / tf.reduce_sum(mask)
  return -ent
