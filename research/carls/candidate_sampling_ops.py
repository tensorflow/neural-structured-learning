# Copyright 2021 Google LLC
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
"""Candidate sampling related ops."""

import typing

from research.carls import context
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls.kernels import gen_dynamic_embedding_ops as de_ops
from research.carls.kernels import gen_sampled_logits_ops
from research.carls.kernels import gen_topk_ops as gen_topk_op
import tensorflow as tf


def top_k(inputs: tf.Tensor,
          k: int,
          de_config: de_config_pb2.DynamicEmbeddingConfig,
          var_name: typing.Text,
          service_address: typing.Text = "",
          timeout_ms: int = -1):
  """Computes logits for the top k closest embeddings to the inputs.

  Args:
    inputs: A float `Tensor` of shape `[batch_size, dim]` representing the
      forward activations of the input network.
    k: An `int` denoting the number of returned keys.
    de_config: A DynamicEmbeddingConfig for configuring the dynamic embedding.
    var_name: A unique name for the operation.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    keys: A string `Tensor` of shape `[batch_size, k]` representing the top k
        keys relative to the input.
    logits: A float `Tensor` of shape `[batch_size, k]` representing the logits
        for the returned keys.

  Raises:
    ValueError: if k is not greater than zero.

  Note: The (keys, logits) pair returned here should not be used for training as
  they only represent biased sampling. Instead, use sampled_softmax_loss()
  for training.
  """
  if not var_name:
    raise ValueError("Must specify a valid var_name.")
  if k <= 0:
    raise ValueError("k must be greater than zero, got %d" % k)

  context.add_to_collection(var_name, de_config)
  resource = de_ops.dynamic_embedding_manager_resource(
      de_config.SerializeToString(), var_name, service_address, timeout_ms)
  return gen_topk_op.topk_lookup(inputs, k, resource)


def sampled_softmax_loss(positive_keys: tf.Tensor,
                         inputs: tf.Tensor,
                         num_samples: int,
                         de_config: de_config_pb2.DynamicEmbeddingConfig,
                         var_name: typing.Text,
                         service_address: typing.Text = "",
                         timeout_ms: int = -1):
  """Compute sampled Softmax loss from given input activations.

  Args:
    positive_keys: A string `Tensor` of shape `[batch_size, None]` representing
      input positive keys.
    inputs: A float `Tensor` of shape `[batch_size, dim]`, representing the
      forward activations of the input network.
    num_samples: An int denoting the returned positive and negative samples.
    de_config: A DynamicEmbeddingConfig for configuring the dynamic embedding.
    var_name: A unique name for the operation.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A float `Tensor` representing the sampled softmax loss.
  """
  logits, labels, _, mask, _ = compute_sampled_logits(positive_keys, inputs,
                                                      num_samples, de_config,
                                                      var_name, service_address,
                                                      timeout_ms)
  tiled_norm = tf.tile(
      tf.maximum(tf.reduce_sum(labels, -1, keepdims=True), 1),
      [1, labels.get_shape()[-1]])
  labels /= tiled_norm
  return tf.reduce_sum(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits)) / tf.reduce_sum(mask)


def sampled_sigmoid_loss(positive_keys: tf.Tensor,
                         inputs: tf.Tensor,
                         num_samples: int,
                         de_config: de_config_pb2.DynamicEmbeddingConfig,
                         var_name: typing.Text,
                         service_address: typing.Text = "",
                         timeout_ms: int = -1):
  """Compute sampled sigmoid loss from given input activations.

  Args:
    positive_keys: A string `Tensor` of shape `[batch_size, None]` representing
      input positive keys.
    inputs: A float `Tensor` of shape `[batch_size, dim]`, representing the
      forward activations of the input network.
    num_samples: An int denoting the returned positive and negative samples.
    de_config: A DynamicEmbeddingConfig for configuring the dynamic embedding.
    var_name: A unique name for the operation.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A float `Tensor` representing the sampled sigmoid loss.
  """
  logits, labels, _, mask, _ = compute_sampled_logits(positive_keys, inputs,
                                                      num_samples, de_config,
                                                      var_name, service_address,
                                                      timeout_ms)
  tiled_norm = tf.tile(
      tf.maximum(tf.reduce_sum(labels, -1, keepdims=True), 1),
      [1, labels.get_shape()[-1]])
  labels /= tiled_norm
  reduced_sum = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)) / tf.reduce_sum(mask)
  return reduced_sum / num_samples


def compute_sampled_logits(positive_keys,
                           inputs,
                           num_samples: int,
                           de_config: de_config_pb2.DynamicEmbeddingConfig,
                           var_name: typing.Text,
                           service_address: typing.Text = "",
                           timeout_ms: int = -1):
  """Computes sampled logits from given positive labels.

  Args:
    positive_keys: A string `Tensor` of shape `[batch_size, None]` representing
      input positive keys.
    inputs: A float `Tensor` of shape `[batch_size, dim]` representing the
      forward activations of the input network.
    num_samples: An int denoting the returned positive and negative samples.
    de_config: A DynamicEmbeddingConfig for configuring the dynamic embedding.
    var_name: A unique name for the operation.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    logits: A float `Tensor` of shape `[batch_size, num_samples]` representing
        the logits for sampled labels.
    labels: A float `Tensor` of shape `[batch_size, num_samples]` with values
        in {0, 1} indicating if the sample is positive or negative.
    keys: A string `Tensor` of shape `[batch_size, num_samples]` representing
        the keys for each sample.
    mask: A float `Tensor` of shape `[batch_size]` representing the 0/1 mask
        of each batch. For example, if all keys in positive_keys[i] are empty,
        mask[i] = 0; otherwise mask[i] = 1.
    weights: A float `Tensor` representing the embeddings of the sampled keys.

  Raises:
    ValueError: If var_name is not specified.
    TypeError: If de_config is an instance of DynamicEmbeddingConfig.
  """
  if not var_name:
    raise ValueError("Must specify a valid name, got %s" % var_name)
  if num_samples < 1:
    raise ValueError("Invalid num_samples: %d" % num_samples)

  context.add_to_collection(var_name, de_config)
  resource = de_ops.dynamic_embedding_manager_resource(
      de_config.SerializeToString(), var_name, service_address, timeout_ms)

  # Create a dummy variable so that the gradients can be passed in.
  grad_placeholder = tf.Variable(0.0)

  keys, labels, expected_counts, mask, weights = (
      gen_sampled_logits_ops.sampled_logits_lookup(positive_keys, inputs,
                                                   num_samples,
                                                   grad_placeholder, resource))

  # Compute sampled logits.
  # Shape of weights: [d1, d2, dn-1, num_samples, embed_dim]
  # Shape of inputs: [d1, d2, dn-1, embed_dim]
  # Shape of output logits: [d1, d2, dn-1, num_samples]

  # [d1, d2, dn-1, embed_dim] -> [d1, d2, dn-1, 1, embed_dim]
  tiled_inputs = tf.expand_dims(inputs, axis=-2)
  # [d1, d2, dn-1, embed_dim] -> [d1, d2, dn-1, num_samples, embed_dim]
  multiples = [1] * (inputs.ndim + 1)
  multiples[-2] = num_samples
  tiled_inputs = tf.tile(tiled_inputs, multiples)
  # [d1, d2, dn-1, num_samples, embed_dim] -> [d1, d2, dn-1, num_samples]
  logits = tf.reduce_sum(weights * tiled_inputs, -1)
  # Sampled logits.
  logits -= tf.math.log(expected_counts)

  return logits, labels, keys, mask, weights


@tf.RegisterGradient("SampledLogitsLookup")
def _sampled_logits_lookup_grad(op, keys_grad, labels_grad,
                                expected_counts_grad, mask_grad, weights_grad):
  """Computes the gradients for SampledLogitsLookup.

  We uses the gradients w.r.t. the weights output of sampled_logits_lookup() to
  update the embeddings/weights of the sampled keys.
  The gradients for the inputs of sampled_logits_lookup should be provided, but
  none of them needs to be back-propagated. So we set all of them to be zeros.

  Args:
    op: The DynamicEmbeddingLookup op.
    keys_grad: The tensor representing the gradient w.r.t. the keys output.
    labels_grad: The tensor representing the gradient w.r.t. the labels output.
    expected_counts_grad: The tensor representing the gradient w.r.t. the
      expected_counts output.
    mask_grad: The tensor representing the gradient w.r.t. the mask output.
    weights_grad: The tensor representing the gradient w.r.t. the weights
      output.

  Returns:
    The gradients w.r.t. the input.
  """
  del keys_grad, labels_grad, expected_counts_grad, mask_grad  # Unused.

  pos_keys_grad, num_samples_grad, dummy_variable_grad, resource_grad = (
      gen_sampled_logits_ops.sampled_logits_lookup_grad(
          keys=op.outputs[0],
          weight_gradients=weights_grad,
          handle=op.inputs[4]))
  # Gradient for the input activation.
  inputs_grad = tf.zeros_like(op.inputs[1])
  return (pos_keys_grad, inputs_grad, num_samples_grad, dummy_variable_grad,
          resource_grad)
