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
"""Operations for dynamic memory and its gradient.

Dynamic memory assumes each input to a hidden layer of a deeping neural network
belongs to a growing number of discrete patterns. This library provides the
basic tools for learning these patterns during model training.
"""

import typing

from research.carls import context
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls.kernels import gen_carls_ops
import tensorflow as tf

# Lookup modes, this is in one-to-one correspondence to
# MemoryLookupRequest::LookupMode.
LOOKUP_WITHOUT_UPDATE = 1
LOOKUP_WITH_UPDATE = 2
LOOKUP_WITH_GROW = 3


def dynamic_gaussian_memory_lookup(inputs: tf.Tensor,
                                   mode: typing.Union[int, tf.Tensor],
                                   config: de_config_pb2.DynamicEmbeddingConfig,
                                   var_name: typing.Text,
                                   service_address: typing.Text = "",
                                   timeout_ms: int = -1):
  """Applies dynamic Gaussian memory to given inputs.

  A Gaussian memory assumes the input pattern can be represented by a number of
  Gaussian clusters. This function returns the closest Gaussian mean, variance
  and the distance between each data and the closest Guassian center.

  This function can be used in conjunction with a DynamicNormalization layer in
  a DNN. The distance between the input and the Gaussian cluster can be used for
  model uncertainty inferece.

  Note that the memory data is only based on the last dimension of the input.
  Hence if the input shape is [d1, d2, ..., dn], it is assumed to contain
  d1*d2*...*dn-1 data points.

  Args:
    inputs: A float `Tensor` of shape [d1, d2, ..., dn] with n > 0.
    mode: An int or a `Tensor` whose value must be one of
      {LOOKUP_WITHOUT_UPDATE, LOOKUP_WITH_UPDATE, LOOKUP_WITH_GROW}.
    config: An instance of DynamicEmbeddingConfig.
    var_name: A unique name for the given op.
    service_address: The address of a knowledge bank service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    - A `Tensor` with the same shape of input representing the mean values.
    - A `Tensor` with the same shape of input representing the variance values.
    - A `Tensor` with the shape [d1, d2, ..., dn-1] representing the distance to
      the cluster center.
    - An int `Tensor` with the shape [d1, d2, ..., dn-1] representing the
    cluster ids.
  Raises:
    TypeError: if dm_config is not an instance of DynamicMemoryConfig.
    ValueError: If layer_name is not specified or mode is not valid.
  """
  if isinstance(mode, int) and mode not in {
      LOOKUP_WITHOUT_UPDATE, LOOKUP_WITH_UPDATE, LOOKUP_WITH_GROW
  }:
    raise ValueError("Invalid mode: %r" % mode)
  else:  # mode is a Tensor
    mode = tf.cast(mode, tf.int32)
  if not var_name:
    raise ValueError("Must specify a valid layer_name.")

  context.add_to_collection(var_name, config)
  resource = gen_carls_ops.dynamic_embedding_manager_resource(
      config.SerializeToString(), var_name, service_address, timeout_ms)

  return gen_carls_ops.dynamic_gaussian_memory_lookup(inputs, mode, resource)


@tf.RegisterGradient("DynamicGaussianMemoryLookup")
def _dynamic_gaussian_memory_lookup_grad(op, mean_grad, variance_grad,
                                         distance_grad, cluster_grad):
  """The gradient for DynamicGaussianMemoryLookup.

  The (mean, variance, distance, cluster_ids) are updated inside the dynamic
  memory based on the input, so we just ignore them. There is also no need to
  back-propagate the gradients for the input since there is no close-form
  formula for, e.g., mean(inputs). Instead, the gradients of the input are
  updated through other parts of a computation graph, e.g., from the loss of
  (input - mean)^2. This is consistent with the batch-normalization as its
  special case.

  Args:
    op: The dynamic_gaussian_memory_lookup op.
    mean_grad: A tensor representing the gradient w.r.t. the first output.
    variance_grad: A tensor representing the gradient w.r.t. the second output.
    distance_grad: A tensor representing the gradient w.r.t. the third output.
    cluster_grad: A tensor representing the gradient w.r.t. the fourth output.

  Returns:
    The gradients w.r.t. the input.
  """
  del mean_grad
  del variance_grad
  del distance_grad
  del cluster_grad
  # Grads for `inputs` and `mode`.
  grads = [tf.zeros_like(op.inputs[i]) for i in range(len(op.inputs) - 1)]
  # grad for `resource` input. tf.zeros_like only accept Tensor-like input.
  grads.append(0)
  return grads
