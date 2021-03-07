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
"""DynamicEmbedding related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls.kernels import gen_dynamic_embedding_ops
import tensorflow as tf


def dynamic_embedding_lookup(keys: tf.Tensor,
                             config: de_config_pb2.DynamicEmbeddingConfig,
                             var_name: typing.Text,
                             service_address: typing.Text = "",
                             skip_gradient_update: bool = False,
                             timeout_ms: int = -1) -> tf.Tensor:
  """Returns the embeddings of from given keys.

  Args:
    keys: A string `Tensor` of shape [batch_size] or [batch_size,
      max_sequence_length] where an empty string would be mapped to an all zero
      embedding.
    config: A DynamicEmbeddingConfig proto that configs the embedding.
    var_name: A unique name for the given embedding.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    skip_gradient_update: A boolean indicating if gradient update is needed.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A `Tensor` of shape with one of below:
    - [batch_size, config.embedding_dimension] if the input Tensor is 1D, or
    - [batch_size, max_sequence_length, config.embedding_dimension] if the
      input is 2D.
  Raises:
    ValueError: If name is not specified.
  """
  if not var_name:
    raise ValueError("Must specify a valid var_name.")

  # If skip_gradient_update is true, reate a dummy variable so that the
  # gradients can be passed in.
  if skip_gradient_update:
    grad_placeholder = tf.constant(0.0)
  else:
    grad_placeholder = tf.Variable(0.0)

  resource = gen_dynamic_embedding_ops.dynamic_embedding_manager_resource(
      config.SerializeToString(), var_name, service_address, timeout_ms)

  return gen_dynamic_embedding_ops.dynamic_embedding_lookup(
      keys, grad_placeholder, resource)


def dynamic_embedding_update(keys: tf.Tensor,
                             values: tf.Tensor,
                             config: de_config_pb2.DynamicEmbeddingConfig,
                             var_name: typing.Text,
                             service_address: typing.Text = "",
                             timeout_ms: int = -1):
  """Updates the embeddings of given keys with given values.

  Args:
    keys: A string `Tensor` of shape [batch] or [batch_size,
      max_sequence_length].
    values: A `Tensor` of shape [batch_size, embedding_dimension] or
      [batch_size, max_sequence_length, embedding_dimension].
    config: A DynamicEmbeddingConfig proto that configs the embedding.
    var_name: A unique name for the given embedding.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A `Tensor` of shape with one of below:
    - [batch_size, config.embedding_dimension] if the input Tensor is 1D, or
    - [batch_size, max_sequence_length, config.embedding_dimension] if the
      input is 2D.
  Raises:
    TypeError: If var_name is not specified.
  """
  if not var_name:
    raise TypeError("Must specify a valid var_name.")

  resource = gen_dynamic_embedding_ops.dynamic_embedding_manager_resource(
      config.SerializeToString(), var_name, service_address, timeout_ms)

  return gen_dynamic_embedding_ops.dynamic_embedding_update(
      keys, values, resource)
