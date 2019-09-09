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
"""Utilities for multimodal integration for Neural Structured Learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs

import tensorflow as tf


def _bimodal_op(x, y, op_config):
  """Applies a bimodal integration operation to the inputs `x` and `y`."""
  if op_config.integration_type == configs.IntegrationType.ADD:
    return x + y
  elif op_config.integration_type == configs.IntegrationType.MUL:
    return x * y
  elif op_config.integration_type == configs.IntegrationType.TUCKER_DECOMP:
    # Given U_i and V_i as weight matrices, the bimodal operation of
    # TUCKER_DECOMP (applied in MUTAN) in the hidden layer is
    # `sum_i ( (x U_i) * (y V_i))`. See Eq (10) in arXiv:1705.06676 for details.
    if len(op_config.hidden_dims) != 3:
      raise ValueError(
          'TUCKER_DECOMP must specify three hidden_dims in IntegrationConfig.')
    hidden_dims = op_config.hidden_dims
    expanded_dims = hidden_dims[1] * hidden_dims[2]
    hidden_x = tf.keras.layers.Dense(expanded_dims, activation=None)(x)
    hidden_y = tf.keras.layers.Dense(expanded_dims, activation=None)(y)
    hidden_layer = hidden_x * hidden_y
    hidden_layer_shape = hidden_layer.get_shape().as_list()
    hidden_layer_reshape = hidden_layer_shape[:-1] + hidden_dims[1:]
    hidden_layer = tf.reshape(hidden_layer, hidden_layer_reshape)
    return tf.reduce_sum(input_tensor=hidden_layer, axis=-1)
  else:
    raise ValueError('Invalid IntegrationType %s.' % op_config.integration_type)


def bimodal_integration(x,
                        y,
                        output_dims,
                        integration_config,
                        reuse=None,
                        scope=None):
  """Computes the bimodal integration between `x` and `y`.

    The inputs `x` and `y` are usually from two different types of input
    sources, e.g., `x` may represent image embeddings and `y` may represent text
    embeddings. This function will integrate bimodal inputs `x` and `y` as
    follows:

    ```
    outputs = fc_layer(activation_fn(integrate(fc_layer(x), fc_layer(y))))
    ```,
    where `fc_layer` represents a fully connected layer.

    When the integration type is (element-wise) 'additive', this function is
    equivalent to concatenating `x` and `y` and passing the result into a
    two-layer perceptron. When the integration type is (element-wise)
    'multiplicative', this function is equivalent to [multimodal low-rank
    bilinear Pooling (MLB)](https://arxiv.org/abs/1610.04325). When the
    integration type is 'tucker_decomp', this function is equivalent to
    [multimodal tensor-based Tucker decomposition
    (MUTAN)](https://arxiv.org/abs/1705.06676).

  Args:
    x: A tensor of rank at least 2 and a static value for the last dimension.
      For example, `[batch_size, depth]`, `[None, None, None, channels]`, etc.
    y: A tensor of the same type and shape as `x`, except that the size of the
      last dimension can be different.
    output_dims: Integer or long, the number of output units.
    integration_config: An instance of `nsl.configs.IntegrationConfig`.
    reuse: Whether or not the fully-connected layers and their variables should
      be reused. To be able to reuse them, `scope` must be specified.
    scope: Optional scope for `variable_scope`.

  Returns:
    The tensor variable representing the result of the series of operations.
  """
  scope_name = 'bimodal_' + integration_config.integration_type.value

  hidden_dims = integration_config.hidden_dims if isinstance(
      integration_config.hidden_dims,
      list) else [integration_config.hidden_dims]

  with tf.compat.v1.variable_scope(scope, scope_name, [x, y], reuse=reuse):
    hidden_x = tf.keras.layers.Dense(
        hidden_dims[0], activation=integration_config.activation_fn)(
            x)
    hidden_y = tf.keras.layers.Dense(
        hidden_dims[0], activation=integration_config.activation_fn)(
            y)
    hidden_layer = _bimodal_op(hidden_x, hidden_y, integration_config)
    return tf.keras.layers.Dense(output_dims, activation=None)(hidden_layer)
