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
"""Libs/utils for multimodal integration for Neural Structured Learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs

import tensorflow as tf


def _bimodal_op(x, y, op_config):
  """Apply bimodal integration operation to inputs."""
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
  """Compute the bimodal integration between x and y.

    The inputs `x` and `y` are usually from two different types of sources,
    e.g., `x` represents image embeddings and `y` represent text embeddings.
    This function will integrate bimodal inputs `x` and `y` by the following:

    `outputs = fc_layer(
        activation_fn(integration_type(fc_layer(x), fc_layer(y))))`

    When the integration_type is (elementwise) 'additive', this function will is
    equivalent to concat `x` and `y` and pass them into a two-layer perception.
    When the integration_type is (elementwise) 'multiplicative', this function
    is equivalent to multimodal low-rank bilinear Pooling (MLB) in
    arXiv:1610.04325.
    When the integration_type is 'tucker_decomp', this function is equivalent to
    multimodal tensor-based Tucker decomposition (MUTAN) in arXiv:1705.06676.

  Args:
    x: A tensor of at least rank 2 and static value for the last dimension; i.e.
      [batch_size, depth], [None, None, None, channels].
    y: A tensor of the same type and shape as `x`, except the size of the last
      dimension can be different.
    output_dims: Integer or long, the number of output units.
    integration_config: IntegrationConfig contains the following configs (or
      hyper-parameters) for computing the hidden integration of `x` and `y`:
      (a) integration_type: Type of integration function to apply.
      (b) hidden_dims: Integer or a list of Integer, the number of hidden units
        in the fully-connected layer(s) before the output layer.
      (c) activation_fn: Activation function to be applied to.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
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
