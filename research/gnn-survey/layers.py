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
"""GNN layers."""
import tensorflow as tf


class GraphConvLayer(tf.keras.layers.Layer):
  """Single graph convolution layer."""

  def __init__(self, output_dim, bias, **kwargs):
    """Initializes the GraphConvLayer.

    Args:
      output_dim: (int) Output dimension of gcn layer
      bias: (bool) Whether bias needs to be added to the layer
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GraphConvLayer, self).__init__(**kwargs)
    self.output_dim = output_dim
    self.bias = bias

  def build(self, input_shape):
    super(GraphConvLayer, self).build(input_shape)
    self.weight = self.add_weight(
        name='weight',
        shape=(input_shape[0][-1], self.output_dim),
        initializer='random_normal',
        trainable=True)
    if self.bias:
      self.b = self.add_weight(
          name='bias',
          shape=(self.output_dim,),
          initializer='random_normal',
          trainable=True)

  def call(self, inputs):
    x, adj = inputs[0], inputs[1]
    x = tf.matmul(adj, x)
    outputs = tf.matmul(x, self.weight)
    if self.bias:
      return self.b + outputs
    else:
      return outputs
