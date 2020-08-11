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
"""Modeling for GNNs."""
from layers import GraphConvLayer
import tensorflow as tf


class GCNBlock(tf.keras.layers.Layer):
  """Graph convolutional block."""

  def __init__(self, hidden_dim, dropout_rate, bias, **kwargs):
    """Initializes a GGN block.

    Args:
      hidden_dim: (int) Dimension of hidden layer.
      dropout_rate: (float) Dropout probability
      bias: (bool) Whether bias needs to be added to gcn layers
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GCNBlock, self).__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.bias = bias

    self._activation = tf.keras.layers.ReLU()
    self._dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    super(GCNBlock, self).build(input_shape)
    self._graph_conv_layer = GraphConvLayer(self.hidden_dim, bias=self.bias)

  def call(self, inputs):
    x = self._graph_conv_layer(inputs)
    x = self._activation(x)
    return self._dropout(x)


class GCN(tf.keras.Model):
  """Graph convolution network for semi-supevised node classification."""

  def __init__(self, num_layers, hidden_dim, num_classes, dropout_rate, bias,
               **kwargs):
    """Initializes a GGN model.

    Args:
      num_layers: (int) Number of gnn layers
      hidden_dim: (list) List of hidden layers dimension
      num_classes: (int) Total number of classes
      dropout_rate: (float) Dropout probability
      bias: (bool) Whether bias needs to be added to gcn layers
      **kwargs: Keyword arguments for tf.keras.Model.
    """
    super(GCN, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate
    self.bias = bias
    # input layer
    self.gc = [
        GCNBlock(self.hidden_dim[0], dropout_rate=dropout_rate, bias=bias),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gc.append(
          GCNBlock(self.hidden_dim[i], dropout_rate=dropout_rate, bias=bias))

    # output layer
    self.classifier = GraphConvLayer(self.num_classes, bias=self.bias)

  def call(self, inputs):
    features, adj = inputs[0], inputs[1]
    for i in range(self.num_layers - 1):
      x = (features, adj)
      features = self.gc[i](x)

    x = (features, adj)
    outputs = self.classifier(x)
    return outputs
