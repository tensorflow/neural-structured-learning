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
from layers import GraphConvLayer, GraphAttnLayer
import tensorflow as tf


class GCNBlock(tf.keras.layers.Layer):
  """Graph convolutional block."""

  def __init__(self, hidden_dim, dropout_rate, sparse, bias, **kwargs):
    """Initializes a GGN block.

    Args:
      hidden_dim: (int) Dimension of hidden layer.
      dropout_rate: (float) Dropout probability
      sparse: (bool) Whether features are sparse
      bias: (bool) Whether bias needs to be added to gcn layers
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GCNBlock, self).__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.sparse = sparse
    self.bias = bias

    self._activation = tf.keras.layers.ReLU()
    self._dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    super(GCNBlock, self).build(input_shape)
    self._graph_conv_layer = GraphConvLayer(self.hidden_dim, sparse=self.sparse,
                                            bias=self.bias)

  def call(self, inputs):
    x = self._graph_conv_layer(inputs)
    x = self._activation(x)
    return self._dropout(x)


class GCN(tf.keras.Model):
  """Graph convolution network for semi-supevised node classification."""

  def __init__(self, num_layers, hidden_dim, num_classes, dropout_rate,
               sparse, bias, **kwargs):
    """Initializes a GGN model.

    Args:
      num_layers: (int) Number of gnn layers
      hidden_dim: (list) List of hidden layers dimension
      num_classes: (int) Total number of classes
      dropout_rate: (float) Dropout probability
      sparse: (bool) Whether features are sparse
      bias: (bool) Whether bias needs to be added to gcn layers
      **kwargs: Keyword arguments for tf.keras.Model.
    """
    super(GCN, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate
    self.sparse = sparse
    self.bias = bias
    # input layer
    self.gc = [
        GCNBlock(self.hidden_dim[0], dropout_rate=dropout_rate,
                 sparse=sparse, bias=bias),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gc.append(
          GCNBlock(self.hidden_dim[i], dropout_rate=dropout_rate,
                   sparse=sparse, bias=bias))

    # output layer
    self.classifier = GraphConvLayer(self.num_classes, sparse=self.sparse, bias=self.bias)

  def call(self, inputs):
    features, adj = inputs[0], inputs[1]
    for i in range(self.num_layers - 1):
      x = (features, adj)
      features = self.gc[i](x)

    x = (features, adj)
    outputs = self.classifier(x)
    return outputs

class GATBlock(tf.keras.layers.Layer):
  """Graph attention block."""

  def __init__(self, hidden_dim, dropout_rate, num_heads, **kwargs):
    """Initializes a GGN block.

    Args:
      hidden_dim: (int) Dimension of hidden layer.
      dropout_rate: (float) Dropout probability
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GATBlock, self).__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.num_heads = num_heads

    self._activation = tf.keras.layers.ELU()
    self._dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    super(GATBlock, self).build(input_shape)
    self._graph_attn_layer = [
      GraphAttnLayer(self.hidden_dim, self.dropout_rate) for _ in range(self.num_heads)]

  def call(self, inputs):
    x = tf.concat([attn(inputs) for attn in self._graph_attn_layer], axis=1)
    x = self._activation(x)
    return self._dropout(x)


class GAT(tf.keras.Model):
  """Graph convolution network for semi-supevised node classification."""

  def __init__(self, num_layers, hidden_dim, num_classes, dropout_rate,
               num_heads, **kwargs):
    """Initializes a GGN model.

    Args:
      num_layers: (int) Number of gnn layers
      hidden_dim: (list) List of hidden layers dimension
      num_classes: (int) Total number of classes
      dropout_rate: (float) Dropout probability
      num_heads: (int) Number of multi-head attentions
      **kwargs: Keyword arguments for tf.keras.Model.
    """
    super(GAT, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate
    self.num_heads = num_heads
    # input layer
    self.gat = [
        GATBlock(self.hidden_dim[0], dropout_rate=dropout_rate, num_heads=num_heads),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gat.append(
          GATBlock(self.hidden_dim[i], dropout_rate=dropout_rate))

    # output layer
    self.classifier = GraphAttnLayer(self.num_classes, dropout_rate=dropout_rate)

  def call(self, inputs):
    features, adj = inputs[0], inputs[1]
    for i in range(self.num_layers - 1):
      x = (features, adj)
      features = self.gat[i](x)

    x = (features, adj)
    outputs = self.classifier(x)
    return outputs

