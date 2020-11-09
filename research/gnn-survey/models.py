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
from layers import GraphAttnLayer
from layers import SparseGraphAttnLayer
from layers import GraphConvLayer
from layers import GraphIsomorphismLayer
import tensorflow as tf


class GCNBlock(tf.keras.layers.Layer):
  """Graph convolutional block."""

  def __init__(self, hidden_dim, dropout_rate, sparse, bias, **kwargs):
    """Initializes a GCN block.

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

    self._graph_conv_layer = GraphConvLayer(
        self.hidden_dim, sparse=self.sparse, bias=self.bias)

  def call(self, inputs):
    x = self._graph_conv_layer(inputs)
    x = self._activation(x)
    return self._dropout(x)


class GCN(tf.keras.Model):
  """Graph convolution network for semi-supevised node classification."""

  def __init__(self, num_layers, hidden_dim, num_classes, dropout_rate, sparse,
               bias, **kwargs):
    """Initializes a GCN model.

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
        GCNBlock(
            self.hidden_dim[0],
            dropout_rate=dropout_rate,
            sparse=sparse,
            bias=bias),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gc.append(
          GCNBlock(
              self.hidden_dim[i],
              dropout_rate=dropout_rate,
              sparse=sparse,
              bias=bias))

    # output layer
    self.classifier = GraphConvLayer(
        self.num_classes, sparse=self.sparse, bias=self.bias)

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

  def __init__(self, hidden_dim, dropout_rate, num_heads, sparse, **kwargs):
    """Initializes a GAT block.

    Args:
      hidden_dim: (int) Dimension of hidden layer.
      dropout_rate: (float) Dropout probability
      num_heads: (int) Number of attention heads.
      sparse: (bool) Whether features are sparse
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GATBlock, self).__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.num_heads = num_heads

    self._activation = tf.keras.layers.ELU()
    self._dropout = tf.keras.layers.Dropout(self.dropout_rate)

    if sparse:
      self._graph_attn_layer = [
        SparseGraphAttnLayer(self.hidden_dim, self.dropout_rate) for _ in range(self.num_heads)]
    else:
      self._graph_attn_layer = [
        GraphAttnLayer(self.hidden_dim, self.dropout_rate) for _ in range(self.num_heads)]

  def call(self, inputs):
    x = tf.concat([attn(inputs) for attn in self._graph_attn_layer], axis=1)
    x = self._activation(x)
    return self._dropout(x)


class GAT(tf.keras.Model):
  """Graph attention network for semi-supevised node classification."""

  def __init__(self, num_layers, hidden_dim, num_classes, dropout_rate,
               num_heads, sparse, **kwargs):
    """Initializes a GAT model.

    Args:
      num_layers: (int) Number of gnn layers
      hidden_dim: (list) List of hidden layers dimension
      num_classes: (int) Total number of classes
      dropout_rate: (float) Dropout probability
      num_heads: (int) Number of multi-head attentions
      sparse: (bool) Whether features are sparse
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
        GATBlock(
            self.hidden_dim[0], dropout_rate=dropout_rate, num_heads=num_heads, sparse=sparse),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gat.append(
          GATBlock(
              self.hidden_dim[i],
              dropout_rate=dropout_rate,
              num_heads=self.num_heads,
              sparse=sparse))

    # output layer
    if sparse:
      self.classifier = SparseGraphAttnLayer(
          self.num_classes,
          dropout_rate=dropout_rate)
    else:
      self.classifier = GraphAttnLayer(
          self.num_classes,
          dropout_rate=dropout_rate)

  def call(self, inputs):
    features, adj = inputs[0], inputs[1]
    for i in range(self.num_layers - 1):
      x = (features, adj)
      features = self.gat[i](x)

    x = (features, adj)
    outputs = self.classifier(x)
    return outputs


class GINBlock(tf.keras.layers.Layer):
  """Graph isomorphism block."""

  def __init__(self, mlp_layers, hidden_dim, dropout_rate, learn_eps, sparse, **kwargs):
    """Initializes a GIN block.

    Args:
      mlp layers: (int) Number of mlp layers in GIN
      hidden_dim: (int) Dimension of hidden layer.
      dropout_rate: (float) Dropout probability
      learn_eps: (bool) Whether to learn the epsilon weighting
      sparse: (bool) Whether features are sparse
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GINBlock, self).__init__(**kwargs)
    self.mlp_layers = mlp_layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.learn_eps = learn_eps
    self.sparse = sparse

    self._activation = tf.keras.layers.ReLU()
    self._dropout = tf.keras.layers.Dropout(self.dropout_rate)

    self._graph_iso_layer = GraphIsomorphismLayer(
        self.mlp_layers, self.hidden_dim, self.dropout_rate, learn_eps=self.learn_eps, sparse=self.sparse)

  def call(self, inputs):
    x = self._graph_iso_layer(inputs)
    x = self._activation(x)
    return self._dropout(x)


class GIN(tf.keras.Model):
  """Graph isomorphism network for semi-supevised node classification."""

  def __init__(self, num_layers, mlp_layers, hidden_dim, num_classes, dropout_rate,
               learn_eps, sparse, **kwargs):
    """Initializes a GIN model.

    Args:
      num_layers: (int) Number of gnn layers
      mlp_layers: (int) Number of mlp layers in GIN
      hidden_dim: (list) List of hidden layers dimension
      num_classes: (int) Total number of classes
      dropout_rate: (float) Dropout probability
      learn_eps: (bool) Whether to learn the epsilon weighting
      sparse: (bool) Whether features are sparse
      **kwargs: Keyword arguments for tf.keras.Model.
    """
    super(GIN, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.mlp_layers = mlp_layers
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate
    self.learn_eps = learn_eps
    self.sparse = sparse
    # input layer
    self.gin = [
        GINBlock(
            self.mlp_layers,
            self.hidden_dim[0],
            dropout_rate=dropout_rate,
            learn_eps=learn_eps,
            sparse=sparse),
    ]

    # hidden layers
    for i in range(1, self.num_layers - 1):
      self.gin.append(
          GINBlock(
              self.mlp_layers,
              self.hidden_dim[0],
              dropout_rate=dropout_rate,
              learn_eps=learn_eps,
              sparse=sparse))

    # output layer
    self.classifier = GraphIsomorphismLayer(
        1, self.num_classes, self.dropout_rate, learn_eps=self.learn_eps, sparse=self.sparse)

  def call(self, inputs):
    features, adj = inputs[0], inputs[1]
    for i in range(self.num_layers - 1):
      x = (features, adj)
      features = self.gin[i](x)

    x = (features, adj)
    outputs = self.classifier(x)
    return outputs
