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

  def __init__(self, output_dim, bias, sparse=True, **kwargs):
    """Initializes the GraphConvLayer.

    Args:
      output_dim: (int) Output dimension of gcn layer
      bias: (bool) Whether bias needs to be added to the layer
      sparse: (bool) sparse: Whether features are sparse
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GraphConvLayer, self).__init__(**kwargs)
    self.output_dim = output_dim
    self.bias = bias
    self.sparse = sparse

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
    if self.sparse:
      x = tf.sparse.sparse_dense_matmul(adj, x)
    else:
      x = tf.matmul(adj, x)
    outputs = tf.matmul(x, self.weight)

    if self.bias:
      return self.b + outputs
    else:
      return outputs


class GraphAttnLayer(tf.keras.layers.Layer):
  """Single graph attention layer."""

  def __init__(self, output_dim, dropout_rate, alpha=0.2, **kwargs):
    """Initializes the GraphAttnLayer.

    Args:
      output_dim: (int) Output dimension of gat layer
      dropout_rate: (float) Dropout probability
      alpha: (float) LeakyReLU angle of alpha
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(GraphAttnLayer, self).__init__(**kwargs)
    self.output_dim = output_dim
    self.dropout_rate = dropout_rate
    self.alpha = alpha

    self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    super(GraphAttnLayer, self).build(input_shape)
    self.weight = self.add_weight(
        name='weight',
        shape=(input_shape[0][-1], self.output_dim),
        initializer='random_normal',
        trainable=True)

    self.attention = self.add_weight(
        name='attention',
        shape=(2 * self.output_dim, 1),
        initializer='random_normal',
        trainable=True)

  def call(self, inputs):
    x, adj = inputs[0], inputs[1]
    x = self.dropout(x)
    x = tf.matmul(x, self.weight)

    # feat shape: [n_data, n_data, output_dim]
    n_data = int(adj.shape[1])

    feat_i = tf.reshape(
        tf.repeat(x, repeats=n_data), [n_data, n_data, self.output_dim])
    feat_j = tf.repeat(tf.expand_dims(x, axis=0), repeats=[n_data], axis=0)

    attn_input = tf.concat([feat_i, feat_j], axis=2)
    # Unnormalized probability.
    energy = tf.squeeze(tf.matmul(attn_input, self.attention), axis=2)
    energy = self.leakyrelu(energy)

    attn = tf.where(adj > 0, energy, -9e15 * tf.ones_like(energy))
    attn = tf.nn.softmax(attn, axis=1)

    attn = self.dropout(attn)
    output = tf.matmul(attn, x)
    return output


class SparseGraphAttnLayer(tf.keras.layers.Layer):
  """ Single sparse graph attention layer."""

  def __init__(self, output_dim, dropout_rate, alpha=0.2, **kwargs):
    """Initializes the GraphAttnLayer.

    Args:
      output_dim: (int) Output dimension of gat layer
      dropout_rate: (float) Dropout probability
      alpha: (float) LeakyReLU angle of alpha
      **kwargs: Keyword arguments for tf.keras.layers.Layer.
    """
    super(SparseGraphAttnLayer, self).__init__(**kwargs)
    self.output_dim = output_dim
    self.dropout_rate = dropout_rate
    self.alpha = alpha

    self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    super(SparseGraphAttnLayer, self).build(input_shape)
    self.weight = self.add_weight(
        name='weight',
        shape=(input_shape[0][-1], self.output_dim),
        initializer='random_normal',
        trainable=True)

    self.attention_row = self.add_weight(
        name='attention',
        shape=(self.output_dim, 1),
        initializer='random_normal',
        trainable=True)

    self.attention_col = self.add_weight(
        name='attention',
        shape=(self.output_dim, 1),
        initializer='random_normal',
        trainable=True)

  def call(self, inputs):
    x, adj = inputs[0], inputs[1]
    x = self.dropout(x)
    x = tf.matmul(x, self.weight)

    attn_row = tf.matmul(x, self.attention_row)
    attn_col = tf.matmul(x, self.attention_col)

    attn = tf.sparse.add(adj * attn_row, adj * tf.transpose(attn_col, perm=[1, 0]))
    attn = tf.sparse.SparseTensor(
      indices=attn.indices,
      values=self.leakyrelu(attn.values),
      dense_shape=attn.dense_shape)

    attn = tf.sparse.reorder(attn)
    attn = tf.sparse.softmax(attn)

    # sparse dropout
    attn = tf.sparse.SparseTensor(
      indices=attn.indices,
      values=self.dropout(attn.values),
      dense_shape=attn.dense_shape)

    output = tf.sparse.sparse_dense_matmul(attn, x)
    return output