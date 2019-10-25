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
"""Neural network encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils


class Encoder(object):
  """Abstract class representing an encoder object."""

  def __init__(self):
    # Collection is used to store tensors useful for analyzing/debugging
    self.collection = {}

  def add_to_collection(self, name, tensor):
    self.collection[name] = tensor

  def get_from_collection(self, name):
    if name not in self.collection:
      return None
    return self.collection[name]

  def make_feed_dict(self):
    raise NotImplementedError("No make_feed_dict implementation")


class EmbeddingLookup(Encoder):
  """A simple embedding lookup encoder."""

  def __init__(self, emb_dim, is_train, train_dropout=1.0,
               input_dim=None, embeddings=None, scope="embeddings",
               use_tanh=False, num_ps_tasks=None):
    super(EmbeddingLookup, self).__init__()
    self.emb_dim = emb_dim
    self.is_train = is_train
    self.dropout = train_dropout
    self.use_tanh = use_tanh
    with tf.variable_scope(scope):
      if embeddings:
        self.embeddings = embeddings
      else:
        partitioner = None
        if num_ps_tasks:
          partitioner = tf.min_max_variable_partitioner(
              max_partitions=num_ps_tasks
          )
        self.embeddings = tf.get_variable(
            "embeddings", shape=(input_dim, self.emb_dim),
            initializer=tf.glorot_uniform_initializer(),
            partitioner=partitioner
        )
    if not embeddings:
      utils.add_variable_summaries(self.embeddings, scope)

  def lookup(self, inputs):
    """Lookup embeddings for inputs."""
    embedding_layer = tf.nn.embedding_lookup(
        self.embeddings, inputs
    )
    if self.use_tanh:
      embedding_layer = tf.nn.tanh(embedding_layer)
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(embedding_layer, self.dropout, name="dropout"),
        lambda: embedding_layer
    )
    return output

  def make_feed_dict(self):
    return {}


class NbrAttentionEmbedding(Encoder):
  """Compose embedding by attending to neighbors using bilinear dot product."""

  def __init__(self, input_dim, is_train, train_dropout=1.0,
               emb_dim=None, proj_w=None, scope="attention"):
    super(NbrAttentionEmbedding, self).__init__()
    self.input_dim = input_dim
    self.scope = scope
    self.is_train = is_train
    self.dropout = train_dropout
    if emb_dim:
      self.emb_dim = emb_dim
    else:
      # Keep embedding dimension same as input node embedding
      self.emb_dim = self.input_dim
    with tf.variable_scope(scope):
      if proj_w:
        self.proj_w = proj_w
      else:
        self.proj_w = tf.get_variable(
            "W_attention", shape=(2 * self.input_dim, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
    if not proj_w:
      utils.add_variable_summaries(self.proj_w, self.scope + "/W_attention")

  def attend(self, node, neighbors, query, nbr_mask, name=""):
    """Bilinear attention with a diagonal matrix of query."""
    node_query = tf.expand_dims(node * query, 1)
    nbr_scores = tf.squeeze(tf.matmul(node_query, neighbors, transpose_b=True),
                            axis=1)
    # mask out non-existing neighbors by adding a large negative number
    nbr_scores += (1 - nbr_mask) * (-1e7)
    # attention_probs = tf.squeeze(tf.nn.softmax(nbr_scores, axis=-1), axis=-1)
    attention_probs = tf.nn.softmax(nbr_scores, axis=-1)
    self.add_to_collection("attention_probs", attention_probs)
    # add summary to monitor attention weights
    utils.add_histogram_summary(attention_probs,
                                self.scope + "/" + name + "/attention_probs")
    attention_emb = tf.reduce_sum(
        tf.expand_dims(attention_probs, -1) * neighbors, 1
    )
    # Now concat attention_emb with node embedding and then project to emb_dim
    concat_emb = tf.concat([node, attention_emb], -1)
    output_emb = tf.matmul(concat_emb, self.proj_w)
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(output_emb, self.dropout, name="dropout"),
        lambda: output_emb
    )
    return output

  def make_feed_dict(self):
    return {}


class SigmoidNbrAttentionEmbedding(Encoder):
  """Compose embedding by attending to neighbors using bilinear dot product."""

  def __init__(self, input_dim, is_train, train_dropout=1.0,
               emb_dim=None, proj_w=None, scope="attention", average=False):
    super(SigmoidNbrAttentionEmbedding, self).__init__()
    self.input_dim = input_dim
    self.scope = scope
    self.is_train = is_train
    self.dropout = train_dropout
    self.average = average
    if emb_dim:
      self.emb_dim = emb_dim
    else:
      # Keep embedding dimension same as input node embedding
      self.emb_dim = self.input_dim
    with tf.variable_scope(scope):
      if proj_w:
        self.proj_w = proj_w
      else:
        self.proj_w = tf.get_variable(
            "W_attention", shape=(2 * self.input_dim, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
    if not proj_w:
      utils.add_variable_summaries(self.proj_w, self.scope + "/W_attention")

  def attend(self, node, neighbors, query, nbr_mask, name=""):
    """Bilinear attention with a diagonal matrix of query."""
    node_query = tf.expand_dims(node * query, 1)
    nbr_scores = tf.squeeze(tf.matmul(node_query, neighbors, transpose_b=True),
                            axis=1)
    # mask out non-existing neighbors by adding a large negative number
    nbr_scores += (1 - nbr_mask) * (-1e7)
    attention_probs = tf.nn.sigmoid(nbr_scores)
    self.add_to_collection("attention_probs", attention_probs)
    # add summary to monitor attention weights
    utils.add_histogram_summary(attention_probs,
                                self.scope + "/" + name + "/attention_probs")
    attention_emb = tf.reduce_sum(
        tf.expand_dims(attention_probs, -1) * neighbors, 1
    )
    if self.average:
      weights_sum = tf.reduce_sum(attention_probs, axis=-1, keep_dims=True)
      attention_emb /= tf.maximum(weights_sum, 1e-6)
    else:
      # apply tanh to normalize
      attention_emb = tf.nn.tanh(attention_emb)
    # Now concat attention_emb with node embedding and then project to emb_dim
    concat_emb = tf.concat([node, attention_emb], -1)
    output_emb = tf.matmul(concat_emb, self.proj_w)
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(output_emb, self.dropout, name="dropout"),
        lambda: output_emb
    )
    return output

  def make_feed_dict(self):
    return {}


class CosineNbrAttentionEmbedding(Encoder):
  """Compose embedding by attending to neighbors using simple dot product.
  Concatenates and projects (node, query_rel) to an embedding that is used to
  attend to the neighbor embeddings.
  """

  def __init__(self, input_dim, is_train, train_dropout=1.0,
               emb_dim=None, proj_e=None, proj_w=None, scope="attention"):
    super(CosineNbrAttentionEmbedding, self).__init__()
    self.input_dim = input_dim
    self.scope = scope
    self.is_train = is_train
    self.dropout = train_dropout
    if emb_dim:
      self.emb_dim = emb_dim
    else:
      # Keep embedding dimension same as input node embedding
      self.emb_dim = self.input_dim
    with tf.variable_scope(scope):
      if proj_e:
        self.proj_e = proj_e
      else:
        self.proj_e = tf.get_variable(
            "W_embed", shape=(2 * self.input_dim, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
      if proj_w:
        self.proj_w = proj_w
      else:
        self.proj_w = tf.get_variable(
            "W_attention", shape=(2 * self.input_dim, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
    if not proj_w:
      utils.add_variable_summaries(self.proj_w, self.scope + "/W_attention")

  def attend(self, node, neighbors, query, nbr_mask, name=""):
    """Bilinear attention with a diagonal matrix of query."""
    node_query = tf.concat([node, query], axis=-1)
    node_emb = tf.matmul(node_query, self.proj_e)
    node_emb = tf.expand_dims(node_emb, 1)
    nbr_scores = tf.squeeze(
        tf.matmul(node_emb, neighbors, transpose_b=True), axis=1
    )
    # mask out non-existing neighbors by adding a large negative number
    nbr_scores += (1 - nbr_mask) * (-1e7)
    attention_probs = tf.squeeze(tf.nn.softmax(nbr_scores, axis=-1))
    self.add_to_collection("attention_probs", attention_probs)
    # add summary to monitor attention weights
    utils.add_histogram_summary(attention_probs,
                                self.scope + "/" + name + "/attention_probs")
    attention_emb = tf.reduce_sum(
        tf.expand_dims(attention_probs, -1) * neighbors, 1
    )
    # Now concat attention_emb with node embedding and then project to emb_dim
    concat_emb = tf.concat([node, attention_emb], -1)
    output_emb = tf.matmul(concat_emb, self.proj_w)
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(output_emb, self.dropout, name="dropout"),
        lambda: output_emb
    )

    return output

  def make_feed_dict(self):
    return {}


class RelAttentionEmbedding(NbrAttentionEmbedding):
  """Compose embedding by attending to neighboring relations."""

  def get_attention_probs(self, query, neighbors, nbr_mask, name=""):
    """Get neighbor attention probabilities given query."""
    query = tf.expand_dims(query, 1)
    nbr_scores = tf.squeeze(tf.matmul(query, neighbors, transpose_b=True),
                            axis=1)
    # mask out non-existing neighbors by adding a large negative number
    nbr_scores += (1 - nbr_mask) * (-1e7)
    # attention_probs = tf.squeeze(tf.nn.softmax(nbr_scores, axis=-1), axis=-1)
    attention_probs = tf.nn.softmax(nbr_scores, axis=-1)
    self.add_to_collection("attention_probs", attention_probs)
    # add summary to monitor attention weights
    utils.add_histogram_summary(attention_probs,
                                self.scope + "/" + name + "/attention_probs")
    return attention_probs

  def attend(self, node, neighbors, query, nbr_mask, name=""):
    """Bilinear attention with a diagonal matrix of query."""
    nbrs_rels, nbrs_ents = neighbors
    attention_probs = self.get_attention_probs(query, nbrs_rels, nbr_mask, name)
    attention_emb = tf.reduce_sum(
        tf.expand_dims(attention_probs, -1) * nbrs_ents, 1
    )
    # Now concat attention_emb with node embedding and then project to emb_dim
    concat_emb = tf.concat([node, attention_emb], -1)
    output_emb = tf.matmul(concat_emb, self.proj_w)
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(output_emb, self.dropout, name="dropout"),
        lambda: output_emb
    )
    return output

  def make_feed_dict(self):
    return {}


class EmbedAlternateSeq(Encoder):
  """Given a sequence of even length, embed pairs of adjacent elements.
     Input is (batchsize, max_seqlength)
     seqlength should be even
     Output is (batchsize, 0.5*max_seqlength, emb_dim)

     This will embed all even and all odd elements of the sequence separately,
     concatenate the even embedding with the odd embeddings and then project the
     result to emb_dim.
     This is useful to project a sequence of [(rel, ent), ...] into a sequence
     of vectors for each (rel, ent) pair.
  """

  def __init__(self, emb_dim, is_train, train_dropout=1.0, input_dim_a=None,
               input_dim_b=None, embeddings_a=None, embeddings_b=None,
               scope="embed_pairs"):
    super(EmbedAlternateSeq, self).__init__()
    self.emb_dim = emb_dim
    self.is_train = is_train
    self.dropout = train_dropout
    with tf.variable_scope(scope):
      if embeddings_a:
        self.embeddings_a = embeddings_a
      else:
        self.embeddings_a = tf.get_variable(
            "embeddings_a", shape=(input_dim_a, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
      if embeddings_b:
        self.embeddings_b = embeddings_b
      else:
        self.embeddings_b = tf.get_variable(
            "embeddings_b", shape=(input_dim_b, self.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
      self.proj_w = tf.get_variable(
          "W_embed_pair", shape=(2 * self.emb_dim, self.emb_dim),
          initializer=tf.glorot_uniform_initializer()
      )
    if not embeddings_a:
      utils.add_variable_summaries(self.embeddings_a, scope)
    if not embeddings_b:
      utils.add_variable_summaries(self.embeddings_b, scope)

  def _lookup(self, embeddings, inputs):
    embedding_layer = tf.nn.embedding_lookup(
        embeddings, inputs
    )
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(embedding_layer, self.dropout, name="dropout"),
        lambda: embedding_layer
    )
    return output

  def embed(self, inputs):
    """Embed the input."""
    seq_embed_a = self._lookup(self.embeddings_a, inputs[:, 0::2])
    seq_embed_b = self._lookup(self.embeddings_b, inputs[:, 1::2])
    seq_embeddings = tf.concat([seq_embed_a, seq_embed_b], axis=-1)
    seq_embeddings_flat = tf.reshape(seq_embeddings, (-1, 2*self.emb_dim))
    final_embeddings_flat = tf.matmul(seq_embeddings_flat, self.proj_w)
    final_embeddings = tf.reshape(
        final_embeddings_flat,
        tf.concat([tf.shape(seq_embeddings)[:2], [self.emb_dim]], 0)
    )
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(final_embeddings, self.dropout, name="dropout"),
        lambda: final_embeddings
    )
    return output


class AverageSeqEncoder(Encoder):
  """Encode a sequence by averaging the input embeddings."""

  def __init__(self, emb_dim, max_seq_len):
    super(AverageSeqEncoder, self).__init__()
    self.max_seq_len = max_seq_len
    self.emb_dim = emb_dim

  def embed(self, inputs, mask):
    inputs = inputs * tf.expand_dims(mask, -1)
    counts = tf.reduce_sum(mask, -1, keep_dims=True)
    sum_inp = tf.reduce_sum(inputs, axis=1)
    output = sum_inp / tf.maximum(counts, 1)
    self.add_to_collection("output", output)
    return output


class PositionSumSeqEncoder(Encoder):
  """Encode a sequence by averaging the input embeddings."""

  def __init__(self, emb_dim, max_seq_len, scope="seqmodel"):
    super(PositionSumSeqEncoder, self).__init__()
    self.max_seq_len = max_seq_len
    self.emb_dim = emb_dim
    with tf.variable_scope(scope):
      self.pos_w = tf.get_variable(
          "position_weights", shape=(self.max_seq_len, self.emb_dim),
          initializer=tf.glorot_uniform_initializer()
      )

  def embed(self, inputs, mask):
    inputs = inputs * tf.expand_dims(mask, -1)
    # counts = tf.reduce_sum(mask, -1, keep_dims=True)
    weights = tf.expand_dims(self.pos_w, 0)
    output = tf.reduce_sum(inputs * weights, axis=1)
    # output = sum_inp / tf.maximum(counts, 1)
    self.add_to_collection("output", output)
    return output


class ConvSeqEncoder(Encoder):
  """Encode a sequence using a Convolution Model."""

  def __init__(
      self, emb_dim, input_dim, max_seq_len, is_train,
      train_dropout=1.0, filter_widths=(1, 2), num_filters=64, scope="CNN",
      nonlinearity="tanh"
  ):
    super(ConvSeqEncoder, self).__init__()
    self.max_seq_len = max_seq_len
    self.emb_dim = emb_dim
    self.filter_widths = filter_widths
    self.num_filters = num_filters
    self.filters = {}
    self.is_train = is_train
    self.dropout = train_dropout
    self.input_dim = input_dim
    self.nonlinearity = nonlinearity
    with tf.variable_scope(scope):
      for filter_width in self.filter_widths:
        filter_shape = [filter_width, self.input_dim, 1, self.num_filters]
        w_filter = tf.get_variable(
            "W_filterwidth%d" % filter_width, shape=filter_shape,
            dtype=tf.float32, initializer=tf.truncated_normal_initializer()
        )
        b = tf.get_variable(
            "b_filterwidth%d" % filter_width, shape=[self.num_filters],
            dtype=tf.float32, initializer=tf.constant_initializer(0.1)
        )
        self.filters[filter_width] = (w_filter, b)
      n_out = self.num_filters * len(self.filter_widths)
      w_final = tf.get_variable(
          "W_affine", shape=[n_out, self.emb_dim],
          initializer=tf.glorot_uniform_initializer()
      )
      b_final = tf.get_variable(
          "b_afffine", shape=[self.emb_dim],
          initializer=tf.constant_initializer(0.01)
      )
      self.proj_params = (w_final, b_final)

  def embed(self, inputs, mask):
    """Embed sequence using a layer of 2d Convolution."""
    # Create CNN
    inp = inputs * tf.expand_dims(tf.cast(mask, tf.float32), -1)
    inp = tf.expand_dims(inp, -1)
    outputs = []
    for filter_width in self.filter_widths:
      w_filter, b = self.filters[filter_width]
      conv = tf.nn.conv2d(inp, w_filter, strides=[1, 1, 1, 1],
                          padding="VALID", name="conv")
      conv_bias = tf.nn.bias_add(conv, b)
      # conv_bias = tf.contrib.layers.batch_norm(conv_bias)
      if self.nonlinearity == "relu":
        conv_bias = tf.nn.relu(conv_bias, name="relu")
      else:
        conv_bias = tf.nn.tanh(conv_bias, name="tanh")
      pooled = tf.nn.max_pool(
          conv_bias, ksize=[1, self.max_seq_len - filter_width + 1, 1, 1],
          strides=[1, 1, 1, 1], padding="VALID", name="max_pool"
      )

      outputs.append(pooled)
    n_out = self.num_filters * len(self.filter_widths)
    h_out = tf.concat(outputs, 3)
    conv_output = tf.reshape(h_out, [-1, n_out])

    proj_w, proj_b = self.proj_params
    conv_embedding = tf.nn.xw_plus_b(conv_output, proj_w, proj_b, name="affine")
    out_mask = tf.greater(tf.reduce_sum(mask, -1, keep_dims=True), 0)
    out_mask = tf.cast(out_mask, tf.float32)
    conv_embedding = conv_embedding * out_mask
    output = tf.cond(
        self.is_train,
        lambda: tf.nn.dropout(conv_embedding, self.dropout, name="dropout"),
        lambda: conv_embedding
    )
    return output


class ConvTextEncoder(Encoder):
  """Encode a text sequence using a convolution model."""

  def __init__(
      self, vocab_size, word_emb_dim, output_emb_dim, max_seq_len, is_train,
      train_dropout=1.0, filter_widths=(3, 5, 7), num_filters=64,
      scope="TextCNN", num_ps_tasks=None, nonlinearity="tanh"
  ):
    super(ConvTextEncoder, self).__init__()
    self.vocab_size = vocab_size
    self.word_emb_dim = word_emb_dim
    self.output_emb_dim = output_emb_dim
    self.is_train = is_train
    self.dropout = train_dropout
    self.max_seq_len = max_seq_len
    with tf.variable_scope(scope):
      self.word_embedding_encoder = EmbeddingLookup(
          word_emb_dim, is_train, train_dropout=train_dropout,
          input_dim=vocab_size, num_ps_tasks=num_ps_tasks
      )
      self.cnn_encoder = ConvSeqEncoder(
          output_emb_dim, word_emb_dim, max_seq_len, is_train,
          train_dropout=train_dropout, filter_widths=filter_widths,
          num_filters=num_filters, nonlinearity=nonlinearity
      )

  def embed(self, inputs, mask):
    word_embeddings = self.word_embedding_encoder.lookup(inputs)
    output = self.cnn_encoder.embed(word_embeddings, mask)
    return output

