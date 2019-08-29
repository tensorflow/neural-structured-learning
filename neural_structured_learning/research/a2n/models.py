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
"""Models for the knowledge graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import encoders
import tensorflow as tf


def attention_kbc_model(config, train_graph, is_train_ph, input_tensors):
  """Use attention model to score candidates for kbc."""
  # with tf.variable_scope(scope, reuse=reuse):
    # if reuse:
    #   tf_scope.reuse_variables()
  s, nbrs_s, r, candidates, nbrs_candidates = input_tensors
  model = {}
  entity_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph, train_dropout=config.entity_encoder_dropout,
      input_dim=train_graph.ent_vocab_size,
      scope="entity_embeddings"
  )
  model["entity_encoder"] = entity_encoder
  if config.use_separate_attention_emb:
    init_entity_encoder = encoders.EmbeddingLookup(
        config.emb_dim, is_train_ph,
        train_dropout=config.init_entity_encoder_dropout,
        input_dim=train_graph.ent_vocab_size, scope="init_entity_embeddings"
    )
  else:
    init_entity_encoder = entity_encoder
  model["init_entity_encoder"] = init_entity_encoder
  relation_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph,
      train_dropout=config.relation_encoder_dropout,
      input_dim=train_graph.rel_vocab_size, scope="relation_embeddings"
  )
  model["relation_encoder"] = relation_encoder

  attention_encoder = encoders.NbrAttentionEmbedding(
      config.emb_dim, is_train_ph,
      train_dropout=config.attention_encoder_dropout,
      emb_dim=config.emb_dim, scope="attention"
  )
  model["attention_encoder"] = attention_encoder

  source_emb = entity_encoder.lookup(s)
  nbrs_source_emb = init_entity_encoder.lookup(nbrs_s)
  relation_emb = relation_encoder.lookup(r)
  candidates_emb = entity_encoder.lookup(candidates)
  candidates_emb_flat = tf.reshape(
      candidates_emb, (-1, entity_encoder.emb_dim)
  )
  nbrs_candidates_emb = init_entity_encoder.lookup(nbrs_candidates)
  if config.max_neighbors:
    max_neighbors = config.max_neighbors
  else:
    max_neighbors = train_graph.max_neighbors
  nbrs_candidates_emb_flat = tf.reshape(
      nbrs_candidates_emb, (-1, max_neighbors, init_entity_encoder.emb_dim)
  )
  relation_emb_expand = tf.expand_dims(relation_emb, 1)
  relation_emb_tile = tf.tile(
      relation_emb_expand,  # [1, config.max_negatives+1, 1]
      tf.concat([[1], tf.shape(candidates)[-1:], [1]], 0)
  )
  relation_emb_tile_flat = tf.reshape(relation_emb_tile, [-1, config.emb_dim])

  # Perform attention to construct feature vectors
  mask_nbrs_s = tf.cast(tf.not_equal(nbrs_s, train_graph.ent_pad), tf.float32)
  mask_nbrs_candidates = tf.cast(
      tf.reshape(tf.not_equal(nbrs_candidates, train_graph.ent_pad),
                 (-1, max_neighbors)),
      tf.float32
  )

  source_vec = attention_encoder.attend(
      source_emb, nbrs_source_emb, relation_emb, mask_nbrs_s, name="source"
  )
  # source_vec_tile = tf.tile(
  #     tf.expand_dims(source_vec, 1), [1, config.max_negatives + 1, 1]
  # )
  candidates_vec = attention_encoder.attend(
      candidates_emb_flat, nbrs_candidates_emb_flat, relation_emb_tile_flat,
      mask_nbrs_candidates, name="candidates"
  )
  candidates_vec = tf.reshape(
      candidates_vec,
      tf.concat([tf.shape(candidates_emb)[:2], [config.emb_dim]], 0)
  )

  # Score candidates
  source_dot_query = source_vec * relation_emb
  scores = tf.squeeze(
      tf.matmul(
          tf.expand_dims(source_dot_query, 1), candidates_vec, transpose_b=True
      ),
      axis=1
  )

  # loss = losses.softmax_crossentropy(logits=candidates_scores, labels=labels)

  return scores, model


def source_attention_kbc_model(
    config, train_graph, is_train_ph, input_tensors,
    model_type="source_attention"
):
  """Use attention model to score candidates for kbc."""
  # with tf.variable_scope(scope, reuse=reuse):
    # if reuse:
    #   tf_scope.reuse_variables()
  if config.clueweb_data:
    s, nbrs_s, text_nbrs_s, text_nbrs_s_emb, r, candidates = input_tensors
  elif config.text_kg_file:
    s, nbrs_s, text_nbrs_s, r, candidates = input_tensors
  else:
    s, nbrs_s, r, candidates = input_tensors
  model = {}
  entity_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph, train_dropout=config.entity_encoder_dropout,
      input_dim=train_graph.ent_vocab_size,
      scope="entity_embeddings", num_ps_tasks=None
  )
  model["entity_encoder"] = entity_encoder
  relation_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph,
      train_dropout=config.relation_encoder_dropout,
      input_dim=train_graph.rel_vocab_size, scope="relation_embeddings"
  )
  model["relation_encoder"] = relation_encoder

  if config.attention_type == "bilinear":
    attention_encoder = encoders.NbrAttentionEmbedding(
        config.emb_dim, is_train_ph,
        train_dropout=config.attention_encoder_dropout,
        emb_dim=config.emb_dim, scope="attention"
    )
  elif config.attention_type == "sigmoid_bilinear":
    attention_encoder = encoders.SigmoidNbrAttentionEmbedding(
        config.emb_dim, is_train_ph,
        train_dropout=config.attention_encoder_dropout,
        emb_dim=config.emb_dim, scope="attention",
        average=False
    )
  elif config.attention_type == "sigmoid_avg_bilinear":
    attention_encoder = encoders.SigmoidNbrAttentionEmbedding(
        config.emb_dim, is_train_ph,
        train_dropout=config.attention_encoder_dropout,
        emb_dim=config.emb_dim, scope="attention",
        average=True
    )
  elif config.attention_type == "cosine":
    attention_encoder = encoders.CosineNbrAttentionEmbedding(
        config.emb_dim, is_train_ph,
        train_dropout=config.attention_encoder_dropout,
        emb_dim=config.emb_dim, scope="attention"
    )
  elif config.attention_type == "relation":
    attention_relation_encoder = encoders.EmbeddingLookup(
        config.emb_dim, is_train_ph,
        train_dropout=config.relation_encoder_dropout,
        input_dim=train_graph.rel_vocab_size,
        scope="attention_relation_embeddings"
    )
    model["attention_relation_encoder"] = attention_relation_encoder
    attention_encoder = encoders.RelAttentionEmbedding(
        config.emb_dim, is_train_ph,
        train_dropout=config.attention_encoder_dropout,
        emb_dim=config.emb_dim, scope="attention"
    )

  model["attention_encoder"] = attention_encoder

  source_emb = entity_encoder.lookup(s)
  relation_emb = relation_encoder.lookup(r)
  candidates_emb = entity_encoder.lookup(candidates)
  if model_type == "source_rel_attention":
    if config.attention_type == "relation":
      # nbrs_rel_emb = relation_encoder.lookup(nbrs_s[:, :, 0])
      nbrs_rel_emb = attention_relation_encoder.lookup(nbrs_s[:, :, 0])
      nbrs_ent_emb = entity_encoder.lookup(nbrs_s[:, :, 1])
      nbrs_source_emb = (nbrs_rel_emb, nbrs_ent_emb)
    else:
      if config.use_separate_attention_emb:
        nbrs_encoder = encoders.EmbedAlternateSeq(
            config.emb_dim, is_train_ph,
            train_dropout=config.init_entity_encoder_dropout,
            input_dim_a=train_graph.rel_vocab_size,
            input_dim_b=train_graph.ent_vocab_size
        )
      else:
        nbrs_encoder = encoders.EmbedAlternateSeq(
            config.emb_dim, is_train_ph,
            train_dropout=config.init_entity_encoder_dropout,
            embeddings_a=relation_encoder.embeddings,
            embeddings_b=entity_encoder.embeddings)
      model["nbrs_encoder"] = nbrs_encoder
      nbrs_s_flat = tf.reshape(nbrs_s, (-1, 2))
      nbrs_source_emb_flat = tf.squeeze(nbrs_encoder.embed(nbrs_s_flat), axis=1)
      if config.max_neighbors:
        max_neighbors = config.max_neighbors
      else:
        max_neighbors = train_graph.ent_vocab_size
      nbrs_source_emb = tf.reshape(nbrs_source_emb_flat,
                                   (-1, max_neighbors, config.emb_dim))
    mask_nbrs_s = tf.cast(
        tf.not_equal(nbrs_s[:, :, 1], train_graph.ent_pad), tf.float32
    )
  elif model_type == "source_path_attention":
    if config.use_separate_attention_emb:
      nbrs_encoder = encoders.EmbedAlternateSeq(
          config.emb_dim, is_train_ph,
          train_dropout=config.init_entity_encoder_dropout,
          input_dim_a=train_graph.rel_vocab_size,
          input_dim_b=train_graph.ent_vocab_size)
    else:
      nbrs_encoder = encoders.EmbedAlternateSeq(
          config.emb_dim, is_train_ph,
          train_dropout=config.init_entity_encoder_dropout,
          embeddings_a=relation_encoder.embeddings,
          embeddings_b=entity_encoder.embeddings)
    model["nbrs_encoder"] = nbrs_encoder
    nbrs_s_flat = tf.reshape(nbrs_s, (-1, config.max_path_length * 2))
    nbrs_source_emb_flat = nbrs_encoder.embed(nbrs_s_flat)
    # path_encoder = encoders.AverageSeqEncoder(config.emb_dim,
    #                                           config.max_path_length)
    path_encoder = encoders.PositionSumSeqEncoder(config.emb_dim,
                                                  config.max_path_length)
    model["path_encoder"] = path_encoder
    path_mask = tf.cast(
        tf.not_equal(
            tf.reshape(nbrs_s[:, :, 1::2], (-1, config.max_path_length)),
            train_graph.ent_pad
        ), tf.float32
    )
    path_embeddings = path_encoder.embed(nbrs_source_emb_flat, path_mask)
    if config.max_neighbors:
      max_neighbors = config.max_neighbors
    else:
      max_neighbors = train_graph.ent_vocab_size
    nbrs_source_emb = tf.reshape(path_embeddings,
                                 (-1, max_neighbors, config.emb_dim))
    mask_nbrs_s = tf.cast(
        tf.not_equal(nbrs_s[:, :, 1], train_graph.ent_pad), tf.float32
    )
  else:
    if config.use_separate_attention_emb:
      init_entity_encoder = encoders.EmbeddingLookup(
          config.emb_dim, is_train_ph,
          train_dropout=config.init_entity_encoder_dropout,
          input_dim=train_graph.ent_vocab_size, scope="init_entity_embeddings"
      )
    else:
      init_entity_encoder = entity_encoder
    model["init_entity_encoder"] = init_entity_encoder
    nbrs_source_emb = init_entity_encoder.lookup(nbrs_s)
    mask_nbrs_s = tf.cast(tf.not_equal(nbrs_s, train_graph.ent_pad), tf.float32)

  if config.text_kg_file or config.clueweb_data:
    if config.text_kg_file:
      max_text_len = config.max_text_len or train_graph.max_text_len
      text_encoder = encoders.ConvTextEncoder(
          train_graph.word_vocab_size, config.emb_dim, config.emb_dim,
          max_text_len, is_train_ph,
          train_dropout=config.text_encoder_dropout,
          filter_widths=map(int, config.text_encoder_filter_widths),
          num_filters=config.text_encoder_num_filters,
          nonlinearity=config.text_encoder_nonlinearity,
          num_ps_tasks=None
      )
      model["text_encoder"] = text_encoder
      text_ents = text_nbrs_s[:, :, 0]
      text_rels = text_nbrs_s[:, :, 1:]
      text_rels_flat = tf.reshape(text_rels, (-1, max_text_len))
      text_mask = tf.cast(
          tf.not_equal(text_rels_flat,
                       train_graph.vocab[train_graph.mask_token]),
          tf.float32
      )
      text_emb_flat = text_encoder.embed(text_rels_flat, text_mask)
      text_emb_dim = config.emb_dim
    else:
      text_ents = text_nbrs_s
      text_emb_flat = tf.reshape(text_nbrs_s_emb, (-1, config.text_emb_dim))
      text_emb_dim = config.text_emb_dim
    text_rels_mask = tf.cast(
        tf.not_equal(text_ents, train_graph.ent_pad), tf.float32
    )
    mask_nbrs_s = tf.concat([mask_nbrs_s, text_rels_mask], axis=1)
    # text_ent_emb = entity_encoder.lookup(text_ents) * tf.expand_dims(
    #     text_rels_mask, -1)
    text_ent_emb = entity_encoder.lookup(text_ents)
    if config.attention_type == "relation":
      text_final_emb = tf.reshape(
          text_emb_flat, (-1, config.max_text_neighbors, text_emb_dim)
      )
      nbrs_rel_emb, nbrs_ent_emb = nbrs_source_emb
      all_nbrs_rel_emb = tf.concat([nbrs_rel_emb, text_final_emb], axis=1)
      all_nbrs_ent_emb = tf.concat([nbrs_ent_emb, text_ent_emb], axis=1)
      nbrs_source_emb = (all_nbrs_rel_emb, all_nbrs_ent_emb)
    else:
      text_ent_emb_flat = tf.reshape(text_ent_emb, (-1, config.emb_dim))
      text_emb_concat = tf.concat([text_emb_flat, text_ent_emb_flat], axis=-1)
      with tf.variable_scope("text_rel_prject"):
        w_project = tf.get_variable(
            "W_project", shape=(text_emb_dim + config.emb_dim, config.emb_dim),
            initializer=tf.glorot_uniform_initializer()
        )
      text_emb = tf.matmul(text_emb_concat, w_project)
      text_final_emb = tf.reshape(
          text_emb, (-1, config.max_text_neighbors, config.emb_dim)
      )
      nbrs_source_emb = tf.concat([nbrs_source_emb, text_final_emb], axis=1)
    # import pdb; pdb.set_trace()

  # Perform attention to construct source feature vectors
  source_vec = attention_encoder.attend(
      source_emb, nbrs_source_emb, relation_emb, mask_nbrs_s, name="source"
  )

  # Score candidates
  source_dot_query = source_vec * relation_emb
  scores = tf.squeeze(
      tf.matmul(
          tf.expand_dims(source_dot_query, 1), candidates_emb, transpose_b=True
      ),
      axis=1
  )

  return scores, model


def distmult_kbc_model(config, train_graph, is_train_ph, input_tensors):
  """Use DistMult model to score candidates for kbc."""
  s, r, candidates = input_tensors
  model = {}
  entity_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph, train_dropout=config.entity_encoder_dropout,
      input_dim=train_graph.ent_vocab_size,
      scope="entity_embeddings", use_tanh=config.use_tanh
  )
  model["entity_encoder"] = entity_encoder

  relation_encoder = encoders.EmbeddingLookup(
      config.emb_dim, is_train_ph,
      train_dropout=config.relation_encoder_dropout,
      input_dim=train_graph.rel_vocab_size, scope="relation_embeddings",
      use_tanh=config.use_tanh
  )
  model["relation_encoder"] = relation_encoder

  source_emb = entity_encoder.lookup(s)
  relation_emb = relation_encoder.lookup(r)
  candidates_emb = entity_encoder.lookup(candidates)

  # Score candidates
  source_dot_query = source_emb * relation_emb
  scores = tf.squeeze(
      tf.matmul(
          tf.expand_dims(source_dot_query, 1), candidates_emb, transpose_b=True
      ),
      axis=1
  )

  return scores, model
