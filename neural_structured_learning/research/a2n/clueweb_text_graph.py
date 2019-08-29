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
"""Used for text_graph in the initial experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import re

from absl import logging
import graph
import numpy as np
import tensorflow as tf


def split_into_words(text, delim=" ", mention_a=None, mention_b=None):
  """Split line into tokens, substitute mentions with placeholders."""
  text = text.strip().split(delim)
  text = [re.sub(r"[0-9]", "0", word.strip()) for word in text if word.strip()]
  if mention_a:
    output = []
    n = 0
    while n < len(text):
      if n == mention_a[0]:
        output.append("__ARG1__")
        n = mention_a[1] + 1
      elif n == mention_b[0]:
        output.append("__ARG2__")
        n = mention_b[1] + 1
      else:
        word = text[n]
        output.append(word)
        n += 1
    return output
  return text


class CWTextGraph(graph.Graph):
  """Augment Graph with clueweb text edges."""

  def __init__(self, text_kg_file, embeddings_file, sentence_vocab_file,
               skip_new=True, subsample=None, **kwargs):
    super(CWTextGraph, self).__init__(**kwargs)
    self._kg_text_file = text_kg_file
    self._embeddings_file = embeddings_file
    self._sentence_vocab_file = sentence_vocab_file
    self._skip_new = skip_new
    self._max_rel_text_len = 0
    self._new_entities = 0
    self._subsample = subsample
    self.kg_text_data = defaultdict(dict)
    # self.vocab = {}
    # self.vocab["__ARG1__"] = 0
    # self.vocab["__ARG2__"] = 1
    # self.word_vocab_size = 2
    # self.max_vocab_size = max_vocab_size
    # self.min_word_freq = min_word_freq
    # sentences is a list of all the sentences
    # self.sentences = []
    # sentence_embeddings is a mapping from the sentence id in sentences
    # to the embedding vector for that sentence
    # self.sentence_embeddings = {}
    # self.create_vocab()
    self.read()
    # self.max_text_len = max_text_len or self._max_rel_text_len
    logging.info("New entities: %d", self._new_entities)
    logging.info("Max relation text len: %d", self._max_rel_text_len)
    num_reachable_e2s = [len(self.kg_text_data[e].keys())
                         for e in self.kg_text_data.keys()]
    logging.debug("Mean target e2: %.2f +- %.2f and Max target e2: %d",
                  np.mean(num_reachable_e2s), np.std(num_reachable_e2s),
                  max(num_reachable_e2s))

  def get_relation_text(self, text_ids):
    output = [self.inverse_word_vocab[tid] for tid in text_ids
              if tid != self.vocab[self.mask_token]]
    output = " ".join(output)
    return output

  def create_vocab(self):
    vocab_freq = defaultdict(int)
    with open(self._kg_text_file, "r") as f:
      nread = 0
      skipped = 0
      for line in f:
        line = line.strip().split("\t")
        if len(line) != 3:
          skipped += 1
          continue
        nread += 1
        text = line[0].strip()
        # mentions = line[2].strip().split(",")
        relation_text_words = split_into_words(text, delim=" ")
        for w in relation_text_words:
          vocab_freq[w] += 1
    if self.max_vocab_size:
      top_words = sorted(vocab_freq, key=vocab_freq.get, reverse=True)
      self.vocab = {w: i for i, w in enumerate(top_words[:self.max_vocab_size])}
    elif self.min_word_freq:
      self.vocab = {}
      for w in vocab_freq:
        if vocab_freq[w] >= self.min_word_freq:
          self.vocab[w] = len(self.vocab)
    else:
      self.max_vocab_size = len(vocab_freq)
      self.vocab = {w: i for i, w in enumerate(vocab_freq.keys())}
    self.mask_token = "MASK"
    self.unk_token = "UNK"
    self.vocab[self.mask_token] = len(self.vocab)
    self.vocab[self.unk_token] = len(self.vocab)
    if self.add_inverse_edge:
      self.vocab[self.inverse_relation_prefix] = len(self.vocab)
    self.word_vocab_size = len(self.vocab)
    logging.info("Vocab created with %d words out of total %d words in raw",
                 self.word_vocab_size, len(vocab_freq))

  def read(self):
    self.sentences = {}
    self.embeddings = []
    self.emb_id_map = {}
    sentence_ids = set([])
    with open(self._kg_text_file, "r") as f:
      nread = 0
      skipped = 0
      num_sentences = 0
      for line in f:
        line = line.strip().split("\t")
        if len(line) != 3:
          skipped += 1
          continue
        nread += 1
        ent1 = line[1].strip()
        ent2 = line[2].strip()
        # mentions = line[-1].strip().split(",")
        sentence_id = int(line[0].strip())
        if self._skip_new and (ent1 not in self.entity_vocab or
                               ent2 not in self.entity_vocab):
          continue
        if ent1 not in self.entity_vocab:
          self.entity_vocab[ent1] = self.ent_vocab_size
          self.ent_vocab_size += 1
          self._new_entities += 1
        e1 = self.entity_vocab[ent1]

        if ent2 not in self.entity_vocab:
          self.entity_vocab[ent2] = self.ent_vocab_size
          self.ent_vocab_size += 1
          self._new_entities += 1
        e2 = self.entity_vocab[ent2]

        if e2 not in self.kg_text_data[e1]:
          self.kg_text_data[e1][e2] = []
        self.kg_text_data[e1][e2].append(sentence_id)
        num_sentences = max(num_sentences, sentence_id)
        sentence_ids.add(sentence_id)

        if nread % 10000 == 0:
          logging.info("read %d lines", nread)
          # break

      logging.info("Read: %d Skipped: %d", nread, skipped)
      # logging.info("Vocab size: %d", len(self.vocab))

    if self._subsample:
      logging.info("Subsampling relations")
      logging.info("Number of sentences before sampling: %d", len(sentence_ids))
      sentence_ids = set([])
      for e1 in self.kg_text_data:
        for e2 in self.kg_text_data[e1]:
          sids = self.kg_text_data[e1][e2]
          # import ipdb; ipdb.set_trace()
          if len(sids) > self._subsample:
            self.kg_text_data[e1][e2] = np.random.choice(
                sids, size=self._subsample, replace=False).tolist()
          for sid in self.kg_text_data[e1][e2]:
            sentence_ids.add(sid)
      logging.info("Sentences after subsampling: %d", len(sentence_ids))
    # self.embeddings = [None for _ in xrange(len(self.sentences))]
    if self._embeddings_file.endswith(".proto"):
      emb_files = [self._embeddings_file]
    else:
      emb_files = tf.gfile.Glob(self._embeddings_file + "/*.proto")
    for fname in emb_files:
      logging.info("Reading from file %s", fname)
      record_iterator = tf.python_io.tf_record_iterator(
          path=fname)
      nread = 0
      for string_record in record_iterator:
        # import ipdb; ipdb.set_trace()
        example = tf.train.Example()
        example.ParseFromString(string_record)
        # txt = example.features.feature["text"].bytes_list.value[0]
        sid = int(example.features.feature["sid"].int64_list.value[0])
        if sid not in sentence_ids:
          continue
        emb = np.asarray(example.features.feature["embedding"].float_list.value)
        self.emb_id_map[sid] = len(self.embeddings)
        self.embeddings.append(emb)
        # self.sentences[sid] = txt
        nread += 1
        if nread % 100000 == 0:
          logging.info("Read %d sentence embeddings", nread)

    assert len(sentence_ids) == len(self.embeddings)
    self._num_outgoing_rels = [sum([len(self.kg_text_data[e1][e2])
                                    for e2 in self.kg_text_data[e1]])
                               for e1 in self.kg_text_data]
    # import ipdb; ipdb.set_trace()
    logging.info(
        "Max outgoing text relations: %d, mean: %.2f +- %.2f",
        np.max(self._num_outgoing_rels), np.mean(self._num_outgoing_rels),
        np.std(self._num_outgoing_rels)
    )
    # self.inverse_word_vocab = {v: k for k, v in self.vocab.iteritems()}
    # import ipdb; ipdb.set_trace()
