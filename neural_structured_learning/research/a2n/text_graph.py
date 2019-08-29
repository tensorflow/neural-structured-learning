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
"""Construct a text graph from words or sentences using similarity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import re

from absl import logging
import graph
import numpy as np


def split_into_words(text, delim=" "):
  text = text.strip().split(delim)
  text = [re.sub(r"[0-9]", "0", word.strip()) for word in text if word.strip()]
  return text


class TextGraph(graph.Graph):
  """Augment Graph with text edges."""

  def __init__(self, text_kg_file, skip_new=True, max_text_len=None,
               max_vocab_size=None, min_word_freq=None, **kwargs):
    super(TextGraph, self).__init__(**kwargs)
    self._kg_text_file = text_kg_file
    self._skip_new = skip_new
    self._max_rel_text_len = 0
    self._new_entities = 0
    self.kg_text_data = defaultdict(dict)
    self.vocab = {}
    self.word_vocab_size = 0
    self.max_vocab_size = max_vocab_size
    self.min_word_freq = min_word_freq
    self.create_vocab()
    self.read_text()
    self.max_text_len = max_text_len or self._max_rel_text_len
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
        if len(line) != 4:
          skipped += 1
          continue
        nread += 1
        text = line[1].strip()
        relation_text_words = split_into_words(text, delim=":")
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

  def read_text(self):
    with open(self._kg_text_file, "r") as f:
      nread = 0
      skipped = 0
      for line in f:
        line = line.strip().split("\t")
        if len(line) != 4:
          skipped += 1
          continue
        nread += 1
        e1 = line[0].strip()
        e2 = line[2].strip()
        if self._skip_new and (e1 not in self.entity_vocab or
                               e2 not in self.entity_vocab):
          continue
        if e1 not in self.entity_vocab:
          self.entity_vocab[e1] = self.ent_vocab_size
          self.ent_vocab_size += 1
          self._new_entities += 1
        e1 = self.entity_vocab[e1]

        if e2 not in self.entity_vocab:
          self.entity_vocab[e2] = self.ent_vocab_size
          self.ent_vocab_size += 1
          self._new_entities += 1
        e2 = self.entity_vocab[e2]

        relation_text = line[1].strip()
        # Delimiter for dependency parsed data is ':'
        relation_text_words = split_into_words(relation_text, delim=":")
        if len(relation_text_words) < 3:
          skipped += 1
          continue
        proc_rtext = []
        for w in relation_text_words:
          if w not in self.vocab:
            # self.vocab[w] = self.word_vocab_size
            # self.word_vocab_size += 1
            w = self.vocab[self.unk_token]
          else:
            w = self.vocab[w]
          proc_rtext.append(w)

        if e2 not in self.kg_text_data[e1]:
          self.kg_text_data[e1][e2] = []
        self.kg_text_data[e1][e2].append(proc_rtext)
        if self.add_inverse_edge:
          inv_r = [self.vocab[self.inverse_relation_prefix]] + proc_rtext
          if e1 not in self.kg_text_data[e2]:
            self.kg_text_data[e2][e1] = []
          self.kg_text_data[e2][e1].append(inv_r)
        self._max_rel_text_len = max(
            self._max_rel_text_len, len(proc_rtext)
        )

        if nread % 100000 == 0:
          logging.info("read %d lines", nread)

      logging.info("Read: %d Skipped: %d", nread, skipped)
      logging.info("Vocab size: %d", len(self.vocab))
    self._num_outgoing_rels = [sum([len(self.kg_text_data[e1][e2])
                                    for e2 in self.kg_text_data[e1]])
                               for e1 in self.kg_text_data]
    logging.info(
        "Max outgoing text relations: %d, mean: %.2f +- %.2f",
        np.max(self._num_outgoing_rels), np.mean(self._num_outgoing_rels),
        np.std(self._num_outgoing_rels)
    )
    self.inverse_word_vocab = {v: k for k, v in self.vocab.iteritems()}
    # import ipdb; ipdb.set_trace()
