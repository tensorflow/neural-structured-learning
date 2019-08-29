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
"""Construct graph for the qangaroo dataset (a reading comprehension dataset).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict

from absl import app
from absl import logging
import ijson
import numpy as np


class QangarooGraph(object):
  """Read and yield examples from QAngaroo dataset."""

  def __init__(self, datapath, skip_new=True):
    self.datapath = datapath
    self._skip_new = skip_new
    self._create_entity_relation_vocab()

  def _create_entity_relation_vocab(self):
    """Read all entities and relations."""
    self.entity_vocab = {}
    entity_vocab_freq = defaultdict(int)
    self.query_entity_vocab = {}
    self.candidates_vocab = {}
    self.relation_vocab = {}
    with open(self.datapath) as f:
      reader = ijson.items(f, "item")
      for example in reader:
        # qent = example["query_entity"]
        # if qent["mid"] != "":
        #   qent_id = qent["mid"]
        # else:
        #   qent_id = qent["name"].strip().lower()
        # if qent_id not in self.query_entity_vocab:
        #   self.query_entity_vocab[qent_id] = len(self.query_entity_vocab)

        relation = example["query"].strip().split(" ")[0]
        if relation not in self.relation_vocab:
          self.relation_vocab[relation] = len(self.relation_vocab)

        all_doc_entities = example["entities"]
        for doc_ents in all_doc_entities:
          for ent in doc_ents:
            if ent["mid"] != "":
              ent_id = ent["mid"]
            else:
              ent_id = ent["name"].strip().lower()
            if ent_id not in self.entity_vocab:
              self.entity_vocab[ent_id] = len(self.entity_vocab)
            entity_vocab_freq[ent_id] += 1
    self.entity_vocab_size = len(self.entity_vocab)
    logging.info("Read %d entities", self.entity_vocab_size)
    # logging.info("Read %d query entities", len(self.query_entity_vocab))
    logging.info("Read %d relations", len(self.relation_vocab))
    logging.info("Avg freq: %.3f +- %.3f",
                 np.mean(entity_vocab_freq.values()),
                 np.std(entity_vocab_freq.values()))
    import ipdb; ipdb.set_trace()


def main(argv):
  del argv
  data = "/public_benchmarks/wikihop_parsed.json"
  graph = QangarooGraph(data)

if __name__ == "__main__":
  app.run(main)

