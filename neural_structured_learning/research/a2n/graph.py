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
"""A class to read and store a graph of tuples in memory.

The graph file is a tab separated file with each line containing a tuple.
The first element of the tuple is source entity, second elemnt is the relation
and the third element is the target entity.

Graph.kg_data stores the graph data in a hash table with source entity as key.
It has the following structure:
{s: {t1: [r1, ...], ...}}
So, kg_data[e1][e2] is a list of all relations filling in (e1, ?, e2).

If Graph.add_reverse_graph is True then reverse tuples are stored in
Graph.reverse_kg_data

If Graph.add_inverse_edge is True then for every relation r, the inverse of that
relation is also added to the graph, so for a (s, r, t) tuple in graph the
tuple (t, inv_r, s) is also added.
Note that only one of add_reverse_graph or add_inverse_edge should be true for
a particular graph.

Graph.tuple_store contains the graph tuple stored as (num_tuples, 3) array.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict

import csv
from absl import logging
import numpy as np


class Graph(object):
  """
    Read a knowledge graph to memory
  """

  def __init__(
      self, kg_file, entity_vocab=False, relation_vocab=False,
      add_reverse_graph=False, add_inverse_edge=False, mode="train",
      max_path_length=None, **kwargs
  ):
    """Init Graph class."""
    del kwargs
    if add_reverse_graph and add_inverse_edge:
      raise ValueError(
          "Only one of add_reverse_graph or add_inverse_edge should be used"
      )
    self._raw_kg_file = kg_file
    self.add_reverse_graph = add_reverse_graph
    self.add_inverse_edge = add_inverse_edge
    if add_inverse_edge:
      self.inverse_relation_prefix = "INVERSE:"
    # vocab maps from name to integer id
    if entity_vocab:
      self.entity_vocab = entity_vocab
    else:
      self.entity_vocab = {}
    if relation_vocab:
      self.relation_vocab = relation_vocab
    else:
      self.relation_vocab = {}
    self.ent_vocab_size = len(self.entity_vocab)
    self.rel_vocab_size = len(self.relation_vocab)
    self._num_edges = 0

    self.kg_data = defaultdict(dict)
    self.next_edges = defaultdict(set)
    if self.add_reverse_graph:
      self.reverse_kg_data = defaultdict(dict)
      self.reverse_next_edges = defaultdict(set)
    self.entity_pad_token = "ePAD"
    self.relation_pad_token = "rPAD"
    # self.no_op_relation = "NO_OP"
    self.max_kg_relations = None
    # self.max_ent_sampled = max_ent_sampled
    self.mode = mode
    self.read_graph(mode)
    # inverse vocab maps from integer id to name
    self.inverse_relation_vocab = {
        v: k for k, v in self.relation_vocab.iteritems()
    }
    self.inverse_entity_vocab = {v: k for k, v in self.entity_vocab.iteritems()}
    if mode == "train":
      self.all_entities = set(self.entity_vocab.values())
      self.max_neighbors = self._max_neighbors()
    self.all_reachable_e2 = defaultdict(set)
    self.all_reachable_e2_reverse = defaultdict(set)
    self.tuple_store = []
    self.create_tuple_store()
    logging.debug("Graph data read")
    logging.info("Entity vocab: %d", self.ent_vocab_size)
    logging.info("Relation vocab: %d", self.rel_vocab_size)
    logging.info("Number of edges: %d", self._num_edges)
    self.max_path_length = max_path_length
    # if mode == "train" and self.max_path_length:
    #   self.store_paths()
    # num_reachable_e2s = map(len, self.all_reachable_e2.values())
    # logging.debug("Mean target e2: %.2f +- %.2f and Max target e2: %d" % (
    #     np.mean(num_reachable_e2s), np.std(num_reachable_e2s),
    #     max(num_reachable_e2s)
    # ))

  def get_inverse_relation_from_name(self, rname):
    """Given a relation name, get the name of inverse relation."""
    if rname.startswith(self.inverse_relation_prefix):
      inv_rname = rname.strip(self.inverse_relation_prefix)
    else:
      inv_rname = self.inverse_relation_prefix + rname
    return inv_rname

  def get_inverse_relation_from_id(self, r):
    """Given a relation id (from vocab), get the id of the inverse relation."""
    rname = self.inverse_relation_vocab[r]
    inv_rname = self.get_inverse_relation_from_name(rname)
    inv_r = self.relation_vocab[inv_rname]
    return inv_r

  def _max_neighbors(self):
    """Helper to find neighbors statistics."""
    max_nbrs = 0
    num_nbrs = []
    max_ent = None
    for e1 in self.kg_data:
      nbrs = set(self.kg_data[e1].keys())
      if self.add_reverse_graph:
        nbrs |= set(self.reverse_kg_data[e1].keys())
      if len(nbrs) > max_nbrs:
        max_nbrs = len(nbrs)
        max_ent = self.inverse_entity_vocab[e1]
      num_nbrs.append(len(nbrs))
    logging.info("Average number of neighbors: %.2f +- %.2f",
                 np.mean(num_nbrs), np.std(num_nbrs))
    logging.info("Max neighbors %d of entity %s", max_nbrs, max_ent)
    return max_nbrs

  def read_graph(self, mode="train"):
    """Read the knowledge graph."""
    logging.debug("Reading graph from %s", self._raw_kg_file)
    with open(self._raw_kg_file, "r") as f:
      kg_file = csv.reader(f, delimiter="\t")
      skipped = 0
      for line in kg_file:
        e1 = line[0].strip()
        if e1 not in self.entity_vocab:
          if mode != "train":
            skipped += 1
            continue
          self.entity_vocab[e1] = self.ent_vocab_size
          self.ent_vocab_size += 1
        e1 = self.entity_vocab[e1]

        r = line[1].strip()
        if r not in self.relation_vocab:
          if mode != "train":
            skipped += 1
            continue
          self.relation_vocab[r] = self.rel_vocab_size
          self.rel_vocab_size += 1
        if self.add_inverse_edge:
          inv_r = self.inverse_relation_prefix + r
          if inv_r not in self.relation_vocab:
            self.relation_vocab[inv_r] = self.rel_vocab_size
            self.rel_vocab_size += 1
          inv_r = self.relation_vocab[inv_r]
        r = self.relation_vocab[r]

        e2 = line[2].strip()
        if e2 not in self.entity_vocab:
          if mode != "train":
            skipped += 1
            continue
          self.entity_vocab[e2] = self.ent_vocab_size
          self.ent_vocab_size += 1
        e2 = self.entity_vocab[e2]

        if e2 not in self.kg_data[e1]:
          self.kg_data[e1][e2] = []
        self.kg_data[e1][e2].append(r)
        self.next_edges[e1].add((r, e2))
        if self.add_inverse_edge:
          if e1 not in self.kg_data[e2]:
            self.kg_data[e2][e1] = []
          self.kg_data[e2][e1].append(inv_r)
          self.next_edges[e2].add((inv_r, e1))
          self._num_edges += 1
        if self.add_reverse_graph:
          if e1 not in self.reverse_kg_data[e2]:
            self.reverse_kg_data[e2][e1] = []
          self.reverse_kg_data[e2][e1].append(r)
          self.reverse_next_edges[e2].add((r, e1))
        # if self.mode != 'train':
        #     self.tuple_store.append((e1, r, e2))

        self._num_edges += 1

    logging.info("Skipped %d tuples in mode %s", skipped, mode)

    # if mode == "train" and self.no_op_relation not in self.relation_vocab:
    #   self.relation_vocab[self.no_op_relation] = self.rV
    #   self.rV += 1
    if mode == "train" and self.entity_pad_token not in self.entity_vocab:
      self.entity_vocab[self.entity_pad_token] = self.ent_vocab_size
      self.ent_vocab_size += 1
    if mode == "train" and self.relation_pad_token not in self.relation_vocab:
      self.relation_vocab[self.relation_pad_token] = self.rel_vocab_size
      self.rel_vocab_size += 1

    self.ent_pad = self.entity_vocab[self.entity_pad_token]
    self.rel_pad = self.relation_vocab[self.relation_pad_token]

    # if self.mode != 'train':
    #     self.tuple_store = np.array(self.tuple_store)
    # self.all_reachable_e2 = defaultdict(set)

    if not self.max_kg_relations:
      max_out = 0
      for e1 in self.kg_data:
        nout = 0
        for e2 in self.kg_data[e1]:
          nout += len(self.kg_data[e1][e2])
        max_out = max(max_out, nout)
      logging.info("Max outgoing rels kg: %d", max_out)
      self.max_kg_relations = max_out

  def create_tuple_store(self, train_graph=None, only_one_hop=False):
    """Create a numpy store for training or validation tuples."""
    self.tuple_store = []
    skipped = 0
    for e1 in self.kg_data:
      for e2 in self.kg_data[e1]:
        if only_one_hop and train_graph:
          reachable = e1 in train_graph.kg_data and \
                      e2 in train_graph.kg_data[e1]
          reachable = reachable or (
              e1 in train_graph.kg_text_data and \
              e2 in train_graph.kg_text_data[e1]
          )
        else:
          reachable = True
        if reachable:
          for r in self.kg_data[e1][e2]:
            self.tuple_store.append((e1, r, e2))
            # if self.mode == "train":
            self.all_reachable_e2[(e1, r)].add(e2)
            # if self.add_reverse_graph:
            self.all_reachable_e2_reverse[(e2, r)].add(e1)
        else:
          skipped += len(self.kg_data[e1][e2])
    self.tuple_store = np.array(self.tuple_store)
    logging.info("Unreachable %s tuples skipped: %d", self.mode, skipped)
    logging.info("Remaining %s tuples: %d", self.mode,
                 self.tuple_store.shape[0])

  def store_paths(self):
    """Find and store all paths from all entities upto max_path_length."""
    self.paths = [defaultdict(list) for _ in range(self.max_path_length)]
    # Add all paths of length 1
    for e in self.kg_data:
      self.paths[0][e] = ["%d %d" % (r, e2) for e2 in self.kg_data[e]
                          for r in self.kg_data[e][e2]]
    # Add all paths of length > 1
    for i in range(1, self.max_path_length):
      for e in self.kg_data:
        for path in self.paths[i-1][e]:
          all_prev_e = map(int, path.strip().split(" ")[1::2])
          last_e = all_prev_e[-1]
          # last_r = int(path.strip().split(" ")[-2])
          if last_e not in self.kg_data:
            continue
          new_paths = [path + " %d %d" % (r, e2) for e2 in self.kg_data[last_e]
                       for r in self.kg_data[last_e][e2]
                       if e2 not in all_prev_e]
          self.paths[i][e] += new_paths
    # import pdb; pdb.set_trace()

  def get_next_kg_actions(
      self, current_ents, query_rels, max_kg_relations=None, mode="train",
      all_answers=None
  ):
    """Get all next actions (edge, next_entity) from current nodes."""
    if not max_kg_relations:
      max_kg_relations = self.max_kg_relations
    actions = np.ones((current_ents.shape[0], max_kg_relations, 2),
                      dtype=np.int32)
    actions[:, :, 0] *= self.entity_vocab[self.entity_pad_token]
    actions[:, :, 1] *= self.relation_vocab[self.relation_pad_token]
    for i in range(current_ents.shape[0]):
      e1 = current_ents[i]
      # actions[i, 0, 0] = e1
      # actions[i, 0, 1] = self.relation_vocab[self.no_op_relation]
      naction = 0
      for e2 in self.kg_data[e1]:
        if naction == max_kg_relations:
          break
        for r in self.kg_data[e1][e2]:
          if naction == max_kg_relations:
            break
          # if r == query_rels[i] and e2 == answers[i]:
          if mode == "train" and r == query_rels[i] and e2 in all_answers[i]:
            actions[i, naction, 0] = self.ePAD
            actions[i, naction, 1] = self.rPAD
          else:
            actions[i, naction, 0] = e2
            actions[i, naction, 1] = r
          naction += 1

    return actions

  def get_next_kg_actions_sampled(
      self, current_ents, all_answers, query_rels, all_negatives,
      max_kg_relations=None
  ):
    """Sample next actions for training."""
    if not max_kg_relations:
      max_kg_relations = self.max_kg_relations
    actions = np.ones((current_ents.shape[0], max_kg_relations, 2),
                      dtype=np.int32)
    actions[:, :, 0] *= self.entity_vocab[self.entity_pad_token]
    actions[:, :, 1] *= self.relation_vocab[self.relation_pad_token]
    for i in range(current_ents.shape[0]):
      e1 = current_ents[i]
      actions[i, 0, 0] = e1
      # actions[i, 0, 1] = self.relation_vocab[self.no_op_relation]
      nactions = 0
      ents = all_answers[i] + all_negatives[i]
      np.random.shuffle(ents)
      answers = set(all_answers[i])
      # negatives = set(all_negatives[i])
      nrels = max_kg_relations / len(ents)
      for e2 in ents:
        # if nactions == self.max_kg_relations:
        #     break
        if nactions >= max_kg_relations:
          logging.info("reached max kg relations")
          break
        # if nactions >= 0.5 * self.max_kg_relations:
        #     # import ipdb; ipdb.set_trace()
        #     break
        # import ipdb; ipdb.set_trace()
        if e2 in self.kg_data[e1]:
          rels = self.kg_data[e1][e2]
        else:
          rels = []
        if len(rels) > nrels:
          rels = np.random.choice(rels, size=nrels, replace=False)
        # if e2 in answers:
          # take all positive relations
        for rel in rels:
          # if nactions >= 0.5 * self.max_kg_relations:
          #     break
          if nactions >= max_kg_relations:
            logging.info("reached max kg relations")
            break
          if rel == query_rels[i] and e2 in answers:
            actions[i, nactions, 0] = self.ePAD
            actions[i, nactions, 1] = self.rPAD
          else:
            actions[i, nactions, 0] = e2
            actions[i, nactions, 1] = rel
          nactions += 1
    return actions
