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
"""A class representing a dataset input pipeline and an iterator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import clueweb_text_graph
import numpy as np
import tensorflow as tf


def get_graph_nbrhd(train_graph, ent, exclude_tuple):
  """Helper to get neighbor entities excluding a particular tuple."""
  es, er, et = exclude_tuple
  neighborhood = [nbr for nbr in train_graph.kg_data[ent]
                  if ent != es or nbr != et or
                  # er not in train_graph.kg_data[ent][nbr]]
                  (train_graph.kg_data[ent][nbr] - set([er]))]
  if train_graph.add_reverse_graph:
    rev_nighborhood = [nbr for nbr in train_graph.reverse_kg_data[ent]
                       if ent != et or nbr != es or
                       # er not in train_graph.reverse_kg_data[ent][nbr]]
                       (train_graph.reverse_kg_data[ent][nbr] - set([er]))]
    neighborhood += rev_nighborhood
  neighborhood = np.array(list(set(neighborhood)), dtype=int)
  return neighborhood


def get_graph_nbrhd_with_rels(train_graph, ent, exclude_tuple):
  """Helper to get neighbor (rels, ents) excluding a particular tuple."""
  es, er, et = exclude_tuple
  neighborhood = [[r, nbr] for nbr in train_graph.kg_data[ent]
                  for r in train_graph.kg_data[ent][nbr]
                  # if r != er]
                  if ent != es or nbr != et or r != er]
  if not neighborhood:
    neighborhood = [[]]
  # if train_graph.add_reverse_graph:
  #   rev_nighborhood = [nbr for nbr in train_graph.reverse_kg_data[ent]
  #                      if ent != et or nbr != es or
  #                      # er not in train_graph.reverse_kg_data[ent][nbr]]
  #                      (train_graph.reverse_kg_data[ent][nbr] - set([er]))]
  #   neighborhood += rev_nighborhood
  neighborhood = np.array(neighborhood, dtype=int)
  return neighborhood


def get_graph_nbrhd_text(train_graph, ent, max_text_len):
  """Helper to get neighbor text relations."""
  neighborhood = []
  for nbr in train_graph.kg_text_data[ent]:
    for text in train_graph.kg_text_data[ent][nbr]:
      text_edge = [nbr] + text
      text_edge = text_edge[:max_text_len+1]
      len_to_pad = max_text_len + 1 - len(text_edge)
      if len_to_pad:
        text_edge += [train_graph.vocab[train_graph.mask_token]] * len_to_pad
      neighborhood.append(text_edge)
  if not neighborhood:
    neighborhood = [[]]
  # if train_graph.add_reverse_graph:
  #   rev_nighborhood = [nbr for nbr in train_graph.reverse_kg_data[ent]
  #                      if ent != et or nbr != es or
  #                      # er not in train_graph.reverse_kg_data[ent][nbr]]
  #                      (train_graph.reverse_kg_data[ent][nbr] - set([er]))]
  #   neighborhood += rev_nighborhood
  neighborhood = np.array(neighborhood, dtype=int)
  return neighborhood


def get_graph_nbrhd_embd_text(train_graph, ent, max_text_nbrs):
  """Helper to get neighbor text relations from embedded data."""
  neighborhood = []
  neighborhood_emb = []
  for nbr in train_graph.kg_text_data[ent]:
    for sid in train_graph.kg_text_data[ent][nbr]:
      neighborhood.append(nbr)
      eid = train_graph.emb_id_map[sid]
      neighborhood_emb.append(train_graph.embeddings[eid])
  if not neighborhood:
    neighborhood = [[]]
    neighborhood_emb = [np.zeros(train_graph.embeddings[0].size)]
  neighborhood = np.array(neighborhood, dtype=int)
  neighborhood_emb = np.array(neighborhood_emb, dtype=np.float32)
  if neighborhood.shape[0] > max_text_nbrs:
    ids = np.random.choice(np.range(neighborhood.shape[0]),
                           size=max_text_nbrs, replace=False)
    neighborhood = neighborhood[ids]
    neighborhood_emb = neighborhood_emb[ids]
  else:
    neighborhood = sample_or_pad(neighborhood, max_text_nbrs,
                                 pad_value=train_graph.ent_pad)
    neighborhood_emb = sample_or_pad(neighborhood_emb, max_text_nbrs,
                                     pad_value=0)

  return neighborhood, neighborhood_emb

# def get_graph_nbrhd_text_target(train_graph, ent, max_text_len):
#   """Helper to get neighbor text relations."""
#   neighborhood = []
#   for nbr in train_graph.kg_text_data[ent]:
#     for text in train_graph.kg_text_data[ent][nbr]:
#       text_edge = [nbr] + text
#       text_edge = text_edge[:max_text_len+1]
#       len_to_pad = max_text_len + 1 - len(text_edge)
#       if len_to_pad:
#         text_edge += [train_graph.vocab[train_graph.mask_token]] * len_to_pad
#       neighborhood.append(text_edge)
#   if not neighborhood:
#     neighborhood = [[]]
#   # if train_graph.add_reverse_graph:
#   #   rev_nighborhood = [nbr for nbr in train_graph.reverse_kg_data[ent]
#   #                      if ent != et or nbr != es or
#   #                      # er not in train_graph.reverse_kg_data[ent][nbr]]
#   #                      (train_graph.reverse_kg_data[ent][nbr] - set([er]))]
#   #   neighborhood += rev_nighborhood
#   neighborhood = np.array(neighborhood, dtype=np.int)
#   return neighborhood


def _proc_paths(paths, er=None, et=None, max_length=1, pad=(-1, -1)):
  """Process path from string to list of ints, exculde query tuple and pad."""
  assert len(pad) == 2
  out = []
  for path in paths:
    p = map(int, path.strip().split(" "))
    p += list(pad)*(max_length - int(0.5*len(p)))
    if er:
      if p[0] == er and p[1] == et:
        continue
    out.append(p)
  return out


def get_graph_nbrhd_paths(train_graph, ent, exclude_tuple):
  """Helper to get neighbor (rels, ents) excluding a particular tuple."""
  es, er, et = exclude_tuple
  neighborhood = []
  for i in range(train_graph.max_path_length):
    if ent == es:
      paths = _proc_paths(train_graph.paths[i][ent], er, et,
                          train_graph.max_path_length,
                          (train_graph.rel_pad, train_graph.ent_pad))
    else:
      paths = _proc_paths(train_graph.paths[i][ent],
                          max_length=train_graph.max_path_length,
                          pad=(train_graph.rel_pad, train_graph.ent_pad))
    neighborhood += paths
  if not neighborhood:
    neighborhood = [[]]
  neighborhood = np.array(neighborhood, dtype=int)
  return neighborhood


def _sample_next_edges(edges, to_sample):
  if len(edges) < to_sample:
    return edges
  sample_ids = np.random.choice(range(len(edges)), size=to_sample,
                                replace=False)
  return [edges[i] for i in sample_ids]


def get_graph_nbrhd_paths_randwalk(train_graph, ent, exclude_tuple,
                                   max_length=1,
                                   max_paths=200, terminate_prob=0.1,
                                   pad=(-1, -1)):
  """Helper to get paths through random walk excluding a particular tuple."""
  _, er, et = exclude_tuple
  nsample_per_step = int(max_paths ** (1.0 / train_graph.max_path_length))
  neighborhood = []
  # paths of length one
  init_edges = list(train_graph.next_edges[ent] - set((er, et)))
  current_paths = _sample_next_edges(init_edges, nsample_per_step)
  current_paths = map(list, current_paths)
  # import pdb; pdb.set_trace()
  # outlog = ""
  for _ in range(max_length-1):
    next_paths = []
    # outlog += "current_paths: " + str(current_paths) + "\n"
    while len(current_paths) > 0:
      path = current_paths.pop()
      # outlog += "path:" + str(path) + "\n"
      if np.random.random() <= terminate_prob:
        # Terminate this path
        neighborhood.append(path + list(pad)*(max_length - int(0.5*len(path))))
        # outlog += "adding to paths, "
        # outlog += "nbd:" + str(neighborhood) + "\n"
      else:
        # Expand the node
        prev_ents = path[1::2]
        last_ent = path[-1]
        next_edges = _sample_next_edges(list(train_graph.next_edges[last_ent]),
                                        nsample_per_step)
        # outlog += "next_edges:" +  str(next_edges) + "\n"
        for r, e2 in next_edges:
          # outlog += "\t edge:" + str(r) + str(e2) + "\n"
          if e2 in prev_ents:
            # outlog += "skipped " + str(e2) + "\n"
            continue
          next_paths.append(path + [r, e2])
        # outlog += "next_paths:" +  str(next_paths) + "\n"
    current_paths = next_paths
  if current_paths:
    for path in current_paths:
      neighborhood.append(path + list(pad)*(max_length - int(0.5*len(path))))

  # outlog += "final: " + str(neighborhood) + "\n"
  # print(outlog)
  if not neighborhood:
    neighborhood = [[]]
  # import pdb; pdb.set_trace()
  neighborhood = np.array(neighborhood, dtype=int)
  return neighborhood


def sample_or_pad(arr, max_size, pad_value=-1):
  """Helper to pad arr along axis 0 to max_size or subsample to max_size."""
  arr_shape = arr.shape
  if arr.size == 0:
    if isinstance(pad_value, list):
      result = np.ones((max_size, len(pad_value)), dtype=arr.dtype) * pad_value
    else:
      result = np.ones((max_size,), dtype=arr.dtype) * pad_value
  elif arr.shape[0] > max_size:
    if arr.ndim == 1:
      result = np.random.choice(arr, size=max_size, replace=False)
    else:
      idx = np.arange(arr.shape[0])
      np.random.shuffle(idx)
      result = arr[idx[:max_size], :]
  else:
    padding = np.ones((max_size-arr.shape[0],) + arr_shape[1:],
                      dtype=arr.dtype)
    if isinstance(pad_value, list):
      for i in range(len(pad_value)):
        padding[..., i] *= pad_value[i]
    else:
      padding *= pad_value
    result = np.concatenate((arr, padding), axis=0)
  # result = np.pad(arr,
  #                 [[0, max_size-arr.shape[0]]] + ([[0, 0]] * (arr.ndim-1)),
  #                 "constant", constant_values=pad_value)
  return result


class Dataset(object):
  """A class representing a training dataset for KB.
    Dataset.dataset is a tf.data.Dataset object and Dataset.iterator is an
    iterator over the dataset.
    Dataset.input_tensors are the input tensors used in downstream model, it
    can be returned by Dataset.get_input_tensors()
    Dataset iteration parameters are:
    batchsize: size of each mini-batch
    num_epochs: number of epochs to iterate over the dataset

    Each tuple is processed to include the neighborhoods of the entities using
    the following parameters:
    max_neighbors: maximum number of entities in the neighborhood, if None this
    is specified from train_graph
    max_negatives: maximum  number of negative entities samples for each example

  """

  def __init__(self, data_graph, train_graph=None, mode="train",
               max_negatives=2, max_neighbors=None, num_epochs=20,
               batchsize=64, model_type="attention",
               max_text_len=None, max_text_neighbors=None, val_graph=None):
    """Initialize the Dataset object."""
    if not train_graph:
      train_graph = data_graph
    self.train_graph = train_graph
    self.data_graph = data_graph
    self.mode = mode
    if mode != "train":
      if max_negatives:
        self.max_negatives = max_negatives
      else:
        self.max_negatives = train_graph.ent_vocab_size - 1
    else:
      if not max_negatives and mode == "train":
        raise ValueError("Must provide max_negatives value for training.")
      self.max_negatives = max_negatives
    if max_neighbors:
      self.max_neighbors = max_neighbors
    else:
      self.max_neighbors = train_graph.max_neighbors
    self.num_epochs = num_epochs
    self.batchsize = batchsize
    self.iterator = None
    self.input_tensors = None
    self.output_shapes = None
    self.model_type = model_type
    self.max_text_len = max_text_len
    self.max_text_neighbors = max_text_neighbors
    self.val_graph = val_graph

  def _tuple_iterator(self):
    """Iterate over training tuples."""
    if self.mode == "train":
      np.random.shuffle(self.data_graph.tuple_store)
    for example in self.data_graph.tuple_store:
      s, r, t = example
      yield s, r, t, False
      # if self.train_graph.add_inverse_edge:
      #   inv_r = self.train_graph.get_inverse_relation_from_id(r)
      #   yield t, inv_r, s, False
      if self.model_type not in \
          ["source_rel_attention", "source_path_attention"]:
        yield s, r, t, True

  def featurize_each_example(self, example_tuple):
    """Convert each example into padded arrays for input to model."""
    s, r, t, reverse = example_tuple
    if not reverse:
      all_targets = self.train_graph.all_reachable_e2[(s, r)]
      if self.mode != "train":
        # add all correct candidate from val/test set
        all_targets |= self.data_graph.all_reachable_e2[(s, r)]
        if self.val_graph:
          # if provided also remove val tuples for testing
          all_targets |= self.val_graph.all_reachable_e2[(s, r)]
    else:
      all_targets = self.train_graph.all_reachable_e2_reverse[(t, r)]
      if self.mode != "train":
        # add all correct candidate from val/test set
        all_targets |= self.data_graph.all_reachable_e2_reverse[(t, r)]
        if self.val_graph:
          # if provided also remove val tuples for testing
          all_targets |= self.val_graph.all_reachable_e2[(s, r)]
        # switch s and t
        s, t = t, s
    candidate_negatives = list(
        self.train_graph.all_entities -
        (all_targets | set([t]) | set([self.train_graph.ent_pad]))
    )
    # if len(candidate_negatives) > self.max_negatives:
    #   negatives = np.random.choice(candidate_negatives,
    #                                size=self.max_negatives,
    #                                replace=False)
    # else:
    #   negatives = np.array(candidate_negatives)
    negatives = sample_or_pad(
        np.array(candidate_negatives, dtype=int), self.max_negatives,
        pad_value=self.train_graph.ent_pad
    )
    # negatives is an array of shape (max_negatives)
    # candidates will have shape (max_negatives + 1), i.e including the target
    candidates = np.insert(negatives, 0, t, axis=0)

    if self.model_type == "source_rel_attention":
      nbrhd_fn = get_graph_nbrhd_with_rels
      pad_value = [self.train_graph.rel_pad, self.train_graph.ent_pad]
    elif self.model_type == "source_path_attention":
      # nbrhd_fn = get_graph_nbrhd_paths
      nbrhd_fn = lambda x, y, z: get_graph_nbrhd_paths_randwalk(
          x, y, z, max_length=self.train_graph.max_path_length,
          max_paths=self.max_neighbors, terminate_prob=0.1,
          pad=(self.train_graph.rel_pad, self.train_graph.ent_pad)
      )
      pad_value = [self.train_graph.rel_pad, self.train_graph.ent_pad] * \
        self.train_graph.max_path_length
    else:
      nbrhd_fn = get_graph_nbrhd
      pad_value = self.train_graph.ent_pad
    if self.model_type == "distmult":
      nbrs_s = np.array([], dtype=int)
      nbrs_candidates = np.array([], dtype=int)
    elif self.model_type in ["source_attention", "source_rel_attention",
                             "source_path_attention"]:
      nbrs_s = sample_or_pad(nbrhd_fn(self.train_graph, s, (s, r, t)),
                             self.max_neighbors,
                             pad_value=pad_value)
      if isinstance(self.train_graph, clueweb_text_graph.CWTextGraph):
        # this func does paddding in there
        text_nbrs_s, text_nbrs_s_emb = get_graph_nbrhd_embd_text(
            self.train_graph, s, self.max_text_neighbors)
      elif self.max_text_len:
        text_pad_value = [self.train_graph.ent_pad] + \
              [self.train_graph.vocab[self.train_graph.mask_token]] * \
              self.max_text_len
        text_nbrs_s = sample_or_pad(
            get_graph_nbrhd_text(self.train_graph, s, self.max_text_len),
            self.max_text_neighbors, pad_value=text_pad_value
        )
      nbrs_candidates = np.array([], dtype=int)
    else:
      nbrs_s = sample_or_pad(nbrhd_fn(self.train_graph, s, (s, r, t)),
                             self.max_neighbors,
                             pad_value=pad_value)
      nbrs_t = sample_or_pad(nbrhd_fn(self.train_graph, t, (s, r, t)),
                             self.max_neighbors,
                             pad_value=pad_value)
      nbrs_negatives = np.array(
          [sample_or_pad(nbrhd_fn(self.train_graph, cand, (s, r, t)),
                         self.max_neighbors,
                         pad_value=pad_value)
           for cand in negatives]
      )
      # import pdb; pdb.set_trace()
      nbrs_candidates = np.concatenate(
          (np.expand_dims(nbrs_t, 0), nbrs_negatives), axis=0
      )
    if self.mode != "train":
      labels = [t]
    else:
      labels = np.zeros(candidates.shape[0], dtype=int)
      labels[0] = 1
      idx = np.arange(candidates.shape[0])
      np.random.shuffle(idx)
      candidates = candidates[idx]
      if self.model_type == "attention":
        nbrs_candidates = nbrs_candidates[idx]
      labels = labels[idx]
    # import ipdb; ipdb.set_trace()
    if isinstance(self.train_graph, clueweb_text_graph.CWTextGraph):
      return s, nbrs_s, text_nbrs_s, r, candidates, nbrs_candidates, labels, \
             text_nbrs_s_emb
    elif self.max_text_len:
      return s, nbrs_s, text_nbrs_s, r, candidates, nbrs_candidates, labels
    return s, nbrs_s, r, candidates, nbrs_candidates, labels

  def create_dataset_iterator(self, num_parallel=64, prefetch=5,
                              shuffle_buffer=-1):
    """Create a tf.data.Dataset input pipeline and a dataset iterator."""
    # dataset = tf.data.Dataset.from_generator(
    #     self._tuple_iterator,
    #     (tf.int64, tf.int64, tf.int64),
    #     (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
    #     # (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64),
    #     # (tf.TensorShape([]), tf.TensorShape([self.max_neighbors]),
    #     #  tf.TensorShape([1]), tf.TensorShape([self.max_negatives + 1])
    #     #  tf.TensorShape([self.max_negatives + 1, self.max_neighbors]),
    #     #  tf.TensorShape([self.max_negatives + 1]))
    # )
    # if device == "worker":
    #   data_device = tf.device("/job:worker")
    # else:
    #   data_device = tf.device("/cpu:0")
    # with data_device:
    dataset = tf.data.Dataset.from_generator(
        self._tuple_iterator, tf.int64, tf.TensorShape([4])
    )
    if self.mode == "train":
      if shuffle_buffer == -1:
        shuffle_buffer = self.train_graph.tuple_store.shape[0]
      dataset = dataset.shuffle(shuffle_buffer)
    # pylint: disable=g-long-lambda
    output_dtypes = [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64]
    if isinstance(self.train_graph, clueweb_text_graph.CWTextGraph):
      output_dtypes.append(tf.int64)
      output_dtypes.append(tf.float32)
    elif self.max_text_len:
      output_dtypes.append(tf.int64)
    dataset = dataset.map(lambda example_tuple: tf.py_func(
        self.featurize_each_example, [example_tuple],
        output_dtypes
    ), num_parallel_calls=num_parallel)
    dataset = dataset.repeat(self.num_epochs)
    dataset = dataset.batch(self.batchsize)
    dataset = dataset.prefetch(prefetch)
    self.dataset = dataset
    _ = self.get_output_shapes()
    # if self.mode != "train":
    #   self.iterator = dataset.make_initializable_iterator()
    # else:
    #   self.iterator = dataset.make_one_shot_iterator()

  def get_output_shapes(self):
    """Set shapes of tensors."""
    if not self.output_shapes:
      s_shape = tf.TensorShape([None])
      r_shape = tf.TensorShape([None])
      candidates_shape = tf.TensorShape([None, None])
      if self.model_type == "distmult":
        nbrs_s_shape = tf.TensorShape([None])
        nbrs_candidates_shape = tf.TensorShape([None])
      elif self.model_type == "source_attention":
        nbrs_s_shape = tf.TensorShape([None, self.max_neighbors])
        nbrs_candidates_shape = tf.TensorShape([None])
      elif self.model_type == "source_rel_attention":
        nbrs_s_shape = tf.TensorShape([None, self.max_neighbors, 2])
        nbrs_candidates_shape = tf.TensorShape([None])
      elif self.model_type == "source_path_attention":
        nbrs_s_shape = tf.TensorShape(
            [None, self.max_neighbors, 2*self.train_graph.max_path_length]
        )
        nbrs_candidates_shape = tf.TensorShape([None])
      else:
        nbrs_s_shape = tf.TensorShape([None, self.max_neighbors])
        nbrs_candidates_shape = tf.TensorShape([None, None, self.max_neighbors])
      labels_shape = tf.TensorShape([None, None])
      if isinstance(self.train_graph, clueweb_text_graph.CWTextGraph):
        text_nbrs_s_shape = tf.TensorShape([None, self.max_text_neighbors])
        text_nbrs_s_emb_shape = tf.TensorShape([None, self.max_text_neighbors,
                                                None])
        self.output_shapes = (s_shape, nbrs_s_shape, text_nbrs_s_shape, r_shape,
                              candidates_shape, nbrs_candidates_shape,
                              labels_shape, text_nbrs_s_emb_shape)
      elif self.max_text_len:
        text_nbrs_s_shape = tf.TensorShape([None, self.max_text_neighbors,
                                            self.max_text_len+1])
        self.output_shapes = (s_shape, nbrs_s_shape, text_nbrs_s_shape, r_shape,
                              candidates_shape, nbrs_candidates_shape,
                              labels_shape)
      else:
        self.output_shapes = (s_shape, nbrs_s_shape, r_shape, candidates_shape,
                              nbrs_candidates_shape, labels_shape)
    return self.output_shapes

  # def set_input_tensors_shape(self):
  #   self.input_tensors[0].set_shape([None])
  #   self.input_tensors[1].set_shape([None, self.max_neighbors])
  #   self.input_tensors[2].set_shape([None])
  #   # self.input_tensors[3].set_shape([None, self.max_negatives + 1])
  #   # self.input_tensors[4].set_shape(
  #   #     [None, self.max_negatives + 1, self.max_neighbors]
  #   # )
  #   # self.input_tensors[5].set_shape([None, self.max_negatives + 1])
  #   self.input_tensors[3].set_shape([None, None])
  #   self.input_tensors[4].set_shape(
  #       [None, None, self.max_neighbors]
  #   )
  #   self.input_tensors[5].set_shape([None, None])

  # def get_input_tensors(self):
  #   if not self.iterator:
  #     self.create_dataset_iterator()
  #   if not self.input_tensors:
  #     self.input_tensors = self.iterator.get_next()
  #     self.set_input_tensors_shape()
  #   return self.input_tensors



