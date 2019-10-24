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
"""Functions for evaluating robustness."""
import logging
import scipy

from gam.data.dataset import GCNDataset
from gam.data.dataset import GraphDataset

import numpy as np

from scipy.sparse import coo_matrix


def compute_percent_correct(data):
  """Compute the ratio of edges connecting two nodes with the same label.

  Args:
    data: A GraphDataset object.

  Returns:
    A float representing the ratio of edges among all graph edges that are
    connecting nodes with similar labels.
  """
  correct_edges = 0
  for edge in data.edges:
    if data.get_labels(edge.src) == data.get_labels(edge.tgt):
      correct_edges = correct_edges + 1
  total_edges = len(data.edges)
  return float(correct_edges) / total_edges, correct_edges, total_edges


def add_noisy_edges(data, target_ratio_correct, symmetrical=True):
  """Add noisy graph edges between nodes with different labels.

  Args:
    data: A GraphDataset object.
    target_ratio_correct: A float representing the target ratio of correct edges
      (i.e. edges connecting nodes with the same label) we want the returned
      graph to have.
    symmetrical: A boolean specifying whether the wrong edges are added in both
      directions (from a source node to a target node, and in reverse as well).

  Returns:
    A GraphDataset object matching the content of `data` in all but the graph
    edges. This new graph will have more more edges, with a total ratio of
    `target_ratio_correct` edges connecting two nodes with similar label.
  """
  # Compute the original number of correct edges.
  ratio_correct, correct_edges, total_edges = compute_percent_correct(data)
  logging.info(
      'Ratio correct edges in the initial dataset: %.2f count_correct = %d  '
      'count all = %d', ratio_correct, correct_edges, total_edges)
  assert target_ratio_correct < ratio_correct, (
      'Cannot achieve requested ratio of %f correct edges by adding more wrong '
      'edges, since the original graph already has %f correct edges.') % (
          target_ratio_correct, ratio_correct)

  # Convert edges to sparse matrix, for faster generation of wrong edges.
  num_edges = len(data.edges)
  num_nodes = data.num_samples
  rows = np.zeros((num_edges,), dtype=data.edges[0].src.dtype)
  cols = np.zeros((num_edges,), dtype=data.edges[0].tgt.dtype)
  vals = np.zeros((num_edges,), dtype=data.edges[0].weight.dtype)
  for i, edge in enumerate(data.edges):
    rows[i] = edge.src
    cols[i] = edge.tgt
    vals[i] = edge.weight
  adj = coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
  adj = adj.tocsr()

  # Add wrong edges.
  num_edges_to_add = int(correct_edges / target_ratio_correct - total_edges)
  logging.info('Adding %d wrong edges...', num_edges_to_add)
  if symmetrical:
    # We add half as many edges, because each edge is now added in both
    # directions.
    num_edges_to_add = num_edges_to_add // 2
  for _ in range(num_edges_to_add):
    added = False
    while not added:
      i = np.random.choice(data.num_samples)
      j = np.random.choice(data.num_samples)
      if (i != j and data.get_labels(i) != data.get_labels(j) and
          not adj[i, j]):
        adj[i, j] = 1
        if symmetrical:
          adj[j, i] = 1
        added = True

  # Convert adj back to edges.
  adj = adj.tocoo()
  edges = [
      GraphDataset.Edge(src, tgt, val)
      for src, tgt, val in zip(adj.row, adj.col, adj.data)]

  if isinstance(data, GCNDataset):
    # Preprocessing of adjacency matrix for simple GCN model and conversion to
    # tuple representation.
    adj_normalized = GCNDataset.normalize_adj(
        adj + scipy.eye(adj.shape[0])).astype(np.float32)
    support = GCNDataset.sparse_to_tuple(adj_normalized)

    data_noisy = data.copy(
        edges=edges,
        support=support)
  else:
    data_noisy = data.copy(edges=edges)

  # Compute the final number of correct edges, to verify it matches the target.
  ratio_correct, _, _ = compute_percent_correct(data_noisy)
  logging.info('Ratio correct edges in the noisy dataset: %.2f', ratio_correct)

  return data_noisy
