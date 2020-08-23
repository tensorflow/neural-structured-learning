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
"""Utilities for GNN."""
import os

from models import GCN, GAT
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def build_model(model_name, num_layers, hidden_dim, num_classes, dropout_rate,
                num_heads, sparse):
  """Create gnn model and initialize parameters weights."""
  # Convert hidden_dim to integers
  for i in range(len(hidden_dim)):
    hidden_dim[i] = int(hidden_dim[i])

  # Only gcn available now
  if model_name == 'gcn':
    model = GCN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        sparse=sparse,
        bias=True)
  elif model_name == 'gat':
    model = GAT(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        num_heads=num_heads)

  return model


def cal_acc(labels, logits):
  indices = tf.math.argmax(logits, axis=1)
  acc = tf.math.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
  return acc.numpy().item()


def encode_onehot(labels):
  """Provides a mapping from string labels to integer indices."""
  label_index = {
      'Case_Based': 0,
      'Genetic_Algorithms': 1,
      'Neural_Networks': 2,
      'Probabilistic_Methods': 3,
      'Reinforcement_Learning': 4,
      'Rule_Learning': 5,
      'Theory': 6,
  }

  # Convert to onehot label
  num_classes = len(label_index)
  onehot_labels = np.zeros((len(labels), num_classes))
  idx = 0
  for s in labels:
    onehot_labels[idx, label_index[s]] = 1
    idx += 1
  return onehot_labels


def normalize_adj_matrix(adj):
  """Normalize adjacency matrix."""
  rowsum = np.array(adj.sum(1))
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
  return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_features(features):
  """Row-normalize feature matrix."""
  rowsum = np.array(features.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_mat_inv = sp.diags(r_inv)
  features = r_mat_inv.dot(features)
  return features

def sparse_matrix_to_tf_sparse_tensor(matrix):
  """Convert scipy sparse matrix to tf sparse tensor"""
  sp_matrix = matrix.tocoo().astype(np.float32)
  indices = tf.convert_to_tensor(
    np.vstack((sp_matrix.row, sp_matrix.col)).T.astype(np.int64))
  values = tf.convert_to_tensor(sp_matrix.data)
  shape = tf.TensorShape(sp_matrix.shape)
  return tf.sparse.SparseTensor(indices, values, shape)

def load_dataset(dataset, sparse_features, normalize_adj):
  """Loads Cora dataset."""
  dir_path = os.path.join('data', dataset)
  content_path = os.path.join(dir_path, '{}.content'.format(dataset))
  citation_path = os.path.join(dir_path, '{}.cites'.format(dataset))

  content = np.genfromtxt(content_path, dtype=np.dtype(str))

  idx = np.array(content[:, 0], dtype=np.int32)
  features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
  labels = encode_onehot(content[:, -1])

  # Dict which maps paper id to data id
  idx_map = {j: i for i, j in enumerate(idx)}
  edges_unordered = np.genfromtxt(citation_path, dtype=np.int32)
  edges = np.array(
      list(map(idx_map.get, edges_unordered.flatten())),
      dtype=np.int32).reshape(edges_unordered.shape)
  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(labels.shape[0], labels.shape[0]),
                      dtype=np.float32)

  # build symmetric adjacency matrix
  adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  # Add self-connection edge
  adj = adj + sp.eye(adj.shape[0])

  features = normalize_features(features)
  if normalize_adj:
    adj = normalize_adj_matrix(adj)

  # 5% for train, 300 for validation, 1000 for test
  idx_train = slice(140)
  idx_val = slice(200, 500)
  idx_test = slice(500, 1500)

  features = tf.convert_to_tensor(np.array(features.todense()))
  labels = tf.convert_to_tensor(np.where(labels)[1])
  if sparse_features:
    adj = sparse_matrix_to_tf_sparse_tensor(adj)
  else:
    adj = tf.convert_to_tensor(np.array(adj.todense()))

  return adj, features, labels, idx_train, idx_val, idx_test
