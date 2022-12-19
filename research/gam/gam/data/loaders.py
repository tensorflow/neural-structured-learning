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
"""Data loaders for Graph Agreement Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import pickle
import sys

from .dataset import Dataset
from .dataset import PlanetoidDataset
import networkx as nx
import numpy as np
from .preprocessing import convert_image
from .preprocessing import split_train_val_unlabeled
from scipy import sparse as sp
import tensorflow_datasets as tfds


def load_data_tf_datasets(dataset_name, target_num_train_per_class,
                          target_num_val, seed):
  """Load and preprocess data from tensorflow_datasets."""
  logging.info('Loading and preprocessing data from tensorflow datasets...')
  # Load train data.
  ds = tfds.load(dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
  ds = tfds.as_numpy(ds)
  train_inputs, train_labels = ds['image'], ds['label']
  # Load test data.
  ds = tfds.load(dataset_name, split=tfds.Split.TEST, batch_size=-1)
  ds = tfds.as_numpy(ds)
  test_inputs, test_labels = ds['image'], ds['label']

  # Remove extra dimensions of size 1.
  train_labels = np.squeeze(train_labels)
  test_labels = np.squeeze(test_labels)

  logging.info('Splitting data...')
  data = split_train_val_unlabeled(train_inputs, train_labels,
                                   target_num_train_per_class, target_num_val,
                                   seed)
  train_inputs = data[0]
  train_labels = data[1]
  val_inputs = data[2]
  val_labels = data[3]
  unlabeled_inputs = data[4]
  unlabeled_labels = data[5]

  logging.info('Converting data to Dataset format...')
  data = Dataset.build_from_splits(
      name=dataset_name,
      inputs_train=train_inputs,
      labels_train=train_labels,
      inputs_val=val_inputs,
      labels_val=val_labels,
      inputs_test=test_inputs,
      labels_test=test_labels,
      inputs_unlabeled=unlabeled_inputs,
      labels_unlabeled=unlabeled_labels,
      feature_preproc_fn=convert_image)
  return data


def load_data_realistic_ssl(dataset_name, data_path, label_map_path):
  """Loads data from the `ealistic Evaluation of Deep SSL Algorithms`."""
  logging.info('Loading data from pickle at %s.', data_path)
  train_set, validation_set, test_set = pickle.load(open(data_path, 'rb'))
  train_inputs = train_set['images']
  train_labels = train_set['labels']
  val_inputs = validation_set['images']
  val_labels = validation_set['labels']
  test_inputs = test_set['images']
  test_labels = test_set['labels']
  # Load label map that specifies which trainining labeles are available.
  train_indices = json.load(open(label_map_path, 'r'))
  train_indices = [
      int(key.encode('ascii', 'ignore')) for key in train_indices['values']
  ]
  train_indices = np.asarray(train_indices)

  # Select the loaded train indices, and make the rest unlabeled.
  unlabeled_mask = np.ones((train_inputs.shape[0],), dtype=bool)
  unlabeled_mask[train_indices] = False
  unlabeled_inputs = train_inputs[unlabeled_mask]
  unlabeled_labels = train_labels[unlabeled_mask]
  train_inputs = train_inputs[train_indices]
  train_labels = train_labels[train_indices]

  # Select a feature preprocessing function, depending on the dataset.
  feature_preproc_fn = ((lambda image: image)
                        if dataset_name == 'cifar10' else convert_image)

  data = Dataset.build_from_splits(
      name=dataset_name,
      inputs_train=train_inputs,
      labels_train=train_labels,
      inputs_val=val_inputs,
      labels_val=val_labels,
      inputs_test=test_inputs,
      labels_test=test_labels,
      inputs_unlabeled=unlabeled_inputs,
      labels_unlabeled=unlabeled_labels,
      feature_preproc_fn=feature_preproc_fn)
  return data


def load_from_planetoid_files(dataset_name, path):
  """Loads Planetoid data in GCN format, as released with the GCN code.

  This function is adapted from https://github.com/tkipf/gcn.

  This function assumes that the following files can be found at the location
  specified by `path`:

  ind.dataset_str.x          => the feature vectors of the training instances
                                as scipy.sparse.csr.csr_matrix object.
  ind.dataset_str.tx         => the feature vectors of the test instances as
                                scipy.sparse.csr.csr_matrix object.
  ind.dataset_str.allx       => the feature vectors of both labeled and
                                unlabeled training instances (a superset of
                                ind.dataset_str.x) as
                                scipy.sparse.csr.csr_matrix object.
  ind.dataset_str.y          => the one-hot labels of the labeled training
                                instances as numpy.ndarray object.
  ind.dataset_str.ty         => the one-hot labels of the test instances as
                                numpy.ndarray object.
  ind.dataset_str.ally       => the labels for instances in
                                ind.dataset_str.allx as numpy.ndarray object.
  ind.dataset_str.graph      => a dict in the format
                                {index: [index_of_neighbor_nodes]} as
                                collections.defaultdict object.
  ind.dataset_str.test.index => the indices of test instances in graph, for
                                the inductive setting as list object.

  Args:
    dataset_name: A string representing the dataset name (e.g., `cora`).
    path: Path to the directory containing the files.

  Returns:
    All data input files loaded (as well the training/test data).
  """

  def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

  def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
      index.append(int(line.strip()))
    return index

  def _load_file(name):
    """Load from data file."""
    filename = 'ind.{}.{}'.format(dataset_name, name)
    filename = os.path.join(path, filename)
    with open(filename, 'rb') as f:
      if sys.version_info > (3, 0):
        return pickle.load(f, encoding='latin1')  # pylint: disable=unexpected-keyword-arg
      else:
        return pickle.load(f)

  x = _load_file('x')
  y = _load_file('y')
  tx = _load_file('tx')
  ty = _load_file('ty')
  allx = _load_file('allx')
  ally = _load_file('ally')
  graph = _load_file('graph')

  filename = 'ind.{}.test.index'.format(dataset_name)
  filename = os.path.join(path, filename)
  test_idx_reorder = _parse_index_file(filename)
  test_idx_range = np.sort(test_idx_reorder)

  if dataset_name == 'citeseer':
    # Fix citeseer dataset (there are some isolated nodes in the graph).
    # Find isolated nodes, add them as zero-vecs into the right position.
    test_idx_range_full = range(
        min(test_idx_reorder),
        max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

  features = sp.vstack((allx, tx)).tolil()
  features[test_idx_reorder, :] = features[test_idx_range, :]
  adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

  labels = np.vstack((ally, ty))
  labels[test_idx_reorder, :] = labels[test_idx_range, :]

  idx_test = test_idx_range.tolist()
  idx_train = range(len(y))
  idx_val = range(len(y), len(y) + 500)

  train_mask = _sample_mask(idx_train, labels.shape[0])
  val_mask = _sample_mask(idx_val, labels.shape[0])
  test_mask = _sample_mask(idx_test, labels.shape[0])

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_mask, :] = labels[train_mask, :]
  y_val[val_mask, :] = labels[val_mask, :]
  y_test[test_mask, :] = labels[test_mask, :]

  return (adj, features, y_train, y_val, y_test, train_mask, val_mask,
          test_mask, labels)


def load_data_planetoid(name, path, splits_path=None, row_normalize=False,
                        data_container_class=PlanetoidDataset):
  """Load Planetoid data."""
  if splits_path is None:
    # Load from file in Planetoid format.
    (adj, features, _, _, _, train_mask, val_mask, test_mask,
     labels) = load_from_planetoid_files(name, path)
  else:
    # Otherwise load from a path where we saved a pickle with random splits.
    logging.info('Loading from splits path: %s', splits_path)
    (adj, features, _, _, _, train_mask, val_mask, test_mask,
     labels) = pickle.load(open(splits_path, 'rb'))

  return data_container_class.build_from_adjacency_matrix(
      name,
      adj,
      features,
      train_mask,
      val_mask,
      test_mask,
      labels,
      row_normalize=row_normalize)
