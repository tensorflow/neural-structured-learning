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
import pickle

from gam.data.dataset import FixedDataset
from gam.data.preprocessing import convert_image
from gam.data.preprocessing import split_train_val_unlabeled

import numpy as np
import tensorflow_datasets as tfds


def load_data_tf_datasets(
    dataset_name, target_num_train_per_class, target_num_val, seed):
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
  data = FixedDataset(train_inputs, train_labels, val_inputs, val_labels,
                      test_inputs, test_labels, unlabeled_inputs,
                      unlabeled_labels, feature_preproc_fn=convert_image)
  return data


def load_data_realistic_ssl(dataset_name, data_path, label_map_path):
  """Loads data from the `ealistic Evaluation of Deep SSL Algorithms`."""
  logging.info('Loading data from pickle at %s.', data_path)
  train_set, validation_set, test_set = pickle.load(
      open(data_path, 'rb'))
  train_inputs = train_set['images']
  train_labels = train_set['labels']
  val_inputs = validation_set['images']
  val_labels = validation_set['labels']
  test_inputs = test_set['images']
  test_labels = test_set['labels']
  # Load label map that specifies which trainining labeles are available.
  train_indices = json.load(open(label_map_path, 'r'))
  train_indices = [int(key.encode('ascii', 'ignore'))
                   for key in train_indices['values']]
  train_indices = np.asarray(train_indices)

  # Select the loaded train indices, and make the rest unlabeled.
  unlabeled_mask = np.ones((train_inputs.shape[0],), dtype=np.bool)
  unlabeled_mask[train_indices] = False
  unlabeled_inputs = train_inputs[unlabeled_mask]
  unlabeled_labels = train_labels[unlabeled_mask]
  train_inputs = train_inputs[train_indices]
  train_labels = train_labels[train_indices]

  # Select a feature preprocessing function, depending on the dataset.
  feature_preproc_fn = ((lambda image: image) if dataset_name == 'cifar10' else
                        convert_image)

  data = FixedDataset(
      train_inputs, train_labels, val_inputs, val_labels, test_inputs,
      test_labels, unlabeled_inputs, unlabeled_labels,
      feature_preproc_fn=feature_preproc_fn)
  return data
