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
"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import tensorflow as tf


class DatasetFn(object):
  """Knowledge Graph dataset class."""

  def __init__(self, data_path, debug):
    """Creates KG dataset object for data loading.

    Args:
      data_path: Path to directory containing train/valid/test pickle files
        produced by process.py.
      debug: boolean indicating whether to use debug mode or not. If true, the
        dataset will only contain 1000 examples for debugging.
    """
    self.data_path = data_path
    self.debug = debug
    self.data = {}
    for split in ['train', 'test', 'valid']:
      file_path = os.path.join(self.data_path, split + '.pickle')
      with open(file_path, 'rb') as in_file:
        self.data[split] = pkl.load(in_file)
    filters_file = open(os.path.join(self.data_path, 'to_skip.pickle'), 'rb')
    self.to_skip = pkl.load(filters_file)
    filters_file.close()
    max_axis = np.max(self.data['train'], axis=0)
    self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
    self.n_predicates = int(max_axis[1] + 1) * 2

  def get_filters(self,):
    """Return filter dict to compute ranking metrics in the filtered setting."""
    return self.to_skip

  def get_examples(self, split):
    """Get examples in a split.

    Args:
      split: String indicating the split to use (train/valid/test).

    Returns:
      examples: tf.data.Dataset contatining KG triples in a split.
    """
    examples = self.data[split]
    if split == 'train':
      copy = np.copy(examples)
      tmp = np.copy(copy[:, 0])
      copy[:, 0] = copy[:, 2]
      copy[:, 2] = tmp
      copy[:, 1] += self.n_predicates // 2
      examples = np.vstack((examples, copy))
    if self.debug:
      examples = examples[:1000]
      examples = examples.astype(np.int64)
    tf_dataset = tf.data.Dataset.from_tensor_slices(examples)
    if split == 'train':
      buffer_size = examples.shape[0]
      tf_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    return tf_dataset

  def get_shape(self):
    """Returns KG dataset shape."""
    return self.n_entities, self.n_predicates, self.n_entities
