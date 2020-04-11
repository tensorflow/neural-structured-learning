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
"""KG dataset pre-processing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle

import numpy as np


def get_idx(path):
  """Maps entities and relations to unique ids.

  Args:
    path: path to directory with raw dataset files (tab-separated
      tain/valid/test triples).

  Returns:
    ent2idx: Dictionary mapping raw entities to unique ids.
    rel2idx: Dictionary mapping raw relations to unique ids.
  """
  entities, relations = set(), set()
  for split in ['train', 'valid', 'test']:
    with open(os.path.join(path, split), 'r') as lines:
      for line in lines:
        lhs, rel, rhs = line.strip().split('\t')
        entities.add(lhs)
        entities.add(rhs)
        relations.add(rel)
  ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
  rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
  return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
  """Map raw dataset file to numpy array with unique ids.

  Args:
    dataset_file: Path to file contatining raw triples in a split.
    ent2idx: Dictionary mapping raw entities to unique ids.
    rel2idx: Dictionary mapping raw relations to unique ids.

  Returns:
    Numpy array of size n_examples x 3 mapping the raw dataset file to ids.
  """
  examples = []
  with open(dataset_file, 'r') as lines:
    for line in lines:
      lhs, rel, rhs = line.strip().split('\t')
      try:
        examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
      except ValueError:
        continue
  return np.array(examples).astype('int64')


def get_filters(examples, n_relations):
  """Create filtering lists for evaluation.

  Args:
    examples: Numpy array of size n_examples x 3 contatining KG triples.
    n_relations: Int indicating the total number of relations in the KG.

  Returns:
    lhs_final: Dictionary mapping queries (entity, relation) to filtered
               entities for left-hand-side prediction.
    rhs_final: Dictionary mapping queries (entity, relation) to filtered
               entities for right-hand-side prediction.
  """
  lhs_filters = collections.defaultdict(set)
  rhs_filters = collections.defaultdict(set)
  for lhs, rel, rhs in examples:
    rhs_filters[(lhs, rel)].add(rhs)
    lhs_filters[(rhs, rel + n_relations)].add(lhs)
  lhs_final = {}
  rhs_final = {}
  for k, v in lhs_filters.items():
    lhs_final[k] = sorted(list(v))
  for k, v in rhs_filters.items():
    rhs_final[k] = sorted(list(v))
  return lhs_final, rhs_final


def process_dataset(path):
  """Maps entities and relations to ids and saves corresponding pickle arrays.

  Args:
    path: Path to dataset directory.

  Returns:
    examples: Dictionary mapping splits to with Numpy array contatining
              corresponding KG triples.
    filters: Dictionary containing filters for lhs and rhs predictions.
  """
  lhs_skip = collections.defaultdict(set)
  rhs_skip = collections.defaultdict(set)
  ent2idx, rel2idx = get_idx(dataset_path)
  examples = {}
  for split in ['train', 'valid', 'test']:
    dataset_file = os.path.join(path, split)
    examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    lhs_filters, rhs_filters = get_filters(examples[split], len(rel2idx))
    lhs_skip.update(lhs_filters)
    rhs_skip.update(rhs_filters)
  filters = {'lhs': lhs_skip, 'rhs': rhs_skip}
  return examples, filters


if __name__ == '__main__':
  for dataset_name in os.listdir('data/'):
    dataset_path = os.path.join('data/', dataset_name)
    dataset_examples, dataset_filters = process_dataset(dataset_path)
    for dataset_split in ['train', 'valid', 'test']:
      save_path = os.path.join(dataset_path, dataset_split + '.pickle')
      with open(save_path, 'wb') as save_file:
        pickle.dump(dataset_examples[dataset_split], save_file)
    with open(os.path.join(dataset_path, 'to_skip.pickle'), 'wb') as save_file:
      pickle.dump(dataset_filters, save_file)
