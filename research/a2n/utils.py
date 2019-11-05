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
"""Utility functions for project A2N."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tensorflow as tf


def combine_dict(init_dict, add_dict):
  """Add add_dict to init_dict and return init_dict."""
  for k, v in add_dict.iteritems():
    init_dict[k] = v
  return init_dict


def add_variable_summaries(var, var_name_scope):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries/' + var_name_scope):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    var_max = tf.reduce_max(var)
    var_min = tf.reduce_min(var)
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', var_max)
    tf.summary.scalar('min', var_min)
    tf.summary.histogram('histogram', var)


def add_histogram_summary(var, var_name_scope):
  """Just adds a histogram summary for the variable."""
  with tf.name_scope('summaries/' + var_name_scope):
    tf.summary.histogram('histogram', var)


def read_entity_name_mapping(entity_names_file):
  """Read mapping from entity mid to names."""
  entity_names = {}
  with open(entity_names_file) as gf:
    if entity_names_file.endswith('.gz'):
      f = gzip.GzipFile(fileobj=gf)
    else:
      f = gf
    for line in f:
      contents = line.strip().split('\t')
      if len(contents) < 2:
        continue
      # mid, name = contents
      mid = contents[0]
      name = contents[1]
      entity_names['/' + mid] = name
  return entity_names


def save_embedding_vocabs(output_dir, graph, entity_names_file=None):
  """Save entity and relation vocabs to file."""
  # Read entity names
  entity_names = None
  if entity_names_file:
    entity_names = read_entity_name_mapping(entity_names_file)
  # Save entity vocab
  with open(output_dir + '/entity_vocab.tsv', 'w+') as f:
    for i in range(graph.ent_vocab_size):
      name = graph.inverse_entity_vocab[i]
      if entity_names and name in entity_names:
        name += '/' + entity_names[name]
      f.write(name + '\n')
  with open(output_dir + '/relation_vocab.tsv', 'w+') as f:
    for i in range(graph.rel_vocab_size):
      f.write(graph.inverse_relation_vocab[i] + '\n')
  if hasattr(graph, 'vocab'):
    with open(output_dir + '/word_vocab.tsv', 'w+') as f:
      for i in range(graph.word_vocab_size):
        f.write(graph.inverse_word_vocab[i] + '\n')
