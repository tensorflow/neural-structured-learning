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
r"""Tool that preprocesses Cora data for Graph Keras trainers.

The Cora dataset can be downloaded from:
https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

In particular, this tool does the following:
(a) Converts Cora data (cora.content) into TF Examples,
(b) Parses the Cora citation graph (cora.cites),
(c) Merges/combines the TF Examples and the graph, and
(d) Writes the training and test data in TF Record format.

The 'cora.content' has the following TSV format:

  publication_id<TAB>word_1<TAB>word_2<TAB>...<TAB>publication_label

Each line of cora.content is a publication that:
- Has an integer 'publication_id'
- Described by a 0/1-valued word vector indicating the absence/presence of the
  corresponding word from the dictionary. In other words, each 'word_k' is
  either 0 or 1.
- Has a string 'publication_label' representing the publication category.

The 'cora.cites' is a TSV file that specifies a graph as a set of edges
representing citation relationships among publications. 'cora.cites' has the
following TSV format:

  source_publication_id<TAB>target_publication_id

Each line of cora.cites represents an edge that 'source_publication_id' cites
'target_publication_id'.

This tool first converts all the 'cora.content' into TF Examples. Then for
training data, this tool merges into each labeled Example the features of that
Example's neighbors according to that instance's edges in the graph. Finally,
the merged training examples are written to a TF Record file. The test data
will be written to a TF Record file w/o joining with the neighbors.

Sample usage:

$ python preprocess_cora_dataset.py --max_nbrs=5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

from absl import app
from absl import flags
from absl import logging
from neural_structured_learning.tools import graph_utils
from neural_structured_learning.tools import pack_nbrs
import six
import tensorflow as tf

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_string(
    'input_cora_content', '/tmp/cora/cora.content',
    """Input file for Cora content that contains ID, words and labels.""")
flags.DEFINE_string('input_cora_graph', '/tmp/cora/cora.cites',
                    """Input file for Cora citation graph in TSV format.""")
flags.DEFINE_integer(
    'max_nbrs', None,
    'The maximum number of neighbors to merge into each labeled Example.')
flags.DEFINE_float(
    'train_percentage', 0.8,
    """The percentage of examples to be created as training data. The rest
    are created as test data.""")
flags.DEFINE_string(
    'output_train_data', '/tmp/cora/train_merged_examples.tfr',
    """Output file for training data merged with graph in TF Record format.""")
flags.DEFINE_string('output_test_data', '/tmp/cora/test_examples.tfr',
                    """Output file for test data in TF Record format.""")


def _int64_feature(*value):
  """Returns int64 tf.train.Feature from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def parse_cora_content(in_file, train_percentage):
  """Converts the Cora content (in TSV) to `tf.train.Example` instances.

  This function parses Cora content (in TSV), converts string labels to integer
  label IDs, randomly splits the data into training and test sets, and returns
  the training and test sets as outputs.

  Args:
    in_file: A string indicating the input file path.
    train_percentage: A float indicating the percentage of training examples
      over the dataset.

  Returns:
    train_examples: A dict with keys being example IDs (string) and values being
    `tf.train.Example` instances.
    test_examples: A dict with keys being example IDs (string) and values being
    `tf.train.Example` instances.
  """
  # Provides a mapping from string labels to integer indices.
  label_index = {
      'Case_Based': 0,
      'Genetic_Algorithms': 1,
      'Neural_Networks': 2,
      'Probabilistic_Methods': 3,
      'Reinforcement_Learning': 4,
      'Rule_Learning': 5,
      'Theory': 6,
  }
  # Fixes the random seed so the train/test split can be reproduced.
  random.seed(1)
  train_examples = {}
  test_examples = {}
  with open(in_file, 'rU') as cora_content:
    for line in cora_content:
      entries = line.rstrip('\n').split('\t')
      # entries contains [ID, Word1, Word2, ..., Label]; 'Words' are 0/1 values.
      words = map(int, entries[1:-1])
      features = {
          'words': _int64_feature(*words),
          'label': _int64_feature(label_index[entries[-1]]),
      }
      example_features = tf.train.Example(
          features=tf.train.Features(feature=features))
      example_id = entries[0]
      if random.uniform(0, 1) <= train_percentage:  # for train/test split.
        train_examples[example_id] = example_features
      else:
        test_examples[example_id] = example_features

  return train_examples, test_examples


def main(unused_argv):
  start_time = time.time()

  # Parses Cora content into TF Examples.
  train_examples, test_examples = parse_cora_content(FLAGS.input_cora_content,
                                                     FLAGS.train_percentage)

  graph = graph_utils.read_tsv_graph(FLAGS.input_cora_graph)
  graph_utils.add_undirected_edges(graph)

  # Joins 'train_examples' with 'graph'. 'test_examples' are used as *unlabeled*
  # neighbors for transductive learning purpose. In other words, the labels of
  # test_examples are not used.
  with tf.io.TFRecordWriter(FLAGS.output_train_data) as writer:
    # Here we call a private function in pack_nbrs to join the examples. This is
    # one-off for demonstration purpose only. Later on we will refactor that
    # function to a public API.
    for merged_example in pack_nbrs._join_examples(  # pylint: disable=protected-access
        train_examples, test_examples, graph, FLAGS.max_nbrs):
      writer.write(merged_example.SerializeToString())

  logging.info('Output training data written to TFRecord file: %s.',
               FLAGS.output_train_data)

  # Writes 'test_examples' out w/o joining with the graph since graph
  # regularization is used only during training, not testing/serving.
  with tf.io.TFRecordWriter(FLAGS.output_test_data) as writer:
    for example in six.itervalues(test_examples):
      writer.write(example.SerializeToString())

  logging.info('Output test data written to TFRecord file: %s.',
               FLAGS.output_test_data)
  logging.info('Total running time: %.2f minutes.',
               (time.time() - start_time) / 60.0)


if __name__ == '__main__':
  # Ensures TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
