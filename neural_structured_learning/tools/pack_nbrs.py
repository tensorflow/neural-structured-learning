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

r"""Program & library to prepare input for graph-based NSL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

from absl import app
from absl import flags
from absl import logging
from neural_structured_learning.tools import graph_utils
import six
import tensorflow as tf


def _read_tfrecord_examples(filename, id_feature_name):
  """Returns a dict containing the Examples read from a TFRecord file.

  Args:
    filename: Name of the TFRecord file to read. Each `tf.train.Example` in the
      input is expected to have a feature named `id` that maps to a singleton
      `bytes_list` value.
    id_feature_name: Name of the singleton `bytes_list` feature in each input
      `tf.train.Example` whose value is the Example's ID.

  Returns:
    A dictionary that maps the ID of each Example to that Example.
  """
  def parse_example(raw_record):
    """Parses and returns a single record of a TFRecord file."""
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    return example

  def get_id(tf_example):
    """Returns the (singleton) value of the Example's `id_feature_name` feature.

    Args:
      tf_example: The `tensorflow.Example` from which to extract the ID feature.
        This is expected to contain a singleton bytes_list value.

    Returns: The ID feature value as a (decoded) string.
    """
    id_feature = tf_example.features.feature[id_feature_name].bytes_list
    return id_feature.value[0].decode('utf-8')

  start_time = time.time()
  logging.info('Reading tf.train.Examples from TFRecord file: %s...', filename)
  result = {}
  for raw_record in tf.data.TFRecordDataset([filename]):
    tf_example = parse_example(raw_record)
    result[get_id(tf_example)] = tf_example
  logging.info('Done reading %d tf.train.Examples from: %s (%.2f seconds).',
               len(result), filename, (time.time() - start_time))
  return result


def _join_examples(seed_exs, nbr_exs, graph, max_nbrs):
  r"""Joins the `seeds` and `nbrs` Examples using the edges in `graph`.

  This generator joins and augments each labeled Example in `seed_exs` with the
  features of at most `max_nbrs` of the seed's neighbors according to the given
  `graph`, and yields each merged result.

  Args:
    seed_exs: A `dict` mapping node IDs to labeled Examples.
    nbr_exs: A `dict` mapping node IDs to unlabeled Examples.
    graph: A `dict`: source -> (target, weight).
    max_nbrs: The maximum number of neighbors to merge into each seed Example,
      or `None` if the number of neighbors per node is unlimited.

  Yields:
    The result of merging each seed Example with the features of its neighbors,
    as described by the module comment.
  """
  # A histogram of the out-degrees of all seed Examples. The keys of this dict
  # range from 0 to 'max_nbrs' (inclusive) if 'max_nbrs' is finite.
  out_degree_count = collections.Counter()

  def has_ex(node_id):
    """Returns true iff 'node_id' is in the 'seed_exs' or 'nbr_exs dict'."""
    result = (node_id in seed_exs) or (node_id in nbr_exs)
    if not result:
      logging.warning('No tf.train.Example found for edge target ID: "%s"',
                      node_id)
    return result

  def lookup_ex(node_id):
    """Returns the Example from `seed_exs` or `nbr_exs` with the given ID."""
    return seed_exs[node_id] if node_id in seed_exs else nbr_exs[node_id]

  def join_seed_to_nbrs(seed_id):
    """Joins the seed with ID `seed_id` to its out-edge graph neighbors.

    This also has the side-effect of maintaining the `out_degree_count`.

    Args:
      seed_id: The ID of the seed Example to start from.

    Returns:
      A list of (nbr_wt, nbr_id) pairs (in decreasing weight order) of the
      seed Example's top `max_nbrs` neighbors. So the resulting list will have
      size at most `max_nbrs`, but it may be less (or even empty if the seed
      Example has no out-edges).
    """
    nbr_dict = graph[seed_id] if seed_id in graph else {}
    nbr_wt_ex_list = [(nbr_wt, nbr_id)
                      for (nbr_id, nbr_wt) in six.iteritems(nbr_dict)
                      if has_ex(nbr_id)]
    result = sorted(nbr_wt_ex_list, reverse=True)[:max_nbrs]
    out_degree_count[len(result)] += 1
    return result

  def merge_examples(seed_ex, nbr_wt_ex_list):
    """Merges neighbor Examples into the given seed Example `seed_ex`.

    Args:
      seed_ex: A labeled Example.
      nbr_wt_ex_list: A list of (nbr_wt, nbr_id) pairs (in decreasing nbr_wt
         order) representing the neighbors of 'seed_ex'.

    Returns:
      The Example that results from merging the features of the neighbor
      Examples (as well as creating a feature for each neighbor's edge weight)
      into `seed_ex`. See the `join()` description above for how the neighbor
      features are named in the result.
    """
    # Make a deep copy of the seed Example to augment.
    merged_ex = tf.train.Example()
    merged_ex.CopyFrom(seed_ex)

    # Add a feature for the number of neighbors.
    merged_ex.features.feature['NL_num_nbrs'].int64_list.value.append(
        len(nbr_wt_ex_list))

    # Enumerate the neighbors, and merge in the features of each.
    for index, (nbr_wt, nbr_id) in enumerate(nbr_wt_ex_list):
      prefix = 'NL_nbr_{}_'.format(index)
      # Add the edge weight value as a new singleton float feature.
      weight_feature = prefix + 'weight'
      merged_ex.features.feature[weight_feature].float_list.value.append(nbr_wt)
      # Copy each of the neighbor Examples features, prefixed with 'prefix'.
      nbr_ex = lookup_ex(nbr_id)
      for (feature_name, feature_val) in six.iteritems(nbr_ex.features.feature):
        new_feature = merged_ex.features.feature[prefix + feature_name]
        new_feature.CopyFrom(feature_val)
    return merged_ex

  start_time = time.time()
  logging.info(
      'Joining seed and neighbor tf.train.Examples with graph edges...')
  for (seed_id, seed_ex) in six.iteritems(seed_exs):
    yield merge_examples(seed_ex, join_seed_to_nbrs(seed_id))
  logging.info(
      'Done creating and writing %d merged tf.train.Examples (%.2f seconds).',
      len(seed_exs), (time.time() - start_time))
  logging.info('Out-degree histogram: %s', sorted(out_degree_count.items()))


def pack_nbrs(labeled_examples_path,
              unlabeled_examples_path,
              graph_path,
              output_training_data_path,
              add_undirected_edges=False,
              max_nbrs=None,
              id_feature_name='id'):
  """Prepares input for graph-based Neural Structured Learning and persists it.

  In particular, this function merges into each labeled training example the
  features from its out-edge neighbor examples according to a supplied
  similarity graph, and persists the resulting (augmented) training data.

  Each `tf.train.Example` read from the files identified by
  `labeled_examples_path` and `unlabeled_examples_path` is expected to have a
  feature that contains its ID (represented as a singleton `bytes_list` value);
  the name of this feature is specified by the value of `id_feature_name`.

  Each edge in the graph specified by `graph_path` is identified by a source
  instance ID, a target instance ID, and an optional edge weight. These edges
  are specified by TSV lines of the following form:

  ```
  source_id<TAB>target_id[<TAB>edge_weight]
  ```

  If no `edge_weight` is specified, it defaults to 1.0. If the input graph is
  not symmetric and if `add_undirected_edges` is `True`, then all edges will be
  treated as bi-directional. To build a graph based on the similarity of
  instances' dense embeddings, see `nsl.tools.build_graph`.

  This function merges into each labeled example the features of that example's
  out-edge neighbors according to that instance's in-edges in the graph. If a
  value is specified for `max_nbrs`, then at most that many neighbors' features
  are merged into each labeled instance (based on which neighbors have the
  largest edge weights, with ties broken using instance IDs).

  Here's how the merging process works. For each labeled example, the features
  of its `i`'th out-edge neighbor will be prefixed by `NL_nbr_<i>_`, with
  indexes `i` in the half-open interval `[0, K)`, where K is the minimum of
  `max_nbrs` and the number of the labeled example's out-edges in the graph. A
  feature named `NL_nbr_<i>_weight` will also be merged into the labeled example
  whose value will be the neighbor's corresponding edge weight. The top
  neighbors to use in this process are selected by consulting the input graph
  and selecting the labeled example's out-edge neighbors with the largest edge
  weight; ties are broken by preferring neighbor IDs with larger lexicographic
  order. Finally, a feature named `NL_num_nbrs` is set on the result (a
  singleton `int64_list`) denoting the number of neighbors `K` merged into the
  labeled example.

  Finally, the merged examples are written to a TFRecord file named by
  `output_training_data_path`.

  Note that this function can also be invoked as a binary from a shell. Sample
  usage:

  `python -m neural_structured_learning.tools.pack_nbrs` [*flags*]
  *labeled.tfr unlabeled.tfr graph.tsv output.tfr*

  For details about this program's flags, run:

  ```
  python -m neural_structured_learning.tools.pack_nbrs --help
  ```

  Args:
    labeled_examples_path: Names a TFRecord file containing labeled
      `tf.train.Example` instances.
    unlabeled_examples_path: Names a TFRecord file containing unlabeled
      `tf.train.Example` instances. This can be an empty string if there are no
      unlabeled examples.
    graph_path: Names a TSV file that specifies a graph as a set of edges
      representing similarity relationships.
    output_training_data_path: Path to a file where the resulting augmented
      training data in the form of `tf.train.Example` instances will be
      persisted in the TFRecord format.
    add_undirected_edges: `Boolean` indicating whether or not to treat adges as
      bi-directional.
    max_nbrs: The maximum number of neighbors to use to generate the augmented
      training data for downstream training.
    id_feature_name: The name of the feature in the input labeled and unlabeled
      `tf.train.Example` objects representing the ID of examples.
  """
  start_time = time.time()

  # Read seed and neighbor TFRecord input files.
  seed_exs = _read_tfrecord_examples(labeled_examples_path, id_feature_name)
  # Unlabeled neighbor input instances are optional. If not provided, all
  # neighbors used will be labeled instances.
  nbr_exs = _read_tfrecord_examples(
      unlabeled_examples_path,
      id_feature_name) if unlabeled_examples_path else {}

  # Read the input graph in TSV format, and conditionally reverse all its edges.
  graph = graph_utils.read_tsv_graph(graph_path)
  if add_undirected_edges:
    graph_utils.add_undirected_edges(graph)

  # Join the edges with the seed and neighbor Examples, and write out the
  # results to the output TFRecord file.
  with tf.io.TFRecordWriter(output_training_data_path) as writer:
    for merged_ex in _join_examples(seed_exs, nbr_exs, graph, max_nbrs):
      writer.write(merged_ex.SerializeToString())
  logging.info('Output written to TFRecord file: %s.',
               output_training_data_path)
  logging.info('Total running time: %.2f minutes.',
               (time.time() - start_time) / 60.0)


def _main(argv):
  """Main function for invoking the `nsl.tools.pack_nbrs` function."""
  flag = flags.FLAGS
  flag.showprefixforinfo = False
  # Check that the correct number of arguments have been provided.
  if len(argv) != 5:
    raise app.UsageError('Invalid number of arguments; expected 4, got %d' %
                         (len(argv) - 1))

  pack_nbrs(
      labeled_examples_path=argv[1],
      unlabeled_examples_path=argv[2],
      graph_path=argv[3],
      output_training_data_path=argv[4],
      add_undirected_edges=flag.add_undirected_edges,
      max_nbrs=flag.max_nbrs,
      id_feature_name=flag.id_feature_name)


if __name__ == '__main__':
  flags.DEFINE_integer(
      'max_nbrs', None,
      'The maximum number of neighbors to merge into each labeled Example.')
  flags.DEFINE_string(
      'id_feature_name', 'id',
      """Name of the singleton bytes_list feature in each input Example
      whose value is the Example's ID.""")
  flags.DEFINE_bool(
      'add_undirected_edges', False,
      """By default, the set of neighbors of a node S are
      only those nodes T such that there is an edge S-->T in the input graph. If
      this flag is True, all edges of the graph will be made symmetric before
      determining each node's neighbors (and in the case where edges S-->T and
      T-->S exist in the input graph with weights w1 and w2, respectively, the
      weight of the symmetric edge will be max(w1, w2)).""")

  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  app.run(_main)
