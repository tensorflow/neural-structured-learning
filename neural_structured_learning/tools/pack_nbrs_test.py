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
"""Tests for neural_structured_learning.tools.pack_nbrs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from neural_structured_learning.tools import graph_utils
from neural_structured_learning.tools import pack_nbrs as pack_nbrs_lib

import tensorflow as tf

from google.protobuf import text_format


def _read_tfrecord_examples(filename):
  """Returns a dict with Examples read from a TFRecord file and keyed by ID."""
  result = {}
  for raw_record in tf.data.TFRecordDataset([filename]):
    tf_example = tf.train.Example()
    tf_example.ParseFromString(raw_record.numpy())
    id_feature = tf_example.features.feature['id'].bytes_list
    result[id_feature.value[0].decode('utf-8')] = tf_example
  return result


def _example_a():
  """Returns the features for node A as a `tf.train.Example` instance."""
  example_str = """
                  features {
                    feature {
                      key: "id"
                      value: { bytes_list { value: [ "A" ] } }
                    }
                    feature {
                      key: "some_int_feature"
                      value: { int64_list { value: [ 1, 1 ] } }
                    }
                    feature {
                      key: "label"
                      value: { int64_list { value: [ 0 ] } }
                    }
                  }
                """
  return text_format.Parse(example_str, tf.train.Example())


def _example_b():
  """Returns the features for node B as a `tf.train.Example` instance."""
  example_str = """
                  features {
                    feature {
                      key: "id"
                      value: { bytes_list { value: [ "B" ] } }
                    }
                    feature {
                      key: "some_int_feature"
                      value: { int64_list { value: [ 2, 2 ] } }
                    }
                    feature {
                      key: "label"
                      value: { int64_list { value: [ 1 ] } }
                    }
                  }
                """
  return text_format.Parse(example_str, tf.train.Example())


def _example_c():
  """Returns the features for node C as a `tf.train.Example` instance."""
  example_str = """
                  features {
                    feature {
                      key: "id"
                      value: { bytes_list { value: [ "C" ] } }
                    }
                    feature {
                      key: "some_int_feature"
                      value: { int64_list { value: [ 3, 3 ] } }
                    }
                    feature {
                      key: "label"
                      value: { int64_list { value: [ 2 ] } }
                    }
                  }
                """
  return text_format.Parse(example_str, tf.train.Example())


def _num_neighbors_example(num_neighbors):
  """Returns a `tf.train.Example` with the 'NL_num_nbrs' feature."""
  example_str = """
                  features {
                    feature {
                      key: "NL_num_nbrs"
                      value: { int64_list { value: [ %d ] } }
                    }
                  }
                """ % (
                    num_neighbors)
  return text_format.Parse(example_str, tf.train.Example())


def _node_as_neighbor(example, neighbor_id, edge_weight):
  """Returns a `tf.train.Example` containing neighbor features."""
  result = tf.train.Example()
  nbr_prefix = 'NL_nbr_{}_'.format(neighbor_id)

  # Add the edge weight value as a new singleton float feature.
  weight_feature = nbr_prefix + 'weight'
  result.features.feature[weight_feature].float_list.value.append(edge_weight)

  # Copy each of the neighbor Example's features, prefixed with 'nbr_prefix'.
  for (feature_name, feature_val) in example.features.feature.items():
    new_feature = result.features.feature[nbr_prefix + feature_name]
    new_feature.CopyFrom(feature_val)
  return result


def _write_examples(examples_file, examples):
  """Writes the given `examples` to the TFRecord file named `examples_file`."""
  with tf.io.TFRecordWriter(examples_file) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


# Note that this is an asymmetric/directed graph.
_GRAPH = {'A': {'B': 0.5, 'C': 0.9}, 'B': {'A': 0.4, 'C': 1.0}}


def _augmented_a_directed_one_nbr():
  """Returns an augmented `tf.train.Example` instance for node A."""
  augmented_a = _example_a()
  augmented_a.MergeFrom(_node_as_neighbor(_example_c(), 0, 0.9))
  augmented_a.MergeFrom(_num_neighbors_example(1))
  return augmented_a


def _augmented_a_directed_two_nbrs():
  """Returns an augmented `tf.train.Example` instance for node A."""
  augmented_a = _example_a()
  augmented_a.MergeFrom(_node_as_neighbor(_example_c(), 0, 0.9))
  augmented_a.MergeFrom(_node_as_neighbor(_example_b(), 1, 0.5))
  augmented_a.MergeFrom(_num_neighbors_example(2))
  return augmented_a


def _augmented_a_undirected_one_nbr():
  """Returns an augmented `tf.train.Example` instance for node A."""
  return _augmented_a_directed_one_nbr()


def _augmented_a_undirected_two_nbrs():
  """Returns an augmented `tf.train.Example` instance for node A."""
  return _augmented_a_directed_two_nbrs()


def _augmented_c_directed():
  """Returns an augmented `tf.train.Example` instance for node C."""
  augmented_c = _example_c()
  augmented_c.MergeFrom(_num_neighbors_example(0))
  return augmented_c


def _augmented_c_undirected_one_nbr_b():
  """Returns an augmented `tf.train.Example` instance for node C with nbr B."""
  augmented_c = _example_c()
  augmented_c.MergeFrom(_node_as_neighbor(_example_b(), 0, 1.0))
  augmented_c.MergeFrom(_num_neighbors_example(1))
  return augmented_c


def _augmented_c_undirected_one_nbr_a():
  """Returns an augmented `tf.train.Example` instance for node C with nbr A."""
  augmented_c = _example_c()
  augmented_c.MergeFrom(_node_as_neighbor(_example_a(), 0, 0.9))
  augmented_c.MergeFrom(_num_neighbors_example(1))
  return augmented_c


def _augmented_c_undirected_two_nbrs():
  """Returns an augmented `tf.train.Example` instance for node C."""
  augmented_c = _example_c()
  augmented_c.MergeFrom(_node_as_neighbor(_example_b(), 0, 1.0))
  augmented_c.MergeFrom(_node_as_neighbor(_example_a(), 1, 0.9))
  augmented_c.MergeFrom(_num_neighbors_example(2))
  return augmented_c


class PackNbrsTest(absltest.TestCase):

  def setUp(self):
    super(PackNbrsTest, self).setUp()
    # Write graph edges (as a TSV file).
    self._graph_path = self._create_tmp_file('graph.tsv')
    graph_utils.write_tsv_graph(self._graph_path, _GRAPH)
    # Write labeled training Examples.
    self._training_examples_path = self._create_tmp_file('train_data.tfr')
    _write_examples(self._training_examples_path, [_example_a(), _example_c()])
    # Write unlabeled neighbor Examples.
    self._neighbor_examples_path = self._create_tmp_file('neighbor_data.tfr')
    _write_examples(self._neighbor_examples_path, [_example_b()])
    # Create output file
    self._output_nsl_training_data_path = self._create_tmp_file(
        'nsl_train_data.tfr')

  def _create_tmp_file(self, filename):
    return self.create_tempfile(filename).full_path

  def testDirectedGraphUnlimitedNbrsNoNeighborExamples(self):
    """Tests pack_nbrs() with an empty second argument (neighbor examples).

    In this case, the edge A-->B is dangling because there will be no Example
    named "B" in the input.
    """
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        '',
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=False)
    expected_nsl_train_data = {
        # Node A has only one neighbor, namely C.
        'A': _augmented_a_directed_one_nbr(),
        # C has no neighbors in the directed case.
        'C': _augmented_c_directed()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)

  def testUndirectedGraphUnlimitedNbrsNoNeighborExamples(self):
    """Tests pack_nbrs() with an empty second argument (neighbor examples).

    In this case, the edge A-->B is dangling because there will be no Example
    named "B" in the input.
    """
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        '',
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=True)
    expected_nsl_train_data = {
        # Node A has only one neighbor, namely C.
        'A': _augmented_a_directed_one_nbr(),
        # C's only neighbor in the undirected case is A.
        'C': _augmented_c_undirected_one_nbr_a()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)

  def testDirectedGraphUnlimitedNbrs(self):
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        self._neighbor_examples_path,
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=False)
    expected_nsl_train_data = {
        'A': _augmented_a_directed_two_nbrs(),
        'C': _augmented_c_directed()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)

  def testDirectedGraphLimitedNbrs(self):
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        self._neighbor_examples_path,
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=False,
        max_nbrs=1)
    expected_nsl_train_data = {
        'A': _augmented_a_directed_one_nbr(),
        'C': _augmented_c_directed()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)

  def testUndirectedGraphUnlimitedNbrs(self):
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        self._neighbor_examples_path,
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=True)
    expected_nsl_train_data = {
        'A': _augmented_a_undirected_two_nbrs(),
        'C': _augmented_c_undirected_two_nbrs()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)

  def testUndirectedGraphLimitedNbrs(self):
    pack_nbrs_lib.pack_nbrs(
        self._training_examples_path,
        self._neighbor_examples_path,
        self._graph_path,
        self._output_nsl_training_data_path,
        add_undirected_edges=True,
        max_nbrs=1)
    expected_nsl_train_data = {
        'A': _augmented_a_undirected_one_nbr(),
        'C': _augmented_c_undirected_one_nbr_b()
    }
    actual_nsl_train_data = _read_tfrecord_examples(
        self._output_nsl_training_data_path)
    self.assertDictEqual(actual_nsl_train_data, expected_nsl_train_data)


if __name__ == '__main__':
  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
