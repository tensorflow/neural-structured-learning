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
"""Tests for neural_structured_learning.tools.build_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cmath

from absl.testing import absltest
from neural_structured_learning.tools import build_graph as build_graph_lib
from neural_structured_learning.tools import graph_utils
import six
import tensorflow as tf

from google.protobuf import text_format


# These embeddings in R^3 are chosen so that any pair of them has a cosine
# similaritt of 0.5.
r3_embeddings = [
    """
      features {
        feature {
          key: "id"
          value: { bytes_list { value: [ "A" ] } }
        }
        feature {
          key: "embedding"
          value: { float_list { value: [ 1, 1, 0 ] } }
        }
      }
    """, """
      features {
        feature {
          key: "id"
          value: { bytes_list { value: [ "B" ] } }
        }
        feature {
          key: "embedding"
          value: { float_list { value: [ 1, 0, 1] } }
        }
      }
    """, """
      features {
        feature {
          key: "id"
          value: { bytes_list { value: [ "C" ] } }
        }
        feature {
          key: "embedding"
          value: { float_list { value: [ 0, 1, 1] } }
        }
      }
    """
]


def write_embeddings(embeddings, embedding_output_path):
  """Writes the given 'embeddings' to a TFRecord file at the given path."""
  with tf.io.TFRecordWriter(embedding_output_path) as writer:
    for example_str in embeddings:
      example = text_format.Parse(example_str, tf.train.Example())
      writer.write(example.SerializeToString())


class BuildGraphTest(absltest.TestCase):
  # These embedding vectors are chosen so that the cosine of the angle between
  # each pair of them is 0.5.

  def _create_embedding_file(self):
    return self.create_tempfile('embeddings.tfr').full_path

  def _create_graph_file(self):
    return self.create_tempfile('graph.tsv').full_path

  def testBuildGraphInvalidLshBitsValue(self):
    with self.assertRaises(ValueError):
      build_graph_lib.build_graph([], None, lsh_bits=-1)

  def testBuildGraphInvalidLshRoundsValue(self):
    with self.assertRaises(ValueError):
      build_graph_lib.build_graph([], None, lsh_bits=1, lsh_rounds=0)

  def testBuildGraphNoThresholdingNoLSH(self):
    """All edges whose weight is greater than 0 are retained."""
    embeddings = r3_embeddings
    embedding_path = self._create_embedding_file()
    write_embeddings(embeddings, embedding_path)
    graph_path = self._create_graph_file()
    build_graph_lib.build_graph([embedding_path],
                                graph_path,
                                similarity_threshold=0)
    g_actual = graph_utils.read_tsv_graph(graph_path)
    self.assertDictEqual(
        g_actual, {
            'A': {
                'B': 0.5,
                'C': 0.5
            },
            'B': {
                'A': 0.5,
                'C': 0.5
            },
            'C': {
                'A': 0.5,
                'B': 0.5
            }
        })

  def testBuildGraphWithThresholdingNoLSH(self):
    """Edges below the similarity threshold are not part of the graph."""
    embeddings = r3_embeddings
    embedding_path = self._create_embedding_file()
    write_embeddings(embeddings, embedding_path)
    graph_path = self._create_graph_file()
    build_graph_lib.build_graph([embedding_path],
                                graph_path,
                                similarity_threshold=0.51)
    g_actual = graph_utils.read_tsv_graph(graph_path)
    self.assertDictEqual(g_actual, {})

  def _build_test_embeddings(self, num_points):
    """Construct `num_points` 2D points arranged equiangularly about the origin.

    Use a magnitude multiplier of 1.1 to test that the cosine similarity
    function is ignoring the magnitutes of the vectors (i.e., is normalizing
    the vectors before taking their dot product).

    Args:
      num_points: the number of points to create. The first is `(1.0, 0.0)`.

    Returns:
      A pair containing the resulting embeddings and the cosine similarity
      between adjacent points.
    """
    rotation = cmath.rect(1.1, 2 * cmath.pi / num_points)
    vector = 1.0 + 0.0j
    embeddings = []
    for node_id in range(num_points):
      embedding = tf.train.Example()
      embedding.features.feature['id'].bytes_list.value.append(
          'id_{}'.format(node_id).encode('utf8'))
      values = embedding.features.feature['embedding'].float_list.value
      values.append(vector.real)
      values.append(vector.imag)
      embeddings.append(text_format.MessageToString(embedding))
      vector = vector * rotation
    # Cosine similarity between adjacent points = 0.951057 for 20 points.
    adjacent_similarity = round(rotation.real / abs(rotation), 6)
    return (embeddings, adjacent_similarity)

  def testBuildGraphWithThresholdWithLSHInsufficientLSHRounds(self):
    """Tests that some edges are lost with insufficient LSH rounds."""
    # Construct the embeddings and write them to a file.
    num_points = 20
    (embeddings, _) = self._build_test_embeddings(num_points)
    embedding_path = self._create_embedding_file()
    write_embeddings(embeddings, embedding_path)

    # Build the graph, and read the results into a dictionary.
    graph_path = self._create_graph_file()
    build_graph_lib.build_graph([embedding_path],
                                graph_path,
                                similarity_threshold=0.9,
                                lsh_bits=2,
                                lsh_rounds=1,
                                random_seed=12345)
    g_actual = graph_utils.read_tsv_graph(graph_path)

    # Check that the graph contains fewer than 2 * N edges
    actual_edge_cnt = 0
    for (unused_src_id, tgt_dict) in six.iteritems(g_actual):
      actual_edge_cnt += len(tgt_dict)
    self.assertEqual(actual_edge_cnt, 2 * len(embeddings) - 8,
                     'Expected some edges not to have been found.')

  def testBuildGraphWithThresholdWithLSHSufficientLSHRounds(self):
    """Tests the case where we use (multiple rounds of) LSH bucketing."""
    # Construct the embeddings and write them to a file.
    num_points = 20
    (embeddings, adjacent_similarity) = self._build_test_embeddings(num_points)
    embedding_path = self._create_embedding_file()
    write_embeddings(embeddings, embedding_path)

    # Build the graph, and read the results into a dictionary.
    graph_path = self._create_graph_file()
    build_graph_lib.build_graph([embedding_path],
                                graph_path,
                                similarity_threshold=0.9,
                                lsh_bits=2,
                                lsh_rounds=4,
                                random_seed=12345)
    g_actual = graph_utils.read_tsv_graph(graph_path)

    # Constuct the expected graph: each point should be a neighbor of the
    # point before it and the point after it in the 'embeddings' sequence.
    # That's because the cosine similarity of adjacent points is ~0.951057,
    # while between every other point it is ~0.809017 (which is below the
    # similarity threshold of 0.9).
    g_expected = {}
    for node_id in range(num_points):
      t_dict = g_expected.setdefault('id_{}'.format(node_id), {})
      t_dict['id_{}'.format((node_id - 1) % num_points)] = adjacent_similarity
      t_dict['id_{}'.format((node_id + 1) % num_points)] = adjacent_similarity
    self.assertDictEqual(g_actual, g_expected)


if __name__ == '__main__':
  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
