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
"""Tests for neural_structured_learning.tools.graph_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from neural_structured_learning.tools import graph_builder
from neural_structured_learning.tools import graph_utils
import tensorflow as tf

from google.protobuf import text_format


class BuildGraphLibTest(absltest.TestCase):

  def _create_embedding_file(self):
    return self.create_tempfile('embeddings.tfr').full_path

  def _create_graph_file(self):
    return self.create_tempfile('graph.tsv').full_path

  def _write_embeddings(self, embedding_output_path):
    example1 = """
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
              """
    example2 = """
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
              """
    example3 = """
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
    # The embedding vectors above are chosen so that the cosine of the angle
    # between each pair of them is 0.5.
    with tf.io.TFRecordWriter(embedding_output_path) as writer:
      for example_str in [example1, example2, example3]:
        example = text_format.Parse(example_str, tf.train.Example())
        writer.write(example.SerializeToString())

  def testGraphBuildingNoThresholding(self):
    """All edges whose weight is greater than 0 are retained."""
    embedding_path = self._create_embedding_file()
    self._write_embeddings(embedding_path)
    graph_path = self._create_graph_file()
    graph_builder.build_graph([embedding_path],
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

  def testGraphBuildingWithThresholding(self):
    """Edges below the similarity threshold are not part of the graph."""
    embedding_path = self._create_embedding_file()
    self._write_embeddings(embedding_path)
    graph_path = self._create_graph_file()
    graph_builder.build_graph([embedding_path],
                              graph_path,
                              similarity_threshold=0.51)
    g_actual = graph_utils.read_tsv_graph(graph_path)
    self.assertDictEqual(g_actual, {})


if __name__ == '__main__':
  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
