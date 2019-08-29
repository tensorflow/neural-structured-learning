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
"""Tests for neural_structured_learning.tools.graph_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl.testing import absltest
from neural_structured_learning.tools import graph_utils

GRAPH = {'A': {'B': 0.5, 'C': 0.9}, 'B': {'A': 0.4, 'C': 1.0}}


class GraphUtilsTest(absltest.TestCase):

  def testAddEdge(self):
    graph = {}
    graph_utils.add_edge(graph, ['A', 'B', '0.5'])
    graph_utils.add_edge(graph, ['A', 'C', 0.7])  # Tests that the edge
    graph_utils.add_edge(graph, ['A', 'C', 0.9])  # ...with maximal weight
    graph_utils.add_edge(graph, ['A', 'C', 0.8])  # ...is used.
    graph_utils.add_edge(graph, ('B', 'A', '0.4'))
    graph_utils.add_edge(graph, ('B', 'C'))  # Tests default weight
    self.assertDictEqual(graph, GRAPH)

  def testAddUndirectedEdges(self):
    g_actual = copy.deepcopy(GRAPH)
    graph_utils.add_undirected_edges(g_actual)
    self.assertDictEqual(
        g_actual, {
            'A': {
                'B': 0.5,
                'C': 0.9
            },
            'B': {
                'A': 0.5,  # Note, changed from 0.4 to 0.5
                'C': 1.0
            },
            'C': {
                'A': 0.9,  # Added
                'B': 1.0   # Added
            }
        })

  def testReadAndWriteTsvGraph(self):
    path = self.create_tempfile('graph.tsv').full_path
    graph_utils.write_tsv_graph(path, GRAPH)
    read_graph = graph_utils.read_tsv_graph(path)
    self.assertDictEqual(read_graph, GRAPH)


if __name__ == '__main__':
  absltest.main()
