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
"""Utility functions for manipulating (weighted) graphs.

The functions in this module assume that weighted graphs are represented by
nested dictionaries, where the outer dictionary maps each edge source ID to
an inner dictionary that maps each edge target ID to that edge's weight. So
for example, the graph containing the edges:

```
A -- 0.5 --> B
A -- 0.9 --> C
B -- 0.4 --> A
B -- 1.0 --> C
C -- 0.8 --> D
```

would be represented by the dictionary:

```
{ "A": { "B": 0.5, "C": 0.9 },
  "B": { "A": 0.4, "C": 1.0 },
  "C": { "D": 0.8 }
}
```

In the documention, we say a graph is represented by a `dict`:
source_id -> (target_id -> weight).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging
import six


def add_edge(graph, edge):
  """Adds an edge to a given graph.

  If an edge between the two nodes already exists, the one with the largest
  weight is retained.

  Args:
    graph: A `dict`: source_id -> (target_id -> weight) to be augmented.
    edge: A `list` (or `tuple`) of the form `[source, target, weight]`, where
      `source` and `target` are strings, and `weight` is a numeric value of
      type `string` or `float`. The 'weight' component is optional; if not
      supplied, it defaults to 1.0.

  Returns:
    `True` if and only if a new edge was added to `graph`.
  """
  source = edge[0]
  target = edge[1]
  weight = float(edge[2]) if len(edge) > 2 else 1.0
  t_dict = graph.setdefault(source, {})
  is_new_edge = target not in t_dict
  if is_new_edge or weight > t_dict[target]:
    t_dict[target] = weight
  return is_new_edge


def add_undirected_edges(graph):
  """Makes all edges of the given `graph` bi-directional.

  Updates `graph` to include a reversed version of each of its edges. Multiple
  edges between the same source and target node IDs are combined by picking the
  edge with the largest weight.

  Args:
    graph: A `dict`: source -> (target -> weight) as returned by the
      `read_tsv_graph` function.

  Returns:
    `None`. Instead, this function has a side-effect on the `graph` argument.
  """
  def all_graph_edges():
    # Make a copy of all source IDs to avoid concurrent iteration failure.
    sources = list(graph.keys())
    for source in sources:
      # Make a copy of source's out-edges to avoid concurrent iteration failure.
      out_edges = dict(graph[source])
      for target, weight in six.iteritems(out_edges):
        yield (source, target, weight)

  start_time = time.time()
  logging.info('Making all edges bi-directional...')
  for s, t, w in all_graph_edges():
    add_edge(graph, [t, s, w])
  logging.info('Done (%.2f seconds). Total graph nodes: %d',
               (time.time() - start_time), len(graph))


def read_tsv_graph(filename):
  r"""Reads the file `filename` containing graph edges in TSV format.

  Args:
    filename: Name of a TSV file specifying the edges of a graph. Each line of
      the input file should be the specification of a single graph edge in the
      form `source\<TAB\>target[\<TAB\>weight]`. If supplied, `weight` should
      be a floating point number; if not supplied, it defaults to 1.0. Multiple
      edges between the same source and target node IDs are combined by picking
      the edge with the largest weight.

  Returns:
    A graph represented as a `dict`: source -> (target -> weight).
  """
  start_time = time.time()
  logging.info('Reading graph file: %s...', filename)
  graph = {}
  edge_cnt = 0
  with open(filename, 'rU') as f:
    for tsv_line in f:
      edge = tsv_line.rstrip('\n').split('\t')
      if add_edge(graph, edge): edge_cnt += 1
  logging.info('Done reading %d edges from: %s (%.2f seconds).', edge_cnt,
               filename, (time.time() - start_time))
  return graph


def write_tsv_graph(filename, graph):
  """Writes the given `graph` to the file `filename` in TSV format.

  Args:
    filename: Name of the file to which TSV output is written. The TSV lines are
      written in the same form as the input expected by `read_tsv_graph()`.
    graph: A `dict` source_id -> (target_id -> weight) representing the graph.

  Returns:
    `None`. Instead, this has the side-effect or writing output to a file.
  """
  start_time = time.time()
  logging.info('Writing graph to TSV file: %s', filename)
  with open(filename, 'w') as f:
    for s, t_dict in six.iteritems(graph):
      for t, w in six.iteritems(t_dict):
        f.write('%s\t%s\t%f\n' % (s, t, w))
  logging.info('Done writing graph to TSV file: %s (%.2f seconds).',
               filename, (time.time() - start_time))
