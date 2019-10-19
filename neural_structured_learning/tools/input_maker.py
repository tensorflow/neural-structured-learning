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
r"""Tool that prepares input for graph-based Neural Structured Learning.

This is a wrapper around the `nsl.tools.pack_nbrs` API. See its documentation
for more details.

USAGE:

`python input_maker.py` [*flags*] *labeled.tfr unlabeled.tfr graph.tsv
output.tfr*

For details about this program's flags, run `python input_maker.py --help`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from neural_structured_learning.tools import input_maker_lib
import tensorflow as tf


def _main(argv):
  """Main function for running the input_maker program."""
  flag = flags.FLAGS
  flag.showprefixforinfo = False
  # Check that the correct number of arguments have been provided.
  if len(argv) != 5:
    raise app.UsageError('Invalid number of arguments; expected 4, got %d' %
                         (len(argv) - 1))

  input_maker_lib.pack_nbrs(argv[1], argv[2], argv[3], argv[4],
                            flag.add_undirected_edges, flag.max_nbrs,
                            flag.id_feature_name)


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
