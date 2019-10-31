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
"""Generate a random graph for testing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("output", None, "Path to output file")
flags.DEFINE_integer("num_entities", 20, "Number of entities in graph")
flags.DEFINE_integer("num_relations", 5, "Number of relation types in graph")
flags.DEFINE_integer("num_edges", 20, "Number of edges in graph")

flags.mark_flag_as_required("output")


def generate_graph():
  with open(FLAGS.output, "w+") as f:
    for _ in range(FLAGS.num_edges):
      s = np.random.randint(FLAGS.num_entities)
      r = np.random.randint(FLAGS.num_relations)
      t = np.random.randint(FLAGS.num_entities)
      while t != s:
        t = np.random.randint(FLAGS.num_entities)
      f.write("%d\t%d\t%d\n" % (s, r, t))


def main(argv):
  del argv
  generate_graph()


if __name__ == "__main__":
  app.run(main)
