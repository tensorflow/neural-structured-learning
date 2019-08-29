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
"""Extract entities from FB15k and write to a file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags


flags.DEFINE_string('train_data', '', 'path to data')
flags.DEFINE_string('dev_data', '', 'path to data')
flags.DEFINE_string('test_data', '', 'path to data')
flags.DEFINE_string('output', 'entities.txt', 'path to output')

FLAGS = flags.FLAGS


def main(argv):
  del argv
  entities = set([])
  for fname in [FLAGS.train_data, FLAGS.dev_data, FLAGS.test_data]:
    with open(fname, 'r') as f:
      for line in f:
        e1, _, e2 = line.strip().split('\t')
        entities.add(e1)
        entities.add(e2)
  entities = list(entities)
  print('Found %d etities' % len(entities))
  with open(FLAGS.output, 'w+') as f:
    for ent in entities:
      f.write(ent + '\n')


if __name__ == '__main__':
  app.run(main)
