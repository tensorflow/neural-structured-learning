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
"""This file splits the dataset into multiple shards for easier processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

flags.DEFINE_string('data', '', 'path to data')
flags.DEFINE_string('output', '', 'path to output dir')
# flags.DEFINE_integer('shards', 20, 'number of shards to split into')
flags.DEFINE_integer('lines_per_file', 1000000, 'lines per file')

FLAGS = flags.FLAGS


def main(argv):
  del argv
  if not tf.gfile.Exists(FLAGS.output):
    tf.gfile.MkDir(FLAGS.output)
  with open(FLAGS.data, 'r') as f:
    nlines = 0
    nshards = 0
    outf = open(FLAGS.output + '/' + 'shard%d.txt' % nshards, 'w+')
    for line in f:
      outf.write(line.strip() + '\n')
      nlines += 1
      if nlines % FLAGS.lines_per_file == 0:
        logging.info('Shard %d done', nshards)
        outf.close()
        nshards += 1
        outf = open(FLAGS.output + '/' + 'shard%d.txt' % nshards, 'w+')
    outf.close()
    logging.info('Wrote to %d shards', nshards)

if __name__ == '__main__':
  app.run(main)
