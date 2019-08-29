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
"""Post-process ClueWeb data with special focus on handling entities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

flags.DEFINE_string('data', '', 'path to data')
flags.DEFINE_string('output', '', 'path to output folder')
flags.DEFINE_string('entities', '', 'path to entities')
flags.DEFINE_integer('n_jobs', 20, 'number of parallel jobs')

FLAGS = flags.FLAGS
entities_set = set([])


def process_file(fname):
  """Process each file."""
  name = fname.split('/')[-1]
  outf = open(FLAGS.output + '/proc_' + name, 'w+')
  nout = 0
  nproc = 0
  with open(fname, 'r') as f:
    for line in f:
      nproc += 1
      if nproc % 100000 == 0:
        logging.info(nproc)
      line = line.strip().split('\\t')
      text = ' '.join(line[:-2])
      if len(text.strip()) < 6:
        continue
      ents = line[-2].strip().split(',')
      mentions = line[-1].strip().split(',')
      out_ents, out_mentions = [], []
      for ent, ment in zip(ents, mentions):
        if ent in entities_set:
          out_ents.append(ent)
          out_mentions.append(ment)
      if len(out_ents) < 2:
        continue
      ents = ','.join(out_ents)
      ments = ','.join(out_mentions)
      outf.write(text + '\t' + ents + '\t' + ments + '\n')
      nout += 1
  outf.close()
  return nout


def main(argv):
  del argv
  global entities_set
  with open(FLAGS.entities, 'r') as f:
    for line in f:
      entities_set.add(line.strip())
  logging.info('Read %d entities', len(entities_set))
  if not tf.gfile.Exists(FLAGS.output):
    tf.gfile.MkDir(FLAGS.output)
  result = process_file(FLAGS.data)
  logging.info('Done, final docs: %s', result)


if __name__ == '__main__':
  app.run(main)
