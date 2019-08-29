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
"""Generate senctence vocab and relations file for ClueWeb data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
# import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

flags.DEFINE_string('data', '', 'path to data')
flags.DEFINE_string('output', '', 'path to output folder')
flags.DEFINE_string('max_text_len', 200, 'maximum length of text')
flags.DEFINE_integer('subsample_per_pair', 10,
                     'number of text edges to sample per pair')

FLAGS = flags.FLAGS


def process_file(fname):
  """Process to read entity pairs and filter by frequency."""
  nproc = 0
  logging.info('Reading file %s', fname)
  sentences = set([])
  with open(fname, 'r') as f:
    for line in f:
      nproc += 1
      if nproc % 100000 == 0:
        logging.info(nproc)
      line = line.strip().split('\t')
      text = line[0].strip()
      if len(text) > FLAGS.max_text_len:
        continue
      sentences.add(text)

  logging.info('Total sentences %d', len(sentences))
  sentences = list(sentences)
  with open(FLAGS.output + '/sentences.txt', 'w+') as f:
    for i in range(len(sentences)):
      sentence = sentences[i]
      f.write(sentence + '\n')

  sentences = {sent: i for i, sent in enumerate(sentences)}

  nproc = 0
  logging.info('Reading pair statistics')
  pair_data = {}
  with open(fname, 'r') as f:
    for line in f:
      nproc += 1
      if nproc % 100000 == 0:
        logging.info(nproc)
      line = line.strip().split('\t')
      text = line[0].strip()
      if len(text) > FLAGS.max_text_len:
        continue
      sid = sentences[text]
      e1 = line[1].strip()
      e2 = line[2].strip()
      if e1 not in pair_data:
        pair_data[e1] = {}
      if e2 not in pair_data[e1]:
        pair_data[e1][e2] = []
      pair_data[e1][e2].append(sid)

  # subsample data
  logging.info('Writing subsampled relation data')
  outf = open(FLAGS.output + '/relation_pairs.txt', 'w+')
  for e1 in pair_data:
    for e2 in pair_data[e1]:
      sents = np.random.choice(pair_data[e1][e2], size=FLAGS.subsample_per_pair,
                               replace=False)
      for sid in sents:
        outf.write(str(sid) + '\t' + e1 + '\t' + e1 + '\n')
  outf.close()


def main(argv):
  del argv
  if not tf.gfile.Exists(FLAGS.output):
    tf.gfile.MkDir(FLAGS.output)
  process_file(FLAGS.data)


if __name__ == '__main__':
  app.run(main)
