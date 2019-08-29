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
"""Process ClueWeb data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

flags.DEFINE_string('data', '', 'path to data')
flags.DEFINE_string('output', '', 'path to output folder')
flags.DEFINE_integer('min_pair_freq', 20, 'min entity pair frequency')
flags.DEFINE_integer('subsample_nrels', 20,
                     'number of relations to subsample per entity pair')

FLAGS = flags.FLAGS


def process_files(file_names):
  """Process to read entity pairs and filter by frequency."""
  nproc = 0
  entity_pairs = defaultdict(int)
  sentences = set([])
  for fname in file_names:
    logging.info('Reading file %s', fname)
    with open(fname, 'r') as f:
      for line in f:
        nproc += 1
        if nproc % 500000 == 0:
          logging.info(nproc)
        line = line.strip().split('\t')
        if len(line[0].strip()) > 200:
          continue
        sentences.add(line[0].strip())
        ents = line[-2].strip().split(',')
        for e1 in ents:
          for e2 in ents:
            if e1 == e2:
              continue
            entity_pairs[e1 + ',' + e2] += 1
    # break

  logging.info('Total entity pairs %d', len(entity_pairs))
  sentences = list(sentences)
  # write sentences
  logging.info('Writing sentences data, total sentences %d', len(sentences))
  with open(FLAGS.output + '/sentences.txt', 'w+') as f:
    for sent in sentences:
      f.write(sent + '\n')

  logging.info('Storing entity pair data ')
  sentences = {sent: i for i, sent in enumerate(sentences)}
  pair_data = {}
  nproc = 0
  for fname in file_names:
    logging.info('Reading file %s', fname)
    with open(fname, 'r') as f:
      for line in f:
        nproc += 1
        if nproc % 500000 == 0:
          logging.info(nproc)
        line = line.strip().split('\t')
        if len(line[0].strip()) > 200:
          continue
        ents = line[-2].strip().split(',')
        sid = sentences[line[0].strip()]
        for e1 in ents:
          for e2 in ents:
            if e1 == e2:
              continue
            if e1 not in pair_data:
              pair_data[e1] = {}
            if e2 not in pair_data[e1]:
              pair_data[e1][e2] = []
            pair_data[e1][e2].append(sid)
    # break

  # subsample data
  sentences = {i: sent for sent, i in sentences.iteritems()}
  logging.info('Writing subsampled relation data')
  outf = open(FLAGS.output + '/relation_pairs.txt', 'w+')
  sent_outf = open(FLAGS.output + '/sentences_subset.txt', 'w+')
  written = set([])
  for e1 in pair_data:
    for e2 in pair_data[e1]:
      if len(pair_data[e1][e2]) < FLAGS.min_pair_freq:
        continue
      sents = np.random.choice(pair_data[e1][e2], size=FLAGS.subsample_nrels,
                               replace=False)
      for sid in sents:
        outf.write(str(sid) + '\t' + e1 + '\t' + e2 + '\n')
        if sid not in written:
          sent_outf.write(str(sid) + '\t' + sentences[sid] +'\n')
          written.add(sid)
  outf.close()
  sent_outf.close()
  logging.info('Subsampled senteces: %d', len(written))


def main(argv):
  del argv
  if not tf.gfile.Exists(FLAGS.output):
    tf.gfile.MkDir(FLAGS.output)
  file_list = tf.gfile.ListDirectory(FLAGS.data)
  file_list = [os.path.join(FLAGS.data, f) for f in file_list]
  process_files(file_list)
  # logging.info('Done, final relations/lines in output: %d', result)


if __name__ == '__main__':
  app.run(main)
