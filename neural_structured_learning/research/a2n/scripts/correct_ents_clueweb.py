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
flags.DEFINE_string('sentences_subset', '', 'path to sent subset')
flags.DEFINE_string('output', '', 'path to output folder')
flags.DEFINE_integer('min_pair_freq', 20, 'min entity pair frequency')
flags.DEFINE_integer('subsample_nrels', 20,
                     'number of relations to subsample per entity pair')

FLAGS = flags.FLAGS


def process_files(file_names):
  """Process to read entity pairs and filter by frequency."""
  logging.info('Reading sentences data')
  sentence_ids = set([])
  sentences = {}
  nsents = 0
  with open(FLAGS.sentences_subset, 'r') as f:
    for line in f:
      sid, sent = line.strip().split('\t')
      sid = int(sid)
      sentences[sid] = sent.strip()
      sentence_ids.add(sid)
      nsents += 1
  logging.info('Read %d sentences', nsents)
  logging.info('Storing entity pair data ')
  sentences = {sent: sid for sid, sent in sentences.iteritems()}
  pair_data = {}
  nproc = 0
  ndata = 0
  for fname in file_names:
    logging.info('Reading file %s', fname)
    with open(fname, 'r') as f:
      for line in f:
        nproc += 1
        if nproc % 500000 == 0:
          logging.info('Read %d, Stored Pairs %d', nproc, ndata)
        line = line.strip().split('\t')
        if len(line[0].strip()) > 200:
          continue
        ents = line[-2].strip().split(',')
        txt = line[0].strip()
        if txt in sentences:
          sid = sentences[txt]
          for e1 in ents:
            for e2 in ents:
              if e1 == e2:
                continue
              if e1 not in pair_data:
                pair_data[e1] = {}
              if e2 not in pair_data[e1]:
                pair_data[e1][e2] = []
              pair_data[e1][e2].append(sid)
              ndata += 1
    if nproc > 50000000:
      break

  # subsample data
  # sentences = {i: sent for sent, i in sentences.iteritems()}
  logging.info('Writing subsampled relation data')
  outf = open(FLAGS.output + '/relation_pairs.txt', 'w+')
  nout = 0
  for e1 in pair_data:
    for e2 in pair_data[e1]:
      sents = list(set(pair_data[e1][e2]))
      if len(sents) < FLAGS.min_pair_freq:
        continue
      sents = np.random.choice(sents, size=FLAGS.subsample_nrels,
                               replace=False)
      for sid in sents:
        outf.write(str(sid) + '\t' + e1 + '\t' + e2 + '\n')
        nout += 1
        # if sid not in written:
        #   sent_outf.write(str(sid) + '\t' + sentences[sid] +'\n')
        #   written.add(sid)
  outf.close()
  logging.info('Final lines: %d', nout)


def main(argv):
  del argv
  if not tf.gfile.Exists(FLAGS.output):
    tf.gfile.MkDir(FLAGS.output)
  file_list = tf.gfile.ListDirectory(FLAGS.data)
  file_list = [os.path.join(FLAGS.data, f) for f in file_list]
  process_files(file_list)


if __name__ == '__main__':
  app.run(main)
