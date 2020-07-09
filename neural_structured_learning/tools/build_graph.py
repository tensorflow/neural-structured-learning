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

r"""Program & library to build a graph from dense features (embeddings)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import time

from absl import app
from absl import flags
from absl import logging
import neural_structured_learning.configs as nsl_configs
import numpy as np
import six
import tensorflow as tf

# Norm used if the computed norm of an embedding is less than this value.
# This value is the same as the default for tf.math.l2_normalize.
_MIN_NORM = np.float64(1e-6)


def _read_tfrecord_examples(filenames, id_feature_name, embedding_feature_name):
  """Reads and returns the embeddings stored in the Examples in `filename`.

  Args:
    filenames: A list of names of TFRecord files containing `tf.train.Example`
      objects.
    id_feature_name: Name of the feature that identifies the Example's ID.
    embedding_feature_name: Name of the feature that identifies the Example's
        embedding.

  Returns:
    A dict mapping each instance ID to its L2-normalized embedding, represented
    by a 1-D numpy.ndarray. The ID is expected to be contained in the singleton
    bytes_list feature named by 'id_feature_name', and the embedding is
    expected to be contained in the float_list feature named by
    'embedding_feature_name'.
  """
  def parse_tf_record_examples(filename):
    """Generator that returns the tensorflow.Examples in `filename`.

    Args:
      filename: Name of the TFRecord file containing tensorflow.Examples.

    Yields:
      The tensorflow.Examples contained in the file.
    """
    for raw_record in tf.data.TFRecordDataset([filename]):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      yield example

  def l2_normalize(v):
    """Returns the L2-norm of the vector `v`.

    Args:
      v: A 1-D vector (either a list or numpy array).

    Returns:
      The L2-normalized version of `v`. The result will have an L2-norm of 1.0.
    """
    l2_norm = np.linalg.norm(v)
    return v / max(l2_norm, _MIN_NORM)

  embeddings = {}
  for filename in filenames:
    start_time = time.time()
    logging.info('Reading tf.train.Examples from TFRecord file: %s...',
                 filename)
    for tf_example in parse_tf_record_examples(filename):
      f_map = tf_example.features.feature
      if id_feature_name not in f_map:
        logging.error('No feature named "%s" found in input Example: %s',
                      id_feature_name, tf_example.ShortDebugString())
        continue
      ex_id = f_map[id_feature_name].bytes_list.value[0].decode('utf-8')
      if embedding_feature_name not in f_map:
        logging.error('No feature named "%s" found in input with ID "%s"',
                      embedding_feature_name, ex_id)
        continue
      embedding_list = f_map[embedding_feature_name].float_list.value
      embeddings[ex_id] = l2_normalize(embedding_list)
    logging.info('Done reading %d tf.train.Examples from: %s (%.2f seconds).',
                 len(embeddings), filename, time.time() - start_time)
  return embeddings


class GraphBuilder(object):
  """Computes the similarity graph from a set of (dense) embeddings."""

  def __init__(self, graph_builder_config):
    """Initializes this GraphBuilder from the given configuration instance.

    Args:
      graph_builder_config: A `nsl.configs.GraphBuilderConfig` instance.

    Raises:
      ValueError: If `lsh_splits < 0` or if `lsh_splits > 0 and lsh_rounds < 1`.
    """
    self.config = graph_builder_config
    if self.config.lsh_splits < 0:
      raise ValueError('lsh_splits < 0')
    if self.config.lsh_splits > 0 and self.config.lsh_rounds < 1:
      raise ValueError('lsh_splits > 0 but lsh_rounds < 1')

    # Keep a set of previously written edges if it's possible we might
    # generate the same edge multiple times. This can happen only if both
    # 'lsh_splits > 0' and 'lsh_rounds > 1'. To save space, we pick a canonical
    # ordering (source < target) for each bi-directional edge. Note that we
    # do not need to store the edge weight as well because for any
    # (source, target) pair, the cosine similarity between them will never
    # change.
    self.edge_set = None
    if self.config.lsh_splits > 0 and self.config.lsh_rounds > 1:
      self.edge_set = set()

  def _is_new_edge(self, src, tgt):
    """Returns `True` iff the edge `src` to `tgt` has not been generated yet."""
    canonical_edge = (src, tgt) if src < tgt else (tgt, src)
    # Remember set size before calling add() because add() returns None.
    # This way we don't have to hash 'canonical_edge' twice.
    set_size_before_add = len(self.edge_set)
    self.edge_set.add(canonical_edge)
    return len(self.edge_set) > set_size_before_add

  def _bucket(self, lsh_matrix, embedding):
    """Returns the bucket ID of the given `embedding` relative to `lsh_matrix`.

    Args:
      lsh_matrix: A random matrix of vectors for computing LSH buckets.
      embedding: A 1-D vector representing the dense embedding for a point.

    Returns:
      The bucket ID, a value in `[0, 2^n)`, where `n = self.config.lsh_splits`.
      Bit `i` of the result (where bit 0 corresponds to the least significant
      bit) is 1 if and only if the dot product of row `i` of `lsh_matrix' and
      `embedding` is positive.
    """
    bucket = 0
    mask = 1
    for x in np.matmul(lsh_matrix, embedding):
      if x > 0.0: bucket = bucket | mask
      mask = mask << 1
    return bucket

  def _generate_lsh_buckets(self, embeddings):
    """Buckets the given `embeddings` according to `config.lsh_splits`.

    The embeddings can be bucketed into a total of at most `2^n` different
    buckets, where `n` is given by the value of `config.lsh_splits`. If `n` is
    not positive, then all of the given `embeddings` keys will be bucketed into
    bucket 0.

    Args:
      embeddings: A `dict`: node_id -> embedding.

    Returns:
      A dictionary mapping bucket IDs to sets of embedding IDs in each bucket.
      The bucket IDs are integers in the half-open interval `[0, 2^n)`, where
      `n = config.lsh_splits`.
    """
    if self.config.lsh_splits <= 0: return {0: set(embeddings.keys())}

    # Generate a random matrix of values in the range [-1.0, 1.0] to use
    # to create the LSH buckets.
    num_dims = next(iter(embeddings.values())).size
    lsh_matrix = np.random.rand(self.config.lsh_splits, num_dims) * 2 - 1

    # Add each embedding to its appropriate bucket
    bucket_map = {}
    for key, embedding in six.iteritems(embeddings):
      s = bucket_map.setdefault(self._bucket(lsh_matrix, embedding), set())
      s.add(key)
    return bucket_map

  def _generate_edges_for_bucket(self, bucket, embeddings):
    """Generates edges based on considering all node pairs in `bucket`.

    Args:
      bucket: A `set` of all node IDs in the same LSH bucket.
      embeddings: A `dict`: node_id -> embedding.

    Yields:
      A tuple (source, target, weight) denoting a (directed) edge from 'source'
      to 'target' with the given edge 'weight'.
    """
    for src, tgt in itertools.combinations(bucket, 2):
      weight = np.dot(embeddings[src], embeddings[tgt])
      if weight >= self.config.similarity_threshold:
        if self.edge_set is None or self._is_new_edge(src, tgt):
          yield (src, tgt, weight)

  def _generate_edges(self, embeddings):
    """Generates edges among pairs of the given `embeddings`.

    This function considers related pairs of nodes in `embeddings`,
    computes the cosine similarity between all such pairs, and yields any edge
    for which the cosine similarity is at least `self.similarity_threshold`.

    Args:
      embeddings: A `dict`: node_id -> embedding.

    Yields:
      A tuple (source, target, weight) denoting a (directed) edge from 'source'
      to 'target' with the given 'weight'.
    """
    for lsh_round in range(max(1, self.config.lsh_rounds)):
      start_time = time.time()
      edge_cnt = 0
      bucket_map = self._generate_lsh_buckets(embeddings)
      logging_prefix = 'LSH bucketing round {}'.format(lsh_round)
      logging.info('%s: created %d bucket(s) in %.2f seconds.', logging_prefix,
                   len(bucket_map),
                   time.time() - start_time)
      for bucket in bucket_map.values():
        for edge in self._generate_edges_for_bucket(bucket, embeddings):
          edge_cnt += 1
          if (edge_cnt % 1000000) == 0:
            logging.info(
                '%s: generated %d new bi-directional edges in %.2f seconds....',
                logging_prefix, edge_cnt,
                time.time() - start_time)
          yield edge
      logging.info(
          '%s completed: generated %d new bi-directional edges in %.2f seconds.',
          logging_prefix, edge_cnt,
          time.time() - start_time)

  def build(self, embedding_files, output_graph_path):
    """Reads embeddings and writes the similarity graph to `output_graph_path`.

    The parameters used to construct the graph are taken from the
    `nsl.configs.GraphBuilderConfig` passed to this class's constructor.

    Args:
      embedding_files: A list of names of TFRecord files containing
        `tf.train.Example` objects, which in turn contain dense embeddings.
      output_graph_path: Name of the file to which the output graph in TSV
        format should be written.
    """
    embeddings = _read_tfrecord_examples(embedding_files,
                                         self.config.id_feature_name,
                                         self.config.embedding_feature_name)
    start_time = time.time()
    logging.info('Building graph and writing edges to TSV file: %s',
                 output_graph_path)
    np.random.seed(self.config.random_seed)
    logging.info('Using random seed value: %s', self.config.random_seed)
    edge_cnt = 0
    with open(output_graph_path, 'w') as f:
      for (src, tgt, wt) in self._generate_edges(embeddings):
        f.write('%s\t%s\t%f\n' % (src, tgt, wt))
        f.write('%s\t%s\t%f\n' % (tgt, src, wt))
        edge_cnt += 1
      logging.info(
          'Wrote graph containing %d bi-directional edges (%.2f seconds).',
          edge_cnt, time.time() - start_time)


def build_graph_from_config(embedding_files, output_graph_path,
                            graph_builder_config):
  """Builds a graph based on dense embeddings and persists it in TSV format.

  This function reads input instances from one or more TFRecord files, each
  containing `tf.train.Example` protos. Each input example is expected to
  contain at least the following 2 features:

  *   `id`: A singleton `bytes_list` feature that identifies each example.
  *   `embedding`: A `float_list` feature that contains the (dense) embedding of
       each example.

  `id` and `embedding` are not necessarily the literal feature names; if your
  features have different names, you can specify them using the
  `graph_builder_config` fields named `id_feature_name` and
  `embedding_feature_name`, respectively.

  This function then uses the node embeddings to compute the edges of a graph.
  Graph construction is configured by the `graph_builder_config` instance. The
  general algorithm is to consider pairs of input examples (each with an ID and
  an associated dense embedding, as described above), and to generate an edge in
  the graph between those two examples if the cosine similarity between the two
  embeddings is at least `graph_builder_config.similarity_threshold`.

  Of course, the number of pairs of points is quadratic in the number of inputs,
  so comparing all pairs of points will take too long for large inputs. To
  address that problem, this implementation can be configured to use locality
  sensitive hashing (LSH) for better performance. The idea behind LSH is to
  randomly hash each input into one or more LSH "buckets" such that points in
  the same bucket are likely to be considered similar. In this way, we need to
  compare just the pairs of points within each bucket for similarity, which can
  lead to dramatically faster running times.

  The `lsh_splits` configuration attribute is used to control the maximum number
  of LSH buckets. In particular, if `lsh_splits` has the value `n`, then there
  can be at most `2^n` LSH buckets. Using a larger value for `lsh_splits` will
  (generally) result in a larger number of buckets, and therefore, smaller
  number of instances in each bucket that need to be compared to each other.
  As a result, increasing `lsh_splits` can lead to dramatically faster running
  times.

  The disadvantage to using too many LSH buckets, however, is that we won't
  create a graph edge between two instances that are highly similar if they
  happen to be randomly hashed into two different LSH buckets. To address
  that problem, the `lsh_rounds` parameter can be used to perform multiple
  rounds of the LSH bucketing process. Even if two similar instances may get
  hashed to different LSH buckets during the first round, they may get hashed
  into the same LSH bucket on a subsequent round. An edge is created in the
  output graph if two intances are hashed into the same bucket and deemed to
  be similar enough on *any* of the LSH rounds (i.e., the resulting graph is the
  *union* of the graph edges generated on each LSH round).

  To illustrate these concepts and how various `lsh_splits` and `lsh_rounds`
  values correlate with graph building running times, we performed multiple runs
  of the graph builder on a dataset containing 50,000 instances, each with a
  100-dimensional embedding. When `lsh_splits = 0`, the program has to compare
  each instance against every other instance, for a total of roughly 2.5B
  comparisons, which takes nearly half an hour to complete and generates a total
  of 35,313 graph edges (when `similarity_threshold = 0.9`). As `lsh_splits` is
  increased, we lose recall (i.e., fewer than 35,313 edges are generated), but
  the recall can then be improved by increasing `lsh_rounds`. This table shows
  the minimum `lsh_rounds` value required to achieve a recall of >= 99.7%
  (except for the `lsh_splits = 1` case), as well as the elapsed running time:

  ```none
  lsh_splits  lsh_rounds    Recall    Running time
      0          N/A        100.0%      27m 46s
      1           2          99.4%      24m 33s
      2           3          99.8%      15m 35s
      3           4          99.7%       9m 37.9s
      4           6          99.9%       7m 07.5s
      5           8          99.9%       4m 59.2s
      6           9          99.7%       3m 01.2s
      7          11          99.8%       2m 02.3s
      8          13          99.8%       1m 20.8s
      9          16          99.7%          58.5s
     10          18          99.7%          43.6s
  ```

  As the table illustrates, by increasing both `lsh_splits` and `lsh_rounds`,
  we can dramatically decrease the running time of the graph builder without
  sacrificing edge recall. We have found that a good rule of thumb is to set
  `lsh_splits >= ceiling(log_2(num_instances / 1000))`, so the expected LSH
  bucket size will be at most 1000. However, if your instances are clustered or
  you want an even faster run, you may want to use a larger `lsh_splits` value.
  Note, however, that when the similarity threshold is lower, recall rates are
  reduced more quickly the larger the value of `lsh_splits` is, so be careful
  not to set that parameter too high for smaller `similarity_threshold` values.

  The generated graph edges are written to the TSV file named by
  `output_graph_path`. Each output edge is represented by a TSV line with the
  following form:

  ```
  source_id<TAB>target_id<TAB>edge_weight
  ```

  All edges in the output will be symmetric (i.e., if edge `A--w-->B` exists in
  the output, then so will edge `B--w-->A`).

  Args:
    embedding_files: A list of names of TFRecord files containing
      `tf.train.Example` objects, which in turn contain dense embeddings.
    output_graph_path: Name of the file to which the output graph in TSV format
      should be written.
    graph_builder_config: A `nsl.configs.GraphBuilderConfig` specifying the
      graph building parameters.

  Raises:
    ValueError: If `lsh_splits < 0` or if `lsh_splits > 0 and lsh_rounds < 1`.
  """
  graph_builder = GraphBuilder(graph_builder_config)
  graph_builder.build(embedding_files, output_graph_path)


def build_graph(embedding_files,
                output_graph_path,
                similarity_threshold=0.8,
                id_feature_name='id',
                embedding_feature_name='embedding',
                lsh_splits=0,
                lsh_rounds=2,
                random_seed=None):
  """Like `nsl.tools.build_graph_from_config`, but with individual parameters.

  This API exists to maintain backward compatibility, but is deprecated in favor
  of using `nsl.tools.build_graph_from_config` instead.

  Args:
    embedding_files: A list of names of TFRecord files containing
      `tf.train.Example` objects, which in turn contain dense embeddings.
    output_graph_path: Name of the file to which the output graph in TSV format
      should be written.
    similarity_threshold: Threshold used to determine which edges to retain in
      the resulting graph.
    id_feature_name: The name of the feature in the input `tf.train.Example`
      objects representing the ID of examples.
    embedding_feature_name: The name of the feature in the input
      `tf.train.Example` objects representing the embedding of examples.
    lsh_splits: Determines the maximum number of LSH buckets into which input
      data points will be bucketed by the graph builder. See the
      `nsl.tools.build_graph_from_config` documentation for details.
    lsh_rounds: The number of rounds of LSH bucketing to perform when
      `lsh_splits > 0`. This is also the number of LSH buckets each point will
      be hashed into.
    random_seed: Value used to seed the random number generator used to perform
      randomized LSH bucketing of the inputs when `lsh_splits > 0`. By default,
      the generator will be initialized randomly, but setting this to any
      integer will initialize it deterministically.

  Raises:
      ValueError: If `lsh_splits < 0` or if `lsh_splits > 0 and lsh_rounds < 1`.
  """
  build_graph_from_config(
      embedding_files, output_graph_path,
      nsl_configs.GraphBuilderConfig(
          id_feature_name=id_feature_name,
          embedding_feature_name=embedding_feature_name,
          similarity_threshold=similarity_threshold,
          lsh_splits=lsh_splits,
          lsh_rounds=lsh_rounds,
          random_seed=random_seed))


def _main(argv):
  """Main function for invoking the `nsl.tools.build_graph` function."""
  flag = flags.FLAGS
  flag.showprefixforinfo = False
  if len(argv) < 3:
    raise app.UsageError(
        'Invalid number of arguments; expected 2 or more, got %d' %
        (len(argv) - 1))

  build_graph_from_config(
      argv[1:-1], argv[-1],
      nsl_configs.GraphBuilderConfig(
          id_feature_name=flag.id_feature_name,
          embedding_feature_name=flag.embedding_feature_name,
          similarity_threshold=flag.similarity_threshold,
          lsh_splits=flag.lsh_splits,
          lsh_rounds=flag.lsh_rounds,
          random_seed=flag.random_seed))


if __name__ == '__main__':
  flags.DEFINE_string(
      'id_feature_name', 'id',
      """Name of the singleton bytes_list feature in each input Example
      whose value is the Example's ID.""")
  flags.DEFINE_string(
      'embedding_feature_name', 'embedding',
      """Name of the float_list feature in each input Example
      whose value is the Example's (dense) embedding.""")
  flags.DEFINE_float(
      'similarity_threshold', 0.8,
      """Lower bound on the cosine similarity required for an edge
      to be created between two nodes.""")
  flags.DEFINE_integer(
      'lsh_splits', 0,
      """On each LSH bucketing round, the space containing the input instances
      will be randomly split/partitioned this many times for better performance,
      resulting in up to 2^(lsh_splits) LSH buckets. The larger your number of
      input instances, the larger this value should be. A good rule of thumb is
      to set `lsh_splits = ceiling(log_2(num_instances / 1000))`.""")
  flags.DEFINE_integer(
      'lsh_rounds', 2,
      """The number of rounds of LSH bucketing to perform when `lsh_splits > 0`.
      This is also the number of LSH buckets each point will be hashed into.""")
  flags.DEFINE_integer(
      'random_seed', None,
      """Value used to seed the random number generator used to perform
      randomized LSH bucketing of the inputs when `lsh_splits > 0`. By default,
      the generator will be initialized randomly, but setting this to any
      integer will initialize it deterministically.""")

  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  app.run(_main)
