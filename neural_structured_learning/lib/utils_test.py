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
"""Tests for neural_structured_learning.lib.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import utils
import numpy as np
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def testNormalizeInf(self):
    with self.cached_session() as sess:
      target_tensor = tf.constant([[1.0, 2.0, -4.0], [-1.0, 5.0, -3.0]])
      normalized_tensor = utils.normalize(target_tensor, 'infinity')
      expected_tensor = tf.constant([[0.25, 0.5, -1.0], [-0.2, 1.0, -0.6]])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testNormalizeInfWithOnes(self):
    with self.cached_session() as sess:
      target_tensor = tf.constant(1.0, shape=[2, 4])
      normalized_tensor = utils.normalize(target_tensor, 'infinity')
      expected_tensor = tf.constant(1.0, shape=[2, 4])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testNormalizeInfWithZero(self):
    with self.cached_session() as sess:
      tensor = tf.constant(0.0, shape=[2, 3])
      normalized_tensor = utils.normalize(tensor, 'infinity')
      expected_tensor = tf.constant(0.0, shape=[2, 3])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testNormalizeL1(self):
    with self.cached_session() as sess:
      # target_tensor = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
      target_tensor = tf.constant(1.0, shape=[2, 4])
      normalized_tensor = utils.normalize(target_tensor, 'l1')
      # L1 norm of target_tensor (other than batch/1st dim) is [4, 4]; therefore
      # target_tensor = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
      expected_tensor = tf.constant(0.25, shape=[2, 4])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testNormalizeL1WithZero(self):
    with self.cached_session() as sess:
      tensor = tf.constant(0.0, shape=[2, 3])
      normalized_tensor = utils.normalize(tensor, 'l1')
      expected_tensor = tf.constant(0.0, shape=[2, 3])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testNormalizeL2(self):
    with self.cached_session() as sess:
      # target_tensor = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
      target_tensor = tf.constant(1.0, shape=[2, 4])
      normalized_tensor = utils.normalize(target_tensor, 'l2')
      # L2 norm of target_tensor (other than batch/1st dim) is:
      # [sqrt(1^2+1^2+1^2+1^2), sqrt(1^2+1^2+1^2+1^2)] = [2, 2], and therefore
      # target_tensor = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
      expected_tensor = tf.constant(0.5, shape=[2, 4])
      sess.run(normalized_tensor)
      self.assertAllEqual(normalized_tensor, expected_tensor)

  def testMaximizeWithinUnitNormInf(self):
    with self.cached_session() as sess:
      weights = tf.constant([[1.0, 2.0, -4.0], [-1.0, 5.0, -3.0]])
      actual = utils.maximize_within_unit_norm(weights, 'infinity')
      sess.run(actual)
      expected = tf.constant([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])
      self.assertAllEqual(actual, expected)

  def testMaximizeWithinUnitNormL1(self):
    with self.cached_session() as sess:
      weights = tf.constant([[3.0, -4.0, -5.0], [1.0, 1.0, 0.0]])
      actual = utils.maximize_within_unit_norm(weights, 'l1')
      sess.run(actual)
      expected = tf.constant([[0.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
      self.assertAllEqual(actual, expected)

  def testMaximizeWithinUnitNormL2(self):
    with self.cached_session() as sess:
      weights = tf.constant([[3.0, -4.0], [-7.0, 24.0]])
      actual = utils.maximize_within_unit_norm(weights, 'l2')
      sess.run(actual)
      # Weights are normalized by their L2 norm: [[5], [25]]
      expected = tf.constant([[0.6, -0.8], [-0.28, 0.96]])
      self.assertAllEqual(actual, expected)

  def testReplicateEmbeddingsWithConstant(self):
    """Test the replicate_embeddings function with constant replicate_times."""
    input_embeddings = tf.constant(
        [[[1, 2, 4], [3, 5, 8]], [[2, 10, 3], [1, 1, 1]], [[4, 8, 1], [8, 4, 1]]
        ],
        dtype='float32')
    output_embeddings = utils.replicate_embeddings(input_embeddings, 2)
    with self.cached_session() as sess:
      self.assertAllEqual([[[1, 2, 4], [3, 5, 8]], [[1, 2, 4], [3, 5, 8]],
                           [[2, 10, 3], [1, 1, 1]], [[2, 10, 3], [1, 1, 1]],
                           [[4, 8, 1], [8, 4, 1]], [[4, 8, 1], [8, 4, 1]]],
                          sess.run(output_embeddings))

  def testReplicateEmbeddingsWithIndexArray(self):
    """Test the replicate_embeddings function with 1-D replicate_times."""
    input_embeddings = tf.constant(
        [[[1, 2, 4], [3, 5, 8]], [[2, 10, 3], [1, 1, 1]], [[4, 8, 1], [8, 4, 1]]
        ],
        dtype='float32')
    replicate_times = tf.constant([2, 0, 1])
    output_embeddings = utils.replicate_embeddings(input_embeddings,
                                                   replicate_times)
    with self.cached_session() as sess:
      self.assertAllEqual([[[1, 2, 4], [3, 5, 8]], [[1, 2, 4], [3, 5, 8]],
                           [[4, 8, 1], [8, 4, 1]]], sess.run(output_embeddings))

  def testReplicateEmbeddingsWithDynamicBatchSize(self):
    """Test the replicate_embeddings function with a dynamic batch size."""
    emb1 = [[1, 2, 3], [3, 2, 1]]
    emb2 = [[4, 5, 6], [6, 5, 4]]
    emb3 = [[7, 8, 9], [9, 8, 7]]
    input_embeddings = np.array([emb1, emb2, emb3], dtype=np.float32)
    replicate_times = np.array([2, 1, 2], dtype=np.int32)

    @tf.function(
        input_signature=(tf.TensorSpec(
            (None, 2, 3), tf.float32), tf.TensorSpec((None,), tf.int32)))
    def _replicate_with_dynamic_batch_size(embeddings, replicate_times):
      return utils.replicate_embeddings(embeddings, replicate_times)

    output_embeddings = self.evaluate(
        _replicate_with_dynamic_batch_size(input_embeddings, replicate_times))
    self.assertAllEqual(output_embeddings, [emb1, emb1, emb2, emb3, emb3])

  def testInvalidRepeatTimes(self):
    """Test the replicate_embeddings function with invalid repeat_times."""
    input_embeddings = tf.constant(
        [[[1, 2, 4], [3, 5, 8]], [[2, 10, 3], [1, 1, 1]], [[4, 8, 1], [8, 4, 1]]
        ],
        dtype='float32')
    replicate_times = tf.constant([-1, 0, 1])
    with self.cached_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        output_embeddings = utils.replicate_embeddings(input_embeddings,
                                                       replicate_times)
        output_embeddings.eval()


class GetTargetIndicesTest(tf.test.TestCase):

  def testGetSecondIndices(self):
    """Test get_target_indices function with AdvTargetType.SECOND."""
    logits = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype='float32')
    labels = tf.constant([2, 1], dtype='int32')
    adv_target_config = configs.AdvTargetConfig(
        target_method=configs.AdvTargetType.SECOND)
    with self.cached_session() as sess:
      self.assertAllEqual(
          tf.constant([1, 0], dtype='int32'),
          sess.run(utils.get_target_indices(logits, labels, adv_target_config)))

  def testGetLeastIndices(self):
    """Test get_target_indices function with AdvTargetType.LEAST."""
    logits = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype='float32')
    labels = tf.constant([2, 1], dtype='int32')
    adv_target_config = configs.AdvTargetConfig(
        target_method=configs.AdvTargetType.LEAST)
    with self.cached_session() as sess:
      self.assertAllEqual(
          tf.constant([0, 2], dtype='int32'),
          sess.run(utils.get_target_indices(logits, labels, adv_target_config)))

  def testGetGroundTruthIndices(self):
    """Test get_target_indices function with AdvTargetType.GROUND_TRUTH."""
    logits = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype='float32')
    labels = tf.constant([2, 1], dtype='int32')
    adv_target_config = configs.AdvTargetConfig(
        target_method=configs.AdvTargetType.GROUND_TRUTH)
    with self.cached_session() as sess:
      self.assertAllEqual(
          tf.constant([2, 1], dtype='int32'),
          sess.run(utils.get_target_indices(logits, labels, adv_target_config)))

  def testGetRandomIndices(self):
    """Test get_target_indices function with AdvTargetType.RANDOM."""
    logits = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype='float32')
    labels = tf.constant([2, 1], dtype='int32')
    adv_target_config = configs.AdvTargetConfig(
        target_method=configs.AdvTargetType.RANDOM, random_seed=1)
    with self.cached_session() as sess:
      self.assertAllEqual(
          tf.constant([0, 2], dtype='int32'),
          sess.run(utils.get_target_indices(logits, labels, adv_target_config)))


def decay_over_time_wrapper(config):

  @tf.function
  def decay_over_time(global_step, init_value=1.0):
    return utils.decay_over_time(global_step, config, init_value)

  return decay_over_time


class DecayOverTimeTest(tf.test.TestCase):

  def testExponentialDecay(self):
    """Test the decay_over_time function with exponential decay applied."""
    init_value = 0.1
    decay_step = 10
    global_step = 5
    decay_rate = 0.96
    expected_value = init_value * decay_rate**(global_step / decay_step)
    config = configs.DecayConfig(decay_step, decay_rate)
    decayed_value = decay_over_time_wrapper(config)(global_step, init_value)
    self.assertAllClose(decayed_value, expected_value, 1e-6)

  def testBoundedDecay(self):
    """Test the decay_over_time function with bounded decay value."""
    init_value = 0.1
    min_value = 0.99
    decay_step = 10
    global_step = 5
    decay_rate = 0.96
    bounded_config = configs.DecayConfig(decay_step, decay_rate, min_value)
    bounded_value = decay_over_time_wrapper(bounded_config)(global_step,
                                                            init_value)
    self.assertAllClose(bounded_value, min_value, 1e-6)

  def testInverseTimeDecay(self):
    """Test the decay_over_time function with inverse time decay applied."""
    init_value = 0.1
    decay_step = 10
    global_step = 5
    decay_rate = 0.9
    expected_value = init_value / (1 + decay_rate * global_step / decay_step)
    config = configs.DecayConfig(
        decay_step, decay_rate, decay_type=configs.DecayType.INVERSE_TIME_DECAY)
    decayed_value = decay_over_time_wrapper(config)(global_step, init_value)
    self.assertAllClose(decayed_value, expected_value, 1e-6)

  def testNaturalExpDecay(self):
    """Test the decay_over_time function with natural exp decay applied."""
    init_value = 0.1
    decay_step = 10
    global_step = 5
    decay_rate = 0.9
    expected_value = init_value * math.exp(
        -decay_rate * global_step / decay_step)
    config = configs.DecayConfig(
        decay_step, decay_rate, decay_type=configs.DecayType.NATURAL_EXP_DECAY)
    decayed_value = decay_over_time_wrapper(config)(global_step, init_value)
    self.assertAllClose(decayed_value, expected_value, 1e-6)

  def testDefaultInitValueWithExponentialDecay(self):
    """Test the decay_over_time function with default init value."""
    decay_step = 10
    global_step = 5
    decay_rate = 0.96
    expected_value = decay_rate**(global_step / decay_step)
    config = configs.DecayConfig(decay_step, decay_rate)
    decayed_value = decay_over_time_wrapper(config)(global_step)
    self.assertAllClose(decayed_value, expected_value, 1e-6)

  def testApplyFeatureMask(self):
    """Test the apply_feature_mask function."""
    features = [[1.0, 1.0], [2.0, 2.0]]
    mask = [0.0, 1.0]
    masked_features = utils.apply_feature_mask(
        tf.constant(features), tf.constant(mask))
    with self.cached_session() as sess:
      actual = sess.run(masked_features)
    self.assertAllClose(actual, [[0.0, 1.0], [0.0, 2.0]], 1e-6)

  def testApplyFeatureMaskWithNone(self):
    """Test the apply_feature_mask function with 'None' feature mask."""
    features = [[1.0, 1.0], [2.0, 2.0]]
    masked_features = utils.apply_feature_mask(tf.constant(features))
    with self.cached_session() as sess:
      actual = sess.run(masked_features)
    self.assertAllClose(actual, features, 1e-6)

  def testApplyFeatureMaskWithInvalidMaskNegative(self):
    """Test the apply_feature_mask function with mask value < 0."""
    features = [[1.0, 1.0], [2.0, 2.0]]
    mask = [-1.0, 1.0]
    # In eager mode, the arguments are validated once `tf.debugging.assert_*` is
    # called (in `utils.apply_feature_mask`). In graph mode, the call to
    # `tf.debugging.assert_*` only creates an Op, and the actual validation
    # happens when the graph is run. The behavior in graph mode may change in
    # the future to validate statically known arguments (e.g. `tf.constant`) at
    # Op-creation time. Enclosing both Op creation and evaluation is
    # an `assertRaises` block handles all cases.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      masked_features = utils.apply_feature_mask(
          tf.constant(features), tf.constant(mask))
      self.evaluate(masked_features)

  def testApplyFeatureMaskWithInvalidMaskTooLarge(self):
    """Test the apply_feature_mask function with mask value > 1."""
    features = [[1.0, 1.0], [2.0, 2.0]]
    mask = [1.0, 2.0]
    # In eager mode, the arguments are validated once `tf.debugging.assert_*` is
    # called (in `utils.apply_feature_mask`). In graph mode, the call to
    # `tf.debugging.assert_*` only creates an Op, and the actual validation
    # happens when the graph is run. The behavior in graph mode may change in
    # the future to validate statically known arguments (e.g. `tf.constant`) at
    # Op-creation time. Enclosing both Op creation and evaluation is
    # an `assertRaises` block handles all cases.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      masked_features = utils.apply_feature_mask(
          tf.constant(features), tf.constant(mask))
      self.evaluate(masked_features)


class UnpackNeighborFeaturesTest(tf.test.TestCase):
  """Tests unpacking of sample feature, neighbor features, and neighbor weights.

    This class currently expects a fixed number of neighbors per sample.
  """

  def testSampleFeatureOnlyExtractionWithNoNeighbors(self):
    """Test sample feature extraction without neighbor features."""
    # Simulate batch size of 1.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'F1': tf.constant([[3.0, 4.0, 5.0]]),
    }

    expected_sample_features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'F1': tf.constant([[3.0, 4.0, 5.0]]),
    }

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=0)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)
    self.assertIsNone(nbr_weights)

    with self.cached_session() as sess:
      sess.run([sample_features, nbr_features])
      self.assertAllEqual(sample_features['F0'], expected_sample_features['F0'])
      self.assertAllEqual(sample_features['F1'], expected_sample_features['F1'])
      self.assertEmpty(nbr_features)

  def testSampleFeatureOnlyExtractionWithNeighbors(self):
    """Test sample feature extraction with neighbor features."""
    # Simulate batch size of 1.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'F1': tf.constant([[3.0, 4.0, 5.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_F1': tf.constant([[3.1, 4.1, 5.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_F0': tf.constant([[1.2, 2.2]]),
        'NL_nbr_1_F1': tf.constant([[3.2, 4.2, 5.2]]),
        'NL_nbr_1_weight': tf.constant([[0.75]]),
    }

    expected_sample_features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'F1': tf.constant([[3.0, 4.0, 5.0]]),
    }

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=0)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)
    self.assertIsNone(nbr_weights)

    with self.cached_session() as sess:
      sess.run([sample_features, nbr_features])
      self.assertAllEqual(sample_features['F0'], expected_sample_features['F0'])
      self.assertAllEqual(sample_features['F1'], expected_sample_features['F1'])
      self.assertEmpty(nbr_features)

  def testBatchedSampleAndNeighborFeatureExtraction(self):
    """Test input contains two samples with one feature and three neighbors."""
    # Simulate a batch size of 2.
    features = {
        'F0': tf.constant(11.0, shape=[2, 2]),
        'NL_nbr_0_F0': tf.constant(22.0, shape=[2, 2]),
        'NL_nbr_0_weight': tf.constant(0.25, shape=[2, 1]),
        'NL_nbr_1_F0': tf.constant(33.0, shape=[2, 2]),
        'NL_nbr_1_weight': tf.constant(0.75, shape=[2, 1]),
        'NL_nbr_2_F0': tf.constant(44.0, shape=[2, 2]),
        'NL_nbr_2_weight': tf.constant(1.0, shape=[2, 1]),
    }

    expected_sample_features = {
        'F0': tf.constant(11.0, shape=[2, 2]),
    }

    # The key in this dictionary will contain the original sample's feature
    # name. The shape of the corresponding tensor will be 6x2, which is the
    # result of doing an interleaved merge of three 2x2 tensors along axis 0.
    expected_neighbor_features = {
        'F0':
            tf.constant([[22.0, 22.0], [33.0, 33.0], [44.0, 44.0], [22.0, 22.0],
                         [33.0, 33.0], [44.0, 44.0]]),
    }
    # The shape of this tensor is 6x1, which is the result of doing an
    # interleaved merge of three 2x1 tensors along axis 0.
    expected_neighbor_weights = tf.constant([[0.25], [0.75], [1.0], [0.25],
                                             [0.75], [1.0]])

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=3)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)

    with self.cached_session() as sess:
      sess.run([sample_features, nbr_features, nbr_weights])
      self.assertAllEqual(sample_features['F0'], expected_sample_features['F0'])
      self.assertAllEqual(nbr_features['F0'], expected_neighbor_features['F0'])
      self.assertAllEqual(nbr_weights, expected_neighbor_weights)

  def testExtraNeighborFeaturesIgnored(self):
    """Test that extra neighbor features are ignored."""
    # Simulate a batch size of 1 for simplicity.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_F0': tf.constant([[1.2, 2.2]]),
        'NL_nbr_1_weight': tf.constant([[0.75]]),
    }

    expected_sample_features = {
        'F0': tf.constant([[1.0, 2.0]]),
    }

    expected_neighbor_features = {
        'F0': tf.constant([[1.1, 2.1]]),
    }
    expected_neighbor_weights = tf.constant([[0.25]])

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=1)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)

    with self.cached_session() as sess:
      sess.run([sample_features, nbr_features, nbr_weights])
      self.assertAllEqual(sample_features['F0'], expected_sample_features['F0'])
      self.assertAllEqual(nbr_features['F0'], expected_neighbor_features['F0'])
      self.assertAllEqual(nbr_weights, expected_neighbor_weights)

  def testEmptyFeatures(self):
    """Test unpack_neighbor_features with empty input."""
    features = {}
    neighbor_config = configs.GraphNeighborConfig(max_neighbors=0)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)
    self.assertIsNone(nbr_weights)

    with self.cached_session() as sess:
      # We create a dummy tensor so that the computation graph is not empty.
      dummy_tensor = tf.constant(1.0)
      sess.run([sample_features, nbr_features, dummy_tensor])
      self.assertEmpty(sample_features)
      self.assertEmpty(nbr_features)

  def testInvalidRank(self):
    """Input containing rank 1 tensors raises ValueError."""
    # Simulate a batch size of 1 for simplicity.
    features = {
        'F0': tf.constant([1.0, 2.0]),
        'NL_nbr_0_F0': tf.constant([1.1, 2.1]),
        'NL_nbr_0_weight': tf.constant([0.25]),
    }

    with self.assertRaises(ValueError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=1)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testInvalidNeighborWeightRank(self):
    """Input containing a rank 3 neighbor weight tensor raises ValueError."""
    features = {
        'F0': tf.constant([1.0, 2.0]),
        'NL_nbr_0_F0': tf.constant([1.1, 2.1]),
        'NL_nbr_0_weight': tf.constant([[[0.25]]]),
    }

    with self.assertRaises(ValueError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=1)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testMissingNeighborFeature(self):
    """Missing neighbor feature raises KeyError."""
    # Simulate a batch size of 1 for simplicity.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_weight': tf.constant([[0.75]]),
    }

    with self.assertRaises(KeyError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testMissingNeighborWeight(self):
    """Missing neighbor weight raises KeyError."""
    # Simulate a batch size of 1 for simplicity.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_F0': tf.constant([[1.2, 2.2]]),
    }

    with self.assertRaises(KeyError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testSampleAndNeighborFeatureShapeIncompatibility(self):
    """Sample feature and neighbor feature have incompatible shapes."""
    # Simulate a batch size of 1 for simplicity.
    # The shape of the sample feature is 1x2 while the shape of the
    # corresponding neighbor feature 1x3.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1, 3.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
    }

    with self.assertRaises(ValueError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=1)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testNeighborFeatureShapeIncompatibility(self):
    """One neighbor feature has an incompatible shape."""
    # Simulate a batch size of 1 for simplicity.
    # The shape of the sample feature and one neighbor feature is 1x2, while the
    # shape of another neighbor feature 1x3.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_F0': tf.constant([[1.2, 2.2, 3.2]]),
        'NL_nbr_1_weight': tf.constant([[0.5]]),
    }

    with self.assertRaises(ValueError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testNeighborWeightShapeIncompatibility(self):
    """One neighbor weight has an incompatibile shape."""
    # Simulate a batch size of 1 for simplicity.
    # The shape of one neighbor weight is 1x2 instead of 1x1.
    features = {
        'F0': tf.constant([[1.0, 2.0]]),
        'NL_nbr_0_F0': tf.constant([[1.1, 2.1]]),
        'NL_nbr_0_weight': tf.constant([[0.25]]),
        'NL_nbr_1_F0': tf.constant([[1.2, 2.2]]),
        'NL_nbr_1_weight': tf.constant([[0.5, 0.75]]),
    }

    with self.assertRaises(ValueError):
      neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)
      utils.unpack_neighbor_features(features, neighbor_config)

  def testSparseFeature(self):
    """Test the case when the sample has a sparse feature."""
    # Simulate batch size of 2.
    features = {
        'F0':
            tf.constant(11.0, shape=[2, 2]),
        'F1':
            tf.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1.0, 2.0], dense_shape=[2,
                                                                          4]),
        'NL_nbr_0_F0':
            tf.constant(22.0, shape=[2, 2]),
        'NL_nbr_0_F1':
            tf.SparseTensor(
                indices=[[1, 0], [1, 1]], values=[3.0, 4.0], dense_shape=[2,
                                                                          4]),
        'NL_nbr_0_weight':
            tf.constant(0.25, shape=[2, 1]),
        'NL_nbr_1_F0':
            tf.constant(33.0, shape=[2, 2]),
        'NL_nbr_1_F1':
            tf.SparseTensor(
                indices=[[0, 2], [1, 3]], values=[5.0, 6.0], dense_shape=[2,
                                                                          4]),
        'NL_nbr_1_weight':
            tf.constant(0.75, shape=[2, 1]),
    }

    expected_sample_features = {
        'F0':
            tf.constant(11.0, shape=[2, 2]),
        'F1':
            tf.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1.0, 2.0], dense_shape=[2,
                                                                          4]),
    }

    # The keys in this dictionary will contain the original sample's feature
    # names.
    expected_neighbor_features = {
        # The shape of the corresponding tensor for 'F0' will be 4x2, which is
        # the result of doing an interleaved merge of two 2x2 tensors along
        # axis 0.
        'F0':
            tf.constant([[22, 22], [33, 33], [22, 22], [33, 33]]),
        # The shape of the corresponding tensor for 'F1' will be 4x4, which is
        # the result of doing an interleaved merge of two 2x4 tensors along
        # axis 0.
        'F1':
            tf.SparseTensor(
                indices=[[1, 2], [2, 0], [2, 1], [3, 3]],
                values=[5.0, 3.0, 4.0, 6.0],
                dense_shape=[4, 4]),
    }
    # The shape of this tensor is 4x1, which is the result of doing an
    # interleaved merge of two 2x1 tensors along axis 0.
    expected_neighbor_weights = tf.constant([[0.25], [0.75], [0.25], [0.75]])

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)
    sample_features, nbr_features, nbr_weights = utils.unpack_neighbor_features(
        features, neighbor_config)

    with self.cached_session() as sess:
      sess.run([sample_features, nbr_features, nbr_weights])
      self.assertAllEqual(sample_features['F0'], expected_sample_features['F0'])
      self.assertAllEqual(sample_features['F1'].values,
                          expected_sample_features['F1'].values)
      self.assertAllEqual(sample_features['F1'].indices,
                          expected_sample_features['F1'].indices)
      self.assertAllEqual(sample_features['F1'].dense_shape,
                          expected_sample_features['F1'].dense_shape)
      self.assertAllEqual(nbr_features['F0'], expected_neighbor_features['F0'])
      self.assertAllEqual(nbr_features['F1'].values,
                          expected_neighbor_features['F1'].values)
      self.assertAllEqual(nbr_features['F1'].indices,
                          expected_neighbor_features['F1'].indices)
      self.assertAllEqual(nbr_features['F1'].dense_shape,
                          expected_neighbor_features['F1'].dense_shape)
      self.assertAllEqual(nbr_weights, expected_neighbor_weights)

  def testDynamicBatchSizeAndFeatureShape(self):
    """Test the case when the batch size and feature shape are both dynamic."""
    # Use a dynamic batch size and a dynamic feature shape. The former
    # corresponds to the first dimension of the tensors defined below, and the
    # latter corresonponds to the second dimension of 'sample_features' and
    # 'neighbor_i_features'.

    feature_specs = {
        'F0': tf.TensorSpec((None, None, 3), tf.float32),
        'NL_nbr_0_F0': tf.TensorSpec((None, None, 3), tf.float32),
        'NL_nbr_0_weight': tf.TensorSpec((None, 1), tf.float32),
        'NL_nbr_1_F0': tf.TensorSpec((None, None, 3), tf.float32),
        'NL_nbr_1_weight': tf.TensorSpec((None, 1), tf.float32)
    }

    # Specify a batch size of 3 and a pre-batching feature shape of 2x3 at run
    # time.
    sample1 = [[1, 2, 3], [3, 2, 1]]
    sample2 = [[4, 5, 6], [6, 5, 4]]
    sample3 = [[7, 8, 9], [9, 8, 7]]
    sample_features = [sample1, sample2, sample3]  # 3x2x3

    neighbor_0_features = [[[1, 3, 5], [5, 3, 1]],
                           [[7, 9, 11], [11, 9, 7]],
                           [[13, 15, 17], [17, 15, 13]]]  # 3x2x3
    neighbor_0_weights = [[0.25], [0.5], [0.75]]  # 3x1

    neighbor_1_features = [[[2, 4, 6], [6, 4, 2]],
                           [[8, 10, 12], [12, 10, 8]],
                           [[14, 16, 18], [18, 16, 14]]]  # 3x2x3
    neighbor_1_weights = [[0.75], [0.5], [0.25]]  # 3x1

    expected_sample_features = {'F0': sample_features}

    features = {
        'F0': sample_features,
        'NL_nbr_0_F0': neighbor_0_features,
        'NL_nbr_0_weight': neighbor_0_weights,
        'NL_nbr_1_F0': neighbor_1_features,
        'NL_nbr_1_weight': neighbor_1_weights
    }

    # The key in this dictionary will contain the original sample's feature
    # name. The shape of the corresponding tensor will be 6x2x3, which is the
    # result of doing an interleaved merge of 2 3x2x3 tensors along axis 0.
    expected_neighbor_features = {
        'F0': [[[1, 3, 5], [5, 3, 1]], [[2, 4, 6], [6, 4, 2]],
               [[7, 9, 11], [11, 9, 7]], [[8, 10, 12], [12, 10, 8]],
               [[13, 15, 17], [17, 15, 13]], [[14, 16, 18], [18, 16, 14]]],
    }
    # The shape of this tensor is 6x1, which is the result of doing an
    # interleaved merge of two 3x1 tensors along axis 0.
    expected_neighbor_weights = [[0.25], [0.75], [0.5], [0.5], [0.75], [0.25]]

    neighbor_config = configs.GraphNeighborConfig(max_neighbors=2)

    @tf.function(input_signature=[feature_specs])
    def _unpack_neighbor_features(features):
      return utils.unpack_neighbor_features(features, neighbor_config)

    sample_feats, nbr_feats, nbr_weights = self.evaluate(
        _unpack_neighbor_features(features))

    self.assertAllEqual(sample_feats['F0'], expected_sample_features['F0'])
    self.assertAllEqual(nbr_feats['F0'], expected_neighbor_features['F0'])
    self.assertAllEqual(nbr_weights, expected_neighbor_weights)


if __name__ == '__main__':
  tf.test.main()
