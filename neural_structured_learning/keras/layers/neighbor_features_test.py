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
"""Tests for neural_structured_learning.keras.layers.neighbor_features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import neural_structured_learning.configs as configs
from neural_structured_learning.keras.layers import neighbor_features as neighbor_features_lib
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def _make_model(neighbor_config, inputs, keep_rank, weight_dtype=None):
  """Makes a model that exercises the `NeighborFeatures` layer.

  The model takes a dictionary of sample features as input and unpacks the
  dictionary into sample features, neighbor features, and neighbor weights.

  Args:
    neighbor_config: An instance of `configs.GraphNeighborConfig`.
    inputs: A `tf.keras.Input` or a nested structure of `tf.keras.Input`s.
    keep_rank: Whether to keep the extra neighborhood size dimention.
    weight_dtype: Optional `tf.DType` for weights.

  Returns:
    An instance of `tf.keras.Model`.
  """
  # Create inputs for neighbor features and unpack.
  neighbor_features_layer = neighbor_features_lib.NeighborFeatures(
      neighbor_config, weight_dtype=weight_dtype)
  sample_features, neighbor_features, neighbor_weights = (
      neighbor_features_layer(inputs, keep_rank=keep_rank))
  return tf.keras.Model(
      inputs=inputs,
      outputs=(sample_features, neighbor_features, neighbor_weights))


@test_util.run_all_in_graph_and_eager_modes
class NeighborFeaturesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for layers.NeighborFeatures."""

  @parameterized.named_parameters([
      ('nokeep_rank', False),
      ('keep_rank', True),
  ])
  def testDense(self, keep_rank):
    """Tests creating image neighbors."""
    # Make fake 8x8 images.
    batch_size = 4
    image_height = 8
    image_width = 8
    features = {
        'image':
            np.random.randint(
                0, 256, size=(batch_size, image_height, image_width,
                              1)).astype(np.uint8),
        'NL_nbr_0_image':
            np.random.randint(
                0, 256, size=(batch_size, image_height, image_width,
                              1)).astype(np.uint8),
        'NL_nbr_1_image':
            np.random.randint(
                0, 256, size=(batch_size, image_height, image_width,
                              1)).astype(np.uint8),
        'NL_nbr_2_image':
            np.random.randint(
                0, 256, size=(batch_size, image_height, image_width,
                              1)).astype(np.uint8),
        'NL_nbr_0_weight':
            np.random.uniform(size=(batch_size, 1)).astype(np.float32),
        'NL_nbr_1_weight':
            np.random.uniform(size=(batch_size, 1)).astype(np.float32),
        'NL_nbr_2_weight':
            np.random.uniform(size=(batch_size, 1)).astype(np.float32),
    }

    num_neighbors = 3
    model = _make_model(
        configs.GraphNeighborConfig(max_neighbors=num_neighbors), {
            'image':
                tf.keras.Input((image_height, image_width, 1),
                               dtype=tf.uint8,
                               name='image'),
        }, keep_rank)
    samples, neighbors, weights = self.evaluate(model(features))
    samples, neighbors = (samples['image'], neighbors['image'])
    # Check that samples are unchanged.
    self.assertAllEqual(samples, features['image'])
    # Check that neighbors and weights are grouped together for each sample.
    for i in range(batch_size):
      self.assertAllEqual(
          neighbors[i] if keep_rank else
          neighbors[(i * num_neighbors):((i + 1) * num_neighbors)],
          np.stack([
              features['NL_nbr_0_image'][i],
              features['NL_nbr_1_image'][i],
              features['NL_nbr_2_image'][i],
          ]))
      self.assertAllEqual(
          weights[i] if keep_rank else np.split(weights, batch_size)[i],
          np.stack([
              features['NL_nbr_0_weight'][i],
              features['NL_nbr_1_weight'][i],
              features['NL_nbr_2_weight'][i],
          ]))

  @parameterized.named_parameters([
      ('nokeep_rank', False),
      ('keep_rank', True),
  ])
  def testSparse(self, keep_rank):
    """Tests the layer with a variable number of neighbors."""
    batch_size = 4
    input_size = 2
    features = {
        'input':
            tf.sparse.from_dense(
                np.random.normal(size=(batch_size, input_size))),
        # Every sample but the last has 1 neighbor.
        'NL_nbr_0_input':
            tf.RaggedTensor.from_row_starts(
                values=np.random.normal(size=(batch_size - 1) * input_size),
                row_starts=[0, 2, 4, 6]).to_sparse(),
        'NL_nbr_0_weight':
            np.expand_dims(np.array([0.9, 0.3, 0.6, 0.]), -1),
        # Only the 1st and 3rd sample have a second neighbor.
        'NL_nbr_1_input':
            tf.RaggedTensor.from_row_starts(
                values=np.random.normal(size=(batch_size - 2) * input_size),
                row_starts=[0, 2, 2, 4]).to_sparse(),
        'NL_nbr_1_weight':
            np.expand_dims(np.array([0.25, 0., 0.75, 0.]), -1),
    }

    model = _make_model(
        configs.GraphNeighborConfig(max_neighbors=2),
        {'input': tf.keras.Input(input_size, dtype=tf.float64)}, keep_rank,
        tf.float64)
    samples, neighbors, weights = self.evaluate(model(features))
    # Check that samples are unchanged.
    self.assertAllClose(samples['input'].values,
                        self.evaluate(features['input'].values))
    # Check that weights are grouped together and have the right shape.
    self.assertAllClose(
        weights,
        np.array([0.9, 0.25, 0.3, 0., 0.6, 0.75, 0.,
                  0.]).reshape((batch_size, 2,
                                1) if keep_rank else (batch_size * 2, 1)))
    # Check that neighbors are grouped together.
    dense_neighbors = self.evaluate(tf.sparse.to_dense(neighbors['input'], -1.))
    neighbor0 = self.evaluate(
        tf.sparse.to_dense(features['NL_nbr_0_input'], -1))
    neighbor1 = self.evaluate(
        tf.sparse.to_dense(features['NL_nbr_1_input'], -1))
    for i in range(batch_size):
      actual = (
          dense_neighbors[i]
          if keep_rank else np.split(dense_neighbors, batch_size)[i])
      self.assertAllEqual(actual, np.stack([neighbor0[i], neighbor1[i]]))


if __name__ == '__main__':
  tf.test.main()
