# Copyright 2021 Google LLC
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
"""Tests for research.carls.models.caml.sparse_features lib."""

from absl.testing import parameterized
from research.carls.models.caml import sparse_features
from research.carls.testing import test_util
import numpy as np
import tensorflow as tf


class FeatureEmbeddingTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FeatureEmbeddingTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()

  def tearDown(self):
    self._service_server.Terminate()
    super(FeatureEmbeddingTest, self).tearDown()

  @parameterized.named_parameters(('with_sigma', 5), ('without_sigma', 0))
  def test_partitioned_dynamic_embedding_lookup_1D_input(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    embed, sigma = sparse_features._partitioned_dynamic_embedding_lookup(
        ['input1', 'input2'],
        config,
        5,
        sigma_dimension,
        'feature_name0_%d' % sigma_dimension,
        service_address=self._kbs_address)

    if sigma_dimension > 0:
      self.assertEqual((2, sigma_dimension), sigma.shape)
      self.assertEqual((2, 5), embed.shape)
    else:
      self.assertAllClose([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], embed.numpy())
      self.assertIsNone(sigma)

  @parameterized.named_parameters(('with_sigma', 3), ('without_sigma', 0))
  def test_partitioned_dynamic_embedding_lookup_2D_input(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    emb, sigma = sparse_features._partitioned_dynamic_embedding_lookup(
        [['input1', ''], ['input2', 'input3']],
        config,
        5,
        sigma_dimension,
        'feature_name1_%d' % sigma_dimension,
        service_address=self._kbs_address)

    if sigma_dimension > 0:
      self.assertEqual((2, 2, sigma_dimension), sigma.shape)
      self.assertEqual((2, 2, 5), emb.shape)
      self.assertAllClose([[[1] * sigma_dimension, [0] * sigma_dimension],
                           [[1] * sigma_dimension, [1] * sigma_dimension]],
                          sigma.numpy())
      self.assertAllClose([[[1] * 5, [0] * 5], [[1] * 5, [1] * 5]], emb.numpy())
    else:
      self.assertAllClose([[[1] * 5, [0] * 5], [[1] * 5, [1] * 5]], emb.numpy())
      self.assertIsNone(sigma)

  @parameterized.named_parameters(('with_sigma', 3), ('without_sigma', 0))
  def test_embed_single_feature_1D_input(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    emb, vc, sigma, input_embed, variables = sparse_features.embed_single_feature(
        ['input1', 'input2'],
        config,
        5,
        sigma_dimension,
        'feature_name2_%d' % sigma_dimension,
        service_address=self._kbs_address)

    if sigma_dimension > 0:
      self.assertIsNotNone(variables)
      self.assertEqual((2, 5), emb.shape)
      self.assertEqual(5, vc.shape)
      self.assertEqual((2, 1), sigma.shape)
      self.assertEqual((2, 5), input_embed.shape)
    else:
      self.assertAllClose([[1] * 5, [1] * 5], emb.numpy())
      self.assertIsNone(vc)
      self.assertIsNone(sigma)
      self.assertAllClose([[1] * 5, [1] * 5], input_embed)

    # Lookup again with given variables. Checks all values are the same.
    new_emb, new_vc, new_sigma, new_input_embed, variables = (
        sparse_features.embed_single_feature(['input1', 'input2'],
                                             config,
                                             5,
                                             sigma_dimension,
                                             'feature_name2_%d' %
                                             sigma_dimension,
                                             variables=variables,
                                             service_address=self._kbs_address))
    if sigma_dimension > 0:
      self.assertIsNotNone(variables)
    self.assertAllClose(emb.numpy(), new_emb.numpy())
    if vc is not None:
      self.assertAllClose(vc.numpy(), new_vc.numpy())
    if sigma is not None:
      self.assertAllClose(sigma.numpy(), new_sigma.numpy())
    self.assertAllClose(input_embed.numpy(), new_input_embed.numpy())

  @parameterized.named_parameters(('with_sigma', 3), ('without_sigma', 0))
  def test_embed_single_feature_2D_input(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    emb, vc, sigma, input_embed, var = sparse_features.embed_single_feature(
        [['input1', ''], ['input2', 'input3']],
        config,
        5,
        sigma_dimension,
        'feature_name3_%d' % sigma_dimension,
        service_address=self._kbs_address)

    if sigma_dimension > 0:
      self.assertIsNotNone(var)
      self.assertEqual((2, 5), emb.shape)
      self.assertEqual(5, vc.shape)
      self.assertEqual((2, 2), sigma.shape)
      self.assertEqual((2, 2, 5), input_embed.shape)
    else:
      self.assertAllClose([[1] * 5, [1] * 5], emb)
      self.assertIsNone(vc)
      self.assertIsNone(sigma)
      self.assertEqual((2, 2, 5), input_embed.shape)

  @parameterized.named_parameters(('with_sigma', 3), ('without_sigma', 0))
  def test_single_feature_lookup_1D(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    fea_embed = sparse_features.SparseFeatureEmbedding(
        config, {'fea': (5, sigma_dimension)},
        op_name='single_feature_%d' % sigma_dimension,
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup(['input1', 'input2'])
    if sigma_dimension > 0:
      self.assertEqual((2, 5), embed.shape)
    else:
      self.assertAllClose([[1] * 5, [1] * 5], embed)
    self.assertEqual(['fea'], list(embed_map.keys()))
    self.assertEqual((2, 5), embed_map['fea'].shape)
    self.assertEqual(['fea'], list(fea_embed._variable_map.keys()))

  @parameterized.named_parameters(('with_sigma', 3), ('without_sigma', 0))
  def test_single_feature_lookup_2D(self, sigma_dimension):
    emb_dim = 5 + sigma_dimension
    config = test_util.default_de_config(emb_dim, [1] * emb_dim)
    fea_embed = sparse_features.SparseFeatureEmbedding(
        config, {'fea': (5, sigma_dimension)},
        op_name='single_feature_%d' % sigma_dimension,
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup([['input1', ''],
                                               ['input2', 'input3']])
    if sigma_dimension > 0:
      self.assertEqual((2, 5), embed.shape)
    else:
      self.assertAllClose([[1] * 5, [1] * 5], embed)
    self.assertEqual(['fea'], list(embed_map.keys()))
    self.assertEqual((2, 2, 5), embed_map['fea'].shape)
    self.assertEqual(['fea'], list(fea_embed._variable_map.keys()))

  def test_multiple_feature_lookup_1D_with_sigma(self):
    fea_embed = sparse_features.SparseFeatureEmbedding(
        self._config, {
            'fea1': (5, 1),
            'fea2': (10, 1)
        },
        op_name='multiple_feature0',
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup({
        'fea1': ['input1', 'input2'],
        'fea2': ['input3', 'input4']
    })
    self.assertEqual((2, 15), embed.shape)
    self.assertLen(embed_map.keys(), 2)
    self.assertIn('fea1', embed_map.keys())
    self.assertIn('fea2', embed_map.keys())
    self.assertEqual((2, 5), embed_map['fea1'].shape)
    self.assertEqual((2, 10), embed_map['fea2'].shape)
    self.assertLen(fea_embed._variable_map.keys(), 2)
    self.assertIn('fea1', fea_embed._variable_map.keys())
    self.assertIn('fea2', fea_embed._variable_map.keys())

  def test_multiple_feature_lookup_1D_without_sigma(self):
    config = test_util.default_de_config(5, [1] * 5)
    fea_embed = sparse_features.SparseFeatureEmbedding(
        config, {
            'fea1': (5, 0),
            'fea2': (5, 0)
        },
        op_name='multiple_feature1',
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup({
        'fea1': ['input1', 'input2'],
        'fea2': ['input3', 'input4']
    })
    self.assertAllClose([[1] * 10, [1] * 10], embed.numpy())
    self.assertLen(embed_map.keys(), 2)
    self.assertIn('fea1', embed_map.keys())
    self.assertIn('fea2', embed_map.keys())
    self.assertEqual((2, 5), embed_map['fea1'].shape)
    self.assertEqual((2, 5), embed_map['fea2'].shape)
    self.assertLen(fea_embed._variable_map.keys(), 2)
    self.assertIn('fea1', fea_embed._variable_map.keys())
    self.assertIn('fea2', fea_embed._variable_map.keys())

  def test_multiple_feature_lookup_2D_with_sigma(self):
    fea_embed = sparse_features.SparseFeatureEmbedding(
        self._config, {
            'fea1': (5, 1),
            'fea2': (10, 1)
        },
        op_name='multiple_feature2',
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup({
        'fea1': [['input1', ''], ['input2', '']],
        'fea2': [['input3', 'input5'], ['input4', 'input6']]
    })
    self.assertEqual((2, 15), embed.shape)
    self.assertLen(embed_map.keys(), 2)
    self.assertIn('fea1', embed_map.keys())
    self.assertIn('fea2', embed_map.keys())
    self.assertEqual((2, 2, 5), embed_map['fea1'].shape)
    self.assertEqual((2, 2, 10), embed_map['fea2'].shape)
    self.assertLen(fea_embed._variable_map.keys(), 2)
    self.assertIn('fea1', fea_embed._variable_map.keys())
    self.assertIn('fea2', fea_embed._variable_map.keys())

  def test_multiple_feature_lookup_2D_without_sigma(self):
    config = test_util.default_de_config(5, [1] * 5)
    fea_embed = sparse_features.SparseFeatureEmbedding(
        config, {
            'fea1': (5, 0),
            'fea2': (5, 0)
        },
        op_name='multiple_feature3',
        service_address=self._kbs_address)
    embed, _, _, embed_map = fea_embed.lookup({
        'fea1': [['input1', ''], ['input2', '']],
        'fea2': [['input3', 'input5'], ['input4', 'input6']]
    })
    self.assertAllClose([[1] * 10, [1] * 10], embed.numpy())
    self.assertLen(embed_map.keys(), 2)
    self.assertIn('fea1', embed_map.keys())
    self.assertIn('fea2', embed_map.keys())
    self.assertEqual((2, 2, 5), embed_map['fea1'].shape)
    self.assertEqual((2, 2, 5), embed_map['fea2'].shape)
    self.assertLen(fea_embed._variable_map.keys(), 2)
    self.assertIn('fea1', fea_embed._variable_map.keys())
    self.assertIn('fea2', fea_embed._variable_map.keys())

  def test_training_logistic(self):
    self._config.gradient_descent_config.learning_rate = 0.05
    fea_embed = sparse_features.SparseFeatureEmbedding(
        self._config, {
            'weather': (10, 2),
            'day_of_week': (10, 2)
        },
        op_name='multiple_feature',
        service_address=self._kbs_address)

    model = tf.keras.models.Sequential(
        [fea_embed, tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    x_train = {
        'weather': [['cold', 'day'], ['hot', ''], ['warm', 'day'], ['warm',
                                                                    '']],
        'day_of_week': [['monday', 'day'], ['tuesday', 'day'], ['sunday', ''],
                        ['saturday', '']],
    }
    y_train = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    # Test the shape of model's output.
    self.assertEqual((4, 2), model(x_train).shape)
    loss_layer = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    logits = model(x_train)
    init_loss = loss_layer(logits, y_train)
    for _ in range(10):
      with tf.GradientTape() as tape:
        logits = model(x_train)
        loss = loss_layer(logits, y_train)
      grads = tape.gradient(loss, model.trainable_weights)
      # Update the trainable variables w.r.t. the logistic loss
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      print('===>loss: ', loss_layer(logits, y_train).numpy())
    # Checks the loss is dropped after 10 steps of training.
    logits = model(x_train)
    final_loss = loss_layer(logits, y_train)
    self.assertLess(final_loss.numpy(), init_loss.numpy())


if __name__ == '__main__':
  tf.test.main()
