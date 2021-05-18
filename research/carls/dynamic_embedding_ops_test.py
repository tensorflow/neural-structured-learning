# Copyright 2020 Google LLC
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
"""Tests for neural_structured_learning.research.carls.dynamic_embedding_ops."""

import itertools
from absl.testing import parameterized
from research.carls import context
from research.carls import dynamic_embedding_ops as de_ops
from research.carls.testing import test_util

import numpy as np
import tensorflow as tf


class DynamicEmbeddingOpsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DynamicEmbeddingOpsTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(DynamicEmbeddingOpsTest, self).tearDown()

  @parameterized.parameters(itertools.product((True, False), (1000, -1)))
  def testLookup_1DInput(self, skip_gradient, timeout_ms):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    embedding = de_ops.dynamic_embedding_lookup(
        ['first'],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient,
        timeout_ms=timeout_ms)
    self.assertAllClose(embedding.numpy(), [[1, 2]])

    embedding = de_ops.dynamic_embedding_lookup(
        ['first', 'second', ''],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[1, 2], [1, 2], [0, 0]])

  @parameterized.parameters({True, False})
  def testLookup_2DInput(self, skip_gradient):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    embedding = de_ops.dynamic_embedding_lookup(
        [['first', 'second'], ['third', '']],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[[1, 2], [1, 2]], [[1, 2], [0, 0]]])

  @parameterized.parameters((True, True), (True, False), (False, False),
                            (False, True))
  def testUpdate_1DInput(self, use_kbs_address, skip_gradient):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    embedding = de_ops.dynamic_embedding_lookup(
        ['first'],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[1, 2]])

    update_res = de_ops.dynamic_embedding_update(
        ['first'],
        tf.constant([[2.0, 4.0]]),
        self._config,
        'emb',
        service_address=self._kbs_address,
    )
    self.assertAllClose(update_res.numpy(), [[2, 4]])

    embedding = de_ops.dynamic_embedding_lookup(
        ['first'],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[2, 4]])

    # Allows keys' shape to be [N, 1] and values shape to be [N, D].
    update_res = de_ops.dynamic_embedding_update(
        [['first']],
        tf.constant([[4.0, 5.0]]),
        self._config,
        'emb',
        service_address=self._kbs_address)
    self.assertAllClose(update_res.numpy(), [[4, 5]])
    embedding = de_ops.dynamic_embedding_lookup(
        ['first'],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[4, 5]])

  @parameterized.parameters({True, False})
  def testUpdate_2DInput(self, skip_gradient):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    embedding = de_ops.dynamic_embedding_lookup(
        [['first', 'second'], ['third', '']],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(), [[[1, 2], [1, 2]], [[1, 2], [0, 0]]])

    # The values for an empty key should be ignored.
    update_res = de_ops.dynamic_embedding_update(
        [['first', 'second'], ['third', '']],
        tf.constant([[[2.0, 4.0], [4.0, 8.0]], [[8.0, 16.0], [16.0, 32.0]]]),
        self._config,
        'emb',
        service_address=self._kbs_address,
    )
    self.assertAllClose(update_res.numpy(),
                        [[[2, 4], [4, 8]], [[8, 16], [0, 0]]])

    embedding = de_ops.dynamic_embedding_lookup(
        [['first', 'second'], ['third', '']],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(),
                        [[[2, 4], [4, 8]], [[8, 16], [0, 0]]])

    # Allows keys' shape to be [N1, N2, 1] and values shape to be [N1, N2, D].
    update_res = de_ops.dynamic_embedding_update(
        [[['first'], ['second']], [['third'], ['']]],
        tf.constant([[[3.0, 5.0], [5.0, 9.0]], [[9.0, 17.0], [17.0, 33.0]]]),
        self._config,
        'emb',
        service_address=self._kbs_address,
    )
    self.assertAllClose(update_res.numpy(),
                        [[[3, 5], [5, 9]], [[9, 17], [0, 0]]])
    embedding = de_ops.dynamic_embedding_lookup(
        [['first', 'second'], ['third', '']],
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=skip_gradient)
    self.assertAllClose(embedding.numpy(),
                        [[[3, 5], [5, 9]], [[9, 17], [0, 0]]])

  def testWrongAddress(self):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    with self.assertRaisesRegex(Exception, 'DynamicEmbeddingManager is NULL.'):
      de_ops.dynamic_embedding_lookup(['first', 'second', ''],
                                      self._config,
                                      'emb',
                                      'wrongaddress',
                                      timeout_ms=10)

  def testTrainingLogistic(self):
    embedding_dimension = 5
    self._config.embedding_dimension = embedding_dimension

    # Set initial embedding to be all zero's.
    init = self._config.knowledge_bank_config.initializer
    for _ in range(embedding_dimension):
      init.default_embedding.value.append(0)

    # Create variables.
    initializer = tf.ones_initializer()
    w = tf.Variable(
        initializer(shape=[embedding_dimension, 1], dtype=tf.float32))
    b = tf.Variable(0.0)
    trainable_variables = [w, b]

    # Create an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # Conducts one step of gradient descent.
    ids = np.array(['yes', 'no', 'good', 'bad'])
    y = np.array([[1], [0], [1], [0]])
    with tf.GradientTape() as tape:
      embedding = de_ops.dynamic_embedding_lookup(
          ids,
          self._config,
          'emb',
          service_address=self._kbs_address,
          skip_gradient_update=False)
      logit = tf.linalg.matmul(embedding, w) + b
      pred = 1 / (1 + tf.exp(-logit))
      loss = y * tf.math.log(pred) + (1 - y) * tf.math.log(1 - pred)
    grads = tape.gradient(loss, trainable_variables)
    # Update the trainable variables w.r.t. the logistic loss
    optimizer.apply_gradients(zip(grads, trainable_variables))

    # Checks that the embeddings are updated.
    new_embedding = de_ops.dynamic_embedding_lookup(
        ids,
        self._config,
        'emb',
        service_address=self._kbs_address,
        skip_gradient_update=False)
    distance = np.sum((new_embedding.numpy() - embedding.numpy())**2)
    self.assertGreater(distance, 0)

    # Checks that the new loss is smaller.
    new_logit = tf.linalg.matmul(new_embedding, w) + b
    new_pred = 1 / (1 + tf.exp(-new_logit))
    new_loss = y * tf.math.log(new_pred) + (1 - y) * tf.math.log(1 - new_pred)
    for old, new in zip(loss.numpy(), new_loss.numpy()):
      self.assertLess(new[0], old[0])

  def _create_dataset(self):
    """Returns a tf.data.Dataset with dynamic embedding as input."""
    dataset = tf.data.Dataset.range(100)
    dataset = dataset.batch(batch_size=4, drop_remainder=True)

    def _parse(example):
      string_ids = tf.strings.as_string(example)
      input_embed = de_ops.dynamic_embedding_lookup(
          string_ids,
          self._config,
          'input_embed',
          service_address=self._kbs_address,
          skip_gradient_update=True)
      return input_embed

    dataset = dataset.map(_parse, num_parallel_calls=2)
    return dataset

  def testDynamicEmbeddingTfDataset(self):
    """Test DynamicEmbedding's compatibility with tf.data.Dataset API."""
    dataset = self._create_dataset()
    for data in dataset:
      self.assertAllEqual([4, 2], data.shape)

  def testDynamicEmbeddingKerasInterface_KerasLayer(self):
    de_layer = de_ops.DynamicEmbeddingLookup(
        self._config, 'embed', service_address=self._kbs_address)
    # 1D case.
    embed = de_layer(np.array(['key1', 'key2', 'key3']))
    self.assertEqual((3, 2), embed.shape)
    # 2D case.
    embed = de_layer(np.array([['key1', 'key2'], ['key3', '']]))
    self.assertEqual((2, 2, 2), embed.shape)

  def testDynamicEmbeddingKerasInterface_KerasModel(self):
    """A simple Logistic Regression Keras model."""
    string_ids = np.array([['yes'], ['no'], ['good'], ['bad']])
    y_train = np.array([[[1, 0]], [[0, 1]], [[1, 0]], [[0, 1]]])

    model = tf.keras.models.Sequential([
        de_ops.DynamicEmbeddingLookup(
            self._config, 'embed', service_address=self._kbs_address),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(string_ids, y_train, epochs=10)
    # Checks that the loss is decreased.
    self.assertLess(history.history['loss'][-1], history.history['loss'][0])


if __name__ == '__main__':
  tf.test.main()
