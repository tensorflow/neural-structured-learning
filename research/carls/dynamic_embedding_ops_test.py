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
from research.carls import dynamic_embedding_ops as de_ops
from research.carls.testing import test_util

import tensorflow as tf


class DynamicEmbeddingOpsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DynamicEmbeddingOpsTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()

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
    with self.assertRaisesRegex(Exception,
                                'Creating DynamicEmbeddingManager failed'):
      de_ops.dynamic_embedding_lookup(['first', 'second', ''],
                                      self._config,
                                      'emb',
                                      'wrongaddress',
                                      timeout_ms=10)


if __name__ == '__main__':
  tf.test.main()
