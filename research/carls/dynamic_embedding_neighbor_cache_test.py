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
"""Tests for neural_structured_learning.research.carls.dynamic_embedding_neighbor_cache."""

from research.carls import context
from research.carls import dynamic_embedding_neighbor_cache as de_nb_cache
from research.carls.testing import test_util

import tensorflow as tf


class DynamicEmbeddingNeighborCacheTest(tf.test.TestCase):

  def setUp(self):
    super(DynamicEmbeddingNeighborCacheTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(DynamicEmbeddingNeighborCacheTest, self).tearDown()

  def testLookup(self):
    init = self._config.knowledge_bank_config.initializer
    init.default_embedding.value.append(1)
    init.default_embedding.value.append(2)
    cache = de_nb_cache.DynamicEmbeddingNeighborCache(
        'nb_cache', self._config, service_address=self._kbs_address)
    embedding = cache.lookup(['first', 'second', ''])
    self.assertAllClose(embedding.numpy(), [[1, 2], [1, 2], [0, 0]])

  def testUpdate(self):
    cache = de_nb_cache.DynamicEmbeddingNeighborCache(
        'nb_cache', self._config, service_address=self._kbs_address)
    update_res = cache.update(['first', 'second', ''],
                              tf.constant([[2.0, 4.0], [4.0, 8.0], [8.0,
                                                                    16.0]]))
    self.assertAllClose(update_res.numpy(), [[2, 4], [4, 8], [0, 0]])

    embedding = cache.lookup(['first', 'second', ''])
    self.assertAllClose(embedding.numpy(), [[2, 4], [4, 8], [0, 0]])


if __name__ == '__main__':
  tf.test.main()
