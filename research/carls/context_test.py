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
"""Tests for neural_structured_learning.research.carls.context."""

from research.carls import context

from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
import tensorflow as tf


class ContextTest(tf.test.TestCase):

  def test_add_to_collection(self):
    config = de_config_pb2.DynamicEmbeddingConfig(embedding_dimension=5)
    context.add_to_collection('first', config)
    context.add_to_collection('second', config)
    context.add_to_collection('first', config)  # ok to add twice.
    self.assertLen(context._knowledge_bank_collections, 2)

    # Empty name.
    with self.assertRaises(ValueError):
      context.add_to_collection('', config)
    # Wrong config type.
    with self.assertRaises(TypeError):
      context.add_to_collection('first', 'config')
    # Checks adding a different config with the same name is not allowed.
    config.embedding_dimension = 10
    with self.assertRaises(ValueError):
      context.add_to_collection('first', config)

  def test_get_all_collection(self):
    config = de_config_pb2.DynamicEmbeddingConfig(embedding_dimension=5)
    context.add_to_collection('first', config)
    context.add_to_collection('second', config)
    collections = context.get_all_collection()
    self.assertLen(collections, 2)
    for key, value in collections:
      self.assertIn(key, {'first', 'second'})
      self.assertProtoEquals(value, config)

  def testClearAllCollection(self):
    config = de_config_pb2.DynamicEmbeddingConfig(embedding_dimension=5)
    context.add_to_collection('first', config)
    context.add_to_collection('second', config)
    collections = context.get_all_collection()
    self.assertLen(collections, 2)
    context.clear_all_collection()
    collections = context.get_all_collection()
    self.assertLen(collections, 0)


if __name__ == '__main__':
  tf.test.main()
