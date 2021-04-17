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
"""Tests for neural_structured_learning.research.carls.candidate_sampling_ops."""

from research.carls import candidate_sampling_ops as cs_ops
from research.carls import context
from research.carls import dynamic_embedding_ops as de_ops
from research.carls.candidate_sampling import candidate_sampler_config_builder as cs_config_builder
from research.carls.testing import test_util

import tensorflow as tf


class CandidateSamplingOpsTest(tf.test.TestCase):

  def setUp(self):
    super(CandidateSamplingOpsTest, self).setUp()
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(CandidateSamplingOpsTest, self).tearDown()

  def test_brute_force_topk(self):
    cs_config = cs_config_builder.build_candidate_sampler_config(
        cs_config_builder.brute_force_topk_sampler('DOT_PRODUCT'))
    de_config = test_util.default_de_config(2, cs_config=cs_config)
    # Add a few embeddings into knowledge bank.
    de_ops.dynamic_embedding_update(['key1', 'key2', 'key3'],
                                    tf.constant([[2.0, 4.0], [4.0, 8.0],
                                                 [8.0, 16.0]]),
                                    de_config,
                                    'emb',
                                    service_address=self._kbs_address)

    keys, logits = cs_ops.top_k([[1.0, 2.0], [-1.0, -2.0]],
                                3,
                                de_config,
                                'emb',
                                service_address=self._kbs_address)
    self.assertAllEqual(
        keys.numpy(),
        [[b'key3', b'key2', b'key1'], [b'key1', b'key2', b'key3']])
    self.assertAllClose(logits.numpy(), [[40, 20, 10], [-10, -20, -40]])

  def test_compute_sampled_logits(self):
    cs_config = cs_config_builder.build_candidate_sampler_config(
        cs_config_builder.negative_sampler(unique=True, algorithm='UNIFORM'))
    de_config = test_util.default_de_config(3, cs_config=cs_config)

    # Add a few embeddings into knowledge bank.
    de_ops.dynamic_embedding_update(['key1', 'key2', 'key3'],
                                    tf.constant([[1.0, 2.0,
                                                  3.0], [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]),
                                    de_config,
                                    'emb',
                                    service_address=self._kbs_address)

    # Sample logits.
    logits, labels, keys, mask, weights = cs_ops.compute_sampled_logits(
        [['key1', ''], ['key2', 'key3']],
        tf.constant([[2.0, 4.0, 1], [-2.0, -4.0, 1]]),
        3,
        de_config,
        'emb',
        service_address=self._kbs_address)

    # Expected results:
    # - Example one returns one positive key {'key2'} and two negative keys
    #   {'key2', 'key3'}.
    # - Example two returns two positive keys {'key2', 'key3'} and one
    #   positive key {'key1'}.
    expected_weights = {
        b'key1': [1, 2, 3],
        b'key2': [4, 5, 6],
        b'key3': [7, 8, 9]
    }
    expected_labels = [{
        b'key1': 1,
        b'key2': 0,
        b'key3': 0
    }, {
        b'key1': 0,
        b'key2': 1,
        b'key3': 1
    }]
    # Logit for example one:
    # - 'key1': [2, 4, 1] * [1, 2, 3] = 13
    # - 'key2': [2, 4, 1] * [4, 5, 6] = 34
    # - 'key3': [2, 4, 1] * [7, 8, 9] = 55
    # Logit for example two:
    # - 'key1': [-2, -4, 1] * [1, 2, 3] = -7
    # - 'key2': [-2, -4, 1] * [4, 5, 6] = -22
    # - 'key3': [-2, -4, 1] * [7, 8, 9] = -37
    expected_logits = [{
        b'key1': 13,
        b'key2': 34,
        b'key3': 55
    }, {
        b'key1': -7,
        b'key2': -22,
        b'key3': -37
    }]
    # Check keys and weights.
    for b in range(2):
      self.assertEqual(1, mask.numpy()[b])

      for key in {b'key1', b'key2', b'key3'}:
        self.assertIn(key, keys.numpy()[b])
      for i in range(3):
        key = keys.numpy()[b][i]
        self.assertAllClose(expected_weights[key], weights.numpy()[b][i])
        self.assertAllClose(expected_labels[b][key], labels.numpy()[b][i])
        self.assertAllClose(expected_logits[b][key], logits.numpy()[b][i])

  def test_compute_sampled_logits_grad(self):
    cs_config = cs_config_builder.build_candidate_sampler_config(
        cs_config_builder.negative_sampler(unique=True, algorithm='UNIFORM'))
    de_config = test_util.default_de_config(3, cs_config=cs_config)

    # Add a few embeddings into knowledge bank.
    de_ops.dynamic_embedding_update(['key1', 'key2', 'key3'],
                                    tf.constant([[1.0, 2.0,
                                                  3.0], [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]),
                                    de_config,
                                    'emb',
                                    service_address=self._kbs_address)

    # A simple one layer NN model.
    # Input data: x = [[1, 2], [3, 4]].
    # Weights from input to logit output layer: W = [[1, 2, 3], [4, 5, 6]].
    # Input activation at output layer i = x*W = [[9, 12, 15], [19, 26, 33]].
    # Logits output therefore becomes E*i, where E are the embeddings of output
    #  keys, i.e., E = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
    # Then the logits output becomes [[78, 186, 294], [170, 404, 638]]
    #
    # If we define the loss to be L = tf.reduced_sum(Logits), then
    # dL/dE = sum_by_key(i) = [[28, 38, 48], [28, 38, 48], [28, 38, 48]].
    # So the expected new embeddings become
    # E - 0.1 * dL/dE = [[-1.8, -1.8, -1.8], [1.2, 1.2, 1.2], [4.2, 4.2, 4.2]].
    weights = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    inputs = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    with tf.GradientTape() as tape:
      logits, _, _, _, _ = cs_ops.compute_sampled_logits(
          [['key1', ''], ['key2', 'key3']],
          tf.matmul(inputs, weights),
          3,
          de_config,
          'emb',
          service_address=self._kbs_address)
      loss = tf.reduce_sum(logits)

    # Applies the gradient descent.
    grads = tape.gradient(loss, weights)

    # The gradients updated by the knowledge bank.
    updated_embedding = de_ops.dynamic_embedding_lookup(
        ['key1', 'key2', 'key3'],
        de_config,
        'emb',
        service_address=self._kbs_address)
    self.assertAllClose(updated_embedding,
                        [[-1.8, -1.8, -1.8], [1.2, 1.2, 1.2], [4.2, 4.2, 4.2]])

    # The gradients w.r.t. the weight W is calculated as
    # dL/dw = dL/di * di/dW = sum_by_dim(E) * x =
    # [12, 15, 18] * [[4, 4, 4], [6, 6, 6]] = [[48, 60, 72], [72, 90, 108]]
    self.assertAllClose(grads, [[48, 60, 72], [72, 90, 108]])


if __name__ == '__main__':
  tf.test.main()
