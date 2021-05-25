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

"""Tests for the evaluator library."""

import os

from absl.testing import parameterized
import gin
from multi_representation_adversary import evaluator
from multi_representation_adversary import resnet
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class EvaluatorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir()
    self.summary_dir = self.create_tempdir()
    gin.parse_config([
        "data.preprocess_image.height = 28",
        "data.preprocess_image.width = 28",
        "data.preprocess_image.num_channels = 1",
        "data.get_test_dataset.batch_size = 1",
        "data.get_test_dataset.dataset = 'mnist'",
        "evaluator.evaluate.epochs = 5",
        "resnet.build_resnet_v1.input_shape = (28, 28, 1)",
        "resnet.build_resnet_v1.depth = 8",
    ])
    for epoch in range(6):
      model = resnet.build_resnet_v1()  # Randomly initialized
      ckpt = tf.train.Checkpoint(model)
      ckpt.save_counter.assign(epoch - 1)  # Next save() uses number <epoch>.
      ckpt.save(os.path.join(self.ckpt_dir, "ckpt"))

  @parameterized.named_parameters([
      ("single_checkpoint", 1, "avg_weight", 1),
      ("average_weights", 3, "avg_weight", 2),
      ("ensmeble", 3, "ensemble", 2),
  ])
  def test_evaluate(self, num_aggregate, aggregation_method,
                    aggregation_interval):
    gin.parse_config([
        f"evaluator.evaluate.num_aggregate = {num_aggregate}",
        f"evaluator.evaluate.aggregation_method = '{aggregation_method}'",
        f"evaluator.evaluate.aggregation_interval = {aggregation_interval}",
        "evaluator.evaluate.representation_list = [('identity', 'linf')]",
        "evaluator.evaluate.should_write_final_predictions = True",
    ])
    with tfds.testing.mock_data(num_examples=10):
      evaluator.evaluate(self.ckpt_dir.full_path, self.summary_dir.full_path)
    self.assertTrue(tf.io.gfile.exists(
        os.path.join(self.summary_dir, "results_identity_linf.tsv")))


class AggregationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir()
    self.w0 = np.array([[2., -1.]])
    self.w1 = np.array([[3., -2.]])
    model = self._build_toy_model()
    ckpt = tf.train.Checkpoint(model=model)
    model.weights[0].assign(self.w0)
    ckpt.save(os.path.join(self.ckpt_dir, "ckpt"))  # ckpt-1
    model.weights[0].assign(self.w1)
    ckpt.save(os.path.join(self.ckpt_dir, "ckpt"))  # ckpt-2

  def _build_toy_model(self):
    # 1-input, 2-output linear model (logit = w * x)
    return tf.keras.Sequential([
        tf.keras.Input(shape=[1], dtype=tf.float32),
        tf.keras.layers.Dense(2, use_bias=False),
    ])

  def test_load_and_aggregate_avg_weight(self):
    model = evaluator.load_and_aggregate(self.ckpt_dir, [1, 2], "avg_weight",
                                         self._build_toy_model)
    self.assertAllClose((self.w0 + self.w1) / 2, model(np.array([[1.]])))

  def test_load_and_aggregate_ensemble(self):
    model = evaluator.load_and_aggregate(self.ckpt_dir, [1, 2], "ensemble",
                                         self._build_toy_model)
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    expected = (softmax(self.w0) + softmax(self.w1)) / 2
    actual = softmax(model(np.array([[1.]])))
    self.assertAllClose(expected, actual)  # Compare after softmax


if __name__ == "__main__":
  tf.test.main()
