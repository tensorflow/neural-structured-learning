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

"""Tests for the trainer library."""

import os

import gin
from multi_representation_adversary import trainer
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class TrainerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir()
    self.summary_dir = self.create_tempdir()

  def test_train(self):
    gin.parse_config([
        "data.preprocess_image.height = 28",
        "data.preprocess_image.width = 28",
        "data.preprocess_image.num_channels = 1",
        "data.get_test_dataset.batch_size = 1",
        "data.get_test_dataset.dataset = 'mnist'",
        "data.get_training_dataset.batch_size = 1",
        "data.get_training_dataset.dataset = 'mnist'",
        "data.get_training_dataset.shuffle_buffer_size = 1",
        "data.get_validation_dataset.batch_size = 1",
        "data.get_validation_dataset.dataset = 'mnist'",
        "data.get_validation_dataset.split = '2'",
        "resnet.build_resnet_v1.input_shape = (28, 28, 1)",
        "resnet.build_resnet_v1.depth = 8",
        "selectors.construct_representation_selector.selection_strategy = 'multiweight'",
        "selectors.construct_representation_selector.sample_freq = 1",
        "selectors.construct_representation_selector.update_freq = 1",
        "trainer.train.epochs = 2",
        "trainer.train.steps_per_epoch = 1",
        "trainer.train.representation_list = [('identity', 'l2'), ('dct', 'l2')]",
    ])
    with tfds.testing.mock_data(num_examples=10):
      trainer.train(self.ckpt_dir.full_path, self.summary_dir.full_path)

    ckpt_path = os.path.join(self.ckpt_dir, "ckpt-2")
    self.assertTrue(tf.io.gfile.exists(ckpt_path + ".index"))
    variables = [name for name, shape in tf.train.list_variables(ckpt_path)]
    self.assertTrue(any(name.startswith("model") for name in variables))
    self.assertTrue(any(name.startswith("selector") for name in variables))


if __name__ == "__main__":
  tf.test.main()
