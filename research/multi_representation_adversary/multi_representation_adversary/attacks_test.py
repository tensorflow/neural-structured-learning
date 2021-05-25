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

"""Tests for the attacks library."""

from absl.testing import parameterized
import gin
from multi_representation_adversary import attacks
import numpy as np
import tensorflow.compat.v2 as tf


class AttacksTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.input_shape = (12, 12, 3)
    self.num_classes = 10
    self.batch_size = 8
    self.batched_input_shape = tuple([self.batch_size] + list(self.input_shape))
    self.model = tf.keras.Sequential([
        tf.keras.Input(shape=self.input_shape, dtype=tf.float32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(self.num_classes),
    ])
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  @parameterized.named_parameters([
      ("single", "linf", True),
      ("with_restart", "linf_restart", True),
      ("no_random_start", "linf", False),
  ])
  def test_linf_attack(self, attack_name, random_start):
    num_iter, step_size, epsilon = 4, 0.02, 0.06
    gin.parse_config([
        f"attacks.linf_config.num_iter = {num_iter}",
        f"attacks.linf_config.step_size = {step_size}",
        f"attacks.linf_config.epsilon = {epsilon}",
        "attacks.union_config.restart = 5",
    ])

    x = tf.random.uniform(shape=self.batched_input_shape)
    y = tf.random.categorical(tf.zeros([self.batch_size, self.num_classes]), 1)
    attack = attacks.construct_attack(attack_name)
    adv_x = attack.attack(tf.constant(x), tf.constant(y), self.model,
                          self.loss_fn, random_start=random_start)

    self.assertAllLessEqual(tf.math.abs(adv_x - x), epsilon + 1e-5)

  @parameterized.named_parameters([
      ("single", "l2"),
      ("with_restart", "l2_restart"),
  ])
  def test_l2_attack(self, attack_name):
    num_iter, step_size, epsilon = 4, 0.2, 0.6
    gin.parse_config([
        f"attacks.l2_config.num_iter = {num_iter}",
        f"attacks.l2_config.step_size = {step_size}",
        f"attacks.l2_config.epsilon = {epsilon}",
        "attacks.union_config.restart = 5",
    ])

    x = tf.random.uniform(shape=self.batched_input_shape)
    y = tf.random.categorical(tf.zeros([self.batch_size, self.num_classes]), 1)
    attack = attacks.construct_attack(attack_name)
    adv_x = attack.attack(tf.constant(x), tf.constant(y), self.model,
                          self.loss_fn, random_start=True)

    diff = tf.reshape(adv_x - x, (self.batch_size, -1)).numpy()
    l2_norm = np.linalg.norm(diff, ord=2, axis=-1)
    self.assertAllLessEqual(l2_norm, epsilon + 1e-5)

  @parameterized.named_parameters([
      ("single", "l1", True),
      ("with_restart", "l1_restart", True),
      ("no_random_start", "l1", False),
  ])
  def test_l1_attack(self, attack_name, random_start):
    num_iter, step_size, epsilon, percentile = 4, 1.0, 2.5, 99
    gin.parse_config([
        f"attacks.l1_config.num_iter = {num_iter}",
        f"attacks.l1_config.step_size = {step_size}",
        f"attacks.l1_config.epsilon = {epsilon}",
        f"attacks.l1_config.percentile = {percentile}",
        "attacks.union_config.restart = 5",
    ])

    x = tf.random.uniform(shape=self.batched_input_shape)
    y = tf.random.categorical(tf.zeros([self.batch_size, self.num_classes]), 1)
    attack = attacks.construct_attack(attack_name)
    adv_x = attack.attack(tf.constant(x), tf.constant(y), self.model,
                          self.loss_fn, random_start=random_start)

    diff = tf.reshape(adv_x - x, (self.batch_size, -1)).numpy()
    l1_norm = np.linalg.norm(diff, ord=1, axis=-1)
    self.assertAllLessEqual(l1_norm, epsilon + 1e-5)
    touched = np.count_nonzero(diff, axis=-1)
    self.assertAllLessEqual(touched,
                            diff.shape[1] * num_iter * percentile / 100)

  def test_no_attack(self):
    x = tf.random.uniform(shape=self.batched_input_shape)
    y = tf.random.categorical(tf.zeros([self.batch_size, self.num_classes]), 1)
    attack = attacks.construct_attack("none")
    adv_x = attack.attack(tf.constant(x), tf.constant(y), self.model,
                          self.loss_fn, random_start=True)
    self.assertAllClose(x, adv_x)

  def test_rotation_attack(self):
    x = tf.random.uniform(shape=self.batched_input_shape)
    y = tf.random.categorical(tf.zeros([self.batch_size, self.num_classes]), 1)
    attack = attacks.construct_attack("union_rotation")
    adv_x = attack.attack(x, y, self.model, self.loss_fn)
    self.assertAllClose(x.shape, adv_x.shape)


if __name__ == "__main__":
  tf.test.main()
