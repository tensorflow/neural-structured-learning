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

"""Tests for the resnet library."""

import gin
from multi_representation_adversary import selectors
import tensorflow.compat.v2 as tf


class SelectorsTest(tf.test.TestCase):

  def test_greedy(self):
    selector = selectors.construct_representation_selector(
        ["a", "b", "c"], "greedy", sample_freq=2, update_freq=1)
    self.assertEqual(0, selector.select(0))  # Initialization value
    selector.update(1, [0.5, 0.3, 0.6])
    self.assertEqual(2, selector.select(1))
    self.assertEqual(2, selector.select(2))
    selector.update(2, [0.5, 0.3, 0.4])
    self.assertEqual(0, selector.select(3))
    self.assertEqual(0, selector.select(4))

  def test_round_robin(self):
    selector = selectors.construct_representation_selector(
        ["a", "b", "c"], "roundrobin", sample_freq=2, update_freq=1)
    selector.update(1, [0.5, 0.8, 0.2])
    expected = [1, 1, 2, 2, 0, 0, 1]
    actual = [selector.select(i) for i in range(1, 8)]
    self.assertEqual(expected, actual)

  def test_multiplicative_weight(self):
    tf.random.set_seed(514)
    selector = selectors.construct_representation_selector(
        ["a", "b", "c"], "multiweight", sample_freq=1, update_freq=1)
    selector.update(1, [0.5, 0.8, 0.2])
    expected = [1, 1, 1, 1, 0, 1, 2]  # Recorded for the seed
    actual = [selector.select(i) for i in range(1, 8)]
    self.assertEqual(expected, actual)

  def test_should_update(self):
    selector = selectors.construct_representation_selector(
        ["a", "b", "c"], "greedy", sample_freq=2, update_freq=2)
    selector.update(1, [0.5, 0.3, 0.6])
    self.assertFalse(selector.should_update(2))
    self.assertTrue(selector.should_update(3))


class EtaSchedulerTest(tf.test.TestCase):

  def test_constant_eta(self):
    gin.parse_config([
        "selectors.eta_scheduler.values=(0.5,)",
        "selectors.eta_scheduler.breakpoints=()",
    ])
    self.assertEqual(0.5, selectors.eta_scheduler(1))
    self.assertEqual(0.5, selectors.eta_scheduler(100))

  def test_piecewise_constant_eta(self):
    gin.parse_config([
        "selectors.eta_scheduler.values=(0.5, 0.25, 0.125)",
        "selectors.eta_scheduler.breakpoints=(100, 150)",
    ])
    self.assertEqual(0.5, selectors.eta_scheduler(1))
    self.assertEqual(0.5, selectors.eta_scheduler(100))
    self.assertEqual(0.25, selectors.eta_scheduler(101))
    self.assertEqual(0.25, selectors.eta_scheduler(150))
    self.assertEqual(0.125, selectors.eta_scheduler(151))


if __name__ == "__main__":
  tf.test.main()
