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
"""Tests for neural_structured_learning.research.neural_clustering.data_generators.partition."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_structured_learning.research.neural_clustering.data_generators import partition
import numpy as np


class PartitionTest(parameterized.TestCase):

  @parameterized.parameters({
      'n': 100,
      'batch_size': 16,
      'alpha': 1
  }, {
      'n': 100,
      'batch_size': 16,
      'alpha': 0.1
  }, {
      'n': 1,
      'batch_size': 16,
      'alpha': 1
  }, {
      'n': 100,
      'batch_size': 1,
      'alpha': 1
  })
  def test_crp_generator_generate_batch(self, n, batch_size, alpha):
    crp_generator = partition.CRPGenerator(alpha=alpha)
    partitions = crp_generator.generate_batch(n, batch_size)
    self.assertLen(partitions, batch_size)
    partition_sums = np.array([pt.sum() for pt in partitions])
    np.testing.assert_array_equal(partition_sums, np.array([n] * batch_size))
    invalid_partitions = np.array([np.sum(pt < 1) for pt in partitions])
    np.testing.assert_array_equal(invalid_partitions, np.zeros(batch_size))

  @parameterized.parameters({
      'n': 100,
      'alpha': 1
  }, {
      'n': 100,
      'alpha': 0.1
  }, {
      'n': 1,
      'alpha': 1
  })
  def test_crp_generator_generate_single(self, n, alpha):
    crp_generator = partition.CRPGenerator(alpha=alpha)
    partitions = crp_generator.generate_single(n)
    partition_sum = partitions.sum()
    self.assertEqual(partition_sum, n)
    no_zero_partition = (partitions >= 1).all()
    self.assertTrue(no_zero_partition)


if __name__ == '__main__':
  absltest.main()
