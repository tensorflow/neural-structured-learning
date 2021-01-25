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
"""Tests for neural_structured_learning.research.neural_clustering.data_generators.mog."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_clustering.data_generators import mog
import numpy as np


class DummyPartitionGenerator():

  def generate_batch(self, n: int, batch_size: int):
    partitions = []
    for _ in range(batch_size):
      k = np.random.randint(1, 10)
      partition = np.random.multinomial(n, np.ones(k) / k, size=1)[0]
      partitions.append(partition)
    return partitions


class MogTest(parameterized.TestCase):

  @parameterized.parameters({
      'x_dim': 5,
      'prior_sigma': 10,
      'sigma_min': 1,
      'sigma_max': 3,
      'n': 100,
      'batch_size': 16
  })
  def test_generate_batch_output_shape(self, x_dim, prior_sigma, sigma_min,
                                       sigma_max, n, batch_size):
    partition_generator = DummyPartitionGenerator()
    mog_generator = mog.MOGGenerator(
        partition_generator=partition_generator,
        x_dim=x_dim,
        prior_sigma=prior_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max)
    data, labels = mog_generator.generate_batch(n=n, batch_size=batch_size)
    np.testing.assert_array_equal(data.shape, np.array([batch_size, n, x_dim]))
    np.testing.assert_array_equal(labels.shape, np.array([batch_size, n]))

  @parameterized.parameters({
      'x_dim': 5,
      'prior_sigma': 10,
      'sigma_min': 1,
      'sigma_max': 3,
      'n': 100,
      'batch_size': 16
  })
  def test_generate_by_partitions_output_shape(self, x_dim, prior_sigma,
                                               sigma_min, sigma_max, n,
                                               batch_size):
    partition_generator = DummyPartitionGenerator()
    mog_generator = mog.MOGGenerator(
        partition_generator=None,
        x_dim=x_dim,
        prior_sigma=prior_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max)
    partitions = partition_generator.generate_batch(n, batch_size)
    data, labels = mog_generator.generate_by_partitions(partitions=partitions)
    np.testing.assert_array_equal(data.shape, np.array([batch_size, n, x_dim]))
    np.testing.assert_array_equal(labels.shape, np.array([batch_size, n]))


if __name__ == '__main__':
  absltest.main()
