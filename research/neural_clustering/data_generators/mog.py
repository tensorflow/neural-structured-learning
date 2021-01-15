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
"""Generative model for Mixture of Gaussians."""

from typing import List, Tuple

import numpy as np


class MOGGenerator():
  """Synthetic data generator for Mixture of Gaussians (MoG).

  Attributes:
    partition_generator: A partition generator with a `generate_batch` method
      for producing the cluster partitions, or `None`. `generate_batch(n,
      batch_size)` should return a list of `np.ndarray` representing a batch of
      partitions, where each array is a sampled partition configuration, and the
      k-th element in the array is the size of the k-th partition. If set to
      `None`, the `generate_by_partitions` method can be used to create MOG with
      an arbitrary partition configuration.
    x_dim: the number of dimensions.
    prior_sigma: the standard deviation for generating the centers of the
      mixture components.
    sigma_min: the minimum standard deviation for each mixture component.
    sigma_max: the maximum standard deviation for each mixture component.
  """

  def __init__(self, partition_generator, x_dim: int, prior_sigma: float,
               sigma_min: float, sigma_max: float):
    self.partition_generator = partition_generator
    self.x_dim = x_dim
    self.prior_sigma = prior_sigma
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max

  def generate_batch(self,
                     n: int,
                     batch_size: int,
                     standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a batch of Gaussian mixtures using the partition_generator.

    Arguments:
      n: The number of data points.
      batch_size: Batch size.
      standardize: Boolean of whether to standardize the data.

    Returns:
      data: A `np.ndarray` of shape `[batch_size, n_points, x_dim]`.
      labels: An int `np.ndarray` of shape `[batch_size, n_points]`.
    """
    if self.partition_generator is None:
      raise ValueError("partition_generator must be set in the constructor.")

    partitions = self.partition_generator.generate_batch(
        n=n, batch_size=batch_size)

    return self.generate_by_partitions(
        partitions=partitions, standardize=standardize)

  def generate_by_partitions(
      self,
      partitions: List[np.ndarray],
      standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a batch of Gaussian mixtures given a batch of partitions.

    Arguments:
      partitions: A list of integer arrays. Each array is a partition
        configuration, and each element in the array is the size of a partition.
        The sum of each array must be the same.
      standardize: Boolean of whether to standardize the data.

    Returns:
      data: A float `np.ndarray` of shape `[batch_size, n_points, x_dim]`.
      labels: An int `np.ndarray` of shape `[batch_size, n_points]`.
    """
    batch_size = len(partitions)
    partition_sums = np.array([pt.sum() for pt in partitions])
    n = partition_sums[0]

    if not np.all(partition_sums == n):
      raise ValueError(
          "All partition configurations in the batch must contain the same total number of points."
      )

    max_num_clusters = max(len(pt) for pt in partitions)

    cumsums = [np.cumsum(np.insert(pt, 0, [0])) for pt in partitions]
    data = np.zeros([batch_size, n, self.x_dim])
    labels = np.zeros([batch_size, n], dtype=np.int32)

    # Generates samples from a mixture of Gaussians. In each iteration, a new
    # cluster of data points is created.
    for k in range(max_num_clusters):
      # Generates the center and standard deviation of each cluster.
      mu = np.random.normal(
          0, self.prior_sigma, size=[batch_size, 1, self.x_dim])
      sigma = np.random.uniform(
          self.sigma_min, self.sigma_max, size=[batch_size, 1, self.x_dim])

      # Generates data points within each cluster.
      for i in range(batch_size):
        if k >= len(partitions[i]):
          continue
        samples = np.random.normal(
            mu[i], sigma[i], size=[partitions[i][k], self.x_dim])
        data[i, cumsums[i][k]:cumsums[i][k + 1], :] = samples
        labels[i, cumsums[i][k]:cumsums[i][k + 1]] = k

    # Shuffles the ordering of the data.
    indices = np.arange(n)
    np.random.shuffle(indices)
    data = data[:, indices, :]
    labels = labels[:, indices]

    # Standardizes data.
    if standardize:
      data -= np.mean(data, axis=1, keepdims=True)
      data /= np.std(data, axis=1, keepdims=True)

    return data, labels
