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
"""Generative models of partition distributions for creating synthetic clustering datasets."""

from typing import List
import numpy as np


class CRPGenerator():
  """Partition generator using the Chinese Restaurant Process (CRP).

  The first element creates the first partition. The n-th element chooses to
  either create a new partition with probability `alpha / (n - 1 + alpha)`, or
  join an existing partition `B_k` with probabiliy `|B_k|/(n - 1 + alpha)` where
  `|B_k|` is the size of `B_k`.

  Attributes:
    alpha: A positive float representing the discount parameter of CRP.
  """

  def __init__(self, alpha: float):
    if alpha <= 0:
      raise ValueError(f"alpha must be a positive value, not {alpha}.")

    self.alpha = alpha

  def generate_batch(self, n: int, batch_size: int) -> List[np.ndarray]:
    """Generates a batch of partitions for n elements.

    Arguments:
      n: The number of data points.
      batch_size: Batch size.

    Returns:
      A list of 2-D `np.ndarray` representing a batch of partitions, where each
      array is a sampled partition configuration from CRP, and the k-th element
      in the array is the size of the k-th partition.
    """
    # A batch of partitions. In each row, the elements before alpha represents
    # the size of each existing partition. alpha is appended for multinomial
    # sampling. Here, we assign the first element to the first partition.
    partitions = np.tile([1, self.alpha], (batch_size, 1))

    # The IDs of the next empty partition across the batch.
    new_partition_ids = np.ones(batch_size, dtype=np.int32)

    # Sequentially samples the partition assignment for each element based on
    # the size of existing partitions and alpha.
    for _ in range(n - 1):
      probs = partitions / partitions.sum(axis=1, keepdims=True)
      sampled_partition_ids = batch_multinomial_sampling(probs)

      # Grows the size of the partition array if the number of partitions in
      # one of the batch instances reachs array size.
      if sampled_partition_ids.max() == new_partition_ids.max():
        partitions = np.append(partitions, np.zeros([batch_size, 1]), axis=1)

      for b, k in enumerate(sampled_partition_ids):
        if k < new_partition_ids[b]:
          # Joins an existing partition.
          partitions[b, k] += 1
        else:
          # Creates an new partition.
          partitions[b, k] = 1
          partitions[b, k + 1] = self.alpha
          new_partition_ids[b] += 1

    # Removes alpha from the array.
    partitions = [
        pt[:new_partition_ids[i]].astype(np.int32)
        for i, pt in enumerate(partitions)
    ]

    return partitions

  def generate_single(self, n: int) -> np.ndarray:
    """Generates partitions for `n` elements.

    Arguments:
      n: The number of data points.

    Returns:
      A 1-D `np.ndarray` in which each element is the size of a partition.
    """
    return self.generate_batch(n, 1)[0]


def batch_multinomial_sampling(probs: np.ndarray) -> np.ndarray:
  """Batched version of multinomial sampling.

  Draws samples from a batch of multinomial distributions simultaneously.
  Note: this can be replaced once numpy provides API support for sampling from
  multinomial distributions simultaneously.

  Arguments:
    probs: A 2-D `np.ndarray` of shape `[batch_size, num_classes]`. A batch of
      probability vectors such that `probs.sum(-1) = 1`

  Returns:
    An 1-D `np.ndarray` representing the sampled indices.
  """
  return (probs.cumsum(-1) >=
          np.random.uniform(size=probs.shape[:-1])[..., None]).argmax(-1)
