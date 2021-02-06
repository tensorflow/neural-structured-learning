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
"""Utility functions."""

import numpy as np


def remap_label_ids(labels: np.ndarray) -> np.ndarray:
  """Remaps labels so that the first occurrence of each label ID is in ascending order.

  This remapping is to preprocess the labels of the training data for NCP.
  Suppose a dataset has 5 points with cluster labels `[1, 0, 3, 0, 2]`. Due to
  the permutation symmetry between clusters, the cluster IDs are exchangeable.
  A trained NCP model would label this dataset with `[0, 1, 2, 1, 3]`, because
  NCP creates cluster `k` before cluster `k + 1` when processing the data points
  sequentially. Thus, we remap labels in this way to prepare the training data.

  Note: If the input labels are not continuous, e.g. missing a cluster due to
  subsampling, the remapped labels will fall into the range `[0,1...k-1]` where
  `k` is the number of unique labels observed in that example. For example, if
  the input labels are `[8, 2, 4, 2, 6]`, the remapped labels will be
  `[0, 1, 2, 1, 3]`.

  Arguments:
    labels: A `np.ndarray` of shape `[num_points]`.

  Returns:
    remapped_labels: A `np.ndarray` of shape `[num_points]`.
    old_id_to_new_id: A dictionary that maps old cluster IDs to new cluster IDs.
  """
  remapped_labels = np.zeros_like(labels)
  old_id_to_new_id = {}
  next_label_id = 0
  for i, label in enumerate(labels):
    if label not in old_id_to_new_id:
      old_id_to_new_id[label] = next_label_id
      next_label_id += 1
    remapped_labels[i] = old_id_to_new_id[label]
  return remapped_labels, old_id_to_new_id


def batch_remap_label_ids(labels: np.ndarray) -> np.ndarray:
  """Remaps a batch of labels so that the first occurrence of each label ID is in ascending order.

  This is a batched version of `remap_label_ids`. The mappings are independent
  between examples in the batch.

  Arguments:
    labels: A `np.ndarray` of shape `[batch_size, num_points]`.

  Returns:
    batch_remapped_labels: A `np.ndarray` of shape `[batch_size, num_points]`.
    batch_old_id_to_new_id: A list of dictionaries that map old cluster IDs to
      new cluster IDs.
  """
  batch_remap_output = [remap_label_ids(label) for label in labels]
  batch_remapped_labels = np.stack([x[0] for x in batch_remap_output])
  batch_old_id_to_new_id = [x[1] for x in batch_remap_output]
  return batch_remapped_labels, batch_old_id_to_new_id
