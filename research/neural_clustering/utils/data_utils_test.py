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
"""Tests for neural_structured_learning.research.neural_clustering.utils.data_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_clustering.utils import data_utils
import numpy as np


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'before': [1, 0, 3, 0, 2],
          'after': [0, 1, 2, 1, 3],
          'expected_old_id_to_new_id': {
              1: 0,
              0: 1,
              3: 2,
              2: 3
          }
      }, {
          'before': [0, 1, 2, 1, 3],
          'after': [0, 1, 2, 1, 3],
          'expected_old_id_to_new_id': {
              0: 0,
              1: 1,
              2: 2,
              3: 3
          }
      }, {
          'before': [8, 2, 4, 2, 6],
          'after': [0, 1, 2, 1, 3],
          'expected_old_id_to_new_id': {
              8: 0,
              2: 1,
              4: 2,
              6: 3
          }
      }, {
          'before': [0, 0, 0, 0, 0],
          'after': [0, 0, 0, 0, 0],
          'expected_old_id_to_new_id': {
              0: 0
          }
      })
  def test_remap_label_ids(self, before, after, expected_old_id_to_new_id):
    unordered_labels = np.array(before)
    expected_labels = np.array(after)

    remapped, old_id_to_new_id = data_utils.remap_label_ids(unordered_labels)
    np.testing.assert_array_equal(remapped, expected_labels)
    self.assertDictEqual(old_id_to_new_id, expected_old_id_to_new_id)

  @parameterized.parameters({
      'before': [[1, 0, 3, 0, 2], [0, 1, 2, 1, 3], [8, 2, 4, 2, 6],
                 [0, 0, 0, 0, 0]],
      'after': [[0, 1, 2, 1, 3], [0, 1, 2, 1, 3], [0, 1, 2, 1, 3],
                [0, 0, 0, 0, 0]],
      'expected_old_id_to_new_id': [{
          1: 0,
          0: 1,
          3: 2,
          2: 3
      }, {
          0: 0,
          1: 1,
          2: 2,
          3: 3
      }, {
          8: 0,
          2: 1,
          4: 2,
          6: 3
      }, {
          0: 0
      }]
  })
  def test_batch_remap_label_ids(self, before, after,
                                 expected_old_id_to_new_id):
    unordered_labels = np.array(before)
    expected = np.array(after)
    batch_remapped, batch_old_id_to_new_id = data_utils.batch_remap_label_ids(
        unordered_labels)
    np.testing.assert_array_equal(batch_remapped, expected)
    self.assertListEqual(batch_old_id_to_new_id, expected_old_id_to_new_id)


if __name__ == '__main__':
  absltest.main()
