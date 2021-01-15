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
"""Tests for neural_structured_learning.research.neural_clustering.models.ncp."""

from absl.testing import parameterized
from neural_structured_learning.research.neural_clustering import models
import numpy as np
import tensorflow as tf


class NcpTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the NCP clutering model."""

  @parameterized.parameters(
      {
          'labels':
              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 2, 0, 0, 0, 0, 0, 3], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 1, 2, 1, 0, 2, 3, 3, 1]]
      }, {
          'labels':
              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
               [2, 3, 2, 0, 2, 2, 2, 2, 2, 1], [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
               [3, 3, 2, 1, 2, 3, 1, 0, 0, 2]]
      })
  def test_train_ncp_wrapper_compute_loss(self, labels):
    """Test for a full forward pass through the NCP model at training time."""
    inner_model = models.NCPWithMLP(32, [16, 16], 32, [16, 16], 32, [16, 16],
                                    [16, 16])
    wrapper_model = models.NCPWrapper(inner_model, sampler=None)

    batch_size = 5
    n = 10
    x_dim = 2
    data = np.random.normal(0, 1, size=[batch_size, n, x_dim])
    labels = np.array(labels)

    logits, _ = wrapper_model(data, labels, training=True)
    loss = wrapper_model.loss_function(logits)

    self.assertGreater(loss, 0.)


if __name__ == '__main__':
  tf.test.main()
