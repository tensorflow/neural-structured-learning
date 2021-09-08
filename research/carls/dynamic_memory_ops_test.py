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
"""Tests for neural_structured_learning.research.carls.dynamic_memory_ops."""

from research.carls import context
from research.carls import dynamic_memory_ops as dm_ops
from research.carls.testing import test_util
import tensorflow as tf


class DynamicMemoryOpsTest(tf.test.TestCase):

  def setUp(self):
    super(DynamicMemoryOpsTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(DynamicMemoryOpsTest, self).tearDown()

  def testGaussianMemoryLookupWithSingleCluster(self):
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=3,
        distance_to_cluster_threshold=0.5,
        bootstrap_steps=0,
        min_variance=1,
        max_num_clusters=1)
    inputs = [[0, 0], [1, 0], [101, 0]]
    mode = dm_ops.LOOKUP_WITH_UPDATE
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)

    # Mean: [(1 + 101) / 3, 0]
    # Variance: [(34^2 + (1-34)^2 + (101-34)^2)/3, 0]
    # Distance: exp(-((34 - 0)^2/2)/ (2 * 2244.6667))
    self.assertAllClose(mean.numpy(), [[34, 0], [34, 0], [34, 0]])
    self.assertAllClose(variance.numpy(),
                        [[2244.6667, 1], [2244.6667, 1], [2244.6667, 1]])
    self.assertAllClose(distance.numpy(), [0.8791941, 0.88577926, 0.6065532])
    self.assertAllClose(cid.numpy(), [0, 0, 0])

    # Switch x and y values of input.
    inputs = [[0, 0], [0, 1], [0, 101]]
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    self.assertAllClose(mean.numpy(), [[0, 34], [0, 34], [0, 34]])
    self.assertAllClose(variance.numpy(),
                        [[1, 2244.6667], [1, 2244.6667], [1, 2244.6667]])
    self.assertAllClose(distance.numpy(), [0.8791941, 0.88577926, 0.6065532])
    self.assertAllClose(cid.numpy(), [0, 0, 0])

    # Lookup without update mode.
    inputs = [[0, 0], [1, 0], [101, 0]]
    mode = dm_ops.LOOKUP_WITHOUT_UPDATE
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    # Returns the same mean and variance as above.
    self.assertAllClose(mean.numpy(), [[0, 34], [0, 34], [0, 34]])
    self.assertAllClose(variance.numpy(),
                        [[1, 2244.6667], [1, 2244.6667], [1, 2244.6667]])
    # [0, 0]: exp(-(((0 - 0)^2/1 + 34^2/2244.6667)/2)/2)
    # [1, 0]: exp(-(((1 - 0)^2/1 + 34^2/2244.6667)/2)/2)
    # [101, 0]: exp(-(((100 - 0)^2/1 + 34^2/2244.6667)/2)/2)
    self.assertAllClose(distance.numpy(), [0.8791941, 0.68471706, 0])
    self.assertAllClose(cid.numpy(), [0, 0, 0])

    # Lookup without grow mode, it's equivalent to update for single cluster.
    inputs = [[10, 0], [40, 0], [70, 0]]
    mode = dm_ops.LOOKUP_WITH_GROW
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    self.assertAllClose(mean.numpy(), [[40, 0], [40, 0], [40, 0]])
    self.assertAllClose(variance.numpy(), [[600, 1], [600, 1], [600, 1]])
    # [10, 0]; exp(-(((10 - 40)^2/600)/2)/2)
    # [40, 0]; exp(-(((40 - 40)^2/600)/2)/2)
    # [70, 0]; exp(-(((70 - 40)^2/600)/2)/2)
    self.assertAllClose(distance.numpy(), [0.6872893, 1, 0.6872893])
    self.assertAllClose(cid.numpy(), [0, 0, 0])

  def testGaussianMemoryLookupWithSingleCluster_3DInput(self):
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=4,
        distance_to_cluster_threshold=0.5,
        bootstrap_steps=0,
        min_variance=1,
        max_num_clusters=1)
    inputs = [[[0, 0], [1, 0]], [[101, 0], [0, 101]]]
    mode = dm_ops.LOOKUP_WITH_UPDATE
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)

    # Mean: [(1 + 101) / 4, 101 / 4]
    self.assertAllClose(
        mean.numpy(),
        [[[25.5, 25.25], [25.5, 25.25]], [[25.5, 25.25], [25.5, 25.25]]])
    self.assertAllClose(variance.numpy(),
                        [[[1900.25, 1912.6875], [1900.25, 1912.6875]],
                         [[1900.25, 1912.6875], [1900.25, 1912.6875]]])
    self.assertAllClose(distance.numpy(),
                        [[0.84460914, 0.85018337], [0.43462682, 0.4336368]])
    self.assertAllClose(cid.numpy(), [[0, 0], [0, 0]])

    mode = dm_ops.LOOKUP_WITH_GROW
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    # Single cluster, do not grow.
    self.assertAllClose(
        mean.numpy(),
        [[[25.5, 25.25], [25.5, 25.25]], [[25.5, 25.25], [25.5, 25.25]]])
    self.assertAllClose(variance.numpy(),
                        [[[1900.25, 1912.6875], [1900.25, 1912.6875]],
                         [[1900.25, 1912.6875], [1900.25, 1912.6875]]])
    self.assertAllClose(distance.numpy(),
                        [[0.84460914, 0.85018337], [0.43462682, 0.4336368]])
    self.assertAllClose(cid.numpy(), [[0, 0], [0, 0]])

  def testGaussianMemoryLookupWithMutiClusterWithGrow(self):
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=3,
        distance_to_cluster_threshold=0.7,
        bootstrap_steps=0,
        min_variance=1,
        max_num_clusters=2)
    inputs = [[0, 0], [1, 0], [101, 0]]
    mode = dm_ops.LOOKUP_WITH_UPDATE
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)

    # Mean: [(1 + 101) / 3, 0]
    # Variance: [(34^2 + (1-34)^2 + (101-34)^2)/3, 0]
    # Distance: exp(-((34 - 0)^2/2)/ (2 * 2244.6667))
    self.assertAllClose(mean.numpy(), [[34, 0], [34, 0], [34, 0]])
    self.assertAllClose(variance.numpy(),
                        [[2244.6667, 1], [2244.6667, 1], [2244.6667, 1]])
    self.assertAllClose(distance.numpy(), [0.8791941, 0.88577926, 0.6065532])
    self.assertAllClose(cid.numpy(), [0, 0, 0])

    mode = dm_ops.LOOKUP_WITH_GROW
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    # A new cluster is formed with single data [101, 0].
    self.assertAllClose(mean.numpy(), [[34, 0], [34, 0], [101, 0]])
    self.assertAllClose(variance.numpy(),
                        [[2244.6667, 1], [2244.6667, 1], [1, 1]])
    self.assertAllClose(distance.numpy(), [0.8791941, 0.88577926, 1])
    self.assertAllClose(cid.numpy(), [0, 0, 1])

  def testGaussianMemoryLookupWithMultiCluster_3DInput(self):
    dm_config = test_util.default_dm_config(
        per_cluster_buffer_size=4,
        distance_to_cluster_threshold=0.5,
        bootstrap_steps=0,
        min_variance=1,
        max_num_clusters=4)
    inputs = [[[0, 0], [1, 0]], [[101, 0], [0, 101]]]
    mode = dm_ops.LOOKUP_WITH_UPDATE
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)

    # Mean: [(1 + 101) / 4, 101 / 4]
    self.assertAllClose(
        mean.numpy(),
        [[[25.5, 25.25], [25.5, 25.25]], [[25.5, 25.25], [25.5, 25.25]]])
    self.assertAllClose(variance.numpy(),
                        [[[1900.25, 1912.6875], [1900.25, 1912.6875]],
                         [[1900.25, 1912.6875], [1900.25, 1912.6875]]])
    self.assertAllClose(distance.numpy(),
                        [[0.84460914, 0.85018337], [0.43462682, 0.4336368]])
    self.assertAllClose(cid.numpy(), [[0, 0], [0, 0]])

    # Lookup with grow, new clusters are formed for [101, 0] and [0, 101].
    mode = dm_ops.LOOKUP_WITH_GROW
    mean, variance, distance, cid = dm_ops.dynamic_gaussian_memory_lookup(
        inputs, mode, dm_config, 'dm_layer', service_address=self._kbs_address)
    self.assertAllClose(mean.numpy(),
                        [[[25.5, 25.25], [25.5, 25.25]], [[101, 0], [0, 101]]])
    self.assertAllClose(
        variance.numpy(),
        [[[1900.25, 1912.6875], [1900.25, 1912.6875]], [[1, 1], [1, 1]]])
    self.assertAllClose(distance.numpy(), [[0.84460914, 0.85018337], [1, 1]])
    self.assertAllClose(cid.numpy(), [[0, 0], [1, 2]])


if __name__ == '__main__':
  tf.test.main()
