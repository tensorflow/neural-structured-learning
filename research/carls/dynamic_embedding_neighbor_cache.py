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
"""DynamicEmbedding implementation of NeighborCacheClient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

from neural_structured_learning.research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from neural_structured_learning.research.carls import dynamic_embedding_ops as de_ops
from neural_structured_learning.research.carls import neighbor_cache_client as nb_cache


class DynamicEmbeddingNeighborCache(nb_cache.NeighborCacheClient):
  """Implementation of the NeighborCacheClient with DynamicEmbedding."""

  def __init__(self,
               key_feature_name: typing.Text,
               config: de_config_pb2.DynamicEmbeddingConfig,
               service_address: typing.Text = "",
               timeout_ms: int = -1):
    """Initializes the `NeighborCacheClient` object.

    Args:
      key_feature_name: feature name of the key in the input `tf.Example`
        instances whose value contains neighbor IDs.
      config: A DynamicEmbeddingConfig proto that configs the embedding.
      service_address: The address of a dynamic embedding service. If empty, the
        value passed from --kbs_address flag will be used instead.
      timeout_ms: Timeout millseconds for the connection. If negative, never
        timout.
    """
    self._key_feature_name = key_feature_name
    self._config = config
    self._service_address = service_address
    self._timeout_ms = timeout_ms

  def lookup(self, neighbor_ids):
    """Looks up neighbor state in the neighbor cache.

    Args:
      neighbor_ids: a string Tensor of shape [batch_size] representing the ids
        of a neighborhood.

    Returns:
      Cached state of neighbor examples; `None` if it doesn't exist.
    """
    return de_ops.dynamic_embedding_lookup(
        neighbor_ids,
        self._config,
        self._key_feature_name,
        self._service_address,
        skip_gradient_update=True,
        timeout_ms=self._timeout_ms)

  def update(self, neighbor_ids, neighbor_state):
    """Updates the neighbor cache with the new state of neighbor examples.

    Args:
      neighbor_ids: a string Tensor of shape [batch_size] representing the ids
        of a neighborhood.
      neighbor_state: a Tensor of shape [batch_size, ...] representing newly
        computed neighbor state(e.g. embeddings, logits) that should be stored
        in the neighbor cache.

    Returns:
      A `Tensor` of shape [batch_size, config.embedding_dimension].
    """
    return de_ops.dynamic_embedding_update(neighbor_ids, neighbor_state,
                                           self._config, self._key_feature_name,
                                           self._service_address,
                                           self._timeout_ms)
