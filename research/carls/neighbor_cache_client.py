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
"""Abstract client class to manage cached state for neighbor examples.

This abstract class will be implemented as a client to lookup and update values
from a cache service.
"""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class NeighborCacheClient(object):
  """Abstract client class to manage cached state for neighbor examples.

  This abstract class will be implemented as a client to lookup and update
  values from a cache service.
  """

  def __init__(self, key_feature_name):
    """Initializes the `NeighborCacheClient` object.

    Args:
      key_feature_name: feature name of the key in the input `tf.Example`
        instances whose value contains neighbor IDs.
    """
    self._key_feature_name = key_feature_name

  @abc.abstractmethod
  def lookup(self, neighbor_ids):
    """Looks up neighbor state in the neighbor cache.

    Args:
      neighbor_ids: a string Tensor of shape [batch_size] representing the ids
        of a neighborhood.

    Returns:
      Cached state of neighbor examples; `None` if it doesn't exist.
    """
    # TODO(thunderfyc): provide decoding utils for None in batched input.
    raise NotImplementedError

  @abc.abstractmethod
  def update(self, neighbor_ids, neighbor_state):
    """Updates the neighbor cache with the new state of neighbor examples.

    Args:
      neighbor_ids: a string Tensor of shape [batch_size] representing the ids
        of a neighborhood.
      neighbor_state: a Tensor of shape [batch_size, ...] representing newly
        computed neighbor state(e.g. embeddings, logits) that should be stored
        in the neighbor cache.
    """
    raise NotImplementedError

  @property
  def key_feature_name(self):
    """Returns the feature name of the key."""
    return self._key_feature_name
