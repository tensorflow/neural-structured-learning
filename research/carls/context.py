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
"""Global context for knowledge bank operations."""

import threading
from typing import Text

from research.carls import dynamic_embedding_config_pb2 as de_config_pb2

# A map from variable name to DynamicEmbeddingConfig.
_knowledge_bank_collections = {}
_lock = threading.Lock()


def add_to_collection(name: Text, config: de_config_pb2.DynamicEmbeddingConfig):
  """Adds given (name, config) pair to global collectionss.

  Args:
    name: A string denoting the variable name.
    config: An instance of DynamicEmbeddingConfig.

  Raises:
    TypeError: Invalid input.
    ValueError: Name is empty, or a different config is added for an existing
    variable.
  """
  if not name:
    raise ValueError("Empty name.")
  if not isinstance(config, de_config_pb2.DynamicEmbeddingConfig):
    raise TypeError("Config is not an instance of DynamicEmbeddingConfig.")
  if name in _knowledge_bank_collections.keys():
    existing_config = _knowledge_bank_collections[name]
    if config.SerializeToString() != existing_config.SerializeToString():
      raise ValueError(
          "Adding a new config for the same var name is not allowed, existing:"
          " %r, new: %r." % (existing_config, config))

  with _lock:
    _knowledge_bank_collections[name] = de_config_pb2.DynamicEmbeddingConfig()
    _knowledge_bank_collections[name].CopyFrom(config)


def get_all_collection():
  """Returns a list of all (name, config) pairs."""
  with _lock:
    return [(key, value) for key, value in _knowledge_bank_collections.items()]


def clear_all_collection():
  """Clears existing all (name, config) pairs."""
  with _lock:
    _knowledge_bank_collections.clear()
