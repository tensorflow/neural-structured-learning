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
"""Functions for saving/loading knowledge bank related data."""

from typing import Text

from research.carls import context
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls.kernels import gen_carls_ops


def save_knowledge_bank(output_directory: Text,
                        service_address: Text = '',
                        timeout_ms: int = -1,
                        append_timestamp: bool = True,
                        var_names=None):
  """Saves knowledge bank data to given output directory.

  Each knowldge bank data will be saved in a subdir:
  `%output_directory%/%var_name%`.

  Args:
    output_directory: A string representing the output directory path.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.
    append_timestamp: A boolean variable indicating if a timestamped dir should
      be added when saving the data.
    var_names: A list of strings represent list of variable names with dynamic
      embedding data to be saved. If not specified, save all data.

  Returns:
    Path to the saved file.
  """
  if not output_directory:
    raise ValueError('Empty output_directory.')

  saved_paths = []
  for name, config in context.get_all_collection():
    if var_names and (name not in var_names):
      continue
    resource = gen_carls_ops.dynamic_embedding_manager_resource(
        config.SerializeToString(), name, service_address, timeout_ms)

    saved_path = gen_carls_ops.save_knowledge_bank(
        output_directory, append_timestamp=append_timestamp, handle=resource)
    saved_paths.append(saved_path)

  return saved_paths


def restore_knowledge_bank(config: de_config_pb2.DynamicEmbeddingConfig,
                           var_name: Text,
                           saved_path: Text,
                           service_address: Text = '',
                           timeout_ms: int = -1) -> None:
  """Restores knowledge bank data (`config`, `name`) from given `saved_path`.

  Args:
    config: A DynamicEmbeddingConfig proto that configs the embedding.
    var_name: A unique name for the given embedding.
    saved_path: A string representing the saved embedding data.
    service_address: The address of a dynamic embedding service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.
  """
  resource = gen_carls_ops.dynamic_embedding_manager_resource(
      config.SerializeToString(), var_name, service_address, timeout_ms)

  gen_carls_ops.restore_knowledge_bank(saved_path, resource)
