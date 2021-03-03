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
"""Utility functions for testing dynamic embedding related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import typing
from absl import flags

from neural_structured_learning.research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from neural_structured_learning.research.carls import kbs_server_helper
from google.protobuf import text_format

FLAGS = flags.FLAGS


def default_de_config(embedding_dimension: int,
                      initial_values: typing.List[float] = None):
  """Creates a default DynamicEmbeddingConfig and starts a local DES server.

  Args:
    embedding_dimension: An positive int specifying embedding dimension.
    initial_values: A list of float with size embedding_dimension if specified.

  Returns:
    A DynamicEmbeddingConfig.
  Raises:
    ValueError: if embedding_dimension is not positive or initial_values is
      given and its size is not embedding_dimension.
  """
  if embedding_dimension <= 0:
    raise ValueError("Invalid embedding_dimension: %d" % embedding_dimension)
  if initial_values is not None and len(initial_values) != embedding_dimension:
    raise ValueError("initial_values's size is not embedding_dimension")
  config = de_config_pb2.DynamicEmbeddingConfig()
  text_format.Parse(
      """
    embedding_dimension: %d
    knowledge_bank_config {
      initializer {
        random_uniform_initializer {
          low: -0.5
          high: 0.5
        }
        use_deterministic_seed: true
      }
      extension {
        [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
      }
    }
  """ % embedding_dimension, config)
  if initial_values:
    default_embed = config.knowledge_bank_config.initializer.default_embedding
    for v in initial_values:
      default_embed.value.append(v)
  return config


def start_kbs_server():
  """Starts a local KBS server and returns its handler."""
  # Starts a local KBS server.
  options = kbs_server_helper.KnowledgeBankServiceOptions(True, -1, 10)
  server = kbs_server_helper.KbsServerHelper(options)
  FLAGS.kbs_address = "localhost:%d" % server.port()
  return server
