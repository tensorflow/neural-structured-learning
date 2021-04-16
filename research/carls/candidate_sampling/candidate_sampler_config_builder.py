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
"""A library for building a CandidateSamplerConfig."""

import typing

from research.carls.candidate_sampling import candidate_sampler_config_pb2 as cs_config_pb2


def negative_sampler(unique: bool,
                     algorithm) -> cs_config_pb2.NegativeSamplerConfig:
  """Builds a NegativeSamplerConfig from given input.

  Args:
    unique: a bool indicating if the samples should be unique.
    algorithm: the sampler algorithm defined by NegativeSamplerConfig.Sampler.

  Returns:
    A NegativeSamplerConfig proto.
  """
  if isinstance(algorithm, typing.Text):
    algorithm = cs_config_pb2.NegativeSamplerConfig.Sampler.Value(algorithm)
  if isinstance(algorithm, int):
    if algorithm not in [
        cs_config_pb2.NegativeSamplerConfig.UNIFORM,
        cs_config_pb2.NegativeSamplerConfig.LOG_UNIFORM
    ]:
      raise ValueError('Invalid sampler type.')
  else:
    raise ValueError('Invalid input: %r' % algorithm)

  return cs_config_pb2.NegativeSamplerConfig(unique=unique, sampler=algorithm)


def brute_force_topk_sampler(
    similarity_type) -> cs_config_pb2.BruteForceTopkSamplerConfig:
  """Returns a BruteForceTopkSamplerConfig based on given similarity type.

  Args:
    similarity_type: A string or an int indicating the type of similarity
      defined in carls.candidate_sampling.SimilarityType.

  Returns:
    An instance of BruteForceTopkSamplerConfig if input is valid.

  Raises:
    ValueError: if input is invalid.
  """
  if isinstance(similarity_type, typing.Text):
    similarity_type = cs_config_pb2.SimilarityType.Value(similarity_type)
  if isinstance(similarity_type, int):
    if similarity_type not in [cs_config_pb2.COSINE, cs_config_pb2.DOT_PRODUCT]:
      raise ValueError('Invalid similarity type.')
  else:
    raise ValueError('Invalid input: %r' % similarity_type)

  return cs_config_pb2.BruteForceTopkSamplerConfig(
      similarity_type=similarity_type)


def build_candidate_sampler_config(
    sampler) -> cs_config_pb2.CandidateSamplerConfig:
  """Builds a CandidateSamplerConfig from given sampler.

  Args:
    sampler: an instance of NegativeSamplerConfig or
      BruteForceTopkSamplerConfig.

  Returns:
    A valid CandidateSamplerConfig.

  Raises:
    ValueError if `sampler` is not valid.
  """
  if not (isinstance(sampler, cs_config_pb2.NegativeSamplerConfig) or
          isinstance(sampler, cs_config_pb2.BruteForceTopkSamplerConfig)):
    raise ValueError(
        'sampler must be one of NegativeSamplerConfig or BruteForceTopkSamplerConfig'
    )

  sampler_config = cs_config_pb2.CandidateSamplerConfig()
  sampler_config.extension.Pack(sampler)
  return sampler_config
