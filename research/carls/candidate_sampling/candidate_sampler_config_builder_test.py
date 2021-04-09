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
"""Tests for candidate_sampler_config_builder."""

from research.carls.candidate_sampling import candidate_sampler_config_builder as cs_config_builder
from research.carls.candidate_sampling import candidate_sampler_config_pb2 as cs_config_pb2

import tensorflow as tf


class CandidateSamplerConfigBuilderTest(tf.test.TestCase):

  def test_log_uniform_sampler(self):
    self.assertProtoEquals("""
      unique: true
    """, cs_config_builder.log_uniform_sampler(True))
    self.assertProtoEquals("""
      unique: false
    """, cs_config_builder.log_uniform_sampler(False))

  def test_brute_force_topk_sampler_success(self):
    self.assertProtoEquals("""
      similarity_type: COSINE
    """, cs_config_builder.brute_force_topk_sampler('COSINE'))
    self.assertProtoEquals(
        """
      similarity_type: COSINE
    """, cs_config_builder.brute_force_topk_sampler(cs_config_pb2.COSINE))
    self.assertProtoEquals(
        """
      similarity_type: DOT_PRODUCT
    """, cs_config_builder.brute_force_topk_sampler('DOT_PRODUCT'))
    self.assertProtoEquals(
        """
      similarity_type: DOT_PRODUCT
    """, cs_config_builder.brute_force_topk_sampler(cs_config_pb2.DOT_PRODUCT))

  def test_brute_force_topk_sampler_failed(self):
    with self.assertRaises(ValueError):
      cs_config_builder.brute_force_topk_sampler(cs_config_pb2.UNKNOWN)
    with self.assertRaises(ValueError):
      cs_config_builder.brute_force_topk_sampler('Unknown type string')
    with self.assertRaises(ValueError):
      cs_config_builder.brute_force_topk_sampler(cs_config_pb2.SampleContext())
    with self.assertRaises(ValueError):
      cs_config_builder.brute_force_topk_sampler(999)

  def test_build_candidate_sampler_config_success(self):
    self.assertProtoEquals(
        """
        extension {
          [type.googleapis.com/carls.candidate_sampling.BruteForceTopkSamplerConfig] {
            similarity_type: COSINE
          }
        }
    """,
        cs_config_builder.build_candidate_sampler_config(
            cs_config_builder.brute_force_topk_sampler('COSINE')))

    self.assertProtoEquals(
        """
        extension {
          [type.googleapis.com/carls.candidate_sampling.LogUniformSamplerConfig] {
            unique: true
          }
        }
    """,
        cs_config_builder.build_candidate_sampler_config(
            cs_config_builder.log_uniform_sampler(True)))

  def test_build_candidate_sampler_config_failed(self):
    with self.assertRaises(ValueError):
      cs_config_builder.build_candidate_sampler_config(100)
    with self.assertRaises(ValueError):
      cs_config_builder.build_candidate_sampler_config('invalid')


if __name__ == '__main__':
  tf.test.main()
