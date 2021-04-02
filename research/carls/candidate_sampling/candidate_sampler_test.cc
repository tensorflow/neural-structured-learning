/*Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "research/carls/candidate_sampling/candidate_sampler.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace carls {
namespace candidate_sampling {
namespace {

class FakeSampler : public CandidateSampler {
 public:
  FakeSampler(const CandidateSamplerConfig& config)
      : CandidateSampler(config) {}

  absl::Status SampleInternal(
      const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
      int num_samples, std::vector<SampledResult>* results) const override {
    return absl::OkStatus();
  }

  int NumOfCandidates() override { return 1; }

  absl::Status InsertOrUpdate(absl::string_view key,
                              const EmbeddingVectorProto& embedding) override {
    return absl::OkStatus();
  }
};

REGISTER_SAMPLER_FACTORY(CandidateSamplerConfig,
                         [](const CandidateSamplerConfig& config)
                             -> std::unique_ptr<CandidateSampler> {
                           return std::unique_ptr<CandidateSampler>(
                               new FakeSampler(config));
                         });

}  // namespace

class CandidateSamplerTest : public ::testing::Test {
 protected:
  CandidateSamplerTest() {}

  std::unique_ptr<CandidateSampler> CreateDefaultSampler() {
    CandidateSamplerConfig config;
    return SamplerFactory::Make(config);
  }
};

TEST_F(CandidateSamplerTest, Create) {
  auto sampler = CreateDefaultSampler();
  ASSERT_NE(sampler, nullptr);
}

}  // namespace candidate_sampling
}  // namespace carls
