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

#include "absl/status/status.h"

namespace carls {
namespace candidate_sampling {

CandidateSampler::CandidateSampler(const CandidateSamplerConfig& config)
    : config_(config) {}

CandidateSampler::~CandidateSampler() {}

absl::Status CandidateSampler::Sample(
    const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
    int num_samples, std::vector<SampledResult>* results) const {
  if (results == nullptr) {
    return absl::InvalidArgumentError("Null input.");
  }
  if (num_samples <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid num_samples: ", num_samples));
  }
  return SampleInternal(knowledge_bank, sample_context, num_samples, results);
}

absl::Status CandidateSampler::InsertOrUpdate(
    absl::string_view key, const EmbeddingVectorProto& embedding) {
  LOG(FATAL) << "Method is not implemented, possibly because it is not needed.";
}

int CandidateSampler::NumOfCandidates() {
  LOG(FATAL) << "Method is not implemented, possibly because it is not needed.";
}

}  // namespace candidate_sampling
}  // namespace carls
