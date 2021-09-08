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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_CANDIDATE_SAMPLING_CANDIDATE_SAMPLER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_CANDIDATE_SAMPLING_CANDIDATE_SAMPLER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "research/carls/base/proto_factory.h"
#include "research/carls/candidate_sampling/candidate_sampler_config.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/knowledge_bank.h"

namespace carls {
namespace candidate_sampling {

// Macro for registering a candidate sampler implementation.
#define REGISTER_SAMPLER_FACTORY(proto_type, factory_type)                   \
  REGISTER_CARLS_FACTORY_0(proto_type, factory_type, CandidateSamplerConfig, \
                           CandidateSampler)

// The base class for a candidate sampler.
// It is responsible for working with a knowledge_bank to construct proper
// data structures for efficient candidate sampling.
class CandidateSampler {
 public:
  virtual ~CandidateSampler();

  // The public interface for generating samples from given sample_context from
  // knowledge bank.
  // NOTE: This interface returns a vector of
  // std::pair<absl::string_view, SampledResult> instead of simply SampledResult
  // to get around a protobuf compatibility issue that the key of SampledResult
  // could mess up with each other when copying to a std::vector<SampledResult>.
  absl::Status Sample(
      const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
      int num_samples,
      std::vector<std::pair<absl::string_view, SampledResult>>* results) const;

  // Adds or update a candidate for sampling.
  virtual absl::Status InsertOrUpdate(absl::string_view key,
                                      const EmbeddingVectorProto& embedding);

  // Returns the number of candidates currently available for sampling.
  virtual int NumOfCandidates();

 protected:
  CandidateSampler(const CandidateSamplerConfig& config);

  // The internal implementation of the Sample() method. The subclass can assume
  // the inputs have been checked (num_samples > 0 and results != nullptr).
  virtual absl::Status SampleInternal(
      const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
      int num_samples,
      std::vector<std::pair<absl::string_view, SampledResult>>* results)
      const = 0;

  const CandidateSamplerConfig config_;
};

REGISTER_CARLS_BASE_CLASS_0(CandidateSamplerConfig, CandidateSampler,
                            SamplerFactory);

}  // namespace candidate_sampling
}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_CANDIDATE_SAMPLING_CANDIDATE_SAMPLER_H_
