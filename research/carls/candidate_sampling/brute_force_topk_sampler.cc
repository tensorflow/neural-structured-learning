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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "research/carls/base/embedding_helper.h"
#include "research/carls/base/top_n.h"
#include "research/carls/candidate_sampling/candidate_sampler.h"
#include "research/carls/candidate_sampling/candidate_sampler_config.pb.h"  // proto to pb
#include "research/carls/embedding.pb.h"  // proto to pb

namespace carls {
namespace candidate_sampling {
namespace {

// Represents an embedding in the knowledge bank, used for top-k comparison.
struct CandidateInfo {
  // The key in the knowledge bank.
  absl::string_view key;

  // The similarity between the activation and the embedding of the `key`.
  float similarity = 0;

  // The embedding of the key. Make a copy to avoid accidental deallocation.
  EmbeddingVectorProto embed;

  CandidateInfo(absl::string_view k, float s) : key(k), similarity(s) {}
};

// Used for the top-k computation.
struct CandidateInfoComparator {
  bool operator()(const CandidateInfo& lhs, const CandidateInfo& rhs) const {
    return lhs.similarity > rhs.similarity;
  }
};

BruteForceTopkSamplerConfig GetTopkConfig(
    const CandidateSamplerConfig& config) {
  return GetExtensionProtoOrDie<CandidateSamplerConfig,
                                BruteForceTopkSamplerConfig>(config);
}

}  // namespace

// A brute-force implementation of the top-k sampler. Each time the Sample()
// method is called, it traverses all the embeddings in a knowledge bank and
// compares their similarities with given input activation.
class BruteForceTopkSampler : public CandidateSampler {
 public:
  BruteForceTopkSampler(const CandidateSamplerConfig& config)
      : CandidateSampler(config), topk_config_(GetTopkConfig(config)) {}

 private:
  absl::Status SampleInternal(
      const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
      int num_samples,
      std::vector<std::pair<absl::string_view, SampledResult>>* results)
      const override;

  const BruteForceTopkSamplerConfig topk_config_;
};

REGISTER_SAMPLER_FACTORY(BruteForceTopkSamplerConfig,
                         [](const CandidateSamplerConfig& config)
                             -> std::unique_ptr<CandidateSampler> {
                           auto topk_config = GetTopkConfig(config);
                           if (topk_config.similarity_type() == UNKNOWN) {
                             LOG(ERROR)
                                 << "Unknown similarity type, cannot create "
                                    "BruteForceTopkSampler.";
                             return nullptr;
                           }
                           return std::unique_ptr<CandidateSampler>(
                               new BruteForceTopkSampler(config));
                         });

absl::Status BruteForceTopkSampler::SampleInternal(
    const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
    int num_samples,
    std::vector<std::pair<absl::string_view, SampledResult>>* results) const {
  if (!sample_context.has_activation()) {
    return absl::InvalidArgumentError("No activation from sample_context.");
  }
  if (knowledge_bank.embedding_dimension() !=
      sample_context.activation().value_size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid embedding dimension from activation, expect ",
                     knowledge_bank.embedding_dimension(), ", got ",
                     sample_context.activation().value_size(), "."));
  }

  std::vector<absl::string_view> all_keys = knowledge_bank.Keys();
  TopN<CandidateInfo, CandidateInfoComparator> topn(num_samples);
  for (auto key : all_keys) {
    EmbeddingVectorProto embed;
    if (!knowledge_bank.Lookup(key, &embed).ok()) {
      continue;
    }
    if (knowledge_bank.embedding_dimension() != embed.value_size()) {
      return absl::InternalError(absl::StrCat(
          "Inconsistent embedding size (", embed.value_size(), " v.s. ",
          sample_context.activation().value_size(), ") for key: ", key));
    }
    float similarity = 0;
    switch (topk_config_.similarity_type()) {
      case DOT_PRODUCT:
        if (ComputeDotProduct(sample_context.activation(), embed,
                              &similarity)) {
          CandidateInfo candidate_info(key, similarity);
          candidate_info.embed = std::move(embed);
          topn.push(std::move(candidate_info));
        }
        break;
      case COSINE:
        if (ComputeCosineSimilarity(sample_context.activation(), embed,
                                    &similarity)) {
          CandidateInfo candidate_info(key, similarity);
          candidate_info.embed = std::move(embed);
          topn.push(std::move(candidate_info));
        }
        break;
      default:
        LOG(FATAL) << "Shouldn't be here. Similarity type: "
                   << topk_config_.similarity_type();
    }
  }

  // Processes results.
  results->clear();
  results->reserve(num_samples);
  std::unique_ptr<std::vector<CandidateInfo>> topn_results(topn.Extract());
  for (auto& candidate_info : *topn_results) {
    SampledResult sampled_result;
    TopkSamplingResult* result = sampled_result.mutable_topk_sampling_result();
    result->set_key(std::string(candidate_info.key));
    result->set_similarity(candidate_info.similarity);
    *(result->mutable_embedding()) = std::move(candidate_info.embed);
    results->push_back({candidate_info.key, std::move(sampled_result)});
  }
  return absl::OkStatus();
}

}  // namespace candidate_sampling
}  // namespace carls
