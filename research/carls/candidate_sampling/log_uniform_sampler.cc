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

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/candidate_sampling/candidate_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace carls {
namespace candidate_sampling {

using tensorflow::random::SimplePhilox;

// Sample the candidates based on log uniform distribution.
class LogUniformSampler : public CandidateSampler {
 public:
  LogUniformSampler(const CandidateSamplerConfig& config)
      : CandidateSampler(config),
        simple_philox_(&random_),
        log_uniform_config_(
            GetExtensionProtoOrDie<CandidateSamplerConfig,
                                   LogUniformSamplerConfig>(config)) {}

 private:
  absl::Status SampleInternal(
      const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
      int num_samples, std::vector<SampledResult>* results) const override;

  // Samples unique keys from all the positive/negative key set.
  absl::Status SampleUnique(const KnowledgeBank& knowledge_bank,
                            std::vector<absl::string_view> positive_keys,
                            const std::vector<absl::string_view>& all_keys,
                            const int num_sampled,
                            std::vector<SampledResult>* results) const;

  // Allows duplicates in the sampling.
  absl::Status SampleWithReplacement(
      const KnowledgeBank& knowledge_bank,
      std::vector<absl::string_view> positive_keys,
      const std::vector<absl::string_view>& all_keys, const int num_sampled,
      std::vector<SampledResult>* results) const;

  // Returns a random number sampled uniformly in [0, n).
  uint32_t Uniform(uint32_t n) const {
    absl::MutexLock l(&mu_);
    return simple_philox_.Uniform(n);
  }

  // Returns a random float sampled uniformly in [0, 1].
  float RandFloat() const {
    absl::MutexLock l(&mu_);
    return simple_philox_.RandFloat();
  }

  SampledResult BuildSampledResult(const KnowledgeBank& knowledge_bank,
                                   absl::string_view key,
                                   const bool is_positive,
                                   const float expected_count) const {
    SampledResult sampled_result;
    auto result = sampled_result.mutable_negative_sampling_result();
    result->set_key(std::string(key));
    result->set_is_positive(is_positive);
    result->set_expected_count(expected_count);
    auto status = knowledge_bank.Lookup(key, result->mutable_embedding());
    if (!status.ok()) {
      LOG(ERROR) << "Lookup failed for key: " << key
                 << " with error: " << status.message();
    }
    return sampled_result;
  }

  mutable absl::Mutex mu_;
  tensorflow::random::PhiloxRandom random_;
  mutable tensorflow::random::SimplePhilox simple_philox_ ABSL_GUARDED_BY(mu_);
  LogUniformSamplerConfig log_uniform_config_;
};

REGISTER_SAMPLER_FACTORY(LogUniformSamplerConfig,
                         [](const CandidateSamplerConfig& config)
                             -> std::unique_ptr<CandidateSampler> {
                           return std::unique_ptr<CandidateSampler>(
                               new LogUniformSampler(config));
                         });

absl::Status LogUniformSampler::SampleInternal(
    const KnowledgeBank& knowledge_bank, const SampleContext& sample_context,
    int num_samples, std::vector<SampledResult>* results) const {
  if (sample_context.positive_key().empty()) {
    return absl::InvalidArgumentError("Empty positive keys.");
  }
  std::vector<absl::string_view> positive_keys(
      sample_context.positive_key().begin(),
      sample_context.positive_key().end());
  std::vector<absl::string_view> all_keys = knowledge_bank.Keys();
  if (log_uniform_config_.unique()) {
    return SampleUnique(knowledge_bank, positive_keys, all_keys, num_samples,
                        results);
  }
  return SampleWithReplacement(knowledge_bank, positive_keys, all_keys,
                               num_samples, results);
}

absl::Status LogUniformSampler::SampleUnique(
    const KnowledgeBank& knowledge_bank,
    std::vector<absl::string_view> positive_keys,
    const std::vector<absl::string_view>& all_keys, const int num_sampled,
    std::vector<SampledResult>* results) const {
  absl::flat_hash_set<absl::string_view> pos_set(positive_keys.begin(),
                                                 positive_keys.end());
  const size_t range = all_keys.size();
  results->clear();
  results->reserve(num_sampled);

  // Case One: too many positive keys than num_sampled, returns num_sampled
  // randomly sampled positive keys.
  if (positive_keys.size() >= num_sampled) {
    // Randomly choose num_sampled from positive set.
    const float prob = static_cast<float>(num_sampled) / positive_keys.size();
    int size = positive_keys.size();
    for (int i = 0; i < num_sampled; ++i, --size) {
      int index = Uniform(size);
      results->push_back(
          BuildSampledResult(knowledge_bank, positive_keys[index],
                             /*is_positive=*/true, /*expected_count=*/prob));
      // Swap out the selected key.
      std::swap(positive_keys[index], positive_keys[size - 1]);
    }
    return absl::OkStatus();
  }

  // Case Two: positive_keys.size() < num_sampled and num_sampled == range,
  // returns all the available keys. We handle this special case for faster
  // processing by avoiding random sampling.
  if (num_sampled == range) {
    // Choose everything.
    for (size_t i = 0; i < range; ++i) {
      SampledResult sampled_result;
      results->push_back(BuildSampledResult(knowledge_bank, all_keys[i],
                                            pos_set.contains(all_keys[i]),
                                            /*expected_count=*/1.0f));
    }
    return absl::OkStatus();
  }

  if (num_sampled > range) {
    return absl::InternalError(
        "num_samples is larger than the total number of availabe candidates in "
        "the knowledge bank. Potentially caused by the positive keys are not "
        "saved to the knowledge bank.");
  }
  // Case Three: positive_keys.size() < num_sampled < range, sample randomly.
  const float prob = static_cast<float>(num_sampled - pos_set.size()) /
                     static_cast<float>(range - pos_set.size());
  for (size_t i = 0; i < num_sampled; ++i) {
    if (i < positive_keys.size()) {
      results->push_back(BuildSampledResult(knowledge_bank, positive_keys[i],
                                            /*is_positive=*/true,
                                            /*expected_count=*/1));
      continue;
    }
    size_t index = Uniform(range);
    while (pos_set.contains(all_keys[index])) {
      index = (index + 1) % range;
    }
    results->push_back(BuildSampledResult(knowledge_bank, all_keys[index],
                                          /*is_positive=*/false,
                                          /*expected_count=*/prob));
    // Insert into positive keyword set to avoid resampling.
    pos_set.insert(all_keys[index]);
  }
  return absl::OkStatus();
}

absl::Status LogUniformSampler::SampleWithReplacement(
    const KnowledgeBank& knowledge_bank,
    std::vector<absl::string_view> positive_keys,
    const std::vector<absl::string_view>& all_keys, const int num_sampled,
    std::vector<SampledResult>* results) const {
  absl::flat_hash_set<absl::string_view> pos_set(positive_keys.begin(),
                                                 positive_keys.end());
  const size_t range = all_keys.size();
  const float log_range = std::log1p(range);  // Computes log(range + 1).
  results->clear();
  results->reserve(num_sampled);
  for (int i = 0; i < num_sampled; ++i) {
    if (i < positive_keys.size()) {
      results->push_back(BuildSampledResult(knowledge_bank, positive_keys[i],
                                            /*is_positive=*/true,
                                            /*expected_count=*/1));
      continue;
    }
    // The following is based on tensorflow/core/kernels/range_sampler.h
    // Sample an element based on the log-uniform distribution.
    const size_t index =
        (static_cast<size_t>(std::exp(RandFloat() * log_range)) - 1) % range;
    // Probability that the given index is sampled.
    const float prob = (log((index + 2.0) / (index + 1.0))) / log_range;
    // numerically stable version of (1 - (1-p)^num_tries).
    const float expected_count =
        -std::expm1((num_sampled - positive_keys.size()) * std::log1p(-prob));
    results->push_back(BuildSampledResult(knowledge_bank, all_keys[index],
                                          pos_set.contains(all_keys[index]),
                                          expected_count));
  }
  return absl::OkStatus();
}

}  // namespace candidate_sampling
}  // namespace carls
