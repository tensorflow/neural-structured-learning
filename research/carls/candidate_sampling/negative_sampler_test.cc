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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/candidate_sampling/candidate_sampler.h"
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace candidate_sampling {
namespace {

class FakeKnowledgeBank : public KnowledgeBank {
 public:
  explicit FakeKnowledgeBank(const int num_keys)
      : KnowledgeBank(
            []() -> KnowledgeBankConfig {
              KnowledgeBankConfig config;
              config.mutable_initializer()->mutable_zero_initializer();
              return config;
            }(),
            /*embedding_dimension=*/1) {
    str_keys_.reserve(num_keys);
    for (int i = 0; i < num_keys; ++i) {
      str_keys_.push_back(absl::StrCat("key", i));
      keys_.push_back(str_keys_.back());
    }
  }

  size_t Size() const override { return keys_.size(); }

  std::vector<absl::string_view> Keys() const { return keys_; }

  // None of the following methods are used.
  absl::Status Lookup(const absl::string_view key,
                      EmbeddingVectorProto* result) const override {
    return absl::OkStatus();
  }

  absl::Status LookupWithUpdate(const absl::string_view key,
                                EmbeddingVectorProto* result) override {
    return absl::OkStatus();
  }

  absl::Status Update(const absl::string_view key,
                      const EmbeddingVectorProto& value) override {
    return absl::OkStatus();
  }

  absl::Status ExportInternal(const std::string& dir,
                              std::string* exported_path) override {
    return absl::OkStatus();
  }

  absl::Status ImportInternal(const std::string& saved_path) override {
    return absl::OkStatus();
  }

  // Never called.
  bool Contains(absl::string_view key) const { return true; }

 private:
  std::vector<std::string> str_keys_;
  std::vector<absl::string_view> keys_;
};

}  // namespace

class NegativeSamplerTest : public ::testing::Test {
 protected:
  NegativeSamplerTest() {}

  std::unique_ptr<CandidateSampler> CreateSampler(
      bool unique, NegativeSamplerConfig::Sampler sampler) {
    CandidateSamplerConfig sampler_config;
    NegativeSamplerConfig ng_sampler_config;
    ng_sampler_config.set_unique(unique);
    ng_sampler_config.set_sampler(sampler);
    sampler_config.mutable_extension()->PackFrom(ng_sampler_config);
    return SamplerFactory::Make(sampler_config);
  }
};

TEST_F(NegativeSamplerTest, Create) {
  ASSERT_TRUE(CreateSampler(true, NegativeSamplerConfig::UNKNOWN) == nullptr);
  ASSERT_TRUE(CreateSampler(true, NegativeSamplerConfig::UNIFORM) != nullptr);
  ASSERT_TRUE(CreateSampler(true, NegativeSamplerConfig::LOG_UNIFORM) !=
              nullptr);
}

TEST_F(NegativeSamplerTest, SampingWithReplacement) {
  const auto algorithms = std::vector<NegativeSamplerConfig::Sampler>{
      NegativeSamplerConfig::LOG_UNIFORM, NegativeSamplerConfig::UNIFORM};
  for (const auto algorithm : algorithms) {
    auto sampler = CreateSampler(false, algorithm);
    SampleContext context;
    context.add_positive_key("key0");
    context.add_positive_key("key1");
    std::vector<std::pair<absl::string_view, SampledResult>> results;

    // Same number of positives and num_samples = num_total_keys.
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/2), context,
                              /*num_samples=*/2, &results));
    ASSERT_EQ(2, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_TRUE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());

    // num_pos = num_total_keys (3) > num_samples (2).
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/3), context,
                              /*num_samples=*/2, &results));
    ASSERT_EQ(2, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_TRUE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());

    //  num_pos < num_total_keys = num_samples
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/3), context,
                              /*num_samples=*/3, &results));
    ASSERT_EQ(3, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_TRUE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());
    // The third one is randomly chosen so we only know the prob.
    if (algorithm == NegativeSamplerConfig::LOG_UNIFORM) {
      EXPECT_FLOAT_EQ(
          0.5, results[2].second.negative_sampling_result().expected_count());
    } else {  // uniform sampler.
      EXPECT_FLOAT_EQ(
          1.0 / 3.0,
          results[2].second.negative_sampling_result().expected_count());
    }
  }
}

TEST_F(NegativeSamplerTest, UniqueSamping) {
  const auto algorithms = std::vector<NegativeSamplerConfig::Sampler>{
      NegativeSamplerConfig::LOG_UNIFORM, NegativeSamplerConfig::UNIFORM};
  for (const auto algorithm : algorithms) {
    auto sampler = CreateSampler(true, algorithm);
    SampleContext context;
    context.add_positive_key("key0");
    context.add_positive_key("key1");
    std::vector<std::pair<absl::string_view, SampledResult>> results;

    // Same number of positives and num_samples = num_total_keys.
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/2), context,
                              /*num_samples=*/2, &results));
    ASSERT_EQ(2, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_TRUE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());

    // num_pos (2) < num_total_keys = num_samples (3)
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/3), context,
                              /*num_samples=*/3, &results));
    ASSERT_EQ(3, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_TRUE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FALSE(results[2].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[2].second.negative_sampling_result().expected_count());

    // num_pos (1) < num_total_keys = num_samples (3)
    context.clear_positive_key();
    context.add_positive_key("key0");
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/3), context,
                              /*num_samples=*/3, &results));
    ASSERT_EQ(3, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_FALSE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FALSE(results[2].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[1].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        1, results[2].second.negative_sampling_result().expected_count());

    // num_pos (1) < num_samples (3) < num_total_keys (5)
    ASSERT_OK(sampler->Sample(FakeKnowledgeBank(/*num_keys=*/5), context,
                              /*num_samples=*/3, &results));
    ASSERT_EQ(3, results.size());
    EXPECT_TRUE(results[0].second.negative_sampling_result().is_positive());
    EXPECT_FALSE(results[1].second.negative_sampling_result().is_positive());
    EXPECT_FALSE(results[2].second.negative_sampling_result().is_positive());
    EXPECT_FLOAT_EQ(
        1, results[0].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        0.5, results[1].second.negative_sampling_result().expected_count());
    EXPECT_FLOAT_EQ(
        0.5, results[2].second.negative_sampling_result().expected_count());
  }
}

}  // namespace candidate_sampling
}  // namespace carls
