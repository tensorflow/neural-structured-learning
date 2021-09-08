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
#include "absl/strings/string_view.h"
#include "research/carls/candidate_sampling/candidate_sampler.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/initializer_helper.h"
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace candidate_sampling {
namespace {

class FakeEmbedding : public KnowledgeBank {
 public:
  explicit FakeEmbedding(const KnowledgeBankConfig& config, int dimension)
      : KnowledgeBank(config, dimension) {}

  absl::Status Lookup(const absl::string_view key,
                      EmbeddingVectorProto* result) const override {
    CHECK(result != nullptr);
    std::string str_key(key);
    if (!data_table_.embedding_table().contains(str_key)) {
      return absl::InvalidArgumentError("Data not found");
    }
    *result = data_table_.embedding_table().find(str_key)->second;
    return absl::OkStatus();
  }

  absl::Status LookupWithUpdate(const absl::string_view key,
                                EmbeddingVectorProto* result) override {
    CHECK(result != nullptr);
    std::string str_key(key);
    if (!data_table_.embedding_table().contains(str_key)) {
      (*data_table_.mutable_embedding_table())[str_key] =
          InitializeEmbedding(embedding_dimension(), config().initializer());
      keys_.push_back(data_table_.embedding_table().find(str_key)->first);
    }
    *result = data_table_.embedding_table().find(str_key)->second;
    return absl::OkStatus();
  }

  absl::Status Update(const absl::string_view key,
                      const EmbeddingVectorProto& value) override {
    std::string str_key(key);
    if (!data_table_.embedding_table().contains(str_key)) {
      (*data_table_.mutable_embedding_table())[str_key] = value;
      keys_.push_back(data_table_.embedding_table().find(str_key)->first);
    } else {
      data_table_.mutable_embedding_table()->at(str_key) = value;
    }
    return absl::OkStatus();
  }

  absl::Status ExportInternal(const std::string& dir,
                              std::string* exported_path) override {
    *exported_path = "fake_checkpoint";
    return absl::OkStatus();
  }

  absl::Status ImportInternal(const std::string& saved_path) override {
    return absl::OkStatus();
  }

  size_t Size() const override { return data_table_.embedding_table_size(); }

  std::vector<absl::string_view> Keys() const { return keys_; }

  bool Contains(absl::string_view key) const { return true; }

 private:
  InProtoKnowledgeBankConfig::EmbeddingData data_table_;
  std::vector<absl::string_view> keys_;
};

REGISTER_KNOWLEDGE_BANK_FACTORY(KnowledgeBankConfig,
                                [](const KnowledgeBankConfig& config,
                                   int dimension)
                                    -> std::unique_ptr<KnowledgeBank> {
                                  return std::unique_ptr<KnowledgeBank>(
                                      new FakeEmbedding(config, dimension));
                                });

}  // namespace

class BruteForceTopkSamplerTest : public ::testing::Test {
 protected:
  BruteForceTopkSamplerTest() = default;

  std::unique_ptr<CandidateSampler> CreateSampler(SimilarityType type) {
    CandidateSamplerConfig sampler_config;
    BruteForceTopkSamplerConfig bf_sampler_config;
    bf_sampler_config.set_similarity_type(type);
    sampler_config.mutable_extension()->PackFrom(bf_sampler_config);
    return SamplerFactory::Make(sampler_config);
  }

  std::unique_ptr<KnowledgeBank> CreateKnowledgeBank(int embedding_dimension) {
    KnowledgeBankConfig config;
    config.mutable_initializer()->mutable_zero_initializer();
    return KnowledgeBankFactory::Make(config, embedding_dimension);
  }
};

TEST_F(BruteForceTopkSamplerTest, Create) {
  EXPECT_EQ(nullptr, CreateSampler(UNKNOWN));
  EXPECT_NE(nullptr, CreateSampler(COSINE));
  EXPECT_NE(nullptr, CreateSampler(DOT_PRODUCT));
}

TEST_F(BruteForceTopkSamplerTest, InvalidInput) {
  auto sampler = CreateSampler(COSINE);
  auto knowledge_bank = CreateKnowledgeBank(2);
  SampleContext context;
  std::vector<std::pair<absl::string_view, SampledResult>> results;

  // Empty context.
  EXPECT_ERROR_EQ(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/1, &results),
      "No activation from sample_context.");
  context.mutable_activation()->add_value(1.0);
  // NULL input.
  EXPECT_ERROR_EQ(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/1, nullptr),
      "Null input.");
  // Invalid num_samples.
  EXPECT_ERROR_EQ(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/0, &results),
      "Invalid num_samples: 0");
  // Inconsitent embedding dimension.
  EXPECT_ERROR_EQ(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/1, &results),
      "Invalid embedding dimension from activation, expect 2, got 1.");
}

TEST_F(BruteForceTopkSamplerTest, CosineSimilarity) {
  auto sampler = CreateSampler(COSINE);
  auto knowledge_bank = CreateKnowledgeBank(2);
  ASSERT_OK(knowledge_bank->Update(
      "key1", ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb")));

  ASSERT_OK(knowledge_bank->Update(
      "key2", ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb")));

  SampleContext context;
  context.mutable_activation()->add_value(1);
  context.mutable_activation()->add_value(2);

  std::vector<std::pair<absl::string_view, SampledResult>> results;
  ASSERT_OK(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/1, &results));
  ASSERT_EQ(1, results.size());
  EXPECT_THAT(results[0].second, EqualsProto<SampledResult>(R"pb(
                topk_sampling_result {
                  key: "key1"
                  embedding { value: 1 value: 2 }
                  similarity: 1
                }
              )pb"));
}

TEST_F(BruteForceTopkSamplerTest, DotProtudctSimilarity) {
  auto sampler = CreateSampler(DOT_PRODUCT);
  auto knowledge_bank = CreateKnowledgeBank(2);
  ASSERT_OK(knowledge_bank->Update(
      "key1", ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb")));

  ASSERT_OK(knowledge_bank->Update(
      "key2", ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb")));

  SampleContext context;
  context.mutable_activation()->add_value(1);
  context.mutable_activation()->add_value(2);

  std::vector<std::pair<absl::string_view, SampledResult>> results;
  ASSERT_OK(
      sampler->Sample(*knowledge_bank, context, /*num_samples=*/1, &results));
  ASSERT_EQ(1, results.size());
  EXPECT_THAT(results[0].second, EqualsProto<SampledResult>(R"pb(
                topk_sampling_result {
                  key: "key2"
                  embedding { value: 3 value: 4 }
                  similarity: 11
                }
              )pb"));
}

}  // namespace candidate_sampling
}  // namespace carls
