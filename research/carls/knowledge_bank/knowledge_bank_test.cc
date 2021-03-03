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

#include "research/carls/knowledge_bank/knowledge_bank.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/initializer_helper.h"

namespace carls {
namespace {

using ::testing::Eq;
using ::testing::EqualsProto;

class FakeEmbedding : public KnowledgeBank {
 public:
  explicit FakeEmbedding(const KnowledgeBankConfig& config, int dimension)
      : KnowledgeBank(config, dimension) {}

  absl::Status Lookup(const absl::string_view key,
                      EmbeddingVectorProto* result) const override {
    CHECK(result != nullptr);
    if (!data_table_.contains(key)) {
      return absl::InvalidArgumentError("Data not found");
    }
    *result = data_table_.find(key)->second;
    return absl::OkStatus();
  }

  absl::Status LookupWithUpdate(const absl::string_view key,
                                EmbeddingVectorProto* result) override {
    CHECK(result != nullptr);
    if (!data_table_.contains(key)) {
      data_table_[key] =
          InitializeEmbedding(embedding_dimension(), config().initializer());
    }
    *result = data_table_.find(key)->second;
    return absl::OkStatus();
  }

  absl::Status Update(const absl::string_view key,
                      const EmbeddingVectorProto& value) override {
    data_table_[key] = value;
    return absl::OkStatus();
  }

  absl::flat_hash_map<std::string, EmbeddingVectorProto> data_table_;
};

REGISTER_KNOWLEDGE_BANK_FACTORY(KnowledgeBankConfig,
                                [](const KnowledgeBankConfig& config,
                                   int dimension)
                                    -> std::unique_ptr<KnowledgeBank> {
                                  return std::unique_ptr<KnowledgeBank>(
                                      new FakeEmbedding(config, dimension));
                                });

}  // namespace

class KnowledgeBankTest : public ::testing::Test {
 protected:
  KnowledgeBankTest() {}

  std::unique_ptr<KnowledgeBank> CreateDefaultStore(int embedding_dimension) {
    KnowledgeBankConfig config;
    config.mutable_initializer()->mutable_zero_initializer();
    return KnowledgeBankFactory::Make(config, embedding_dimension);
  }
};

TEST_F(KnowledgeBankTest, Basic) {
  auto store = CreateDefaultStore(10);

  EXPECT_EQ(10, store->embedding_dimension());
}

TEST_F(KnowledgeBankTest, LookupAndUpdate) {
  auto store = CreateDefaultStore(2);
  EmbeddingInitializer initializer;
  initializer.mutable_zero_initializer();
  EmbeddingVectorProto value = InitializeEmbedding(2, initializer);
  EXPECT_OK(store->Update("key1", value));

  EmbeddingVectorProto result;
  EXPECT_OK(store->Lookup("key1", &result));
  EXPECT_THAT(result, EqualsProto(R"(
                value: 0 value: 0
              )"));
}

TEST_F(KnowledgeBankTest, BatchLookupAndUpdate) {
  auto store = CreateDefaultStore(2);
  EmbeddingInitializer initializer;
  initializer.mutable_zero_initializer();
  EmbeddingVectorProto value1 = InitializeEmbedding(2, initializer);
  EmbeddingVectorProto value2 = InitializeEmbedding(2, initializer);
  EXPECT_THAT(
      store->BatchUpdate({"key1", "key2"}, {value1, value2}),
      Eq(std::vector<absl::Status>{absl::OkStatus(), absl::OkStatus()}));

  std::vector<absl::variant<EmbeddingVectorProto, std::string>> value_or_errors;
  store->BatchLookup({"key1", "key2", "key3"}, &value_or_errors);
  ASSERT_EQ(3, value_or_errors.size());
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(
        absl::holds_alternative<EmbeddingVectorProto>(value_or_errors[i]));
    EXPECT_THAT(absl::get<EmbeddingVectorProto>(value_or_errors[i]),
                EqualsProto(R"(
                  value: 0 value: 0
                )"));
  }
  ASSERT_TRUE(absl::holds_alternative<std::string>(value_or_errors[2]));
  EXPECT_EQ("Data not found", absl::get<std::string>(value_or_errors[2]));
}

TEST_F(KnowledgeBankTest, BatchLookupWithUpdate) {
  auto store = CreateDefaultStore(2);

  std::vector<absl::variant<EmbeddingVectorProto, std::string>> value_or_errors;
  store->BatchLookupWithUpdate({"key1", "key2", "key3"}, &value_or_errors);
  ASSERT_EQ(3, value_or_errors.size());
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(
        absl::holds_alternative<EmbeddingVectorProto>(value_or_errors[i]));
    EXPECT_THAT(absl::get<EmbeddingVectorProto>(value_or_errors[i]),
                EqualsProto(R"(
                  value: 0 value: 0
                )"));
  }
}

}  // namespace carls
