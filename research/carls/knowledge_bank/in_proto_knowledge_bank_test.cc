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
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/knowledge_bank.h"

namespace carls {

using ::testing::EqualsProto;

class InProtoKnowledgeBankTest : public ::testing::Test {
 protected:
  InProtoKnowledgeBankTest() {}

  std::unique_ptr<KnowledgeBank> CreateDefaultStore(int embedding_dimension) {
    KnowledgeBankConfig config;
    config.mutable_initializer()->mutable_zero_initializer();
    InProtoKnowledgeBankConfig in_proto_config;
    config.mutable_extension()->PackFrom(in_proto_config);
    return KnowledgeBankFactory::Make(config, embedding_dimension);
  }
};

TEST_F(InProtoKnowledgeBankTest, LookupAndUpdate) {
  auto store = CreateDefaultStore(2);
  EmbeddingVectorProto value;
  value.add_value(1.0f);
  value.add_value(2.0f);
  EXPECT_OK(store->Update("key1", value));

  EmbeddingVectorProto result;
  EXPECT_OK(store->Lookup("key1", &result));
  EXPECT_THAT(result, EqualsProto(R"(
                value: 1 value: 2
              )"));

  EXPECT_FALSE(store->Lookup("key2", &result).ok());
}

TEST_F(InProtoKnowledgeBankTest, LookupWithUpdate) {
  auto store = CreateDefaultStore(2);
  EmbeddingVectorProto result;
  ASSERT_OK(store->LookupWithUpdate("key1", &result));
  EXPECT_THAT(result, EqualsProto(R"(
                tag: "key1" value: 0 value: 0 weight: 1
              )"));

  // Checks that weight is incremented by 1.
  ASSERT_OK(store->LookupWithUpdate("key1", &result));
  EXPECT_THAT(result, EqualsProto(R"(
                tag: "key1" value: 0 value: 0 weight: 2
              )"));
}

}  // namespace carls
