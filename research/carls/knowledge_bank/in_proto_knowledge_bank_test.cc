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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/knowledge_bank.h"
#include "research/carls/testing/test_helper.h"

namespace carls {

using ::testing::TempDir;

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
  EXPECT_TRUE(store->Update("key1", value).ok());

  EmbeddingVectorProto result;
  EXPECT_TRUE(store->Lookup("key1", &result).ok());
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                value: 1 value: 2
              )pb"));

  EXPECT_FALSE(store->Lookup("key2", &result).ok());

  // Checks size and keys of embedding.
  EXPECT_EQ(1, store->Size());
  ASSERT_EQ(1, store->Keys().size());
  EXPECT_EQ("key1", store->Keys()[0]);
}

TEST_F(InProtoKnowledgeBankTest, BatchLookupAndUpdate) {
  auto store = CreateDefaultStore(2);

  // Initialize a batch of keys and values.
  const int batch_size = 100;
  std::vector<std::string> str_keys;
  str_keys.reserve(batch_size);
  std::vector<EmbeddingVectorProto> values;
  values.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    str_keys.push_back(absl::StrCat("key", i));
    EmbeddingVectorProto value;
    value.add_value(i);
    value.add_value(i + 1.0f);
    values.push_back(std::move(value));
  }
  std::vector<absl::string_view> keys(str_keys.begin(), str_keys.end());

  // Checks the BatchUpdate and BatchLookup.
  auto statuses = store->BatchUpdate(keys, values);
  for (const auto& status : statuses) {
    ASSERT_TRUE(status.ok());
  }

  // Checks the BatchLookup results.
  std::vector<absl::variant<EmbeddingVectorProto, std::string>> results;
  store->BatchLookup(keys, &results);
  ASSERT_EQ(batch_size, results.size());
  int i = 0;
  for (const auto& result : results) {
    ASSERT_TRUE(absl::holds_alternative<EmbeddingVectorProto>(result));
    auto proto = absl::get<EmbeddingVectorProto>(result);
    EXPECT_FLOAT_EQ(i, proto.value(0));
    EXPECT_FLOAT_EQ(++i, proto.value(1));
  }

  // Checks size and keys of embedding.
  EXPECT_EQ(batch_size, store->Size());
  ASSERT_EQ(batch_size, store->Keys().size());
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(absl::StrCat("key", i), store->Keys()[i]);
  }
}

TEST_F(InProtoKnowledgeBankTest, LookupWithUpdate) {
  auto store = CreateDefaultStore(2);
  EmbeddingVectorProto result;
  ASSERT_TRUE(store->LookupWithUpdate("key1", &result).ok());
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key1"
                value: 0
                value: 0
                weight: 1
              )pb"));

  // Checks that weight is incremented by 1.
  ASSERT_TRUE(store->LookupWithUpdate("key1", &result).ok());
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key1"
                value: 0
                value: 0
                weight: 2
              )pb"));

  // Checks size and keys of embedding.
  EXPECT_EQ(1, store->Size());
  ASSERT_EQ(1, store->Keys().size());
  EXPECT_EQ("key1", store->Keys()[0]);
}

TEST_F(InProtoKnowledgeBankTest, BatchLookupWithUpdate) {
  auto store = CreateDefaultStore(2);

  // Initialize a batch of keys.
  const int batch_size = 100;
  std::vector<std::string> str_keys;
  str_keys.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    str_keys.push_back(absl::StrCat("key", i));
  }
  std::vector<absl::string_view> keys(str_keys.begin(), str_keys.end());

  // Checks the BatchLookupWithUpdate returns the zero initialized values.
  std::vector<absl::variant<EmbeddingVectorProto, std::string>>
          results;
  store->BatchLookupWithUpdate(keys, &results);
  int i = 0;
  for (const auto& result : results) {
    ASSERT_TRUE(absl::holds_alternative<EmbeddingVectorProto>(result));
    auto proto = absl::get<EmbeddingVectorProto>(result);
    EXPECT_EQ(absl::StrCat("key", i++), proto.tag());
    EXPECT_FLOAT_EQ(0, proto.value(0));
    EXPECT_FLOAT_EQ(0, proto.value(1));
    EXPECT_FLOAT_EQ(1, proto.weight());
  }

  // Checks size and keys of embedding.
  EXPECT_EQ(batch_size, store->Size());
  ASSERT_EQ(batch_size, store->Keys().size());
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(absl::StrCat("key", i), store->Keys()[i]);
  }
}

TEST_F(InProtoKnowledgeBankTest, Export) {
  auto store = CreateDefaultStore(2);

  // Even the time changes, the length should always be the same.
  std::string exported_path;
  ASSERT_TRUE(store->Export(TempDir(), "", &exported_path).ok());
  EXPECT_EQ(JoinPath(TempDir(), "embedding_store_meta_data.pbtxt"),
            exported_path);

  KnowledgeBankCheckpointMetaData meta_data;
  ASSERT_TRUE(ReadTextProto(exported_path, &meta_data).ok());
  EXPECT_EQ(JoinPath(TempDir(), "in_proto_embedding_data.pbbin"),
            meta_data.checkpoint_saved_path());
}

TEST_F(InProtoKnowledgeBankTest, Import) {
  auto store = CreateDefaultStore(2);

  // Some updates.
  EmbeddingVectorProto result;
  EXPECT_TRUE(store->LookupWithUpdate("key1", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key2", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key3", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key2", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key2", &result).ok());

  // Now saves a checkpoint.
  std::string exported_path;
  ASSERT_TRUE(store->Export(TempDir(), "", &exported_path).ok());

  // Some updates.
  EXPECT_TRUE(store->LookupWithUpdate("key1", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key4", &result).ok());
  EXPECT_TRUE(store->LookupWithUpdate("key5", &result).ok());

  // Checks size and keys of embedding.
  EXPECT_EQ(5, store->Size());
  ASSERT_EQ(5, store->Keys().size());
  EXPECT_EQ("key1", store->Keys()[0]);
  EXPECT_EQ("key2", store->Keys()[1]);
  EXPECT_EQ("key3", store->Keys()[2]);
  EXPECT_EQ("key4", store->Keys()[3]);
  EXPECT_EQ("key5", store->Keys()[4]);

  // Import previous state.
  ASSERT_TRUE(store->Import(exported_path).ok());

  // Checks size and keys of embedding again.
  EXPECT_EQ(3, store->Size());
  ASSERT_EQ(3, store->Keys().size());
}

}  // namespace carls
