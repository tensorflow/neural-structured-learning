/*Copyright 2021 Google LLC

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
#include "leveldb/db.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/base/thread_bundle.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/knowledge_bank.h"
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace {

using ::testing::Eq;
using ::testing::TempDir;

std::string UniqueFilename() {
  static int64_t counter = 0;
  return absl::StrCat(counter++, "data.db");
}

}  // namespace

class LeveldbKnowledgeBankTest : public ::testing::Test {
 protected:
  LeveldbKnowledgeBankTest() {}

  std::unique_ptr<KnowledgeBank> CreateKnowledgeBank(
      const int embedding_dimension, const std::string& leveldb_address,
      const int num_in_memory_partitions,
      const int max_in_memory_write_buffer_size) {
    KnowledgeBankConfig config;
    config.mutable_initializer()->mutable_zero_initializer();
    LeveldbKnowledgeBankConfig leveldb_config;
    leveldb_config.set_leveldb_address(leveldb_address);
    leveldb_config.set_num_in_memory_partitions(num_in_memory_partitions);
    leveldb_config.set_max_in_memory_write_buffer_size(
        max_in_memory_write_buffer_size);
    config.mutable_extension()->PackFrom(leveldb_config);
    return KnowledgeBankFactory::Make(config, embedding_dimension);
  }
};

TEST_F(LeveldbKnowledgeBankTest, Create) {
  KnowledgeBankConfig config;
  config.mutable_initializer()->mutable_zero_initializer();
  // No extension.
  EXPECT_TRUE(KnowledgeBankFactory::Make(config, /*embedding_dimension=*/10) ==
              nullptr);

  LeveldbKnowledgeBankConfig leveldb_config;
  config.mutable_extension()->PackFrom(leveldb_config);
  // Empty leveldb_address.
  EXPECT_TRUE(KnowledgeBankFactory::Make(config, /*embedding_dimension=*/10) ==
              nullptr);

  leveldb_config.set_leveldb_address(JoinPath(TempDir(), "data.db"));
  config.mutable_extension()->PackFrom(leveldb_config);
  // Invalid num_in_memory_partitions.
  EXPECT_TRUE(KnowledgeBankFactory::Make(config, /*embedding_dimension=*/10) ==
              nullptr);

  leveldb_config.set_num_in_memory_partitions(10);
  config.mutable_extension()->PackFrom(leveldb_config);
  // Invalid max_in_memory_write_buffer_size.
  EXPECT_TRUE(KnowledgeBankFactory::Make(config, /*embedding_dimension=*/10) ==
              nullptr);

  // A valid case.
  leveldb_config.set_max_in_memory_write_buffer_size(1);
  config.mutable_extension()->PackFrom(leveldb_config);
  EXPECT_TRUE(KnowledgeBankFactory::Make(config, /*embedding_dimension=*/10) !=
              nullptr);
}

TEST_F(LeveldbKnowledgeBankTest, LookupWithUpdate) {
  auto knowledge_bank = CreateKnowledgeBank(
      /*embedding_dimension=*/2,
      /*leveldb_address=*/
      JoinPath(TempDir(), UniqueFilename()),
      /*num_in_memory_partitions=*/2,
      /*max_in_memory_write_buffer_size=*/1);
  EmbeddingVectorProto result;
  ASSERT_OK(knowledge_bank->LookupWithUpdate("first", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "first"
                value: 0
                value: 0
                weight: 1
              )pb"));
  ASSERT_OK(knowledge_bank->LookupWithUpdate("second", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "second"
                value: 0
                value: 0
                weight: 1
              )pb"));

  // Lookup the same key again with new value.
  auto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(
      R"pb(
        tag: "first" value: 2 value: 3 weight: 4
      )pb");
  ASSERT_OK(knowledge_bank->Update("first", proto));
  ASSERT_OK(knowledge_bank->LookupWithUpdate("first", &result));
  // Weight is incremented by 1.
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "first"
                value: 2
                value: 3
                weight: 5
              )pb"));

  EXPECT_OK(knowledge_bank->Lookup("first", &result));
  EXPECT_OK(knowledge_bank->Lookup("second", &result));
  EXPECT_NOT_OK(knowledge_bank->Lookup("third", &result));

  // Check Keys(), Size(), and Contains().
  auto keys = knowledge_bank->Keys();
  EXPECT_THAT(keys, Eq(std::vector<absl::string_view>{"first", "second"}));
  EXPECT_EQ(2, knowledge_bank->Size());
  EXPECT_TRUE(knowledge_bank->Contains("first"));
  EXPECT_TRUE(knowledge_bank->Contains("second"));
  EXPECT_FALSE(knowledge_bank->Contains("third"));
}

TEST_F(LeveldbKnowledgeBankTest, LoadDataFromLevelDb) {
  const std::string db_address = JoinPath(TempDir(), UniqueFilename());
  // Write a few embeddings into the DB.
  {
    std::unique_ptr<leveldb::DB> db_ptr;
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, db_address, &db);
    ASSERT_OK(status);
    db_ptr.reset(db);

    EmbeddingVectorProto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      tag: "key1"
      value: 1
      value: 2
    )pb");
    leveldb::WriteOptions writeOptions;
    db->Put(writeOptions, "key1", proto.SerializeAsString());

    proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      tag: "key2"
      value: 3
      value: 4
    )pb");
    db->Put(writeOptions, "key2", proto.SerializeAsString());
  }

  // Checks the data is loaded into knowledge bank.
  auto knowledge_bank =
      CreateKnowledgeBank(/*embedding_dimension=*/2,
                          /*leveldb_address=*/db_address,
                          /*num_in_memory_partitions=*/2,
                          /*max_in_memory_write_buffer_size=*/1);
  EmbeddingVectorProto result;
  ASSERT_OK(knowledge_bank->Lookup("key1", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key1"
                value: 1
                value: 2
              )pb"));
  ASSERT_OK(knowledge_bank->Lookup("key2", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key2"
                value: 3
                value: 4
              )pb"));

  EXPECT_NOT_OK(knowledge_bank->Lookup("key3", &result));

  // Check Keys(), Size(), and Contains().
  auto keys = knowledge_bank->Keys();
  EXPECT_THAT(keys, Eq(std::vector<absl::string_view>{"key1", "key2"}));
  EXPECT_EQ(2, knowledge_bank->Size());
  EXPECT_TRUE(knowledge_bank->Contains("key1"));
  EXPECT_TRUE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));
}

TEST_F(LeveldbKnowledgeBankTest, Update) {
  auto knowledge_bank = CreateKnowledgeBank(
      /*embedding_dimension=*/2,
      /*leveldb_address=*/JoinPath(TempDir(), UniqueFilename()),
      /*num_in_memory_partitions=*/2,
      /*max_in_memory_write_buffer_size=*/1);
  // Updates a few embeddings.
  EmbeddingVectorProto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "key1"
    value: 1
    value: 2
  )pb");
  EXPECT_OK(knowledge_bank->Update("key1", proto));
  EXPECT_OK(knowledge_bank->Update("key2", proto));
  // Updates again with a new embedding.
  proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "key2"
    value: 3
    value: 4
  )pb");
  EXPECT_OK(knowledge_bank->Update("key2", proto));

  // Checks the result.
  EmbeddingVectorProto result;
  ASSERT_OK(knowledge_bank->Lookup("key1", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key1"
                value: 1
                value: 2
              )pb"));

  ASSERT_OK(knowledge_bank->Lookup("key2", &result));
  EXPECT_THAT(result, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "key2"
                value: 3
                value: 4
              )pb"));
  EXPECT_NOT_OK(knowledge_bank->Lookup("key3", &result));

  // Check Keys(), Size(), and Contains().
  auto keys = knowledge_bank->Keys();
  EXPECT_THAT(keys, Eq(std::vector<absl::string_view>{"key1", "key2"}));
  EXPECT_EQ(2, knowledge_bank->Size());
  EXPECT_TRUE(knowledge_bank->Contains("key1"));
  EXPECT_TRUE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));
}

TEST_F(LeveldbKnowledgeBankTest, ExportAndImport) {
  auto knowledge_bank = CreateKnowledgeBank(
      /*embedding_dimension=*/2,
      /*leveldb_address=*/JoinPath(TempDir(), UniqueFilename()),
      /*num_in_memory_partitions=*/2,
      /*max_in_memory_write_buffer_size=*/1);

  // Updates a few embeddings.
  EmbeddingVectorProto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "key1"
    value: 1
    value: 2
  )pb");
  EXPECT_OK(knowledge_bank->Update("key1", proto));
  proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "key2"
    value: 3
    value: 4
  )pb");
  EXPECT_OK(knowledge_bank->Update("key2", proto));

  std::string ckpt_path;
  ASSERT_OK(knowledge_bank->Export(TempDir(), "embed", &ckpt_path));
  EXPECT_EQ(JoinPath(TempDir(), "embed", "embedding_store_meta_data.pbtxt"),
            ckpt_path);

  // Exports again, nothing should be exported.
  ASSERT_OK(knowledge_bank->Export(TempDir(), "embed", &ckpt_path));

  // Check Keys(), Size(), and Contains().
  auto keys = knowledge_bank->Keys();
  EXPECT_THAT(keys, Eq(std::vector<absl::string_view>{"key1", "key2"}));
  EXPECT_EQ(2, knowledge_bank->Size());
  EXPECT_TRUE(knowledge_bank->Contains("key1"));
  EXPECT_TRUE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));

  // Update more keys that won't be exported.
  proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "key3"
    value: 5
    value: 6
  )pb");
  EXPECT_OK(knowledge_bank->Update("key3", proto));
  EXPECT_TRUE(knowledge_bank->Contains("key3"));

  const std::string new_db_address = JoinPath(TempDir(), UniqueFilename());
  // Write a few embeddings into the DB.
  {
    std::unique_ptr<leveldb::DB> db_ptr;
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, new_db_address, &db);
    ASSERT_OK(status);
    db_ptr.reset(db);

    EmbeddingVectorProto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      tag: "key4"
      value: 4
      value: 5
    )pb");
    leveldb::WriteOptions writeOptions;
    db->Put(writeOptions, "key4", proto.SerializeAsString());

    proto = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      tag: "key5"
      value: 6
      value: 7
    )pb");
    db->Put(writeOptions, "key5", proto.SerializeAsString());
  }
  KnowledgeBankCheckpointMetaData meta_data;
  meta_data.set_checkpoint_saved_path(new_db_address);
  const std::string metadata_path = JoinPath(TempDir(), "meta_data.pbtxt");
  ASSERT_OK(WriteTextProto(metadata_path, meta_data, /*can_overwrite=*/true));

  ASSERT_OK(knowledge_bank->Import(metadata_path));
  EXPECT_FALSE(knowledge_bank->Contains("key1"));
  EXPECT_FALSE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));  // Updated key not exported.
  EXPECT_TRUE(knowledge_bank->Contains("key4"));
  EXPECT_TRUE(knowledge_bank->Contains("key5"));

  // First releases the DB by creating a new empty KnowledgeBank.
  knowledge_bank = CreateKnowledgeBank(
      /*embedding_dimension=*/2,
      /*leveldb_address=*/JoinPath(TempDir(), UniqueFilename()),
      /*num_in_memory_partitions=*/2,
      /*max_in_memory_write_buffer_size=*/1);
  EXPECT_FALSE(knowledge_bank->Contains("key1"));
  EXPECT_FALSE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));
  // Add a new key to the embedding.
  EXPECT_OK(knowledge_bank->Update("key4", proto));

  // Imports the previous DB.
  ASSERT_OK(knowledge_bank->Import(ckpt_path));
  EXPECT_TRUE(knowledge_bank->Contains("key1"));
  EXPECT_TRUE(knowledge_bank->Contains("key2"));
  EXPECT_FALSE(knowledge_bank->Contains("key3"));
  // Checks the newly added key is no longer there.
  EXPECT_FALSE(knowledge_bank->Contains("key4"));

  // Import a non-existent DB.
  EXPECT_NOT_OK(knowledge_bank->Import("fake DB"));
}

// This test is known to be flaky.
TEST_F(LeveldbKnowledgeBankTest, AsyncLookupWithUpdate) {
  auto knowledge_bank = CreateKnowledgeBank(
      /*embedding_dimension=*/2,
      /*leveldb_address=*/JoinPath(TempDir(), UniqueFilename()),
      /*num_in_memory_partitions=*/10,
      /*max_in_memory_write_buffer_size=*/1);

  // Test the robustness of asynchronous access of KnowledgeBank.
  ThreadBundle b("test", /*num_threads=*/100);
  for (int i = 0; i < 10; ++i) {
    b.Add([i, &knowledge_bank]() {
      const std::string key = absl::StrCat("key", i);
      EmbeddingVectorProto proto;
      proto.set_tag(key);
      proto.add_value(2 * i);
      proto.add_value(2 * i + 1);
      ASSERT_OK(knowledge_bank->Update(key, proto));
    });
  }
  b.JoinAll();

  // Check Keys(), Size(), and Contains().
  auto keys = knowledge_bank->Keys();
  std::set<absl::string_view> key_set(keys.begin(), keys.end());
  EXPECT_EQ(10, knowledge_bank->Size());
  for (int i = 0; i < 10; ++i) {
    const std::string key = absl::StrCat("key", i);
    EXPECT_TRUE(knowledge_bank->Contains(key));
    EXPECT_TRUE(key_set.find(key) != key_set.end());
  }
}

}  // namespace carls
