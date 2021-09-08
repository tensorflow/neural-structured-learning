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

#include "research/carls/memory_store/memory_store.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace memory_store {
namespace {

using ::testing::TempDir;

class FakeMemory : public MemoryStore {
 public:
  explicit FakeMemory(const MemoryStoreConfig& config) : MemoryStore(config) {}

  absl::Status BatchLookupInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override {
    *results = std::vector<MemoryLookupResult>(inputs.size());
    return absl::OkStatus();
  }

  absl::Status BatchLookupWithUpdateInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override {
    *results = std::vector<MemoryLookupResult>(inputs.size());
    return absl::OkStatus();
  }

  absl::Status BatchLookupWithGrowInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override {
    *results = std::vector<MemoryLookupResult>(inputs.size());
    return absl::OkStatus();
  }

  absl::Status ExportInternal(const std::string& dir,
                              std::string* exported_path) override {
    return absl::OkStatus();
  }

  absl::Status ImportInternal(const std::string& dir) override {
    return absl::OkStatus();
  }
};

REGISTER_MEMORY_STORE_FACTORY(MemoryStoreConfig,
                              [](const MemoryStoreConfig& config) {
                                return absl::make_unique<FakeMemory>(config);
                              });

}  // namespace

class MemoryStoreTest : public ::testing::Test {
 protected:
  MemoryStoreTest() = default;
};

TEST_F(MemoryStoreTest, BatchLookup) {
  MemoryStoreConfig config;
  auto store = MemoryStoreFactory::Make(config);
  ASSERT_TRUE(store != nullptr);

  std::vector<MemoryLookupResult> results;
  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  EXPECT_NOT_OK(store->BatchLookup(inputs, &results));

  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  EXPECT_OK(store->BatchLookup(inputs, &results));
}

TEST_F(MemoryStoreTest, BatchLookupWithUpdate) {
  MemoryStoreConfig config;
  auto store = MemoryStoreFactory::Make(config);
  ASSERT_TRUE(store != nullptr);

  std::vector<MemoryLookupResult> results;
  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  EXPECT_NOT_OK(store->BatchLookupWithUpdate(inputs, &results));

  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  EXPECT_OK(store->BatchLookupWithUpdate(inputs, &results));
}

TEST_F(MemoryStoreTest, BatchLookupWithGrow) {
  MemoryStoreConfig config;
  auto store = MemoryStoreFactory::Make(config);
  ASSERT_TRUE(store != nullptr);

  std::vector<MemoryLookupResult> results;
  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  EXPECT_NOT_OK(store->BatchLookupWithGrow(inputs, &results));

  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  EXPECT_OK(store->BatchLookupWithGrow(inputs, &results));
}

TEST_F(MemoryStoreTest, Export) {
  MemoryStoreConfig config;
  auto store = MemoryStoreFactory::Make(config);
  ASSERT_TRUE(store != nullptr);

  // Non-existent Dir.
  std::string exported_path;
  EXPECT_ERROR_CONTAIN(store->Export("", "", &exported_path),
                       "Nonexistent export_directory:");

  // A valid input.
  EXPECT_OK(store->Export(TempDir(), "", &exported_path));
}

TEST_F(MemoryStoreTest, Import) {
  MemoryStoreConfig config;
  auto store = MemoryStoreFactory::Make(config);
  ASSERT_TRUE(store != nullptr);

  // Empty Dir.
  EXPECT_NOT_OK(store->Import(""));
  // A valid input.
  EXPECT_OK(store->Import("file.pbtxt"));
}

}  // namespace memory_store
}  // namespace carls
