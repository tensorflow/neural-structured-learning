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

#include "research/carls/base/proto_factory.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/testing/test_proto3.pb.h"  // proto to pb

namespace carls {
namespace {

// Macro for registering an embedding store implementation.
#define REGISTER_TEST_CLASS_FACTORY(proto_type, factory_type) \
  REGISTER_KNOWLEDGE_BANK_FACTORY_0(proto_type, factory_type, \
                                    TestBaseProto3Def, FakeEmbeddingStoreBase)

class FakeEmbeddingStoreBase {
 public:
  virtual ~FakeEmbeddingStoreBase() = default;

  virtual int Lookup(int num) = 0;
};

REGISTER_KNOWLEDGE_BANK_BASE_CLASS_0(TestBaseProto3Def, FakeEmbeddingStoreBase,
                                     EmbeddingStoreFactory);

class FakeEmbeddingStore : public FakeEmbeddingStoreBase {
 public:
  int Lookup(int num) override { return num; }
};

REGISTER_TEST_CLASS_FACTORY(TestExtendedProto3Def,
                            [](const TestBaseProto3Def& config)
                                -> std::unique_ptr<FakeEmbeddingStoreBase> {
                              return std::unique_ptr<FakeEmbeddingStoreBase>(
                                  new FakeEmbeddingStore());
                            });

}  // namespace

TEST(ProtoFactoryTest, InProtoEmbeddingStore) {
  TestBaseProto3Def base_proto;
  TestExtendedProto3Def extended_proto;
  base_proto.mutable_extension()->PackFrom(extended_proto);
  std::unique_ptr<FakeEmbeddingStoreBase> embedding_store =
      EmbeddingStoreFactory::Make(base_proto);
  ASSERT_TRUE(embedding_store != nullptr);
  EXPECT_EQ(100, embedding_store->Lookup(100));
}

}  // namespace carls
