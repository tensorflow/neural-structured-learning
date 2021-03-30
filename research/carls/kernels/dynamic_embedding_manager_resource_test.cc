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

#include "research/carls/kernels/dynamic_embedding_manager_resource.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/time.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/kbs_server_helper.h"

namespace carls {

class DynamicEmbeddingManagerResourceTest : public ::testing::Test {
 protected:
  DynamicEmbeddingManagerResourceTest() {}
};

TEST_F(DynamicEmbeddingManagerResourceTest, Basic) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  const std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config =
      ParseTextProtoOrDie<DynamicEmbeddingConfig>(R"(
        embedding_dimension: 100
        knowledge_bank_config {
          initializer { zero_initializer {} }
          extension {
            [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
          }
        }
        gradient_descent_config {
          learning_rate: 0.1
          sgd {}
        }
      )");

  DynamicEmbeddingManagerResource resource(config, "embed", address,
                                           absl::Seconds(10));
  EXPECT_TRUE(resource.manager() != nullptr);
  EXPECT_EQ("DEM resource", resource.DebugString());
}

}  // namespace carls
