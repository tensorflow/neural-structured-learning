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

#include "research/carls/knowledge_bank/initializer_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "research/carls/embedding.pb.h"  // proto to pb

namespace carls {

using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(InitializerHelperTest, ValidateInitializer) {
  EmbeddingInitializer initializer;

  // Empty initializer.
  EXPECT_THAT(ValidateInitializer(/*embedding_dimension=*/1, initializer),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Initializer is not supported: ")));

  // Default initializer.
  initializer.mutable_default_embedding()->add_value(1.0);
  EXPECT_THAT(
      ValidateInitializer(/*embedding_dimension=*/2, initializer),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Inconsistent dimension of default_embedding: ")));
  initializer.mutable_default_embedding()->add_value(1.0);
  EXPECT_OK(ValidateInitializer(/*embedding_dimension=*/2, initializer));

  // Zero initializer.
  initializer.mutable_zero_initializer();
  EXPECT_OK(ValidateInitializer(/*embedding_dimension=*/2, initializer));

  // Random uniform initializer.
  initializer.mutable_random_uniform_initializer()->set_low(1.0);
  initializer.mutable_random_uniform_initializer()->set_high(-1.0);
  EXPECT_THAT(ValidateInitializer(/*embedding_dimension=*/2, initializer),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid (low, high) pair: ")));
  initializer.mutable_random_uniform_initializer()->set_low(-1.0);
  initializer.mutable_random_uniform_initializer()->set_high(1.0);
  EXPECT_OK(ValidateInitializer(/*embedding_dimension=*/2, initializer));

  // Random normal initializer.
  initializer.mutable_random_normal_initializer()->set_mean(1.0);
  initializer.mutable_random_normal_initializer()->set_stddev(-1.0);
  EXPECT_THAT(ValidateInitializer(/*embedding_dimension=*/2, initializer),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("stddev should be greater than 0.")));
  initializer.mutable_random_normal_initializer()->set_stddev(1.0);
  EXPECT_OK(ValidateInitializer(/*embedding_dimension=*/2, initializer));
}

TEST(InitializerHelperTest, InitializeEmbedding) {
  EmbeddingInitializer initializer;

  // Default embedding.
  initializer.mutable_default_embedding()->add_value(1.0);
  initializer.mutable_default_embedding()->add_value(2.0);
  EXPECT_THAT(InitializeEmbedding(2, initializer), EqualsProto(R"(
                value: 1.0
                value: 2.0
              )"));

  // Zero initializer.
  initializer.mutable_zero_initializer();
  EXPECT_THAT(InitializeEmbedding(2, initializer), EqualsProto(R"(
                value: 0.0
                value: 0.0
              )"));

  // Random uniform initializer.
  initializer.mutable_random_uniform_initializer()->set_low(-1.0);
  initializer.mutable_random_uniform_initializer()->set_high(1.0);
  EXPECT_EQ(2, InitializeEmbedding(2, initializer).value_size());

  // Random normal initializer.
  initializer.mutable_random_normal_initializer()->set_mean(1.0);
  initializer.mutable_random_normal_initializer()->set_stddev(1.0);
  EXPECT_EQ(2, InitializeEmbedding(2, initializer).value_size());
}

TEST(InitializerHelperTest, InitializeEmbeddingWithDeterministicSeed) {
  EmbeddingInitializer initializer;
  absl::Mutex mu;
  std::seed_seq seed_seq({1, 2, 3});
  RandomEngine eng(seed_seq);

  // Default embedding.
  initializer.mutable_default_embedding()->add_value(1.0);
  initializer.mutable_default_embedding()->add_value(2.0);
  EXPECT_THAT(InitializeEmbeddingWithSeed(2, initializer, &eng, &mu),
              EqualsProto(R"(
                value: 1.0 value: 2.0
              )"));

  // Zero initializer.
  initializer.mutable_zero_initializer();
  EXPECT_THAT(InitializeEmbeddingWithSeed(2, initializer, &eng, &mu),
              EqualsProto(R"(
                value: 0.0 value: 0.0
              )"));

  // Random uniform initializer.
  initializer.mutable_random_uniform_initializer()->set_low(-1.0);
  initializer.mutable_random_uniform_initializer()->set_high(1.0);
  EXPECT_THAT(InitializeEmbeddingWithSeed(2, initializer, &eng, &mu),
              EqualsProto(R"(
                value: -0.20330858 value: -0.6722764
              )"));

  // Random normal initializer.
  initializer.mutable_random_normal_initializer()->set_mean(1.0);
  initializer.mutable_random_normal_initializer()->set_stddev(1.0);
  EXPECT_THAT(InitializeEmbeddingWithSeed(2, initializer, &eng, &mu),
              EqualsProto(R"(
                value: 1.8027948 value: 1.2439967
              )"));
}

}  // namespace carls
