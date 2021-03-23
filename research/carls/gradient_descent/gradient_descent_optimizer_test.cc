/* Copyright 2021 Google LLC. All Rights Reserved.

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

#include "research/carls/gradient_descent/gradient_descent_optimizer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/testing/test_helper.h"

namespace carls {

class GradientDescentOptimizerTest : public ::testing::Test {
 protected:
  GradientDescentOptimizerTest() {
    var1_ = ParseTextProtoOrDie<EmbeddingVectorProto>(
        R"pb(
          tag: "first" weight: 10 value: 1 value: 2
        )pb");
    grad1_ = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      value: 1 value: 1
    )pb");
    var2_ = ParseTextProtoOrDie<EmbeddingVectorProto>(
        R"pb(
          tag: "second" weight: 20 value: 100 value: 200
        )pb");
    grad2_ = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
      value: 10 value: 10
    )pb");
  }

  EmbeddingVectorProto var1_, var2_;
  EmbeddingVectorProto grad1_, grad2_;
};

TEST_F(GradientDescentOptimizerTest, SGD) {
  // Test the case when Optimizer is not set.
  GradientDescentConfig config;
  auto gd_result = GradientDescentOptimizer::Create(2, config);
  EXPECT_TRUE(gd_result == nullptr);

  config = ParseTextProtoOrDie<GradientDescentConfig>(R"pb(
    learning_rate: 0.1
    sgd {}
  )pb");
  gd_result =
      GradientDescentOptimizer::Create(/*embedding_dimension=*/2, config);
  ASSERT_TRUE(gd_result != nullptr);

  // One vector.
  std::string error_msg;
  auto update_result = gd_result->Apply({var1_}, {&grad1_}, &error_msg);
  ASSERT_EQ(1, update_result.size());
  EXPECT_THAT(update_result[0], EqualsProto<EmbeddingVectorProto>(R"(
                tag: "first"
                value: 0.9
                value: 1.9
                weight: 10
              )"));

  // Two vectors.
  update_result =
      gd_result->Apply({var1_, var2_}, {&grad1_, &grad2_}, &error_msg);
  ASSERT_EQ(2, update_result.size());
  EXPECT_THAT(update_result[0], EqualsProto<EmbeddingVectorProto>(R"(
                tag: "first"
                value: 0.9
                value: 1.9
                weight: 10
              )"));
  EXPECT_THAT(update_result[1], EqualsProto<EmbeddingVectorProto>(R"(
                tag: "second"
                value: 99
                value: 199
                weight: 20
              )"));
}

TEST_F(GradientDescentOptimizerTest, Adagrad) {
  // Test the case when init_accumulator_value is not set.
  GradientDescentConfig config = ParseTextProtoOrDie<GradientDescentConfig>(R"(
    learning_rate: 0.1
    adagrad {}
  )");
  auto gd_result =
      GradientDescentOptimizer::Create(/*embedding_dimension=*/2, config);
  EXPECT_TRUE(gd_result == nullptr);

  config.mutable_adagrad()->set_init_accumulator_value(0.1);
  gd_result = GradientDescentOptimizer::Create(2, config);
  ASSERT_TRUE(gd_result != nullptr);

  // One vector.
  std::string error_msg;
  auto update_result = gd_result->Apply({var1_}, {&grad1_}, &error_msg);
  ASSERT_EQ(1, update_result.size());
  EXPECT_THAT(update_result[0], EqualsProto<EmbeddingVectorProto>(R"(
                tag: "first"
                value: 0.9046537
                value: 1.9046538
                weight: 10
              )"));

  // Apply to the same variable again, the result is no longer the same.
  update_result = gd_result->Apply({var1_}, {&grad1_}, &error_msg);
  ASSERT_TRUE(!update_result.empty());
  ASSERT_EQ(1, update_result.size());
  EXPECT_NE(update_result[0].value(0), 0.9046537);
  EXPECT_NE(update_result[0].value(1), 1.9046538);
}

}  // namespace carls
