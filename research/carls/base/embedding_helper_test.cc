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

#include "research/carls/base/embedding_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/testing/test_helper.h"

namespace carls {

TEST(EmbeddingHelperTest, ToEmbeddingVectorProto) {
  InMemoryEmbeddingVector vec;
  auto proto = ToEmbeddingVectorProto(vec);
  EXPECT_THAT(proto, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "" weight: 1
              )pb"));

  vec.vec.resize(3);
  vec.vec << 1, 2, 3;
  vec.tag = "first";
  vec.weight = 0.5;
  proto = ToEmbeddingVectorProto(vec);
  EXPECT_THAT(proto, EqualsProto<EmbeddingVectorProto>(R"pb(
                tag: "first"
                value: 1
                value: 2
                value: 3
                weight: 0.5
              )pb"));
}

TEST(EmbeddingHelperTest, ToInMemoryEmbeddingVector) {
  auto proto = ParseTextProtoOrDie<EmbeddingVectorProto>(
      R"pb(
        tag: "first" value: 1 value: 2 value: 3 weight: 0.5
      )pb");
  auto vec = ToInMemoryEmbeddingVector(proto);
  InMemoryEmbeddingVector expected_vec("first", 0.5, {1, 2, 3});
  EXPECT_EQ(vec.tag, expected_vec.tag);
  EXPECT_NEAR(vec.weight, expected_vec.weight, 1e-6);
  ASSERT_EQ(vec.vec.size(), expected_vec.vec.size());
  for (int i = 0; i < vec.vec.size(); ++i) {
    EXPECT_FLOAT_EQ(vec.vec[i], expected_vec.vec[i]);
  }
}

TEST(EmbeddingHelperTest, ToTensorFlowTensor) {
  EmbeddingVectorProto proto;
  auto tensor = ToTensorFlowTensor(proto);
  EXPECT_EQ(0, tensor.flat<float>().size());

  proto = ParseTextProtoOrDie<EmbeddingVectorProto>(
      R"pb(
        tag: "first" value: 1 value: 2 value: 3 weight: 0.5
      )pb");
  tensor = ToTensorFlowTensor(proto);
  auto values = tensor.flat<float>();
  ASSERT_EQ(3, values.size());
  EXPECT_FLOAT_EQ(1, values(0));
  EXPECT_FLOAT_EQ(2, values(1));
  EXPECT_FLOAT_EQ(3, values(2));
}

TEST(EmbeddingHelperTest, ComputeCosineSimilarity) {
  const auto proto_first = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "first"
    value: 0.5
    value: 1
  )pb");
  const auto proto_second = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "second"
    value: 1
    value: 0.5
  )pb");
  Eigen::VectorXf vec_first = ToInMemoryEmbeddingVector(proto_first).vec;
  Eigen::VectorXf vec_second = ToInMemoryEmbeddingVector(proto_second).vec;

  float result = -1;
  // Proto to Proto
  ASSERT_TRUE(ComputeCosineSimilarity(proto_first, proto_second, &result));
  EXPECT_FLOAT_EQ(0.8, result);
  // Proto to Vector
  ASSERT_TRUE(ComputeCosineSimilarity(proto_first, vec_second, &result));
  EXPECT_FLOAT_EQ(0.8, result);
  // Vector to Proto
  ASSERT_TRUE(ComputeCosineSimilarity(vec_first, proto_second, &result));
  EXPECT_FLOAT_EQ(0.8, result);
  // Vector to Vector
  ASSERT_TRUE(ComputeCosineSimilarity(vec_first, vec_second, &result));
  EXPECT_FLOAT_EQ(0.8, result);
}

TEST(EmbeddingHelperTest, ComputeCosineSimilarity_InvalidInput) {
  const auto proto_first = ParseTextProtoOrDie<EmbeddingVectorProto>(
      R"pb(
        tag: "first" value: 0 value: 1 value: 2
      )pb");
  const auto proto_second = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "second"
    value: 1
    value: 2
  )pb");
  const auto proto_third = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "third"
    value: 0
    value: 0
  )pb");
  Eigen::VectorXf vec_first = ToInMemoryEmbeddingVector(proto_first).vec;
  Eigen::VectorXf vec_second = ToInMemoryEmbeddingVector(proto_second).vec;

  float result = 0;
  // Inconsistent embedding dimensions.
  EXPECT_FALSE(ComputeCosineSimilarity(proto_first, proto_second, &result));
  // NULL input.
  EXPECT_FALSE(ComputeCosineSimilarity(proto_first, proto_first, nullptr));
  // All zeros input.
  EXPECT_FALSE(ComputeCosineSimilarity(proto_second, proto_third, &result));
}

TEST(EmbeddingHelperTest, ComputeDotProduct) {
  const auto proto_first = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "first"
    value: 1
    value: 2
  )pb");
  const auto proto_second = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "second"
    value: 3
    value: 4
  )pb");
  Eigen::VectorXf vec_first = ToInMemoryEmbeddingVector(proto_first).vec;
  Eigen::VectorXf vec_second = ToInMemoryEmbeddingVector(proto_second).vec;

  float result = -1;
  // Proto to Proto
  ASSERT_TRUE(ComputeDotProduct(proto_first, proto_second, &result));
  EXPECT_FLOAT_EQ(11, result);
  // Proto to Vector
  ASSERT_TRUE(ComputeDotProduct(proto_first, vec_second, &result));
  EXPECT_FLOAT_EQ(11, result);
  // Vector to Proto
  ASSERT_TRUE(ComputeDotProduct(vec_first, proto_second, &result));
  EXPECT_FLOAT_EQ(11, result);
  // Vector to Vector
  ASSERT_TRUE(ComputeDotProduct(vec_first, vec_second, &result));
  EXPECT_FLOAT_EQ(11, result);
}

TEST(EmbeddingHelperTest, ComputeDotProduct_InvalidInput) {
  const auto proto_first = ParseTextProtoOrDie<EmbeddingVectorProto>(
      R"pb(
        tag: "first" value: 1 value: 2 value: 3
      )pb");
  const auto proto_second = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    tag: "second"
    value: 4
    value: 5
  )pb");
  Eigen::VectorXf vec_first = ToInMemoryEmbeddingVector(proto_first).vec;
  Eigen::VectorXf vec_second = ToInMemoryEmbeddingVector(proto_second).vec;

  float result = -1;
  // Inconsistent embedding dimensions.
  EXPECT_FALSE(ComputeDotProduct(proto_first, proto_second, &result));
  // NULL input.
  EXPECT_FALSE(ComputeDotProduct(proto_first, proto_first, nullptr));
}

}  // namespace carls
