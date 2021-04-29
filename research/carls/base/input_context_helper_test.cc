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

#include "research/carls/base/input_context_helper.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/input_context.pb.h"  // proto to pb
#include "research/carls/testing/test_helper.h"

namespace carls {

using ::testing::UnorderedElementsAre;

TEST(InputContextHelperTest, BuildInputFeatureTest) {
  // String feature.
  auto string_feature =
      BuildInputFeature(std::vector<std::string>{"val1", "val2"});
  EXPECT_THAT(string_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "val1" } }
                feature_value { bytes_feature { value: "val2" } }
              )pb"));

  // Int feature.
  auto int64_feature = BuildInputFeature(std::vector<int>{1, 2});
  EXPECT_THAT(int64_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { int64_feature { value: 1 } }
                feature_value { int64_feature { value: 2 } }
              )pb"));

  // Float feature.
  auto float_feature = BuildInputFeature(std::vector<float>{1, 2});
  EXPECT_THAT(float_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { float_feature { value: 1 } }
                feature_value { float_feature { value: 2 } }
              )pb"));

  // uint64 feature.
  auto uint64_feature = BuildInputFeature(std::vector<uint64_t>{1, 2});
  EXPECT_THAT(uint64_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { uint64_feature { value: 1 } }
                feature_value { uint64_feature { value: 2 } }
              )pb"));
}

TEST(InputContextHelperTest, BuildInputFeatureWithWeightsTest) {
  InputFeature input_feature;
  EXPECT_NOT_OK(
      BuildInputFeatureWithWeights(std::vector<std::string>{"val1", "val2"},
                                   std::vector<float>{1.0}, &input_feature));
  // String feature.
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<std::string>{"val1", "val2"}, std::vector<float>{1.0, 2.0},
      &input_feature));
  EXPECT_THAT(input_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "val1" weight: 1 } }
                feature_value { bytes_feature { value: "val2" weight: 2 } }
              )pb"));

  // Int feature.
  InputFeature int64_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<int>{1, 2}, std::vector<float>{4.0, 5.0}, &int64_feature));
  EXPECT_THAT(int64_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { int64_feature { value: 1 weight: 4 } }
                feature_value { int64_feature { value: 2 weight: 5 } }
              )pb"));

  // uint64 feature.
  InputFeature uint64_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(std::vector<uint64_t>{1, 2},
                                         std::vector<float>{4.0, 5.0},
                                         &uint64_feature));
  EXPECT_THAT(uint64_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { uint64_feature { value: 1 weight: 4 } }
                feature_value { uint64_feature { value: 2 weight: 5 } }
              )pb"));

  // Float feature.
  InputFeature float_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<float>{1, 2}, std::vector<float>{1.0, 2.0}, &float_feature));
  EXPECT_THAT(float_feature, EqualsProto<InputFeature>(R"pb(
                feature_value { float_feature { value: 1 weight: 1 } }
                feature_value { float_feature { value: 2 weight: 2 } }
              )pb"));
}

TEST(InputContextHelperTest, FindFeatureValuesByNameTest) {
  InputContext input_context;

  //////////////////////////// String feature //////////////////////////////////
  std::vector<std::string> string_feature;
  // Empty context.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "string_fea", &string_feature));

  (*input_context.mutable_feature())["string_fea1"] =
      BuildInputFeature(std::vector<std::string>{"value1", "value2"});

  // No such feature.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "string_fea", &string_feature));

  // Desired feature.
  EXPECT_TRUE(
      FindFeatureValuesByName(input_context, "string_fea1", &string_feature));
  ASSERT_EQ(2, string_feature.size());
  EXPECT_EQ("value1", string_feature[0]);
  EXPECT_EQ("value2", string_feature[1]);

  ///////////////////////// string view feature ////////////////////////////////
  std::vector<absl::string_view> string_view_feature;
  // Empty context.
  EXPECT_FALSE(FindFeatureValuesByName(input_context, "string_fea",
                                       &string_view_feature));

  // Desired feature.
  EXPECT_TRUE(FindFeatureValuesByName(input_context, "string_fea1",
                                      &string_view_feature));
  ASSERT_EQ(2, string_view_feature.size());
  EXPECT_EQ("value1", string_view_feature[0]);
  EXPECT_EQ("value2", string_view_feature[1]);

  //////////////////////////// Int feature //////////////////////////////////
  std::vector<int> int64_feature;
  // Empty context.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "int_fea", &int64_feature));

  (*input_context.mutable_feature())["int_fea1"] =
      BuildInputFeature(std::vector<int>{1, 2});

  // No such feature.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "int_fea", &int64_feature));

  // Desired feature.
  EXPECT_TRUE(
      FindFeatureValuesByName(input_context, "int_fea1", &int64_feature));
  ASSERT_EQ(2, int64_feature.size());
  EXPECT_EQ(1, int64_feature[0]);
  EXPECT_EQ(2, int64_feature[1]);

  ///////////////////////////// Float feature //////////////////////////////////
  std::vector<float> float_feature;
  // Empty context.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "float_fea", &float_feature));

  (*input_context.mutable_feature())["float_fea1"] =
      BuildInputFeature(std::vector<float>{1.0f, 2.0f});

  // No such feature.
  EXPECT_FALSE(
      FindFeatureValuesByName(input_context, "float_fea", &float_feature));

  // Desired feature.
  EXPECT_TRUE(
      FindFeatureValuesByName(input_context, "float_fea1", &float_feature));
  EXPECT_EQ(2, float_feature.size());
  EXPECT_FLOAT_EQ(1.0f, float_feature[0]);
  EXPECT_FLOAT_EQ(2.0f, float_feature[1]);
}

TEST(InputContextHelperTest, FindFeatureWeightsByName_StringInput) {
  InputContext input_context;
  std::vector<float> weights;

  // Empty context.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  InputFeature input_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<std::string>{"value1", "value2"},
      std::vector<float>{1.0, 2.0}, &input_feature));
  (*input_context.mutable_feature())["fea1"] = input_feature;

  // No such feature.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  // Desired feature.
  EXPECT_TRUE(FindFeatureWeightsByName(input_context, "fea1", &weights));
  EXPECT_EQ(2, weights.size());
  EXPECT_FLOAT_EQ(1.0, weights[0]);
  EXPECT_FLOAT_EQ(2.0, weights[1]);
}

TEST(InputContextHelperTest, FindFeatureWeightsByName_IntInput) {
  InputContext input_context;
  std::vector<float> weights;

  // Empty context.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  InputFeature input_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<int>{1, 2}, std::vector<float>{1.0, 2.0}, &input_feature));
  (*input_context.mutable_feature())["fea1"] = input_feature;

  // No such feature.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  // Desired feature.
  EXPECT_TRUE(FindFeatureWeightsByName(input_context, "fea1", &weights));
  EXPECT_EQ(2, weights.size());
  EXPECT_FLOAT_EQ(1.0, weights[0]);
  EXPECT_FLOAT_EQ(2.0, weights[1]);
}

TEST(InputContextHelperTest, FindFeatureWeightsByName_Uint64Input) {
  InputContext input_context;
  std::vector<float> weights;

  // Empty context.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  InputFeature input_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(std::vector<uint64_t>{1, 2},
                                         std::vector<float>{1.0, 2.0},
                                         &input_feature));
  (*input_context.mutable_feature())["fea1"] = input_feature;

  // No such feature.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  // Desired feature.
  EXPECT_TRUE(FindFeatureWeightsByName(input_context, "fea1", &weights));
  EXPECT_EQ(2, weights.size());
  EXPECT_FLOAT_EQ(1.0, weights[0]);
  EXPECT_FLOAT_EQ(2.0, weights[1]);
}

TEST(InputContextHelperTest, FindFeatureWeightsByName_FloatInput) {
  InputContext input_context;
  std::vector<float> weights;

  // Empty context.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  InputFeature input_feature;
  ASSERT_OK(BuildInputFeatureWithWeights(
      std::vector<float>{1, 2}, std::vector<float>{1.0, 2.0}, &input_feature));
  (*input_context.mutable_feature())["fea1"] = input_feature;

  // No such feature.
  EXPECT_FALSE(FindFeatureWeightsByName(input_context, "fea", &weights));

  // Desired feature.
  EXPECT_TRUE(FindFeatureWeightsByName(input_context, "fea1", &weights));
  EXPECT_EQ(2, weights.size());
  EXPECT_FLOAT_EQ(1.0, weights[0]);
  EXPECT_FLOAT_EQ(2.0, weights[1]);
}

TEST(InputContextHelperTest, AddFeatureOrDieTest) {
  InputContext input_context;

  AddFeatureOrDie("fea1", BuildInputFeature(std::vector<float>{1.0f, 2.0f}),
                  &input_context);
  EXPECT_TRUE(FeatureExists(input_context, "fea1"));

  std::vector<float> features;
  EXPECT_TRUE(FindFeatureValuesByName(input_context, "fea1", &features));
  EXPECT_EQ(2, features.size());

  EXPECT_DEATH(
      AddFeatureOrDie("fea1", BuildInputFeature(std::vector<float>{1.0f, 2.0f}),
                      &input_context),
      "");
}

TEST(InputContextHelperTest, AddOrUpdateFeatureTest) {
  InputContext input_context;

  // Add a feature.
  AddOrUpdateFeature("fea1", BuildInputFeature(std::vector<float>{1.0f, 2.0f}),
                     &input_context);
  EXPECT_TRUE(FeatureExists(input_context, "fea1"));

  std::vector<float> features;
  EXPECT_TRUE(FindFeatureValuesByName(input_context, "fea1", &features));
  EXPECT_EQ(2, features.size());
  EXPECT_FLOAT_EQ(1.0, features[0]);
  EXPECT_FLOAT_EQ(2.0, features[1]);

  // Update a feature.
  AddOrUpdateFeature("fea1",
                     BuildInputFeature(std::vector<float>{4.0f, 5.0f, 6.0f}),
                     &input_context);
  EXPECT_TRUE(FeatureExists(input_context, "fea1"));

  EXPECT_TRUE(FindFeatureValuesByName(input_context, "fea1", &features));
  EXPECT_EQ(3, features.size());
}

TEST(InputContextHelperTest, FeatureExistsTest) {
  InputContext input_context;

  EXPECT_FALSE(FeatureExists(input_context, ""));
  EXPECT_FALSE(FeatureExists(input_context, "fea1"));

  (*input_context.mutable_feature())["fea2"] =
      BuildInputFeature(std::vector<float>{1.0f, 2.0f});

  EXPECT_TRUE(FeatureExists(input_context, "fea2"));
}

TEST(InputContextHelperTest, Merge_NoOverlapFeature) {
  InputContext input_context1 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value { feature_value { bytes_feature { value: "val1" } } }
    }
  )pb");

  InputContext input_context2 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key2"
      value { feature_value { bytes_feature { value: "val2" } } }
    }
  )pb");

  InputContext combined_input_context;
  ASSERT_OK(Merge({input_context1, input_context2},
                  /*allow_overlap_features=*/false,
                  /*dedup_overlap_string_values=*/false,
                  &combined_input_context));
  ASSERT_EQ(2, combined_input_context.feature_size());
  EXPECT_THAT(combined_input_context.feature().at("key1"),
              EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "val1" } }
              )pb"));
  EXPECT_THAT(combined_input_context.feature().at("key2"),
              EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "val2" } }
              )pb"));

  EXPECT_NOT_OK(
      Merge({input_context1, input_context1}, /*allow_overlap_features=*/false,
            /*dedup_overlap_string_values=*/false, &combined_input_context));
}

TEST(InputContextHelperTest, Merge_WithOverlapFeature) {
  InputContext input_context1 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value { feature_value { bytes_feature { value: "val1" } } }
    }
  )pb");

  InputContext input_context2 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value { feature_value { bytes_feature { value: "val2" } } }
    }
  )pb");

  InputContext merged_input_context;
  ASSERT_OK(Merge({input_context1, input_context2},
                  /*allow_overlap_features=*/true,
                  /*dedup_overlap_string_values=*/true, &merged_input_context));
  EXPECT_THAT(merged_input_context, EqualsProto<InputContext>(R"pb(
                feature {
                  key: "key1"
                  value {
                    feature_value { bytes_feature { value: "val1" } }
                    feature_value { bytes_feature { value: "val2" } }
                  }
                }
              )pb"));
}

TEST(InputContextHelperTest, Merge_WithOverlapFeatureAndValue) {
  InputContext input_context1 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value { bytes_feature { value: "val1" } }
        feature_value { bytes_feature { value: "val2" } }
      }
    }
  )pb");

  InputContext input_context2 = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value { feature_value { bytes_feature { value: "val2" } } }
    }
  )pb");

  InputContext merged_input_context;
  EXPECT_OK(
      Merge({input_context1, input_context2}, /*allow_overlap_features=*/true,
            /*dedup_overlap_string_values=*/false, &merged_input_context));
  EXPECT_THAT(merged_input_context, EqualsProto<InputContext>(R"pb(
                feature {
                  key: "key1"
                  value {
                    feature_value { bytes_feature { value: "val1" } }
                    feature_value { bytes_feature { value: "val2" } }
                    feature_value { bytes_feature { value: "val2" } }
                  }
                }
              )pb"));

  EXPECT_OK(Merge({input_context1, input_context2},
                  /*allow_overlap_features=*/true,
                  /*dedup_overlap_string_values=*/true, &merged_input_context));
  EXPECT_THAT(merged_input_context, EqualsProto<InputContext>(R"pb(
                feature {
                  key: "key1"
                  value {
                    feature_value { bytes_feature { value: "val1" } }
                    feature_value { bytes_feature { value: "val2" } }
                  }
                }
              )pb"));
}

TEST(InputContextHelperTest, DebugString_BytesFeature) {
  InputContext input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value {
          bytes_feature { value: "val1" weight: 1.0 debug_info: "debug1" }
        }
        feature_value {
          bytes_feature { value: "val2" weight: 2.0 debug_info: "debug2" }
        }
      }
    }
  )pb");

  const std::string expected_result(R"(feature {
  key: "key1"
  bytes_values: [val1, val2]
  weights: [1, 2]
  debug_infos: [debug1, debug2]
})");
  EXPECT_EQ(expected_result, DebugString(input_context));
}

TEST(InputContextHelperTest, DebugString_FloatFeature) {
  InputContext input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value {
          float_feature { value: 1.0 weight: 1.0 debug_info: "debug1" }
        }
        feature_value {
          float_feature { value: 2.0 weight: 2.0 debug_info: "debug2" }
        }
      }
    }
  )pb");

  const std::string expected_result(R"(feature {
  key: "key1"
  float_values: [1, 2]
  weights: [1, 2]
  debug_infos: [debug1, debug2]
})");
  EXPECT_EQ(expected_result, DebugString(input_context));
}

TEST(InputContextHelperTest, DebugString_Int64Feature) {
  InputContext input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value {
          int64_feature { value: 1 weight: 1.0 debug_info: "debug1" }
        }
        feature_value {
          int64_feature { value: 2 weight: 2.0 debug_info: "debug2" }
        }
      }
    }
  )pb");

  const std::string expected_result(R"(feature {
  key: "key1"
  int64_values: [1, 2]
  weights: [1, 2]
  debug_infos: [debug1, debug2]
})");
  EXPECT_EQ(expected_result, DebugString(input_context));
}

TEST(InputContextHelperTest, DebugString_Uint64Feature) {
  InputContext input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value {
          uint64_feature { value: 1 weight: 1.0 debug_info: "debug1" }
        }
        feature_value {
          uint64_feature { value: 2 weight: 2.0 debug_info: "debug2" }
        }
      }
    }
  )pb");

  const std::string expected_result(R"(feature {
  key: "key1"
  uint64_values: [1, 2]
  weights: [1, 2]
  debug_infos: [debug1, debug2]
})");
  EXPECT_EQ(expected_result, DebugString(input_context));
}

TEST(InputContextHelperTest, BuildInputFeatureWithDebugInfo_StringInput) {
  std::vector<std::string> value_list{"first", "second", "third"};
  std::vector<std::string> debug_infos{"first_debug", "second_debug",
                                       "third_debug"};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithDebugInfo(value_list, debug_infos, &input_feature));
  EXPECT_THAT(input_feature, EqualsProto<InputFeature>(R"pb(
                feature_value {
                  bytes_feature { value: "first" debug_info: "first_debug" }
                }
                feature_value {
                  bytes_feature { value: "second" debug_info: "second_debug" }
                }
                feature_value {
                  bytes_feature { value: "third" debug_info: "third_debug" }
                }
              )pb"));
}

TEST(InputContextHelperTest, BuildInputFeatureWithDebugInfo_IntInput) {
  std::vector<int> value_list{1, 2, 3};
  std::vector<std::string> debug_infos{"first_debug", "second_debug",
                                       "third_debug"};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithDebugInfo(value_list, debug_infos, &input_feature));
  EXPECT_THAT(
      input_feature, EqualsProto<InputFeature>(R"pb(
        feature_value { int64_feature { value: 1 debug_info: "first_debug" } }
        feature_value { int64_feature { value: 2 debug_info: "second_debug" } }
        feature_value { int64_feature { value: 3 debug_info: "third_debug" } }
      )pb"));
}

TEST(InputContextHelperTest, BuildInputFeatureWithDebugInfo_Uint64Input) {
  std::vector<uint64_t> value_list{1, 2, 3};
  std::vector<std::string> debug_infos{"first_debug", "second_debug",
                                       "third_debug"};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithDebugInfo(value_list, debug_infos, &input_feature));
  EXPECT_THAT(
      input_feature, EqualsProto<InputFeature>(R"pb(
        feature_value { uint64_feature { value: 1 debug_info: "first_debug" } }
        feature_value { uint64_feature { value: 2 debug_info: "second_debug" } }
        feature_value { uint64_feature { value: 3 debug_info: "third_debug" } }
      )pb"));
}

TEST(InputContextHelperTest, BuildInputFeatureWithDebugInfo_FloatInput) {
  std::vector<float> value_list{1.0, 2.0, 3.0};
  std::vector<std::string> debug_infos{"first_debug", "second_debug",
                                       "third_debug"};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithDebugInfo(value_list, debug_infos, &input_feature));
  EXPECT_THAT(
      input_feature, EqualsProto<InputFeature>(R"pb(
        feature_value { float_feature { value: 1.0 debug_info: "first_debug" } }
        feature_value {
          float_feature { value: 2.0 debug_info: "second_debug" }
        }
        feature_value { float_feature { value: 3.0 debug_info: "third_debug" } }
      )pb"));
}

TEST(InputContextHelperTest, FindFeatureValuesAndWeights_StringInput) {
  std::vector<std::string> value_list{"first", "second", "third"};
  std::vector<float> weight_list{1.0, 2.0, 3.0};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithWeights(value_list, weight_list, &input_feature));
  std::map<std::string, float> value_and_weights;
  ASSERT_OK(FindFeatureValuesAndWeights<std::string>(
      input_feature, /*weight_position=*/0, &value_and_weights));
  EXPECT_EQ(3, value_and_weights.size());
  EXPECT_FLOAT_EQ(1.0, value_and_weights["first"]);
  EXPECT_FLOAT_EQ(2.0, value_and_weights["second"]);
  EXPECT_FLOAT_EQ(3.0, value_and_weights["third"]);
}

TEST(InputContextHelperTest, FindFeatureValuesAndWeights_FloatInput) {
  std::vector<float> value_list{4.0, 5.0, 6.0};
  std::vector<float> weight_list{1.0, 2.0, 3.0};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithWeights(value_list, weight_list, &input_feature));
  std::map<float, float> value_and_weights;
  ASSERT_OK(FindFeatureValuesAndWeights<float>(
      input_feature, /*weight_position=*/0, &value_and_weights));
  EXPECT_EQ(3, value_and_weights.size());
  EXPECT_FLOAT_EQ(1.0, value_and_weights[4.0f]);
  EXPECT_FLOAT_EQ(2.0, value_and_weights[5.0f]);
  EXPECT_FLOAT_EQ(3.0, value_and_weights[6.0f]);
}

TEST(InputContextHelperTest, FindFeatureValuesAndWeights_IntInput) {
  std::vector<int> value_list{4, 5, 6};
  std::vector<float> weight_list{1.0, 2.0, 3.0};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithWeights(value_list, weight_list, &input_feature));
  std::map<int, float> value_and_weights;
  ASSERT_OK(FindFeatureValuesAndWeights<int>(
      input_feature, /*weight_position=*/0, &value_and_weights));
  EXPECT_EQ(3, value_and_weights.size());
  EXPECT_FLOAT_EQ(1.0, value_and_weights[4]);
  EXPECT_FLOAT_EQ(2.0, value_and_weights[5]);
  EXPECT_FLOAT_EQ(3.0, value_and_weights[6]);
}

TEST(InputContextHelperTest, FindFeatureValuesAndWeights_Uint64Input) {
  std::vector<uint64_t> value_list{4, 5, 6};
  std::vector<float> weight_list{1.0, 2.0, 3.0};
  InputFeature input_feature;
  ASSERT_OK(
      BuildInputFeatureWithWeights(value_list, weight_list, &input_feature));
  std::map<uint64_t, float> value_and_weights;
  ASSERT_OK(FindFeatureValuesAndWeights<uint64_t>(
      input_feature, /*weight_position=*/0, &value_and_weights));
  EXPECT_EQ(3, value_and_weights.size());
  EXPECT_FLOAT_EQ(1.0, value_and_weights[4]);
  EXPECT_FLOAT_EQ(2.0, value_and_weights[5]);
  EXPECT_FLOAT_EQ(3.0, value_and_weights[6]);
}

TEST(InputContextHelperTest, GetAllFeatureNames) {
  InputContext input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {}
    }
    feature {
      key: "key2"
      value {}
    }
    feature {
      key: "key3"
      value {}
    }
  )pb");

  InputContext pruned_input_context;
  ASSERT_OK(Prune(input_context, /*max_values_per_feature=*/1,
                  &pruned_input_context));
  EXPECT_THAT(GetAllFeatureNames(pruned_input_context),
              UnorderedElementsAre("key1", "key2", "key3"));
}

TEST(InputContextHelperTest, ToInputContext) {
  auto example = ParseTextProtoOrDie<tensorflow::Example>(R"pb(
    features: {
      feature: {
        key: "feature1"
        value: {
          bytes_list: { value: [ "/g/12qfbnl8r", "/g/11csq8_wcp", "/m/0h9mv" ] }
        }
      }
      feature: {
        key: "feature2"
        value: { bytes_list: { value: [ "anglet", "plus", "profil" ] } }
      }
      feature: {
        key: "feature3"
        value: { int64_list: { value: [ 6842, 312482, 1394 ] } }
      }
      feature: {
        key: "feature4"
        value: { float_list: { value: [ 100, 200, 300 ] } }
      }
    }
  )pb");

  auto result = ToInputContext(example);
  ASSERT_TRUE(result.feature().contains("feature1"));
  EXPECT_THAT(result.feature().at("feature1"), EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "/g/12qfbnl8r" } }
                feature_value { bytes_feature { value: "/g/11csq8_wcp" } }
                feature_value { bytes_feature { value: "/m/0h9mv" } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("feature2"));
  EXPECT_THAT(result.feature().at("feature2"), EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "anglet" } }
                feature_value { bytes_feature { value: "plus" } }
                feature_value { bytes_feature { value: "profil" } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("feature3"));
  EXPECT_THAT(result.feature().at("feature3"), EqualsProto<InputFeature>(R"pb(
                feature_value { int64_feature { value: 6842 } }
                feature_value { int64_feature { value: 312482 } }
                feature_value { int64_feature { value: 1394 } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("feature4"));
  EXPECT_THAT(result.feature().at("feature4"), EqualsProto<InputFeature>(R"pb(
                feature_value { float_feature { value: 100 } }
                feature_value { float_feature { value: 200 } }
                feature_value { float_feature { value: 300 } }
              )pb"));
}

TEST(InputContextHelperTest, Prune) {
  const auto input_context = ParseTextProtoOrDie<InputContext>(R"pb(
    feature {
      key: "key1"
      value {
        feature_value { bytes_feature { value: "val1" weight: 1.0 } }
        feature_value { bytes_feature { value: "val2" weight: 2.0 } }
      }
    }
    feature {
      key: "key2"
      value {
        feature_value { float_feature { value: 2.0 weight: 1.0 } }
        feature_value { float_feature { value: 1.0 weight: 0.0 } }
      }
    }
    feature {
      key: "key3"
      value { feature_value { int64_feature { value: 2 weight: 10.0 } } }
    }
    feature {
      key: "key4"
      value { feature_value { uint64_feature { value: 4 weight: 10.0 } } }
    }
  )pb");

  InputContext result;
  ASSERT_OK(Prune(input_context, /*max_values_per_feature=*/1, &result));
  ASSERT_TRUE(result.feature().contains("key1"));
  EXPECT_THAT(result.feature().at("key1"), EqualsProto<InputFeature>(R"pb(
                feature_value { bytes_feature { value: "val2" weight: 2 } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("key2"));
  EXPECT_THAT(result.feature().at("key2"), EqualsProto<InputFeature>(R"pb(
                feature_value { float_feature { value: 2 weight: 1 } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("key3"));
  EXPECT_THAT(result.feature().at("key3"), EqualsProto<InputFeature>(R"pb(
                feature_value { int64_feature { value: 2 weight: 10 } }
              )pb"));
  ASSERT_TRUE(result.feature().contains("key4"));
  EXPECT_THAT(result.feature().at("key4"), EqualsProto<InputFeature>(R"pb(
                feature_value { uint64_feature { value: 4 weight: 10 } }
              )pb"));
}

}  // namespace carls
