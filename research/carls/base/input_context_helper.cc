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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "research/carls/base/status_helper.h"
#include "research/carls/base/top_n.h"
#include "tensorflow/core/example/feature.pb.h"  // proto to pb

namespace carls {
namespace {

using tensorflow::Example;

template <typename ValueType>
absl::Status CheckInputInputFeature(const std::vector<ValueType>& value_list,
                                    const std::vector<float>& weight_list,
                                    const std::vector<std::string>& debug_infos,
                                    InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0);
  RET_CHECK_TRUE(input_feature != nullptr);
  input_feature->Clear();
  return absl::OkStatus();
}

}  // namespace

bool FeatureExists(const InputContext& input_context,
                   const std::string& feature_name) {
  return input_context.feature().contains(feature_name);
}

// FeatureType = string.
template <>
InputFeature BuildInputFeature(const std::vector<std::string>& value_list) {
  InputFeature feature;
  for (const auto& value : value_list) {
    feature.add_feature_value()->mutable_bytes_feature()->set_value(value);
  }
  return feature;
}

// FeatureType = int.
template <>
InputFeature BuildInputFeature(const std::vector<int>& value_list) {
  InputFeature feature;
  for (const auto& value : value_list) {
    feature.add_feature_value()->mutable_int64_feature()->set_value(value);
  }
  return feature;
}

// FeatureType = int64.
template <>
InputFeature BuildInputFeature(const std::vector<int64_t>& value_list) {
  InputFeature feature;
  for (const auto& value : value_list) {
    feature.add_feature_value()->mutable_int64_feature()->set_value(value);
  }
  return feature;
}

// FeatureType = uint64.
template <>
InputFeature BuildInputFeature(const std::vector<uint64_t>& value_list) {
  InputFeature feature;
  for (const auto& value : value_list) {
    feature.add_feature_value()->mutable_uint64_feature()->set_value(value);
  }
  return feature;
}

// FeatureType = float.
template <>
InputFeature BuildInputFeature(const std::vector<float>& value_list) {
  InputFeature feature;
  for (const auto& value : value_list) {
    feature.add_feature_value()->mutable_float_feature()->set_value(value);
  }
  return feature;
}

// FeatureType = string.
template <>
absl::Status BuildInputFeatureWithWeights(
    const std::vector<std::string>& value_list,
    const std::vector<float>& weight_list, InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0)
      << "Weights' size must be multiples of values' size.";
  input_feature->Clear();
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_bytes_feature();
    bytes_feature->set_value(value_list[i]);
    for (int j = 0; j < multiple; ++j) {
      bytes_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = int.
template <>
absl::Status BuildInputFeatureWithWeights(const std::vector<int>& value_list,
                                          const std::vector<float>& weight_list,
                                          InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0)
      << "Weights' size must be multiples of values' size.";
  input_feature->Clear();
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* int64_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    int64_feature->set_value(value_list[i]);
    for (int j = 0; j < multiple; ++j) {
      int64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = int64.
template <>
absl::Status BuildInputFeatureWithWeights(
    const std::vector<int64_t>& value_list,
    const std::vector<float>& weight_list, InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0)
      << "Weights' size must be multiples of values' size.";
  input_feature->Clear();
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* int64_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    int64_feature->set_value(value_list[i]);
    for (int j = 0; j < multiple; ++j) {
      int64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = uint64.
template <>
absl::Status BuildInputFeatureWithWeights(
    const std::vector<uint64_t>& value_list,
    const std::vector<float>& weight_list, InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0)
      << "Weights' size must be multiples of values' size.";
  input_feature->Clear();
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* uint64_feature =
        input_feature->add_feature_value()->mutable_uint64_feature();
    uint64_feature->set_value(value_list[i]);
    for (int j = 0; j < multiple; ++j) {
      uint64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = float.
template <>
absl::Status BuildInputFeatureWithWeights(const std::vector<float>& value_list,
                                          const std::vector<float>& weight_list,
                                          InputFeature* input_feature) {
  RET_CHECK_TRUE(!value_list.empty());
  RET_CHECK_TRUE(weight_list.size() % value_list.size() == 0)
      << "Weights' size must be multiples of values' size.";
  const int multiple = weight_list.size() / value_list.size();
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* float_feature =
        input_feature->add_feature_value()->mutable_float_feature();
    float_feature->set_value(value_list[i]);
    for (int j = 0; j < multiple; ++j) {
      float_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = string
template <>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<std::string>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_bytes_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
  }
  return absl::OkStatus();
}

// FeatureType = int64
template <>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<int64_t>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
  }
  return absl::OkStatus();
}

// FeatureType = uint64
template <>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<uint64_t>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_uint64_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
  }
  return absl::OkStatus();
}

// FeatureType = int
template <>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<int>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
  }
  return absl::OkStatus();
}

// FeatureType = float
template <>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<float>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_TRUE(value_list.size() == debug_infos.size());
  input_feature->Clear();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_float_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
  }
  return absl::OkStatus();
}

// FeatureType = string.
template <>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<std::string>& value_list,
    const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_OK(CheckInputInputFeature(value_list, weight_list, debug_infos,
                                      input_feature));
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* bytes_feature =
        input_feature->add_feature_value()->mutable_bytes_feature();
    bytes_feature->set_value(value_list[i]);
    bytes_feature->set_debug_info(debug_infos[i]);
    for (int j = 0; j < multiple; ++j) {
      bytes_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = int.
template <>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<int>& value_list, const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_OK(CheckInputInputFeature(value_list, weight_list, debug_infos,
                                      input_feature));
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* int64_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    int64_feature->set_value(value_list[i]);
    int64_feature->set_debug_info(debug_infos[i]);
    for (int j = 0; j < multiple; ++j) {
      int64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = int64.
template <>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<int64_t>& value_list,
    const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_OK(CheckInputInputFeature(value_list, weight_list, debug_infos,
                                      input_feature));
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* int64_feature =
        input_feature->add_feature_value()->mutable_int64_feature();
    int64_feature->set_value(value_list[i]);
    int64_feature->set_debug_info(debug_infos[i]);
    for (int j = 0; j < multiple; ++j) {
      int64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = uint64.
template <>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<uint64_t>& value_list,
    const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_OK(CheckInputInputFeature(value_list, weight_list, debug_infos,
                                      input_feature));
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* uint64_feature =
        input_feature->add_feature_value()->mutable_uint64_feature();
    uint64_feature->set_value(value_list[i]);
    uint64_feature->set_debug_info(debug_infos[i]);
    for (int j = 0; j < multiple; ++j) {
      uint64_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = float.
template <>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<float>& value_list, const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature) {
  RET_CHECK_OK(CheckInputInputFeature(value_list, weight_list, debug_infos,
                                      input_feature));
  const int multiple = weight_list.size() / value_list.size();
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto* float_feature =
        input_feature->add_feature_value()->mutable_float_feature();
    float_feature->set_value(value_list[i]);
    float_feature->set_debug_info(debug_infos[i]);
    for (int j = 0; j < multiple; ++j) {
      float_feature->add_weight(weight_list[i * multiple + j]);
    }
  }
  return absl::OkStatus();
}

// FeatureType = string.
template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<std::string>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.bytes_feature().value());
  }
  return !results->empty();
}

template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<absl::string_view>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.bytes_feature().value());
  }
  return !results->empty();
}

// FeatureType = int.
template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<int>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.int64_feature().value());
  }
  return !results->empty();
}

// FeatureType = int64.
template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<int64_t>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.int64_feature().value());
  }
  return !results->empty();
}

// FeatureType = uint64.
template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<uint64_t>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.uint64_feature().value());
  }
  return !results->empty();
}

// FeatureType = float.
template <>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<float>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    results->push_back(feature_value.float_feature().value());
  }
  return !results->empty();
}

bool FindFeatureWeights(const InputFeature& input_feature,
                        std::vector<float>* results) {
  for (const auto& feature_value : input_feature.feature_value()) {
    switch (feature_value.feature_case()) {
      case FeatureValue::kBytesFeature:
        for (const auto weight : feature_value.bytes_feature().weight()) {
          results->push_back(weight);
        }
        break;
      case FeatureValue::kInt64Feature:
        for (const auto weight : feature_value.int64_feature().weight()) {
          results->push_back(weight);
        }
        break;
      case FeatureValue::kUint64Feature:
        for (const auto weight : feature_value.uint64_feature().weight()) {
          results->push_back(weight);
        }
        break;
      case FeatureValue::kFloatFeature:
        for (const auto weight : feature_value.float_feature().weight()) {
          results->push_back(weight);
        }
        break;
      case FeatureValue::FEATURE_NOT_SET:
        return false;
    }
  }
  return !results->empty();
}

bool FindFeatureWeightsByName(const InputContext& input_context,
                              const std::string& feature_name,
                              std::vector<float>* results) {
  results->clear();
  if (!input_context.feature().contains(feature_name)) {
    return false;
  }
  return FindFeatureWeights(input_context.feature().find(feature_name)->second,
                            results);
}

void AddFeatureOrDie(const std::string& feature_name,
                     const InputFeature& feature, InputContext* input_context) {
  CHECK(input_context != nullptr);
  CHECK(input_context->feature().find(feature_name) ==
        input_context->feature().end());
  (*input_context->mutable_feature())[feature_name] = feature;
}

void AddOrUpdateFeature(const std::string& feature_name,
                        const InputFeature& feature,
                        InputContext* input_context) {
  CHECK(input_context != nullptr);
  (*input_context->mutable_feature())[feature_name] = feature;
}

absl::Status Merge(const std::vector<InputContext>& values,
                   const bool allow_overlap_features,
                   const bool dedup_overlap_string_values,
                   InputContext* input_context) {
  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>
      string_values_set;
  input_context->Clear();
  for (const auto& value : values) {
    for (const auto& pair : value.feature()) {
      const auto& feature_name = pair.first;
      const auto& input_feature = pair.second;
      if (!input_context->feature().contains(feature_name)) {
        (*input_context->mutable_feature())[feature_name] = input_feature;
        if (allow_overlap_features && dedup_overlap_string_values) {
          for (const auto& feature_value : input_feature.feature_value()) {
            if (feature_value.has_bytes_feature()) {
              string_values_set[feature_name].insert(
                  feature_value.bytes_feature().value());
            }
          }
        }
        continue;
      }
      if (!allow_overlap_features) {
        return absl::InvalidArgumentError(
            absl::StrCat("Overlapping feature name: ", feature_name));
      }
      // Now merge features values.
      for (const auto& feature_value : input_feature.feature_value()) {
        const bool should_dedup_string_feature =
            dedup_overlap_string_values && feature_value.has_bytes_feature();
        if (should_dedup_string_feature &&
            string_values_set[feature_name].contains(
                feature_value.bytes_feature().value())) {
          continue;
        }
        *(*input_context->mutable_feature())[feature_name].add_feature_value() =
            feature_value;
        if (should_dedup_string_feature) {
          string_values_set[feature_name].insert(
              feature_value.bytes_feature().value());
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status Prune(const InputContext& input, const int max_values_per_feature,
                   InputContext* input_context) {
  RET_CHECK_TRUE(max_values_per_feature > 0);
  static auto cmp = [](const FeatureValue& lhs,
                       const FeatureValue& rhs) -> bool {
    float lhs_weight = 0;
    float rhs_weight = 0;
    // Bytes.
    if (lhs.has_bytes_feature() && !lhs.bytes_feature().weight().empty()) {
      lhs_weight = lhs.bytes_feature().weight(0);
    }
    if (rhs.has_bytes_feature() && !rhs.bytes_feature().weight().empty()) {
      rhs_weight = rhs.bytes_feature().weight(0);
    }
    // Float.
    if (lhs.has_float_feature() && !lhs.float_feature().weight().empty()) {
      lhs_weight = lhs.float_feature().weight(0);
    }
    if (rhs.has_float_feature() && !rhs.float_feature().weight().empty()) {
      rhs_weight = rhs.float_feature().weight(0);
    }
    // Int64.
    if (lhs.has_int64_feature() && !lhs.int64_feature().weight().empty()) {
      lhs_weight = lhs.int64_feature().weight(0);
    }
    if (rhs.has_int64_feature() && !rhs.int64_feature().weight().empty()) {
      rhs_weight = rhs.int64_feature().weight(0);
    }
    // Uint64.
    if (lhs.has_uint64_feature() && !lhs.uint64_feature().weight().empty()) {
      lhs_weight = lhs.uint64_feature().weight(0);
    }
    if (rhs.has_uint64_feature() && !rhs.uint64_feature().weight().empty()) {
      rhs_weight = rhs.uint64_feature().weight(0);
    }

    return lhs_weight > rhs_weight;
  };
  input_context->Clear();
  for (const auto& pair : input.feature()) {
    const auto& name = pair.first;
    const auto& input_feature = pair.second;
    if (input_feature.feature_value_size() <= max_values_per_feature) {
      *(*input_context->mutable_feature())[name].mutable_feature_value() =
          input_feature.feature_value();
      continue;
    }
    TopN<FeatureValue, decltype(cmp)> topn_result(max_values_per_feature, cmp);
    for (const auto& feature_value : input_feature.feature_value()) {
      topn_result.push(feature_value);
    }
    std::vector<FeatureValue> topn_results = std::move(*topn_result.Extract());
    for (auto& feature_value : topn_results) {
      *(*input_context->mutable_feature())[name].add_feature_value() =
          std::move(feature_value);
    }
  }
  return absl::OkStatus();
}

std::string DebugString(const InputContext& input_context) {
  std::vector<std::string> lines;
  for (const auto& pair : input_context.feature()) {
    const auto& feature_name = pair.first;
    const auto& feature_value = pair.second;
    lines.push_back("feature {");
    lines.push_back(absl::StrCat("  key: \"", feature_name, "\""));
    if (feature_value.feature_value().empty()) {
      continue;
    }
    std::vector<std::string> values;
    std::vector<std::string> weights;
    std::vector<std::string> debug_infos;
    values.reserve(feature_value.feature_value().size());
    weights.reserve(feature_value.feature_value().size());
    std::string feature_prefix;
    for (const auto& value : feature_value.feature_value()) {
      std::vector<std::string> per_feature_weights;
      if (value.feature_case() == FeatureValue::kBytesFeature) {
        if (feature_prefix.empty()) {
          feature_prefix = "bytes_";
        }
        values.push_back(value.bytes_feature().value());
        for (const auto weight : value.bytes_feature().weight()) {
          per_feature_weights.push_back(absl::StrCat(weight));
        }
        if (!value.bytes_feature().debug_info().empty()) {
          debug_infos.push_back(value.bytes_feature().debug_info());
        }
      } else if (value.feature_case() == FeatureValue::kFloatFeature) {
        if (feature_prefix.empty()) {
          feature_prefix = "float_";
        }
        values.push_back(absl::StrCat(value.float_feature().value()));
        for (const auto weight : value.float_feature().weight()) {
          per_feature_weights.push_back(absl::StrCat(weight));
        }
        if (!value.float_feature().debug_info().empty()) {
          debug_infos.push_back(value.float_feature().debug_info());
        }
      } else if (value.feature_case() == FeatureValue::kInt64Feature) {
        if (feature_prefix.empty()) {
          feature_prefix = "int64_";
        }
        values.push_back(absl::StrCat(value.int64_feature().value()));
        for (const auto weight : value.int64_feature().weight()) {
          per_feature_weights.push_back(absl::StrCat(weight));
        }
        if (!value.int64_feature().debug_info().empty()) {
          debug_infos.push_back(value.int64_feature().debug_info());
        }
      } else if (value.feature_case() == FeatureValue::kUint64Feature) {
        if (feature_prefix.empty()) {
          feature_prefix = "uint64_";
        }
        values.push_back(absl::StrCat(value.uint64_feature().value()));
        for (const auto weight : value.uint64_feature().weight()) {
          per_feature_weights.push_back(absl::StrCat(weight));
        }
        if (!value.uint64_feature().debug_info().empty()) {
          debug_infos.push_back(value.uint64_feature().debug_info());
        }
      } else {  // FeatureValue::FEATURE_NOT_SET
        values.push_back("FEATURE_NOT_SET");
      }
      if (!per_feature_weights.empty()) {
        if (per_feature_weights.size() > 1) {
          weights.push_back(
              absl::StrCat("(", absl::StrJoin(per_feature_weights, ", "), ")"));
        } else {
          weights.push_back(absl::StrCat(per_feature_weights[0]));
        }
      }
    }
    lines.push_back(absl::StrCat("  ", feature_prefix, "values: [",
                                 absl::StrJoin(values, ", "), "]"));
    if (!weights.empty()) {
      lines.push_back(
          absl::StrCat("  weights: [", absl::StrJoin(weights, ", "), "]"));
    }
    if (!debug_infos.empty()) {
      lines.push_back(absl::StrCat("  debug_infos: [",
                                   absl::StrJoin(debug_infos, ", "), "]"));
    }
    lines.push_back("}");
  }
  return absl::StrJoin(lines, "\n");
}

InputContext ToInputContext(const Example& example) {
  InputContext input_context;
  for (const auto& pair : example.features().feature()) {
    const auto& feature_name = pair.first;
    const auto& feature_value = pair.second;
    InputFeature input_feature;
    if (feature_value.has_bytes_list()) {
      std::vector<std::string> byte_features(
          feature_value.bytes_list().value().begin(),
          feature_value.bytes_list().value().end());
      input_feature = BuildInputFeature(byte_features);
    } else if (feature_value.has_float_list()) {
      std::vector<float> float_features(
          feature_value.float_list().value().begin(),
          feature_value.float_list().value().end());
      input_feature = BuildInputFeature(float_features);
    } else {
      std::vector<int64_t> int64_features(
          feature_value.int64_list().value().begin(),
          feature_value.int64_list().value().end());
      input_feature = BuildInputFeature(int64_features);
    }
    AddFeatureOrDie(feature_name, input_feature, &input_context);
  }
  return input_context;
}

std::vector<std::string> GetAllFeatureNames(const InputContext& input_context) {
  std::vector<std::string> names;
  names.reserve(input_context.feature().size());
  for (const auto& pair : input_context.feature()) {
    names.push_back(pair.first);
  }
  return names;
}

}  // namespace carls
