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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_INPUT_CONTEXT_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_INPUT_CONTEXT_HELPER_H_

#include "absl/status/status.h"
#include "research/carls/input_context.pb.h"  // proto to pb
#include "tensorflow/core/example/example.pb.h"  // proto to pb

namespace carls {

// If any feature value exists given a feature name.
bool FeatureExists(const InputContext& input_context,
                   const std::string& feature_name);

// Build a KeywordFeature from a list of feature values.
// FeatureType can be one of string, float or int.
template <typename FeatureType>
InputFeature BuildInputFeature(const std::vector<FeatureType>& value_list);

// Build an InputFeature from a list of feature values and weights.
// FeatureType can be one of string, float or int.
template <typename FeatureType>
absl::Status BuildInputFeatureWithWeights(
    const std::vector<FeatureType>& value_list,
    const std::vector<float>& weight_list, InputFeature* input_feature);

// Build an InputFeature from a list of feature values and debug_infos.
// FeatureType can be one of string, float or int.
template <typename FeatureType>
absl::Status BuildInputFeatureWithDebugInfo(
    const std::vector<FeatureType>& value_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature);

// Build an InputFeature from a list of feature values, weights and debug info.
// FeatureType can be one of string, float or int.
template <typename FeatureType>
absl::Status BuildInputFeatureWithWeightsAndDebugInfo(
    const std::vector<FeatureType>& value_list,
    const std::vector<float>& weight_list,
    const std::vector<std::string>& debug_infos, InputFeature* input_feature);

// Returns a list of feature values from given InputFeature.
template <typename FeatureType>
bool FindFeatureValues(const InputFeature& input_feature,
                       std::vector<FeatureType>* results);

// Returns a list of feature values matching the feature name.
// FeatureType can be one of string, float or int.
// It also supports FeatureType = absl::string_view when the features' type is
// bytes. But the caller should make sure input_context outlives results.
template <typename FeatureType>
bool FindFeatureValuesByName(const InputContext& input_context,
                             const std::string& feature_name,
                             std::vector<FeatureType>* results) {
  results->clear();
  if (!input_context.feature().contains(feature_name)) {
    return false;
  }
  return FindFeatureValues<FeatureType>(
      input_context.feature().find(feature_name)->second, results);
}

// Returns a list of feature weights from given InputFeature.
bool FindFeatureWeights(const InputFeature& input_feature,
                        std::vector<float>* results);

// Returns a list of feature weights matching the feature name.
// FeatureType can be one of string, float or int.
bool FindFeatureWeightsByName(const InputContext& input_context,
                              const std::string& feature_name,
                              std::vector<float>* results);

// Returns a list of feature values from given InputFeature.
// A feature can have multiple weights, weight_position specifies which weight
// to use.
template <typename FeatureType,
          typename MapContainerType = std::map<FeatureType, float>>
absl::Status FindFeatureValuesAndWeights(const InputFeature& input_feature,
                                         int weight_position,
                                         MapContainerType* container);

// Returns a list of feature values from given InputContext.
template <typename FeatureType,
          typename MapContainerType = std::map<FeatureType, float>>
absl::Status FindFeatureValuesAndWeightsByName(
    const InputContext& input_context, const std::string& feature_name,
    MapContainerType* container) {
  if (!input_context.feature().contains(feature_name)) {
    return absl::InvalidArgumentError("Given feature name does not exist.");
  }
  return FindFeatureValuesAndWeights<FeatureType, MapContainerType>(
      *input_context.feature().find(feature_name));
}

// Add feature into a InputContext, die if the feature name already exists.
void AddFeatureOrDie(const std::string& feature_name,
                     const InputFeature& feature, InputContext* input_context);

// Add or update feature in a InputContext,
void AddOrUpdateFeature(const std::string& feature_name,
                        const InputFeature& feature,
                        InputContext* input_context);

// Merge multiple InputContext into one.
// If allow_overlap_features = true, merge the values of the same feature name,
// otherwise, report error.
// If dedup_overlap_string_values = true and feature value is of type string,
// only keep the first seen value under each feature name, otherwise, allow
// duplicated string values.
absl::Status Merge(const std::vector<InputContext>& values,
                   bool allow_overlap_features,
                   bool dedup_overlap_string_values,
                   InputContext* input_context);

// Prunes off values with small weights and only keeps max_values_per_feature
// values per feature.
absl::Status Prune(const InputContext& input, int max_values_per_feature,
                   InputContext* input_context);

// Output a compact human-readable string representing the content of given
// input_context.
std::string DebugString(const InputContext& input_context);

// Convert Example to InputContext without the weight and debug_info fields.
InputContext ToInputContext(const tensorflow::Example& example);

// Gets all the keys of the input_context.feature().
std::vector<std::string> GetAllFeatureNames(const InputContext& input_context);

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Implementation ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename FeatureType, typename MapContainerType>
absl::Status FindFeatureValuesAndWeights(const InputFeature& input_feature,
                                         int weight_position,
                                         MapContainerType* container) {
  std::vector<FeatureType> feature_values;
  std::vector<float> feature_weights;
  if (!FindFeatureValues(input_feature, &feature_values)) {
    return absl::InvalidArgumentError("Looking for feature values failed.");
  }
  if (!FindFeatureWeights(input_feature, &feature_weights)) {
    return absl::InvalidArgumentError("Looking for feature weights failed.");
  }
  int num_weights = feature_weights.size() / feature_values.size();
  if (num_weights == 0) {
    return absl::InvalidArgumentError("No weights in the input.");
  }
  if (weight_position >= num_weights) {
    return absl::InvalidArgumentError("Invalid weight_position: ");
  }
  if (feature_weights.size() % feature_values.size() != 0) {
    return absl::InvalidArgumentError(
        "Number of weights is not multiples of feature values.");
  }
  for (size_t i = 0; i < feature_values.size(); ++i) {
    (*container)[feature_values[i]] =
        feature_weights[i * num_weights + weight_position];
  }
  return absl::OkStatus();
}

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_INPUT_CONTEXT_HELPER_H_
