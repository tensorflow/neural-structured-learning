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

#include "research/carls/embedding.pb.h"  // proto to pb

namespace carls {
namespace {

// Returns a 1-D zero tensor with given dimension.
EmbeddingVectorProto InitTensor(const int dimension, const float init_value) {
  CHECK_GT(dimension, 0);
  EmbeddingVectorProto result;
  result.mutable_value()->Reserve(dimension);
  for (int i = 0; i < dimension; ++i) {
    result.add_value(init_value);
  }
  return result;
}

}  // namespace

// Static
std::unique_ptr<GradientDescentOptimizer> GradientDescentOptimizer::Create(
    const int embedding_dimension, const GradientDescentConfig& config) {
  if (config.optimizer_case() == GradientDescentConfig::OPTIMIZER_NOT_SET) {
    LOG(ERROR) << "Optimizer is not set.";
    return nullptr;
  }

  if (embedding_dimension <= 0) {
    LOG(ERROR) << "Invalid embedding_dimension: " << embedding_dimension;
    return nullptr;
  }

  // Checks learning rate.
  if (config.learning_rate() <= 0) {
    LOG(ERROR) << "Invalid learning rate: " << config.learning_rate();
    return nullptr;
  }

  // Checks params for AdaGrad optimizer.
  if (config.optimizer_case() == GradientDescentConfig::kAdagrad) {
    if (config.adagrad().init_accumulator_value() <= 0) {
      LOG(ERROR)
          << "init_accumulator_value must be positive for ADAGRAD optimizer.";
      return nullptr;
    }
  }
  return absl::make_unique<GradientDescentOptimizer>(embedding_dimension,
                                                     config);
}

GradientDescentOptimizer::GradientDescentOptimizer(
    const int embedding_dimension, const GradientDescentConfig& config)
    : embedding_dimension_(embedding_dimension),
      learning_rate_(config.learning_rate()),
      config_(config) {}

std::vector<EmbeddingVectorProto> GradientDescentOptimizer::Apply(
    const std::vector<EmbeddingVectorProto>& variables,
    const std::vector<const EmbeddingVectorProto*>& gradients,
    std::string* error_msg) {
  CHECK(error_msg != nullptr);
  if (variables.empty()) {
    *error_msg = "Empty variables.";
    return {};
  }
  if (variables.size() != gradients.size()) {
    *error_msg = absl::StrCat("Inconsistent (variables, gradients) sizes: (",
                              variables.size(), ", ", gradients.size(), ")");
    return {};
  }
  std::vector<EmbeddingVectorProto> results(variables.size());
  for (size_t i = 0; i < variables.size(); ++i) {
    if (variables[i].value_size() != embedding_dimension_ ||
        gradients[i]->value_size() != embedding_dimension_) {
      *error_msg =
          absl::StrCat("Inconsistent variable and gradient value size: ",
                       variables[i].value_size(), " v.s. ",
                       gradients[i]->value_size(), " for input ", i);
      return {};
    }

    switch (config_.optimizer_case()) {
      case GradientDescentConfig::kSgd:
        results[i] = ApplyGradientDescent(variables[i], *gradients[i]);
        break;
      case GradientDescentConfig::kAdagrad:
        results[i] = ApplyAdagrad(variables[i], *gradients[i]);
        break;
      default:
        LOG(FATAL) << "Unsupported optimizer: " << config_.optimizer_case();
    }

    results[i].set_tag(variables[i].tag());
    results[i].set_weight(variables[i].weight());
  }
  return results;
}

EmbeddingVectorProto GradientDescentOptimizer::ApplyGradientDescent(
    const EmbeddingVectorProto& var, const EmbeddingVectorProto& grad) {
  EmbeddingVectorProto result;
  result.mutable_value()->Reserve(var.value_size());
  for (int i = 0; i < var.value_size(); ++i) {
    result.add_value(var.value(i) - grad.value(i) * learning_rate_);
  }
  return result;
}

EmbeddingVectorProto GradientDescentOptimizer::ApplyAdagrad(
    const EmbeddingVectorProto& var, const EmbeddingVectorProto& grad) {
  EmbeddingVectorProto result;
  result.mutable_value()->Reserve(embedding_dimension_);
  const auto& key = var.tag();
  absl::MutexLock l(&params_mu_);
  if (!params_[kAccum].contains(key)) {
    params_[kAccum][key] = InitTensor(
        embedding_dimension_, config_.adagrad().init_accumulator_value());
  }

  auto* accum = params_[kAccum][key].mutable_value();
  for (int i = 0; i < embedding_dimension_; ++i) {
    *accum->Mutable(i) += grad.value(i) * grad.value(i);
    result.add_value(var.value(i) -
                     grad.value(i) * learning_rate_ / std::sqrt(accum->Get(i)));
  }
  return result;
}

}  // namespace carls
