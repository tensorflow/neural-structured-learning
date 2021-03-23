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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_GRADIENT_DESCENT_GRADIENT_DESCENT_OPTIMIZER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_GRADIENT_DESCENT_GRADIENT_DESCENT_OPTIMIZER_H_

#include <glog/logging.h>
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/gradient_descent/gradient_descent_config.pb.h"  // proto to pb

namespace carls {

// List of non-input argument names used in
// tensorflow/core/kernels/training_ops.h
constexpr char kAccum[] = "accum";
constexpr char kAccumUpdate[] = "accum_update";

class GradientDescentOptimizer {
 public:
  // Create helps check the validality of the input config.
  static std::unique_ptr<GradientDescentOptimizer> Create(
      const int embedding_dimension, const GradientDescentConfig& config);

  GradientDescentOptimizer(const int embedding_dimension,
                           const GradientDescentConfig& config);

  // Applies the gradients to the variables using given optimizer.
  // Note that we require EmbeddingVectorProto::tag to be a valid index key
  // after normalization to locate the correct per-update variables.
  std::vector<EmbeddingVectorProto> Apply(
      const std::vector<EmbeddingVectorProto>& variables,
      const std::vector<const EmbeddingVectorProto*>& gradients,
      std::string* error_msg);

 private:
  // Implementation of the basic SGD algorithm.
  EmbeddingVectorProto ApplyGradientDescent(const EmbeddingVectorProto& var,
                                            const EmbeddingVectorProto& grad);

  // Implementation of the ApplyAdagrad algorithm.
  EmbeddingVectorProto ApplyAdagrad(const EmbeddingVectorProto& var,
                                    const EmbeddingVectorProto& grad);

  const int embedding_dimension_;
  const float learning_rate_;
  const GradientDescentConfig config_;

  // Mutex for params_.
  absl::Mutex params_mu_;

  // A map from parameter name to EmbeddingVectorProto. For example, the accum
  // variable for key 'abc' in Adagrad can be accessed by
  // params_['accum']['abc']. We protect it by a Mutex to guarantee consistent
  // update for each batch.
  absl::node_hash_map<std::string,
                      absl::node_hash_map<std::string, EmbeddingVectorProto>>
      params_ ABSL_GUARDED_BY(params_mu_);
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_GRADIENT_DESCENT_GRADIENT_DESCENT_OPTIMIZER_H_
