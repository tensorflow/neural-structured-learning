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

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/random/random.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/embedding.pb.h"  // proto to pb

namespace carls {

absl::Status ValidateInitializer(const int embedding_dimension,
                                 const EmbeddingInitializer& initializer) {
  if (initializer.has_default_embedding()) {
    if (embedding_dimension != initializer.default_embedding().value_size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Inconsistent dimension of default_embedding: ",
                       initializer.default_embedding().value_size(),
                       ", expect ", embedding_dimension));
    }
    return absl::OkStatus();
  }
  if (initializer.has_zero_initializer()) {
    return absl::OkStatus();
  } else if (initializer.has_random_uniform_initializer()) {
    const auto& init = initializer.random_uniform_initializer();
    if (init.high() <= init.low()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid (low, high) pair: (", init.low(), ", ", init.high(), ")"));
    }
    return absl::OkStatus();
  } else if (initializer.has_random_normal_initializer()) {
    const auto& init = initializer.random_normal_initializer();
    if (init.stddev() <= 0) {
      return absl::InvalidArgumentError("stddev should be greater than 0.");
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Initializer is not supported: ", initializer.DebugString()));
}

EmbeddingVectorProto InitializeEmbedding(
    const int embedding_dimension, const EmbeddingInitializer& initializer) {
  if (initializer.has_default_embedding()) {
    return initializer.default_embedding();
  }
  EmbeddingVectorProto result;
  result.mutable_value()->Reserve(embedding_dimension);
  if (initializer.has_zero_initializer()) {
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(0.0f);
    }
    return result;
  }
  if (initializer.has_random_uniform_initializer()) {
    absl::SharedBitGen bit_gen;
    const auto& init = initializer.random_uniform_initializer();
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(absl::Uniform<float>(bit_gen, init.low(), init.high()));
    }
    return result;
  }
  if (initializer.has_random_normal_initializer()) {
    absl::SharedBitGen bit_gen;
    const auto& init = initializer.random_normal_initializer();
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(
          absl::Gaussian<double>(bit_gen, init.mean(), init.stddev()));
    }
    return result;
  }

  LOG(FATAL) << "Initializer is not supported: " << initializer;
  return result;
}

EmbeddingVectorProto InitializeEmbeddingWithSeed(
    const int embedding_dimension, const EmbeddingInitializer& initializer,
    RandomEngine* engine, absl::Mutex* mu) {
  CHECK(engine != nullptr);
  CHECK(mu != nullptr);
  if (initializer.has_default_embedding()) {
    return initializer.default_embedding();
  }
  EmbeddingVectorProto result;
  result.mutable_value()->Reserve(embedding_dimension);
  if (initializer.has_zero_initializer()) {
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(0.0f);
    }
    return result;
  }
  if (initializer.has_random_uniform_initializer()) {
    absl::MutexLock l(mu);
    const auto& init = initializer.random_uniform_initializer();
    std::uniform_real_distribution<float> distribution(init.low(), init.high());
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(distribution(*engine));
    }
    return result;
  }
  if (initializer.has_random_normal_initializer()) {
    absl::MutexLock l(mu);
    const auto& init = initializer.random_normal_initializer();
    std::normal_distribution<double> distribution(init.mean(), init.stddev());
    for (int i = 0; i < embedding_dimension; ++i) {
      result.add_value(distribution(*engine));
    }
    return result;
  }

  LOG(FATAL) << "Initializer is not supported: " << initializer;
  return result;
}

}  // namespace carls
