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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_EMBEDDING_STORE_INITIALIZER_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_EMBEDDING_STORE_INITIALIZER_HELPER_H_

#include <random>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/knowledge_bank/initializer.pb.h"  // proto to pb

namespace carls {

using RandomEngine = std::mt19937;

// A util function to check if a given initializer config is valid.
absl::Status ValidateInitializer(const int embedding_dimension,
                                 const EmbeddingInitializer& initializer);

// Returns an initial value for an embedding based on EmbeddingInitializer.
// We do not check the validality of the input to trade for speed.
EmbeddingVectorProto InitializeEmbedding(
    const int embedding_dimension, const EmbeddingInitializer& initializer);

// Initializes an embedding based on given generator and Mutex.
// This is to support deterministic mode in EmbeddingInitializer.
// Example:
//   std::seed_seq seq({1, 2});
//   RandomEngine gen(seed);
//   absl::Mutex mu;
//   auto result = InitializeEmbeddingWithSeed(10, init, &gen, &mu);
// Note that a Mutex is need to protect multi-thread access to RandomEngine.
EmbeddingVectorProto InitializeEmbeddingWithSeed(
    const int embedding_dimension, const EmbeddingInitializer& initializer,
    RandomEngine* engine, absl::Mutex* mu);

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_EMBEDDING_STORE_INITIALIZER_HELPER_H_
