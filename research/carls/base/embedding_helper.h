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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_EMBEDDING_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_EMBEDDING_HELPER_H_

#include <string>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "tensorflow/core/framework/tensor.h"

namespace carls {

// An in-memory representation of an EmbeddingVectorProto.
struct InMemoryEmbeddingVector {
  std::string tag;
  float weight;
  Eigen::VectorXf vec;

  InMemoryEmbeddingVector() : weight(1.0f) {}
  InMemoryEmbeddingVector(const std::string& t, float w,
                          const std::vector<float>& vec_values);
};

// Converts from EmbeddingVector to EmbeddingVectorProto.
EmbeddingVectorProto ToEmbeddingVectorProto(
    const InMemoryEmbeddingVector& embedding);

// Converts from EmbeddingVectorProto to EmbeddingVector.
InMemoryEmbeddingVector ToInMemoryEmbeddingVector(
    const EmbeddingVectorProto& proto);

// Converts from EmbeddingVectorProto to tensorflow::Tensor.
tensorflow::Tensor ToTensorFlowTensor(const EmbeddingVectorProto& proto);

// Computes the consine similarity between two vectors represented by different
// types (Eigen::VectorXf/EmbeddingVectorProto).
// Returns false when the inputs are invalid to compute the cosine similarity.
template <typename FirstType, typename SecondType>
bool ComputeCosineSimilarity(const FirstType& first, const SecondType& second,
                             float* result);

// Computes the dot product between two vectors represented by different
// types (Eigen::VectorXf/EmbeddingVectorProto).
// Returns false when the inputs are invalid.
template <typename FirstType, typename SecondType>
bool ComputeDotProduct(const FirstType& first, const SecondType& second,
                       float* result);

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_EMBEDDING_HELPER_H_
