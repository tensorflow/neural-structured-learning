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

namespace carls {

InMemoryEmbeddingVector::InMemoryEmbeddingVector(
    const std::string& t, float w, const std::vector<float>& vec_values)
    : tag(t), weight(w) {
  vec = Eigen::VectorXf::Zero(vec_values.size());
  for (size_t i = 0; i < vec_values.size(); ++i) {
    vec[i] = vec_values[i];
  }
}

EmbeddingVectorProto ToEmbeddingVectorProto(
    const InMemoryEmbeddingVector& embedding) {
  EmbeddingVectorProto proto;
  proto.set_tag(embedding.tag);
  proto.set_weight(embedding.weight);
  proto.mutable_value()->Reserve(embedding.vec.size());
  for (int i = 0; i < embedding.vec.size(); ++i) {
    proto.add_value(embedding.vec[i]);
  }
  return proto;
}

InMemoryEmbeddingVector ToInMemoryEmbeddingVector(
    const EmbeddingVectorProto& proto) {
  return InMemoryEmbeddingVector(
      proto.tag(), proto.weight(),
      std::vector<float>(proto.value().begin(), proto.value().end()));
}

tensorflow::Tensor ToTensorFlowTensor(const EmbeddingVectorProto& proto) {
  tensorflow::Tensor output(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({proto.value_size()}));
  auto vec = output.vec<float>();
  for (int i = 0; i < proto.value_size(); ++i) {
    vec(i) = proto.value(i);
  }
  return output;
}

template <>
bool ComputeCosineSimilarity(const Eigen::VectorXf& first,
                             const Eigen::VectorXf& second, float* result) {
  if (result == nullptr) {
    return false;
  }
  if (first.size() != second.size() || first.size() == 0) {
    return false;
  }
  const float norm = first.norm() * second.norm();
  if (std::abs(norm) < 1e-6) {
    return false;
  }
  *result = first.dot(second) / norm;
  return true;
}

template <>
bool ComputeCosineSimilarity(const EmbeddingVectorProto& first,
                             const EmbeddingVectorProto& second,
                             float* result) {
  return ComputeCosineSimilarity(ToInMemoryEmbeddingVector(first).vec,
                                 ToInMemoryEmbeddingVector(second).vec, result);
}

template <>
bool ComputeCosineSimilarity(const Eigen::VectorXf& first,
                             const EmbeddingVectorProto& second,
                             float* result) {
  return ComputeCosineSimilarity(first, ToInMemoryEmbeddingVector(second).vec,
                                 result);
}

template <>
bool ComputeCosineSimilarity(const EmbeddingVectorProto& first,
                             const Eigen::VectorXf& second, float* result) {
  return ComputeCosineSimilarity(ToInMemoryEmbeddingVector(first).vec, second,
                                 result);
}

}  // namespace carls
