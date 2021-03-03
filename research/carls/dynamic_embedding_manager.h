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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_DYNAMIC_EMBEDDING_MANAGER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_DYNAMIC_EMBEDDING_MANAGER_H_

#include <string>

#include "absl/status/status.h"
#include "research/carls/dynamic_embedding_config.pb.h"  // proto to pb
#include "research/carls/knowledge_bank_grpc_service.h"
#include "tensorflow/core/framework/tensor.h"

namespace carls {

// Responsible for communicating with a KnowledgeBankService stub within
// Tensorflow C++ Operation code. Each instance of DynamicEmbeddingManager only
// works for one session.
class DynamicEmbeddingManager {
 public:
  // Connects to a KBS server and starts a session.
  // Returns a nullptr if input parameters are invalid.
  static std::unique_ptr<DynamicEmbeddingManager> Create(
      const DynamicEmbeddingConfig& config, const std::string& name,
      const std::string& kbs_address,
      absl::Duration timeout = absl::InfiniteDuration());

  DynamicEmbeddingManager(
      std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub,
      const DynamicEmbeddingConfig& config, const std::string& session_handle);

  // Prepares KnowledgeBankService::LookupRequest from given input and
  // calls DES server.
  // If a given key is empty, the output tensor is filled with zero values.
  absl::Status Lookup(const tensorflow::Tensor& keys, bool update,
                      tensorflow::Tensor* output);

  // Updates the embedding values of given keys by calling
  // KnowledgeBankService::UpdateRequest.
  // If there are duplicated keys, it only updates the value of the last seen
  // one.
  absl::Status UpdateValues(const tensorflow::Tensor& keys,
                            const tensorflow::Tensor& values);

 private:
  absl::Status LookupInternal(const tensorflow::Tensor& keys, bool update,
                              LookupResponse* response);

  std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub_;
  const DynamicEmbeddingConfig config_;
  const std::string session_handle_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_DYNAMIC_EMBEDDING_MANAGER_H_
