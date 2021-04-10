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

  // Update the gradients of the embeddings for given keys.
  absl::Status UpdateGradients(const tensorflow::Tensor& keys,
                               const tensorflow::Tensor& grads);

  // Returns DynamicEmbeddingConfig.
  const DynamicEmbeddingConfig& config() { return config_; }

  // Samples negative keys from given positive keys and compute the dot products
  // between the embeddings of the positive/negative keys and the input
  // activations.
  //
  // If update = true, new embeddings are dynamically allocated for new
  // positive keys, which is often used in training.
  //
  // Note that for a logit layer with activation x in the last layer, one needs
  // to append an extra 1 to the input activations to obtain wx + b, where [w,
  // b] is the embedding of a particular output key.
  //
  // The `output_labels` indicates if the corresponding `output_keys` is a
  // positive or negative sample, and the `output_expected_counts` represents
  // the sampling probability. Please refer to
  // carls.candidate_sampling.NegativeSamplingResult for details.
  //
  // `output_mask` indicates whether `positive_keys` of an entry in the input
  // batch are all invalid (empty).
  //
  // `output_embedding` returns the embeddings of the sampled keys. It should
  // be allocated as [batch_size, num_samples, embed_dim]. This is needed for
  // computing the gradients w.r.t. the input_activations.
  absl::Status NegativeSamplingWithLogits(
      const tensorflow::Tensor& positive_keys,
      const tensorflow::Tensor& input_activations, int num_samples, bool update,
      tensorflow::Tensor* output_keys, tensorflow::Tensor* output_logits,
      tensorflow::Tensor* output_labels,
      tensorflow::Tensor* output_expected_counts,
      tensorflow::Tensor* output_masks, tensorflow::Tensor* output_embeddings);

  // Return top k closest embeddings to each of the input activations.
  // Note that for a logit layer with activation x, one need to append an extra
  // 1 to the input activations to obtain wx + b, where [w, b] is the embedding
  // of a particular output key.
  absl::Status TopK(const tensorflow::Tensor& input_activations, int k,
                    tensorflow::Tensor* output_keys,
                    tensorflow::Tensor* output_logits);

  // Calls the KnowledgeBankService::Export RPC.
  absl::Status Export(const std::string& output_dir,
                      std::string* exported_path);

  // Calls the KnowledgeBankService::Import RPC.
  absl::Status Import(const std::string& saved_path);

 private:
  // Check validity of input for both UpdateValues() and UpdateGradients().
  absl::Status CheckInputForUpdate(const tensorflow::Tensor& keys,
                                   const tensorflow::Tensor& values);

  // Internal implementation of the Lookup() method.
  absl::Status LookupInternal(const tensorflow::Tensor& keys, bool update,
                              LookupResponse* response);

  std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub_;
  const DynamicEmbeddingConfig config_;
  const std::string session_handle_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_DYNAMIC_EMBEDDING_MANAGER_H_
