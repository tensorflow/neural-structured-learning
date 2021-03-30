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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KERNELS_DYNAMIC_EMBEDDING_MANAGER_RESOURCE_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KERNELS_DYNAMIC_EMBEDDING_MANAGER_RESOURCE_H_

#include "research/carls/dynamic_embedding_config.pb.h"  // proto to pb
#include "research/carls/dynamic_embedding_manager.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace carls {

// Uses a ResourceOpKernel to manage the DynamicEmbeddingManager object, which
// talks to a KnowledgeBankService.
class DynamicEmbeddingManagerResource : public tensorflow::ResourceBase {
 public:
  DynamicEmbeddingManagerResource(const DynamicEmbeddingConfig& config,
                                  const std::string& var_name,
                                  const std::string& kbs_address,
                                  absl::Duration timeout);
  ~DynamicEmbeddingManagerResource() override = default;

  std::string DebugString() const override { return "DEM resource"; }

  DynamicEmbeddingManager* manager() { return manager_.get(); }

 private:
  std::unique_ptr<DynamicEmbeddingManager> manager_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KERNELS_DYNAMIC_EMBEDDING_MANAGER_RESOURCE_H_
