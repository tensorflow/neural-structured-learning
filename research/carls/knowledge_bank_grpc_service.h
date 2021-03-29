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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_GRPC_SERVICE_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_GRPC_SERVICE_H_

#include <string>

#include "grpcpp/support/status.h"  // net
#include "absl/synchronization/mutex.h"
#include "research/carls/gradient_descent/gradient_descent_optimizer.h"
#include "research/carls/knowledge_bank/knowledge_bank.h"
#include "research/carls/knowledge_bank_service.grpc.pb.h"

namespace carls {

// Implementation of KnowledgeBank Service for embedding lookup/update.
class KnowledgeBankGrpcServiceImpl final
    : public /*grpc_gen::*/KnowledgeBankService::Service {
 public:
  KnowledgeBankGrpcServiceImpl();
  ~KnowledgeBankGrpcServiceImpl() override;
  KnowledgeBankGrpcServiceImpl(const KnowledgeBankGrpcServiceImpl&) = delete;
  KnowledgeBankGrpcServiceImpl& operator=(const KnowledgeBankGrpcServiceImpl&) =
      delete;

  // Starts a session by generating a session_handle from given name and config,
  // creating an instance of EmbeddingStore if it does not exist and
  // returning the session_handle to the caller for lookup/update/sample, etc.
  grpc::Status StartSession(grpc::ServerContext* context,
                            const StartSessionRequest* request,
                            StartSessionResponse* response) override;

  // Implements the Lookup method of KnowledgeBankService.
  grpc::Status Lookup(grpc::ServerContext* context,
                      const LookupRequest* request,
                      LookupResponse* response) override;

  // Implements the Update method of KnowledgeBankService.
  grpc::Status Update(grpc::ServerContext* context,
                      const UpdateRequest* request,
                      UpdateResponse* response) override;

  // Implements the Export method of KnowledgeBankService.
  grpc::Status Export(grpc::ServerContext* context,
                      const ExportRequest* request,
                      ExportResponse* response) override;

  // Implements the Import method of KnowledgeBankService.
  grpc::Status Import(grpc::ServerContext* context,
                      const ImportRequest* request,
                      ImportResponse* response) override;

  // Returns the number of KnowledgeBank already loaded into KBS.
  size_t KnowledgeBankSize();

 private:
  grpc::Status StartSessionIfNecessary(const std::string& session_handle);

  // Protects maps lookup and update.
  absl::Mutex map_mu_;

  // Maps from session_handle to KnowledgeBank.
  absl::node_hash_map<std::string, std::unique_ptr<KnowledgeBank>> kb_map_;
  // Maps from session_handle to GradientDescentOptimizer.
  absl::node_hash_map<std::string, std::unique_ptr<GradientDescentOptimizer>>
      gd_map_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_GRPC_SERVICE_H_
