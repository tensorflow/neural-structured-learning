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

#include "research/carls/knowledge_bank_grpc_service.h"

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/base/status_helper.h"

namespace carls {
namespace {

using grpc::Status;
using grpc::StatusCode;

}  // namespace

KnowledgeBankGrpcServiceImpl::KnowledgeBankGrpcServiceImpl() {}

KnowledgeBankGrpcServiceImpl::~KnowledgeBankGrpcServiceImpl() {}

Status KnowledgeBankGrpcServiceImpl::StartSession(
    grpc::ServerContext* context, const StartSessionRequest* request,
    StartSessionResponse* response) {
  if (request->name().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Name is empty.");
  }
  const std::string session_handle = request->SerializeAsString();
  const auto status = StartSessionIfNecessary(session_handle);
  if (!status.ok()) {
    return status;
  }
  response->set_session_handle(session_handle);
  return Status::OK;
}

Status KnowledgeBankGrpcServiceImpl::Lookup(grpc::ServerContext* context,
                                            const LookupRequest* request,
                                            LookupResponse* response) {
  if (request->session_handle().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "session_handle is empty.");
  }
  if (request->key().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Empty input keys.");
  }
  const auto status = StartSessionIfNecessary(request->session_handle());
  if (!status.ok()) {
    return status;
  }
  std::vector<absl::string_view> keys(request->key().begin(),
                                      request->key().end());

  absl::ReaderMutexLock lock(&map_mu_);
  std::vector<absl::variant<EmbeddingVectorProto, std::string>> value_or_errors;
  if (request->update()) {
    kb_map_[request->session_handle()]->BatchLookupWithUpdate(keys,
                                                              &value_or_errors);
  } else {
    kb_map_[request->session_handle()]->BatchLookup(keys, &value_or_errors);
  }
  if (value_or_errors.size() != keys.size()) {
    return Status(StatusCode::INTERNAL,
                  "Inconsistent result returned by BatchLookup()");
  }
  auto& embedding_table = *response->mutable_embedding_table();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!absl::holds_alternative<EmbeddingVectorProto>(value_or_errors[i])) {
      continue;
    }
    embedding_table[request->key(i)] =
        std::move(absl::get<EmbeddingVectorProto>(value_or_errors[i]));
  }
  return Status::OK;
}

Status KnowledgeBankGrpcServiceImpl::Update(grpc::ServerContext* context,
                                            const UpdateRequest* request,
                                            UpdateResponse* response) {
  if (request->session_handle().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "session_handle is empty.");
  }
  if (request->values().empty() && request->gradients().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "input is empty.");
  }
  const auto status = StartSessionIfNecessary(request->session_handle());
  if (!status.ok()) {
    return status;
  }

  if (!request->values().empty()) {
    std::vector<absl::string_view> keys;
    std::vector<EmbeddingVectorProto> values;
    keys.reserve(request->values_size());
    values.reserve(request->values_size());
    for (const auto& iter : request->values()) {
      keys.push_back(iter.first);
      values.push_back(iter.second);
    }

    absl::WriterMutexLock lock(&map_mu_);
    kb_map_[request->session_handle()]->BatchUpdate(keys, values);
  }

  if (!request->gradients().empty()) {
    // Collect variables and gradients.
    std::vector<absl::string_view> keys;
    std::vector<absl::string_view> valid_keys;
    std::vector<EmbeddingVectorProto> embeddings;
    std::vector<const EmbeddingVectorProto*> gradients;
    keys.reserve(request->gradients().size());
    valid_keys.reserve(request->gradients().size());
    embeddings.reserve(request->gradients().size());
    gradients.reserve(request->gradients().size());
    for (auto& pair : request->gradients()) {
      keys.push_back(pair.first);
      gradients.push_back(&pair.second);
    }

    absl::WriterMutexLock lock(&map_mu_);
    if (!gd_map_.contains(request->session_handle())) {
      return Status(StatusCode::INTERNAL,
                    "Optimizer is not created, did you forget to add "
                    "gradient_descent_config in DynamicEmbeddingConfig?");
    }

    // Step One: find the embeddings of given keys.
    std::vector<absl::variant<EmbeddingVectorProto, std::string>>
        value_or_errors;
    kb_map_[request->session_handle()]->BatchLookup(keys, &value_or_errors);

    if (value_or_errors.size() != keys.size()) {
      return Status(StatusCode::INTERNAL,
                    "Inconsistent result returned by BatchLookup()");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!absl::holds_alternative<EmbeddingVectorProto>(value_or_errors[i])) {
        continue;
      }
      valid_keys.push_back(keys[i]);
      embeddings.push_back(
          std::move(absl::get<EmbeddingVectorProto>(value_or_errors[i])));
    }
    if (valid_keys.empty()) {
      return Status(StatusCode::INTERNAL, "No valid keys for gradient update.");
    }

    // Step Two: apply gradient update.
    std::string error_msg;
    auto updated_embeddings = gd_map_[request->session_handle()]->Apply(
        embeddings, gradients, &error_msg);
    if (updated_embeddings.empty()) {
      return Status(
          StatusCode::INTERNAL,
          absl::StrCat("Applying gradient update returned error: ", error_msg));
    }

    // Step Three: update the embeddings.
    kb_map_[request->session_handle()]->BatchUpdate(valid_keys,
                                                    updated_embeddings);
  }
  return Status::OK;
}

grpc::Status KnowledgeBankGrpcServiceImpl::Sample(grpc::ServerContext* context,
                                                  const SampleRequest* request,
                                                  SampleResponse* response) {
  if (request->session_handle().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "session_handle is empty.");
  }
  if (request->sample_context().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "No sample context.");
  }
  const auto status = StartSessionIfNecessary(request->session_handle());
  if (!status.ok()) {
    return status;
  }
  absl::MutexLock lock(&map_mu_);
  auto& knowledge_bank = *kb_map_[request->session_handle()];
  // Add new keys into the knowledge bank if necessary.
  if (request->update()) {
    absl::flat_hash_set<absl::string_view> keys;
    for (const auto& context : request->sample_context()) {
      for (const auto& key : context.positive_key()) {
        if (!knowledge_bank.Contains(key)) {
          keys.insert(key);
        }
      }
    }
    if (!keys.empty()) {
      std::vector<absl::variant<EmbeddingVectorProto, std::string>> results;
      std::vector<absl::string_view> positive_keys(keys.begin(), keys.end());
      knowledge_bank.BatchLookupWithUpdate(positive_keys, &results);
    }
  }

  for (const auto& sample_context : request->sample_context()) {
    std::vector<candidate_sampling::SampledResult> results;
    auto status = cs_map_[request->session_handle()]->Sample(
        knowledge_bank, sample_context, request->num_samples(), &results);
    if (!status.ok()) {
      return ToGrpcStatus(status);
    }
    auto* samples = response->add_samples();
    for (auto& result : results) {
      *samples->add_sampled_result() = std::move(result);
    }
  }
  return Status::OK;
}

Status KnowledgeBankGrpcServiceImpl::Export(grpc::ServerContext* context,
                                            const ExportRequest* request,
                                            ExportResponse* response) {
  if (request->session_handle().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "session_handle is empty.");
  }
  if (request->export_directory().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "export_directory is empty.");
  }
  const auto status = StartSessionIfNecessary(request->session_handle());
  if (!status.ok()) {
    return status;
  }
  StartSessionRequest start_request;
  start_request.ParseFromString(request->session_handle());
  absl::MutexLock lock(&map_mu_);
  return ToGrpcStatus(kb_map_[request->session_handle()]->Export(
      request->export_directory(), start_request.name(),
      response->mutable_knowledge_bank_saved_path()));
}

Status KnowledgeBankGrpcServiceImpl::Import(grpc::ServerContext* context,
                                            const ImportRequest* request,
                                            ImportResponse* response) {
  if (request->session_handle().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "session_handle is empty.");
  }
  if (request->knowledge_bank_saved_path().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "knowledge_bank_saved_path is empty.");
  }
  const auto status = StartSessionIfNecessary(request->session_handle());
  if (!status.ok()) {
    return status;
  }
  absl::MutexLock lock(&map_mu_);
  return ToGrpcStatus(kb_map_[request->session_handle()]->Import(
      request->knowledge_bank_saved_path()));
}

size_t KnowledgeBankGrpcServiceImpl::KnowledgeBankSize() {
  absl::ReaderMutexLock lock(&map_mu_);
  return kb_map_.size();
}

Status KnowledgeBankGrpcServiceImpl::StartSessionIfNecessary(
    const std::string& session_handle) {
  StartSessionRequest request;
  request.ParseFromString(session_handle);
  absl::MutexLock lock(&map_mu_);
  if (!kb_map_.contains(session_handle)) {
    // Creates a new KnowledgeBank.
    auto knowledge_bank =
        KnowledgeBankFactory::Make(request.config().knowledge_bank_config(),
                                   request.config().embedding_dimension());
    if (knowledge_bank == nullptr) {
      return Status(StatusCode::INTERNAL, "Creating KnowledgeBank failed.");
    }
    kb_map_[session_handle] = std::move(knowledge_bank);
  }
  if (request.config().has_gradient_descent_config() &&
      !gd_map_.contains(session_handle)) {
    auto optimizer = GradientDescentOptimizer::Create(
        request.config().embedding_dimension(),
        request.config().gradient_descent_config());
    if (optimizer == nullptr) {
      return Status(StatusCode::INTERNAL,
                    "Creating GradientDescentOptimizer failed.");
    }
    gd_map_[session_handle] = std::move(optimizer);
  }
  if (request.config().has_candidate_sampler_config() &&
      !cs_map_.contains(session_handle)) {
    auto sampler = candidate_sampling::SamplerFactory::Make(
        request.config().candidate_sampler_config());
    if (sampler == nullptr) {
      return Status(StatusCode::INTERNAL, "Creating CandidateSampler failed.");
    }
    cs_map_[session_handle] = std::move(sampler);
  }
  return Status::OK;
}

}  // namespace carls
