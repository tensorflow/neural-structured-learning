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

#include "research/carls/dynamic_embedding_manager.h"

#include "grpcpp/create_channel.h"  // net
// Placeholder for internal channel credential  // net
#include "grpcpp/security/credentials.h"  // net
#include "grpcpp/support/time.h"  // net
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "grpc/impl/codegen/gpr_types.h"
#include "grpcpp/impl/codegen/client_context.h"  // third_party
#include "research/carls/base/status_helper.h"

ABSL_FLAG(std::string, kbs_address, "", "Address to a KBS server.");
ABSL_FLAG(double, des_rpc_deadline_sec, 10,
          "Timeout for connecting to a DES server.");

namespace carls {
namespace {

using ::tensorflow::tstring;

#ifndef INTERNAL_DIE_IF_NULL
#define INTERNAL_DIE_IF_NULL(val) DieIfNull(__FILE__, __LINE__, #val, (val))
#endif

template <typename T>
T DieIfNull(const char* file, int line, const char* exprtext, T&& t) {
  CHECK(t != nullptr) << exprtext;
  return std::forward<T>(t);
}

}  // namespace

// Static.
// This function is called for constructing a DynamicEmbeddingManager with given
// parameters.
std::unique_ptr<DynamicEmbeddingManager> DynamicEmbeddingManager::Create(
    const DynamicEmbeddingConfig& config, const std::string& name,
    const std::string& kbs_address, absl::Duration timeout) {
  const std::string service_address =
      kbs_address.empty() ? absl::GetFlag(FLAGS_kbs_address) : kbs_address;
  if (service_address.empty()) {
    LOG(ERROR) << "kbs_address is empty.";
    return nullptr;
  }

  // Starts a channel to DES and creates a stub.
  std::shared_ptr<grpc::ChannelCredentials> credentials =
      grpc::InsecureChannelCredentials();
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(service_address, credentials);
  if (channel == nullptr) {
    LOG(ERROR) << "grpc::CreateChannel() failed.";
    return nullptr;
  }
  std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub =
      /*grpc_gen::*/KnowledgeBankService::NewStub(channel);
  if (stub == nullptr) {
    LOG(ERROR) << "Creating KnowledgeBankService stub failed.";
    return nullptr;
  }

  // Starts a session.
  StartSessionRequest request;
  *request.mutable_config() = config;
  request.set_name(name);
  StartSessionResponse response;
  grpc::ClientContext context;
  if (timeout != absl::InfiniteDuration()) {
    context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }
  context.set_wait_for_ready(true);
  auto status = stub->StartSession(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "StartSession failed with error: " << status.error_message();
    return nullptr;
  }
  if (response.session_handle().empty()) {
    LOG(ERROR) << "StartSession returned empty session_handle.";
    return nullptr;
  }
  return absl::make_unique<DynamicEmbeddingManager>(std::move(stub), config,
                                                    response.session_handle());
}

DynamicEmbeddingManager::DynamicEmbeddingManager(
    std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub,
    const DynamicEmbeddingConfig& config, const std::string& session_handle)
    : stub_(std::move(INTERNAL_DIE_IF_NULL(stub))),
      config_(config),
      session_handle_(session_handle) {}

absl::Status DynamicEmbeddingManager::Lookup(const tensorflow::Tensor& keys,
                                             bool update,
                                             tensorflow::Tensor* output) {
  CHECK(output != nullptr);
  if (!(keys.dims() == 1 || keys.dims() == 2)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input dimension must be either 1 or 2, got ", keys.dims()));
  }
  if (keys.NumElements() == 0) {
    return absl::InvalidArgumentError("No input.");
  }
  LookupResponse lookup_response;
  const auto lookup_status = LookupInternal(keys, update, &lookup_response);
  if (!lookup_status.ok()) {
    return lookup_status;
  }

  // Process results.
  // This is usually used in the case when the input is a batch of keys.
  if (keys.dims() == 1) {
    const auto key_values = keys.flat<tstring>();
    auto output_values = output->matrix<float>();  // (key_size, dim_size)
    for (int i = 0; i < keys.NumElements(); ++i) {
      const auto& key = key_values(i);
      if (key.empty()) {
        for (int d = 0; d < config_.embedding_dimension(); ++d) {
          output_values(i, d) = 0;
        }
        continue;
      }

      const auto& embedding_table = lookup_response.embedding_table();
      const auto lookup_iter = embedding_table.find(key);
      if (lookup_iter == embedding_table.end()) {
        return absl::InternalError(absl::StrCat(
            std::string(key), " is not in the Lookup result, unexpected."));
      }
      const auto& embedding = lookup_iter->second;
      for (int d = 0; d < embedding.value_size(); ++d) {
        output_values(i, d) = embedding.value(d);
      }
    }
    return absl::OkStatus();
  }

  // keys.dims() == 2
  // This is usually used in the case when the input is a batch of sequences.
  const auto key_values = keys.matrix<tstring>();
  // (batch_size, max_sequence_size, dim_size)
  auto output_values = output->tensor<float, 3>();
  for (int b = 0; b < keys.dim_size(0); ++b) {
    for (int i = 0; i < keys.dim_size(1); ++i) {
      const auto& key = key_values(b, i);
      if (key.empty()) {
        for (int d = 0; d < config_.embedding_dimension(); ++d) {
          output_values(b, i, d) = 0;
        }
        continue;
      }

      const auto& embedding_table = lookup_response.embedding_table();
      const auto lookup_iter = embedding_table.find(key);
      if (lookup_iter == embedding_table.end()) {
        return absl::InternalError(absl::StrCat(
            std::string(key), " is not in the Lookup result, unexpected."));
      }
      const auto& embedding = lookup_iter->second;
      for (int d = 0; d < embedding.value_size(); ++d) {
        output_values(b, i, d) = embedding.value(d);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status DynamicEmbeddingManager::CheckInputForUpdate(
    const tensorflow::Tensor& keys, const tensorflow::Tensor& values) {
  if (keys.NumElements() == 0) {
    return absl::InvalidArgumentError("Input key is empty.");
  }
  const int num_keys = keys.NumElements();
  const int emb_dim = values.dim_size(values.dims() - 1);
  const int num_values = values.NumElements() / emb_dim;

  if (num_keys != num_values) {
    return absl::InvalidArgumentError(
        absl::StrCat("Inconsistent keys size and values size: ", num_keys,
                     " v.s. ", num_values));
  }
  if (emb_dim != config_.embedding_dimension()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Inconsistent embedding dimension, got ", emb_dim,
                     " expect ", config_.embedding_dimension()));
  }
  return absl::OkStatus();
}

absl::Status DynamicEmbeddingManager::UpdateValues(
    const tensorflow::Tensor& keys, const tensorflow::Tensor& values) {
  auto status = CheckInputForUpdate(keys, values);
  if (!status.ok()) {
    return status;
  }
  const auto key_values = keys.flat<tstring>();
  const int num_keys = keys.NumElements();
  const int emb_dim = values.dim_size(values.dims() - 1);

  UpdateRequest update_request;
  update_request.set_session_handle(session_handle_);
  // shape (key_size, dim_size)
  const auto emb_values = values.flat_inner_dims<float>();
  for (int b = 0; b < num_keys; ++b) {
    const std::string key_value = key_values(b);
    if (key_value.empty()) {
      continue;
    }
    auto* emb = &(*update_request.mutable_values())[key_value];
    emb->clear_value();
    // If a key shows up in a batch multiple times, do not add up.
    for (int i = 0; i < emb_dim; ++i) {
      emb->add_value(emb_values(b, i));
    }
  }

  UpdateResponse update_response;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_des_rpc_deadline_sec))));
  return ToAbslStatus(
      stub_->Update(&context, update_request, &update_response));
}

absl::Status DynamicEmbeddingManager::LookupInternal(
    const tensorflow::Tensor& keys, bool update, LookupResponse* response) {
  CHECK(response != nullptr);
  const auto key_values = keys.flat<tstring>();

  LookupRequest request;
  request.set_update(update);
  request.set_session_handle(session_handle_);
  for (int i = 0; i < keys.NumElements(); ++i) {
    if (!key_values(i).empty()) {
      request.add_key(std::string(key_values(i)));
    }
  }

  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_des_rpc_deadline_sec))));
  return ToAbslStatus(stub_->Lookup(&context, request, response));
}

absl::Status DynamicEmbeddingManager::UpdateGradients(
    const tensorflow::Tensor& keys, const tensorflow::Tensor& grads) {
  auto status = CheckInputForUpdate(keys, grads);
  if (!status.ok()) {
    return status;
  }
  const auto key_values = keys.flat<tstring>();
  const int num_keys = keys.NumElements();
  const int emb_dim = grads.dim_size(grads.dims() - 1);

  // Prepare update request.
  UpdateRequest update_request;
  update_request.set_session_handle(session_handle_);

  auto grad_values = grads.flat_inner_dims<float>();  // (num_keys, dim_size)
  for (int b = 0; b < num_keys; ++b) {
    const std::string& key_value = key_values(b);
    if (key_value.empty()) {
      continue;
    }
    auto* emb = &(*update_request.mutable_gradients())[key_value];
    // Initializes the embedding values if they are not set.
    while (emb->value_size() < emb_dim) {
      emb->add_value(0.0);
    }
    // If a key shows up in a batch multiple times, add their gradients up.
    for (int i = 0; i < emb_dim; ++i) {
      emb->set_value(i, emb->value(i) + grad_values(b, i));
    }
  }

  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_des_rpc_deadline_sec))));
  UpdateResponse update_response;
  return ToAbslStatus(
      stub_->Update(&context, update_request, &update_response));
}

}  // namespace carls
