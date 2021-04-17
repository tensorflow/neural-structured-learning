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
ABSL_FLAG(double, kbs_rpc_deadline_sec, 10,
          "Timeout for connecting to a DES server.");

namespace carls {
namespace {

using ::tensorflow::Tensor;
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

absl::Status DynamicEmbeddingManager::Lookup(const Tensor& keys, bool update,
                                             Tensor* output) {
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
    const Tensor& keys, const Tensor& values) {
  RET_CHECK_TRUE(keys.NumElements() > 0) << "Input key is empty.";
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

absl::Status DynamicEmbeddingManager::UpdateValues(const Tensor& keys,
                                                   const Tensor& values) {
  RET_CHECK_OK(CheckInputForUpdate(keys, values));
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
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
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
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  return ToAbslStatus(stub_->Lookup(&context, request, response));
}

absl::Status DynamicEmbeddingManager::UpdateGradients(const Tensor& keys,
                                                      const Tensor& grads) {
  RET_CHECK_OK(CheckInputForUpdate(keys, grads));
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
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  UpdateResponse update_response;
  return ToAbslStatus(
      stub_->Update(&context, update_request, &update_response));
}

absl::Status DynamicEmbeddingManager::NegativeSampling(
    const Tensor& positive_keys, const Tensor& input_activations,
    const int num_samples, const bool update, Tensor* output_keys,
    Tensor* output_labels, Tensor* output_expected_counts, Tensor* output_masks,
    Tensor* output_embeddings) {
  RET_CHECK_TRUE(config_.embedding_dimension() > 0)
      << "Invalid embedding dimension:" << config_.embedding_dimension();
  RET_CHECK_TRUE(num_samples > 0);

  // Shape of input: [d1, d2, ..., inner_dim].
  const int dims = input_activations.dims();
  const int inner_dim = input_activations.dim_size(dims - 1);
  RET_CHECK_TRUE(inner_dim == config_.embedding_dimension())
      << inner_dim << " v.s. " << config_.embedding_dimension();
  const int batch_size =
      input_activations.NumElements() / config_.embedding_dimension();

  // Processes positive keys.
  SampleRequest sample_request;
  sample_request.set_session_handle(session_handle_);
  sample_request.set_num_samples(num_samples);
  sample_request.set_update(update);
  const auto pos_key_values = positive_keys.flat_inner_dims<tstring>();
  RET_CHECK_TRUE(pos_key_values.dimension(0) == batch_size)
      << pos_key_values.dimension(0) << " v.s. " << batch_size;
  for (int b = 0; b < batch_size; ++b) {
    auto* sample_context = sample_request.add_sample_context();
    for (int i = 0; i < positive_keys.dim_size(1); ++i) {
      if (!pos_key_values(b, i).empty()) {
        sample_context->add_positive_key(std::string(pos_key_values(b, i)));
      }
    }
  }

  // Calls the Sample RPC.
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  SampleResponse sample_response;
  RET_CHECK_OK(stub_->Sample(&context, sample_request, &sample_response));
  RET_CHECK_TRUE(sample_response.samples_size() == batch_size);

  // Process sampled results.
  auto output_keys_values = output_keys->flat_inner_dims<tstring>();
  auto label_values = output_labels->flat_inner_dims<float>();
  auto expected_count_values = output_expected_counts->flat_inner_dims<float>();
  auto mask_values = output_masks->flat<float>();
  auto embedding_values = output_embeddings->flat_inner_dims<float, 3>();
  for (int b = 0; b < batch_size; ++b) {
    // Use auto& such that we can directly move some contents of samples into
    // the output for efficiency.
    auto& samples = *sample_response.mutable_samples(b);

    // If no sample result is returned, set the default values for output
    // tensors.
    if (samples.sampled_result().empty()) {
      mask_values(b) = 0.0f;
      for (int i = 0; i < num_samples; ++i) {
        output_keys_values(b, i) = "";
        label_values(b, i) = 0;
        expected_count_values(b, i) = 1;
        for (int d = 0; d < config_.embedding_dimension(); ++d) {
          embedding_values(b, i, d) = 0.0f;
        }
      }
      continue;
    }
    mask_values(b) = 1.0f;

    // Processes the output tensors.
    RET_CHECK_TRUE(samples.sampled_result_size() == num_samples);
    RET_CHECK_TRUE(samples.sampled_result(0).has_negative_sampling_result());
    for (int i = 0; i < samples.sampled_result_size(); ++i) {
      auto& result = *samples.mutable_sampled_result(i)
                          ->mutable_negative_sampling_result();
      const auto& embedding = result.embedding();
      label_values(b, i) = result.is_positive() ? 1.0 : 0.0;
      expected_count_values(b, i) = result.expected_count();
      output_keys_values(b, i) = std::move(result.key());
      for (int d = 0; d < config_.embedding_dimension(); ++d) {
        embedding_values(b, i, d) = embedding.value(d);
      }
    }
  }

  return absl::OkStatus();
}

absl::Status DynamicEmbeddingManager::TopK(
    const tensorflow::Tensor& input_activations, const int k,
    tensorflow::Tensor* output_keys, tensorflow::Tensor* output_logits) {
  RET_CHECK_TRUE(config_.embedding_dimension() > 0)
      << "Invalid embedding dimension:" << config_.embedding_dimension();
  RET_CHECK_TRUE(k > 0);

  // Shape of input: batch_size x hidden_size.
  const int dims = input_activations.dims();
  const int inner_dim = input_activations.dim_size(dims - 1);
  RET_CHECK_TRUE(inner_dim == config_.embedding_dimension());
  const int batch_size =
      input_activations.NumElements() / config_.embedding_dimension();

  // Processes SampleRequest.
  SampleRequest sample_request;
  sample_request.set_session_handle(session_handle_);
  sample_request.set_num_samples(k);
  auto activation_value = input_activations.flat_inner_dims<float>();
  for (int b = 0; b < batch_size; ++b) {
    auto* sample_context = sample_request.add_sample_context();
    for (int i = 0; i < config_.embedding_dimension(); ++i) {
      sample_context->mutable_activation()->add_value(activation_value(b, i));
    }
  }

  // Calls the Sample RPC.
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  SampleResponse sample_response;
  RET_CHECK_OK(stub_->Sample(&context, sample_request, &sample_response));
  RET_CHECK_TRUE(sample_response.samples_size() == batch_size);

  // Process topk results.
  auto output_keys_values = output_keys->flat_inner_dims<tstring>();
  auto logits_values = output_logits->flat_inner_dims<float>();
  for (int b = 0; b < batch_size; ++b) {
    const auto& samples = sample_response.samples(b);
    RET_CHECK_TRUE(samples.sampled_result_size() == k);
    for (int i = 0; i < k; ++i) {
      auto& result = samples.sampled_result(i).topk_sampling_result();
      logits_values(b, i) = result.similarity();
      output_keys_values(b, i) = std::move(result.key());
    }
  }
  return absl::OkStatus();
}

absl::Status DynamicEmbeddingManager::Export(const std::string& output_dir,
                                             std::string* exported_path) {
  CHECK(exported_path != nullptr);
  ExportRequest request;
  request.set_session_handle(session_handle_);
  request.set_export_directory(output_dir);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  ExportResponse response;
  auto status = stub_->Export(&context, request, &response);
  if (!status.ok()) {
    return ToAbslStatus(status);
  }
  *exported_path = response.knowledge_bank_saved_path();
  return absl::OkStatus();
}

absl::Status DynamicEmbeddingManager::Import(const std::string& saved_path) {
  ImportRequest request;
  request.set_session_handle(session_handle_);
  request.set_knowledge_bank_saved_path(saved_path);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       absl::ToChronoSeconds(absl::Seconds(
                           absl::GetFlag(FLAGS_kbs_rpc_deadline_sec))));
  ImportResponse response;
  return ToAbslStatus(stub_->Import(&context, request, &response));
}

}  // namespace carls
