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

#include "research/carls/knowledge_bank/knowledge_bank.h"

#include "absl/status/status.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/knowledge_bank/initializer_helper.h"

namespace carls {
namespace {

constexpr char kSavedMetadataFilename[] = "embedding_store_meta_data.pbtxt";

}  // namespace

KnowledgeBank::KnowledgeBank(const KnowledgeBankConfig& config,
                             const int embedding_dimension)
    : config_(config), embedding_dimension_(embedding_dimension) {
  CHECK_GT(embedding_dimension_, 0);
  absl::Status status =
      ValidateInitializer(embedding_dimension_, config_.initializer());
  CHECK(status.ok()) << status;
}

KnowledgeBank::~KnowledgeBank() {}

void KnowledgeBank::BatchLookup(
    const std::vector<absl::string_view>& keys,
    std::vector<absl::variant<EmbeddingVectorProto, std::string>>*
        value_or_errors) const {
  CHECK(value_or_errors != nullptr);
  if (keys.empty()) {
    return;
  }
  value_or_errors->clear();
  value_or_errors->reserve(keys.size());
  for (const absl::string_view& key : keys) {
    EmbeddingVectorProto result;
    const auto status = Lookup(key, &result);
    if (!status.ok()) {
      value_or_errors->push_back(std::string(status.message()));
    } else {
      value_or_errors->push_back(std::move(result));
    }
  }
}

void KnowledgeBank::BatchLookupWithUpdate(
    const std::vector<absl::string_view>& keys,
    std::vector<absl::variant<EmbeddingVectorProto, std::string>>*
        value_or_errors) {
  CHECK(value_or_errors != nullptr);
  if (keys.empty()) {
    return;
  }
  value_or_errors->clear();
  value_or_errors->reserve(keys.size());
  for (const absl::string_view& key : keys) {
    EmbeddingVectorProto result;
    const auto status = LookupWithUpdate(key, &result);
    if (!status.ok()) {
      value_or_errors->push_back(std::string(status.message()));
    } else {
      value_or_errors->push_back(std::move(result));
    }
  }
}

std::vector<absl::Status> KnowledgeBank::BatchUpdate(
    const std::vector<absl::string_view>& keys,
    const std::vector<EmbeddingVectorProto>& values) {
  CHECK(keys.size() == values.size());
  std::vector<absl::Status> statuses;
  if (keys.empty()) {
    return statuses;
  }
  statuses.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    statuses.emplace_back(Update(keys[i], values[i]));
  }
  return statuses;
}

absl::Status KnowledgeBank::Export(const std::string& export_directory,
                                   const std::string& subdir,
                                   std::string* checkpoint) {
  CHECK(checkpoint != nullptr);
  if (!IsDirectory(export_directory).ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Nonexistent export_directory:", export_directory));
  }
  const std::string dirname = JoinPath(export_directory, subdir);
  if (!IsDirectory(dirname).ok()) {
    // This directory may have already been created by other tasks.
    auto status = RecursivelyCreateDir(dirname);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "RecursivelyCreateDir failed with error:", status.message()));
    }
  }
  std::string exported_path;
  auto status = ExportInternal(dirname, &exported_path);
  if (!status.ok()) {
    return status;
  }

  KnowledgeBankCheckpointMetaData meta_data;
  *meta_data.mutable_config() = config_;
  *meta_data.mutable_checkpoint_saved_path() = exported_path;
  // Output Summary.
  *checkpoint = JoinPath(dirname, kSavedMetadataFilename);
  status = WriteTextProto(*checkpoint, meta_data, /*can_overwrite=*/true);
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

absl::Status KnowledgeBank::Import(const std::string& saved_path) {
  KnowledgeBankCheckpointMetaData meta_data;
  auto status = ReadTextProto(saved_path, &meta_data);
  if (!status.ok()) {
    return status;
  }
  if (!meta_data.checkpoint_saved_path().empty()) {
    auto status = ImportInternal(meta_data.checkpoint_saved_path());
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace carls
