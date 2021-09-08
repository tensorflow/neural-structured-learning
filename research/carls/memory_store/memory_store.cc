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

#include "research/carls/memory_store/memory_store.h"

#include "research/carls/base/file_helper.h"
#include "research/carls/base/status_helper.h"

namespace carls {
namespace memory_store {

MemoryStore::MemoryStore(const MemoryStoreConfig& ms_config)
    : ms_config_(ms_config) {}

MemoryStore::~MemoryStore() {}

absl::Status MemoryStore::BatchLookup(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  RET_CHECK_TRUE(!inputs.empty());
  RET_CHECK_TRUE(results != nullptr);
  return BatchLookupInternal(inputs, results);
}

absl::Status MemoryStore::BatchLookupWithUpdate(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  RET_CHECK_TRUE(!inputs.empty());
  RET_CHECK_TRUE(results != nullptr);
  return BatchLookupWithUpdateInternal(inputs, results);
}

absl::Status MemoryStore::BatchLookupWithGrow(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  RET_CHECK_TRUE(!inputs.empty());
  RET_CHECK_TRUE(results != nullptr);
  return BatchLookupWithGrowInternal(inputs, results);
}

absl::Status MemoryStore::Export(const std::string& export_directory,
                                 const std::string& subdir,
                                 std::string* exported_path) {
  auto status = IsDirectory(export_directory);
  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Nonexistent export_directory:", export_directory));
  }
  const std::string dirname = JoinPath(export_directory, subdir);
  if (!IsDirectory(dirname).ok()) {
    // This directory may have already been created by other tasks.
    status = RecursivelyCreateDir(dirname);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "RecursivelyCreateDir failed with error:", status.message()));
    }
  }
  return ExportInternal(dirname, exported_path);
}

absl::Status MemoryStore::Import(const std::string& saved_path) {
  RET_CHECK_TRUE(!saved_path.empty());
  return ImportInternal(saved_path);
}

}  // namespace memory_store
}  // namespace carls
