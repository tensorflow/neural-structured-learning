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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_MEMORY_STORE_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_MEMORY_STORE_H_

#include <vector>

#include "absl/status/status.h"
#include "research/carls/base/proto_factory.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/memory_store/memory_store_config.pb.h"  // proto to pb

namespace carls {
namespace memory_store {

// Macro for registering an embedding store implementation.
#define REGISTER_MEMORY_STORE_FACTORY(proto_type, factory_type)         \
  REGISTER_CARLS_FACTORY_0(proto_type, factory_type, MemoryStoreConfig, \
                           MemoryStore)

// Base class for MemoryStore implementation.
class MemoryStore {
 public:
  virtual ~MemoryStore();

  // Looks up memory without changing any internal state.
  // This is usually used in inference model.
  absl::Status BatchLookup(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results);

  // Looks up memory and update internal representation of memory result based
  // on given input.
  // For example, in Gaussian memory, it may update the estimated mean and
  // variance based on new input. Returns the lookup result after the memory is
  // updated.
  absl::Status BatchLookupWithUpdate(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results);

  // Looks up memory and also grows memory clusters if necessary.
  // Note that when memory is growing, its contents are also updated.
  absl::Status BatchLookupWithGrow(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results);

  // Exports current data to a timestamped output directory with given subdir,
  // e.g., %export_directory%/%subdir%
  // It returns the name of the full file path of the export directory upon
  // success.
  absl::Status Export(const std::string& export_directory,
                      const std::string& subdir, std::string* exported_path);

  // Restores the state of the memory from the given saved path.
  absl::Status Import(const std::string& saved_path);

 protected:
  explicit MemoryStore(const MemoryStoreConfig& ms_config);

 private:
  // Internal implementation of the BatchLookup method.
  virtual absl::Status BatchLookupInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) = 0;

  // Internal implementation of the BatchLookupWithUpdate method.
  virtual absl::Status BatchLookupWithUpdateInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) = 0;

  // Internal implementation of the BatchLookupWithGrow method.
  virtual absl::Status BatchLookupWithGrowInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) = 0;

  // Internal implementation of the Export() method.
  virtual absl::Status ExportInternal(const std::string& dir,
                                      std::string* exported_path) = 0;

  // Internal implementation of the Import() method.
  virtual absl::Status ImportInternal(const std::string& saved_path) = 0;

  const MemoryStoreConfig ms_config_;
};

REGISTER_CARLS_BASE_CLASS_0(MemoryStoreConfig, MemoryStore, MemoryStoreFactory);

}  // namespace memory_store
}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_MEMORY_STORE_H_
