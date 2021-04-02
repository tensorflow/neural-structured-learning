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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_KNOWLEDGE_BANK_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_KNOWLEDGE_BANK_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "research/carls/base/proto_factory.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/knowledge_bank_config.pb.h"  // proto to pb

namespace carls {

// Macro for registering a knowledge bank implementation.
#define REGISTER_KNOWLEDGE_BANK_FACTORY(proto_type, factory_type)         \
  REGISTER_CARLS_FACTORY_1(proto_type, factory_type, KnowledgeBankConfig, \
                           KnowledgeBank, int)

// Base class for an KnowledgeBank.
class KnowledgeBank {
 public:
  virtual ~KnowledgeBank();

  // Looks up the embedding for a single key without updating the embedding.
  virtual absl::Status Lookup(const absl::string_view key,
                              EmbeddingVectorProto* result) const = 0;

  // Looks up the embedding for a single key, and if the key is not found,
  // create a new embedding based on given initializer in KnowledgeBankConfig.
  virtual absl::Status LookupWithUpdate(const absl::string_view key,
                                        EmbeddingVectorProto* result) = 0;

  // Updates the embedding of a single key.
  // The function may move the value into the storage.
  virtual absl::Status Update(const absl::string_view key,
                              const EmbeddingVectorProto& value) = 0;

  // Batch lookup for the given keys.
  // Returns a vector of variant [EmbeddingVectorProto, error message] with the
  // same length as the input keys.
  virtual void BatchLookup(
      const std::vector<absl::string_view>& keys,
      std::vector<absl::variant<EmbeddingVectorProto, std::string>>*
          value_or_errors) const;

  // Batch lookup with update for the given keys.
  // Returns a vector of variant [EmbeddingVectorProto, error message] with the
  // same length as the input keys.
  virtual void BatchLookupWithUpdate(
      const std::vector<absl::string_view>& keys,
      std::vector<absl::variant<EmbeddingVectorProto, std::string>>*
          value_or_errors);

  // Batch update.
  // Since the update is done one by one, it is not guaranteed that
  // atomic commit/cancellation if only part of the updates are successful.
  virtual std::vector<absl::Status> BatchUpdate(
      const std::vector<absl::string_view>& keys,
      const std::vector<EmbeddingVectorProto>& values);

  // Exports current data to a timestamped output directory with given subdir,
  // e.g., %export_directory%/%subdir%
  // The checkpoint contains the full file path of the saved binary proto of the
  // KnowledgeBankConfig upon success.
  absl::Status Export(const std::string& export_directory,
                      const std::string& subdir, std::string* checkpoint);

  // Restores the stage of the embedding from the given saved path.
  absl::Status Import(const std::string& saved_path);

  // Returns embedding dimension.
  int embedding_dimension() const { return embedding_dimension_; }

  // Returns KnowledgeBankConfig.
  const KnowledgeBankConfig& config() { return config_; }

  // Returns the total number of keys in the knowledge store.
  virtual size_t Size() const = 0;

  // Returns the list of keys in the knowledge bank.
  virtual std::vector<absl::string_view> Keys() const = 0;

 protected:
  KnowledgeBank(const KnowledgeBankConfig& config,
                const int embedding_dimension);

  // Internal implementation of the Export() method.
  // The caller is expected to export the files to %output_dir%/%subdir%.
  virtual absl::Status ExportInternal(const std::string& dir,
                                      std::string* exported_path) = 0;

  // Internal implementation of the Import() method.
  virtual absl::Status ImportInternal(const std::string& saved_path) = 0;

  KnowledgeBankConfig config_;
  const int embedding_dimension_;
};

REGISTER_CARLS_BASE_CLASS_1(KnowledgeBankConfig, KnowledgeBank,
                            KnowledgeBankFactory, int);

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KNOWLEDGE_BANK_KNOWLEDGE_BANK_H_
