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

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/knowledge_bank/initializer_helper.h"
#include "research/carls/knowledge_bank/knowledge_bank.h"

namespace carls {

// An implementation of KnowledgeBank using protocol buffer as its internal
// storage of embedding data.
class InProtoKnowledgeBank : public KnowledgeBank {
 public:
  InProtoKnowledgeBank(const KnowledgeBankConfig& config, int dimension)
      : KnowledgeBank(config, dimension) {}

 private:
  // Implementation of the Lookup interface.
  absl::Status Lookup(const absl::string_view key,
                      EmbeddingVectorProto* result) const override;

  // Implementation of the LookupWithUpdate interface.
  absl::Status LookupWithUpdate(const absl::string_view key,
                                EmbeddingVectorProto* result) override;

  // Updates the embedding of a single key.
  absl::Status Update(const absl::string_view key,
                      const EmbeddingVectorProto& value) override;

  mutable absl::Mutex mu_;
  InProtoKnowledgeBankConfig in_proto_config_ ABSL_GUARDED_BY(mu_);
};

REGISTER_KNOWLEDGE_BANK_FACTORY(
    InProtoKnowledgeBankConfig,
    [](const KnowledgeBankConfig& config,
       int dimension) -> std::unique_ptr<KnowledgeBank> {
      if (dimension <= 0) {
        LOG(ERROR) << "Invalid dimension: " << dimension;
        return nullptr;
      }
      auto status = ValidateInitializer(dimension, config.initializer());
      if (!status.ok()) {
        LOG(ERROR) << status;
        return nullptr;
      }
      return std::unique_ptr<KnowledgeBank>(
          new InProtoKnowledgeBank(config, dimension));
    });

absl::Status InProtoKnowledgeBank::Lookup(const absl::string_view key,
                                          EmbeddingVectorProto* result) const {
  CHECK(result != nullptr);
  absl::ReaderMutexLock l(&mu_);
  const auto& embedding_table =
      in_proto_config_.embedding_data().embedding_table();
  const auto lookup_iter = embedding_table.find(std::string(key));
  if (lookup_iter == embedding_table.end()) {
    return absl::InvalidArgumentError(absl::StrCat("Key is not found: ", key));
  }
  *result = lookup_iter->second;
  return absl::OkStatus();
}

absl::Status InProtoKnowledgeBank::LookupWithUpdate(
    const absl::string_view key, EmbeddingVectorProto* result) {
  absl::WriterMutexLock l(&mu_);
  auto* embedding_table =
      in_proto_config_.mutable_embedding_data()->mutable_embedding_table();
  std::string key_str(key);
  if (!embedding_table->contains(key_str)) {
    // Insert a new embedding.
    EmbeddingVectorProto embed =
        InitializeEmbedding(embedding_dimension(), config().initializer());
    embed.set_tag(key_str);
    (*embedding_table)[key_str] = std::move(embed);
  }
  auto& value = (*embedding_table)[key_str];
  // Incement frequency by one for each lookup with update.
  value.set_weight(value.weight() + 1);
  *result = value;
  return absl::OkStatus();
}

absl::Status InProtoKnowledgeBank::Update(const absl::string_view key,
                                          const EmbeddingVectorProto& value) {
  absl::WriterMutexLock l(&mu_);
  (*in_proto_config_.mutable_embedding_data()
        ->mutable_embedding_table())[std::string(key)] = value;
  return absl::OkStatus();
}

}  // namespace carls
