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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "leveldb/db.h"
#include "research/carls/base/async_node_hash_map.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/base/status_helper.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/knowledge_bank/initializer_helper.h"
#include "research/carls/knowledge_bank/knowledge_bank.h"

namespace carls {
namespace {

constexpr char kMetaDataOutputBaseName[] = "leveldb_embedding_metadata.txt";

}  // namespace

// An implementation of KnowledgeBank using LevelDB as its internal storage of
// embedding data. All methods of this class are thread-safe.
class LeveldbKnowledgeBank : public KnowledgeBank {
 public:
  LeveldbKnowledgeBank(const KnowledgeBankConfig& config, int dimension)
      : KnowledgeBank(config, dimension),
        leveldb_config_(
            GetExtensionProtoOrDie<KnowledgeBankConfig,
                                   LeveldbKnowledgeBankConfig>(config)),
        embedding_data_(leveldb_config_.num_in_memory_partitions(),
                        leveldb_config_.max_in_memory_write_buffer_size(),
                        [](const std::deque<EmbeddingVectorProto>& data)
                            -> EmbeddingVectorProto { return data.back(); }) {
    auto absl_status = LoadDataFromLevelDb(leveldb_config_.leveldb_address(),
                                           /*create_if_missing=*/true);
    CHECK(absl_status.ok()) << absl_status.message();
  }

 private:
  // Implementation of the Lookup interface.
  absl::Status Lookup(const absl::string_view key,
                      EmbeddingVectorProto* result) const
      ABSL_LOCKS_EXCLUDED(load_db_mu_) override;

  // Implementation of the LookupWithUpdate interface.
  absl::Status LookupWithUpdate(const absl::string_view key,
                                EmbeddingVectorProto* result)
      ABSL_LOCKS_EXCLUDED(load_db_mu_, keys_mu_) override;

  // Updates the embedding of a single key.
  absl::Status Update(const absl::string_view key,
                      const EmbeddingVectorProto& value)
      ABSL_LOCKS_EXCLUDED(load_db_mu_, keys_mu_) override;

  // Implementation of the ExportInternal interface.
  // Exports the embeddings of updated keys and save these keys into a txt file.
  absl::Status ExportInternal(const std::string& dir,
                              std::string* exported_path)
      ABSL_LOCKS_EXCLUDED(load_db_mu_, keys_mu_) override;

  // Implementation of the ImportInternal interface.
  // It treats `saved_path` as a LevelDB address and attempts to reload the
  // embedding data from it.
  // Note that a LevelDB can only be opened by one process.
  absl::Status ImportInternal(const std::string& saved_path)
      ABSL_LOCKS_EXCLUDED(load_db_mu_, keys_mu_) override;

  // Returns the size of the current embedding data.
  size_t Size() const ABSL_LOCKS_EXCLUDED(load_db_mu_) override {
    absl::ReaderMutexLock rl(&load_db_mu_);
    return embedding_data_.size();
  }

  // Implementation of the Keys interface.
  std::vector<absl::string_view> Keys() const
      ABSL_LOCKS_EXCLUDED(load_db_mu_) override {
    absl::ReaderMutexLock l(&keys_mu_);
    return keys_;
  }

  // Implementation of the Contains interface.
  bool Contains(absl::string_view key) const
      ABSL_LOCKS_EXCLUDED(load_db_mu_) override {
    absl::ReaderMutexLock rl(&load_db_mu_);
    return embedding_data_.find(std::string(key)) != embedding_data_.end();
  }

  void ClearInternalData() ABSL_EXCLUSIVE_LOCKS_REQUIRED(load_db_mu_)
      ABSL_LOCKS_EXCLUDED(keys_mu_) {
    absl::MutexLock l(&keys_mu_);
    // Makes sure not other thread is accessing embedding_data_ when
    // ClearInternalData() is called.
    embedding_data_.clear();
    keys_.clear();
    updated_keys_.clear();
    keys_set_.clear();
  }

  // Loads all the data into memory.
  absl::Status LoadDataFromLevelDb(const std::string& db_path,
                                   bool create_if_missing)
      ABSL_LOCKS_EXCLUDED(load_db_mu_, keys_mu_);

  // LevelDB related.
  std::unique_ptr<leveldb::DB> leveldb_;
  const LeveldbKnowledgeBankConfig leveldb_config_;

  // Mutex for updating all the internal data, e.g., in LoadDataFromLevelDb().
  mutable absl::Mutex load_db_mu_;
  // Stores the in-memory (key, embedding) data for efficient lookup/update.
  // async_node_hash_map is sharded based on the keys to improve parallelism.
  // Note that only a read share of the lock is required even when doing writes
  // to this field, since async_node_hash_map is thread-safe.
  mutable async_node_hash_map<std::string, EmbeddingVectorProto> embedding_data_
      ABSL_GUARDED_BY(load_db_mu_);

  // The list of keys of the embedding, used for the Keys() method.
  mutable absl::Mutex keys_mu_;
  std::vector<absl::string_view> keys_ ABSL_GUARDED_BY(keys_mu_);
  // Keeps track of all the available keys.
  absl::flat_hash_set<absl::string_view> keys_set_ ABSL_GUARDED_BY(keys_mu_);
  // The set of keys that are updated but not exported.
  absl::flat_hash_set<std::string> updated_keys_ ABSL_GUARDED_BY(keys_mu_);
};

REGISTER_KNOWLEDGE_BANK_FACTORY(
    LeveldbKnowledgeBankConfig,
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
      // Checks LeveldbKnowledgeBankConfig.
      LeveldbKnowledgeBankConfig leveldb_config;
      config.extension().UnpackTo(&leveldb_config);
      if (leveldb_config.leveldb_address().empty()) {
        LOG(ERROR) << "leveldb_address is empty.";
        return nullptr;
      }
      if (leveldb_config.num_in_memory_partitions() <= 0) {
        LOG(ERROR) << "Invalid num_in_memory_partitions: "
                   << leveldb_config.num_in_memory_partitions();
        return nullptr;
      }
      if (leveldb_config.max_in_memory_write_buffer_size() <= 0) {
        LOG(ERROR) << "Invalid max_in_memory_write_buffer_size: "
                   << leveldb_config.max_in_memory_write_buffer_size();
        return nullptr;
      }
      return std::unique_ptr<KnowledgeBank>(
          new LeveldbKnowledgeBank(config, dimension));
    });

absl::Status LeveldbKnowledgeBank::Lookup(const absl::string_view key,
                                          EmbeddingVectorProto* result) const {
  absl::ReaderMutexLock rl(&load_db_mu_);
  const std::string str_key(key);
  const auto iter = embedding_data_.find(str_key);
  if (iter == embedding_data_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Key is not found: ", str_key));
  }
  *result = iter->second;
  return absl::OkStatus();
}

absl::Status LeveldbKnowledgeBank::LookupWithUpdate(
    const absl::string_view key, EmbeddingVectorProto* result) {
  // Use a reader locker since embedding_data_ is thread-safe.
  absl::ReaderMutexLock rl(&load_db_mu_);
  std::string str_key(key);
  if (embedding_data_.find(str_key) == embedding_data_.end()) {
    // Insert a new embedding.
    EmbeddingVectorProto embed =
        InitializeEmbedding(embedding_dimension(), config().initializer());
    embed.set_tag(str_key);
    embedding_data_.insert_or_assign(str_key, std::move(embed));
    absl::string_view strview_key = embedding_data_.find(str_key)->first;
    absl::WriterMutexLock l(&keys_mu_);
    if (keys_set_.find(strview_key) == keys_set_.end()) {
      keys_.push_back(strview_key);
      keys_set_.insert(strview_key);
    }
    updated_keys_.insert(embedding_data_.find(str_key)->first);
  }
  auto& embed = embedding_data_.find(str_key)->second;
  embed.set_weight(embed.weight() + 1);
  *result = embedding_data_.find(str_key)->second;
  return absl::OkStatus();
}

absl::Status LeveldbKnowledgeBank::Update(const absl::string_view key,
                                          const EmbeddingVectorProto& value) {
  // Use a reader locker since embedding_data_ is thread-safe.
  absl::ReaderMutexLock rl(&load_db_mu_);
  const std::string str_key(key);
  embedding_data_.insert_or_assign(str_key, value);
  {
    absl::string_view strview_key = embedding_data_.find(str_key)->first;
    absl::WriterMutexLock l(&keys_mu_);
    updated_keys_.insert(str_key);
    if (keys_set_.find(strview_key) == keys_set_.end()) {
      keys_.push_back(strview_key);
      keys_set_.insert(strview_key);
    }
  }
  return absl::OkStatus();
}

absl::Status LeveldbKnowledgeBank::ExportInternal(const std::string& dir,
                                                  std::string* exported_path) {
  absl::ReaderMutexLock rl(&load_db_mu_);
  *exported_path = leveldb_config_.leveldb_address();
  absl::MutexLock l(&keys_mu_);
  leveldb::WriteOptions options;
  for (const auto& key : updated_keys_) {
    const std::string str_key(key);
    leveldb_->Put(options, str_key,
                  embedding_data_.find(str_key)->second.SerializeAsString());
  }
  RET_CHECK_OK(WriteFileString(JoinPath(dir, kMetaDataOutputBaseName),
                               absl::StrJoin(keys_, "\n"),
                               /*can_overwrite=*/true));
  updated_keys_.clear();
  return absl::OkStatus();
}

absl::Status LeveldbKnowledgeBank::ImportInternal(
    const std::string& saved_path) {
  return LoadDataFromLevelDb(saved_path, /*create_if_missing=*/false);
}

absl::Status LeveldbKnowledgeBank::LoadDataFromLevelDb(
    const std::string& db_path, const bool create_if_missing) {
  if (!create_if_missing && !IsDirectory(db_path).ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent LevelDB path: ", db_path));
  }

  auto start = absl::Now();
  leveldb::DB* db = nullptr;
  leveldb::Options options;
  options.create_if_missing = create_if_missing;

  leveldb::Status status = leveldb::DB::Open(options, db_path, &db);
  if (!status.ok()) {
    return absl::InternalError(status.ToString());
  }
  RET_CHECK_TRUE(db != nullptr);

  // Use a write lock to prevent other methods from accessing the internal data
  // during loading.
  absl::WriterMutexLock wl(&load_db_mu_);
  leveldb_.reset(db);

  ClearInternalData();

  // Iterate over each item in the database and print them
  std::unique_ptr<leveldb::Iterator> it(
      leveldb_->NewIterator(leveldb::ReadOptions()));

  absl::MutexLock l(&keys_mu_);
  unsigned int count = 0;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    EmbeddingVectorProto proto;
    if (!proto.ParseFromString(it->value().ToString())) {
      return absl::InternalError(
          "Parsing input data to EmbeddingVectorProto failed.");
    }
    const std::string str_key = it->key().ToString();
    embedding_data_.insert_or_assign(str_key, std::move(proto));
    absl::string_view strview_key = embedding_data_.find(str_key)->first;
    keys_.push_back(strview_key);
    keys_set_.insert(strview_key);
    ++count;
  }
  LOG(ERROR) << "Loading " << count << " keys took " << absl::Now() - start;

  if (!it->status().ok()) {
    return absl::InternalError(absl::StrCat(
        "Scanning LevelDb failed with error: ", it->status().ToString()));
  }
  return absl::OkStatus();
}

}  // namespace carls
