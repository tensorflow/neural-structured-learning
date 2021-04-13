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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_ASYNC_NODE_HASH_MAP_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_ASYNC_NODE_HASH_MAP_H_

#include <deque>
#include <functional>
#include <iostream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/internal/raw_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace carls {

// async_node_hash_map is a special hash map designed for efficient embedding
// lookup/update, with the following features:
// - Thread-safe: asynchronous accesses to read/write items are safe.
// - Pointer stable: pointer/reference to key/value is guaranteed to be valid
//   through its life time.
// - Customizable: users can design their own aggregation strategy on
//   conflicting writing requests.
// - Consistency: if the aggregation strategy is deterministic, the same
//   insert/find sequence always results in the same result.
//
// As an example, if there are multiple requests for embedding update:
// {insert_or_assign("key", [1, 2]), insert_or_assign("key", [3, 4]),
// insert_or_assign("key", [5, 6])}, the embedding of 'key' would be (lazy)
// updated only when one of the following happens
// - A map["key"] or map.find("key") request is received.
// - The number of pending update requests reached the given
//   `max_write_buffer_size`.
// The updated request would be based on given aggregating function, i.e.,
// map["key"] = aggregator({[1, 2], [3, 4], [5, 6]}).
// Note that we do not take into account the original value/embedding of "key".
//
template <class Key, class Value,
          class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq = absl::container_internal::hash_default_eq<Key>,
          class Alloc = std::allocator<std::pair<const Key, Value>>>
class async_node_hash_map {
  using NodeHashMap = absl::node_hash_map<Key, Value, Hash, Eq, Alloc>;
  // P is Policy. It's passed as a template argument to support maps that have
  // incomplete types as values, as in unordered_map<K, IncompleteType>.
  // MappedReference<> may be a non-reference type.
  template <class P>
  using MappedReference = decltype(P::value(
      std::addressof(std::declval<typename NodeHashMap::reference>())));

  using Policy = absl::container_internal::NodeHashMapPolicy<Key, Value>;

 public:
  using key_type = typename NodeHashMap::key_type;
  using mapped_type = typename NodeHashMap::mapped_type;

  template <class K>
  using key_arg = typename NodeHashMap::template key_arg<K>;

  // Constructor for async_node_hash_map.
  //
  // `num_partitions` decides the parallelism for the async_node_hash_map.
  // It is usually set to be no less than the number of CPUs available.
  //
  // `max_write_buffer_size` is the maximal buffer size for the value of a key
  // to be updated.
  //
  // `aggregator` is a function that converts a list of `Value` into a single
  // `Value`, the input is given in the order of arrival of the requests.
  // If `aggregator` = nullptr, disable lazy update, and `max_write_buffer_size`
  // would not have any effect.
  //
  // REQUIRED: num_partitions > 0 and max_write_buffer_size.
  async_node_hash_map(
      int num_partitions, int max_write_buffer_size,
      std::function<Value(const std::deque<Value>&)> aggregator)
      : num_partitions_(num_partitions),
        max_write_buffer_size_(max_write_buffer_size),
        aggregator_(aggregator) {
    assert(num_partitions_ > 0);
    assert(max_write_buffer_size > 0);

    partitioned_hash_maps_.reserve(num_partitions);
    partitioned_mu_.reserve(num_partitions);
    partitioned_update_buffer_.reserve(num_partitions);
    for (int p = 0; p < num_partitions; ++p) {
      partitioned_hash_maps_.emplace_back(new NodeHashMap());
      partitioned_mu_.emplace_back(new absl::Mutex());
      partitioned_update_buffer_.emplace_back();
    }
  }

  // Iterator class that wraps around a list of node_hash_map::iterator's.
  class iterator {
    friend class async_node_hash_map;

   public:
    using reference = typename NodeHashMap::iterator::reference;
    using pointer = typename NodeHashMap::iterator::pointer;

    iterator() {}

    // PRECONDITION: not an end() iterator.
    reference operator*() const { return *iterators_[current_partition_]; }

    // PRECONDITION: not an end() iterator.
    pointer operator->() const { return &operator*(); }

    // The operator ++iter.
    iterator& operator++() {
      ++iterators_[current_partition_];
      // If the iterator has reaches to the end of its partition, find the
      // next partition with non-empty elements.
      while (current_partition_ < iterators_.size() - 1 &&
             iterators_[current_partition_] ==
                 end_iterators_[current_partition_]) {
        ++current_partition_;
      }
      return *this;
    }
    // PRECONDITION: not an end() iterator.
    // The operator iter++;
    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.iterators_[a.current_partition_] ==
             b.iterators_[b.current_partition_];
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return a.iterators_[a.current_partition_] !=
             b.iterators_[b.current_partition_];
    }

   private:
    iterator(std::vector<typename NodeHashMap::iterator> iterators,
             std::vector<typename NodeHashMap::iterator> end_iterators,
             int current_partition)
        : iterators_(std::move(iterators)),
          end_iterators_(std::move(end_iterators)),
          current_partition_(current_partition) {
      while (current_partition_ < iterators_.size() - 1 &&
             iterators_[current_partition_] ==
                 end_iterators_[current_partition_]) {
        ++current_partition_;
      }
      assert(current_partition_ >= 0);
      assert(current_partition_ < iterators_.size());
    }

    // Keeps track of current state of all NodeHashMap::iterators, such that
    // the current element can be retrieved by iterators_[current_partition_].
    std::vector<typename NodeHashMap::iterator> iterators_;
    // The end iterators for all partitions. This is needed for early version
    // of absl::node_hash_map whose end() points to the address after the last
    // element.
    std::vector<typename NodeHashMap::iterator> end_iterators_;
    // The current partition of the iterators_.
    unsigned int current_partition_ = 0;
  };

  iterator begin() {
    return iterator(get_begin_iterators(), get_end_iterators(), 0);
  }
  iterator end() {
    return iterator(get_end_iterators(), get_end_iterators(), num_partitions_);
  }

  // Returns true if the hash map is empty.
  bool empty() const {
    for (unsigned int p = 0; p < num_partitions_; ++p) {
      absl::MutexLock l(partitioned_mu_[p].get());
      if (!partitioned_hash_maps_[p]->empty()) {
        return false;
      }
    }
    return true;
  }

  // Return the total size of the partitioned hash maps.
  size_t size() const {
    size_t size = 0;
    for (unsigned int p = 0; p < num_partitions_; ++p) {
      absl::MutexLock l(partitioned_mu_[p].get());
      size += partitioned_hash_maps_[p]->size();
    }
    return size;
  }

  // Clears all partitioned hash maps.
  void clear() {
    for (unsigned int p = 0; p < num_partitions_; ++p) {
      absl::MutexLock l(partitioned_mu_[p].get());
      partitioned_hash_maps_[p]->clear();
      partitioned_update_buffer_[p].clear();
    }
  }

  // The API of insert_or_assign().
  //
  // The last two template parameters ensure that both arguments are rvalues
  // (lvalue arguments are handled by the overloads below). This is necessary
  // for supporting bitfield arguments.
  template <class K = key_type, class V = mapped_type, K* = nullptr,
            V* = nullptr>
  std::pair<iterator, bool> insert_or_assign(key_arg<K>&& k, V&& v) {
    return insert_or_assign_impl(std::forward<K>(k), std::forward<V>(v));
  }

  template <class K = key_type, class V = mapped_type, K* = nullptr>
  std::pair<iterator, bool> insert_or_assign(key_arg<K>&& k, const V& v) {
    return insert_or_assign_impl(std::forward<K>(k), v);
  }

  template <class K = key_type, class V = mapped_type, V* = nullptr>
  std::pair<iterator, bool> insert_or_assign(const key_arg<K>& k, V&& v) {
    return insert_or_assign_impl(k, std::forward<V>(v));
  }

  template <class K = key_type, class V = mapped_type>
  std::pair<iterator, bool> insert_or_assign(const key_arg<K>& k, const V& v) {
    return insert_or_assign_impl(k, v);
  }

  // Returns the parition number of the given key.
  unsigned int get_partition(const Key& key) const {
    return tensorflow::Fingerprint32(key) % num_partitions_;
  }

  // The API of find().
  template <class K = key_type>
  iterator find(const key_arg<K>& key) {
    const int p = get_partition(key);
    auto iterators = get_begin_iterators();
    auto end_iterators = get_end_iterators();
    absl::MutexLock l(partitioned_mu_[p].get());
    // First checks value buffer. If not empty, update the hash map with
    // aggregated value.
    if (aggregator_ != nullptr) {
      if (partitioned_update_buffer_[p].find(key) !=
          partitioned_update_buffer_[p].end()) {
        auto aggregated_value = aggregator_(partitioned_update_buffer_[p][key]);
        auto pair = partitioned_hash_maps_[p]->insert_or_assign(
            key, std::move(aggregated_value));
        iterators[p] = std::move(pair.first);
        partitioned_update_buffer_[p].erase(key);
      }
    }

    auto iter = partitioned_hash_maps_[p]->find(key);
    // key is not found, return an iterator containing all end()'s.'
    if (iter == partitioned_hash_maps_[p]->end()) {
      return iterator(std::move(iterators), std::move(end_iterators),
                      num_partitions_);
    }
    iterators[p] = std::move(iter);
    return iterator(std::move(iterators), std::move(end_iterators), p);
  }

  // Checks if the given key is already in the partitioned hash maps.
  template <class K = key_type>
  bool contains(const key_arg<K>& key) const {
    const int p = get_partition(key);
    absl::MutexLock l(partitioned_mu_[p].get());
    return partitioned_hash_maps_[p]->contains(key);
  }

  // The API of operator [].
  template <class K = key_type, class P = Policy, K* = nullptr>
  MappedReference<P> operator[](key_arg<K>&& key) {
    const int p = get_partition(key);
    absl::MutexLock l(partitioned_mu_[p].get());
    if (aggregator_ != nullptr) {
      // First checks value buffer. If not empty, update the hash map with
      // aggregated value.
      if (partitioned_update_buffer_[p].find(key) !=
          partitioned_update_buffer_[p].end()) {
        auto aggregated_value = aggregator_(partitioned_update_buffer_[p][key]);
        partitioned_hash_maps_[p]->insert_or_assign(
            key, std::move(aggregated_value));
        partitioned_update_buffer_[p].erase(key);
      }
    }
    return Policy::value(
        &*partitioned_hash_maps_[p]->try_emplace(std::forward<K>(key)).first);
  }

  template <class K = key_type, class P = Policy>
  MappedReference<P> operator[](const key_arg<K>& key) {
    const int p = get_partition(key);
    absl::MutexLock l(partitioned_mu_[p].get());
    if (aggregator_ != nullptr) {
      // First checks value buffer. If not empty, update the hash map with
      // aggregated value.
      if (partitioned_update_buffer_[p].find(key) !=
          partitioned_update_buffer_[p].end()) {
        auto aggregated_value = aggregator_(partitioned_update_buffer_[p][key]);
        partitioned_hash_maps_[p]->insert_or_assign(
            key, std::move(aggregated_value));
        partitioned_update_buffer_[p].erase(key);
      }
    }
    return Policy::value(&*partitioned_hash_maps_[p]->try_emplace(key).first);
  }

 private:
  using ValueUpdateBuffer = absl::flat_hash_map<Key, std::deque<Value>>;

  // Implementation of the insert_or_assign API.
  template <class K, class V>
  std::pair<iterator, bool> insert_or_assign_impl(K&& key, V&& value) {
    const int p = get_partition(key);
    auto iterators = get_begin_iterators();
    auto end_iterators = get_end_iterators();
    absl::MutexLock l(partitioned_mu_[p].get());
    if (aggregator_ == nullptr || max_write_buffer_size_ == 1 ||
        partitioned_hash_maps_[p]->find(key) ==
            partitioned_hash_maps_[p]->end()) {
      // Add the element without buffering.
      auto pair = partitioned_hash_maps_[p]->insert_or_assign(key, value);
      iterators[p] = std::move(pair.first);
      return {iterator(std::move(iterators), std::move(end_iterators), p),
              pair.second};
    }
    // Inserts the value to the buffer and ejects oldest ones if it is full.
    partitioned_update_buffer_[p][key].push_back(value);
    while (partitioned_update_buffer_[p][key].size() > max_write_buffer_size_) {
      partitioned_update_buffer_[p][key].pop_front();
    }
    return {iterator(std::move(iterators), std::move(end_iterators), p), false};
  }

  std::vector<typename NodeHashMap::iterator> get_begin_iterators() {
    std::vector<typename NodeHashMap::iterator> iterators;
    iterators.reserve(num_partitions_ + 1);
    for (unsigned int p = 0; p < num_partitions_; ++p) {
      absl::MutexLock l(partitioned_mu_[p].get());
      iterators.push_back(partitioned_hash_maps_[p]->begin());
    }
    iterators.push_back(end_);
    return iterators;
  }

  std::vector<typename NodeHashMap::iterator> get_end_iterators() {
    std::vector<typename NodeHashMap::iterator> iterators;
    iterators.reserve(num_partitions_ + 1);
    for (unsigned int p = 0; p < num_partitions_; ++p) {
      absl::MutexLock l(partitioned_mu_[p].get());
      iterators.push_back(partitioned_hash_maps_[p]->end());
    }
    iterators.push_back(end_);
    return iterators;
  }

  // Attributes from contructor.
  const unsigned int num_partitions_;
  const unsigned int max_write_buffer_size_;
  std::function<Value(const std::deque<Value>&)> aggregator_;

  // Partitioned hash map for efficient parallel map access.
  mutable std::vector<std::unique_ptr<NodeHashMap>> partitioned_hash_maps_;
  // Mutexes that protects the access to the partitioned map.
  mutable std::vector<std::unique_ptr<absl::Mutex>> partitioned_mu_;
  // Partitioned key to updated values buffer.
  mutable std::vector<ValueUpdateBuffer> partitioned_update_buffer_;
  // A spacial iterator denoting the end of all partitions.
  const typename NodeHashMap::iterator end_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_ASYNC_NODE_HASH_MAP_H_
