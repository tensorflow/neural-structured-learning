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

#include "research/carls/base/async_node_hash_map.h"

#include <thread>  // NOLINT

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace carls {

TEST(AsyncNodeHashTest, SinglePartition_NoAggregator) {
  async_node_hash_map<std::string, std::string> map(
      /*num_partitions=*/1, /*max_write_buffer_size=*/1, nullptr);
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());

  // Tests insert_or_assign().
  map.insert_or_assign("first", "first_value");
  EXPECT_TRUE(map.contains("first"));
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(1, map.size());

  // For subsequent updates, only the last element is recorded.
  map.insert_or_assign("first", "v2");
  map.insert_or_assign("first", "v3");
  map.insert_or_assign("first", "v4");
  EXPECT_EQ("v4", map["first"]);
  EXPECT_EQ("v4", map["first"]);  // Lookup again.
  EXPECT_EQ("v4", map.find("first")->second);

  // Tests operator []
  map["second"] = "second_value";
  EXPECT_TRUE(map.contains("second"));
  EXPECT_EQ(2, map.size());

  // Tests find()
  EXPECT_NE(map.find("second"), map.end());
  EXPECT_EQ(map.find("missing"), map.end());

  // Tests begin() and end().
  auto iter = map.begin();
  std::set<std::string> keys;
  while (iter != map.end()) {
    keys.insert(iter->first);
    iter++;
  }
  EXPECT_EQ(2, keys.size());
  EXPECT_TRUE(keys.find("first") != keys.end());
  EXPECT_TRUE(keys.find("second") != keys.end());

  // Tests for-iterators.
  map.insert_or_assign("third", "third_value");
  keys.clear();
  for (const auto& pair : map) {
    keys.insert(pair.first);
  }
  EXPECT_EQ(3, keys.size());
  EXPECT_TRUE(keys.find("first") != keys.end());
  EXPECT_TRUE(keys.find("second") != keys.end());
  EXPECT_TRUE(keys.find("third") != keys.end());
  EXPECT_EQ(3, map.size());

  // Tests clear().
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());
}

TEST(AsyncNodeHashTest, SinglePartition_WithAggregator) {
  // Returns the joined values..
  auto aggregator = [](const std::deque<std::string>& values) -> std::string {
    return absl::StrJoin(values, ",");
  };
  async_node_hash_map<std::string, std::string> map(/*num_partitions=*/1,
                                                    /*max_write_buffer_size=*/5,
                                                    aggregator);
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());

  // For the first insert request, the key is always updated.
  map.insert_or_assign("first", "v1");
  EXPECT_TRUE(map.contains("first"));
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(1, map.size());
  EXPECT_EQ("v1", map["first"]);

  // For subsequent updates, only the first element is recorded.
  map.insert_or_assign("first", "v2");
  map.insert_or_assign("first", "v3");
  map.insert_or_assign("first", "v4");
  EXPECT_EQ("v2,v3,v4", map["first"]);
  EXPECT_EQ("v2,v3,v4", map["first"]);  // Lookup again.
  EXPECT_EQ("v2,v3,v4", map.find("first")->second);

  // Buffer overflow, old value is automatically ejected.
  // "v5" is flushed out, so the aggregated result should be "v6,v7,v8,v9,v10".
  map.insert_or_assign("first", "v5");
  map.insert_or_assign("first", "v6");
  map.insert_or_assign("first", "v7");
  map.insert_or_assign("first", "v8");
  map.insert_or_assign("first", "v9");
  map.insert_or_assign("first", "v10");
  EXPECT_EQ("v6,v7,v8,v9,v10", map["first"]);
}

TEST(AsyncNodeHashTest, MultiplePartition_NoAggregator) {
  async_node_hash_map<std::string, std::string> map(/*num_partitions=*/5,
                                                    /*max_write_buffer_size=*/5,
                                                    /*aggregator=*/nullptr);
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());

  // Tests insert_or_assign().
  map.insert_or_assign("first", "first_value");
  EXPECT_TRUE(map.contains("first"));
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(1, map.size());

  // For subsequent updates, only the last element is recorded.
  map.insert_or_assign("first", "v2");
  map.insert_or_assign("first", "v3");
  map.insert_or_assign("first", "v4");
  EXPECT_EQ("v4", map["first"]);
  EXPECT_EQ("v4", map["first"]);  // Lookup again.
  EXPECT_EQ("v4", map.find("first")->second);

  // Tests operator []
  map["second"] = "second_value";
  EXPECT_TRUE(map.contains("second"));
  EXPECT_EQ(2, map.size());

  // Tests find()
  EXPECT_NE(map.find("second"), map.end());
  EXPECT_EQ(map.find("missing"), map.end());

  // Tests begin() and end().
  auto iter = map.begin();
  std::set<std::string> keys;
  while (iter != map.end()) {
    keys.insert(iter->first);
    iter++;
  }
  EXPECT_EQ(2, keys.size());
  EXPECT_TRUE(keys.find("first") != keys.end());
  EXPECT_TRUE(keys.find("second") != keys.end());

  // Tests for-iterators
  map.insert_or_assign("third", "third_value");
  keys.clear();
  for (const auto& pair : map) {
    keys.insert(pair.first);
  }
  EXPECT_EQ(3, keys.size());
  EXPECT_TRUE(keys.find("first") != keys.end());
  EXPECT_TRUE(keys.find("second") != keys.end());
  EXPECT_TRUE(keys.find("third") != keys.end());
  EXPECT_EQ(3, map.size());

  // Tests clear().
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());
}

TEST(AsyncNodeHashTest, MultiplePartition_WithAggregator) {
  // Returns the joined values..
  auto aggregator = [](const std::deque<std::string>& values) -> std::string {
    return absl::StrJoin(values, ",");
  };
  async_node_hash_map<std::string, std::string> map(/*num_partitions=*/5,
                                                    /*max_write_buffer_size=*/5,
                                                    aggregator);
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, map.size());

  // For the first insert request, the key is always updated.
  map.insert_or_assign("key", "v1");
  EXPECT_TRUE(map.contains("key"));
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(1, map.size());
  EXPECT_EQ("v1", map["key"]);

  // For subsequent updates, only the first element is recorded.
  map.insert_or_assign("key", "v2");
  map.insert_or_assign("key", "v3");
  map.insert_or_assign("key", "v4");
  EXPECT_EQ("v2,v3,v4", map["key"]);
  EXPECT_EQ("v2,v3,v4", map["key"]);  // Lookup again.
  EXPECT_EQ("v2,v3,v4", map.find("key")->second);

  // Buffer overflow, old value is automatically ejected.
  // "v5" is flushed out, so the aggregated result should be "v6,v7,v8,v9,v10".
  map.insert_or_assign("key", "v5");
  map.insert_or_assign("key", "v6");
  map.insert_or_assign("key", "v7");
  map.insert_or_assign("key", "v8");
  map.insert_or_assign("key", "v9");
  map.insert_or_assign("key", "v10");
  EXPECT_EQ("v6,v7,v8,v9,v10", map["key"]);

  // A call to map.find() would enforce aggregator to be called.
  map.insert_or_assign("key", "v11");
  map.insert_or_assign("key", "v12");
  map.insert_or_assign("key", "v13");
  auto pair = map.find("key");
  EXPECT_EQ("v11,v12,v13", pair->second);
  EXPECT_EQ("v11,v12,v13", map["key"]);  // Lookup again.
}

TEST(AsyncNodeHashTest, MultiplePartition_WithAggregator_MultiThreading) {
  // Returns the joined values..
  auto aggregator = [](const std::deque<std::string>& values) -> std::string {
    return absl::StrJoin(values, ",");
  };
  const int buffer_size = 100;
  async_node_hash_map<std::string, std::string> map(/*num_partitions=*/10,
                                                    buffer_size, aggregator);

  // Run 100 parallel threads to update the value for "key".
  std::vector<std::thread> threads;
  for (int i = 0; i < buffer_size; ++i) {
    threads.emplace_back(
        [&map, i]() { map.insert_or_assign("key", absl::StrCat("v", i)); });
  }

  // Waits for all threads to finish.
  for (auto& thread : threads) {
    thread.join();
  }

  // Checks that all the final keys is a concatenation of "v0", "v1", ... "v99"
  // in random order.
  std::set<std::string> keys = absl::StrSplit(map["key"], ',');
  EXPECT_EQ(99, keys.size());
  // The first value for "key" is applied immediate, so we check only one
  // element is missing.
  int num_missed_value = 0;
  for (int i = 0; i < buffer_size; ++i) {
    if (keys.find(absl::StrCat("v", i)) == keys.end()) ++num_missed_value;
  }
  EXPECT_EQ(1, num_missed_value);
}

TEST(AsyncNodeHashTest, PointerPersistency) {
  async_node_hash_map<std::string, std::string> map(
      /*num_partitions=*/10, /*max_write_buffer_size=*/1,
      /*aggregator=*/nullptr);

  std::vector<absl::string_view> keys;
  std::vector<absl::string_view> values;

  const int num_keys = 100;
  for (int i = 0; i < num_keys; ++i) {
    const std::string key = absl::StrCat("key", i);
    map[key] = absl::StrCat("v", i);
    auto pair = map.find(key);
    keys.push_back(pair->first);
    values.push_back(pair->second);
  }

  // Checks that the string view of keys and values are persistent.
  for (int i = 0; i < num_keys; ++i) {
    EXPECT_EQ(absl::StrCat("key", i), keys[i]);
    EXPECT_EQ(absl::StrCat("v", i), values[i]);
  }
}

}  // namespace carls
