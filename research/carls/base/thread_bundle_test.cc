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

#include "research/carls/base/thread_bundle.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace carls {

TEST(ThreadBundleTest, LocalThreadPool) {
  ThreadBundle b("Test", 1);
  absl::Mutex mu;
  int sum = 0;
  for (int i = 0; i <= 10; ++i) {
    b.Add([&sum, &mu, i]() {
      absl::MutexLock l(&mu);
      sum += i;
    });
  }
  b.JoinAll();
  // 0 + 1 + 2 + .. + 10 = 55.
  EXPECT_EQ(55, sum);
}

TEST(ThreadBundleTest, GlobalThreadPool) {
  ThreadBundle b;
  // A simple matrix computation using multi-threading.
  const std::vector<std::vector<float>> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const std::vector<float> vec{1, 2, 3};
  std::vector<float> result{0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    b.Add([&matrix, &vec, &result, i]() {
      int sum = 0;
      for (int j = 0; j < 3; ++j) {
        sum += matrix[i][j] * vec[j];
      }
      result[i] = sum;
    });
  }
  b.JoinAll();
  EXPECT_FLOAT_EQ(14, result[0]);  // [1, 2, 3] * [1, 2, 3]^T
  EXPECT_FLOAT_EQ(32, result[1]);  // [4, 5, 6] * [1, 2, 3]^T
  EXPECT_FLOAT_EQ(50, result[2]);  // [7, 8, 9] * [1, 2, 3]^T
}

TEST(ThreadBundleTest, WithoutJoin) {
  int sum = 0;
  absl::Mutex mu;
  {
    ThreadBundle b;
    for (int i = 0; i <= 10; ++i) {
      b.Add([&sum, &mu, i]() {
        absl::MutexLock l(&mu);
        sum += i;
        // Makes sure the thread is slow.
        absl::SleepFor(absl::Milliseconds(10));
      });
    }
    // ThreadBundle destructor invokes Join().
  }
  // 0 + 1 + 2 + .. + 10 = 55.
  EXPECT_EQ(55, sum);
}

TEST(ThreadBundleTest, JoinAllWithDeadline_NotAllThreadsFinish) {
  ThreadBundle b;
  std::atomic<int> sum{0};
  // A fast worker increments sum by 1.
  b.Add([&sum]() { sum += 1; });
  // A slow worker increments sum by 10.
  b.Add([&sum]() {
    absl::SleepFor(absl::Milliseconds(100));
    sum += 10;
  });
  // Wait for only 10 ms., so slow worker does not complete.
  EXPECT_FALSE(b.JoinAllWithDeadline(absl::Now() + absl::Milliseconds(10)));
  // Checks only fast worker is run.
  EXPECT_EQ(1, sum);
}

TEST(ThreadBundleTest, JoinAllWithDeadline_AllThreadsFinish) {
  ThreadBundle b;
  std::atomic<int> sum{0};
  // A worker increments sum by 1.
  b.Add([&sum]() { sum += 1; });
  // Another worker increments sum by 10.
  b.Add([&sum]() { sum += 10; });
  // Wait for 10 ms such that all threads can finish.
  EXPECT_TRUE(b.JoinAllWithDeadline(absl::Now() + absl::Milliseconds(10)));
  // Checks the sum is 11.
  EXPECT_EQ(11, sum);
}

}  // namespace carls
