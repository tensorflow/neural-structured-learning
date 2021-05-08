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

#include "absl/base/call_once.h"
#include "absl/flags/flag.h"
#include "tensorflow/core/platform/env.h"

ABSL_FLAG(int, num_threads_for_global_thread_bundle, 1000,
          "Number of threads for the default global thread pool.");

namespace carls {
namespace {

using ::tensorflow::thread::ThreadPool;

constexpr char kDefaultThreadPooleName[] = "GlobalThreadBundle";

ThreadPool* g_thread_pool = nullptr;

}  // namespace

ThreadBundle::ThreadBundle() { Init(); }

ThreadBundle::ThreadBundle(const std::string& name, int num_threads)
    : thread_pool_(absl::make_unique<ThreadPool>(tensorflow::Env::Default(),
                                                 name, num_threads)) {}

ThreadBundle::~ThreadBundle() {
  JoinAll();
}

void ThreadBundle::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    CHECK(g_thread_pool == nullptr);
    g_thread_pool = new ThreadPool(
        tensorflow::Env::Default(), kDefaultThreadPooleName,
        absl::GetFlag(FLAGS_num_threads_for_global_thread_bundle));
  });
}

void ThreadBundle::Add(std::function<void()> f) {
  Inc();
  auto job = [f, this]() {
    f();
    Dec();
  };
  if (thread_pool_ != nullptr) {
    thread_pool_->Schedule(job);
  } else {
    g_thread_pool->Schedule(job);
  }
}

void ThreadBundle::JoinAll() { JoinAllWithDeadline(absl::InfiniteFuture()); }

bool ThreadBundle::JoinAllWithDeadline(absl::Time deadline) {
  absl::MutexLock l(&mu_);
  while (count_ > 0) {
    if (cv_.WaitWithDeadline(&mu_, deadline)) return false;
  }
  return true;
}

void ThreadBundle::Inc() {
  absl::MutexLock l(&mu_);
  ++count_;
}

void ThreadBundle::Dec() {
  absl::MutexLock l(&mu_);
  if (--count_ == 0) cv_.SignalAll();
}

}  // namespace carls
