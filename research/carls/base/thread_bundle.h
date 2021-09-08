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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_THREAD_BUNDLE_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_THREAD_BUNDLE_H_

#include <atomic>
#include <functional>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/threadpool.h"

namespace carls {

// ThreadBundle provides a simple interface for multi-threaded computation.
//
// Example Usage:
//
//   ThreadBundle b;
//   while (!done) {
//     ...
//     b.Add(...);
//   }
//   b.JoinAll();
//
class ThreadBundle {
 public:
  // Constructs a ThreadBundle whose threads will run out of a default global
  // ThreadPool.
  ThreadBundle();

  // Constructs a ThreadBundle whose threads will run out of new ThreadPool
  // created just for this ThreadBundle.
  // `name` and `num_threads` are used for the contructor of ThreadPool.
  // REQUIRED: num_threads > 0.
  ThreadBundle(const std::string& name, int num_threads);

  ThreadBundle(const ThreadBundle&) = delete;
  ThreadBundle& operator=(const ThreadBundle&) = delete;

  // Waits until all threads are finished.
  ~ThreadBundle();

  // Adds a new worker/function to the ThreadBundle and runs it immediately.
  void Add(std::function<void()> f);

  // Join all pending workers. Returns immediately if the bundle is empty.
  // It is illegal to Add() new workers after calling JoinAll().
  void JoinAll();

  // Like JoinAll(), but with a deadline.
  // Returns true if all workers are finished, false if not all workers
  // finished by time `deadline`.
  bool JoinAllWithDeadline(absl::Time deadline);

 private:
  // Called by contructors for initializing global variables.
  void Init();

  // Increase the reference count by 1.
  void Inc();

  // Decrease the reference count by 1.
  void Dec();

  // Local thread pool initialized from constructor.
  std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool_;
  // Mutex for waiting/signaling.
  mutable absl::Mutex mu_;
  mutable absl::CondVar cv_;
  // The reference count for number of active workers.
  uint64_t count_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_THREAD_BUNDLE_H_
