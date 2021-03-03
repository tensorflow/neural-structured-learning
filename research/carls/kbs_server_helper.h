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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KBS_SERVER_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KBS_SERVER_HELPER_H_

#include <string>

#include "grpcpp/server.h"  // third_party
#include "research/carls/knowledge_bank_grpc_service.h"

namespace carls {

// Options for starting a KnowledgeBankService (KBS) server. For example:
//   auto options = KnowledgeBankServiceOptions(true, -1, 10);
//   KbsServerHelper server(options);
//   server.WaitForTermination();
// Directly calling KnowledgeBankServiceOptions() returns the default values.
struct KnowledgeBankServiceOptions {
  // Should run locally or on remote clusters.
  bool run_locally;

  // The port to run, auto-select if port < 0.
  int port;

  // Number of threads to run, default to 100.
  int num_threads;

  KnowledgeBankServiceOptions()
      : run_locally(true), port(-1), num_threads(100) {}

  KnowledgeBankServiceOptions(bool local, int port, int num_threads)
      : run_locally(local), port(port), num_threads(num_threads) {}
};

// Manages an active KBS server such that a server can be asynchronously
// stopped. This is a common interface shared by both C++ and Python.
class KbsServerHelper {
 public:
  // Starts the server based on given options.
  explicit KbsServerHelper(const KnowledgeBankServiceOptions& kbs_options);

  // Blocks the calling thread and keeps the server running.
  void WaitForTermination();

  // Terminates the server.
  void Terminate();

  // Returns the address of the server.
  std::string address() { return address_; }

  // Returns the port of the server.
  int port() { return port_; }

 private:
  int port_;
  std::string address_;
  std::unique_ptr<KnowledgeBankGrpcServiceImpl> service_impl_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_KBS_SERVER_HELPER_H_
