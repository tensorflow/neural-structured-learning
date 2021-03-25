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

#include "research/carls/kbs_server_helper.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

// Placeholder for internal server credential  // net
// Placeholder for netutil
#include "grpcpp/server_builder.h"  // third_party

namespace carls {
namespace {

// https://en.wikipedia.org/wiki/Ephemeral_port
#define MAX_EPHEMERAL_PORT 60999
#define MIN_EPHEMERAL_PORT 32768

bool IsPortAvailable(int* port, bool is_tcp) {
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  const int fd = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, MAX_EPHEMERAL_PORT);
  if (fd < 0) {
    LOG(ERROR) << "socket() failed: " << strerror(errno);
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exists.
  int one = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt() failed: " << strerror(errno);
    if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(*port));
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    LOG(WARNING) << "bind(port=" << *port << ") failed: " << strerror(errno);
    if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
    return false;
  }

  // Get the bound port number.
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) <
      0) {
    LOG(WARNING) << "getsockname() failed: " << strerror(errno);
    if (close(fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
    return false;
  }
  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }
  if (close(fd) < 0) {
    LOG(ERROR) << "close() failed: " << strerror(errno);
  }
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

// TODO(xzeng): refactor to base/net.h
int PickUnusedPortOrDie() {
  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  unsigned int seed = time(NULL);
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = getpid() % (MAX_EPHEMERAL_PORT - MIN_EPHEMERAL_PORT) +
             MIN_EPHEMERAL_PORT;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand_r(&seed) % (MAX_EPHEMERAL_PORT - MIN_EPHEMERAL_PORT) +
             MIN_EPHEMERAL_PORT;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }
    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }
    chosen_ports.insert(port);
    return port;
  }
  return 0;
}

}  // namespace


KbsServerHelper::KbsServerHelper(const KnowledgeBankServiceOptions& kbs_options)
    : service_impl_(new KnowledgeBankGrpcServiceImpl()) {
  port_ = kbs_options.port;
  if (port_ < 0) {  // Auto select a port.
    port_ = /*net_util::*/PickUnusedPortOrDie();
  }
  address_ = absl::StrCat("[::]:", port_);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.RegisterService(service_impl_.get());
  server_ = builder.BuildAndStart();
  CHECK(server_ != nullptr);
  LOG(INFO) << "Server started at: " << address_;
}

void KbsServerHelper::WaitForTermination() { server_->Wait(); }

void KbsServerHelper::Terminate() { server_->Shutdown(); }

}  // namespace carls
