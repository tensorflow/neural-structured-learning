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

#include <thread>  // NOLINT

#include "grpcpp/create_channel.h"  // net
// Placeholder for internal channel credential  // net
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carls {

class KbsServerHelperTest : public ::testing::Test {
 protected:
  KbsServerHelperTest() {}
};

TEST_F(KbsServerHelperTest, Create) {
  const KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
}

TEST_F(KbsServerHelperTest, WaitForTermination) {
  const KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);

  auto fn = [&helper]() { helper.WaitForTermination(); };

  std::thread t(fn);
  helper.Terminate();
  t.join();
}

TEST_F(KbsServerHelperTest, StubConnection) {
  const KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  const std::string kbs_address = absl::StrCat("localhost:", helper.port());

  std::shared_ptr<grpc::ChannelCredentials> credentials =
      grpc::Loas2Credentials(grpc::Loas2CredentialsOptions());
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(kbs_address, credentials);
  ASSERT_TRUE(channel != nullptr);
  std::unique_ptr</*grpc_gen::*/KnowledgeBankService::Stub> stub =
      /*grpc_gen::*/KnowledgeBankService::NewStub(channel);
  ASSERT_TRUE(stub != nullptr);
}

}  // namespace carls
