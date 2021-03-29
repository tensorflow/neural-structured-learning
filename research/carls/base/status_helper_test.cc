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

#include "research/carls/base/status_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace carls {

TEST(ProtoFactoryTest, ToAbslStatus_Grpc) {
  EXPECT_EQ(absl::OkStatus(), ToAbslStatus(grpc::Status::OK));

  auto grpc_status = grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Error.");
  auto absl_status = ToAbslStatus(grpc_status);
  EXPECT_EQ("Error.", absl_status.message());
  EXPECT_TRUE(absl::IsInvalidArgument(absl_status));
}

TEST(ProtoFactoryTest, ToAbslStatus_TensorFlow) {
  EXPECT_EQ(absl::OkStatus(), ToAbslStatus(tensorflow::Status::OK()));

  auto tf_status =
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "Error.");
  auto absl_status = ToAbslStatus(tf_status);
  EXPECT_EQ("Error.", absl_status.message());
  EXPECT_TRUE(absl::IsInvalidArgument(absl_status));
}

TEST(ProtoFactoryTest, ToGrpcStatus) {
  EXPECT_TRUE(ToGrpcStatus(absl::OkStatus()).ok());

  auto absl_status = absl::InvalidArgumentError("Error.");
  auto grpc_status = ToGrpcStatus(absl_status);
  EXPECT_EQ("Error.", grpc_status.error_message());
  EXPECT_EQ(grpc::StatusCode::INVALID_ARGUMENT, grpc_status.error_code());
}

}  // namespace carls