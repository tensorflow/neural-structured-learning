/* Copyright 2021 Google LLC. All Rights Reserved.

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

#include "research/carls/testing/test_helper.h"

#include "grpcpp/support/status.h"  // net
#include "absl/status/status.h"
#include "research/carls/base/status_helper.h"
#include "research/carls/testing/test_proto2.pb.h"  // proto to pb
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"  // proto to pb

namespace carls {

TEST(TestHelperTest, EqualsProto) {
  TestBaseProto2Def proto;
  proto.set_name("Audrey");
  EXPECT_THAT(proto, EqualsProto(proto));
}

TEST(TestHelperTest, EqualsProtoText) {
  TestBaseProto2Def proto;
  proto.set_name("Audrey");
  EXPECT_THAT(proto, EqualsProto<TestBaseProto2Def>(R"pb(
                name: "Audrey"
              )pb"));
}

TEST(TestHelperTest, AbslStatusChecks) {
  EXPECT_OK(absl::OkStatus());
  EXPECT_NOT_OK(absl::InternalError("First error."));
  EXPECT_ERROR(absl::InternalError("First error."), "First error.");
  ASSERT_OK(absl::OkStatus());
  ASSERT_ERROR(absl::InternalError("Second error."), "Second error.");
}

TEST(TestHelperTest, GrpcStatusChecks) {
  EXPECT_OK(grpc::Status::OK);
  EXPECT_OK(ToGrpcStatus(absl::OkStatus()));
  EXPECT_NOT_OK(ToGrpcStatus(absl::InternalError("First error.")));
  EXPECT_ERROR(ToGrpcStatus(absl::InternalError("First error.")),
               "First error.");
  ASSERT_OK(grpc::Status::OK);
  ASSERT_ERROR(ToGrpcStatus(absl::InternalError("Second error.")),
               "Second error.");
}

TEST(TestHelperTest, TensoFlowStatusChecks) {
  EXPECT_OK(tensorflow::Status::OK());
  EXPECT_NOT_OK(
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "First error."));
  EXPECT_ERROR(
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "First error."),
      "First error.");
  ASSERT_OK(tensorflow::Status::OK());
  ASSERT_ERROR(
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "Second error."),
      "Second error.");
}

}  // namespace carls
