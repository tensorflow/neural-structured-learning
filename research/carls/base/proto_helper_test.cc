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

#include "research/carls/base/proto_helper.h"

#include "google/protobuf/any.pb.h"  // proto to pb
#include "gtest/gtest.h"
#include "research/carls/testing/test_proto2.pb.h"  // proto to pb
#include "research/carls/testing/test_proto3.pb.h"  // proto to pb

namespace carls {

TEST(ProtoHelperTest, GetProto2ExtensionType) {
  // Without extension.
  TestBaseProto2Def test_proto;
  EXPECT_EQ(
      "carls.TestBaseProto2Def",
      GetExtensionType<Proto2Extension>(test_proto, test_proto.GetTypeName()));

  // With extension.
  test_proto.MutableExtension(TestExtendedProto2Def::ext);
  EXPECT_EQ(
      "carls.TestExtendedProto2Def",
      GetExtensionType<Proto2Extension>(test_proto, test_proto.GetTypeName()));
}

TEST(ProtoHelperTest, GetProto3AnyExtensionType) {
  // Without extension.
  TestBaseProto3Def test_proto;
  EXPECT_EQ("carls.TestBaseProto3Def",
            GetExtensionType<Proto3AnyField>(test_proto, "extension"));

  // With extension.
  TestExtendedProto3Def extension;
  test_proto.mutable_extension()->PackFrom(extension);
  EXPECT_EQ("carls.TestExtendedProto3Def",
            GetExtensionType<Proto3AnyField>(test_proto, "extension"));
}

TEST(ProtoHelperTest, ParseTextProtoOrDie) {
  TestBaseProto2Def test_proto = ParseTextProtoOrDie<TestBaseProto2Def>(R"(
    name: "test proto"
  )");
  EXPECT_DEATH(ParseTextProtoOrDie<TestBaseProto2Def>("invalid: 'name'"), "");
}

}  // namespace carls
