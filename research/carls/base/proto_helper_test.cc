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
#include "research/carls/base/file_helper.h"
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

TEST(ProtoHelperTest, WriteAndReadBinaryProto) {
  TestBaseProto2Def test_proto = ParseTextProtoOrDie<TestBaseProto2Def>(R"(
    name: "test proto"
  )");
  std::string filepath =
      JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "/proto1.bin");
  auto status = WriteBinaryProto(filepath, test_proto, /*can_overwrite=*/true);
  EXPECT_TRUE(status.ok());

  TestBaseProto2Def test_proto_result;
  ASSERT_TRUE(ReadBinaryProto(filepath, &test_proto_result).ok());
  EXPECT_EQ("test proto", test_proto_result.name());

  // Test can_overwrite = false.
  status = WriteBinaryProto(filepath, test_proto, /*can_overwrite=*/false);
  EXPECT_FALSE(status.ok());
}

TEST(ProtoHelperTest, WriteAndReadTextProto) {
  TestBaseProto2Def test_proto =
      ParseTextProtoOrDie<TestBaseProto2Def>("name: 'test proto'");
  std::string filepath =
      JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "/proto2.bin");
  auto status = WriteTextProto(filepath, test_proto, /*can_overwrite=*/true);
  EXPECT_TRUE(status.ok());

  TestBaseProto2Def test_proto_result;
  ASSERT_TRUE(ReadTextProto(filepath, &test_proto_result).ok());
  EXPECT_EQ("test proto", test_proto_result.name());

  // Test can_overwrite = false.
  status = WriteTextProto(filepath, test_proto, /*can_overwrite=*/false);
  EXPECT_FALSE(status.ok());

  // Checks the content of the saved file.
  std::string content;
  ASSERT_TRUE(ReadFileString(filepath, &content).ok());
  EXPECT_EQ("name: \"test proto\"\n", content);
}

}  // namespace carls
