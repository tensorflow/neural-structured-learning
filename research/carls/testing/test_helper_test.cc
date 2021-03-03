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

#include "research/carls/testing/test_proto2.pb.h"  // proto to pb

namespace carls {

TEST(TestHelperTest, EqualsProto) {
  TestBaseProto2Def proto;
  proto.set_name("Audrey");
  EXPECT_THAT(proto, EqualsProto(proto));
}

TEST(TestHelperTest, EqualsProtoText) {
  TestBaseProto2Def proto;
  proto.set_name("Audrey");
  EXPECT_THAT(proto, EqualsProto<TestBaseProto2Def>(R"(
                name: "Audrey"
              )"));
}

}  // namespace carls
