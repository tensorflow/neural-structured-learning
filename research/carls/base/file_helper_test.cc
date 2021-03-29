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

#include "research/carls/base/file_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carls {

using ::testing::TempDir;

TEST(FileHelperTest, JoinPath) {
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo/", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "/bar"));
  EXPECT_EQ("foo/bar", JoinPath("foo", "bar"));
  EXPECT_EQ("/foo/bar/", JoinPath("/foo/bar", "/"));
  EXPECT_EQ("/foo/bar", JoinPath("/", "foo", "bar"));
}

TEST(FileHelperTest, WriteAndReadFileString) {
  std::string filepath = JoinPath(TempDir(), "/data");

  std::string data("saved data");
  ASSERT_TRUE(WriteFileString(filepath, data, /*can_overwrite=*/true).ok());

  std::string result;
  ASSERT_TRUE(ReadFileString(filepath, &result).ok());
  EXPECT_EQ("saved data", result);

  // Cannot overwrite.
  EXPECT_FALSE(WriteFileString(filepath, data, /*can_overwrite=*/false).ok());

  // Non-existent file.
  filepath = JoinPath(TempDir(), "/non_exist_data");
  EXPECT_FALSE(ReadFileString(filepath, &result).ok());
}

TEST(FileHelperTest, IsDirectory) {
  std::string dirname = JoinPath(TempDir(), "/dir/subdir");
  ASSERT_TRUE(RecursivelyCreateDir(dirname).ok());
  EXPECT_TRUE(IsDirectory(dirname).ok());
}

}  // namespace carls
