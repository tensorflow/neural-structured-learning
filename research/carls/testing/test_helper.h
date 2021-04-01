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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_TESTING_TEST_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_TESTING_TEST_HELPER_H_

#include <glog/logging.h>
#include "google/protobuf/message.h" // proto import
#include "google/protobuf/text_format.h" // proto import
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace carls {
namespace internal {

template <typename StatusType>
std::string GetErrorMessage(const StatusType& status);

}  // namespace internal

// A simple implementation of a proto matcher comparing string representations.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const google::protobuf::Message& expected)/*proto2*/
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p, testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const google::protobuf::Message& message) {/*proto2*/
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(message));
}

template <typename ProtoType>
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const std::string& asciipb) {
  ProtoType proto;
  CHECK(google::protobuf::TextFormat::ParseFromString(asciipb, &proto));/*proto2*/
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(proto));
}

// Macros for testing the results of functions that return Status or
// StatusOr<T> (for any type T).
#undef EXPECT_OK
#define EXPECT_OK(expression) EXPECT_TRUE(expression.ok())

#define EXPECT_NOT_OK(expression) EXPECT_FALSE(expression.ok())

#define EXPECT_ERROR(expression, err_msg) \
  ASSERT_FALSE(expression.ok());          \
  EXPECT_EQ(err_msg, internal::GetErrorMessage(expression));

#undef ASSERT_OK
#define ASSERT_OK(expression) ASSERT_TRUE(expression.ok())

#define ASSERT_ERROR(expression, err_msg) \
  ASSERT_FALSE(expression.ok());          \
  ASSERT_EQ(err_msg, internal::GetErrorMessage(expression));

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_TESTING_TEST_HELPER_H_
