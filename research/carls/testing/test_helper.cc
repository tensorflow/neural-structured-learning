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
#include "tensorflow/core/platform/status.h"

namespace carls {
namespace internal {

template <>
std::string GetErrorMessage(const absl::Status& status) {
  return std::string(status.message());
}

template <>
std::string GetErrorMessage(const grpc::Status& status) {
  return status.error_message();
}

template <>
std::string GetErrorMessage(const tensorflow::Status& status) {
  return status.error_message();
}

}  // namespace internal
}  // namespace carls
