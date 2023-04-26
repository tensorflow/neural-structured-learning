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

#include <string>

#include "grpcpp/support/status.h"  // net
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"

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

// Starting from TF 2.13, `tensorflow::Status` will be an alias to
// `absl::Status`, thus, we don't define the specialization in that case.
template <class T,
          std::enable_if_t<!std::is_same<T, absl::Status>::value, bool> = true>
std::string GetErrorMessage(const tensorflow::Status& status) {
// On April 2023, there is not yet an official release of Tensorflow which
// includes `message().` One will need to wait for the release following 2.12.0.
// The code can be updated to just be the else branch after such release exists.
#if TF_GRAPH_DEF_VERSION < 1467
  return status.error_message();
#else
  return std::string(status.message());
#endif
}

}  // namespace internal
}  // namespace carls
