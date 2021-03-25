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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_STATUS_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_STATUS_HELPER_H_

#include "absl/status/status.h"
#include "grpcpp/impl/codegen/client_context.h"  // third_party
#include "tensorflow/core/platform/status.h"

namespace carls {

// Converts from grpc::Status to absl::Status.
absl::Status ToAbslStatus(const grpc::Status& status);

// Converts from tensorflow::Status to absl::Status.
absl::Status ToAbslStatus(const tensorflow::Status& status);

// Converts from absl::Status to grpc::Status.
grpc::Status ToGrpcStatus(const absl::Status& status);

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_STATUS_HELPER_H_
