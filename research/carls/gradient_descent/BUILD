# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//research/carls:bazel/build_rules.bzl",
    "carls_cc_proto_library",
    "carls_py_proto_library",
)

package(
    default_visibility = ["//research/carls:internal"],
    licenses = ["notice"],  # Apache 2.0
)

carls_cc_proto_library(
    name = "gradient_descent_config_cc_proto",
    srcs = ["gradient_descent_config.proto"],
)

carls_py_proto_library(
    name = "gradient_descent_config_py_pb2",
    srcs = ["gradient_descent_config.proto"],
    deps = [":gradient_descent_config_cc_proto"],
)

cc_library(
    name = "gradient_descent_optimizer",
    srcs = ["gradient_descent_optimizer.cc"],
    hdrs = ["gradient_descent_optimizer.h"],
    deps = [
        ":gradient_descent_config_cc_proto",
        "//research/carls:embedding_cc_proto",
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "@com_github_google_glog//:glog",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "gradient_descent_optimizer_test",
    srcs = ["gradient_descent_optimizer_test.cc"],
    deps = [
        ":gradient_descent_optimizer",
        "//research/carls/base:proto_helper",
        "//research/carls/testing:test_helper",
        "@com_google_googletest//:gtest_main",
    ],
)
