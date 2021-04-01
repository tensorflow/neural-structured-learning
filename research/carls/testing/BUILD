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

# Description:
# Code for testing CARLS related components.

load(
    "//research/carls:bazel/build_rules.bzl",
    "carls_cc_proto_library",
)

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    packages = [
        "//research/...",
    ],
)

carls_cc_proto_library(
    name = "test_proto2_cc_proto",
    testonly = 1,
    srcs = ["test_proto2.proto"],
)

carls_cc_proto_library(
    name = "test_proto3_cc_proto",
    testonly = 1,
    srcs = ["test_proto3.proto"],
)

cc_library(
    name = "test_helper",
    testonly = 1,
    srcs = ["test_helper.cc"],
    hdrs = ["test_helper.h"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@tensorflow_includes//:includes",
    ],
)

cc_test(
    name = "test_helper_test",
    srcs = ["test_helper_test.cc"],
    deps = [
        ":test_helper",
        ":test_proto2_cc_proto",
        "//research/carls/base:status_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tensorflow_includes//:includes",
    ],
)

py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//research/carls:dynamic_embedding_config_py_pb2",
        "//research/carls:kbs_server_helper_pybind",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)
