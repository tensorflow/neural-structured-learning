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
# Code related to MemoryStore component for CARLS.

load("//research/carls:bazel/build_rules.bzl", "carls_cc_proto_library", "carls_py_proto_library")

package(
    default_visibility = ["//research/carls:internal"],
    licenses = ["notice"],  # Apache 2.0
)

carls_cc_proto_library(
    name = "memory_distance_config_cc_proto",
    srcs = ["memory_distance_config.proto"],
)

carls_py_proto_library(
    name = "memory_distance_config_py_pb2",
    srcs = ["memory_distance_config.proto"],
    deps = [
        ":memory_distance_config_cc_proto",
    ],
)

carls_cc_proto_library(
    name = "gaussian_memory_config_cc_proto",
    srcs = ["gaussian_memory_config.proto"],
    deps = [
        ":memory_distance_config_cc_proto",
        "//research/carls:embedding_cc_proto",
    ],
)

carls_py_proto_library(
    name = "gaussian_memory_config_py_pb2",
    srcs = ["gaussian_memory_config.proto"],
    deps = [
        ":gaussian_memory_config_cc_proto",
        ":memory_distance_config_py_pb2",
    ],
)

carls_cc_proto_library(
    name = "memory_store_config_cc_proto",
    srcs = ["memory_store_config.proto"],
    deps = [
        ":gaussian_memory_config_cc_proto",
    ],
)

carls_py_proto_library(
    name = "memory_store_config_py_pb2",
    srcs = ["memory_store_config.proto"],
    deps = [
        ":gaussian_memory_config_py_pb2",
        ":memory_store_config_cc_proto",
    ],
)

cc_library(
    name = "distance_helper",
    srcs = ["distance_helper.cc"],
    hdrs = ["distance_helper.h"],
    deps = [
        ":memory_distance_config_cc_proto",
        "@com_github_google_glog//:glog",
    ],
)

cc_test(
    name = "distance_helper_test",
    srcs = ["distance_helper_test.cc"],
    deps = [
        ":distance_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_store",
    srcs = ["memory_store.cc"],
    hdrs = ["memory_store.h"],
    deps = [
        ":memory_store_config_cc_proto",
        "//research/carls:embedding_cc_proto",
        "//research/carls/base:file_helper",
        "//research/carls/base:proto_factory",
        "//research/carls/base:status_helper",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "memory_store_test",
    srcs = ["memory_store_test.cc"],
    deps = [
        ":memory_store",
        "//research/carls/base:proto_helper",
        "//research/carls/testing:test_helper",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gaussian_memory",
    srcs = ["gaussian_memory.cc"],
    deps = [
        ":distance_helper",
        ":gaussian_memory_config_cc_proto",
        ":memory_store",
        "//research/carls/base:embedding_helper",
        "//research/carls/base:file_helper",
        "//research/carls/base:proto_helper",
        "//research/carls/base:status_helper",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
    ],
    alwayslink = 1,
)

cc_test(
    name = "gaussian_memory_test",
    srcs = ["gaussian_memory_test.cc"],
    deps = [
        ":gaussian_memory",
        ":gaussian_memory_config_cc_proto",
        ":memory_distance_config_cc_proto",
        ":memory_store",
        "//research/carls:embedding_cc_proto",
        "//research/carls/base:file_helper",
        "//research/carls/base:proto_helper",
        "//research/carls/testing:test_helper",
        "@com_google_googletest//:gtest_main",
    ],
)
