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

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = [
        ":file_helper",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "proto_helper_test",
    srcs = ["proto_helper_test.cc"],
    deps = [
        ":file_helper",
        ":proto_helper",
        "//research/carls/testing:test_proto2_cc_proto",
        "//research/carls/testing:test_proto3_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "proto_factory",
    hdrs = [
        "proto_factory.h",
    ],
    deps = [
        ":proto_helper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "proto_factory_test",
    srcs = ["proto_factory_test.cc"],
    deps = [
        ":proto_factory",
        "//research/carls/testing:test_proto3_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "status_helper",
    srcs = ["status_helper.cc"],
    hdrs = ["status_helper.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_codegen_base",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

cc_test(
    name = "status_helper_test",
    srcs = ["status_helper_test.cc"],
    deps = [
        ":status_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "file_helper",
    srcs = ["file_helper.cc"],
    hdrs = ["file_helper.h"],
    deps = [
        ":status_helper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tensorflow_solib//:framework_lib",
    ],
)

cc_test(
    name = "file_helper_test",
    srcs = ["file_helper_test.cc"],
    deps = [
        ":file_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)