# Copyright 2020 Google LLC
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
# Codes related to KnowledgeBank component for CARLS.

# Placeholder for cc proto.

package(
    default_visibility = [
        ":internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    packages = [
        "//research/...",
    ],
)

proto_library(
    name = "initializer_proto",
    srcs = ["initializer.proto"],
    deps = ["//research/carls:embedding_proto"],
)

cc_proto_library(
    name = "initializer_cc_proto",
    deps = [":initializer_proto"],
)

proto_library(
    name = "knowledge_bank_config_proto",
    srcs = ["knowledge_bank_config.proto"],
    deps = [
        ":initializer_proto",
        "//research/carls:embedding_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "knowledge_bank_config_cc_proto",
    deps = [":knowledge_bank_config_proto"],
)

cc_library(
    name = "initializer_helper",
    srcs = ["initializer_helper.cc"],
    hdrs = ["initializer_helper.h"],
    deps = [
        ":initializer_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "//research/carls:embedding_cc_proto",
        "@com_github_google_glog//:glog",
    ],
)

cc_test(
    name = "initializer_helper_test",
    srcs = ["initializer_helper_test.cc"],
    deps = [
        ":initializer_helper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//research/carls:embedding_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "knowledge_bank",
    srcs = ["knowledge_bank.cc"],
    hdrs = ["knowledge_bank.h"],
    deps = [
        ":initializer_helper",
        ":knowledge_bank_config_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//research/carls:embedding_cc_proto",
        "//research/carls:proto_factory",
    ],
)

cc_test(
    name = "knowledge_bank_test",
    srcs = ["knowledge_bank_test.cc"],
    deps = [
        ":initializer_cc_proto",
        ":initializer_helper",
        ":knowledge_bank",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "in_proto_knowledge_bank",
    srcs = ["in_proto_knowledge_bank.cc"],
    deps = [
        ":initializer_helper",
        ":knowledge_bank",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "in_proto_knowledge_bank_test",
    srcs = ["in_proto_knowledge_bank_test.cc"],
    deps = [
        ":in_proto_knowledge_bank",
        ":initializer_cc_proto",
        ":initializer_helper",
        ":knowledge_bank",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
