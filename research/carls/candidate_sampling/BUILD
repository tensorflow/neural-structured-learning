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
# Build rules for the candidate sampling component of CARLS.

load("//research/carls:bazel/build_rules.bzl", "carls_cc_proto_library", "carls_py_proto_library")

package(
    default_visibility = ["//research/carls:internal"],
    licenses = ["notice"],  # Apache 2.0
)

carls_cc_proto_library(
    name = "candidate_sampler_config_cc_proto",
    srcs = ["candidate_sampler_config.proto"],
    deps = [
        "//research/carls:embedding_cc_proto",
    ],
)

carls_py_proto_library(
    name = "candidate_sampler_config_py_pb2",
    srcs = ["candidate_sampler_config.proto"],
    deps = [
        ":candidate_sampler_config_cc_proto",
        "//research/carls:embedding_py_pb2",
    ],
)

cc_library(
    name = "candidate_sampler",
    srcs = ["candidate_sampler.cc"],
    hdrs = ["candidate_sampler.h"],
    deps = [
        ":candidate_sampler_config_cc_proto",
        "//research/carls/base:proto_factory",
        "//research/carls/knowledge_bank",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "candidate_sampler_test",
    srcs = ["candidate_sampler_test.cc"],
    deps = [
        ":candidate_sampler",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "log_uniform_sampler",
    srcs = ["log_uniform_sampler.cc"],
    deps = [
        ":candidate_sampler",
        "//research/carls/base:proto_helper",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "log_uniform_sampler_test",
    srcs = ["log_uniform_sampler_test.cc"],
    deps = [
        ":candidate_sampler",
        ":log_uniform_sampler",
        "//research/carls/knowledge_bank:initializer_cc_proto",
        "//research/carls/testing:test_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "brute_force_topk_sampler",
    srcs = ["brute_force_topk_sampler.cc"],
    deps = [
        ":candidate_sampler",
        ":candidate_sampler_config_cc_proto",
        "//research/carls:embedding_cc_proto",
        "//research/carls/base:embedding_helper",
        "//research/carls/base:top_n",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "brute_force_topk_sampler_test",
    srcs = ["brute_force_topk_sampler_test.cc"],
    deps = [
        ":brute_force_topk_sampler",
        ":candidate_sampler",
        "//research/carls:embedding_cc_proto",
        "//research/carls/knowledge_bank:initializer_cc_proto",
        "//research/carls/knowledge_bank:initializer_helper",
        "//research/carls/testing:test_helper",
        "@com_google_googletest//:gtest_main",
    ],
)
