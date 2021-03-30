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
# Build rules for Dynamic Embedding Kernels.

load("//research/carls:bazel/build_rules.bzl", "carls_gen_op_wrapper_py")

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
    name = "dynamic_embedding_manager_resource",
    srcs = ["dynamic_embedding_manager_resource.cc"],
    hdrs = ["dynamic_embedding_manager_resource.h"],
    deps = [
        "//research/carls:dynamic_embedding_config_cc_proto",
        "//research/carls:dynamic_embedding_manager",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "dynamic_embedding_manager_resource_test",
    srcs = ["dynamic_embedding_manager_resource_test.cc"],
    deps = [
        ":dynamic_embedding_manager_resource",
        "//research/carls:kbs_server_helper_lib",
        "//research/carls/base:proto_helper",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_embedding_ops",
    srcs = ["dynamic_embedding_ops.cc"],
    deps = [
        ":dynamic_embedding_manager_resource",
        "//research/carls:dynamic_embedding_config_cc_proto",
        "//research/carls:dynamic_embedding_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

carls_gen_op_wrapper_py(
    name = "gen_dynamic_embedding_ops_py",
    out = "gen_dynamic_embedding_ops.py",
    kernel_lib = ":dynamic_embedding_ops",
)

cc_library(
    name = "io_ops",
    srcs = ["io_ops.cc"],
    deps = [
        ":dynamic_embedding_manager_resource",
        "//research/carls:constants",
        "//research/carls:dynamic_embedding_config_cc_proto",
        "//research/carls:dynamic_embedding_manager",
        "//research/carls/base:file_helper",
        "@com_google_absl//absl/strings:str_format",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

carls_gen_op_wrapper_py(
    name = "gen_io_ops_py",
    out = "gen_io_ops.py",
    kernel_lib = ":io_ops",
)
