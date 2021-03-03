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
# Build rules for CARLS APIs in Neural Structured Learning.

# Placeholder for py clif.
# Placeholder for internal Python strict & test compatibility macro.
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
# Placeholder for cc proto.
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

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

py_library(
    name = "carls",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_neighbor_cache",
        ":graph_regularization",
        ":neighbor_cache_client",
    ],
)

proto_library(
    name = "embedding_proto",
    srcs = ["embedding.proto"],
)

cc_proto_library(
    name = "embedding_cc_proto",
    deps = [":embedding_proto"],
)

proto_library(
    name = "dynamic_embedding_config_proto",
    srcs = ["dynamic_embedding_config.proto"],
    deps = ["//research/carls/knowledge_bank:knowledge_bank_config_proto"],
)

cc_proto_library(
    name = "dynamic_embedding_config_cc_proto",
    deps = [":dynamic_embedding_config_proto"],
)

py_proto_library(
    name = "dynamic_embedding_config_py_pb2",
    deps = [":dynamic_embedding_config_proto"],
)

proto_library(
    name = "knowledge_bank_service_proto",
    srcs = ["knowledge_bank_service.proto"],
    deps = [
        ":dynamic_embedding_config_proto",
        ":embedding_proto",
    ],
)

cc_proto_library(
    name = "knowledge_bank_service_cc_proto",
    deps = [":knowledge_bank_service_proto"],
)

cc_grpc_library(
    name = "knowledge_bank_service_cc_grpc_proto",
    srcs = [":knowledge_bank_service_proto"],
    grpc_only = True,
    deps = [":knowledge_bank_service_cc_proto"],
)

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_github_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "proto_helper_test",
    srcs = ["proto_helper_test.cc"],
    deps = [
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

py_library(
    name = "neighbor_cache_client",
    srcs = ["neighbor_cache_client.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # package six
    ],
)

py_library(
    name = "graph_regularization",
    srcs = ["graph_regularization.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//neural_structured_learning/keras:graph_regularization",
        # package tensorflow
    ],
)

py_test(
    name = "graph_regularization_test",
    timeout = "long",
    srcs = ["graph_regularization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_regularization",
        ":neighbor_cache_client",
        # package protobuf,
        # package absl/testing:parameterized
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

cc_library(
    name = "knowledge_bank_grpc_service",
    srcs = ["knowledge_bank_grpc_service.cc"],
    hdrs = ["knowledge_bank_grpc_service.h"],
    deps = [
        ":knowledge_bank_service_cc_grpc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//research/carls/knowledge_bank",
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "knowledge_bank_grpc_service_test",
    srcs = ["knowledge_bank_grpc_service_test.cc"],
    deps = [
        ":knowledge_bank_grpc_service",
        ":proto_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kbs_server_helper_lib",
    srcs = ["kbs_server_helper.cc"],
    hdrs = ["kbs_server_helper.h"],
    deps = [
        ":knowledge_bank_grpc_service",
        # Placeholder for alternative grpc++
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "kbs_server_helper_lib_test",
    srcs = ["kbs_server_helper_test.cc"],
    deps = [
        ":kbs_server_helper_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "kbs_server_helper_pybind",
    srcs = ["kbs_server_helper_pybind.cc"],
    deps = [
        ":kbs_server_helper_lib",
    ],
)

py_test(
    name = "kbs_server_helper_test",
    srcs = ["kbs_server_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kbs_server_helper",
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

cc_library(
    name = "dynamic_embedding_manager",
    srcs = ["dynamic_embedding_manager.cc"],
    hdrs = ["dynamic_embedding_manager.h"],
    deps = [
        ":dynamic_embedding_config_cc_proto",
        ":knowledge_bank_grpc_service",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "@org_tensorflow//tensorflow/core:framework",
        "@com_github_grpc_grpc//:gpr_codegen",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_codegen_base",
    ],
)

cc_test(
    name = "dynamic_embedding_manager_test",
    srcs = ["dynamic_embedding_manager_test.cc"],
    deps = [
        ":dynamic_embedding_manager",
        ":kbs_server_helper_lib",
        ":proto_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "dynamic_embedding_ops_py",
    srcs = ["dynamic_embedding_ops.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_embedding_config_py_pb2",
        "//research/carls/kernels:dynamic_embedding_ops",
        "//research/carls/kernels:gen_dynamic_embedding_ops_py",
        # package tensorflow
    ],
)

py_test(
    name = "dynamic_embedding_ops_test",
    size = "small",
    srcs = ["dynamic_embedding_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_ops_py",
        # package absl/testing:parameterized
        "//research/carls/testing:test_util",
        # package tensorflow
    ],
)

py_library(
    name = "dynamic_embedding_neighbor_cache",
    srcs = ["dynamic_embedding_neighbor_cache.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_embedding_config_py_pb2",
        ":dynamic_embedding_ops_py",
        ":neighbor_cache_client",
    ],
)

py_test(
    name = "dynamic_embedding_neighbor_cache_test",
    size = "small",
    srcs = ["dynamic_embedding_neighbor_cache_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_neighbor_cache",
        "//research/carls/testing:test_util",
        # package tensorflow
    ],
)
