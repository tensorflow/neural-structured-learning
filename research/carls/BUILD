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

load(
    "//research/carls:bazel/build_rules.bzl",
    "carls_cc_grpc_library",
    "carls_cc_proto_library",
    "carls_py_proto_library",
    "carls_pybind_extension",
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

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

carls_cc_proto_library(
    name = "input_context_cc_proto",
    srcs = ["input_context.proto"],
)

carls_py_proto_library(
    name = "input_context_py_pb2",
    srcs = ["input_context.proto"],
    deps = [":input_context_cc_proto"],
)

carls_cc_proto_library(
    name = "embedding_cc_proto",
    srcs = ["embedding.proto"],
    deps = [
        ":input_context_cc_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

carls_py_proto_library(
    name = "embedding_py_pb2",
    srcs = ["embedding.proto"],
    deps = [
        ":embedding_cc_proto",
        ":input_context_py_pb2",
    ],
)

carls_cc_proto_library(
    name = "dynamic_embedding_config_cc_proto",
    srcs = ["dynamic_embedding_config.proto"],
    deps = [
        "//research/carls/candidate_sampling:candidate_sampler_config_cc_proto",
        "//research/carls/gradient_descent:gradient_descent_config_cc_proto",
        "//research/carls/knowledge_bank:knowledge_bank_config_cc_proto",
        "//research/carls/memory_store:memory_store_config_cc_proto",
    ],
)

carls_py_proto_library(
    name = "dynamic_embedding_config_py_pb2",
    srcs = ["dynamic_embedding_config.proto"],
    deps = [
        ":dynamic_embedding_config_cc_proto",
        "//research/carls/candidate_sampling:candidate_sampler_config_py_pb2",
        "//research/carls/gradient_descent:gradient_descent_config_py_pb2",
        "//research/carls/knowledge_bank:knowledge_bank_config_py_pb2",
        "//research/carls/memory_store:memory_store_config_py_pb2",
    ],
)

carls_cc_proto_library(
    name = "knowledge_bank_service_cc_proto",
    srcs = ["knowledge_bank_service.proto"],
    deps = [
        ":dynamic_embedding_config_cc_proto",
        ":embedding_cc_proto",
        "//research/carls/candidate_sampling:candidate_sampler_config_cc_proto",
        "//research/carls/memory_store:memory_store_config_cc_proto",
    ],
)

carls_cc_grpc_library(
    name = "knowledge_bank_service_cc_grpc_proto",
    srcs = ["knowledge_bank_service.proto"],
    deps = [":knowledge_bank_service_cc_proto"],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_config_py_pb2",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
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

cc_library(
    name = "knowledge_bank_grpc_service",
    srcs = ["knowledge_bank_grpc_service.cc"],
    hdrs = ["knowledge_bank_grpc_service.h"],
    deps = [
        ":knowledge_bank_service_cc_grpc_proto",
        "//research/carls/candidate_sampling:brute_force_topk_sampler",
        "//research/carls/candidate_sampling:candidate_sampler",
        "//research/carls/candidate_sampling:negative_sampler",
        "//research/carls/gradient_descent:gradient_descent_optimizer",
        "//research/carls/knowledge_bank",
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "//research/carls/memory_store",
        "//research/carls/memory_store:gaussian_memory",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "knowledge_bank_grpc_service_test",
    srcs = ["knowledge_bank_grpc_service_test.cc"],
    deps = [
        ":knowledge_bank_grpc_service",
        "//research/carls/base:proto_helper",
        "//research/carls/testing:test_helper",
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

carls_pybind_extension(
    name = "kbs_server_helper_pybind",
    srcs = ["kbs_server_helper_pybind.cc"],
    module_name = "pywrap_kbs_server_helper_pybind",
    deps = [
        ":kbs_server_helper_lib",
        "@pybind11",
    ],
)

py_test(
    name = "kbs_server_helper_test",
    srcs = ["kbs_server_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kbs_server_helper_pybind",
        # package tensorflow
    ],
)

cc_library(
    name = "dynamic_embedding_manager",
    srcs = ["dynamic_embedding_manager.cc"],
    hdrs = ["dynamic_embedding_manager.h"],
    deps = [
        ":dynamic_embedding_config_cc_proto",
        ":knowledge_bank_grpc_service",
        "//research/carls/base:status_helper",
        "//research/carls/knowledge_bank:in_proto_knowledge_bank",
        "//research/carls/memory_store:gaussian_memory_config_cc_proto",
        "@com_github_grpc_grpc//:gpr",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

cc_test(
    name = "dynamic_embedding_manager_test",
    srcs = ["dynamic_embedding_manager_test.cc"],
    deps = [
        ":dynamic_embedding_manager",
        ":kbs_server_helper_lib",
        "//research/carls/base:proto_helper",
        "//research/carls/candidate_sampling:candidate_sampler_config_cc_proto",
        "//research/carls/testing:test_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
        "@tensorflow_solib//:framework_lib",
    ],
)

py_library(
    name = "dynamic_embedding_ops_py",
    srcs = ["dynamic_embedding_ops.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":context",
        ":dynamic_embedding_config_py_pb2",
        "//research/carls/kernels:gen_carls_ops_py",
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
        ":context",
        ":dynamic_embedding_ops_py",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "dynamic_memory_ops_py",
    srcs = ["dynamic_memory_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":context",
        ":dynamic_embedding_config_py_pb2",
        "//research/carls/kernels:gen_carls_ops_py",
        # package tensorflow
    ],
)

py_test(
    name = "dynamic_memory_ops_test",
    size = "small",
    srcs = ["dynamic_memory_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context",
        ":dynamic_memory_ops_py",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "dynamic_normalization_py",
    srcs = ["dynamic_normalization.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_memory_ops_py",
        # package tensorflow
    ],
)

py_test(
    name = "dynamic_normalization_test",
    size = "small",
    srcs = ["dynamic_normalization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context",
        ":dynamic_normalization_py",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "candidate_sampling_ops_py",
    srcs = ["candidate_sampling_ops.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":context",
        ":dynamic_embedding_config_py_pb2",
        "//research/carls/kernels:gen_carls_ops_py",
    ],
)

py_test(
    name = "candidate_sampling_ops_test",
    size = "small",
    srcs = ["candidate_sampling_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":candidate_sampling_ops_py",
        ":context",
        ":dynamic_embedding_ops_py",
        "//research/carls/candidate_sampling:candidate_sampler_config_builder_py",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
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
        ":context",
        ":dynamic_embedding_neighbor_cache",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
        # package tensorflow
    ],
)

py_library(
    name = "io_ops",
    srcs = ["io_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":context",
        ":dynamic_embedding_config_py_pb2",
        "//research/carls/kernels:gen_carls_ops_py",
    ],
)

py_test(
    name = "io_ops_test",
    srcs = ["io_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context",
        ":dynamic_embedding_ops_py",
        ":io_ops",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        # package tensorflow
    ],
)
