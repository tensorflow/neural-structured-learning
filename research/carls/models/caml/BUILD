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
# Build rules for CARLS models for Context Aware Machine Learning (CAML).

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
    name = "sparse_features_cc_proto",
    srcs = ["sparse_features.proto"],
)

carls_py_proto_library(
    name = "sparse_features_config_py_pb2",
    srcs = ["sparse_features_config.proto"],
    deps = [":sparse_features_cc_proto"],
)

py_library(
    name = "sparse_features_py",
    srcs = ["sparse_features.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//research/carls:dynamic_embedding_config_py_pb2",
        "//research/carls:dynamic_embedding_ops_py",
        # package tensorflow
        # package six
    ],
)

py_test(
    name = "sparse_features_test",
    size = "small",
    srcs = ["sparse_features_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sparse_features_py",
        "//research/carls/testing:test_util",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)
