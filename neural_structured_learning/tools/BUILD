# Copyright 2019 Google LLC
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

# Placeholder for loading py_binary
# Placeholder for internal Python strict & test compatibility macro.

# Description:
# Build rules for tools in Neural Structured Learning.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "tools",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":build_graph_lib",
        ":graph_utils",
        ":pack_nbrs_lib",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package six
    ],
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_utils",
        # package absl/testing:absltest
    ],
)

py_library(
    name = "build_graph_lib",
    srcs = ["build_graph.py"],
    srcs_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package absl/logging
        "//neural_structured_learning/configs",
        # package numpy
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "build_graph_test",
    srcs = ["build_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":build_graph_lib",
        ":graph_utils",
        # package protobuf,
        # package absl/testing:absltest
        # package six
        # package tensorflow
    ],
)

py_binary(
    name = "build_graph",
    srcs = ["build_graph.py"],
    python_version = "PY3",
    deps = [
        ":build_graph_lib",
    ],
)

py_library(
    name = "pack_nbrs_lib",
    srcs = ["pack_nbrs.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_utils",
        # package absl:app
        # package absl/flags
        # package absl/logging
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "pack_nbrs_test",
    srcs = ["pack_nbrs_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_utils",
        ":pack_nbrs_lib",
        # package protobuf,
        # package absl/testing:absltest
        # package tensorflow
    ],
)

py_binary(
    name = "pack_nbrs",
    srcs = ["pack_nbrs.py"],
    python_version = "PY3",
    deps = [
        ":pack_nbrs_lib",
        # package tensorflow
    ],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        "//neural_structured_learning",
        # package tensorflow_docs/api_generator
    ],
)
