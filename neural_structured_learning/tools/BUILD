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

# Placeholder for internal Python version compatibility macro.

# Description:
# Build rules for tools in Neural Structured Learning.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "tools",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":build_graph_lib",
        ":graph_utils",
        ":pack_nbrs_lib",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/absl/logging",
        # package six,
    ],
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_library(
    name = "build_graph_lib",
    srcs = ["build_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        # package numpy,
        # package six,
        # package tensorflow,
    ],
)

py_binary(
    name = "build_graph",
    srcs = ["build_graph.py"],
    python_version = "PY3",
    deps = [":build_graph_lib"],
)

py_library(
    name = "pack_nbrs_lib",
    srcs = ["pack_nbrs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        # package six,
        # package tensorflow,
    ],
)

py_binary(
    name = "pack_nbrs",
    srcs = ["pack_nbrs.py"],
    python_version = "PY3",
    deps = [":pack_nbrs_lib"],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    deps = [
        "//neural_structured_learning",
        "//third_party/py/tensorflow_docs/api_generator",
    ],
)
