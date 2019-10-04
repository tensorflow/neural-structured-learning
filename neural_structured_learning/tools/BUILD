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
        ":graph_builder_lib",
        ":graph_utils",
        ":input_maker_lib",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # package absl/logging
        # package six
    ],
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        # package absl/testing:absltest
    ],
)

py_library(
    name = "graph_builder_lib",
    srcs = ["graph_builder_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        # package absl/logging
        # package numpy
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "graph_builder_lib_test",
    srcs = ["graph_builder_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_builder_lib",
        ":graph_utils",
        # package protobuf,
        # package absl/testing:absltest
        # package tensorflow
    ],
)

py_binary(
    name = "graph_builder_main",
    srcs = ["graph_builder_main.py"],
    python_version = "PY3",
    deps = [
        ":graph_builder_lib",
        # package absl:app
        # package absl/flags
        # package tensorflow
    ],
)

py_library(
    name = "input_maker_lib",
    srcs = ["input_maker_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        # package absl/logging
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "input_maker_lib_test",
    srcs = ["input_maker_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_utils",
        ":input_maker_lib",
        # package protobuf,
        # package absl/testing:absltest
        # package tensorflow
    ],
)

py_binary(
    name = "input_maker",
    srcs = ["input_maker.py"],
    python_version = "PY3",
    deps = [
        ":input_maker_lib",
        # package absl:app
        # package absl/flags
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
