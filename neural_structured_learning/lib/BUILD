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

# Placeholder for internal Python strict compatibility macro.
# Placeholder for internal Python version compatibility macro.

# Description:
# Build rules for libraries and ops in Neural Structured Learning.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "lib",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":abstract_gen_neighbor",
        ":adversarial_neighbor",
        ":distances",
        ":regularizer",
        ":utils",
    ],
)

# The build targets below are alphabetically sorted.

py_library(
    name = "abstract_gen_neighbor",
    srcs = ["abstract_gen_neighbor.py"],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "adversarial_neighbor",
    srcs = ["adversarial_neighbor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":abstract_gen_neighbor",
        ":utils",
        # package absl/logging
        # package tensorflow
    ],
)

py_test(
    name = "adversarial_neighbor_test",
    srcs = ["adversarial_neighbor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":adversarial_neighbor",
        "//neural_structured_learning/configs",
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

py_library(
    name = "distances",
    srcs = ["distances.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//neural_structured_learning/configs",
        # package tensorflow
    ],
)

py_test(
    name = "distances_test",
    srcs = ["distances_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distances",
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
    ],
)

py_library(
    name = "regularizer",
    srcs = ["regularizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distances",
        ":utils",
        # package tensorflow
    ],
)

py_test(
    name = "regularizer_test",
    srcs = ["regularizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":regularizer",
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//neural_structured_learning/configs",
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)
