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
# Build rules for custom Keras layers in Neural Structured Learning.

package(
    default_visibility = ["//neural_structured_learning:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":neighbor_features",
        ":pairwise_distance",
    ],
)

py_test(
    name = "layers_test",
    size = "medium",
    srcs = [
        "layers_test.py",
    ],
    deps = [
        ":layers",
        # package absl/testing:parameterized
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

py_library(
    name = "neighbor_features",
    srcs = ["neighbor_features.py"],
    srcs_version = "PY2AND3",
    deps = [
        # package attr
        "//neural_structured_learning/configs",
        "//neural_structured_learning/lib:utils",
        # package tensorflow
    ],
)

py_test(
    name = "neighbor_features_test",
    srcs = [
        "neighbor_features_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":neighbor_features",
        # package absl/testing:parameterized
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

py_library(
    name = "pairwise_distance",
    srcs = [
        "pairwise_distance.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # package attr
        # package enum
        "//neural_structured_learning/configs",
        "//neural_structured_learning/lib:distances",
        "//neural_structured_learning/lib:utils",
        # package tensorflow
    ],
)

py_test(
    name = "pairwise_distance_test",
    srcs = [
        "pairwise_distance_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":pairwise_distance",
        "//neural_structured_learning/configs",
        # package numpy
        # package scipy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)
