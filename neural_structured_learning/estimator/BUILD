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

# Description:
# Build rules for Estimator APIs in Neural Structured Learning.

# Placeholder for internal Python strict compatibility macro.
# Placeholder for internal Python version compatibility macro.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "estimator",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":adversarial_regularization",
        ":graph_regularization",
    ],
)

py_library(
    name = "adversarial_regularization",
    srcs = ["adversarial_regularization.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//neural_structured_learning/configs",
        "//neural_structured_learning/lib",
        # package tensorflow
    ],
)

py_test(
    name = "adversarial_regularization_test",
    srcs = ["adversarial_regularization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

py_library(
    name = "graph_regularization",
    srcs = ["graph_regularization.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//neural_structured_learning/configs",
        "//neural_structured_learning/lib:distances",
        "//neural_structured_learning/lib:utils",
        # package tensorflow
    ],
)

py_test(
    name = "graph_regularization_test",
    srcs = ["graph_regularization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        # package protobuf,
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)
