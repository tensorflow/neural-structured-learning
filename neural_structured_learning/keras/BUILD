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

# Placeholder for internal Python strict & test compatibility macro.

# Description:
# Build rules for Keras APIs in Neural Structured Learning.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    deps = [
        ":adversarial_regularization",
        ":graph_regularization",
        "//neural_structured_learning/keras/layers",
    ],
)

py_library(
    name = "adversarial_regularization",
    srcs = ["adversarial_regularization.py"],
    deps = [
        # package attr
        "//neural_structured_learning/configs",
        "//neural_structured_learning/lib:adversarial_neighbor",
        # package six
        # package tensorflow
    ],
)

py_test(
    name = "adversarial_regularization_multi_device_test",
    srcs = ["adversarial_regularization_multi_device_test.py"],
    deps = [
        ":adversarial_regularization",
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
    ],
)

py_test(
    name = "adversarial_regularization_test",
    srcs = ["adversarial_regularization_test.py"],
    deps = [
        ":adversarial_regularization",
        # package absl/testing:parameterized
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)

py_library(
    name = "graph_regularization",
    srcs = ["graph_regularization.py"],
    deps = [
        "//neural_structured_learning/configs",
        "//neural_structured_learning/keras/layers",
        "//neural_structured_learning/lib",
        # package tensorflow
    ],
)

py_test(
    name = "graph_regularization_test",
    timeout = "long",
    srcs = ["graph_regularization_test.py"],
    deps = [
        ":graph_regularization",
        # package protobuf,
        # package absl/testing:parameterized
        "//neural_structured_learning/configs",
        # package numpy
        # package tensorflow
        # package tensorflow framework_test_lib,
    ],
)
