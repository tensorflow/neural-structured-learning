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
# Build rules for TensorFlow Neural Structured Learning.

# Internal annotation for sync
# Placeholder for internal Python strict compatibility macro.
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "neural_structured_learning",
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "neural_structured_learning",
    srcs = ["__init__.py"],
    deps = [
        ":version",
        "//neural_structured_learning/configs",
        "//neural_structured_learning/estimator",
        "//neural_structured_learning/experimental",
        "//neural_structured_learning/keras",
        "//neural_structured_learning/lib",
        "//neural_structured_learning/tools",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
