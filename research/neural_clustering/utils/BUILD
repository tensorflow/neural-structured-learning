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

# Placeholder for internal Python strict & test compatibility macro.

# Description:
# Build rules for utils in Neural Clustering.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":data_utils",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        # package numpy
    ],
)

py_test(
    name = "data_utils_test",
    srcs = ["data_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_utils",
        # package absl/testing:absltest
        # package absl/testing:parameterized
        # package numpy
    ],
)
