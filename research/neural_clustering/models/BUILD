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
# Build rules for models in Neural Clustering.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":ncp_base",
        ":ncp_models",
        ":ncp_wrapper",
    ],
)

py_library(
    name = "ncp_base",
    srcs = ["ncp_base.py"],
    srcs_version = "PY3",
    deps = [
        # package numpy
        # package tensorflow
    ],
)

py_library(
    name = "ncp_models",
    srcs = ["ncp_models.py"],
    srcs_version = "PY3",
    deps = [
        ":ncp_base",
        # package tensorflow
    ],
)

py_library(
    name = "ncp_wrapper",
    srcs = ["ncp_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//research/neural_clustering/utils:data_utils",
        # package tensorflow
    ],
)

py_test(
    name = "ncp_test",
    srcs = ["ncp_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        # package absl/testing:parameterized
        # package numpy
        # package tensorflow
    ],
)
