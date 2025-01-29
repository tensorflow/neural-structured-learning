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
# Build rules for data modules in Neural Clustering.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "data_generators",
    srcs = ["__init__.py"],
    deps = [
        ":mog",
        ":partition",
    ],
)

py_library(
    name = "partition",
    srcs = ["partition.py"],
    deps = [
        # package numpy
    ],
)

py_test(
    name = "partition_test",
    srcs = ["partition_test.py"],
    deps = [
        ":partition",
        # package absl/testing:absltest
        # package absl/testing:parameterized
        # package numpy
    ],
)

py_library(
    name = "mog",
    srcs = ["mog.py"],
    deps = ["//third_party/py/numpy"],
)

py_test(
    name = "mog_test",
    srcs = ["mog_test.py"],
    deps = [
        ":mog",
        # package absl/testing:absltest
        # package absl/testing:parameterized
        # package numpy
    ],
)
