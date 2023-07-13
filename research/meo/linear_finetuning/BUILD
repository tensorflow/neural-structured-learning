# Copyright 2022 Google LLC
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

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//research/meo:__subpackages__"],
)

licenses(["notice"])

pytype_strict_library(
    name = "linear_finetuning_lib",
    srcs = [
        "linear_finetuning.py",
    ],
    srcs_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package absl/logging
        "//research/meo/mlp_baseline:mlp_baseline_lib",
        "//research/meo/mlp_baseline:multiple_decoders_lib",
        # package numpy
        # package tensorflow:tensorflow_no_contrib
    ],
)

pytype_strict_binary(
    name = "linear_finetuning",
    srcs = ["linear_finetuning.py"],
    python_version = "PY3",
    deps = [
        ":linear_finetuning_lib",
        # package absl:app
        # package absl/flags
        # package absl/logging
    ],
)
