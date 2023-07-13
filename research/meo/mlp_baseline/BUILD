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
    name = "mlp_baseline_lib",
    srcs = [
        "configs.py",
        "data_utils.py",
        "extended_model.py",
        "losses.py",
        "mlp_baseline.py",
        "obfuscations.py",
    ],
    srcs_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package absl/logging
        # package numpy
        # package tensorflow:tensorflow_no_contrib
        # package tensorflow_datasets
        # package tensorflow_hub
    ],
)

pytype_strict_binary(
    name = "mlp_baseline",
    srcs = ["mlp_baseline.py"],
    python_version = "PY3",
    deps = [
        ":mlp_baseline_lib",
        # package absl:app
        # package absl/flags
        # package absl/logging
    ],
)

pytype_strict_library(
    name = "multiple_decoders_lib",
    srcs = [
        "configs.py",
        "data_utils.py",
        "extended_model.py",
        "losses.py",
        "multiple_decoders.py",
        "obfuscations.py",
    ],
    srcs_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package absl/logging
        # package numpy
        # package tensorflow:tensorflow_no_contrib
        # package tensorflow_datasets
        # package tensorflow_hub
        "@org_tensorflow//tensorflow_text:ops_lib",
    ],
)

pytype_strict_binary(
    name = "multiple_decoders",
    srcs = ["multiple_decoders.py"],
    python_version = "PY3",
    deps = [
        ":multiple_decoders_lib",
        # package absl:app
        # package absl/flags
        # package absl/logging
    ],
)
