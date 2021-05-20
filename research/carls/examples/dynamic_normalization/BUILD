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

# Description:
# Code for DynamicNormalization related applications.

package(
    default_visibility = ["//research/carls:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "image_dataset_eval_main",
    srcs = ["image_dataset_eval_main.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//research/carls:dynamic_embedding_config_py_pb2",
        "//research/carls:dynamic_memory_ops_py",
        "//research/carls:dynamic_normalization_py",
        "//research/carls:kbs_server_helper_pybind",
        "@com_google_absl_py//absl/testing:parameterized",
        # package numpy
        # package six
        # package tensorflow
        # package tensorflow_datasets
    ],
)
