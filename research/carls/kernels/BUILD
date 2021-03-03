# Copyright 2020 Google LLC
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
# Build rules for Dynamic Embedding Kernels.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    packages = [
        "//neural_structured_learning/...",
    ],
)

cc_library(
    name = "dynamic_embedding_ops",
    srcs = ["dynamic_embedding_ops.cc"],
    deps = [
        "//research/carls:dynamic_embedding_config_cc_proto",
        "//research/carls:dynamic_embedding_manager",
        "@org_tensorflow//tensorflow/cc:grad_op_registry",
        "@org_tensorflow//tensorflow/cc:gradients",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_dynamic_embedding_ops_py",
    out = "gen_dynamic_embedding_ops.py",
    deps = [":dynamic_embedding_ops"],
)
