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
# Code for testing DynamicEmbedding related components.

# Placeholder for internal Python strict compatibility macro.
# Placeholder for cc proto.

package(
    default_visibility = [
        ":internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    packages = [
        "//research/...",
    ],
)

proto_library(
    name = "test_proto2_proto",
    srcs = ["test_proto2.proto"],
)

cc_proto_library(
    name = "test_proto2_cc_proto",
    testonly = 1,
    deps = [":test_proto2_proto"],
)

proto_library(
    name = "test_proto3_proto",
    srcs = ["test_proto3.proto"],
    deps = [
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "test_proto3_cc_proto",
    testonly = 1,
    deps = [":test_proto3_proto"],
)

py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    data = ["//research/carls:kbs_server_helper_pybind.so"],
    srcs_version = "PY3",
    deps = [
        # package protobuf,
        # package absl/flags
        "//research/carls:dynamic_embedding_config_py_pb2",
        # package tensorflow
    ],
)

cc_library(
    name = "test_helper",
    testonly = 1,
    hdrs = ["test_helper.h"],
    deps = [
        "@com_google_googletest//:gtest  # gunit_for_library_testonly",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "test_helper_test",
    srcs = ["test_helper_test.cc"],
    deps = [
        ":test_helper",
        ":test_proto2_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
