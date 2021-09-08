# TODO(ppham27): describe this package.
# Placeholder for internal Python strict compatibility macro.

package(
    default_visibility = ["//neural_structured_learning:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":gnn",
    ],
)

py_library(
    name = "gnn",
    srcs = ["gnn.py"],
    srcs_version = "PY3",
    deps = [
        # package graph_nets
        "//neural_structured_learning/configs",
        "//neural_structured_learning/keras",
        "//neural_structured_learning/lib",
        # package sonnet/v2
        # package tensorflow
    ],
)
