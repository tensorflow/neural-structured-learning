# TODO(ppham27): describe this package.
# Placeholder for internal Python strict compatibility macro.

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//neural_structured_learning:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    deps = [
        ":gnn",
    ],
)

py_library(
    name = "gnn",
    srcs = ["gnn.py"],
    deps = [
        # package graph_nets
        "//neural_structured_learning/configs",
        "//neural_structured_learning/keras",
        "//neural_structured_learning/lib",
        # package sonnet/v2
        # package tensorflow
    ],
)
