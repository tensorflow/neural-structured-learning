# Placeholder for loading py_binary

package(
    default_applicable_licenses = ["//neural_structured_learning:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_binary(
    name = "graph_keras_mlp_cora",
    srcs = ["graph_keras_mlp_cora.py"],
    python_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package absl/logging
        # package attr
        "//neural_structured_learning",
        # package tensorflow
    ],
)

py_binary(
    name = "graph_nets_cora_graph_regularization",
    srcs = ["graph_nets_cora_graph_regularization.py"],
    python_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package graph_nets
        "//neural_structured_learning",
        "//neural_structured_learning/experimental:gnn",
        # package tensorflow
    ],
)

py_binary(
    name = "graph_nets_cora_gcn",
    srcs = ["graph_nets_cora_gcn.py"],
    python_version = "PY3",
    deps = [
        # package absl:app
        # package absl/flags
        # package graph_nets
        "//neural_structured_learning",
        "//neural_structured_learning/experimental:gnn",
        # package tensorflow
    ],
)
