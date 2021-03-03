package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "graph_keras_mlp_cora",
    srcs = ["graph_keras_mlp_cora.py"],
    data = ["//research/carls:kbs_server_helper_pybind.so"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # package protobuf,
        # package absl/flags
        "//neural_structured_learning/keras:graph_regularization",
        "//research/carls:dynamic_embedding_config_py_pb2",
        "//research/carls:dynamic_embedding_neighbor_cache",
        "//research/carls:dynamic_embedding_ops_py",
        "//research/carls:graph_regularization",
        # package tensorflow
    ],
)
