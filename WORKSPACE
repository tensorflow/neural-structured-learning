workspace(name = "org_tensorflow_neural_structured_learning")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# farmhash
load("//research/carls/knowledge_bank/farmhash:workspace.bzl", farmhash = "repo")
farmhash()

# leveldb
load("//research/carls/knowledge_bank/leveldb:workspace.bzl", leveldb = "repo")
leveldb()

# rocksdb
load("//research/carls/knowledge_bank/rocksdb:workspace.bzl", rocksdb = "repo")
rocksdb()

# absl
http_archive(
    name = "com_google_absl",
    sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",  # SHARED_ABSL_SHA
    strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl_py",
    sha256 = "b3b94472335becdb3c8ef5132e84ec3d17f408526a84740a70b94c839b0894e6",
    strip_prefix = "abseil-py-c99edd8e3dffe3667a9f086db86aa259927ac429",
    urls = [
        "https://github.com/abseil/abseil-py/archive/c99edd8e3dffe3667a9f086db86aa259927ac429.tar.gz",
    ],
)

# Flags
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

# Logging
http_archive(
    name = "com_github_google_glog",
    urls = ["https://github.com/google/glog/archive/96a2f23dca4cc7180821ca5f32e526314395d26a.zip"],
    strip_prefix = "glog-96a2f23dca4cc7180821ca5f32e526314395d26a",
    sha256 = "6281aa4eeecb9e932d7091f99872e7b26fa6aacece49c15ce5b14af2b7ec050f",
)

# Testing
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/master.zip"],
     strip_prefix = "googletest-master",
)


# Use local tf to avoid error that tensorflow objects already registered.
# Use custom protoc to make sure all protoc are built on the version of local tf
# curl -L "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" | sha256
load("//research/carls:bazel/repo.bzl", "cc_tf_configure")
cc_tf_configure()

# Load protobuf compiler, protobuf and gRPC.
# They MUST be in sync with TensorFlow's corresponding versions defined in
# TENSORFLOW_DIR/tensorflow/workspace2.bzl

# Protobuffer
http_archive(
    name = "com_google_protobuf",
    sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
    strip_prefix = "protobuf-3.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    ],
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# gRPC
http_archive(
    name = "com_github_grpc_grpc",
    patch_cmds = [
        """sed -i.bak 's/"python",/"python3",/g' third_party/py/python_configure.bzl""",
        """sed -i.bak 's/PYTHONHASHSEED=0/PYTHONHASHSEED=0 python3/g' bazel/cython_library.bzl""",
        # remove -Werror from @upb/BUILD
        """sed -z -i.bak 's/name = \"upb\",/name = \"upb\",\\npatch_cmds=[\"\"\" """ +
        """sed -i.bak '\\''s\\/\\\"-Werror\\\"\\/#\\\"-Werror\\\"\\/g'\\'' BUILD\"\"\"],/' bazel/grpc_deps.bzl""",
    ],
    sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
    strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
        "https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# For using C++ in python
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-203508e14aab7309892a1c5f7dd05debda22d9a5",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/203508e14aab7309892a1c5f7dd05debda22d9a5.zip"],
    sha256 = "75922da3a1bdb417d820398eb03d4e9bd067c4905a4246d35a44c01d62154d91",
)

# Pybind
http_archive(
    name = "pybind11",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
    ],
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
)

# License rules.
http_archive(
    name = "rules_license",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
    ],
    sha256 = "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
)
