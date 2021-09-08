"""Provides the repository macro to import rocksdb."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    """Imports rocksdb."""

    ROCKSDB_VERSION = "6.15.5"
    ROCKSDB_SHA256 = "d7b994e1eb4dff9dfefcd51a63f86630282e1927fc42a300b93c573c853aa5d0"

    http_archive(
        name = "rocksdb",
        build_file = "//research/carls/third_party/rocksdb:rocksdb.BUILD",
        sha256 = ROCKSDB_SHA256,
        strip_prefix = "rocksdb-{version}".format(version = ROCKSDB_VERSION),
        url = "https://github.com/facebook/rocksdb/archive/v{version}.tar.gz".format(version = ROCKSDB_VERSION),
    )

    # A dependency of rocksdb that is required for rocksdb::ClockCache.
    http_archive(
        name = "tbb",
        build_file = "//research/carls/third_party/rocksdb:tbb.BUILD",
        sha256 = "b182c73caaaabc44ddc5ad13113aca7e453af73c1690e4061f71dfe4935d74e8",
        strip_prefix = "oneTBB-2021.1.1",
        url = "https://github.com/oneapi-src/oneTBB/archive/v2021.1.1.tar.gz",
    )

    http_archive(
        name = "gflags",
        sha256 = "ce2931dd537eaab7dab78b25bec6136a0756ca0b2acbdab9aec0266998c0d9a7",
        strip_prefix = "gflags-827c769e5fc98e0f2a34c47cef953cc6328abced",
        url = "https://github.com/gflags/gflags/archive/827c769e5fc98e0f2a34c47cef953cc6328abced.tar.gz",
    )
