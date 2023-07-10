"""Provides the repository macro to import leveldb."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    """Imports leveldb."""

    # Attention: tools parse and update these lines.
    LEVELDB_VERSION = "1.23"
    LEVELDB_SHA256 = "9a37f8a6174f09bd622bc723b55881dc541cd50747cbd08831c2a82d620f6d76"

    http_archive(
        name = "com_google_leveldb",
        build_file = "//research/carls/knowledge_bank/leveldb:leveldb.BUILD",
        patch_cmds = [
            """mkdir leveldb; cp include/leveldb/* leveldb""",
        ],
        sha256 = LEVELDB_SHA256,
        strip_prefix = "leveldb-{version}".format(version = LEVELDB_VERSION),
        urls = [
            "https://github.com/google/leveldb/archive/refs/tags/{version}.tar.gz".format(version = LEVELDB_VERSION),
        ],
    )
