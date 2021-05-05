/*Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_FILE_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_FILE_HELPER_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace carls {
namespace internal {

// Internal implementation of JoinPath.
std::string JoinPathImpl(std::initializer_list<absl::string_view> paths);

}  // namespace internal

// Join multiple paths together and also removes the unnecessary path
// separators.
//
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = JoinPath("/mydir", filename);
// string path = JoinPath(FLAGS_test_srcdir, filename);
// string path = JoinPath("/full", "path", "to", "filename");
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

// Reads the content of a file into given string output.
absl::Status ReadFileString(const std::string& filepath, std::string* output);

// Writes the given content to a given file.
// When can_overwrite = false, do not overwrite existing file and return an
// error message.
// When can_overwrite = true, existing file would be overwritten.
absl::Status WriteFileString(const std::string& filepath,
                             absl::string_view content, bool can_overwrite);

// Checks if given path is a directory.
absl::Status IsDirectory(const std::string& path);

// Returns the part of the path before the final "/", EXCEPT:
// * If there is a single leading "/" in the path, the result will be the
//   leading "/".
// * If there is no "/" in the path, the result is the empty prefix of the
//   input string.
absl::string_view Dirname(absl::string_view path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
// Note that this function's behavior differs from the Unix basename
// command if path ends with "/". For such paths, this function returns the
// empty string.
absl::string_view Basename(absl::string_view path);

// Return the parts of the path, split on the final "/".  If there is no
// "/" in the path, the first part of the output is empty and the second
// is the input. If the only "/" in the path is the first character, it is
// the first part of the output.
std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path);

// Creates a path if it doesn't exist.
absl::Status RecursivelyCreateDir(const std::string& dirname);

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_BASE_FILE_HELPER_H_
