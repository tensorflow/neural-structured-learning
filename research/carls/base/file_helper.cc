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

#include "research/carls/base/file_helper.h"

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "research/carls/base/status_helper.h"
#include "tensorflow/core/platform/env.h"

namespace carls {
namespace internal {

bool IsAbsolutePath(absl::string_view path) {
  return !path.empty() && path[0] == '/';
}

std::string JoinPathImpl(std::initializer_list<absl::string_view> paths) {
  std::string result;

  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        absl::StrAppend(&result, path.substr(1));
      } else {
        absl::StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        absl::StrAppend(&result, path);
      } else {
        absl::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

}  // namespace internal

absl::Status ReadFileString(const std::string& filepath, std::string* output) {
  CHECK(output != nullptr);
  tensorflow::Env* env = tensorflow::Env::Default();

  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file;
  auto tf_status = env->NewReadOnlyMemoryRegionFromFile(filepath, &file);
  if (!tf_status.ok()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrCat("Reading file failed with error:",
                                     tf_status.error_message()));
  }
  *output = std::string(static_cast<const char*>(file->data()), file->length());
  return absl::OkStatus();
}

absl::Status WriteFileString(const std::string& filepath,
                             absl::string_view content, bool can_overwrite) {
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!can_overwrite && env->FileExists(filepath).ok()) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        absl::StrCat("File already exists: ", filepath));
  }
  std::unique_ptr<tensorflow::WritableFile> file;
  auto tf_status = env->NewWritableFile(filepath, &file);
  if (!tf_status.ok()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrCat("Creating file failed with error:",
                                     tf_status.error_message()));
  }
  tf_status = file->Append(content);
  if (!tf_status.ok()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrCat("Appending file failed with error:",
                                     tf_status.error_message()));
  }
  tf_status = file->Close();
  if (!tf_status.ok()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrCat("Closing file failed with error:",
                                     tf_status.error_message()));
  }
  return absl::OkStatus();
}

absl::Status IsDirectory(const std::string& path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  return carls::ToAbslStatus(env->IsDirectory(path));
}

absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  auto pos = path.find_last_of('/');

  // Handle the case with no '/' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(path.substr(0, 0), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));

  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

absl::Status RecursivelyCreateDir(const std::string& dirname) {
  tensorflow::Env* env = tensorflow::Env::Default();
  return carls::ToAbslStatus(env->RecursivelyCreateDir(dirname));
}

}  // namespace carls
