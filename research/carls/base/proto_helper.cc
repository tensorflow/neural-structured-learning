/* Copyright 2021 Google LLC. All Rights Reserved.

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

#include "research/carls/base/proto_helper.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "research/carls/base/file_helper.h"

namespace carls {
namespace {
constexpr char kAnyMessageName[] = "Any";
}  // namespace

template <>
std::string GetExtensionType<Proto2Extension>(const google::protobuf::Message& message,/*proto2*/
                                              const std::string& name) {
  const google::protobuf::Reflection* reflection = message.GetReflection();/*proto2*/
  std::vector<const google::protobuf::FieldDescriptor*> fields;/*proto2*/
  reflection->ListFields(message, &fields);
  for (const google::protobuf::FieldDescriptor* fd : fields) {/*proto2*/
    if (!fd->is_extension()) continue;
    if (fd->containing_type()->full_name() == name) {
      return GetExtensionType<Proto2Extension>(
          reflection->GetMessage(message, fd),
          fd->extension_scope()->full_name());
    }
  }
  return name;  // No more extension.
}

template <>
std::string GetExtensionType<Proto3AnyField>(const google::protobuf::Message& message,/*proto2*/
                                             const std::string& name) {
  const google::protobuf::Reflection* reflection = message.GetReflection();/*proto2*/
  std::vector<const google::protobuf::FieldDescriptor*> fields;/*proto2*/
  reflection->ListFields(message, &fields);
  for (const google::protobuf::FieldDescriptor* fd : fields) {/*proto2*/
    if (fd->name() == name) {
      const auto& any_message = reflection->GetMessage(message, fd);
      if (any_message.GetDescriptor()->name() != kAnyMessageName) {
        // Only accept one Any field.
        continue;
      }
      google::protobuf::Any any;
      CHECK(any.ParseFromString(any_message.SerializeAsString()));
      std::vector<std::string> comps = absl::StrSplit(any.type_url(), '/');
      return comps.back();
    }
  }
  return message.GetDescriptor()->full_name();  // No more extension.
}

absl::Status WriteBinaryProto(const std::string& filename,
                              const google::protobuf::Message& proto,/*proto2*/
                              bool can_overwrite) {
  if (!proto.IsInitialized()) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Cannot serialize proto, missing required field ",
                     proto.InitializationErrorString()));
  }
  return WriteFileString(filename, proto.SerializeAsString(), can_overwrite);
}

absl::Status ReadBinaryProto(const std::string& filename,
                             google::protobuf::Message* proto) {/*proto2*/
  CHECK(proto != nullptr);
  std::string data;
  auto status = ReadFileString(filename, &data);
  if (!status.ok()) {
    return status;
  }

  if (!proto->ParsePartialFromString(data)) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Could not parse file contents of ", filename,
                     " as wire-format protobuf of type ",
                     proto->GetTypeName()));
  }
  if (!proto->IsInitialized()) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        absl::StrCat("Could not parse file contents of ",
                                     filename, ", result uninitialized: ",
                                     proto->InitializationErrorString()));
  }

  return absl::OkStatus();
}

absl::Status WriteTextProto(const std::string& filename,
                            const google::protobuf::Message& proto, bool can_overwrite) {/*proto2*/
  std::string text_proto;
  if (!proto.IsInitialized()) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Cannot serialize proto, missing required field ",
                     proto.InitializationErrorString()));
  }
  if (!google::protobuf::TextFormat::PrintToString(proto, &text_proto)) {/*proto2*/
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat(
            "Failed to convert proto to text for saving to ", filename,
            " (this generally stems from massive protobufs that either "
            "exhaust memory or overflow a 32-bit buffer somewhere)."));
  }
  return WriteFileString(filename, text_proto, can_overwrite);
}

absl::Status ReadTextProto(const std::string& filename,
                           google::protobuf::Message* proto) {/*proto2*/
  std::string text_proto;
  auto status = ReadFileString(filename, &text_proto);
  if (!status.ok()) {
    return status;
  }

  google::protobuf::TextFormat::Parser parser;/*proto2*/
  if (!parser.ParseFromString(text_proto, proto)) {
    return absl::Status(
        absl::StatusCode::kInternal,
        absl::StrCat("Parsing text proto failed when reading file: ",
                     filename));
  }
  return absl::OkStatus();
}

}  // namespace carls
