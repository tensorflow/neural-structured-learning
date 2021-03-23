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

#include "google/protobuf/any.pb.h"  // proto to pb
#include "absl/strings/str_split.h"

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

}  // namespace carls
