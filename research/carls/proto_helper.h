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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_HELPER_H_

#include <string>

#include <glog/logging.h>
#include "google/protobuf/descriptor.h" // proto import
#include "google/protobuf/message.h" // proto import
#include "google/protobuf/text_format.h" // proto import

namespace carls {

// Use this as the template type to GetExtensionType() if the extension is
// specified as a proto2 extension.
struct Proto2Extension {};

// Use this as the template type to GetExtensionType() if the extension is
// specified in the Any field of a base message.
struct Proto3AnyField {};

// Get the type of the extension defined in a proto Message, whose base type is
// given. An extension can be specified by the following two ways:
// - From proto2 extension:
//   message BaseMessage {
//     extensions 1000 to max;
//   }
//   In this case, call
//     GetExtensionType<Proto2Extension>(base_message, base_message_name),
//   where base_message_name is the name of the base_message ("BaseMessage").
//
// - From an Any field (in proto3):
//   message BaseMessage {
//     google.protobuf.Any extension = 1;
//   }
//   In this case, call
//      GetExtensionType<Proto3AnyField>(base_message, extension_name),
//   where extension_name is the name for the Any field that is expected to be
//   an extension.
//
template <typename extension_type>
std::string GetExtensionType(const google::protobuf::Message& message,/*proto2*/
                             const std::string& name);

// Parses a given text into a protocol buffer.
template <typename ProtoType>
ProtoType ParseTextProtoOrDie(const std::string& proto_text) {
  ProtoType proto;
  CHECK(google::protobuf::TextFormat::ParseFromString(proto_text, &proto));/*proto2*/
  return proto;
}

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_HELPER_H_
