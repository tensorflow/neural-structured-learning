/*Copyright 2020 Google LLC

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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_FACTORY_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_FACTORY_H_

#include "google/protobuf/descriptor.h" // proto import
#include "google/protobuf/message.h" // proto import
#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "research/carls/proto_helper.h"

namespace carls {

// The name used for recognizing extension from the Any field.
constexpr absl::string_view ExtensionName() { return "extension"; }

// Macros for registering a base class with factory type.
#define REGISTER_KNOWLEDGE_BANK_BASE_CLASS_0(base_proto_type, base_class_type, \
                                             base_factory_type)                \
  using base_factory_type =                                                    \
      ::carls::FactoryBase<base_proto_type, base_class_type>

// Macros for registering a newly derived factory.
#define REGISTER_KNOWLEDGE_BANK_FACTORY_0(proto_type, factory_type,         \
                                          base_proto_type, base_class_type) \
  static bool g_##proto_type##_##base_proto_type##__object = [] {           \
    ::carls::FactoryBase<base_proto_type, base_class_type>::Register(       \
        proto_type::default_instance().GetDescriptor()->full_name(),        \
        factory_type);                                                      \
    return true;                                                            \
  }();

// Macros for registering a base class with one additional param.
#define REGISTER_KNOWLEDGE_BANK_BASE_CLASS_1(base_proto_type, base_class_type, \
                                             base_factory_type, params1_type)  \
  using base_factory_type =                                                    \
      ::carls::FactoryBase<base_proto_type, base_class_type, params1_type>

// Macros for registering a newly derived factory with one addition param.
#define REGISTER_KNOWLEDGE_BANK_FACTORY_1(                                    \
    proto_type, factory_type, base_proto_type, base_class_type, params1_type) \
  static bool g_##proto_type##_##base_proto_type##__object = [] {             \
    ::carls::FactoryBase<base_proto_type, base_class_type, params1_type>::    \
        Register(proto_type::default_instance().GetDescriptor()->full_name(), \
                 factory_type);                                               \
    return true;                                                              \
  }();

// The template factory class for all protocol buffer initiated factory methods.
// ProtoType:   the based protocol buffer for creating new instances.
// ClassType:   the interface (base class) that the factory produces.
// Params:      additional types that can be used for dependency injection, etc.
template <typename ProtoType, typename ClassType, typename... Params>
class FactoryBase {
 public:
  using FactoryType = std::function<std::unique_ptr<ClassType>(
      const ProtoType& def, Params... params)>;

  FactoryBase(const FactoryBase&) = delete;
  FactoryBase& operator=(const FactoryBase&) = delete;
  virtual ~FactoryBase() = default;

  // Get ClassType from protocol buffer definition.
  static std::unique_ptr<ClassType> Make(const ProtoType& def,
                                         Params... params) {
    std::string type_name = ::carls::GetExtensionType< ::carls::Proto3AnyField>(
        def, std::string(ExtensionName()));
    if (type_name.empty()) {
      LOG(ERROR) << "Empty type name.";
      return nullptr;
    }
    auto factory = GetFactoryByName(type_name);
    if (factory == nullptr) {
      LOG(ERROR) << "Cannot get factory for type_name: " << type_name;
      return nullptr;
    }
    return factory(def, params...);
  }

  // Register a factory.
  static void Register(const std::string& full_name, FactoryType factory) {
    absl::call_once(once_, InitInternalMaps);

    absl::MutexLock l(&factories_lock_);
    (*factories_)[full_name] = std::move(factory);
  }

  // Initializes factories_, this should be called only once.
  static void InitInternalMaps() {
    absl::MutexLock l(&factories_lock_);
    factories_ = new absl::node_hash_map<
        std::string,
        typename FactoryBase<ProtoType, ClassType, Params...>::FactoryType>();
  }

 protected:
  FactoryBase() = default;

 private:
  static FactoryType GetFactoryByName(const std::string& name) {
    absl::call_once(once_, InitInternalMaps);

    CHECK(!name.empty());
    absl::MutexLock l(&factories_lock_);
    CHECK(factories_ != nullptr);
    FactoryType factory = (*factories_)[name];
    if (factory == nullptr) {
      LOG(ERROR) << "Missing factory for type: " << name
                 << ". Please register one using REGISTER_***_FACTORY macro.";
    }
    return factory;
  }

  // For initializing factories_ .
  static absl::once_flag once_;

  // Access to factories needs to be lock protected since multiple threads may
  // attempt to add new factories into it during program startup.
  static absl::Mutex factories_lock_;
  static absl::node_hash_map<std::string, FactoryType>* factories_
      ABSL_GUARDED_BY(factories_lock_) ABSL_PT_GUARDED_BY(factories_lock_);
};

template <typename ProtoType, typename ClassType, typename... Params>
absl::once_flag FactoryBase<ProtoType, ClassType, Params...>::once_;

template <typename ProtoType, typename ClassType, typename... Params>
ABSL_CONST_INIT absl::Mutex
    FactoryBase<ProtoType, ClassType, Params...>::factories_lock_(  // NOLINT
        absl::kConstInit);

template <typename ProtoType, typename ClassType, typename... Params>
absl::node_hash_map<std::string, typename FactoryBase<ProtoType, ClassType,
                                                      Params...>::FactoryType>*
    FactoryBase<ProtoType, ClassType, Params...>::factories_ = nullptr;

}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_PROTO_FACTORY_H_
