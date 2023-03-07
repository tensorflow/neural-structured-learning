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

#include "research/carls/kernels/dynamic_embedding_manager_resource.h"

#include "tensorflow/core/protobuf/error_codes.pb.h"  // proto to pb

namespace carls {
namespace {

using ::tensorflow::OpKernelConstruction;
using ::tensorflow::ResourceOpKernel;
using ::tensorflow::Status;
using ::tensorflow::errors::InvalidArgument;

absl::Duration ms_to_duration(int timeout_ms) {
  if (timeout_ms < 0) {
    return absl::InfiniteDuration();
  } else {
    return absl::Milliseconds(timeout_ms);
  }
}

}  // namespace

REGISTER_OP("DynamicEmbeddingManagerResource")
    .Output("handle: resource")
    .Attr("serialized_config: string")
    .Attr("var_name: string")
    .Attr("kbs_address: string = ''")
    .Attr("timeout_ms: int = -1")
    .Attr("container: string = ''")    // Required: using the default container.
    .Attr("shared_name: string = ''")  // Required: private resource.
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Constructs a `DynamicEmbeddingManagerResource` that connects to
DynamicEmbeddingManager. The resource allows ops to share the stub across calls.

Note that var_name is used for identifying the dynamic embedding data, and
shared_name is used to indicate if the resource is private or not.

serialized_config: A serialized DynamicEmbeddingConfig proto.
var_name: A unique name for the given embedding.
kbs_address: The address of a dynamic embedding service. If empty, the value
             passed from --kbs_address flag will be used instead.
timeout_ms: timeout duration in miliseconds.
)doc");

DynamicEmbeddingManagerResource::DynamicEmbeddingManagerResource(
    const DynamicEmbeddingConfig& config, const std::string& var_name,
    const std::string& kbs_address, absl::Duration timeout)
    : ResourceBase(),
      manager_(DynamicEmbeddingManager::Create(config, var_name, kbs_address,
                                               timeout)) {}

class DynamicEmbeddingManagerResourceOp
    : public ResourceOpKernel<DynamicEmbeddingManagerResource> {
 public:
  explicit DynamicEmbeddingManagerResourceOp(OpKernelConstruction* context)
      : ResourceOpKernel<DynamicEmbeddingManagerResource>(context) {
    std::string serialized_config;
    OP_REQUIRES_OK(context,
                   context->GetAttr("serialized_config", &serialized_config));
    OP_REQUIRES(context, config_.ParseFromString(serialized_config),
                InvalidArgument("Cannot deserialize DynamicEmbeddingConfig "
                                "from serialized_config."));

    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES(context, !var_name_.empty(),
                InvalidArgument("var_name is empty."));
    OP_REQUIRES_OK(context, context->GetAttr("kbs_address", &kbs_address_));
    int timeout_ms;
    OP_REQUIRES_OK(context, context->GetAttr("timeout_ms", &timeout_ms));
    timeout_ = ms_to_duration(timeout_ms);
  }

 private:
  Status CreateResource(DynamicEmbeddingManagerResource** ret) override {
    *ret = new DynamicEmbeddingManagerResource(config_, var_name_, kbs_address_,
                                               timeout_);
    if ((*ret)->manager() == nullptr) {
      return Status(static_cast<tensorflow::errors::Code>(
                        absl::StatusCode::kFailedPrecondition),
                    "DynamicEmbeddingManager is NULL.");
    }
    return tensorflow::OkStatus();
  }

  DynamicEmbeddingConfig config_;
  std::string var_name_;
  std::string kbs_address_;
  absl::Duration timeout_;

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicEmbeddingManagerResourceOp);
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicEmbeddingManagerResource").Device(tensorflow::DEVICE_CPU),
    DynamicEmbeddingManagerResourceOp);

}  // namespace carls
