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

#include <string>

#include "research/carls/dynamic_embedding_config.pb.h"  // proto to pb
#include "research/carls/dynamic_embedding_manager.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace carls {
namespace {

using ::Eigen::VectorXf;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::ResourceBase;
using ::tensorflow::ResourceOpKernel;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::FailedPrecondition;
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

REGISTER_OP("DynamicEmbeddingLookup")
    .Input("keys: string")
    .Input("grad_placeholder: float")
    .Input("handle: resource")
    .Output("values: float")
    .Doc(R"doc(
An operation that returns the embedding of a given set of keys.

keys: A string Tensor of shape [batch_size] or [batch_size, max_sequence_length]
      where an empty string would be mapped to an all zero embedding.
grad_placeholder: A dummy Tensor so that the gradients can be passed in.
handle: a handle to DynamicEmbeddingManagerResource.
values: A Tensor of shape [batch_size, embedding_dimension] if the
        input Tensor is 1D, or a Tensor of shape
        [batch_size, max_sequence_length, embedding_dimension] if the
        input is 2D.
)doc");

REGISTER_OP("DynamicEmbeddingUpdate")
    .Input("keys: string")
    .Input("values: float")
    .Input("handle: resource")
    .Output("success: float")
    .Doc(R"doc(
An operation that updates the embeddings of a given set of keys in the dynamic
embedding service.

keys: A string `Tensor` of shape [batch] or [batch_size,
      max_sequence_length].
values: A `Tensor` of shape [batch_size, embedding_dimension] or
        [batch_size, max_sequence_length, embedding_dimension].
handle: a handle to DynamicEmbeddingManagerResource.
)doc");

// Uses a ResourceOpKernel to manage the DynamicEmbeddingManager object, which
// talks to a KnowledgeBankService.
class DynamicEmbeddingManagerResource : public ResourceBase {
 public:
  DynamicEmbeddingManagerResource(const DynamicEmbeddingConfig& config,
                                  const std::string& var_name,
                                  const std::string& kbs_address,
                                  absl::Duration timeout)
      : ResourceBase(),
        manager_(DynamicEmbeddingManager::Create(config, var_name,
                                                 kbs_address, timeout)) {}
  ~DynamicEmbeddingManagerResource() override = default;

  std::string DebugString() const override { return "DEM resource"; }

  DynamicEmbeddingManager* manager() { return manager_.get(); }

 private:
  std::unique_ptr<DynamicEmbeddingManager> manager_;
};

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
    return Status::OK();
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

class DynamicEmbeddingLookupOp : public OpKernel {
 public:
  explicit DynamicEmbeddingLookupOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~DynamicEmbeddingLookupOp() override = default;

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const Tensor& keys_batch = context->input(0);
    OP_REQUIRES(context, keys_batch.dims() == 1 || keys_batch.dims() == 2,
                InvalidArgument("keys dimension must be either 1 or 2."));
    const int batch_size = keys_batch.dim_size(0);

    Tensor* output_tensor = nullptr;
    const int embedding_dimension =
        resource->manager()->config().embedding_dimension();
    if (keys_batch.dims() == 1) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(
                         0, TensorShape({batch_size, embedding_dimension}),
                         &output_tensor));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(
                         0,
                         TensorShape({batch_size, keys_batch.dim_size(1),
                                      embedding_dimension}),
                         &output_tensor));
    }

    auto status =
        resource->manager()->Lookup(keys_batch, /*update=*/true, output_tensor);
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(std::string(status.message())));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicEmbeddingLookup").Device(tensorflow::DEVICE_CPU),
    DynamicEmbeddingLookupOp);

class DynamicEmbeddingUpdateOp : public OpKernel {
 public:
  explicit DynamicEmbeddingUpdateOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~DynamicEmbeddingUpdateOp() override = default;

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));
    const int embedding_dimension =
        resource->manager()->config().embedding_dimension();

    const Tensor& keys_batch = context->input(0);
    const Tensor& values_batch = context->input(1);
    OP_REQUIRES(context, values_batch.dims() == keys_batch.dims() + 1,
                InvalidArgument("values' dimension != keys' dimension + 1."));
    OP_REQUIRES(
        context,
        values_batch.dim_size(values_batch.dims() - 1) == embedding_dimension,
        InvalidArgument("values last dimension size must equal to "
                        "embedding_dimension in config."));
    for (int d = 0; d < keys_batch.dims(); ++d) {
      OP_REQUIRES(context, values_batch.dim_size(d) == keys_batch.dim_size(d),
                  InvalidArgument(
                      "Mismatch between the dimensions of keys and values."));
    }

    auto status = resource->manager()->UpdateValues(keys_batch, values_batch);
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(std::string(status.message())));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, values_batch.shape(),
                                                     &output_tensor));
    *output_tensor = values_batch;
    auto output_value = output_tensor->flat_inner_dims<float>();
    auto keys_value = keys_batch.flat<tensorflow::tstring>();
    for (int i = 0; i < keys_batch.NumElements(); ++i) {
      if (keys_value(i).empty()) {
        for (int j = 0; j < embedding_dimension; ++j) {
          output_value(i, j) = 0.0;
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicEmbeddingUpdate").Device(tensorflow::DEVICE_CPU),
    DynamicEmbeddingUpdateOp);

}  // namespace carls
