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

#include <vector>

#include "research/carls/dynamic_embedding_config.pb.h"  // proto to pb
#include "research/carls/dynamic_embedding_manager.h"
#include "research/carls/kernels/dynamic_embedding_manager_resource.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace carls {
namespace {

using ::tensorflow::int64;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::FailedPrecondition;
using ::tensorflow::errors::Internal;

}  // namespace

REGISTER_OP("TopkLookup")
    .Input("inputs: float")
    .Input("k: int32")
    .Input("handle: resource")
    .Output("keys: string")
    .Output("logits: float")
    .Doc(R"doc(
An operation that returns the logits of sampled keys from given input.

It looks up the top-k closest embeddings `w` to the given input `x`, and
computes the inner product `<w, x>`  as the logits.

This is usually used in the target layer of a neural network, e.g., a softmax
output. It can handle unseen labels and dynamically allocates new embeddings
for them during training.

inputs: A float `Tensor` of shape `[batch_size, dim]` representing the
        forward activations of the input network.
k: An `int` denoting the number of returned keys.
handle: A handle to DynamicEmbeddingManagerResource.
keys: the top k closest keys.
logits: the computed logits for the top k cloeset keys.
)doc");

class TopkLookupOp : public OpKernel {
 public:
  explicit TopkLookupOp(OpKernelConstruction* context) : OpKernel(context) {}

  ~TopkLookupOp() override = default;

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));
    const Tensor& input = context->input(0);
    const int k = context->input(1).scalar<int>()();

    std::vector<int64> dims;
    dims.reserve(input.dims() - 1);
    for (int d = 0; d < input.dims() - 1; ++d) {
      dims.push_back(input.dim_size(d));
    }
    dims.push_back(k);

    Tensor* output_keys = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(dims), &output_keys));

    Tensor* output_logits = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(dims),
                                                     &output_logits));

    auto status =
        resource->manager()->TopK(input, k, output_keys, output_logits);
    OP_REQUIRES(context, status.ok(),
                Internal(absl::StrCat(
                    "DynamicEmbeddingManager::TopK failed with error: ",
                    status.message())));
  }
};

REGISTER_KERNEL_BUILDER(Name("TopkLookup").Device(tensorflow::DEVICE_CPU),
                        TopkLookupOp);

}  // namespace carls
