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

REGISTER_OP("SampledLogitsLookup")
    .Input("positive_keys: string")
    .Input("inputs: float")
    .Input("num_samples: int32")
    .Input("grad_placeholder: float")
    .Input("handle: resource")
    .Output("keys: string")
    .Output("labels: float")
    .Output("expected_counts: float")
    .Output("mask: float")
    .Output("embedding: float")
    .Doc(R"doc(
An operation that samples negative keys from given positive keys.

This is usually used in the logit layer of a neural network, e.g., a softmax
output. It can handle unseen labels and dynamically allocates new embeddings
for them during training.

positive_keys: A string Tensor of shape [d1, ..., dN, max_length]
               where the last dimension holds positive keys with various
               lengths.
inputs: A float Tensor of shape [d1, ..., dN, embed_dim], where
        where `embed_dim` must be the same as set in DynamicEmbeddingConfig.
num_samples: An int indicating the number of returned samples.
grad_placeholder: A dummy Tensor so that the gradients can be passed in.
handle: A handle to DynamicEmbeddingManagerResource.
keys: A float Tensor of shape [d1, ..., dN, num_samples] holding both
      given positive and sampled negative samples.
labels: A float Tensor of shape [d1, d2, ..., dn-1, num_samples] indicating if
        a sample is positive (1.0) or negative (0.0).
expected_counts: A float Tensor of shape [d1, ..., dN, num_samples]
                 indicating the probability/expected-count of each sample.
mask: A float Tensor of shape [d1, ..., dN] indicating if a given entry is
      valid (1.0) or not (0.0). For example, if all keys in positive_keys[i]
      are empty, mask[i] = 0; otherwise mask[i] = 1.
embedding: A float Tensor of shape [d1, ..., dN, num_samples, embed_dim]
           returning the embedding/weights of each positive/negative keys.
           This is used in the gradient op for gradient update.
)doc");

REGISTER_OP("SampledLogitsLookupGrad")
    .Input("keys: string")
    .Input("weight_gradients: float")
    .Input("handle: resource")
    .Output("keys_grad: float")
    .Output("num_samples_grad: float")
    .Output("dummy_variable_gradients: float")
    .Output("resource_grad: float")
    .Doc(R"doc(
An operation that returns the fake gradients for the dummy variable and string
input.

keys: sampled keys returned from SampledLogitsLookup.
weight_gradients: weights for the embeddings of the sampled keys.
handle: A handle to DynamicEmbeddingManagerResource.
keys_grad: gradient for the keys.
num_samples_grad: gradient for the `num_samples` input.
dummy_variable_gradients: gradient for the grad_placeholder.
resource_grad: gradient for the resource input.
)doc");

class SampledLogitsLookupOp : public OpKernel {
 public:
  explicit SampledLogitsLookupOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~SampledLogitsLookupOp() override = default;

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 4),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));
    const Tensor& positive_keys = context->input(0);
    const Tensor& input = context->input(1);
    const int num_sampled = context->input(2).scalar<int>()();

    std::vector<int64> dims;
    dims.reserve(input.dims() - 1);
    for (int d = 0; d < input.dims() - 1; ++d) {
      dims.push_back(input.dim_size(d));
    }
    dims.push_back(num_sampled);

    Tensor* keys_output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(dims), &keys_output));

    Tensor* label_output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape(dims), &label_output));

    Tensor* expected_count_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape(dims),
                                                     &expected_count_output));

    Tensor* mask_output = nullptr;
    auto mask_dims = dims;
    mask_dims.pop_back();
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape(mask_dims),
                                                     &mask_output));

    dims.push_back(resource->manager()->config().embedding_dimension());
    Tensor* output_embed = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, TensorShape(dims), &output_embed));

    auto status = resource->manager()->NegativeSampling(
        positive_keys, input, num_sampled, /*update=*/true, keys_output,
        label_output, expected_count_output, mask_output, output_embed);
    OP_REQUIRES(context, status.ok(),
                Internal(absl::StrCat(
                    "DynamicEmbeddingManager::SampleLogits failed with error: ",
                    status.message())));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SampledLogitsLookup").Device(tensorflow::DEVICE_CPU),
    SampledLogitsLookupOp);

class SampledLogitsLookupGradOp : public OpKernel {
 public:
  explicit SampledLogitsLookupGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const Tensor& keys = context->input(0);
    const Tensor& grads = context->input(1);

    // Updates the gradients for the
    auto status = resource->manager()->UpdateGradients(keys, grads);
    OP_REQUIRES(context, status.ok(),
                Internal(absl::StrCat("UpdateGradients() failed with error: ",
                                      std::string(status.message()))));

    Tensor* output_tensor = nullptr;
    // Gradient for the keys input.
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({1}), &output_tensor));
    auto output = output_tensor->scalar<float>();
    output() = 0.0;

    // Gradient for the num_samples input.
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape({1}), &output_tensor));
    output_tensor->scalar<float>()() = 0;

    // Gradient for the placeholder variable, required by the TF framework.
    OP_REQUIRES_OK(
        context, context->allocate_output(2, TensorShape({1}), &output_tensor));
    output_tensor->scalar<float>()() = 0;

    // Gradient for the resource, required by the TF framework.
    OP_REQUIRES_OK(
        context, context->allocate_output(3, TensorShape({1}), &output_tensor));
    output_tensor->scalar<float>()() = 0;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SampledLogitsLookupGrad").Device(tensorflow::DEVICE_CPU),
    SampledLogitsLookupGradOp);

}  // namespace carls
