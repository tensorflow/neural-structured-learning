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
#include "research/carls/kernels/dynamic_embedding_manager_resource.h"
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
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::FailedPrecondition;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::DimensionOrConstant;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

// Sets the last dimension of the input to be embedding_dimension.
Status SetInputShape(InferenceContext* c) {
  int embedding_dimension;
  TF_RETURN_IF_ERROR(c->GetAttr("embedding_dimension", &embedding_dimension));
  DynamicEmbeddingConfig config;
  ShapeHandle input_shape = c->input(0);
  auto rank = c->Rank(input_shape);
  if (rank > 0) {
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      dims.push_back(c->Dim(input_shape, i));
    }
    dims.push_back(c->MakeDim(DimensionOrConstant(embedding_dimension)));
    c->set_output(0, c->MakeShape(dims));
  } else {
    // Otherwise (rank == 0), assume input shape is 1D.
    ShapeHandle output_shape =
        c->Matrix(InferenceContext::kUnknownDim,
                  DimensionOrConstant(embedding_dimension));
    c->set_output(0, output_shape);
  }
  return tensorflow::OkStatus();
}

}  // namespace

REGISTER_OP("DynamicEmbeddingLookup")
    .Input("keys: string")
    .Input("grad_placeholder: float")
    .Input("handle: resource")
    .Output("values: float")
    .Attr("embedding_dimension: int")
    .SetShapeFn([](InferenceContext* c) { return SetInputShape(c); })
    .Doc(R"doc(
An operation that returns the embedding of a given set of keys.

keys: A string Tensor of shape [batch_size] or [batch_size, max_sequence_length]
      where an empty string would be mapped to an all zero embedding.
grad_placeholder: A dummy Tensor so that the gradients can be passed in.
handle: A handle to DynamicEmbeddingManagerResource.
values: A Tensor of shape [batch_size, embedding_dimension] if the
        input Tensor is 1D, or a Tensor of shape
        [batch_size, max_sequence_length, embedding_dimension] if the
        input is 2D.
)doc");

REGISTER_OP("DynamicEmbeddingUpdate")
    .Input("keys: string")
    .Input("values: float")
    .Input("handle: resource")
    .Output("results: float")
    .Attr("embedding_dimension: int")
    .SetShapeFn([](InferenceContext* c) { return SetInputShape(c); })
    .Doc(R"doc(
An operation that updates the embeddings of a given set of keys in the dynamic
embedding service.

keys: A string `Tensor` of shape [batch] or [batch_size,
      max_sequence_length].
values: A `Tensor` of shape [batch_size, embedding_dimension] or
        [batch_size, max_sequence_length, embedding_dimension].
handle: A handle to DynamicEmbeddingManagerResource.
results: A `Tensor` of the same shape as `values` representing the updated
         embeddings of keys (same as `values` for non-empty keys and all-zero
         embeddings for empty keys).
)doc");

REGISTER_OP("DynamicEmbeddingLookupGrad")
    .Input("keys: string")
    .Input("gradients: float")
    .Input("handle: resource")
    .Output("keys_gradients: float")
    .Output("dummy_variable_gradients: float")
    .Output("resource_gradients: float")
    .Doc(R"doc(
An operation that updates the gradients of the given `keys` by calling the
knowledge bank service and returns the fake gradients for the
DynamicEmbeddingLookup op.

keys: A string `Tensor` of shape [batch] or [batch_size,
      max_sequence_length].
gradients: A `Tensor` of shape [batch_size, embedding_dimension] or
        [batch_size, max_sequence_length, embedding_dimension].
handle: A handle to DynamicEmbeddingManagerResource.
keys_gradients: A float `Tensor` representing the fake gradient of the keys
                input.
dummy_variable_gradients: A float `Tensor` representing the fake gradient of the
                          dummy variable input.
resource_gradients: A float `Tensor` representing the fake gradient of the
                    resource input.
)doc");

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

    // Checks input dimensions.
    // The rank of (keys, values) can either be (N, N + 1) or
    // (N, N) when the last dimension of the keys is 1.
    int keys_dim = keys_batch.dims();
    if (values_batch.dims() == keys_batch.dims() &&
        keys_batch.dim_size(keys_dim - 1) == 1) {
      --keys_dim;
    }
    OP_REQUIRES(
        context, values_batch.dims() == keys_dim + 1,
        InvalidArgument(absl::StrCat(
            "Incompatible input dimensions, got (dim(keys), dim(values)): (",
            keys_batch.dims(), ", ", values_batch.dims(), ").")));
    OP_REQUIRES(
        context,
        values_batch.dim_size(values_batch.dims() - 1) == embedding_dimension,
        InvalidArgument("values' last dimension size must equal to "
                        "embedding_dimension in config."));
    for (int d = 0; d < values_batch.dims() - 1; ++d) {
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

class DynamicEmbeddingLookupGradOp : public OpKernel {
 public:
  explicit DynamicEmbeddingLookupGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const Tensor& keys_batch = context->input(0);
    const Tensor& grad_input = context->input(1);

    auto status = resource->manager()->UpdateGradients(keys_batch, grad_input);
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(std::string(status.message())));

    // Gradient for the keys input.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({1}), &output_tensor));
    auto output = output_tensor->scalar<float>();
    output() = 0.0;

    // Gradient for the dummy input, required by the TF framework.
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape({1}), &output_tensor));
    output_tensor->scalar<float>()() = 0;

    // Gradient for the resource, required by the TF framework.
    OP_REQUIRES_OK(
        context, context->allocate_output(2, TensorShape({1}), &output_tensor));
    output_tensor->scalar<float>()() = 0;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicEmbeddingLookupGrad").Device(tensorflow::DEVICE_CPU),
    DynamicEmbeddingLookupGradOp);

}  // namespace carls
