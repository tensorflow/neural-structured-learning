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

#include "research/carls/dynamic_embedding_manager.h"
#include "research/carls/kernels/dynamic_embedding_manager_resource.h"
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
using ::tensorflow::errors::FailedPrecondition;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;

}  // namespace

REGISTER_OP("DynamicGaussianMemoryLookup")
    .Input("input: float")
    .Input("mode: int32")
    .Input("handle: resource")
    .Output("mean: float")
    .Output("variance: float")
    .Output("distance: float")
    .Output("cluster_id: int32")
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      c->set_output(0, input_shape);
      c->set_output(1, input_shape);
      auto rank = c->Rank(input_shape);
      if (rank > 0) {
        std::vector<DimensionHandle> dims;
        dims.reserve(rank);
        for (int i = 0; i < rank - 1; ++i) {
          dims.push_back(c->Dim(input_shape, i));
        }
        c->set_output(2, c->MakeShape(dims));
      }
      return tensorflow::OkStatus();
    })
    .Doc(R"doc(
A Gaussian memory assumes the input pattern can be represented by a number of
Gaussian clusters.

input: a float Tensor with shape [d1, d2, ..., dn].
mode: an int corresponding to  MemoryLookupRequest::LookupMode defined in
      research/carls/knowledge_bank_service.proto
mean: a float Tensor with the same shape as input representing the closest
      Gaussian center to the input.
variance: a float Tensor with the same shape as input representing the variance
          of the closest Gaussian cluster.
distance: a float Tensor with shape [d1, d2, ..., dn-1] representing the
          distances of each input to its closest Gaussian center.
cluster_id: an int Tensor with the shape [d1, d2, ..., dn-1] representing the
            cluster ids.

Note that the memory data is only based on the last dimension of the input.
Hence if the input shape is [d1, d2, ..., dn], it is assumed to contain
d1*d2*...*dn-1 data points.

This function can be used in conjunction with a DynamicNormalization layer.
The distance between the input and the Gaussian cluster can be used for model
uncertainty inferece.
)doc");

class DynamicGaussianMemoryLookupOp : public OpKernel {
 public:
  explicit DynamicGaussianMemoryLookupOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~DynamicGaussianMemoryLookupOp() override = default;

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const Tensor& input_batch = context->input(0);
    const int32_t mode = context->input(1).scalar<int32_t>()();

    // Allocates output.
    Tensor* output_mean = nullptr;
    Tensor* output_variance = nullptr;
    Tensor* output_distance = nullptr;
    Tensor* output_cluster_id = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_batch.shape(),
                                                     &output_mean));
    OP_REQUIRES_OK(context, context->allocate_output(1, input_batch.shape(),
                                                     &output_variance));
    auto shape = input_batch.shape();
    shape.RemoveDim(shape.dims() - 1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, shape, &output_distance));
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, shape, &output_cluster_id));

    // Calls DynamicMemoryManager::LookupGaussianCluster.
    auto status = resource->manager()->LookupGaussianCluster(
        input_batch, mode, output_mean, output_variance, output_distance,
        output_cluster_id);
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(std::string(status.message())));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicGaussianMemoryLookup").Device(tensorflow::DEVICE_CPU),
    DynamicGaussianMemoryLookupOp);

}  // namespace carls
