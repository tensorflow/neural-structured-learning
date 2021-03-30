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

#include <string>

#include "absl/strings/str_format.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/constants.h"
#include "research/carls/dynamic_embedding_config.pb.h"  // proto to pb
#include "research/carls/dynamic_embedding_manager.h"
#include "research/carls/kernels/dynamic_embedding_manager_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace carls {
namespace {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::errors::FailedPrecondition;

// Generates a dir name based on current time YYYYMMDDHHMMSSMsMsMs.
std::string TimestampedDirname() {
  absl::Time t = absl::Now();
  absl::TimeZone tz = absl::LocalTimeZone();
  auto e = tz.At(t);
  return absl::StrFormat("%4lld%02d%02d_%02d%02d_%02d%03lld", e.cs.year(),
                         e.cs.month(), e.cs.day(), e.cs.hour(), e.cs.minute(),
                         e.cs.second(), e.subsecond / absl::Milliseconds(1));
}

}  // namespace

REGISTER_OP("SaveKnowledgeBank")
    .Input("output_directory: string")
    .Input("append_timestamp: bool")
    .Input("handle: resource")
    .Output("updated_checkpoint: string")
    .Doc(R"doc(
An operation that saves the current state of knowledge bank.

output_directory: A string representing the output directory for saved
                  checkpoint.
append_timestamp: A boolean indicating if a timestamped subdirectory should be
                  created or not.
handle: A handle to DynamicEmbeddingManagerResource.
updated_checkpoint: A string representing the path to the saved checkpoint.
)doc");

REGISTER_OP("RestoreKnowledgeBank")
    .Input("saved_path: string")
    .Input("handle: resource")
    .Doc(R"doc(
An operation that restores the saved state of knowledge bank.

saved_path: A string representing the path to the saved checkpoint.
handle: A handle to DynamicEmbeddingManagerResource.
)doc");

class SaveKnowledgeBankOp : public OpKernel {
 public:
  explicit SaveKnowledgeBankOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 2),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const std::string& output_dir = context->input(0).scalar<tstring>()();
    const bool append_timestamp = context->input(1).scalar<bool>()();

    // Creates a timestamped subdir under output_dir for this checkpoint.
    const std::string output_directory = JoinPath(
        output_dir, append_timestamp ? absl::StrCat(kExportDataSubDir, "_",
                                                    TimestampedDirname())
                                     : kExportDataSubDir);
    auto status = RecursivelyCreateDir(output_directory);
    OP_REQUIRES(
        context, status.ok(),
        FailedPrecondition(absl::StrCat(
            "Creating directory failed with error: ", status.message())));

    std::string exported_path;
    status = resource->manager()->Export(output_directory, &exported_path);
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(absl::StrCat(
                    "Calling Export() failed with error: ", status.message())));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({1}), &output_tensor));
    output_tensor->scalar<tstring>()() = exported_path;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SaveKnowledgeBank").Device(tensorflow::DEVICE_CPU),
    SaveKnowledgeBankOp);

class RestoreKnowledgeBankOp : public OpKernel {
 public:
  explicit RestoreKnowledgeBankOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DynamicEmbeddingManagerResource* resource = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1),
                                           &resource));
    OP_REQUIRES(context, resource->manager() != nullptr,
                FailedPrecondition("Creating DynamicEmbeddingManager failed."));

    const Tensor& saved_path = context->input(0);

    auto status = resource->manager()->Import(saved_path.scalar<tstring>()());
    OP_REQUIRES(context, status.ok(),
                FailedPrecondition(absl::StrCat(
                    "Calling Export() failed with error: ", status.message())));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("RestoreKnowledgeBank").Device(tensorflow::DEVICE_CPU),
    RestoreKnowledgeBankOp);

}  // namespace carls
