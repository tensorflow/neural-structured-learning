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

#include "research/carls/base/status_helper.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/abi.h"

#if defined(TF_HAS_STACKTRACE)
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#endif

namespace carls {
namespace internal {
namespace {

static absl::Status MakeStatus(absl::StatusCode code,
                               const std::string& message) {
  return absl::Status(code, message);
}

// Function to create a pretty stacktrace.
inline std::string CurrentStackTrace() {
#if defined(TF_HAS_STACKTRACE)
  std::stringstream ss("");
  ss << "*** Begin stack trace ***" << std::endl;

  // Get the mangled stack trace.
  int buffer_size = 128;
  void* trace[128];
  buffer_size = backtrace(trace, buffer_size);

  for (int i = 0; i < buffer_size; ++i) {
    const char* symbol = "";
    Dl_info info;
    if (dladdr(trace[i], &info)) {
      if (info.dli_sname != nullptr) {
        symbol = info.dli_sname;
      }
    }

    std::string demangled = tensorflow::port::MaybeAbiDemangle(symbol);
    if (demangled.length()) {
      ss << "\t" << demangled << std::endl;
    } else {
      ss << "\t" << symbol << std::endl;
    }
  }

  ss << "*** End stack trace ***" << std::endl;
  return ss.str();
#else
  return std::string();
#endif  // defined(TF_HAS_STACKTRACE)
}

// Log the error at the given severity, optionally with a stack trace.
// If log_severity is NUM_SEVERITIES, nothing is logged.
static void LogError(const absl::Status& status, const char* filename, int line,
                     int log_severity, bool should_log_stack_trace) {
  if (TF_PREDICT_TRUE(log_severity != tensorflow::NUM_SEVERITIES)) {
    std::string stack_trace;
    if (should_log_stack_trace) {
      stack_trace = absl::StrCat("\n", CurrentStackTrace());
    }
    switch (log_severity) {
      case tensorflow::INFO:
        LOG(INFO) << status << stack_trace;
        break;
      case tensorflow::WARNING:
        LOG(WARNING) << status << stack_trace;
        break;
      case tensorflow::ERROR:
        LOG(ERROR) << status << stack_trace;
        break;
      case tensorflow::FATAL:
        LOG(FATAL) << status << stack_trace;
        break;
      case tensorflow::NUM_SEVERITIES:
        break;
      default:
        LOG(FATAL) << "Unknown LOG severity " << log_severity;
    }
  }
}

// Make a Status with a code, error message and payload,
// and also send it to LOG(<log_severity>) using the given filename
// and line (unless should_log is false, or log_severity is
// NUM_SEVERITIES).  If should_log_stack_trace is true, the stack
// trace is included in the log message (ignored if should_log is
// false).
static absl::Status MakeError(const char* filename, int line,
                              absl::StatusCode code, const std::string& message,
                              bool should_log, int log_severity,
                              bool should_log_stack_trace) {
  if (TF_PREDICT_FALSE(code == absl::StatusCode::kOk)) {
    LOG(ERROR) << "Cannot create error with status OK";
    code = absl::StatusCode::kUnknown;
  }
  const absl::Status status = MakeStatus(code, message);
  if (TF_PREDICT_TRUE(should_log)) {
    LogError(status, filename, line, log_severity, should_log_stack_trace);
  }
  return status;
}

}  // namespace

// This method is written out-of-line rather than in the header to avoid
// generating a lot of inline code for error cases in all callers.
void MakeErrorStream::CheckNotDone() const { impl_->CheckNotDone(); }

MakeErrorStream::Impl::Impl(const char* file, int line, absl::StatusCode code,
                            MakeErrorStream* error_stream,
                            bool is_logged_by_default)
    : file_(file),
      line_(line),
      code_(code),
      is_done_(false),
      should_log_(is_logged_by_default),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {}

MakeErrorStream::Impl::Impl(const absl::Status& status,
                            PriorMessageHandling prior_message_handling,
                            const char* file, int line,
                            MakeErrorStream* error_stream)
    : file_(file),
      line_(line),
      // Make sure we show some error, even if the call is incorrect.
      code_(!status.ok() ? status.code() : absl::StatusCode::kUnknown),
      prior_message_handling_(prior_message_handling),
      prior_message_(status.message()),
      is_done_(false),
      // Error code type is not visible here, so we can't call
      // IsLoggedByDefault.
      should_log_(true),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {
  DCHECK(!status.ok()) << "Attempted to append/prepend error text to status OK";
}

MakeErrorStream::Impl::~Impl() {
  // Note: error messages refer to the public MakeErrorStream class.

  if (!is_done_) {
    LOG(ERROR) << "MakeErrorStream destructed without getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

absl::Status MakeErrorStream::Impl::GetStatus() {
  // Note: error messages refer to the public MakeErrorStream class.

  // Getting a Status object out more than once is not harmful, but
  // it doesn't match the expected pattern, where the stream is constructed
  // as a temporary, loaded with a message, and then casted to Status.
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream got Status more than once: " << file_ << ":"
               << line_ << " " << stream_.str();
  }

  is_done_ = true;

  const std::string& stream_str = stream_.str();
  const std::string str = prior_message_handling_ == kAppendToPriorMessage
                              ? absl::StrCat(prior_message_, stream_str)
                              : absl::StrCat(stream_str, prior_message_);
  if (TF_PREDICT_FALSE(str.empty())) {
    return MakeError(
        file_, line_, code_,
        absl::StrCat(str, "Error without message at ", file_, ":", line_),
        true /* should_log */, tensorflow::ERROR /* log_severity */,
        should_log_stack_trace_);
  } else {
    return MakeError(file_, line_, code_, str, should_log_, log_severity_,
                     should_log_stack_trace_);
  }
}

void MakeErrorStream::Impl::CheckNotDone() const {
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream shift called after getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

}  // namespace internal

absl::Status ToAbslStatus(const grpc::Status& status) {
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

absl::Status ToAbslStatus(const tensorflow::Status& status) {
  return absl::Status(static_cast<absl::StatusCode>(status.code()),
                      status.error_message());
}

grpc::Status ToGrpcStatus(const absl::Status& status) {
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      std::string(status.message()));
}

}  // namespace carls
