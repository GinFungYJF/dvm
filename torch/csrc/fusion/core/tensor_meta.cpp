/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fusion/core/tensor_meta.h"

namespace dvm_torch {

const char *FlushReasonName(FlushReason reason) {
  switch (reason) {
    case FlushReason::kNonFusibleOp: return "NonFusibleOp";
    case FlushReason::kStreamSwitch: return "StreamSwitch";
    case FlushReason::kHostRead: return "HostRead";
    case FlushReason::kWriteAlias: return "WriteAlias";
    case FlushReason::kTensorResize: return "TensorResize";
    case FlushReason::kEventRecord: return "EventRecord";
    case FlushReason::kEventWait: return "EventWait";
    case FlushReason::kMemoryPressure: return "MemoryPressure";
    case FlushReason::kGraphTaskExit: return "GraphTaskExit";
    case FlushReason::kGraphTooLarge: return "GraphTooLarge";
    case FlushReason::kCaptureBegin: return "CaptureBegin";
    case FlushReason::kProcessExit: return "ProcessExit";
    case FlushReason::kManual: return "Manual";
    default: return "Unknown";
  }
}

dvm::DataType ToDvmDtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return dvm::kFloat32;
    case at::kHalf: return dvm::kFloat16;
    case at::kBFloat16: return dvm::kBFloat16;
    case at::kInt: return dvm::kInt32;
    case at::kLong: return dvm::kInt64;
    case at::kBool: return dvm::kBool;
    default: return dvm::kDataTypeEnd;
  }
}

}  // namespace dvm_torch
