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

#ifndef _DVM_TORCH_FUSION_CORE_TENSOR_META_H_
#define _DVM_TORCH_FUSION_CORE_TENSOR_META_H_

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <dvm.h>

#include <cstdint>

namespace dvm_torch {

/** Three-state model. The tensor itself is always a real at::Tensor. */
enum class TensorState : uint8_t {
  kExternal,    ///< Not part of the graph.
  kPending,     ///< Storage allocated, data_ptr valid, contents not computed.
  kMaterialized ///< Contents computed.
};

/** Reasons for flushing the pending graph. */
enum class FlushReason : uint8_t {
  kNonFusibleOp = 0,
  kStreamSwitch,
  kHostRead,
  kWriteAlias,
  kTensorResize,
  kEventRecord,
  kEventWait,
  kMemoryPressure,
  kGraphTaskExit,
  kGraphTooLarge,
  kCaptureBegin,
  kProcessExit,
  kManual,
  kNumReasons,
};

const char *FlushReasonName(FlushReason reason);

struct TrimPolicy {
  size_t max_ops = 512;
  size_t pending_ops = 0;
  size_t check_interval = 64;

  bool ShouldTrim() const { return pending_ops >= max_ops; }
  void OnEmit() { ++pending_ops; }
  void Reset() { pending_ops = 0; }
};

dvm::DataType ToDvmDtype(at::ScalarType t);

}  // namespace dvm_torch

#endif  // _DVM_TORCH_FUSION_CORE_TENSOR_META_H_
