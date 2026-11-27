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

#ifndef _DVM_TORCH_ADAPTER_ADAPTER_H_
#define _DVM_TORCH_ADAPTER_ADAPTER_H_

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <cstdint>

namespace dvm_torch {

/** Backend capability set. Replaces torch_npu's GetSocVersion()/GetTaskQueueEnable(). */
struct Caps {
  bool native_bf16 = true;
  bool has_async_queue = false;
  int vector_align = 32;
  int prefetch_depth = 0;
};

/**
 * @brief Per-backend execution hooks. Only CUDA is implemented here; the Ascend
 * adapter stays in torch_npu because it needs private NPU symbols.
 */
class DeviceAdapter {
 public:
  virtual ~DeviceAdapter() = default;

  virtual bool IsDeviceTensor(const at::Tensor &t) const = 0;
  virtual int64_t DeviceIndex(const at::Tensor &t) const = 0;
  virtual void *CurrentStream() const = 0;
  virtual at::Tensor Allocate(c10::IntArrayRef shape, const at::TensorOptions &opts) = 0;
  virtual Caps Capabilities() const = 0;
  virtual void *WsAlloc(size_t size) = 0;
  virtual void WsFree(void *ptr) = 0;
  virtual void CheckLastError() = 0;
  /** True while a CUDA graph capture is in flight on the current stream. */
  virtual bool CapturingNow() const = 0;
};

DeviceAdapter &GetAdapter();

}  // namespace dvm_torch

#endif  // _DVM_TORCH_ADAPTER_ADAPTER_H_
