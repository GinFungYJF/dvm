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
#include "adapter/adapter.h"

#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace dvm_torch {
namespace {

class CudaAdapter final : public DeviceAdapter {
 public:
  bool IsDeviceTensor(const at::Tensor &t) const override {
    return t.defined() && t.is_cuda() && t.is_contiguous();
  }

  int64_t DeviceIndex(const at::Tensor &t) const override { return t.get_device(); }

  void *CurrentStream() const override {
    return static_cast<void *>(c10::cuda::getCurrentCUDAStream().stream());
  }

  at::Tensor Allocate(c10::IntArrayRef shape, const at::TensorOptions &opts) override {
    // A real tensor: metadata and data_ptr are valid, only the contents are
    // deferred. aten::empty takes no tensor argument, so the coverage guard does
    // not intercept it and the graph is not flushed by this allocation.
    return at::empty(shape, opts);
  }

  Caps Capabilities() const override { return Caps{}; }

  void *WsAlloc(size_t size) override {
    return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
        size, c10::cuda::getCurrentCUDAStream().stream());
  }

  void WsFree(void *ptr) override {
    if (ptr != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
  }

  void CheckLastError() override { C10_CUDA_CHECK(cudaGetLastError()); }

  bool CapturingNow() const override {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
      cudaGetLastError();  // clear, treat as not capturing
      return false;
    }
    return status != cudaStreamCaptureStatusNone;
  }
};

}  // namespace

DeviceAdapter &GetAdapter() {
  static CudaAdapter instance;
  return instance;
}

}  // namespace dvm_torch
