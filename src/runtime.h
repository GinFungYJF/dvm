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

#ifndef _DVM_RUNTIME_H_
#define _DVM_RUNTIME_H_

#include <cstddef>
#include <cstdint>

#ifdef DVM_ARCH_GPU

#include <cuda_runtime_api.h>

using aclrtDrvMemHandle = void *;

namespace dvm::rt {

using Stream = void *;
using Event = void *;

inline int GetDeviceCount(uint32_t *count) {
  int device_count = 0;
  auto err = cudaGetDeviceCount(&device_count);
  *count = static_cast<uint32_t>(device_count);
  return static_cast<int>(err);
}
inline int SetDevice(int device_id) { return static_cast<int>(cudaSetDevice(device_id)); }
inline int GetDevice(int *device_id) { return static_cast<int>(cudaGetDevice(device_id)); }

inline int CreateStream(Stream *stream) {
  return static_cast<int>(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(stream)));
}
inline int DestroyStream(Stream stream) {
  return static_cast<int>(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}
inline int SynchronizeStream(Stream stream) {
  auto err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    // Consume the sticky launch error so later cleanup (cudaFree) can succeed;
    // the call still reports the original error to the caller.
    (void)cudaGetLastError();
  }
  return static_cast<int>(err);
}

inline int Malloc(void **ptr, size_t size) { return static_cast<int>(cudaMalloc(ptr, size)); }
inline int Free(void *ptr) { return static_cast<int>(cudaFree(ptr)); }
inline int MallocHost(void **ptr, size_t size) { return static_cast<int>(cudaMallocHost(ptr, size)); }
inline int FreeHost(void *ptr) { return static_cast<int>(cudaFreeHost(ptr)); }
inline int MemToDev(void *dst, const void *src, size_t count) {
  return static_cast<int>(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}
inline int MemFromDev(void *dst, const void *src, size_t count) {
  return static_cast<int>(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}
inline int Memset(void *dst, int value, size_t size) { return static_cast<int>(cudaMemset(dst, value, size)); }

inline int CreateEvent(Event *event) {
  return static_cast<int>(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(event)));
}
inline int DestroyEvent(Event event) {
  return static_cast<int>(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
}
inline int RecordEvent(Event event, Stream stream) {
  return static_cast<int>(
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)));
}
inline int EventElapsedTime(float *ms, Event start, Event end) {
  return static_cast<int>(
    cudaEventElapsedTime(ms, reinterpret_cast<cudaEvent_t>(start), reinterpret_cast<cudaEvent_t>(end)));
}

}  // namespace dvm::rt

#else

#include "acl/acl_rt.h"

namespace dvm::rt {

using Stream = aclrtStream;
using Event = aclrtEvent;

inline int GetDeviceCount(uint32_t *count) { return aclrtGetDeviceCount(count); }
inline int SetDevice(int device_id) { return aclrtSetDevice(device_id); }
inline int GetDevice(int *device_id) { return aclrtGetDevice(device_id); }

inline int CreateStream(Stream *stream) {
  return aclrtCreateStreamWithConfig(stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
}
inline int DestroyStream(Stream stream) { return aclrtDestroyStream(stream); }
inline int SynchronizeStream(Stream stream) { return aclrtSynchronizeStream(stream); }

inline int Malloc(void **ptr, size_t size) { return aclrtMalloc(ptr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH); }
inline int Free(void *ptr) { return aclrtFree(ptr); }
inline int MallocHost(void **ptr, size_t size) { return aclrtMallocHost(ptr, size); }
inline int FreeHost(void *ptr) { return aclrtFreeHost(ptr); }
inline int MemToDev(void *dst, const void *src, size_t count) {
  return aclrtMemcpy(dst, count, src, count, ACL_MEMCPY_HOST_TO_DEVICE);
}
inline int MemFromDev(void *dst, const void *src, size_t count) {
  return aclrtMemcpy(dst, count, src, count, ACL_MEMCPY_DEVICE_TO_HOST);
}
inline int Memset(void *dst, int value, size_t size) { return aclrtMemset(dst, size, value, size); }

inline int CreateEvent(Event *event) { return aclrtCreateEventExWithFlag(event, ACL_EVENT_TIME_LINE); }
inline int DestroyEvent(Event event) { return aclrtDestroyEvent(event); }
inline int RecordEvent(Event event, Stream stream) { return aclrtRecordEvent(event, stream); }
inline int EventElapsedTime(float *ms, Event start, Event end) {
  return aclrtEventElapsedTime(ms, start, end);
}

}  // namespace dvm::rt

#endif  // DVM_ARCH_GPU

#endif  // _DVM_RUNTIME_H_
