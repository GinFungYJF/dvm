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

#ifndef _DVM_VM_GPU_H_
#define _DVM_VM_GPU_H_

#include <cstdint>

namespace dvm::gpu {
#define GPU_VM_ACCESS_OPS(X) \
  X(F_LOAD, V_LOAD) \
  X(F_STORE, V_STORE) \
  X(F_LOAD_DUMMY, V_LOAD_DUMMY)

#define GPU_VM_SIMD_OPS(X) \
  X(F_NOP, V_NOP) \
  X(F_COPY, V_COPY) \
  X(F_BROADCAST_S, V_BROADCAST_S) \
  X(F_BROADCAST_S_B8, V_BROADCAST_S_B8) \
  X(F_BROADCAST_S_B16, V_BROADCAST_S_B16) \
  X(F_BROADCAST_X_B8, V_BROADCAST_X_B8) \
  X(F_BROADCAST_X_B16, V_BROADCAST_X_B16) \
  X(F_BROADCAST_X_B32, V_BROADCAST_X_B32) \
  X(F_BROADCAST_Y, V_BROADCAST_Y) \
  X(F_CAST_FP32_TO_FP16, V_CAST_FP32_TO_FP16) \
  X(F_CAST_FP32_TO_BF16, V_CAST_FP32_TO_BF16) \
  X(F_CAST_FP32_TO_INT32, V_CAST_FP32_TO_INT32) \
  X(F_CAST_FP16_TO_BOOL, V_CAST_FP16_TO_BOOL) \
  X(F_CAST_FP16_TO_FP32, V_CAST_FP16_TO_FP32) \
  X(F_CAST_FP16_TO_INT32, V_CAST_FP16_TO_INT32) \
  X(F_CAST_BF16_TO_FP32, V_CAST_BF16_TO_FP32) \
  X(F_CAST_BF16_TO_INT32, V_CAST_BF16_TO_INT32) \
  X(F_CAST_INT32_TO_FP16, V_CAST_INT32_TO_FP16) \
  X(F_CAST_INT32_TO_FP32, V_CAST_INT32_TO_FP32) \
  X(F_CAST_BOOL_TO_FP16, V_CAST_BOOL_TO_FP16) \
  X(F_SQRT, V_SQRT) \
  X(F_SQRT_FP16, V_SQRT_FP16) \
  X(F_ABS, V_ABS) \
  X(F_ABS_FP16, V_ABS_FP16) \
  X(F_ABS_INT32, V_ABS_INT32) \
  X(F_LOG, V_LOG) \
  X(F_LOG_FP16, V_LOG_FP16) \
  X(F_EXP, V_EXP) \
  X(F_EXP_FP16, V_EXP_FP16) \
  X(F_ROUND, V_ROUND) \
  X(F_FLOOR, V_FLOOR) \
  X(F_CEIL, V_CEIL) \
  X(F_TRUNC, V_TRUNC) \
  X(F_ISFINITE, V_ISFINITE) \
  X(F_ISFINITE_FP16, V_ISFINITE_FP16) \
  X(F_ISFINITE_BF16, V_ISFINITE_BF16) \
  X(F_LOGICAL_NOT_BOOL, V_LOGICAL_NOT_BOOL) \
  X(F_ADDS, V_ADDS) \
  X(F_ADDS_FP16, V_ADDS_FP16) \
  X(F_ADDS_BF16, V_ADDS_BF16) \
  X(F_ADDS_INT32, V_ADDS_INT32) \
  X(F_MULS, V_MULS) \
  X(F_MULS_FP16, V_MULS_FP16) \
  X(F_MULS_BF16, V_MULS_BF16) \
  X(F_MULS_INT32, V_MULS_INT32) \
  X(F_DIVS, V_DIVS) \
  X(F_DIVS_FP16, V_DIVS_FP16) \
  X(F_SDIV, V_SDIV) \
  X(F_SDIV_FP16, V_SDIV_FP16) \
  X(F_MAXS, V_MAXS) \
  X(F_MAXS_FP16, V_MAXS_FP16) \
  X(F_MAXS_BF16, V_MAXS_BF16) \
  X(F_MAXS_INT32, V_MAXS_INT32) \
  X(F_MINS, V_MINS) \
  X(F_MINS_FP16, V_MINS_FP16) \
  X(F_MINS_BF16, V_MINS_BF16) \
  X(F_MINS_INT32, V_MINS_INT32) \
  X(F_CMPS, V_CMPS) \
  X(F_CMPS_FP16, V_CMPS_FP16) \
  X(F_CMPS_BF16, V_CMPS_BF16) \
  X(F_CMPS_INT32, V_CMPS_INT32) \
  X(F_CMPS_BOOL, V_CMPS_BOOL) \
  X(F_ADD, V_ADD) \
  X(F_ADD_FP16, V_ADD_FP16) \
  X(F_ADD_BF16, V_ADD_BF16) \
  X(F_ADD_INT32, V_ADD_INT32) \
  X(F_SUB, V_SUB) \
  X(F_SUB_FP16, V_SUB_FP16) \
  X(F_SUB_BF16, V_SUB_BF16) \
  X(F_SUB_INT32, V_SUB_INT32) \
  X(F_MUL, V_MUL) \
  X(F_MUL_FP16, V_MUL_FP16) \
  X(F_MUL_BF16, V_MUL_BF16) \
  X(F_MUL_INT32, V_MUL_INT32) \
  X(F_DIV, V_DIV) \
  X(F_DIV_FP16, V_DIV_FP16) \
  X(F_MIN, V_MIN) \
  X(F_MIN_FP16, V_MIN_FP16) \
  X(F_MIN_BF16, V_MIN_BF16) \
  X(F_MIN_INT32, V_MIN_INT32) \
  X(F_MAX, V_MAX) \
  X(F_MAX_FP16, V_MAX_FP16) \
  X(F_MAX_BF16, V_MAX_BF16) \
  X(F_MAX_INT32, V_MAX_INT32) \
  X(F_CMP, V_CMP) \
  X(F_CMP_FP16, V_CMP_FP16) \
  X(F_CMP_BF16, V_CMP_BF16) \
  X(F_CMP_INT32, V_CMP_INT32) \
  X(F_CMP_BOOL, V_CMP_BOOL) \
  X(F_SEL, V_SEL) \
  X(F_SEL_FP16, V_SEL_FP16) \
  X(F_SEL_BF16, V_SEL_BF16) \
  X(F_SEL_INT32, V_SEL_INT32) \
  X(F_POW, V_POW) \
  X(F_LOGICAL_AND_BOOL, V_LOGICAL_AND_BOOL) \
  X(F_LOGICAL_OR_BOOL, V_LOGICAL_OR_BOOL)

enum GpuVmOpId : uint8_t {
#define GPU_VM_ENUM_ENTRY(gpu, host) gpu,
  GPU_VM_ACCESS_OPS(GPU_VM_ENUM_ENTRY)
  GPU_VM_SIMD_OPS(GPU_VM_ENUM_ENTRY)
#undef F_VM_ENUM_ENTRY
  F_OP_COUNT
};

static_assert(F_OP_COUNT <= 0xff, "GPU VM opcode must fit in 8 bits");

struct VmParam {
  uint64_t code[4096 / sizeof(uint64_t)];
};

int InitVM(uint64_t local_mem_size);
int LaunchVM(const VmParam &param, uint32_t block_dim, void *stream);
}  // namespace dvm::gpu

#endif  // _DVM_VM_GPU_H_
