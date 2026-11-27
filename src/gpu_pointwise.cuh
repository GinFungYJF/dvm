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

#ifndef DVM_GPU_POINTWISE_CUH_
#define DVM_GPU_POINTWISE_CUH_

#include <cuda_runtime.h>
#include "gpu_pointwise.h"

namespace dvm::gpu {
namespace {
struct alignas(kPointwiseLanes * sizeof(float)) PointwiseValues {
  float v[kPointwiseLanes];
};

__device__ __forceinline__ PointwiseValues Splat(float scalar) {
  PointwiseValues result;
#pragma unroll
  for (uint32_t i = 0; i < kPointwiseLanes; ++i) result.v[i] = scalar;
  return result;
}

struct PointwiseIndex {
  uint32_t dense, row, column;
};

struct PointwiseTemporaries {
  PointwiseValues previous;
  PointwiseValues saved[kPointwiseSaved];

  __device__ __forceinline__ PointwiseValues Read(const PointwiseOperand &operand, PointwiseIndex at) const {
    switch (operand.source) {
      case Source::Dense: return *reinterpret_cast<const PointwiseValues *>(operand.address + at.dense);
      case Source::Column: return *reinterpret_cast<const PointwiseValues *>(operand.address + at.column);
      case Source::Row: return Splat(operand.address[at.row]);
      case Source::Scalar: return Splat(*operand.address);
      case Source::Immediate: return Splat(operand.literal);
      case Source::Previous: return previous;
      case Source::Saved:
        // Constant indices let the compiler keep the temporaries in registers.
#pragma unroll
        for (uint32_t i = 0; i < kPointwiseSaved; ++i) {
          if (operand.slot == i) return saved[i];
        }
    }
    __builtin_unreachable();
  }

  __device__ __forceinline__ void Save(int slot) {
#pragma unroll
    for (int i = 0; i < kPointwiseSaved; ++i) {
      if (i == slot) saved[i] = previous;
    }
  }
};

__device__ __forceinline__ void WritePointwise(const PointwiseInstruction &insn, PointwiseIndex at,
                                               PointwiseValues value) {
  if (!insn.output) return;
  switch (insn.output_index) {
    case Index::Dense:
      *reinterpret_cast<PointwiseValues *>(insn.output + at.dense) = value;
      break;
    case Index::Column:
      if (at.row == 0) *reinterpret_cast<PointwiseValues *>(insn.output + at.column) = value;
      break;
    case Index::Row:
      if (at.column == 0) insn.output[at.row] = value.v[0];
      break;
    case Index::Scalar:
      if (at.dense == 0) insn.output[0] = value.v[0];
      break;
  }
}

__device__ __forceinline__ PointwiseValues Evaluate(const PointwiseInstruction &insn,
                                                    const PointwiseTemporaries &values, PointwiseIndex at) {
  auto a = values.Read(insn.inputs[0], at);
  if (insn.op == Arithmetic::Tanh) {
#pragma unroll
    for (auto &lane : a.v) lane = tanhf(lane);
    return a;
  }
  auto b = values.Read(insn.inputs[1], at);
  if (insn.op == Arithmetic::Fma) {
    auto c = values.Read(insn.inputs[2], at);
#pragma unroll
    for (uint32_t i = 0; i < kPointwiseLanes; ++i) {
      a.v[i] = __fmaf_rn(insn.negate_product ? -a.v[i] : a.v[i], b.v[i], c.v[i]);
    }
    return a;
  }
#pragma unroll
  for (uint32_t i = 0; i < kPointwiseLanes; ++i) {
    switch (insn.op) {
      case Arithmetic::Add: a.v[i] = __fadd_rn(a.v[i], b.v[i]); break;
      case Arithmetic::Sub: a.v[i] = __fsub_rn(a.v[i], b.v[i]); break;
      case Arithmetic::Mul: a.v[i] = __fmul_rn(a.v[i], b.v[i]); break;
      case Arithmetic::Div: a.v[i] = __fdiv_rn(a.v[i], b.v[i]); break;
      default: __builtin_unreachable();
    }
  }
  return a;
}

__device__ __forceinline__ void RunPointwise(const PointwiseProgram &program, uint32_t group) {
  uint32_t dense = group * kPointwiseLanes;
  if (dense >= program.elements) return;
  uint32_t row = dense / program.width;
  PointwiseIndex at{dense, row, dense - row * program.width};
  PointwiseTemporaries values{};
  for (int pc = 0; pc < program.count; ++pc) {
    const auto &insn = program.ops[pc];
    values.previous = Evaluate(insn, values, at);
    values.Save(insn.save_slot);
    WritePointwise(insn, at, values.previous);
  }
}
}  // namespace
}  // namespace dvm::gpu
#endif  // DVM_GPU_POINTWISE_CUH_
