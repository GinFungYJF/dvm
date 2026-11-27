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

#ifndef DVM_GPU_POINTWISE_H_
#define DVM_GPU_POINTWISE_H_

#include <cstdint>
#ifndef _VM_KERNEL_
#include <vector>
#endif
#include "isa.h"
#include "vm_gpu.h"

namespace dvm {
class Code;
class NDObject;
namespace gpu {
constexpr int kPointwiseMaxOps = 12;
constexpr int kPointwiseSaved = 3;
constexpr int kPointwiseMaxInputs = 3;
constexpr uint32_t kPointwiseThreads = 128;
constexpr uint32_t kPointwiseLanes = 4;
enum class Index : uint8_t { Dense, Row, Column, Scalar };
enum class Source : uint8_t { Dense, Row, Column, Scalar, Immediate, Previous, Saved };
enum class Arithmetic : uint8_t { Add, Sub, Mul, Div, Tanh, Fma };

struct PointwiseOperand {
  union {
    const float *address;
    float literal;
    uint32_t slot;
  };
  Source source;
};

struct PointwiseInstruction {
  PointwiseOperand inputs[kPointwiseMaxInputs];
  float *output;
  Arithmetic op;
  Index output_index;
  int8_t save_slot;
  bool negate_product;
};

struct PointwiseProgram {
  PointwiseInstruction ops[kPointwiseMaxOps];
  uint32_t elements, width;
  uint8_t count, saved;
};

// Inline vector bytecode: the normal VM header, one PC-relative instruction,
// a terminator, and its operand table. Each block runs one tile.
struct PointwiseCode {
  enum Word : uint32_t { kEntry = 1, kInstruction, kEnd, kProgram };
  static constexpr uint32_t kSize = kProgram * sizeof(uint64_t) + sizeof(PointwiseProgram);
  static constexpr uint64_t kHead = (uint64_t{F_POINTWISE} << V_HEAD_ID_OFFSET) |
    (uint64_t{kProgram - kInstruction} << V_HEAD_EXT_OFFSET) | (1ull << V_HEAD_SIZE_OFFSET) |
    (1ull << V_HEAD_SIMD_FLAG_OFFSET);
  static constexpr uint64_t kVectorEntry = (1ull << V_ENTRY_V_TILE_BODY_OFFSET) | V_ENTRY_TYPE_V |
    ((kSize / sizeof(uint64_t) - kInstruction) << V_ENTRY_CODE_SIZE_OFFSET);

  static bool Matches(const VmParam &param, uint32_t threads, uint32_t shared_bytes) {
    return param.code[kEntry] == kVectorEntry && param.code[kInstruction] == kHead && param.code[kEnd] == 0 &&
           threads == kPointwiseThreads && shared_bytes == 0;
  }
};
static_assert(sizeof(PointwiseProgram) % sizeof(uint64_t) == 0 && PointwiseCode::kSize <= sizeof(VmParam));

#ifndef _VM_KERNEL_
bool TryEmitPointwise(const std::vector<NDObject *> &objects, Code &code);
#endif
}  // namespace gpu
}  // namespace dvm
#endif  // DVM_GPU_POINTWISE_H_
