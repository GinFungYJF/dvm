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

#include "gpu_pointwise.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include "ops.h"

namespace dvm::gpu {
namespace {
uint32_t Numel(const IntArrayRef &shape) {
  uint32_t n = 1;
  for (size_t i = 0; i < shape.size; ++i) {
    auto dim = shape.data[i];
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() / n) return 0;
    n *= static_cast<uint32_t>(dim);
  }
  return n;
}

// Right-align dimensions exactly as tensor broadcasting does. Leading dimensions
// can be flattened together only when they are all dense or all broadcast.
bool GetIndex(const IntArrayRef &shape, const IntArrayRef &domain, Index &index) {
  if (shape.size > domain.size) return false;
  const size_t pad = domain.size - shape.size;
  bool dense_rows = true;
  bool one_row = true;
  bool dense_columns = false;
  bool exact = true;
  for (size_t i = 0; i < domain.size; ++i) {
    auto dim = i < pad ? 1 : shape.data[i - pad];
    if (dim != 1 && dim != domain.data[i]) return false;
    exact = exact && dim == domain.data[i];
    if (i + 1 == domain.size) {
      dense_columns = dim != 1;
    } else {
      dense_rows = dense_rows && dim == domain.data[i];
      one_row = one_row && dim == 1;
    }
  }
  if (!dense_rows && !one_row) return false;
  if (exact) {
    index = Index::Dense;
  } else if (one_row) {
    index = dense_columns ? Index::Column : Index::Scalar;
  } else {
    index = Index::Row;
  }
  return true;
}

struct Access {
  uintptr_t address;
  size_t bytes;
  Index index;
  int producer;  // -1 for a load
  bool Overlaps(const Access &other) const {
    return address <= other.address ? other.address - address < bytes : address - other.address < other.bytes;
  }
};

class PointwiseBuilder {
 public:
  explicit PointwiseBuilder(PointwiseProgram &program) : program_(program) {}

  bool Build(const std::vector<NDObject *> &objects) {
    // Each binary instruction can introduce two loads, two broadcasts and one store.
    if (objects.size() > 6 * kPointwiseMaxOps) return false;
    for (auto op : objects) {
      if (op->type_id_ != kFloat32 || !op->shape_ref_) return false;
      if (op->obj_id_ == kStore) {
        auto size = Numel(*op->shape_ref_);
        if (!size) return false;
        if (!domain_ || size > program_.elements ||
            (size == program_.elements && op->shape_ref_->size > domain_->size)) {
          domain_ = op->shape_ref_;
          program_.elements = size;
        }
      }
    }
    if (!domain_) return false;
    program_.width = domain_->size ? domain_->data[domain_->size - 1] : 1;
    if (program_.width % kPointwiseLanes != 0) return false;
    for (auto op : objects) {
      Index index;
      if (!GetIndex(*op->shape_ref_, *domain_, index)) return false;
      switch (op->obj_id_) {
        case kLoad:
          if (!AddAccess(op, index, -1)) return false;
          break;
        case kStore: {
          int producer = Find(op->lhs_);
          if (producer < 0 || program_.ops[producer].output || !AddAccess(op, index, producer)) return false;
          program_.ops[producer].output = static_cast<float *>(static_cast<NDAccess *>(op)->addr_.gm);
          program_.ops[producer].output_index = index;
          break;
        }
        case kBroadcastTo:
          // The binary normalizer inserts these right-aligned broadcasts.
          if (op->CheckFlag(OBJ_FLAG_EAGER) || !GetIndex(*op->lhs_->shape_ref_, *op->shape_ref_, index)) return false;
          break;
        case kBinary:
        case kBinaryS:
          if (!AddInstruction(op)) return false;
          break;
        default:
          return false;
      }
    }
    return program_.count >= 2 && AliasesSafe();
  }

 private:
  int Find(const NDObject *op) const {
    for (int i = program_.count - 1; i >= 0; --i) {
      if (results_[i] == op) return i;
    }
    return -1;
  }

  bool AddAccess(NDObject *op, Index index, int producer) {
    const auto access = static_cast<NDAccess *>(op);
    const auto address = reinterpret_cast<uintptr_t>(access->addr_.gm);
    const bool packed = index == Index::Dense || index == Index::Column;
    const auto alignment = packed ? kPointwiseLanes * sizeof(float) : alignof(float);
    if (op->flags_ != OBJ_FLAG_EAGER || access->stride_ || access->offset_bytes_ || !address ||
        address % alignment || access_count_ == std::size(accesses_)) return false;
    accesses_[access_count_++] = {address, sizeof(float) * Numel(*op->shape_ref_), index, producer};
    return true;
  }

  bool AliasesSafe() const {
    for (size_t i = 0; i < access_count_; ++i) {
      const auto &out = accesses_[i];
      if (out.producer < 0) continue;
      for (size_t j = 0; j < access_count_; ++j) {
        const auto &other = accesses_[j];
        if (i == j || !out.Overlaps(other)) continue;
        // Only the final instruction can update a dense input in place. Every
        // thread reads its own old element before writing that same element.
        if (other.producer >= 0 || j > i || out.producer != program_.count - 1 ||
            out.index != Index::Dense || other.index != Index::Dense || out.address != other.address ||
            out.bytes != other.bytes) return false;
      }
    }
    return true;
  }

  bool Operand(NDObject *op, PointwiseOperand &operand) {
    while (op->obj_id_ == kBroadcastTo) op = op->lhs_;
    if (op->obj_id_ == kLoad) {
      Index index;
      if (!GetIndex(*op->shape_ref_, *domain_, index)) return false;
      switch (index) {
        case Index::Dense: operand.source = Source::Dense; break;
        case Index::Row: operand.source = Source::Row; break;
        case Index::Column: operand.source = Source::Column; break;
        case Index::Scalar: operand.source = Source::Scalar; break;
      }
      operand.address = static_cast<const float *>(static_cast<NDAccess *>(op)->addr_.gm);
      return true;
    }
    int producer = Find(op);
    if (producer < 0) return false;
    if (producer == program_.count - 1) {
      operand.source = Source::Previous;
    } else {
      auto &slot = program_.ops[producer].save_slot;
      if (slot < 0) {
        if (program_.saved == kPointwiseSaved) return false;
        slot = program_.saved++;
      }
      operand.source = Source::Saved;
      operand.slot = slot;
    }
    return true;
  }

  bool AddInstruction(NDObject *op) {
    if (program_.count == kPointwiseMaxOps) return false;
    auto &insn = program_.ops[program_.count];
    insn.save_slot = -1;
    if (!Operand(op->lhs_, insn.lhs)) return false;
    if (op->obj_id_ == kBinary) {
      if (!Operand(op->rhs_, insn.rhs)) return false;
      switch (static_cast<BinaryOp *>(op)->GetOpType()) {
        case kAdd: insn.op = Arithmetic::Add; break;
        case kSub: insn.op = Arithmetic::Sub; break;
        case kMul: insn.op = Arithmetic::Mul; break;
        case kDiv: insn.op = Arithmetic::Div; break;
        default: return false;
      }
    } else {
      auto scalar = static_cast<BinaryScalarOp *>(op);
      if (scalar->IsScalarRef()) return false;
      insn.rhs.source = Source::Immediate;
      auto bits = scalar->GetScalar();
      std::memcpy(&insn.rhs.literal, &bits, sizeof(float));
      switch (scalar->GetOpType()) {
        case kAdds: insn.op = Arithmetic::Add; break;
        case kMuls: insn.op = Arithmetic::Mul; break;
        case ksDiv: std::swap(insn.lhs, insn.rhs); [[fallthrough]];
        case kDivs: insn.op = Arithmetic::Div; break;
        default: return false;
      }
    }
    results_[program_.count++] = op;
    return true;
  }

  PointwiseProgram &program_;
  const IntArrayRef *domain_{nullptr};
  NDObject *results_[kPointwiseMaxOps];
  Access accesses_[3 * kPointwiseMaxOps];
  size_t access_count_{0};
};

}  // namespace

bool TryEmitPointwise(const std::vector<NDObject *> &objects, Code &code) {
  static const bool enabled = [] {
    const char *setting = std::getenv("DVM_GPU_POINTWISE");
    return setting == nullptr || std::strcmp(setting, "1") == 0;
  }();
  if (!enabled) return false;
  PointwiseProgram program{};
  if (!PointwiseBuilder(program).Build(objects)) return false;

  static_assert(Code::HeadSize() == PointwiseCode::kInstruction * sizeof(uint64_t));
  code.Alloc(PointwiseCode::kSize);
  code.data_size_ = PointwiseCode::kSize;
  code.target_ = Code::kTargetVec;
  code.threads_ = kPointwiseThreads;
  code.shared_bytes_ = 0;
  code.block_dim_ = CeilDiv<uint32_t>(program.elements, kPointwiseThreads * kPointwiseLanes);
  code.UpdateHead(PointwiseCode::kVectorEntry);
  auto words = reinterpret_cast<uint64_t *>(code.data_);
  words[PointwiseCode::kInstruction] = PointwiseCode::kHead;
  words[PointwiseCode::kEnd] = 0;
  std::memcpy(words + PointwiseCode::kProgram, &program, sizeof(program));
  return true;
}
}  // namespace dvm::gpu
