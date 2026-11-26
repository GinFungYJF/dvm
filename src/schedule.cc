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

#include <algorithm>
#include "schedule.h"

namespace dvm {
void VectorSchedule::SpaceInit() {
  space_records_.clear();
  for (auto op : kernel_->objects_) {
    if (auto ndd = op->Ndd()) {
      space_records_.push_back({0, ndd});
    }
  }
}

void VectorSchedule::SpaceSplit(int dim, int64_t npart, int64_t nfactor) {
  ASSERT(npart > 0 && nfactor > 0);
  auto &space = kernel_->DimSpace();
  size_t space_size = space.size();
  ASSERT(dim >= 0 && static_cast<size_t>(dim) < space_size);
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      size_t n = dims.size();
      int64_t orig_dim = dims[dim];
      dims.resize(n + 1);
      for (size_t i = n; i > static_cast<size_t>(dim) + 1; --i) {
        dims[i] = dims[i - 1];
      }
      if (orig_dim != 1) {
        dims[dim] = nfactor;
        dims[dim + 1] = npart;
      } else {
        dims[dim] = 1;
        dims[dim + 1] = 1;
      }
    }
  }
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < space_size) stride.resize(space_size, 0);
    int64_t orig_stride = stride[dim];
    size_t n = stride.size();
    stride.resize(n + 1);
    for (size_t j = n; j > static_cast<size_t>(dim) + 1; --j) {
      stride[j] = stride[j - 1];
    }
    stride[dim] = orig_stride;
    stride[dim + 1] = orig_stride * nfactor;
  }
}

void VectorSchedule::SpaceTrans(int dim1, int dim2) {
  auto &space = kernel_->DimSpace();
  ASSERT(dim1 >= 0 && static_cast<size_t>(dim1) < space.size());
  ASSERT(dim2 >= 0 && static_cast<size_t>(dim2) < space.size());
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      std::swap(r.ndd->dims[dim1], r.ndd->dims[dim2]);
    }
  }
  size_t nd_size = space.size();
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < nd_size) stride.resize(nd_size, 0);
    std::swap(stride[dim1], stride[dim2]);
  }
}

void VectorSchedule::SaveSpace() {
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 1) {
          r.bcast_mask |= (1u << i);
        }
      }
    }
  }
}

class VectorDupHelper {
 public:
  VectorDupHelper(VectorKernel *kernel, int dup_num, RelocAddr *relocs, uint64_t total_quota)
      : kernel_(kernel), dup_num_(dup_num), dup_idx_(0), block_begin_(0), relocs_(relocs),
        remain_quota_(total_quota) {
    uint64_t dup = static_cast<uint64_t>(dup_num);
    // Reserve 1 core per segment up front (free_core_ = freely allocatable
    // cores); each segment gets its reserved core back via the +1 in DoAppend.
    uint64_t total_core = RoundUp(dup, g_system.CoreNum());
    free_core_ = total_core - dup;
    uint64_t code_reserve = dup_num * kernel_->ReserveCodeSize();
    encoder_.Reset(&kernel_->code_, Code::kTargetVec, dup_num, code_reserve, total_core);
  }

  void Reset() {
    int op_index = 0;
    for (auto op : kernel_->objects_) {
      op->Clear(op_index++);
    }
  }

  void DoAppend(uint64_t quota, uint64_t cap_core = 0) {
    ASSERT(dup_idx_ < dup_num_);
    ASSERT(remain_quota_ >= quota && quota > 0);
    dup_idx_++;
    kernel_->PrepareTiling();
    ASSERT(kernel_->tile_size_);
    auto code_begin = encoder_.ProgData();
    uint64_t code_reserve = kernel_->ReserveCodeSize();
    uint64_t core_limit = CeilDiv(quota * free_core_, remain_quota_) + 1;
    if (cap_core != 0 && core_limit > cap_core) {
      core_limit = cap_core;
    }
    remain_quota_ -= quota;
    auto code_end = kernel_->DoCodeGen(core_limit, code_begin, code_reserve);
    uint64_t code_size = code_end - code_begin;
    uint64_t block_dim = kernel_->CompactBlockDim(core_limit);
    ASSERT(kernel_->visit_ == nullptr);
    auto prog = encoder_.Append(Code::GenEntryV(kernel_->tile_num_, block_dim, code_size), code_size);
    encoder_.AssignAiv(block_begin_, block_dim, prog);
    block_begin_ += block_dim;
    for (auto op : kernel_->static_ops_) {
      if (!(op->flags_ & OBJ_FLAG_DEAD)) {
        auto &r = static_cast<NDAccess *>(op)->addr_;
        auto bind = relocs_++;
        bind->Update(r.reloc_);
        kernel_->code_.BindOpFast(*bind, r);
      }
    }
    free_core_ -= block_dim - 1;
  }

  void Append(uint64_t quota, uint64_t cap_core = 0) {
    Reset();
    DoAppend(quota, cap_core);
  }

  void Submit() {
    ASSERT(dup_idx_ == dup_num_);
    encoder_.Submit(block_begin_);
  }

 protected:
  VectorKernel *kernel_;
  PCodeEncoder encoder_;
  int dup_num_;
  int dup_idx_;
  uint64_t block_begin_;
  RelocAddr *relocs_;
  uint64_t free_core_;
  uint64_t remain_quota_;
};

SchGenHelper::~SchGenHelper() { delete []reloc_array_; }
int64_t SchGenHelper::CodeGen() { return -1; }

void SchGenHelper::AllocStride(NDAccess *acc) {
  if (ext_strides_.empty()) {
    ext_strides_.resize(kernel_->static_ops_.size());
  }
  auto &ext = ext_strides_[ext_stride_used_++];
  ext.acc = acc;
  acc->stride_ = &ext.stride;
  auto &dims = acc->nd_.dims();
  auto n = dims.size();
  ext.stride.resize(n);
  int64_t cur_stride = ITEM_SIZE[acc->type_id_];
  for (size_t i = 0; i < n; ++i) {
    ext.stride[i] = cur_stride;
    cur_stride *= dims[i];
  }
  if (acc->obj_id_ == kStore) {
    auto input = acc->lhs_;
    if (input->obj_id_ == kRemovePad) {
      acc->lhs_ = input->lhs_;
      ext.rm_pad = input;
    } else if (input->obj_id_ == kReduce && input->CheckFlag(OBJ_FLAG_REDUCE_RMPAD_EN)) {
      input->flags_ &= ~OBJ_FLAG_REDUCE_RMPAD_EN;
      ext.rm_pad = acc;
    }
  }
}

void SchGenHelper::ResetStrides() {
  for (size_t i = 0; i < ext_stride_used_; ++i) {
    auto &ext = ext_strides_[i];
    ext.acc->stride_ = nullptr;
    if (ext.rm_pad) {
      if (ext.acc->lhs_->obj_id_ == kReduce) {
        ext.acc->lhs_->SetFlag(OBJ_FLAG_REDUCE_RMPAD_EN);
      } else {
        ext.acc->lhs_ = ext.rm_pad;
      }
      ext.rm_pad = nullptr;
    }
  }
  ext_stride_used_ = 0;
}

FractalSchGen::FractalSchGen(VectorKernel *kernel) : SchGenHelper(kernel) {
  for (auto op : kernel->static_ops_) {
    EXCEPTION_IF(!static_cast<NDAccess *>(op)->IsSupportView(), "unsupport view op");
  }
}

int64_t FractalSchGen::CodeGen() {
  constexpr int64_t min_dim_limit = 6;
  int h_idx = -1;
  int64_t item_size = 0;
  uint64_t fractal_mask = 0;
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto load = static_cast<NDAccess *>(kernel_->static_ops_[i]);
    if (load->stride_ == nullptr) {
      continue;
    }
    auto &stride = *load->stride_;
    auto &dims = load->Ndd()->dims;
    int64_t type_size = ITEM_SIZE[load->type_id_];
    if (stride[0] != type_size && dims[0] >= min_dim_limit) {
      if (h_idx == -1) {
        if (type_size != 2 && type_size != 4) {
          continue;
        }
        for (size_t j = 1; j < stride.size(); ++j) {
          if (stride[j] == type_size) {
            if (dims[j] >= min_dim_limit) {
              h_idx = static_cast<int>(j);
              item_size = type_size;
              load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
              fractal_mask |= 1ull << i;
            }
            break;
          }
        }
      } else if (static_cast<size_t>(h_idx) < stride.size() && type_size == item_size && stride[h_idx] == type_size &&
                 dims[h_idx] >= min_dim_limit) {
        load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
        fractal_mask |= 1ull << i;
      }
    }
  }
  if (h_idx < 0) {
    return -1;
  }

  class FractalDunGen {
   public:
    FractalDunGen(SchGenHelper &gen, int h_idx, uint64_t item_size)
        : gen_(gen), space_(gen.kernel_->DimSpace()), item_size_(item_size), h_idx_(h_idx) {
      w_fractal_ = g_system.Arch() == AiCoreArch::kAiCore_C310 ? 128 : 16;
      h_fractal_ = item_size == 2 ? 16 : 8;
      int64_t w_size = space_[0];
      w_body_ = w_size / w_fractal_;
      w_tail_ = w_size - w_body_ * w_fractal_;
      int64_t h_size = space_[h_idx];
      h_body_ = h_size / h_fractal_;
      h_tail_ = h_size - h_body_ * h_fractal_;
    }

    void CodeGen(uint64_t fractal_mask) {
      const int dup_num = (1 + (w_body_ && w_tail_)) * (1 + (h_body_ && h_tail_));
      const int64_t w_npart = w_body_ + (w_tail_ ? 1 : 0);
      const int64_t h_npart = h_body_ + (h_tail_ ? 1 : 0);
      gen_.SpaceInit();
      gen_.SpaceSplit(0, w_npart, w_fractal_);
      h_idx_ += 1;
      gen_.SpaceTrans(1, h_idx_);
      gen_.SpaceSplit(1, h_npart, h_fractal_);
      gen_.SaveSpace();
      part_base_ = 1;
      size_t w_part_dim = h_idx_ + 1;
      size_.resize(space_.size());
      size_[2] = space_[2];
      for (size_t i = 3; i < space_.size(); ++i) {
        size_[i] = space_[i];
        if (i != w_part_dim) {
          part_base_ *= space_[i];
        }
      }
      // Weight body/tail programs by their estimated element work.
      const int64_t quota_w_tail = w_tail_ ? std::max<int64_t>(w_tail_, 16) : 0;
      const uint64_t total_element_quota =
        (h_body_ * h_fractal_ + h_tail_) * (w_body_ * w_fractal_ + quota_w_tail);
      VectorDupHelper helper(gen_.kernel_, dup_num, gen_.ReserveReloc(gen_.kernel_->static_ops_.size() * 4),
                             total_element_quota);
      if (w_tail_ > 1) {
        if (h_body_) {
          GenDup(helper, 0, w_body_, w_tail_, h_fractal_, h_body_, 1);
        }
        if (h_tail_) {
          GenDup(helper, h_body_, w_body_, w_tail_, h_tail_, 1, 1);
        }
      } else if (w_tail_ == 1) { // lead dim is folded
        ClearFractal(fractal_mask, gen_.kernel_->static_ops_);
        if (h_body_) {
          GenDup(helper, 0, w_body_, w_tail_, h_fractal_, h_body_, 1);
        }
        if (h_tail_) {
          GenDup(helper, h_body_, w_body_, w_tail_, h_tail_, 1, 1);
        }
        SetFractal(fractal_mask, gen_.kernel_->static_ops_);
      }
      if (h_tail_ && w_body_) {
        GenDup(helper, h_body_, 0, w_fractal_, h_tail_, 1, w_body_);
      }
      if (w_body_ && h_body_) {
        GenDup(helper, 0, 0, w_fractal_, h_fractal_, h_body_, w_body_);
      }
      helper.Submit();
    }

    struct RectPart {
      int64_t element_count;
      int64_t tile_size;
      int64_t group_count;
      int64_t fractal_offset;
    };

    struct PartOffset {
      int factor_dim;
      int part_dim;
      int64_t offset;
    };

    static int64_t SelectExactDivisor(int64_t value, int64_t limit) {
      if (value <= 1 || limit <= 1) {
        return 1;
      }
      for (int64_t candidate = std::min(value, limit); candidate > 1;) {
        const int64_t quotient = value / candidate;
        if (value % candidate == 0) {
          return candidate;
        }
        // Skip to just before the current value / candidate quotient interval.
        candidate = value / (quotient + 1);
      }
      return 1;
    }

    static std::vector<RectPart> BuildBodyTailParts(int64_t body_count, int64_t tail_elements,
                                                    int64_t full_element_count, int64_t tile_size) {
      std::vector<RectPart> parts;
      parts.reserve(2);
      if (tail_elements) {
        parts.push_back({tail_elements, 1, 1, body_count});
      }
      if (body_count) {
        ASSERT(body_count % tile_size == 0);
        parts.push_back({full_element_count, tile_size, body_count / tile_size, 0});
      }
      return parts;
    }

    void CodeGenRect(uint64_t fractal_mask) {
      // Keep the UB row stride 32B aligned and target a 512B continuous MTE3.
      // Exact H/W divisors make every body Tile uniform, so only element tails
      // need separate programs.
      constexpr int64_t target_row_bytes = 512;
      const int64_t fractal_bytes = w_fractal_ * h_fractal_ * item_size_;
      const int64_t tile_fractal_limit = gen_.kernel_->AnalyzeTileSizeLimit() / (w_fractal_ * h_fractal_);
      const int64_t ub_fractal_capacity = g_system.LocalMemSize() / fractal_bytes;
      const int64_t fractal_row_bytes = w_fractal_ * item_size_;
      const int64_t max_w_inner =
        std::max<int64_t>(1, std::min(w_body_, std::min(tile_fractal_limit, target_row_bytes / fractal_row_bytes)));
      int64_t w_inner = SelectExactDivisor(w_body_, max_w_inner);
      const int64_t max_h_inner = std::max<int64_t>(1, std::min(h_body_, tile_fractal_limit / w_inner));
      const int64_t w_fractal_count = w_body_ + (w_tail_ ? 1 : 0);
      const int64_t h_fractal_count = h_body_ + (h_tail_ ? 1 : 0);
      int64_t h_inner = SelectExactDivisor(h_body_, max_h_inner);
      const auto rect_ub_fractals = [](int64_t h, int64_t w) { return 2 * h * w + 3 * h + 1; };
      while (rect_ub_fractals(h_inner, w_inner) > ub_fractal_capacity) {
        h_inner = SelectExactDivisor(h_body_, h_inner - 1);
        if (h_inner == 1 && rect_ub_fractals(h_inner, w_inner) > ub_fractal_capacity) {
          w_inner = SelectExactDivisor(w_body_, w_inner - 1);
        }
      }

      const auto h_parts = BuildBodyTailParts(h_body_, h_tail_, h_fractal_, h_inner);
      const auto w_parts = BuildBodyTailParts(w_body_, w_tail_, w_fractal_, w_inner);
      gen_.SpaceInit();
      gen_.SpaceSplit(0, w_fractal_count, w_fractal_);
      h_idx_ += 1;
      gen_.SpaceTrans(1, h_idx_);
      gen_.SpaceSplit(1, h_fractal_count, h_fractal_);

      constexpr int h_part_dim = 2;
      gen_.SpaceSplit(h_part_dim, CeilDiv(h_fractal_count, h_inner), h_inner);
      const int w_part_dim = h_idx_ + 2;
      gen_.SpaceSplit(w_part_dim, CeilDiv(w_fractal_count, w_inner), w_inner);
      const int h_group_dim = w_part_dim;
      const int w_group_dim = w_part_dim + 1;
      gen_.SpaceTrans(h_part_dim + 1, w_part_dim);
      // Use [width element, width tile, height element, height tile] for every program.
      gen_.SpaceTrans(2, 3);
      gen_.SpaceTrans(1, 2);
      gen_.SaveSpace();
      const DimArray rect_size = gen_.kernel_->DimSpace();

      part_base_ = 1;
      for (size_t i = FractalRowMajorAxes::kRank; i < rect_size.size(); ++i) {
        if (static_cast<int>(i) != h_group_dim && static_cast<int>(i) != w_group_dim) {
          part_base_ *= rect_size[i];
        }
      }

      const int program_count = static_cast<int>(h_parts.size() * w_parts.size());
      VectorDupHelper helper(gen_.kernel_, program_count,
                             gen_.ReserveReloc(gen_.kernel_->static_ops_.size() * program_count),
                             h_fractal_count * w_fractal_count);
      SetFractalRowMajor(fractal_mask, gen_.kernel_->static_ops_);
      for (const auto &h : h_parts) {
        for (const auto &w : w_parts) {
          const bool lead_folded = w.element_count == 1;
          if (lead_folded) {
            ClearFractal(fractal_mask, gen_.kernel_->static_ops_);
            ClearFractalRowMajor(fractal_mask, gen_.kernel_->static_ops_);
          }
          DimArray part_size = rect_size;
          part_size[FractalRowMajorAxes::kWidthElement] = w.element_count;
          part_size[FractalRowMajorAxes::kHeightElement] = h.element_count;
          part_size[FractalRowMajorAxes::kHeightTile] = h.tile_size;
          part_size[FractalRowMajorAxes::kWidthTile] = w.tile_size;
          part_size[h_group_dim] = h.group_count;
          part_size[w_group_dim] = w.group_count;
          gen_.ApplySubSpace(part_size);
          for (const auto &record : gen_.space_records_) {
            record.ndd->strides.resize(0);
          }
          UpdateViewOffset({FractalRowMajorAxes::kHeightElement, FractalRowMajorAxes::kHeightTile, h.fractal_offset},
                           {FractalRowMajorAxes::kWidthElement, FractalRowMajorAxes::kWidthTile, w.fractal_offset});
          const uint64_t part_fractal_count = h.tile_size * h.group_count * w.tile_size * w.group_count;
          helper.Append(part_fractal_count, part_base_ * part_fractal_count);
          if (lead_folded) {
            SetFractal(fractal_mask, gen_.kernel_->static_ops_);
            SetFractalRowMajor(fractal_mask, gen_.kernel_->static_ops_);
          }
        }
      }
      ClearFractalRowMajor(fractal_mask, gen_.kernel_->static_ops_);
      helper.Submit();
    }

    void UpdateViewOffset(PartOffset h, PartOffset w) {
      for (auto *op : gen_.kernel_->static_ops_) {
        auto *acc = static_cast<NDAccess *>(op);
        const auto &stride = *acc->stride_;
        const uint32_t bcast_mask = gen_.GetBCast(op);
        uint64_t offset = 0;
        // A broadcast dimension has no independent GM tile. Keep its base address
        // while advancing the output and non-broadcast inputs to the next tile.
        if ((bcast_mask & (1u << h.factor_dim)) == 0) {
          offset += stride[h.part_dim] * static_cast<uint64_t>(h.offset);
        }
        if ((bcast_mask & (1u << w.factor_dim)) == 0) {
          offset += stride[w.part_dim] * static_cast<uint64_t>(w.offset);
        }
        acc->ViewUpdate(offset);
      }
    }

    void GenDup(VectorDupHelper &helper, int64_t h_part_off, int64_t w_part_off, int64_t w_fac_size, int64_t h_fac_size,
                int64_t h_part_size, int64_t w_part_size) {
      constexpr int w_factor_dim = 0;
      constexpr int h_factor_dim = 1;
      constexpr int h_part_dim = 2;
      int w_part_dim = h_idx_ + 1;
      size_[w_factor_dim] = w_fac_size;
      size_[h_factor_dim] = h_fac_size;
      size_[h_part_dim] = h_part_size;
      size_[w_part_dim] = w_part_size;
      gen_.ApplySubSpace(size_);
      UpdateViewOffset({h_factor_dim, h_part_dim, h_part_off}, {w_factor_dim, w_part_dim, w_part_off});
      const uint64_t part_num = h_part_size * w_part_size;
      const uint64_t quota_w_fac_size = std::max<int64_t>(w_fac_size, 16);
      const uint64_t element_quota = part_num * h_fac_size * quota_w_fac_size;
      helper.Append(element_quota, part_base_ * part_num);
    }

    static void SetFractal(uint64_t mask, const std::vector<NDObject *> &objects) {
      while (mask) {
        auto idx = 63 - __builtin_clzll(mask);
        mask &= ~(1ull << idx);
        objects[idx]->flags_ |= OBJ_FLAG_VIEW_LOAD_FRACTAL;
      }
    }

    static void ClearFractal(uint64_t mask, const std::vector<NDObject *> &objects) {
      while (mask) {
        auto idx = 63 - __builtin_clzll(mask);
        mask &= ~(1ull << idx);
        objects[idx]->flags_ &= ~OBJ_FLAG_VIEW_LOAD_FRACTAL;
      }
    }

    static void SetFractalRowMajor(uint64_t mask, const std::vector<NDObject *> &objects) {
      while (mask) {
        auto idx = 63 - __builtin_clzll(mask);
        mask &= ~(1ull << idx);
        objects[idx]->flags_ |= OBJ_FLAG_VIEW_LOAD_FRACTAL_ROW_MAJOR;
      }
    }

    static void ClearFractalRowMajor(uint64_t mask, const std::vector<NDObject *> &objects) {
      while (mask) {
        auto idx = 63 - __builtin_clzll(mask);
        mask &= ~(1ull << idx);
        objects[idx]->flags_ &= ~OBJ_FLAG_VIEW_LOAD_FRACTAL_ROW_MAJOR;
      }
    }

    SchGenHelper &gen_;
    const DimArray &space_;
    int64_t h_fractal_;
    int64_t w_fractal_;
    int64_t w_body_;
    int64_t w_tail_;
    int64_t h_body_;
    int64_t h_tail_;
    DimArray size_;
    uint64_t part_base_;
    uint64_t item_size_;
    int h_idx_;
  };

  for (auto op : kernel_->static_ops_) {
    auto acc = static_cast<NDAccess *>(op);
    if (acc->stride_ == nullptr) {
      AllocStride(acc);
    }
  }
  FractalDunGen gen(*this, h_idx, item_size);
  if (g_system.Arch() == AiCoreArch::kAiCore_C220) {
    gen.CodeGenRect(fractal_mask);
  } else {
    gen.CodeGen(fractal_mask);
  }
  ResetStrides();
  FractalDunGen::ClearFractal(fractal_mask, kernel_->static_ops_);
  return 0;
}

namespace {
template <typename T>
void TravelInput(std::vector<NDObject *> &stack, NDObject *root, const T &func) {
  stack.push_back(root);
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (func(top)) {
      top->ForInput([&stack](NDObject *in) { stack.push_back(in); });
    }
  }
}

uint64_t AddPengLoad(std::vector<NDObject *> &ios, const std::vector<NDObject *> &pend_load) {
  uint64_t num = 0;
  for (auto op : pend_load) {
    if (std::find(ios.begin(), ios.end(), op) == ios.end()) {
        ios.push_back(op);
        num++;
    }
  }
  return num;
}

void ApplySliceSpace(SliceOp *slice) {
  for (auto &sd : slice->sdims_) {
    if (int i = sd.index; slice->nd_[i] == 1 && slice->lhs_->nd_[i] > 1) {
      std::vector<NDObject *> stack;
      TravelInput(stack, slice->lhs_, [i](NDObject *op) {
        if (auto ndd = op->Ndd()) {
          ndd->dims[i] = 1;
        }
        return op->nd_[i] > 1;
      });
    }
  }
}

void ApplySplitSpace(SplitOp *split) {
  size_t d = split->main_->split_dim_;
  if (split->nd_[d] == 1 && split->lhs_->nd_[d] > 1) {
    std::vector<NDObject *> stack;
    TravelInput(stack, split->lhs_, [d](NDObject *op) {
      if (auto ndd = op->Ndd()) {
        ndd->dims[d] = 1;
      }
      return op->nd_[d] > 1;
    });
  }
}

void ApplyViewOpSpace(NDObject *op) {
  if (op->obj_id_ == kSliceOp) {
    return ApplySliceSpace(static_cast<SliceOp *>(op));
  } else if (op->obj_id_ == kSplitOp) {
    return ApplySplitSpace(static_cast<SplitOp *>(op));
  }
}

uint64_t UpdateSliceOffset(SliceOp *slice, uint32_t bcast_mask, const DimArray &stride, uint64_t byte_offset) {
  for (auto &sd : slice->sdims_) {
    if (!((bcast_mask >> sd.index) & 1)) {
      byte_offset += sd.begin * stride[sd.index];
    }
  }
  return byte_offset;
}

uint64_t UpdateSplitOffset(SplitOp *split, uint32_t bcast_mask, const DimArray &stride, uint64_t byte_offset) {
  auto *main = split->main_;
  if (size_t split_dim = main->split_dim_; split_dim < stride.size() && !((bcast_mask >> split_dim) & 1)) {
    byte_offset += split->slice_idx_ * main->split_size_ * stride[split_dim];
  }
  return byte_offset;
}

uint64_t UpdateViewOpOffset(NDObject *op, uint32_t bcast_mask, const DimArray &stride, uint64_t byte_offset) {
  if (op->obj_id_ == kSliceOp) {
    return UpdateSliceOffset(static_cast<SliceOp *>(op), bcast_mask, stride, byte_offset);
  } else if (op->obj_id_ == kSplitOp) {
    return UpdateSplitOffset(static_cast<SplitOp *>(op), bcast_mask, stride, byte_offset);
  }
  return byte_offset;
}
}

ConcatSchGen::ConcatSchGen(VectorKernel *kernel, ConcatOp *concat, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), concat_(concat) {
  constexpr int kInvalidDomain = -1;
  constexpr int kConcatDomain = -2;
  for (auto op : objects) {
    op->reuse_dep_ = kInvalidDomain;
  }
  concat->reuse_dep_ = kConcatDomain;
  std::vector<NDObject *> stack;
  int slice_cnt = concat->slices_.size();
  slice_ios_.resize(slice_cnt);
  for (int i = 0; i < slice_cnt; ++i) {
    auto &s = slice_ios_[i];
    TravelInput(stack, concat->slices_[i].input, [&s, i](NDObject *in) {
      if (in->reuse_dep_ == i) {
        return false;
      }
      if (in->IsLoad()) {
        s.ios.push_back(in);
        s.load_num++;
      }
      in->reuse_dep_ = i;
      return true;
    });
  }
  std::vector<NDObject *> pend_load;
  for (size_t k = 0; k < kernel->load_num_; ++k) {
    kernel->static_ops_[k]->xbuf_ = k;
  }
  for (size_t k = kernel->load_num_; k < kernel->static_ops_.size(); ++k) {
    auto store = kernel->static_ops_[k];
    store->xbuf_ = k;
    int joined = kInvalidDomain;
    TravelInput(stack, store, [&joined, &pend_load, kInvalidDomain](NDObject *in) -> bool {
      if (in->IsLoad()) {
        auto it = std::find_if(pend_load.begin(), pend_load.end(), [in](NDObject *x) { return x == in; });
        if (it == pend_load.end()) {
          pend_load.push_back(in);
        }
      }
      if (auto dom = in->reuse_dep_; dom != kInvalidDomain) {
        if (joined == kInvalidDomain) {
          joined = dom;
        }
        EXCEPTION_IF(joined != dom && !in->IsLoad(), "concat multi domain depend");
        return false;
      }
      return true;
    });
    store->reuse_dep_ = joined;
    if (joined == kConcatDomain) {
      for (auto &s : slice_ios_) {
        s.ios.push_back(store);
      }
    } else if (joined >= 0) {
      slice_ios_[joined].ios.push_back(store);
      for (size_t i = 0; i < slice_ios_.size(); ++i) {
        if (i != static_cast<size_t>(joined)) {
          slice_ios_[i].deads.push_back(store);
        }
      }
    } else {
      DvmException("store is not joined");  // TODO: add seperate path if joined is invalid
    }
    if (!pend_load.empty()) {
      for (auto op : pend_load) {
        EXCEPTION_IF(
          (op->reuse_dep_ >= 0 && joined == kConcatDomain) || (op->reuse_dep_ == kConcatDomain && joined >= 0),
          "concat multi domain depend");
        op->reuse_dep_ = joined;
      }
      if (joined == kConcatDomain) {
        for (auto &s : slice_ios_) {
          s.load_num += AddPengLoad(s.ios, pend_load);;
        }
      } else {
        slice_ios_[joined].load_num += AddPengLoad(slice_ios_[joined].ios, pend_load);;
      }
      pend_load.clear();
    }
  }
  for (auto &s : slice_ios_) {
    std::sort(s.ios.begin(), s.ios.end(), [](NDObject *a, NDObject *b) { return a->xbuf_ < b->xbuf_; });
    for (size_t i = 0; i < s.ios.size(); ++i) {
      if (s.ios[i]->reuse_dep_ != kConcatDomain) {
        s.slice_mask |= 1ull << i;
      }
    }
  }
}

int64_t ConcatSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  DimArray size = concat_->nd_.dims();
  int cat_dim = concat_->CatDim();
  for (auto &s : slice_ios_) {
    s.bcast_mask = 0;
    for (size_t j = 0; j < s.ios.size(); ++j) {
      if ((s.slice_mask >> j) & 1) continue;  // concat input partition: skip
      auto acc = static_cast<NDAccess *>(s.ios[j]);
      if (acc->nd_[cat_dim] == 1) {
        s.bcast_mask |= (1ULL << j);
      }
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int dup_num = concat_->slices_.size();
  auto &static_ops = kernel_->static_ops_;
  VectorDupHelper helper(kernel_, dup_num, ReserveReloc(static_ops.size() * dup_num), size[cat_dim]);
  size_t cat_offset = 0;
  auto ctx = concat_->PartialInit();
  for (int cat_idx = 0; cat_idx < dup_num; ++cat_idx) {
    auto &slice = concat_->slices_[cat_idx];
    auto cat_size = slice.size;
    size[cat_dim] = cat_size;
    concat_->PartialSet(slice.input);
    ApplySubSpace(size);
    helper.Reset();
    auto &entry = slice_ios_[cat_idx];
    for (size_t j = 0; j < entry.ios.size(); ++j) {
      auto acc = static_cast<NDAccess *>(entry.ios[j]);
      if (!((entry.slice_mask >> j) & 1)) {
        auto bcast = (entry.bcast_mask >> j) & 1;
        acc->ViewUpdate(bcast ? 0 : (*acc->stride_)[cat_dim] * static_cast<uint64_t>(cat_offset));
      }
    }
    for (auto op : entry.deads) {
      op->flags_ |= OBJ_FLAG_DEAD;
    }
    std::swap(static_ops, entry.ios);
    kernel_->load_num_ = entry.load_num;
    cat_offset += cat_size;
    helper.DoAppend(cat_size);
    std::swap(static_ops, entry.ios);
  }
  concat_->PartialRecover(ctx);
  helper.Submit();
  ResetStrides();
  return 0;
}

SplitSchGen::SplitSchGen(VectorKernel *kernel, SplitOpM *split, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), split_(split) {
  ASSERT(split_->slice_idx_ == 0);
  constexpr int kInvalidDomain = -1;
  constexpr int kSplitDomain = -2;
  constexpr int kSliceDomain = -3;
  for (auto op : objects) {
    op->reuse_dep_ = kInvalidDomain;
  }
  std::vector<NDObject *> stack;
  std::vector<NDObject *> pend_load;
  TravelInput(stack, split->lhs_, [&pend_load, kSplitDomain](NDObject *in) {
    if (in->reuse_dep_ == kSplitDomain) {
      return false;
    }
    in->reuse_dep_ = kSplitDomain;
    if (in->IsLoad()) {
      pend_load.push_back(in);
    }
    return true;
  });
  slice_ios_.resize(split->siblings_.size());
  for (int i = 0; i < static_cast<int>(split->siblings_.size()); ++i) {
    split->siblings_[i]->reuse_dep_ = i;
    slice_ios_[i].ios = pend_load;
    slice_ios_[i].load_num = pend_load.size();
  }
  pend_load.clear();
  for (uint64_t i = kernel->load_num_; i < kernel->static_ops_.size(); ++i) {
    auto store = kernel->static_ops_[i];
    int joined = kInvalidDomain;
    TravelInput(stack, store, [&](NDObject *in) -> bool {
      auto dom = in->reuse_dep_;
      if (in->IsLoad() && dom != kSplitDomain) {
        auto it = std::find_if(pend_load.begin(), pend_load.end(), [in](NDObject *x) { return x == in; });
        if (it == pend_load.end()) {
          pend_load.push_back(in);
        }
      }
      if (dom != kInvalidDomain && dom != kSliceDomain) {
        if (joined == kInvalidDomain) {
          joined = dom;
        }
        EXCEPTION_IF(joined != dom, "split multi domain depend");
        return false;
      }
      return true;
    });
    store->reuse_dep_ = joined;
    if (joined == kSplitDomain) {
      for (auto &s : slice_ios_) {
        s.ios.push_back(store);
      }
    } else if (joined >= 0) {
      slice_ios_[joined].ios.push_back(store);
      for (size_t i = 0; i < slice_ios_.size(); ++i) {
        if (i != static_cast<size_t>(joined)) {
          slice_ios_[i].deads.push_back(store);
        }
      }
    } else {
      DvmException("store is not joined");  // TODO: add seperate path if joined is invalid
    }
    if (!pend_load.empty()) {
      if (joined == kSplitDomain) {
        for (auto op : pend_load) {
          op->reuse_dep_ = kSplitDomain;
        }
        for (auto &s : slice_ios_) {
          s.load_num += AddPengLoad(s.ios, pend_load);
        }
      } else {
        for (auto op : pend_load) {
          op->reuse_dep_ = kSliceDomain;
        }
        slice_ios_[joined].load_num += AddPengLoad(slice_ios_[joined].ios, pend_load);
      }
      pend_load.clear();
    }
  }
  for (uint64_t i = 0; i < kernel->static_ops_.size(); ++i) {
    kernel->static_ops_[i]->xbuf_ = i;
  }
  for (auto &s : slice_ios_) {
    std::sort(s.ios.begin(), s.ios.end(), [](NDObject *a, NDObject *b) { return a->xbuf_ < b->xbuf_; });
    for (size_t i = 0; i < s.ios.size(); ++i) {
      if (s.ios[i]->reuse_dep_ != kSplitDomain) {
        s.slice_mask |= 1ull << i;
      }
    }
    EXCEPTION_IF(s.slice_mask >> s.load_num == 0, "split has no slice store");
  }
}

int64_t SplitSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  int split_dim = split_->split_dim_;
  for (auto &s : slice_ios_) {
    s.bcast_mask = 0;
    for (size_t j = 0; j < s.ios.size(); ++j) {
      if ((s.slice_mask >> j) & 1) continue;
      auto acc = static_cast<NDAccess *>(s.ios[j]);
      if (acc->nd_[split_dim] == 1) {
        s.bcast_mask |= 1ull << j;
      }
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int slice_num = static_cast<int>(split_->siblings_.size());
  int64_t split_dim_size = split_->lhs_->nd_[split_dim];
  auto &static_ops = kernel_->static_ops_;
  VectorDupHelper helper(kernel_, slice_num, ReserveReloc(static_ops.size() * slice_num), split_dim_size);
  int64_t split_offset = 0;
  DimArray size = split_->nd_.dims();
  for (int slice_idx = 0; slice_idx < slice_num; ++slice_idx) {
    int64_t split_size =
      slice_idx < slice_num - 1 ? split_->split_size_ : split_dim_size - split_->split_size_ * (slice_num - 1);
    size[split_dim] = split_size;
    ApplySubSpace(size);
    helper.Reset();
    auto &entry = slice_ios_[slice_idx];
    for (size_t j = 0; j < entry.ios.size(); ++j) {
      auto acc = static_cast<NDAccess *>(entry.ios[j]);
      if (!((entry.slice_mask >> j) & 1)) {
        auto bcast = (entry.bcast_mask >> j) & 1;
        acc->ViewUpdate(bcast ? 0 : (*acc->stride_)[split_dim] * static_cast<uint64_t>(split_offset));
      }
    }
    for (auto op : entry.deads) {
      op->flags_ |= OBJ_FLAG_DEAD;
    }
    std::swap(static_ops, entry.ios);
    kernel_->load_num_ = entry.load_num;
    split_offset += split_size;
    helper.DoAppend(split_size);
    std::swap(static_ops, entry.ios);
  }
  helper.Submit();
  ResetStrides();
  return 0;
}

void DupTilingSchGen::DupCodeGen(int split_dim, int64_t body_size, int64_t tail_size) {
  DimArray dim_space = kernel_->DimSpace();
  int64_t split_size = dim_space[split_dim];
  auto &static_ops = kernel_->static_ops_;
  uint64_t bcast_mask = 0;
  for (size_t i = 0; i < static_ops.size(); ++i) {
    auto acc = static_cast<NDAccess *>(static_ops[i]);
    if (acc->nd_[split_dim] == 1) {
      bcast_mask |= 1ull << i;
    }
    if (acc->stride_ == nullptr) {
      AllocStride(acc);
    }
  }
  auto reloc_ios = [&static_ops, split_dim, bcast_mask](int64_t offset) {
    for (size_t i = 0; i < static_ops.size(); ++i) {
      auto acc = static_cast<NDAccess *>(static_ops[i]);
      acc->ViewUpdate(bcast_mask >> i & 1ull ? 0 : (*acc->stride_)[split_dim] * offset);
    }
  };
  SpaceInit();
  SaveSpace();
  VectorDupHelper helper(kernel_, 2, ReserveReloc(static_ops.size() * 2), split_size);
  dim_space[split_dim] = tail_size;
  ApplySubSpace(dim_space);
  reloc_ios(body_size);
  helper.Append(tail_size);
  dim_space[split_dim] = body_size;
  ApplySubSpace(dim_space);
  reloc_ios(0);
  helper.Append(body_size);
  helper.Submit();
  ResetStrides();
}

SliceSchGen::SliceSchGen(VectorKernel *kernel, NDObject *slice, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), slice_(static_cast<SliceOp *>(slice)) {
  for (size_t i = 0; i < kernel->load_num_; ++i) {
    kernel->static_ops_[i]->reuse_dep_ = i;
  }
  std::vector<NDObject *> stack;
  TravelInput(stack, slice, [this](NDObject *in) {
    if (in->IsLoad()) {
      full_io_mask_ |= 1ull << in->reuse_dep_;
    }
    return true;
  });
  for (size_t i = kernel->load_num_; i < kernel->static_ops_.size(); ++i) {
    bool full_dom = true;
    TravelInput(stack, kernel->static_ops_[i], [slice, &full_dom](NDObject *in) {
      if (in == slice) {
        full_dom = false;
      }
      return full_dom;
    });
    if (full_dom) {
      full_io_mask_ |= 1ull << i;
    } else {
      slice_dom_ = kernel->static_ops_[i];
    }
  }
}

int64_t SliceSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  int dup_num = 1;
  uint64_t full_quota = 0;
  uint64_t slice_quota = slice_->nd_.dims().prod();
  if (full_io_mask_ >> kernel_->load_num_) {
    dup_num = 2;
    full_quota = slice_->lhs_->nd_.dims().prod();
  }
  auto &static_ops = kernel_->static_ops_;
  VectorDupHelper helper(kernel_, dup_num, ReserveReloc(static_ops.size() * dup_num), full_quota + slice_quota);
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto load = static_cast<NDAccess *>(static_ops[i]);
    if (((full_io_mask_ >> i) & 1) && load->stride_ == nullptr) {
      AllocStride(load);
    }
  }
  DimArray slice_space = slice_dom_->nd_.dims();
  if (dup_num == 2) {
    helper.Reset();
    std::vector<NDObject *> ios;
    uint64_t load_num = 0;
    ios.reserve(static_ops.size());
    for (size_t i = 0; i < static_ops.size(); ++i) {
      if ((full_io_mask_ >> i) & 1) {
        ios.push_back(static_ops[i]);
        if (i < kernel_->load_num_) {
          load_num++;
        }
      } else if (i >= kernel_->load_num_) {
        static_ops[i]->flags_ |= OBJ_FLAG_DEAD;
      }
    }
    auto dom = slice_->lhs_;
    std::swap(static_ops, ios);
    std::swap(kernel_->load_num_, load_num);
    std::swap(kernel_->dom_, dom);
    helper.DoAppend(full_quota);
    std::swap(static_ops, ios);
    std::swap(kernel_->load_num_, load_num);
    std::swap(kernel_->dom_, dom);
  }
  helper.Reset();
  ApplySubSpace(slice_space);
  ApplySliceSpace(slice_);
  if (dup_num == 2) {
    for (size_t i = kernel_->load_num_; i < static_ops.size(); ++i) {
      if ((full_io_mask_ >> i) & 1) {
        static_ops[i]->flags_ |= OBJ_FLAG_DEAD;
      }
    }
  }
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    if ((full_io_mask_ >> i) & 1) {
      auto load = static_cast<NDAccess *>(static_ops[i]);
      auto offset = UpdateSliceOffset(slice_, GetBCast(load), (*load->stride_), 0);
      load->ViewUpdate(offset);
    }
  }
  helper.DoAppend(slice_quota);
  helper.Submit();
  ResetStrides();
  return 0;
}

namespace {
class GroupCollector {
 public:
  using LoadInfo = GeneralViewSchGen::LoadInfo;
  struct VisitInfo {
    uint64_t view_mask;
    uint32_t count{0};
    uint32_t load_count{0};
  };
  GroupCollector(const std::vector<NDObject *> &objects) {
    visited_.resize(objects.size());
    int index = 0;
    for (auto op : objects) {
      op->index_ = index++;
    }
  }
  void Collect(NDObject *root) {
    view_mask_ = 0;
    visit_cnt_++;
    loads_.clear();
    stack_.push_back(root);
    visited_[root->index_].view_mask = 0;
    visited_[root->index_].count = visit_cnt_;
    while (!stack_.empty()) {
      auto top = stack_.back();
      stack_.pop_back();
      auto &visit = visited_[top->index_];
      uint64_t top_mask = visit.view_mask;
      if (top->IsViewOp()) {
        uint64_t mask = 1ULL << GetViewIndex(top);
        top_mask |= mask;
        view_mask_ |= mask;
        if (top->obj_id_ == kConcat) {
          EXCEPTION_IF((top_mask & ~mask) != 0, "split/slice is not allowed directly or indirectly after concat");
          continue;
        }
      } else if (top->IsLoad()) {
        auto io_idx = GetIOIndex(top);
        if (visit.load_count < visit_cnt_) {
          auto &info = loads_.emplace_back();
          info.view_mask = top_mask;
          info.io_idx = io_idx;
          visit.load_count = visit_cnt_;
        } else {
          auto it = std::find_if(loads_.begin(), loads_.end(),
                                 [io_idx](const LoadInfo &info) { return info.io_idx == io_idx; });
          ASSERT(it != loads_.end());
          it->view_mask = top_mask;
        }
      }
      top->ForInput([&](NDObject *in) {
        auto &info = visited_[in->index_];
        if (info.count < visit_cnt_) {
          info.view_mask = top_mask;
          info.count = visit_cnt_;
          stack_.push_back(in);
        } else if (info.view_mask != top_mask) {
          info.view_mask |= top_mask;
          stack_.push_back(in);
        }
      });
    }
  }
  void MergeTo(std::vector<LoadInfo> &load_infos) {
    for (auto &info : loads_) {
      int ins_idx = -1;
      for (size_t k = 0; k < load_infos.size(); ++k) {
        if (load_infos[k].io_idx >= info.io_idx) {
          ins_idx = k;
          break;
        }
      }
      if (ins_idx == -1) {
        load_infos.push_back(info);
      } else if (load_infos[ins_idx].io_idx != info.io_idx) {
        load_infos.emplace_back();
        for (int i = load_infos.size() - 1; i > ins_idx; --i) {
          load_infos[i] = load_infos[i - 1];
        }
        load_infos[ins_idx] = info;
      }
    }
  }

  static void SetViewIndex(NDObject *op, int i) { op->reuse_dep_ = i; }
  static int GetViewIndex(NDObject *op) { return op->reuse_dep_; }
  static void SetIOIndex(NDObject *op, int i) { op->reuse_dep_ = i; }
  static int GetIOIndex(NDObject *op) { return op->reuse_dep_; }

  std::vector<NDObject *> stack_;
  std::vector<LoadInfo> loads_;
  std::vector<VisitInfo> visited_;
  uint32_t visit_cnt_{0};
  uint64_t view_mask_;
};
}  // namespace

GeneralViewSchGen::GeneralViewSchGen(VectorKernel *kernel, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel) {
  GroupCollector gc(objects);
  auto &static_ops = kernel->static_ops_;
  load_bcast_mask_.resize(kernel->load_num_, 0);
  for (int i = 0; i < static_cast<int>(kernel->load_num_); ++i) {
    gc.SetIOIndex(static_ops[i], i);
  }
  uint64_t concat_mask = 0;
  for (auto *op : objects) {
    if (op->IsViewOp()) {
      int view_idx = view_ops_.size();
      gc.SetViewIndex(op, view_idx);
      view_ops_.push_back(op);
      if (op->obj_id_ == kConcat) {
        EXCEPTION_IF(concat_mask != 0, "at most one concat op in the same kernel");
        concat_view_idx_ = view_idx;
        concat_mask = 1ULL << view_idx;
      }
    }
  }
  if (concat_mask) {
    auto concat = static_cast<ConcatOp *>(view_ops_[concat_view_idx_]);
    size_t cat_size = concat->slices_.size();
    concat_groups_.resize(cat_size);
    for (size_t i = 0; i < cat_size; ++i) {
      gc.Collect(concat->slices_[i].input);
      gc.MergeTo(concat_groups_[i].load_infos);
      concat_groups_[i].view_mask = gc.view_mask_;
    }
  }
  int store_num = static_cast<int>(static_ops.size() - kernel->load_num_);
  for (int i = 0; i < store_num; ++i) {
    auto *store = static_ops[i + kernel->load_num_];
    gc.Collect(store);
    if (gc.view_mask_ & concat_mask) {
      for (auto &g : concat_groups_) {
        for (auto &info : gc.loads_) {
          info.view_mask |= concat_mask;
        }
        gc.MergeTo(g.load_infos);
        g.static_ops.push_back(store);
        g.view_mask |= gc.view_mask_;
      }
    } else {
      size_t index = groups_.size();
      if (gc.view_mask_) {
        for (size_t j = 0; j < groups_.size(); ++j) {
          if (groups_[j].view_mask == gc.view_mask_) {
            index = j;
            break;
          }
        }
      }
      if (index == groups_.size()) {
        auto &g = groups_.emplace_back();
        g.view_mask = gc.view_mask_;
      }
      gc.MergeTo(groups_[index].load_infos);
      groups_[index].static_ops.push_back(store);
    }
  }
  auto &temp_ops = gc.stack_;
  auto prepend_loads = [&static_ops, &temp_ops](const std::vector<LoadInfo> &infos, std::vector<NDObject *> &ios) {
    temp_ops.clear();
    temp_ops.reserve(infos.size());
    for (auto &i : infos) {
      temp_ops.push_back(static_ops[i.io_idx]);
    }
    ios.insert(ios.begin(), temp_ops.begin(), temp_ops.end());
  };
  for (auto &g : groups_) {
    prepend_loads(g.load_infos, g.static_ops);
    g.dom = g.static_ops.back();
  }
  for (auto &g : concat_groups_) {
    prepend_loads(g.load_infos, g.static_ops);
  }
}

int64_t GeneralViewSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  auto &static_ops = kernel_->static_ops_;
  size_t orig_load_num = kernel_->load_num_;

  for (size_t i = 0; i < orig_load_num; ++i) {
    auto load = static_cast<NDAccess *>(static_ops[i]);
    load_bcast_mask_[i] = GetBCast(load);
    if (load->stride_ == nullptr) {
      AllocStride(load);
    }
  }

  int dup_num = static_cast<int>(groups_.size());
  uint64_t total_quota = 0;
  uint64_t concat_quota = 0;
  for (auto &g : groups_) {
    g.space = g.dom->nd_.dims();
    g.quota = static_cast<uint64_t>(g.space.prod());
    total_quota += g.quota;
  }
  if (!concat_groups_.empty()) {
    dup_num += static_cast<int>(concat_groups_.size());
    concat_quota = static_cast<uint64_t>(view_ops_[concat_view_idx_]->nd_.dims().prod());
    total_quota += concat_quota;
  }
  VectorDupHelper helper(kernel_, dup_num, ReserveReloc(static_ops.size() * dup_num), total_quota);

  auto update_store_dead = [&static_ops, orig_load_num](const std::vector<NDObject *> &ios, size_t load_num) {
    for (size_t i = orig_load_num; i < static_ops.size(); ++i) {
      static_ops[i]->flags_ |= OBJ_FLAG_DEAD;
    }
    for (size_t i = load_num; i < ios.size(); ++i) {
      ios[i]->flags_ &= ~OBJ_FLAG_DEAD;
    }
  };
  auto apply_view_space = [this](uint64_t view_mask) {
    for (size_t i = 0; i < view_ops_.size(); ++i) {
      if ((view_mask >> i) & 1) {
        ApplyViewOpSpace(view_ops_[i]);
      }
    }
  };
  if (concat_quota) {
    for (size_t i = orig_load_num; i < static_ops.size(); ++i) {
      if (auto io= static_cast<NDAccess *>(static_ops[i]); io->stride_ == nullptr) {
        AllocStride(io);
      }
    }
    auto concat = static_cast<ConcatOp *>(view_ops_[concat_view_idx_]);
    ConcatOp::PartialCtx ctx = concat->PartialInit();
    int cat_dim = concat->CatDim();
    DimArray concat_space = concat->nd_.dims();
    concat_quota /= concat_space[cat_dim];
    int64_t slice_start = 0;
    for (size_t gi = 0; gi < concat_groups_.size(); ++gi) {
      helper.Reset();
      auto &g = concat_groups_[gi];
      size_t load_num = g.load_infos.size();
      update_store_dead(g.static_ops, load_num);
      std::swap(static_ops, g.static_ops);
      kernel_->load_num_ = load_num;

      auto &slice = concat->slices_[gi];
      concat->PartialSet(slice.input);
      concat_space[cat_dim] = slice.size;

      for (size_t i = 0; i < load_num; ++i) {
        auto *acc = static_cast<NDAccess *>(static_ops[i]);
        uint64_t byte_offset = 0;
        uint32_t bcast_mask = load_bcast_mask_[g.load_infos[i].io_idx];
        auto view_mask = g.load_infos[i].view_mask;
        uint32_t chain_mask = bcast_mask;
        if ((view_mask >> concat_view_idx_) & 1) {
          for (size_t j = 0; j < view_ops_.size(); ++j) {
            if ((view_mask >> j) & 1 && j != static_cast<size_t>(concat_view_idx_)) {
              chain_mask |= GetBCast(view_ops_[j]);
            }
          }
        }
        for (size_t j = 0; j < view_ops_.size(); ++j) {
          if ((view_mask >> j) & 1) {
            if (j == static_cast<size_t>(concat_view_idx_)) {
              if (!((chain_mask >> cat_dim) & 1)) {
                byte_offset += (*acc->stride_)[cat_dim] * slice_start;
              }
            } else {
              byte_offset = UpdateViewOpOffset(view_ops_[j], bcast_mask, *acc->stride_, byte_offset);
            }
          }
        }
        acc->ViewUpdate(byte_offset);
      }
      for (size_t i = load_num; i < static_ops.size(); ++i) {
        auto *acc = static_cast<NDAccess *>(static_ops[i]);
        acc->ViewUpdate((*acc->stride_)[cat_dim] * slice_start);
      }
      ApplySubSpace(concat_space);
      apply_view_space(g.view_mask);
      helper.DoAppend(concat_quota * slice.size);
      slice_start += slice.size;
      std::swap(static_ops, g.static_ops);
    }
    concat->PartialRecover(ctx);
  }

  for (auto &g : groups_) {
    helper.Reset();
    size_t load_num = g.load_infos.size();
    update_store_dead(g.static_ops, load_num);
    std::swap(static_ops, g.static_ops);
    kernel_->load_num_ = load_num;
    for (size_t i = 0; i < load_num; ++i) {
      auto *acc = static_cast<NDAccess *>(static_ops[i]);
      uint64_t byte_offset = 0;
      auto bcast_mask = load_bcast_mask_[g.load_infos[i].io_idx];
      auto view_mask = g.load_infos[i].view_mask;
      for (size_t j = 0; j < view_ops_.size(); ++j) {
        if ((view_mask >> j) & 1) {
          byte_offset = UpdateViewOpOffset(view_ops_[j], bcast_mask, *acc->stride_, byte_offset);
        }
      }
      acc->ViewUpdate(byte_offset);
    }
    ApplySubSpace(g.space);
    apply_view_space(g.view_mask);
    helper.DoAppend(g.quota);
    std::swap(static_ops, g.static_ops);
  }
  helper.Submit();
  ResetStrides();
  kernel_->load_num_ = orig_load_num;
  return 0;
}

SchGenHelper *BuildViewSch(VectorKernel *kernel, const std::vector<NDObject *> &objects) {
  NDObject *view = nullptr;
  for (auto op : objects) {
    if (op->obj_id_ == kSliceOp || op->obj_id_ == kConcat ||
        (op->obj_id_ == kSplitOp && static_cast<SplitOp *>(op)->slice_idx_ == 0)) {
      if (view == nullptr) {
        view = op;
      } else {
        return new GeneralViewSchGen(kernel, objects);
      }
    }
  }
  if (view == nullptr) {
    return nullptr;
  }
  if (view->obj_id_ == kConcat) {
    return new ConcatSchGen(kernel, static_cast<ConcatOp *>(view), objects);
  } else if (view->obj_id_ == kSplitOp) {
    return new SplitSchGen(kernel, static_cast<SplitOpM *>(view), objects);
  } else {
    ASSERT(view->obj_id_ == kSliceOp);
    return new SliceSchGen(kernel, view, objects);
  }
}
}  // namespace dvm
