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
#include <array>
#include <cmath>
#include <limits>
#include "runtime.h"
#include "kernel.h"
#include "xkernel.h"
#include "msprof.h"
#include "ops_m.h"

namespace dvm {
namespace {
constexpr std::array<uint32_t, 7> kTileV2Sizes = {1024, 512, 256, 128, 64, 32, 16};
constexpr std::array<uint32_t, 20> kTileV4Sizes = {
  1024, 768, 512, 384, 320, 256, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16};
constexpr uint64_t MIN_TATL_BLOCK_SIZE = 4096;
constexpr double CUBE_BOUND_RATIO = 0.85;
// Ascend950 AICoreMemoryRates.l2_rate / BF16 Cube throughput. Clock cancels.
constexpr double kL2BytesPerCubeOp = 100.0 / 8192;
constexpr float kDdrBandwidthRatio = 1.0f;
constexpr uint64_t kPerCoreL2ReserveElements[kAiCoreArchEnd] = {
  256 * 128 * 8,  // C220
  256 * 256 * 8,  // C310
};

void MatMulBatchShapeProp(const IntArrayRef *lhs, const IntArrayRef *rhs, ShapeWithRef &shape, int64_t &sym_dim_next) {
  ASSERT(lhs != nullptr && rhs != nullptr && lhs->size >= 2 && rhs->size >= 2);
  auto rank = std::max(lhs->size, rhs->size);
  shape.Resize(rank);
  auto lhs_offset = rank - lhs->size;
  auto rhs_offset = rank - rhs->size;
  for (size_t i = 0; i + 2 < rank; ++i) {
    auto lhs_dim = i < lhs_offset ? 1 : lhs->data[i - lhs_offset];
    auto rhs_dim = i < rhs_offset ? 1 : rhs->data[i - rhs_offset];
    if (lhs_dim == rhs_dim || rhs_dim == 1) {
      shape[i] = lhs_dim;
    } else if (lhs_dim == 1) {
      shape[i] = rhs_dim;
    } else {
      shape[i] = sym_dim_next--;
    }
  }
}

class TileHelper {
 public:
  struct TileCand {
    uint32_t m0{0};
    uint32_t n0{0};
    uint32_t k0{0};
    uint32_t m_loop{0};
    uint32_t n_loop{0};
    uint32_t core_loop{0};
    uint32_t block_dim{0};
  };

  TileHelper(const CubeOp *mm, const vCubeOp *op) : mm_(mm), op_(op) {
    auto bias_size = mm->bias_ ? g_system.BtSize() : 0;
    l0c_max_ = g_system.L0CSize() / sizeof(float);
    l1_max_ = (g_system.L1Size() / 2 - bias_size) / ITEM_SIZE[mm->lhs_->type_id_];
    core_num_ = g_system.CoreNum(CoreType::kAIC);
    round_m_ = RoundUp<uint32_t>(mm->m_real_, CubeOp::BLOCK_SIZE);
    round_n_ = RoundUp<uint32_t>(mm->n_real_, CubeOp::BLOCK_SIZE);
    round_k_ = RoundUp<uint32_t>(mm->k_real_, CubeOp::BLOCK_SIZE);
    n_align_max_ = mm->bias_ != nullptr ? g_system.BtSize() / sizeof(float) : MATMUL_ALIGN_MAX;
  }

  bool GetCandidate(uint32_t x, uint32_t y, TileCand *candidate) const;
  bool GetCandidateMN(uint32_t m0, uint32_t n0, TileCand *candidate) const;

 private:
  const CubeOp *mm_;
  const vCubeOp *op_;
  uint64_t l0c_max_;
  uint64_t l1_max_;
  uint32_t core_num_;
  uint32_t round_m_;
  uint32_t round_n_;
  uint32_t round_k_;
  uint32_t n_align_max_;
};

bool TileHelper::GetCandidate(uint32_t x, uint32_t y, TileCand *candidate) const {
  uint32_t m0, n0, k0;
  if (!mm_->trans_a_) {
    k0 = x;
    n0 = y;
    if (k0 > round_k_ || n0 > round_n_ || l1_max_ <= k0 * n0) return false;
    uint64_t mx = std::min(l0c_max_ / n0, (l1_max_ - k0 * n0) / k0);
    m0 = RoundDown<uint32_t>(mx, mx > CubeOp::CUBE_BLOCK_SIZE ? CubeOp::CUBE_BLOCK_SIZE : CubeOp::BLOCK_SIZE);
    ASSERT((k0 * n0 < l1_max_) && (m0 > 0));
    m0 = std::min({m0, MATMUL_ALIGN_MAX, round_m_});
  } else if (!mm_->trans_b_) {
    m0 = x;
    n0 = y;
    if (m0 > round_m_ || n0 > round_n_) return false;
    uint64_t kx = l1_max_ / (m0 + n0);
    k0 = RoundDown<uint32_t>(kx, kx > CubeOp::CUBE_BLOCK_SIZE ? CubeOp::CUBE_BLOCK_SIZE : CubeOp::BLOCK_SIZE);
    if (m0 * n0 > l0c_max_ || k0 == 0) return false;
    k0 = std::min({k0, MATMUL_ALIGN_MAX, round_k_});
  } else {
    k0 = x;
    m0 = y;
    if (k0 > round_k_ || m0 > round_m_ || l1_max_ <= k0 * m0) return false;
    uint64_t nx = std::min(l0c_max_ / m0, (l1_max_ - k0 * m0) / k0);
    n0 = RoundDown<uint32_t>(nx, nx > CubeOp::CUBE_BLOCK_SIZE ? CubeOp::CUBE_BLOCK_SIZE : CubeOp::BLOCK_SIZE);
    ASSERT((k0 * m0 < l1_max_) && (n0 > 0));
    n0 = std::min({n0, MATMUL_ALIGN_MAX, round_n_});
  }
  if (n0 > n_align_max_ || n0 * k0 + m0 * k0 > l1_max_) return false;
  if ((n0 > CubeOp::BLOCK_SIZE && n0 / 2 >= mm_->n_real_) ||
      (m0 > CubeOp::BLOCK_SIZE && m0 / 2 >= mm_->m_real_)) {
    return false;
  }
  candidate->m0 = m0;
  candidate->n0 = n0;
  candidate->k0 = k0;
  candidate->m_loop = CeilDiv(op_->m_real, m0);
  candidate->n_loop = CeilDiv(op_->n_real, n0);
  candidate->core_loop = candidate->m_loop * candidate->n_loop * mm_->batch_c0_ * mm_->batch_c1_;
  candidate->block_dim = candidate->core_loop < core_num_ ? candidate->core_loop : core_num_;
  return true;
}

bool TileHelper::GetCandidateMN(uint32_t m0, uint32_t n0, TileCand *candidate) const {
  if (m0 > round_m_ || n0 > round_n_ || n0 > n_align_max_ || m0 * n0 > l0c_max_) return false;
  // Match the device's L0 K slice; the host k0 stores the L1 depth.
  const uint32_t item_size = ITEM_SIZE[mm_->lhs_->type_id_];
  const uint32_t l0_elements = g_system.L0ABSize() / (2 * item_size);
  const uint32_t max_mn = std::max(m0, n0);
  const uint32_t l0_k = std::min(round_k_, RoundDown(l0_elements / max_mn, CubeOp::BLOCK_SIZE));
  if (l0_k == 0) return false;
  const uint32_t k_align = !mm_->trans_a_ || mm_->trans_b_ ? 512 / item_size : CubeOp::BLOCK_SIZE;
  const uint32_t max_steps = std::min(CeilDiv(round_k_, l0_k), 8u);
  uint32_t l1_k = 0;
  uint64_t selected_bytes = 0;
  for (uint32_t step = 1; step <= max_steps; ++step) {
    const uint32_t depth = std::min(round_k_, l0_k * step);
    const uint64_t transfer_bytes = static_cast<uint64_t>(max_mn) * depth * item_size;
    if (depth > MATMUL_ALIGN_MAX || static_cast<uint64_t>(m0 + n0) * depth > l1_max_ ||
        transfer_bytes * 4 > g_system.L1Size()) {
      break;
    }
    const bool needs_alignment = l1_k % k_align != 0;
    const bool prefer_alignment = needs_alignment && depth % (256 / item_size) == 0;
    const bool grow_transfer = selected_bytes < 48 * 1024 && (needs_alignment || depth % k_align == 0);
    if (l1_k == 0 || prefer_alignment || grow_transfer) {
      l1_k = depth;
      selected_bytes = transfer_bytes;
    }
  }
  if (l1_k == 0) return false;
  candidate->m0 = m0;
  candidate->n0 = n0;
  candidate->k0 = l1_k;
  candidate->m_loop = CeilDiv(op_->m_real, m0);
  candidate->n_loop = CeilDiv(op_->n_real, n0);
  const uint64_t core_loop = static_cast<uint64_t>(candidate->m_loop) * candidate->n_loop *
                             mm_->batch_c0_ * mm_->batch_c1_;
  if (core_loop > std::numeric_limits<uint32_t>::max()) return false;
  candidate->core_loop = core_loop;
  candidate->block_dim = std::min(candidate->core_loop, core_num_);
  return true;
}

template <size_t N, typename Visit, typename Stop>
void ForEachCubeTilePair(const std::array<uint32_t, N> &candidates, Visit &&visit, Stop &&stop) {
  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y <= x; ++y) {
      const uint32_t tile_x = candidates[x];
      const uint32_t tile_y = candidates[y];
      visit(tile_x, tile_y);
      if (tile_x != tile_y) {
        visit(tile_y, tile_x);
      }
      if (stop()) {
        return;
      }
    }
  }
}
} // namespace

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b)
    : NDObject(lhs, rhs, lhs->type_id_, kCubeOp), trans_a_(trans_a), trans_b_(trans_b) {
  shape_ref_ = &shape_;
  nd_.data = &ndd_;
}

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias)
    : CubeOp(lhs, rhs, trans_a, trans_b) {
  bias_ = bias;
}

void CubeOp::InferTactics(Tactics &t) const {
  t.lhs_pad_size = 0;
  t.rhs_pad_size = 0;
  t.enable_splitk = false;
  t.enable_bias_cast = false;
  t.lhs_nz = false;
  t.rhs_nz = false;
  if (g_system.Arch() == kAiCore_C310) return;

  auto CheckPadOrND2NZ = [this](int64_t size, int64_t &pad, bool &nd2nz) {
    constexpr int64_t max_nd2nz_stride = std::numeric_limits<uint16_t>::max();
    // GMM does not support NZ inputs yet.
    if (obj_id_ == kCubeOp && RoundUp(size, ALIGN_256) > max_nd2nz_stride) {
      nd2nz = true;
      return;
    }
    if (size % ALIGN_128 == 0 || (size < ALIGN_512 && size % ALIGN_32 == 0)) {
      return;
    }
    pad = ALIGN_256 - size % ALIGN_256;
  };
  CheckPadOrND2NZ(trans_a_ ? m_align_ : k_align_, t.lhs_pad_size, t.lhs_nz);
  CheckPadOrND2NZ(trans_b_ ? k_align_ : n_align_, t.rhs_pad_size, t.rhs_nz);

  int64_t m_real = m_real_;
  if (batch_fold_) {
    m_real *= lhs_->nd_[2];
    if (lhs_->nd_.size() > 3) m_real *= lhs_->nd_[3];
  }
  if (k_real_ > MAX_SPLIT_K) {
    const uint64_t input_size = (m_real + n_real_) * k_real_ * ITEM_SIZE[lhs_->type_id_];
    const uint64_t output_size = m_real * n_real_ * sizeof(float);
    const uint64_t min_output_size = g_system.L0CSize() * g_system.CoreNum(CoreType::kAIC);

    uint64_t input_limit = g_system.L2Size();
    if (!trans_a_ && trans_b_) {
      input_limit = g_system.L2Size() * 3 / 2;
    }
    uint64_t split_num = 1;
    if (output_size >= min_output_size && input_size > input_limit) {
      uint64_t input_budget;
      if (output_size < g_system.L2Size() / 2)
        input_budget = (g_system.L2Size() - output_size) * 0.8;
      else {
        input_budget = g_system.L2Size() / 2;
      }
      split_num = CeilDiv(input_size, input_budget);
    }
    if (split_num > 1) {
      const uint64_t stride = RoundUp<uint64_t>(CeilDiv<uint64_t>(k_real_, split_num), ALIGN_256);
      t.k_stride = std::clamp<uint64_t>(stride, MIN_SPLIT_K, MAX_SPLIT_K);
      t.enable_splitk = true;
    }
  }

  if (bias_ && bias_->type_id_ == kBFloat16) {
    t.enable_bias_cast = true;
  }
}

void CubeOp::NormalizeCube() {
  m_align_ = trans_a_ ? lhs_->nd_[0] : lhs_->nd_[1];
  // Only pad the rows, which may result in matrices A and B where some K matrices are padded and some are not.
  // Therefore, we take the maximum among them.
  ka_align_ = trans_a_ ? lhs_->nd_[1] : lhs_->nd_[0];
  kb_align_ = trans_b_ ? rhs_->nd_[0] : rhs_->nd_[1];
  k_align_ = std::max(ka_align_, kb_align_);
  n_align_ = trans_b_ ? rhs_->nd_[1] : rhs_->nd_[0];
  if (!set_real_) {
    m_real_ = m_align_;
    k_real_ = k_align_;
    n_real_ = n_align_;
  }
  NormalizeOutput();
}

void CubeOp::NormalizeOutput() {
  size_t n = std::max(lhs_->nd_.size(), rhs_->nd_.size());
  ndd_.dims.resize(n);
  ndd_.dims[0] = n_real_;
  ndd_.dims[1] = m_real_;
  for (size_t i = 2; i < n; ++i) {
    auto dim1 = i < lhs_->nd_.size() ? lhs_->nd_[i] : 1;
    auto dim2 = i < rhs_->nd_.size() ? rhs_->nd_[i] : 1;
    // TODO: nd_[2] = dim1 >= dim2 ? dim1 : dim2;
    if (dim1 != 1 && dim2 != 1 && dim2 != dim1) {
      ASSERT(0);
    }
    ndd_.dims[i] = std::max(dim1, dim2);
  }
  shape_.Resize(n);
  for (size_t i = 0; i < ndd_.size(); ++i) {
    shape_[i] = nd_[ndd_.size() - 1 - i];
  }
}

void CubeOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<CubeOp *>(op);
  auto *lhs = self->lhs_->shape_ref_;
  auto *rhs = self->rhs_->shape_ref_;
  MatMulBatchShapeProp(lhs, rhs, self->shape_, sym_dim_next);
  self->shape_[self->shape_.size - 2] = self->trans_a_ ? lhs->data[lhs->size - 1] : lhs->data[lhs->size - 2];
  self->shape_[self->shape_.size - 1] = self->trans_b_ ? rhs->data[rhs->size - 2] : rhs->data[rhs->size - 1];
}

float CubeOp::CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0) {
  const float l2_ddr_ratio = g_system.L2DdrBandwidthRatio();
  float a_coef = l2_ddr_ratio;
  float b_coef = l2_ddr_ratio;
  auto m_loop = CeilDiv(op->m_real, m0);
  auto n_loop = CeilDiv(op->n_real, n0);
  if (m_loop == 0 || n_loop == 0) {
    return std::numeric_limits<float>::max();
  }
  auto core_need = m_loop * n_loop;
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  auto l2_num = g_system.L2Size() / ITEM_SIZE[lhs_->type_id_];
  uint32_t block_dim = core_need < core_num ? core_need : core_num;
  uint32_t m_once = block_dim < n_loop ? m0 : block_dim / n_loop * m0;

  uint32_t n_once = block_dim < n_loop ? core_num * n0 : op->n_real;
  if (m_once * op->k_real > l2_num) {
    a_coef = kDdrBandwidthRatio;
  }
  if (n_once * op->k_real > l2_num) {
    b_coef = kDdrBandwidthRatio;
  }
  // calibrate bandwidth
  a_coef = a_coef * block_dim / core_num;
  b_coef = b_coef * block_dim / core_num;
  return static_cast<float>(m_real_) * static_cast<float>(n_loop) / a_coef +
         static_cast<float>(n_real_) * static_cast<float>(m_loop) / b_coef;
}

void CubeOp::Tile(vCubeOp *op) {
  auto pri_flag = m_align_ < n_align_ ? false : true;
  auto m_round = RoundUp(static_cast<uint32_t>(m_align_), BLOCK_SIZE);
  auto n_round = RoundUp(static_cast<uint32_t>(n_align_), BLOCK_SIZE);
  auto pri_axis = pri_flag ? m_round : n_round;
  auto axis = pri_flag ? n_round : m_round;
  auto axis_max = AXES_ALIGN_SIZE / ITEM_SIZE[lhs_->type_id_];
  auto pri_axis0_max = pri_axis < axis_max ? pri_axis : axis_max;
  auto axis0_max = axis < axis_max ? axis : axis_max;
  auto l0c_num = g_system.L0CSize() / sizeof(float);
  uint32_t pri_axis0_init = BLOCK_SIZE;
  uint32_t axis0_init = BLOCK_SIZE;
  // The maximum value can be returned by cost function.
  // cost function: 1.0f / (a_coef * n0) + 1.0f / (b_coef * m0)
  // a_coef and b_coef are not less than 1 / core_num, which is 0.04
  // m0 and n0 are not less than BLOCK_SIZE, which is 16
  float min_cost = std::numeric_limits<float>::max();
  // m0, n0
  for (uint32_t pri_axis0 = pri_axis0_init; pri_axis0 <= pri_axis0_max; pri_axis0 *= 2) {
    for (uint32_t axis0 = axis0_init; axis0 <= axis0_max; axis0 *= 2) {
      if (pri_axis0 * axis0 > l0c_num) {
        break;
      }
      auto m0 = pri_flag ? pri_axis0 : axis0;
      auto n0 = pri_flag ? axis0 : pri_axis0;
      auto cost = CostFunc(op, m0, n0);
      if (cost < min_cost) {
        min_cost = cost;
        op->m0 = m0;
        op->n0 = n0;
      }
    }
  }
  // k0
  uint32_t cubeBlockSize = CUBE_BLOCK_SIZE;
  uint32_t kBlockSize = BLOCK_SIZE;
  auto bias_size = bias_ ? g_system.BtSize() : 0;
  auto l1_ping_pong_num = (g_system.L1Size() / 2 - bias_size) / ITEM_SIZE[lhs_->type_id_];
  auto k0_max = l1_ping_pong_num / (op->m0 + op->n0);
  op->k0 =
    k0_max < cubeBlockSize ? RoundDown<uint32_t>(k0_max, kBlockSize) : RoundDown<uint32_t>(k0_max, cubeBlockSize);
  if (op->k0 > CONST_512) {
    op->k0 = RoundDown(op->k0, CONST_512);
  }
  if (op->k0 > op->k_real) {
    op->k0 = op->k_real;
    if (op->k0 % BLOCK_SIZE) {
      op->k0 += BLOCK_SIZE - op->k0 % BLOCK_SIZE;
    }
  }
  m0_ = op->m0;
  n0_ = op->n0;
  k0_ = op->k0;
}

NDObject *CubeOp::Clone(CloneHelper &h) {
  return new CubeOp(h.GetClone(lhs_), h.GetClone(rhs_), trans_a_, trans_b_, bias_ ? h.GetClone(bias_) : nullptr);
}

void CubeOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "MatMul";
  if (verbose) {
    oss << "<" << trans_a_ << ", " << trans_b_ << ">";
  }
}

void CubeOp::TileV1(vCubeOp *op) {
  if (g_system.SocName() == kAscend910B4 || g_system.SocName() == kAscend910B3) {
    uint32_t swizzle_type = m_align_ < n_align_ ? V_CUBE_SWIZ_VISIT_zN : V_CUBE_SWIZ_VISIT_nZ;

    TileV2(op, swizzle_type);
    return;
  }
  Tile(op);
  op->swizzle = vCubeOp::SwizzleEncode(op->m_real > op->n_real ? V_CUBE_SWIZ_VISIT_nZ : V_CUBE_SWIZ_VISIT_zN,
                                       DEFAULT_SWIZZLE_COUNT);
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  op->m_loop = CeilDiv(m_real_, m0_);
  op->n_loop = CeilDiv(n_real_, n0_);
  core_loop_ = op->m_loop * op->n_loop * batch_c0_ * batch_c1_;
  block_dim_ = core_loop_ < core_num ? core_loop_ : core_num;
}

static uint32_t GetSwizzle(uint64_t major, uint64_t minor, uint64_t major_loop, uint64_t minor_loop, uint64_t k_real,
                           bool major_align, bool minor_align, uint32_t block_dim, float &mincost) {
  const float l2_ddr_ratio = g_system.L2DdrBandwidthRatio();
  const uint64_t cache_line = g_system.L2CacheLineSize() / ITEM_SIZE[kFloat16];
  uint32_t core_num = g_system.CoreNum(CoreType::kAIC);
  // Cost upper bound at block_dim=1 and the minimum tile size:
  // 2 / ((1 / core_num) * CubeOp::BLOCK_SIZE).
  const float kSwizzleEarlyStopCost = 2.0f * static_cast<float>(core_num) / CubeOp::BLOCK_SIZE;
  uint64_t cache_limit =
    (g_system.L2Size() / ITEM_SIZE[kFloat16] - kPerCoreL2ReserveElements[g_system.Arch()] * core_num) / k_real;
  uint64_t swizzle_cnt = 0;
  uint64_t minsize = major * major_loop + minor * minor_loop;
  for (uint64_t cnt = std::min(static_cast<uint64_t>(block_dim), major_loop); cnt >= 1; --cnt) {
    float major_hit, minor_hit;
    uint64_t width = std::min(block_dim * 2 / cnt, minor_loop);
    uint64_t major_need = major * cnt * 2;
    uint64_t minor_need = minor * width;
    if (major_align) major_need = RoundUp(major_need, cache_line);
    if (minor_align) minor_need = RoundUp(minor_need, cache_line);
    if (minor * minor_loop + major_need * 2 < cache_limit) {
      uint64_t size = major * cnt + minor * width;
      if (size >= minsize && mincost < kSwizzleEarlyStopCost) continue;
      minsize = size;
      major_hit = static_cast<float>(minor_loop - 1) / minor_loop;
      minor_hit = static_cast<float>(major_loop - 1) / major_loop;
    } else if (major_need + minor_need < cache_limit) {
      major_hit = static_cast<float>(minor_loop - 1) / minor_loop;
      minor_hit = static_cast<float>(major_loop - ((major_loop - 1) / cnt + 1)) / major_loop;
    } else {
      major_hit = static_cast<float>(width - 1) / width;
      minor_hit = static_cast<float>(cnt - 1) / cnt;
      if (major_align && major < cache_line) {
        uint64_t size = major * cnt;
        major_hit = major_hit * size / RoundUp(size, cache_line);
      }
      if (minor_align && minor < cache_line) {
        uint64_t size = minor * width;
        minor_hit = minor_hit * size / RoundUp(size, cache_line);
      }
      float k_hit = static_cast<float>(cache_limit) / (major_need + minor_need);
      major_hit *= k_hit;
      minor_hit *= k_hit;
    }
    float major_coef = l2_ddr_ratio / (major_hit + (1.0f - major_hit) * l2_ddr_ratio);
    float minor_coef = l2_ddr_ratio / (minor_hit + (1.0f - minor_hit) * l2_ddr_ratio);
    if (block_dim < core_num) {
      major_coef = major_coef * block_dim / core_num;
      minor_coef = minor_coef * block_dim / core_num;
    }
    float cost = 1.0f / (major_coef * static_cast<float>(minor)) + 1.0f / (minor_coef * static_cast<float>(major));
    // std::cout << "swizzle=(" << cnt << ", " << width << "), hit=(" << major_hit << ", " << minor_hit << "), coef=("
    // << major_coef << ", " << minor_coef << "), cost=" << cost << std::endl;
    if (cost < mincost) {
      mincost = cost;
      swizzle_cnt = cnt;
    }
  }
  return swizzle_cnt;
}

void CubeOp::TileV2(vCubeOp *op, uint32_t swizzle_type) {
  float mincost = std::numeric_limits<float>::max();
  TileHelper tile_helper(this, op);
  auto tile_select = [&](uint32_t x, uint32_t y) {
    TileHelper::TileCand candidate;
    if (!tile_helper.GetCandidate(x, y, &candidate)) return;
    auto [m0, n0, k0, m_loop, n_loop, core_loop, block_dim] = candidate;
    uint32_t swizzle_cnt = m_align_ < n_align_
                             ? GetSwizzle(n0, m0, n_loop, m_loop, k_real_, !trans_b_, trans_a_, block_dim, mincost)
                             : GetSwizzle(m0, n0, m_loop, n_loop, k_real_, trans_a_, !trans_b_, block_dim, mincost);
    if (swizzle_cnt) {
      op->m0 = m0_ = m0;
      op->n0 = n0_ = n0;
      op->k0 = k0_ = k0;
      if (swizzle_type == V_CUBE_SWIZ_VISIT_DIAGONAL_Z) {
        swizzle_cnt = DEFAULT_DIAGONAL_SWIZZLE_COUNT;
      }
      op->swizzle = vCubeOp::SwizzleEncode(swizzle_type, swizzle_cnt);
      block_dim_ = block_dim;
      core_loop_ = core_loop;
    }
  };
  block_dim_ = 0;
  ForEachCubeTilePair(kTileV2Sizes, tile_select, [&] { return block_dim_ > 0; });
}

void CubeOp::GenTiling(vCubeOp *op) {
  static int tiling_ver = -1;
  if (tiling_ver == -1) {
    const char *ver = getenv("DVM_MATMUL_TILING");
    tiling_ver = ver != nullptr ? std::stoi(ver) : (g_system.Arch() == kAiCore_C310 ? 4 : 3);
  }
  switch (tiling_ver) {
    case 1: {
      TileV1(op);
      break;
    }
    case 2: {
      uint32_t swizzle_type = m_align_ < n_align_ ? V_CUBE_SWIZ_VISIT_zN : V_CUBE_SWIZ_VISIT_nZ;
      TileV2(op, swizzle_type);
      break;
    }
    case 3: {
      TileV2(op, V_CUBE_SWIZ_VISIT_DIAGONAL_Z);
      break;
    }
    case 4: {
      TileV4(op);
      break;
    }
    default:
      break;
  }
}

void CubeOp::TileV4(vCubeOp *op) {
  const uint32_t core_num = g_system.CoreNum(CoreType::kAIC);
  const uint32_t item_size = ITEM_SIZE[lhs_->type_id_];
  const uint64_t batch = static_cast<uint64_t>(batch_c0_) * batch_c1_;
  const uint32_t round_m = RoundUp(op->m_real, BLOCK_SIZE);
  const uint32_t round_n = RoundUp(op->n_real, BLOCK_SIZE);
  const uint32_t round_k = RoundUp(op->k_real, BLOCK_SIZE);
  const uint32_t l0_elements = g_system.L0ABSize() / (2 * item_size);
  const uint64_t l0c_elements = g_system.L0CSize() / sizeof(float);
  const double bandwidth_ratio = g_system.L2DdrBandwidthRatio();
  const double shape_ratio = 1.0 / op->m_real + 1.0 / op->n_real;
  const double input_bytes =
      static_cast<double>(batch) * (static_cast<double>(op->m_real) + op->n_real) * op->k_real * item_size;
  const double cache_usage = std::max(1.0, input_bytes / g_system.L2Size());
  double cube_edge = kL2BytesPerCubeOp + cache_usage * (1 - bandwidth_ratio) * shape_ratio -
                     (1 + bandwidth_ratio) / op->k_real;
  // Use the raw bandwidth edge to choose input alignment and K reserves before searching.
  const uint32_t seed_m = std::min(round_m, 256u);
  const uint32_t seed_n = std::min<uint64_t>(round_n, RoundDown(l0c_elements / seed_m, uint64_t{BLOCK_SIZE}));
  const bool memory_bound = 1.0 / seed_m + 1.0 / seed_n > cube_edge;
  uint32_t inner_align;
  uint32_t min_l0_k;
  uint32_t k_inner_align;
  if (memory_bound) {
    inner_align = 128 / item_size;
    min_l0_k = BLOCK_SIZE;
    k_inner_align = (batch == 1 ? 256u : 512u) / item_size;
  } else {
    inner_align = 64 / item_size;
    min_l0_k = 128 / item_size;
    k_inner_align = 512 / item_size;
  }
  min_l0_k = std::min(round_k, min_l0_k);
  const uint32_t min_l1_k = std::min(round_k, !trans_a_ || trans_b_ ? k_inner_align : BLOCK_SIZE);
  const uint32_t m_align = trans_a_ ? inner_align : BLOCK_SIZE;
  // Align short-K outputs to 256 bytes for Fixpipe bandwidth.
  const uint32_t n_align = op->k_real < 1.0 / (shape_ratio * bandwidth_ratio) ? 256 / item_size :
                           (trans_b_ ? BLOCK_SIZE : inner_align);
  auto Balance = [&](const TileHelper::TileCand &tile) {
    const uint64_t tiles = tile.core_loop;
    const uint64_t main_waves = CeilDiv(tiles, uint64_t{core_num}) - 1;
    const uint64_t tail_split = core_num / (tiles - main_waves * core_num);
    const uint64_t padded_m = op->m_real <= BLOCK_SIZE ? op->m_real : tile.m0;
    const uint64_t padded_n = op->n_real <= BLOCK_SIZE ? op->n_real : tile.n0;
    double tail_wave = 1.0;
    // CANN's tail score assumes device-side subdivision of the final wave.
    // The current kernel executes whole tiles, so tail_wave < 1 overestimates
    // load balance. Matching device-side subdivision is deferred to a follow-up.
    if (main_waves && batch == 1 && padded_m * padded_n / tail_split >= MIN_TATL_BLOCK_SIZE) {
      const uint64_t split_m = std::sqrt(tail_split);
      const uint64_t split_n = tail_split / split_m;
      tail_wave = 1.0 / (split_m * split_n);
    }
    return static_cast<double>(batch) * op->m_real * op->n_real /
           (core_num * (main_waves + tail_wave) * padded_m * padded_n);
  };
  TileHelper helper(this, op);
  TileHelper::TileCand best;
  double best_ratio = std::numeric_limits<double>::max();
  double best_balance = 0;
  double best_score = std::numeric_limits<double>::max();
  cube_edge *= CUBE_BOUND_RATIO;
  auto tile_select = [&](uint32_t m, uint32_t n) {
    TileHelper::TileCand candidate;
    if (!helper.GetCandidateMN(m, n, &candidate)) return;
    if ((m % m_align != 0 && m != round_m) || (n % n_align != 0 && n != round_n)) return;
    const uint32_t max_mn = std::max(m, n);
    if (max_mn * min_l0_k > l0_elements ||
        static_cast<uint64_t>(max_mn) * min_l1_k * item_size * 4 > g_system.L1Size()) return;
    const double ratio = 1.0 / candidate.m0 + 1.0 / candidate.n0;
    if (best_balance >= 0.9 && ratio > best_ratio && ratio > cube_edge && cube_edge > 0) return;
    const double balance = Balance(candidate);
    const bool cube_better = ratio <= cube_edge && balance > best_balance;
    const double score = ratio / balance;
    if (cube_better || score < best_score || (std::abs(score - best_score) < 1e-9 && balance > best_balance)) {
      if (cube_better) cube_edge = ratio;
      best = candidate;
      best_ratio = ratio;
      best_balance = balance;
      best_score = score;
    }
  };
  ForEachCubeTilePair(kTileV4Sizes, tile_select, [] { return false; });
  EXCEPTION_IF(best.block_dim == 0, "TileV4 cannot encode MatMul tile");
  op->m0 = m0_ = best.m0;
  op->n0 = n0_ = best.n0;
  op->k0 = k0_ = best.k0;
  core_loop_ = best.core_loop;
  block_dim_ = best.block_dim;
  // Use the existing nZ walk with CANN's four-tile window.
  const uint32_t window = std::min(4u, CeilDiv(op->m_real, op->m0));
  op->swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_nZ, window);
}

void CubeOp::CodeGen(vCubeOp *op, CubeTuner *tuner) {
  op->m_align = m_align_;
  op->n_align = n_align_;
  op->k_align = k_align_;
  op->ka_align = ka_align_;
  op->kb_align = kb_align_;
  op->m_real = m_real_;
  op->n_real = n_real_;
  op->k_real = k_real_;
  op->offset_a = offset_a_;
  op->offset_b = offset_b_;
  auto a = static_cast<NDAccess *>(lhs_);
  op->gm_a = a->addr_.data;
  uint32_t batch_a1 = a->nd_.size() > 2 ? static_cast<uint32_t>(a->nd_[2]) : 1;
  uint32_t batch_a0 = a->nd_.size() > 3 ? static_cast<uint32_t>(a->nd_[3]) : 1;
  auto b = static_cast<NDAccess *>(rhs_);
  op->gm_b = b->addr_.data;
  uint32_t batch_b1 = b->nd_.size() > 2 ? static_cast<uint32_t>(b->nd_[2]) : 1;
  uint32_t batch_b0 = b->nd_.size() > 3 ? static_cast<uint32_t>(b->nd_[3]) : 1;
  if (batch_fold_) {
    auto batch_fold = batch_a1 * batch_a0;
    batch_a1 = 1;
    batch_a0 = 1;
    op->m_align = m_align_ *= batch_fold;
    op->m_real = m_real_ *= batch_fold;
  }
  batch_c0_ = std::max(batch_a0, batch_b0);
  batch_c1_ = std::max(batch_a1, batch_b1);
  op->batch_cast = 0;
  if (batch_c0_ != batch_a0) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_A0;
  if (batch_c0_ != batch_b0) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_B0;
  if (batch_c1_ != batch_a1) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_A1;
  if (batch_c1_ != batch_b1) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_B1;
  if (op->batch_cast) op->batch_cast |= batch_c1_ << V_CUBE_BCAST_C1_OFFSET;
  auto c = static_cast<NDAccess *>(output_);
  op->gm_c = c->addr_.data;
  op->flags = trans_a_ ? V_CUBE_FLAG_TRANS_A : 0;
  if (trans_b_) op->flags |= V_CUBE_FLAG_TRANS_B;
  if (lhs_->CheckFlag(OBJ_FLAG_LOAD_NZ)) op->flags |= V_CUBE_FLAG_INPUT_A_NZ;
  if (rhs_->CheckFlag(OBJ_FLAG_LOAD_NZ)) op->flags |= V_CUBE_FLAG_INPUT_B_NZ;
  if (type_id_ == dvm::kFloat32) op->flags |= V_CUBE_FLAG_OUT_FP32;
  if (atomic_add_) op->flags |= V_CUBE_FLAG_ATOMIC_ADD;
  if (bias_) {
    op->flags |= V_CUBE_FLAG_WITH_BIAS;
    op->flags |= V_CUBE_FLAG_BIAS_FP32 * (bias_->type_id_ == kFloat32);
    op->gm_bias = static_cast<NDAccess *>(bias_)->addr_.data;
  }
  ASSERT(lhs_->type_id_ == dvm::kFloat16 || lhs_->type_id_ == dvm::kBFloat16);
  uint64_t dtype = lhs_->type_id_ == dvm::kFloat16 ? vCubeOp::FP16 : vCubeOp::BF16;
  op->flags |= dtype << V_CUBE_FLAG_DTYPE_OFFSET;
  if (tuner) {
    tuner->GenTile(this, op);
  } else {
    GenTiling(op);
  }
  op->m_loop = CeilDiv(op->m_real, op->m0);
  op->n_loop = CeilDiv(op->n_real, op->n0);
  op->k_loop = CeilDiv(op->k_real, op->k0);
  op->group_num = core_loop_;
}

GmmOp::GmmOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias, NDObject *group_list,
             GmmSplitType group_type, GmmListType group_list_type)
    : CubeOp(lhs, rhs, trans_a, trans_b, bias),
      group_list_(group_list),
      group_type_(group_type),
      group_list_type_(group_list_type) {
  obj_id_ = kGmmOp;
}

NDObject *GmmOp::Clone(CloneHelper &h) {
  auto bias = bias_ ? h.GetClone(bias_) : nullptr;
  return new GmmOp(h.GetClone(lhs_), h.GetClone(rhs_), trans_a_, trans_b_, bias, group_list_, group_type_,
                   group_list_type_);
}

void GmmOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "GroupedMatMul";
  if (verbose) {
    oss << "<" << trans_a_ << ", " << trans_b_ << ", group_type=" << static_cast<int>(group_type_)
        << ", group_list_type=" << static_cast<int>(group_list_type_) << ">";
  }
}

void GmmOp::NormalizeOutput() {
  if (group_type_ == kSplit_M) {
    ASSERT(rhs_->shape_ref_->size == 3);
    ASSERT(group_list_->shape_ref_->data[0] == rhs_->shape_ref_->data[0]);
    ASSERT(bias_ == nullptr || bias_->shape_ref_->data[0] == rhs_->shape_ref_->data[0]);
    ndd_.dims.resize(2);
    ndd_.dims[0] = n_real_;
    ndd_.dims[1] = m_real_;
  }
  if (group_type_ == kSplit_K) {
    ndd_.dims.resize(3);
    ndd_.dims[0] = n_real_;
    ndd_.dims[1] = m_real_;
    ndd_.dims[2] = group_list_->shape_ref_->data[0];
  }
  shape_.Resize(ndd_.dims.size());
  for (size_t i = 0; i < ndd_.size(); ++i) {
    shape_[i] = ndd_[ndd_.size() - 1 - i];
  }
}

void GmmOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<GmmOp *>(op);
  auto *lhs = self->lhs_->shape_ref_;
  auto *rhs = self->rhs_->shape_ref_;
  auto m_dim = self->trans_a_ ? lhs->data[lhs->size - 1] : lhs->data[lhs->size - 2];
  auto n_dim = self->trans_b_ ? rhs->data[rhs->size - 2] : rhs->data[rhs->size - 1];
  if (self->group_type_ == kSplit_M) {
    self->shape_.Resize(2);
    self->shape_[0] = m_dim;
    self->shape_[1] = n_dim;
    return;
  }
  if (self->group_type_ == kSplit_K) {
    self->shape_.Resize(3);
    self->shape_[0] = self->group_list_->shape_ref_->data[0];
    self->shape_[1] = m_dim;
    self->shape_[2] = n_dim;
    return;
  }
  CubeOp::ShapeProp(op, sym_dim_next);
}

void GmmOp::GenTiling(vCubeOp *op) {
  Tile(op);
  op->swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_Z, DEFAULT_DIAGONAL_SWIZZLE_COUNT);
}

void GmmOp::CodeGen(vCubeOp *op, CubeTuner *tuner) {
  CubeOp::CodeGen(op, nullptr);
  batch_c0_ = 1;
  batch_c1_ = 1;
  op->batch_cast = 0;
  op->flags |= V_CUBE_FLAG_GROUPED_LIST;
  if (group_type_ == kSplit_K) {
    op->flags |= V_CUBE_FLAG_GROUP_K;
  }
  if (group_list_type_ == GmmListType::kListSize) {
    op->flags |= V_CUBE_FLAG_GMM_SIZE_MODE;
  }
  op->gm_group_list = static_cast<NDAccess *>(group_list_)->addr_.data;
  op->group_list_size = group_list_->shape_ref_->data[0];

  block_dim_ = g_system.CoreNum(CoreType::kAIC);
  if (group_type_ == kSplit_K) {
    core_loop_ = op->m_loop * op->n_loop * op->group_list_size;
  }
  if (group_type_ == kSplit_M) {
    core_loop_ = block_dim_;
  }
  op->group_num = core_loop_;
}

void GmmOp::InferTactics(Tactics &t) const {
  CubeOp::InferTactics(t);
  if (group_type_ == kSplit_K) {
    t.enable_splitk = false;
  }
}

constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;

class ManualCubeTuner : public CubeTuner {
 public:
  explicit ManualCubeTuner(const TuningInfo &info) : CubeTuner(kUnknownTuner), info_(info) {}
  void GenTile(CubeOp *op, vCubeOp *code) override {
    op->m0_ = code->m0 = info_.m0;
    op->n0_ = code->n0 = info_.n0;
    op->k0_ = code->k0 = info_.k0;
    code->swizzle = info_.swizzle;
    op->core_loop_ = info_.core_loop;
    op->block_dim_ = info_.block_dim;
  }

 private:
  const TuningInfo &info_;
};

CubeTuner::~CubeTuner() {}

void OnlineCubeTuner::GenTile(CubeOp *op, vCubeOp *code) {
  auto &tuning_table = CubeTuner::CacheTable();
  TuningInfo &best_tuning = tuning_table[GenKey(op, code)];
  if (best_tuning.swizzle == 0) {
    void *dev_M_, *dev_N_, *dev_O_;
    ERROR_CHECK(rt::Malloc(&dev_M_, op->lhs_->Size() + 512));
    ERROR_CHECK(rt::Malloc(&dev_N_, op->rhs_->Size() + 512));
    ERROR_CHECK(rt::Malloc(&dev_O_, op->Size() + 512));
    TuneData td;
    td.kernel.Reset(KernelType::kCube, 0);
    auto m_input = td.kernel.Load(dev_M_, op->lhs_->shape_ref_, DataType::kFloat16);
    auto n_input = td.kernel.Load(dev_N_, op->rhs_->shape_ref_, DataType::kFloat16);
    auto matmul = new CubeOp(m_input, n_input, op->trans_a_, op->trans_b_);
    if (op->type_id_ == dvm::kFloat32) matmul->SetOutFp32(false);
    td.kernel.GetImpl()->Append(matmul);
    matmul->SetRealShape(op->nd_[1], op->nd_[0], op->k_real_, 0, 0);  // in batch fold m_real != op->nd_[1]
    (void)td.kernel.Store(dev_O_, matmul);
    TileV3(td, op, code);
    best_tuning = td.best_para;
    rt::Free(dev_M_);
    rt::Free(dev_N_);
    rt::Free(dev_O_);
  }
  code->m0 = op->m0_ = best_tuning.m0;
  code->n0 = op->n0_ = best_tuning.n0;
  code->k0 = op->k0_ = best_tuning.k0;
  code->swizzle = best_tuning.swizzle;
  op->core_loop_ = best_tuning.core_loop;
  op->block_dim_ = best_tuning.block_dim;
}

void OnlineCubeTuner::TileV3(TuneData &td, CubeOp *mm, vCubeOp *op) {
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  uint32_t block_dim = 0;
  TileHelper tile_helper(mm, op);
  auto tile_select = [&](uint32_t x, uint32_t y) {
    TileHelper::TileCand candidate;
    if (!tile_helper.GetCandidate(x, y, &candidate)) return;
    auto [m0, n0, k0, m_loop, n_loop, core_loop, candidate_block_dim] = candidate;
    block_dim = candidate_block_dim;
    // 3. select swizzle
    for (uint32_t cnt = std::min(block_dim, m_loop); cnt >= 1; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_nZ, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min(block_dim, n_loop); cnt >= 1; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_zN, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min({n_loop, m_loop, block_dim}); cnt >= V_CUBE_DIAGONAL_MIN_DIM; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_N, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
      swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_Z, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
  };
  ForEachCubeTilePair(kTileV2Sizes, tile_select, [&] { return block_dim == core_num; });
}

void OnlineCubeTuner::Tuning(TuneData &td, const TuningInfo &parameter) {
  rt::Stream stream;
  ERROR_CHECK(rt::CreateStream(&stream));
  ManualCubeTuner tuner(parameter);
  static_cast<MixKernel *>(td.kernel.GetImpl())->SetTuner(&tuner);
  td.kernel.CodeGen();
  RepeatProfiler profiler;
  profiler.Reset();
  uint32_t test_num = 10;
  for (uint32_t i = 0; i < test_num; i++) {
    profiler.RecordStart(stream);
    ERROR_CHECK(td.kernel.Launch(nullptr, 0, nullptr, stream));
    profiler.RecordEnd(stream);
  }
  ERROR_CHECK(rt::DestroyStream(stream));
  auto mean_time = (profiler.total_us_ - profiler.min_us_ - profiler.max_us_) / (test_num - 2);
  if (mean_time < td.best_time) {
    td.best_time = mean_time;
    td.best_para = parameter;
  }
}

LazyCubeTuner::~LazyCubeTuner() {
  for (auto it = context_.begin(); it != context_.end(); ++it) {
    delete it->second;
  }
  for (auto w = wrap_->next_w_; w != wrap_;) {
    auto next = w->next_w_;
    delete w;
    w = next;
  }
  delete wrap_;
}

void LazyCubeTuner::GenTile(CubeOp *op, vCubeOp *code) {
  constexpr size_t kCacheTableMax = 2000;
  auto &tuning_table = CubeTuner::CacheTable();
  auto key = GenKey(op, code);
  TuningInfo *info;
  uint32_t space_idx = (uint32_t)-1;
  auto it = tuning_table.find(key);
  if (it != tuning_table.end()) {
    info = &it->second;
  } else if (tuning_table.size() < kCacheTableMax) {
    Context *ctx;
    auto ctx_it = context_.find(key);
    if (ctx_it == context_.end()) {
      ctx = new Context();
      BuildTileSpace(op, code, ctx->tile_space);
      context_[key] = ctx;
    } else {
      ctx = ctx_it->second;
    }
    auto &current_space = ctx->tuning_stage == kTileTuning ? ctx->tile_space : ctx->swizzle_space;
    int space_size = current_space.size();
    constexpr int repeat = 2;
    if (ctx->gen_cnt == space_size * repeat) {
      if (ctx->gen_cnt != ctx->run_cnt) {
        // means some tiling configure in the current generated tiling space have not been launched,
        // wait launch completed
        std::unique_lock<std::mutex> lock(ctx->mutex_);
        ctx->cond_var_.wait(lock, [ctx] { return ctx->gen_cnt == ctx->run_cnt; });
      }
      int cur_idx = ctx->best_idx >= 0 ? (ctx->best_idx % space_size) : 0;
      if (ctx->tuning_stage == kSwizzleTuning) {  // tuning finished
        auto it = tuning_table.emplace(key, *ctx->swizzle_space[cur_idx]);
        info = &it.first->second;
        context_.erase(key);
        delete ctx;
      } else {  // tile_space finished, switch to swizzle_space
        BuildSwizzleSpace(code, ctx->tile_space[cur_idx], ctx->swizzle_space);
        ctx->tuning_stage = kSwizzleTuning;
        ctx->run_cnt = 0;
        ctx->best_idx = -1;
        ctx->next_idx = 1;
        ctx->gen_cnt = 1;
        space_idx = 0;
        info = ctx->swizzle_space[space_idx];
      }
    } else {
      space_idx = ctx->next_idx;
      info = current_space[space_idx];
      ctx->next_idx = (space_idx + 1) % space_size;
      ctx->gen_cnt++;
    }
    if (space_idx != (uint32_t)-1) {
      auto wrap = wrap_->next_w_;
      if (wrap->ctx_) {
        wrap = new _TunerWarp();
        wrap->next_w_ = wrap_->next_w_;
        wrap_->next_w_ = wrap;
      }
      wrap->ctx_ = ctx;
      wrap_ = wrap;
      ASSERT(cur_code_ != nullptr);
      cur_code_->InsertWrap(wrap);
    }
  } else {
    op->GenTiling(code);
    return;
  }
  code->unique_id = space_idx;  // reuse
  code->m0 = op->m0_ = info->m0;
  code->n0 = op->n0_ = info->n0;
  code->k0 = op->k0_ = info->k0;
  code->swizzle = info->swizzle;
  op->core_loop_ = info->core_loop;
  op->block_dim_ = info->block_dim;
}

void LazyCubeTuner::BuildTileSpace(CubeOp *op, vCubeOp *code, std::vector<TuningInfo *> &space) {
  auto l0c_max = g_system.L0CSize() / FP32_SIZE;
  auto bias_size = op->bias_ ? g_system.BtSize() : 0;
  auto l1_max = (g_system.L1Size() / 2 - bias_size) / ITEM_SIZE[op->lhs_->type_id_];
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  uint32_t round_m = RoundUp<uint32_t>(op->m_real_, BLOCK_SIZE);
  uint32_t round_n = RoundUp<uint32_t>(op->n_real_, BLOCK_SIZE);
  uint32_t round_k = RoundUp<uint32_t>(op->k_real_, BLOCK_SIZE);
  uint32_t n_align_max = op->bias_ != nullptr ? g_system.BtSize() / sizeof(float) : MATMUL_ALIGN_MAX;
  uint32_t block_dim = 0;
  auto tile_select = [&](uint32_t m0, uint32_t n0) {
    if (m0 > round_m || n0 > round_n || n0 > n_align_max) return;
    uint32_t kx = l1_max / (m0 + n0);
    uint32_t k0 = RoundDown<uint32_t>(kx, kx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
    if (k0 == 0) return;
    k0 = std::min({k0, MATMUL_ALIGN_MAX, round_k});
    if (m0 * n0 > l0c_max || n0 * k0 + m0 * k0 > l1_max) return;
    uint32_t m_loop = CeilDiv(code->m_real, m0);
    uint32_t n_loop = CeilDiv(code->n_real, n0);
    uint32_t core_loop = m_loop * n_loop * op->batch_c0_ * op->batch_c1_;
    block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    uint32_t cnt = 7;
    space.push_back(
      new TuningInfo(m0, n0, k0, vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_nZ, cnt), core_loop, block_dim));
    space.push_back(
      new TuningInfo(m0, n0, k0, vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_zN, cnt), core_loop, block_dim));
  };
  ForEachCubeTilePair(kTileV2Sizes, tile_select, [&] { return block_dim == core_num; });
}

void LazyCubeTuner::BuildSwizzleSpace(vCubeOp *code, TuningInfo *best_tile, std::vector<TuningInfo *> &space) {
  uint32_t m_loop = CeilDiv(code->m_real, (uint32_t)best_tile->m0);
  uint32_t n_loop = CeilDiv(code->n_real, (uint32_t)best_tile->n0);
  for (uint32_t cnt = std::min(best_tile->block_dim, m_loop); cnt >= 1; --cnt) {
    space.push_back(
      new TuningInfo(best_tile->m0, best_tile->n0, best_tile->k0, cnt, best_tile->core_loop, best_tile->block_dim));
  }
  for (uint32_t cnt = std::min(best_tile->block_dim, n_loop); cnt >= 1; --cnt) {
    space.push_back(new TuningInfo(best_tile->m0, best_tile->n0, best_tile->k0, 1u << 16 | cnt, best_tile->core_loop,
                                   best_tile->block_dim));
  }
}

int LazyCubeTuner::_TunerWarp::LaunchWrap(void *workspace, void *stream) {
  TimeProfiler profiler;
  profiler.RecordStart(stream);
  auto err = next_->LaunchWrap(workspace, stream);
  float time = profiler.RecordEnd(stream);
  auto ctx = const_cast<Context *>(ctx_);
  if (!err && (ctx->best_idx < 0 || time < ctx->best_time)) {
    ctx->best_idx = ctx->run_cnt;
    ctx->best_time = time;
  }
  ctx->run_cnt++;
  if (ctx->run_cnt == ctx->gen_cnt) {
    std::unique_lock<std::mutex> lock(ctx->mutex_);
    ctx->cond_var_.notify_all();
  }
  ctx_ = nullptr;
  return err;
}
}  // namespace dvm
