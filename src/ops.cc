/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include "ops.h"
#include "ops_m.h"
#include "kernel.h"
#include "comm.h"

namespace dvm {
namespace {
struct InsnIdTable {
  const char *name;
  vSimdInsnID ids[DataType::kDataTypeEnd];
};

static const InsnIdTable unary_id_list[kUnaryTypeEnd] = {
  // must keep consistent order with UnaryType
  {"Sqrt", {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE, V_NONE}},
  {"Abs", {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_ABS_INT32, V_NONE}},
  {"Log", {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE, V_NONE}},
  {"Exp", {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE, V_NONE}},
  {"Reciprocal", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"IsFinite", {V_NONE, V_ISFINITE_FP16, V_ISFINITE_BF16, V_ISFINITE, V_NONE, V_NONE}},
  {"LogicalNot", {V_LOGICAL_NOT_BOOL, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"Round", {V_NONE, V_NONE, V_NONE, V_ROUND, V_NONE, V_NONE}},
  {"Floor", {V_NONE, V_NONE, V_NONE, V_FLOOR, V_NONE, V_NONE}},
  {"Ceil", {V_NONE, V_NONE, V_NONE, V_CEIL, V_NONE, V_NONE}},
  {"Trunc", {V_NONE, V_NONE, V_NONE, V_TRUNC, V_NONE, V_NONE}}};

static const InsnIdTable binary_id_list[] = {
  // must keep consistent order with BinaryType
  {"Equal", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"NotEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Greater", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"GreaterEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Less", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"LessEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Add", {V_NONE, V_ADD_FP16, V_ADD_BF16, V_ADD, V_ADD_INT32, V_ADD_INT64}},
  {"Sub", {V_NONE, V_SUB_FP16, V_SUB_BF16, V_SUB, V_SUB_INT32, V_SUB_INT64}},
  {"Mul", {V_NONE, V_MUL_FP16, V_MUL_BF16, V_MUL, V_MUL_INT32, V_NONE}},
  {"Div", {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE, V_NONE}},
  {"Pow", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},  // power: individual implement
  {"Maximum", {V_NONE, V_MAX_FP16, V_MAX_BF16, V_MAX, V_MAX_INT32, V_NONE}},
  {"Minimum", {V_NONE, V_MIN_FP16, V_MIN_BF16, V_MIN, V_MIN_INT32, V_NONE}},
  {"LogicalAnd", {V_LOGICAL_AND_BOOL, V_MIN_FP16, V_MAX_BF16, V_MIN, V_MIN_INT32, V_NONE}},
  {"LogicalOr", {V_LOGICAL_OR_BOOL, V_MAX_FP16, V_MIN_BF16, V_MAX, V_MAX_INT32, V_NONE}}};

static const InsnIdTable binarys_id_list[] = {
  // must keep consistent order with BinarySOpType
  {"Equal", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"NotEqual", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"Greater", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"GreaterEqual", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"Less", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"LessEqual", {V_CMPS_BOOL, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32, V_NONE}},
  {"Add", {V_NONE, V_ADDS_FP16, V_ADDS_BF16, V_ADDS, V_ADDS_INT32, V_NONE}},
  {"Mul", {V_NONE, V_MULS_FP16, V_MULS_BF16, V_MULS, V_MULS_INT32, V_NONE}},
  {"Div", {V_NONE, V_DIVS_FP16, V_NONE, V_DIVS, V_NONE, V_NONE}},
  {"ScalarDiv", {V_NONE, V_SDIV_FP16, V_NONE, V_SDIV, V_NONE, V_NONE}},
  {"Maximum", {V_NONE, V_MAXS_FP16, V_MAXS_BF16, V_MAXS, V_MAXS_INT32, V_NONE}},
  {"Minimum", {V_NONE, V_MINS_FP16, V_MINS_BF16, V_MINS, V_MINS_INT32, V_NONE}}};

static const vSimdInsnID cast_id_list[DataType::kDataTypeEnd][DataType::kDataTypeEnd] = {
  {V_NONE, V_CAST_BOOL_TO_FP16, V_NONE, V_NONE, V_NONE, V_NONE},                                           // V_BOOL
  {V_CAST_FP16_TO_BOOL, V_NONE, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32, V_NONE},                // V_FLOAT16
  {V_NONE, V_NONE, V_NONE, V_CAST_BF16_TO_FP32, V_CAST_BF16_TO_INT32, V_NONE},                             // V_BFLOAT16
  {V_NONE, V_CAST_FP32_TO_FP16, V_CAST_FP32_TO_BF16, V_NONE, V_CAST_FP32_TO_INT32, V_CAST_FP32_TO_INT64},  // V_FLOAT32
  {V_NONE, V_CAST_INT32_TO_FP16, V_NONE, V_CAST_INT32_TO_FP32, V_NONE, V_CAST_INT32_TO_INT64},             // V_INT32
  {V_NONE, V_NONE, V_NONE, V_CAST_INT64_TO_FP32, V_CAST_INT64_TO_INT32, V_NONE},                           // V_INT64
};

static const vSimdInsnID reduce_x_list[][ReduceType::kReduceTypeEnd] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_X_FP16, V_RMIN_X_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_X, V_RMAX_X, V_RMIN_X},          // V_FLOAT32
};

static const vSimdInsnID reduce_y_list[][ReduceType::kReduceTypeEnd] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_Y_FP16, V_RMIN_Y_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_Y, V_RMAX_Y, V_RMIN_Y},          // V_FLOAT32
};

static const scode_t reduce_init_value[][ReduceType::kReduceTypeEnd] = {
  {0, 0, 0},                    // V_BOOL
  {0, 0xFC00, 0x7C00},          // V_FLOAT16
  {0, 0, 0},                    // V_BFLOAT16
  {0, 0xFF800000, 0x7F800000},  // V_FLOAT32
};

static const vSimdInsnID removepad_id_list[SIMD_DTYPE_END] = {V_NONE, V_REMOVEPAD_U16, V_REMOVEPAD_U16, V_REMOVEPAD, V_REMOVEPAD};

uint64_t EmitCopy(bcodeptr_t insn, uint64_t xd, uint64_t xn, uint64_t bytes) {
  if (xd == xn) {
    return vNop::Encode(insn);
  }
  vCopy op;
  op.xd = xd;
  op.xn = xn;
  op.lenburst = CeilDiv(bytes, 32UL);
  return vCopy::Encode(insn, V_COPY, op);
}

inline ReduceOp *GetStoreReduceOp(NDObject *lhs) {
  if (lhs->obj_id_ == kReduce) {
    return static_cast<ReduceOp *>(lhs);
  }
  if (lhs->obj_id_ == kRemovePad && lhs->lhs_->obj_id_ == kReduce) {
    return static_cast<ReduceOp *>(lhs->lhs_);
  }
  return nullptr;
}

inline uint64_t GetAtomicStoreType(DataType type_id) {
  if (type_id == kFloat32) {
    return V_ATOMIC_FP32;
  } else if (type_id == kBFloat16) {
    return V_ATOMIC_BF16;
  } else {
    ASSERT(type_id == kFloat16);
    return V_ATOMIC_FP16;
  }
}

uint64_t SelectSimdWidth(uint64_t iter_size, DataType type_id) {
  uint64_t block_width = ITEM_SIMD_WIDTH_MAX[type_id] >> 3;
  ASSERT(iter_size % block_width == 0);
  uint64_t iter_block = iter_size / block_width;
  if (iter_block <= 8) return iter_size;
  uint64_t simd_block = 8;
  uint64_t repeat = iter_block / simd_block;
  while (repeat * simd_block != iter_block) {
    repeat = iter_block / (--simd_block);
  }
  return simd_block * block_width;
}

uint64_t EmitClearPad(uint64_t *pc, NDObject *op, uint64_t iter_tail, uint64_t simd_width, uint64_t scalar = 0) {
  const static vSimdInsnID id_list[SIMD_DTYPE_END] = {V_NONE, V_CLR_PAD_B16, V_CLR_PAD_B16, V_CLR_PAD, V_CLR_PAD};
  vClearPad clr_op;
  clr_op.xd = op->xbuf_;
  clr_op.iter_size = op->nd_.lead_dim();
  clr_op.iter_stride = op->nd_.lead_stride();
  clr_op.iter_num = op->nd_.stride_back() / clr_op.iter_stride;
  clr_op.simd_width = simd_width;
  clr_op.iter_tail = iter_tail;
  clr_op.scalar = scalar;
  return vClearPad::Encode(pc, id_list[op->type_id_], clr_op);
}

void UpdateRoundTile(bool pointwise, int64_t num, DimArray &round_tile) {
  if (!pointwise) {
    if (round_tile.size() % 2 == 0) {
      round_tile.push_back(num);
    } else {
      round_tile.back() *= num;
    }
  } else {
    if (!round_tile.empty()) {
      if (round_tile.size() % 2 == 0) {
        round_tile.back() *= num;
      } else {
        round_tile.push_back(num);
      }
    }
  }
}

scode_t GetRedInitScalarEncode(int red_op, DataType op_type) {
  ASSERT(op_type == kFloat32 || op_type == kFloat16);
  return reduce_init_value[op_type][red_op];
}

float GetRedInitScalar(int red_op) {
  if (red_op == V_RED_SUM) {
    return 0;
  } else if (red_op == V_RED_MAX) {
    return -std::numeric_limits<float>::infinity();
  } else if (red_op == V_RED_MIN) {
    return std::numeric_limits<float>::infinity();
  }
  return 0;
}

uint64_t GetDynXBufBegin(const VectorKernel &k, NDObject *cur) {
  uint64_t xnum = 1;
  for (size_t i = 0; i < k.static_ops_.size(); ++i) {
    auto op = i < k.load_num_ ? k.static_ops_[i] : k.static_ops_[i]->lhs_;
    if (op->index_ > cur->index_) {
      xnum++;
    }
  }
  return cur->xbuf_ + xnum * k.tile_size_ * ITEM_SIZE[k.MaxType()];
}
}  // namespace

bool CollectRoundTile(const DimArray &nd, const TileParam &tp, DimArray &round_tile) {
  bool pointwise = false;
  if (tp.num > 1) {
    for (int i = tp.start; i <= tp.end; ++i) {
      if (nd[i] != 1) {
        pointwise = true;
        break;
      }
    }
    UpdateRoundTile(pointwise, tp.num, round_tile);
  }
  return pointwise;
}

void BuildDimRounds(const DimArray &round_tile, uint64_t rounds[]) {
  switch (round_tile.size()) {
    case 1: {
      auto r1 = round_tile[0];
      rounds[0] = r1;
      break;
    }
    case 2: {
      auto r1 = round_tile[0] * round_tile[1];
      auto r2 = round_tile[1];
      rounds[0] = r2 << 32 | r1;
      break;
    }
    case 3: {
      auto r1 = round_tile[0] * round_tile[1] * round_tile[2];
      auto r2 = round_tile[1];
      auto r3 = round_tile[2];
      rounds[0] = r2 << 32 | r1;
      rounds[1] = r3;
      break;
    }
    case 4: {
      auto r1 = round_tile[0] * round_tile[1] * round_tile[2] * round_tile[3];
      auto r2 = round_tile[1];
      auto r3 = round_tile[2] * round_tile[3];
      auto r4 = round_tile[3];
      rounds[0] = r2 << 32 | r1;
      rounds[1] = r4 << 32 | r3;
      break;
    }
    default:
      EXCEPTION_IF(true, "multi-broadcast rank exceed max limit(4)");
      break;
  }
}

void BuildRoundTile(const VectorKernel &k, const NDSpaceData &ndd, bool sharded_round, DimArray &round_tile) {
  round_tile.resize(0);
  int dim_size = static_cast<int>(k.DimSpace().size());
  if (sharded_round) {
    if (auto *shard = k.shard_; shard->sink) {
      for (int i = dim_size - 1; i > shard->base + 1; --i) {
        if (auto factor = shard->dom->operator[](i); factor > 1) {
          UpdateRoundTile(ndd.PointwiseTile(i), factor, round_tile);
        }
      }
    }
  } else {
    const auto &tr = k.tile_region_;
    if (tr.depth == 0) {
      return;
    }
    int cur_end = dim_size - 1;
    for (int i = 0; i < tr.depth - 1; ++i) {
      int start = tr.starts[i];
      int end = cur_end;
      if (int64_t num = tr.spaces[i]; num > 1) {
        UpdateRoundTile(ndd.PointwiseTile(start, end), num, round_tile);
      }
      cur_end = start - 1;
    }
    int start = tr.starts[tr.depth - 1];
    int end = cur_end;
    if (int64_t num = tr.last_num; num > 1) {
      UpdateRoundTile(ndd.PointwiseTile(start, end), num, round_tile);
    }
    // lead_fold has num=1, no contribution to round_tile
  }
}

template <int AFFINE>
void BroadReduceFoldProp(const DimArray &small_dim, const DimArray &big_dim, PropRange &range) {
  int state = 0;  // -1 - broadcast/reduce; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  int range_base = range.base;
  if (int small_size = small_dim.size(); range_base >= small_size) {
    if (!small_size) {
      return;
    }
    for (int i = small_size; i < std::min<int>(range_base + 1, big_dim.size()); ++i) {
      if (big_dim[i] != 1) {
        state = -1;
        break;
      }
    }
    range_base = small_size - 1;
    new_depth = range.base - range_base;
  }
  for (int i = range_base; i > range.base - range.depth; --i) {
    if ((state == -1 && small_dim[i] != 1) || (state == 1 && small_dim[i] != big_dim[i])) {
      break;
    }
    if (state == 0) {
      if (small_dim[i] != 1)
        state = 1;
      else if (small_dim[i] != big_dim[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < AFFINE) {
    range.affine = AFFINE;
  }
  range.depth = new_depth;
}
template void BroadReduceFoldProp<PropRange::BROADCAST>(const DimArray &, const DimArray &, PropRange &);
template void BroadReduceFoldProp<PropRange::REDUCE>(const DimArray &, const DimArray &, PropRange &);

template <int AFFINE>
void BroadReduceTileCollect(const DimArray &small_dim, const DimArray &big_dim, TileInfo &info) {
  int state = 0;  // -1 - broadcast/reduce; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  int depth_size = std::min<int>(info.lead_depth, small_dim.size());
  for (int i = 0; i < depth_size; ++i) {
    if ((state == -1 && small_dim[i] != 1) || (state == 1 && small_dim[i] != big_dim[i])) {
      break;
    }
    if (state == 0) {
      if (small_dim[i] != 1)
        state = 1;
      else if (small_dim[i] != big_dim[i])
        state = -1;
    }
    new_depth++;
  }
  if (state) {
    if (state == -1 && info.lead_affine < AFFINE) {
      info.lead_affine = AFFINE;
    }
    info.lead_depth = new_depth;
  }
}
template void BroadReduceTileCollect<PropRange::BROADCAST>(const DimArray &, const DimArray &, TileInfo &);
template void BroadReduceTileCollect<PropRange::REDUCE>(const DimArray &, const DimArray &, TileInfo &);

vSimdInsnID GetBinaryInsnID(BinaryType op, DataType dtype) { return binary_id_list[op].ids[dtype]; }
vSimdInsnID GetCastInsnID(DataType from, DataType to) { return cast_id_list[from][to]; }

std::ostream &operator<<(std::ostream &oss, const IntArrayRef &shape) {
  oss << "[";
  for (size_t i = 0; i < shape.size; i++) {
    if (i) {
      oss << ",";
    }
    oss << shape.data[i];
  }
  oss << "]";
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const Float16 &scalar) {
  oss << static_cast<float>(scalar);
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const BFloat16 &scalar) {
  oss << static_cast<float>(scalar);
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const DimArray &nd) {
  oss << "[";
  if (nd.size() > 0) {
    for (size_t i = 0; i < nd.size() - 1; ++i) {
      oss << nd[i] << ",";
    }
    oss << nd.back();
  }
  oss << "]";
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const NDSpace &nd) {
  oss << nd.dims();
  return oss;
}

void DumpScalarCode(std::ostringstream &oss, scode_t code, DataType type) {
  switch (type) {
    case kFloat16: {
      oss << Float16(static_cast<uint16_t>(code));
      break;
    }
    case kBFloat16: {
      oss << BFloat16(static_cast<uint16_t>(code));
      break;
    }
    case kFloat32: {
      union {
        float val;
        uint32_t code;
      } data;
      data.code = code;
      oss << data.val;
      break;
    }
    default: {
      oss << code;
      break;
    }
  }
}

MemPool<512, 8192> NDObject::mem_pool_;

static void CustomDimChanged(NDObject *op) { static_cast<CustomOp *>(op)->DimChanged(); }
static void CustomFoldProp(NDObject *op, PropRange &r) { static_cast<CustomOp *>(op)->FoldProp(r); }
static void CustomTileCollect(NDObject *op, TileInfo &t) { static_cast<CustomOp *>(op)->TileCollect(t); }
static void CustomShapeProp(NDObject *op, int64_t &s) { static_cast<CustomOp *>(op)->ShapeProp(s); }

static constexpr ObjectMeta GenObjectMeta() {
  struct BaseData {
    CodeGenTmpl tmpl;
    uint32_t flags;
    void (*dim_changed)(NDObject *);
    void (*fold_prop)(NDObject *, PropRange &);
    void (*tile_collect)(NDObject *, TileInfo &);
    void (*shape_prop)(NDObject *, int64_t &);
  };
  constexpr uint32_t F_NS = ObjectMeta::kNddShared;
  constexpr uint32_t F_IP = ObjectMeta::kInplaceProp;
  constexpr uint32_t F_LR = ObjectMeta::kLhsReuse;
  constexpr uint32_t F_RR = ObjectMeta::kRhsReuse;
  constexpr uint32_t F_LD = ObjectMeta::kLhsDom;
  constexpr uint32_t F_DM = ObjectMeta::kDom;
  BaseData data[] = {
    {kGenLoad, 0, nullptr, nullptr, nullptr},                                                            // LoadDummy
    {kGenComm, 0, nullptr, nullptr, nullptr},                                                            // MultiLoad
    {kGenLoad, F_IP, nullptr, nullptr, nullptr},                                                         // GlobalAccess
    {kGenLoad, F_IP, nullptr, nullptr, NDObject::TileCollectSimt},                                       // GatherLoad
    {kGenLoad, 0, NDViewLoad::DimChanged, NDAccess::FoldProp, NDAccess::TileCollect},                    // ViewLoad
    {kGenLoad, F_IP, nullptr, NDAccess::FoldProp, NDAccess::TileCollect},                                // Load
    {kGenStore, F_NS | F_LD, NDViewStore::DimChanged, NDAccess::FoldProp, NDAccess::TileCollect},        // ViewStore
    {kGenStore, F_IP | F_NS | F_LD, nullptr, NDAccess::FoldProp, NDAccess::TileCollect},                 // Store
    {kGenComm, F_LR | F_LD, nullptr, ReduceScatterOp::FoldProp, ReduceScatterOp::TileCollect,
     ReduceScatterOp::ShapeProp},                                                                     // ReduceScatter
    {kGenComm, 0, nullptr, AllGatherOp::FoldProp, AllGatherOp::TileCollect, AllGatherOp::ShapeProp},  // AllGather
    {kGenComm, 0, nullptr, nullptr, CommOp::TileCollect, nullptr},                                    // AllGatherV2
    {kGenComm, F_LR, nullptr, nullptr, CommOp::TileCollect, nullptr},                                 // AllReduce
    {kGenSimd1, F_IP | F_LR, nullptr, nullptr, nullptr, ReshapeOp::ShapeProp},                        // Reshape
    {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                                       // Copy
    {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                                       // Unary
    {kGenSimd2, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr, BinaryOp::ShapeProp},           // Binary
    {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr},                                              // Cast
    {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr, nullptr},                                     // Extract
    {kGenFlex, F_IP | F_NS, nullptr, nullptr, nullptr, nullptr},                                      // Pack
    {kGenSimd1, F_IP | F_NS | F_LR | F_DM, nullptr, nullptr, nullptr},                                // BinaryS
    {kGenFlex, F_DM, nullptr, BroadcastOp::FoldProp, BroadcastOp::TileCollect, BroadcastOp::ShapeProp},  // BroadcastTo
    {kGenSimd0, F_IP, nullptr, nullptr, nullptr},                                                        // BroadcastS
    {kGenFlex, F_LR | F_LD, nullptr, ReduceOp::FoldProp, ReduceOp::TileCollect, ReduceOp::ShapeProp},    // Reduce
    {kGenSimd3, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr, SelectOp::ShapeProp},   // Select
    {kGenSimd1, F_LD, nullptr, nullptr, nullptr},                                            // ElemAny
    {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                              // RemovePad
    {kGenFlex, F_IP | F_NS, nullptr, nullptr, nullptr, PowerOp::ShapeProp},                  // Power
    {kGenFlex, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr, CompareOp::ShapeProp},  // Compare
    {kGenFlex, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                               // CompareS
    {kGenSimd1, F_DM, OneHotOp::DimChanged, OneHotOp::FoldProp, OneHotOp::TileCollect, OneHotOp::ShapeProp},  // OneHot
    {kGenSimd1, F_LR, nullptr, nullptr, nullptr, PermuteOp::ShapeProp},                                       // Permute
    {kGenSimd1, F_LR, ConcatOp::DimChanged, ConcatOp::FoldProp, ConcatOp::TileCollect, ConcatOp::ShapeProp}, // Concat
    {kGenSimd1, F_LR, SplitOp::DimChanged, SplitOp::FoldProp, SplitOp::TileCollect, SplitOp::ShapeProp},  // SplitOp
    {kGenSimd1, F_LR, SliceOp::DimChanged, SliceOp::FoldProp, SliceOp::TileCollect, SliceOp::ShapeProp},  // SliceOp
    {kGenCustom, F_DM, CustomDimChanged, CustomFoldProp, CustomTileCollect, CustomShapeProp},                 // Custom
    {kGenExtOut, F_NS, nullptr, nullptr, nullptr, nullptr},                                                   // ExtOut
    {kGenSimd0, 0, nullptr, nullptr, nullptr, CubeOp::ShapeProp},                                             // CubeOp
    {kGenSimd0, 0, nullptr, nullptr, nullptr, GmmOp::ShapeProp},                                              // GmmOp
  };

  ObjectMeta meta;
  for (size_t i = 0; i < sizeof(data) / sizeof(BaseData); ++i) {
    meta.flags[i] = data[i].flags;
    meta.tmpl[i] = data[i].tmpl;
    meta.dim_changed[i] = data[i].dim_changed;
    meta.fold_prop[i] = data[i].fold_prop;
    meta.tile_collect[i] = data[i].tile_collect;
    meta.shape_prop[i] = data[i].shape_prop;
  }
  return meta;
}

const ObjectMeta NDObject::meta_ = GenObjectMeta();

class AtomicCleanWrap : public CodeWrap {
 public:
  AtomicCleanWrap(NDAccess *store, int red_op) {
    clear_shape_.data = &clear_shape_data_;
    clear_shape_.size = 1;
    auto type = store->type_id_;
    KernelBuilder b(&kernel_);
    auto op = b.Broadcast(GetRedInitScalar(red_op), &clear_shape_, type);
    if (store->obj_id_ == ObjectType::kStore) {
      store_ = static_cast<NDStore *>(b.Store(store->addr_.gm, op));
    } else {
      ASSERT(store->obj_id_ == ObjectType::kViewStore);
      auto stride = static_cast<NDViewStore *>(store)->dst_stride_ref_;
      store_ = static_cast<NDStore *>(b.Store(store->addr_.gm, op, stride));
    }
  }

  void CodeGen(Code &code, NDAccess *store) {
    clear_shape_data_ = std::accumulate(store->shape_ref_->data, store->shape_ref_->data + store->shape_ref_->size, 1LL,
                                        std::multiplies{});
    kernel_.CodeGen();
    code.BindOpFast(store_->addr_, store->addr_);
  }

  void CodeGenView(Code &code, NDAccess *store) {
    if (store->obj_id_ == ObjectType::kStore) {
      CodeGen(code, store);
    } else {
      clear_shape_ = *store->shape_ref_;
      kernel_.CodeGen();
      code.BindOpFast(store_->addr_, store->addr_);
    }
  }

  int LaunchWrap(void *workspace, void *stream) override {
    kernel_.code_.Launch(nullptr, stream);
    return next_->LaunchWrap(workspace, stream);
  }

  void DasWrap(std::ostringstream &oss) override {
    kernel_.code_.DisAssemble(oss);
    oss << std::endl;
    next_->DasWrap(oss);
  }

  void CollectWrap(std::vector<Code *> &codes) override {
    kernel_.code_.Collect(codes);
    next_->CollectWrap(codes);
  }

 private:
  VKernelD kernel_;
  NDStore *store_;
  IntArrayRef clear_shape_;
  int64_t clear_shape_data_;
};

int64_t NDObject::Size() {
  return std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) *
         ITEM_SIZE[type_id_];
}

void NDObject::Tile(const TileParam &tp) {
  if (auto ndd = Ndd(); ndd != nullptr) {
    auto &dims = ndd->dims;
    bool pointwise = false;
    for (int i = tp.start; i <= tp.end; ++i) {
      if (dims[i] != 1) {
        pointwise = true;
        ndd->pointwise_tile_mask |= (1u << i);
      }
      dims[i] = 1;
    }
    if (pointwise) {  // broadcast source or reduce des: all 1, donot need to tile
      dims[tp.start] = tp.tile;
    }
  }
}

void NDObject::Shard(const ShardParam &sp) {
  if (auto ndd = Ndd(); ndd != nullptr) {
    auto &dims = ndd->dims;
    for (size_t i = sp.base; i < dims.size(); ++i) {
      if (dims[i] != 1) {
        ndd->pointwise_tile_mask |= (1u << i);
      }
      dims[i] = i < static_cast<size_t>(sp.base + ShardParam::PARTIAL_SIZE) && dims[i] != 1 ? sp.tile[i - sp.base] : 1;
    }
  }
}

NDObject *NDObject::Clone(CloneHelper &h) {
  DvmException("[BUG] NDObject::Clone is not inheritted!");
  return nullptr;
}

void NDObject::Dump(bool verbose, std::ostringstream &oss) { oss << "NDObject"; }

void NDObject::Inspect(Inspector &sp) {}

void NDObject::TileCollectSimt(NDObject *op, TileInfo &info) { info.flags |= ObjectMeta::kSimt; }

bool NDAccess::IsSupportView() const {
  if (obj_id_ == kViewLoad || obj_id_ == kViewStore || obj_id_ == kLoadDummy || obj_id_ == kLoad) {
    return true;
  }
  if (obj_id_ == kStore && (lhs_->obj_id_ != kReduce || !g_system.deterministic_) && lhs_->obj_id_ != kElementAny) {
    return true;
  }
  return false;
}

void NDAccess::TileCollect(NDObject *op, TileInfo &info) {
  auto acc = static_cast<NDAccess *>(op);
  if (acc->stride_ == nullptr) {
    return;
  }
  info.code_reserve += (op->nd_.size() + 1) * sizeof(uint64_t);
  auto &stride = *acc->stride_;
  if (op->CheckFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL)) {
    info.lead_depth = std::min(info.lead_depth, 1);
  }
  if (int max_depth = stride.size(); info.lead_depth > max_depth) {
    info.lead_depth = max_depth;
  }
  auto &dims = acc->nd_.dims();
  for (int d = 1; d < info.lead_depth; ++d) {
    if (stride[d - 1] * dims[d - 1] != stride[d]) {
      info.lead_depth = d;
      break;
    }
  }
  if (g_system.Arch() == AiCoreArch::kAiCore_C220) { // viewloadx
    constexpr uint32_t EXT_WS = 1024;
    if ((static_cast<uint64_t>(stride[0]) > ITEM_SIZE[op->type_id_] || dims[0] == 1) && info.ext_ws < EXT_WS) {
      info.ext_ws = EXT_WS;
    }
    if (info.event_reserve < 2) {
      info.event_reserve = 2;
    }
  }
}

void NDAccess::FoldProp(NDObject *op, PropRange &range) {
  auto acc = static_cast<NDAccess *>(op);
  if (acc->stride_ == nullptr) {
    return;
  }
  auto &stride = *acc->stride_;
  int stride_size = stride.size();
  if (range.base >= stride_size) {
    if (int max_depth = range.base - stride_size + 1; range.depth > max_depth) {
      range.depth = max_depth;
    }
  } else {
    auto &dims = acc->nd_.dims();
    if (op->CheckFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL_ROW_MAJOR)) {
      if (range.base < FractalRowMajorAxes::kRank) {
        range.depth = 1;
        range.affine = PropRange::NO_TILING;
      } else if (range.depth > range.base - FractalRowMajorAxes::kRank + 1) {
        range.depth = range.base - FractalRowMajorAxes::kRank + 1;
      }
    } else if (op->CheckFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL)) {
      constexpr int fixed_depth = 2;
      if (range.base < fixed_depth) {
        range.depth = 1;
      } else if (range.depth > range.base - fixed_depth + 1) {
        range.depth = range.base - fixed_depth + 1;
      }
    }
    for (int d = 1; d < range.depth; ++d) {
      if (stride[range.base - d] * dims[range.base - d] != stride[range.base - d + 1]) {
        range.depth = d;
        break;
      }
    }
  }
}

uint64_t NDLoadDummy::Emit(VectorKernel &k) {
  *insn_ = vMakeAccHead(V_LOAD_DUMMY, 0, 1);
  return 1;
}

uint64_t NDGlobalAccess::Emit(VectorKernel &k) { return 0; }

NDObject *NDGlobalAccess::Clone(CloneHelper &h) {
  return new NDGlobalAccess(addr_.gm, h.GetClone(shape_ref_), type_id_);
}

void NDGlobalAccess::Dump(bool verbose, std::ostringstream &oss) { oss << "GlobalAccess"; }

NDObject *NDLoadDummy::Clone(CloneHelper &h) { return new NDLoadDummy(type_id_); }

void NDLoadDummy::Dump(bool verbose, std::ostringstream &oss) { oss << "LoadDummy"; }

void NDLoad::Shard(const ShardParam &sp) {
  if (sp.sink) {
    flags_ |= OBJ_FLAG_LOAD_SHARD_ROUND;
    if (ndd_[sp.base + 1] == 1 && sp.tile[sp.base + 1] > 1) {
      flags_ |= OBJ_FLAG_LOAD_SHARD_BCAST1;
    }
    if (ndd_[sp.base] == 1 && sp.tile[sp.base] > 1) {
      flags_ |= OBJ_FLAG_LOAD_SHARD_BCAST0;
    }
  }
  NDObject::Shard(sp);
}

uint64_t NDLoad::Emit(VectorKernel &k) {
  if (stride_) {
    return EmitView(k);
  }
  uint64_t rounds[2];
  DimArray round_tile;
  BuildRoundTile(k, ndd_, (flags_ & OBJ_FLAG_LOAD_SHARD_ROUND) != 0, round_tile);
  if (!round_tile.empty()) {
    BuildDimRounds(round_tile, rounds);
  }
  int64_t lead_align = ndd_.lead_stride();
  int64_t lead_dim = ndd_.lead_dim();
  uint64_t src_tile_stride_ = ndd_.stride_back() / lead_align * lead_dim;
  int64_t tail_size = k.GetTailSize(&ndd_);
  if (auto shard = k.shard_) {
    constexpr int SHARD_M = 1;
    constexpr int SHARD_N = 0;
    if (flags_ & OBJ_FLAG_LOAD_FROM_CC) {
      *insn_ = vMakeAccHead(V_LOAD_DUMMY, 0, 1);
      vCubeCopy op;
      op.xd = xbuf_;
      op.xn = k.GetVisitor<MixVisitCoder>()->ub_base_;
      op.lenburst = GetBlocks(src_tile_stride_);
      op.tile_offset = (k.tile_num_ + 1) / 2;
      return vCubeCopy::Encode(insn_ + 1, V_COPY_CUBE_TILE, op) + 1;
    } else if (flags_ & OBJ_FLAG_LOAD_PINGPONG) {
      vPingPongLoad op;
      op.from = addr_.gm;
      op.xn = xbuf_;
      op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
      op.body_iter = ndd_.stride_back() / lead_align;
      op.tail_iter = (tail_size == 0 || k.GetTailDim() <= ndd_.lead_idx()) ? op.body_iter : op.body_iter / ndd_[k.GetTailDim()] * tail_size;
      op.iter_size = lead_dim * ITEM_SIZE[type_id_];
      op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
      op.pingpong = 0;
      op.pingpong_stride = shard->tile[SHARD_N] * shard->tile[SHARD_M] * ITEM_SIZE[type_id_];
      op.round_rank = round_tile.size();
      addr_.Update(insn_ + vPingPongLoad::RELOC_OFFSET);
      return vPingPongLoad::Encode(insn_, vAccInsnID::V_PINGPONG_LOAD, op, rounds);
    } else {
      vSLoad op;
      op.gm = addr_.gm;
      op.xn = xbuf_;
      op.tile_stride = src_tile_stride_;
      op.pad_size = lead_align - lead_dim;
      op.broadcast_m = flags_ & OBJ_FLAG_LOAD_SHARD_BCAST1;
      op.broadcast_n = flags_ & OBJ_FLAG_LOAD_SHARD_BCAST0;
      op.round_rank = round_tile.size();
      op.type_size = ITEM_SIZE[type_id_];
      addr_.Update(insn_ + vSLoad::RELOC_OFFSET);
      k.GetVisitor<MixVisitCoder>()->AddReloc(insn_, vSLoad::SHARD_OFFSET);
      return vSLoad::Encode(insn_, vAccInsnID::V_SLOAD, op, rounds);
    }
  }  // end shard_
  vLoad op;
  op.from = addr_.gm;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
  op.body_iter = ndd_.stride_back() / lead_align;
  op.iter_size = lead_dim * ITEM_SIZE[type_id_];
  op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
  if (op.body_iter == 1) {
    op.tail_iter = tail_size == 0 ? op.iter_size : tail_size * ITEM_SIZE[type_id_];
  } else {
    op.tail_iter = tail_size == 0 ? op.body_iter : op.body_iter / ndd_[k.GetTailDim()] * tail_size;
    if (op.pad_size == 0) {
      op.tail_iter *= op.iter_size;
      op.iter_size *= op.body_iter;
      op.body_iter = 1;
    }
  }
  op.round_rank = round_tile.size();
  addr_.Update(insn_ + vLoad::RELOC_OFFSET);
  return vLoad::Encode(insn_, vAccInsnID::V_LOAD, op, rounds);
}

uint64_t NDLoad::EmitView(VectorKernel &k) {
  DimArray &src_stride_ = *stride_;
  int tail_dim = k.tile_region_.depth ? k.tile_region_.starts[k.tile_region_.depth - 1] : ndd_.size() - 1;
  int64_t tail_size = k.GetTailSize(&ndd_);
  DimArray fold_dim, fold_stride;
  int lead_idx, fold_idx, last_dim;
  if (flags_ & OBJ_FLAG_VIEW_LOAD_FRACTAL_ROW_MAJOR) {
    ASSERT(k.tile_region_.depth == 0 || k.tile_region_.starts[k.tile_region_.depth - 1] >= FractalRowMajorAxes::kRank);
    constexpr int w_dim = FractalRowMajorAxes::kWidthElement;
    constexpr int h_dim = FractalRowMajorAxes::kHeightElement;
    fold_dim[0] = ndd_.dims[w_dim];
    fold_dim[1] = ndd_.dims[h_dim];
    fold_stride[0] = src_stride_[w_dim];
    fold_stride[1] = src_stride_[h_dim];
    lead_idx = h_dim;
    last_dim = h_dim;
    fold_idx = 1;
    for (int d = 2; d <= tail_dim; ++d) {
      const int i = FractalRowMajorAxes::PhysicalAxis(d);
      if (ndd_.dims[i] == 1) continue;
      // Keep the rectangular Tile iteration axis separate from its first outer group.
      const bool first_outer = d >= FractalRowMajorAxes::kRank && last_dim < FractalRowMajorAxes::kRank;
      if (first_outer || last_dim == lead_idx ||
          src_stride_[last_dim] * ndd_.dims[last_dim] != src_stride_[i]) {
        fold_idx++;
        fold_dim[fold_idx] = ndd_.dims[i];
        fold_stride[fold_idx] = src_stride_[i];
      } else {
        fold_dim[fold_idx] *= ndd_.dims[i];
      }
      last_dim = i;
    }
  } else {
    if (flags_ & OBJ_FLAG_VIEW_LOAD_FRACTAL) {
      ASSERT(tail_dim >= 2);
      fold_dim[0] = ndd_.dims[0];
      fold_dim[1] = ndd_.dims[1];
      fold_stride[0] = src_stride_[0];
      fold_stride[1] = src_stride_[1];
      lead_idx = 1;
      last_dim = 1;
      fold_idx = 1;
    } else {
      lead_idx = std::min(ndd_.lead_idx(), tail_dim);
      fold_dim[0] = ndd_.dims[lead_idx];
      fold_stride[0] = src_stride_[lead_idx];
      last_dim = lead_idx;
      if (nd_.stride(lead_idx) == nd_[lead_idx]) {
        for (int i = last_dim + 1; i <= tail_dim; ++i) {
          if (ndd_.dims[i] == 1) continue;
          if (src_stride_[last_dim] * ndd_.dims[last_dim] != src_stride_[i]) {
            break;
          }
          fold_dim[0] *= nd_[i];
          last_dim = i;
        }
        lead_idx = last_dim;
      }
      fold_idx = 0;
    }
    for (int i = last_dim + 1; i <= tail_dim; ++i) {
      if (ndd_.dims[i] == 1) continue;
      if (last_dim == lead_idx || src_stride_[last_dim] * ndd_.dims[last_dim] != src_stride_[i]) {
        fold_idx++;
        fold_dim[fold_idx] = ndd_.dims[i];
        fold_stride[fold_idx] = src_stride_[i];
      } else {
        fold_dim[fold_idx] *= ndd_.dims[i];
      }
      last_dim = i;
    }
  }
  int tile_start = fold_idx + 1;
  if (auto &tr = k.tile_region_; tr.depth > 0) {
    auto tile_end = [this, &tr](int d) { return d >= 1 ? tr.starts[d - 1] - 1 : ndd_.size(); };
    fold_idx++;
    fold_dim[fold_idx] = tr.last_num;
    if (ndd_.PointwiseTile(tail_dim, tile_end(tr.depth - 1))) {
      fold_stride[fold_idx] = src_stride_[tail_dim] * ndd_.dims[tail_dim];
    } else {
      fold_stride[fold_idx] = 0;
    }
    for (int d = tr.depth - 2; d >= 0; --d) {
      auto start = tr.starts[d];
      auto stride = ndd_.PointwiseTile(start, tile_end(d)) ? src_stride_[start] : 0;
      if (fold_stride[fold_idx] * fold_dim[fold_idx] != stride) {
        fold_idx++;
        fold_dim[fold_idx] = tr.spaces[d];
        fold_stride[fold_idx] = stride;
      } else {
        fold_dim[fold_idx] *= tr.spaces[d];
      }
    }
  }
  fold_dim.resize(fold_idx + 1);
  fold_stride.resize(fold_idx + 1);
  int64_t item_size = ITEM_SIZE[type_id_];
  if (fold_stride[0] == item_size) {
    vViewLoad op;
    uint64_t *var_insn = insn_ + vViewLoad::VAR_OFFSET;
    op.xd = xbuf_;
    op.from = addr_.data;
    op.offset = offset_bytes_;
    op.iter_size = fold_dim[0] * item_size;
    // loop space
    int loop_start;
    uint64_t dst_stride = ndd_.stride(lead_idx) * item_size;
    if (tile_start > 1 && fold_stride[0] * fold_dim[0] <= fold_stride[1]) {
      op.src_gap = fold_stride[1] - fold_stride[0] * fold_dim[0];
      op.dst_gap = (ndd_.stride(lead_idx) * item_size - op.iter_size) >> 5;
      op.iter_num = fold_dim[1];
      loop_start = 2;
      dst_stride *= fold_dim[1];
    } else {
      op.src_gap = 0;
      op.dst_gap = 0;
      op.iter_num = 1;
      loop_start = 1;
    }
    op.loop_depth = tile_start - loop_start;
    for (int i = loop_start; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeLoop(fold_dim[i], dst_stride, fold_stride[i]);
      dst_stride *= fold_dim[i];
    }
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size) {
      op.tail_size = op.tail_size / ndd_[tail_dim] * tail_size;
    }
    if (op.loop_depth == 0 && loop_start == 1) {
      op.tail_size *= item_size;
    }
    // tile space
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewLoad::RELOC_OFFSET);
    return vViewLoad::Encode(insn_, V_LOAD_VIEW, op);
  } else if (flags_ & OBJ_FLAG_VIEW_LOAD_FRACTAL) {
    vViewLoadT op;
    op.xd = xbuf_;
    op.from = addr_.data;
    op.offset = offset_bytes_;
    op.item_size = item_size;
    op.w_fractal = fold_dim[0];
    op.h_fractal = fold_dim[1];
    op.w_gap = fold_stride[0] - op.h_fractal * item_size;
    op.row_stride_blocks = op.item_size == 2 || op.w_fractal <= 8 ? 1 : 2;
    op.is_width_iter = 0;
    if (tile_start > 2) {
      op.iter_size = fold_dim[2];
      op.iter_stride = fold_stride[2];
      op.loop_depth = tile_start - 3;
    } else { // tile_start == 2
      op.iter_size = 1;
      op.iter_stride = 0;
      op.loop_depth = 0;
    }
    if (tile_start == 2) {
      // No dim2 iteration axis: h_fractal describes the H tail, so the iteration tail is 1.
      op.tail_size = 1;
    } else {
      op.tail_size = fold_dim[tile_start - 1];
      if (tail_size) {
        op.tail_size = op.tail_size / ndd_[tail_dim] * tail_size;
      }
    }
    if (g_system.Arch() == AiCoreArch::kAiCore_C220) {
      op.ws = GetDynXBufBegin(k, this);
      op.ws_size = std::min((g_system.LocalMemSize() - op.ws) / (512 * 3), op.iter_size);
      ASSERT(op.ws_size);
    } else {
      op.ws = 0;
      op.ws_size = 0;
    }
    uint64_t *var_insn = insn_ + vViewLoadT::VAR_OFFSET;
    if (flags_ & OBJ_FLAG_VIEW_LOAD_FRACTAL_ROW_MAJOR) {
      constexpr int h_tile_dim = FractalRowMajorAxes::kHeightTile;
      constexpr int w_tile_dim = FractalRowMajorAxes::kWidthTile;
      const uint64_t row_stride_bytes = ndd_.stride(w_tile_dim) * item_size;
      const uint64_t tile_stride_bytes = ndd_.stride(h_tile_dim) * item_size;
      const bool is_height_iter = ndd_[h_tile_dim] > 1 &&
                                  op.iter_size == static_cast<uint64_t>(ndd_[h_tile_dim]) &&
                                  op.iter_stride == static_cast<uint64_t>(src_stride_[h_tile_dim]);
      const bool is_width_iter = ndd_[w_tile_dim] > 1 &&
                                 op.iter_size == static_cast<uint64_t>(ndd_[w_tile_dim]) &&
                                 op.iter_stride == static_cast<uint64_t>(src_stride_[w_tile_dim]);
      if (op.loop_depth) {
        uint64_t remaining_inner_tiles =
          is_width_iter ? ndd_[h_tile_dim] : (is_height_iter ? ndd_[w_tile_dim] : 1);
        const uint64_t inner_dst_stride_bytes =
          is_width_iter ? op.h_fractal * row_stride_bytes : op.w_fractal * item_size;
        uint64_t outer_dst_stride_bytes =
          tile_stride_bytes * (is_height_iter || is_width_iter ? 1 : op.iter_size);
        for (int i = 3; i < tile_start; ++i, ++var_insn) {
          uint64_t loop_dst_stride_bytes;
          if (remaining_inner_tiles > 1) {
            ASSERT(fold_dim[i] == static_cast<int64_t>(remaining_inner_tiles));
            loop_dst_stride_bytes = inner_dst_stride_bytes;
            remaining_inner_tiles = 1;
          } else {
            loop_dst_stride_bytes = outer_dst_stride_bytes;
            outer_dst_stride_bytes *= fold_dim[i];
          }
          *var_insn = vViewLoad::EncodeLoop(fold_dim[i], loop_dst_stride_bytes, fold_stride[i]);
        }
      }
      ASSERT(row_stride_bytes % SIMD_BLOCK_SIZE == 0);
      const uint64_t row_stride_blocks = row_stride_bytes / SIMD_BLOCK_SIZE;
      ASSERT(row_stride_blocks < (1ull << 13));
      op.row_stride_blocks = row_stride_blocks;
      op.is_width_iter = is_width_iter;
    } else if (op.loop_depth) {
      uint64_t dst_stride = ndd_.stride(1) * fold_dim[2] * item_size;
      for (int i = 3; i < tile_start; ++i, ++var_insn) {
        *var_insn = vViewLoad::EncodeLoop(fold_dim[i], dst_stride, fold_stride[i]);
        dst_stride *= fold_dim[i];
      }
    }
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewLoadT::RELOC_OFFSET);
    return vViewLoadT::Encode(insn_, V_LOAD_VIEW_TRANS, op);
  } else {
    EXCEPTION_IF(item_size != 2 && item_size != 4, "view load x only support itemsize 2\4");
    vViewLoadX op;
    op.xd = xbuf_;
    op.from = addr_.data;
    op.offset = offset_bytes_;
    op.type_size = item_size;
    op.ws = GetDynXBufBegin(k, this);
    auto ws_block_num = (g_system.LocalMemSize() - op.ws) / (SIMD_BLOCK_SIZE * 2);
    op.ws_size = RoundDown(ws_block_num, SIMD_BLOCK_SIZE / item_size);
    op.iter_size = fold_dim[0];
    op.iter_stride = fold_stride[0];
    op.loop_depth = tile_start - 1;
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size) {
      op.tail_size = op.tail_size / ndd_[tail_dim] * tail_size;
    }
    uint64_t *var_insn = insn_ + vViewLoadX::VAR_OFFSET;
    uint64_t dst_stride = ndd_.stride(lead_idx) * item_size;
    for (int i = 1; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeLoop(fold_dim[i], dst_stride, fold_stride[i]);
      dst_stride *= fold_dim[i];
    }
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewLoadX::RELOC_OFFSET);
    return vViewLoadX::Encode(insn_, V_LOAD_VIEW_X, op);
  }
}

void NDLoad::Normalize(std::vector<NDObject *> &run_ops) {
  auto dims = shape_ref_->size;
  ndd_.Reset(dims);
  for (size_t i = 0; i < shape_ref_->size; i++) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
  }
}

NDObject *NDLoad::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  return new NDLoad(addr_.gm, shape_ref, type_id_);
}

void NDLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "Load"; }

void NDGatherLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(src_shape_ref_->size >= 1);
  ASSERT(index_ && index_->shape_ref_);
  axis_ = axis_ >= 0 ? axis_ : static_cast<int>(src_shape_ref_->size) + axis_;
  auto *index_shape_ref = index_->shape_ref_;
  gather_size_ = 1;
  inner_size_ = 1;
  gather_dim_size_ = static_cast<uint64_t>(src_shape_ref_->data[axis_]);
  if (gather_mode_ == vGatherLoad::kElementGather) {
    shape_.Resize(index_shape_ref->size);
    for (size_t i = 0; i < index_shape_ref->size; ++i) {
      shape_[i] = index_shape_ref->data[i];
    }
    gather_size_ = static_cast<uint64_t>(index_shape_ref->data[axis_]);
    for (size_t i = axis_ + 1; i < src_shape_ref_->size; ++i) {
      inner_size_ *= static_cast<uint64_t>(src_shape_ref_->data[i]);
    }
  } else {
    shape_.Resize(src_shape_ref_->size - 1 + index_shape_ref->size);
    size_t out_idx = 0;
    for (int i = 0; i < axis_; ++i) {
      shape_[out_idx++] = src_shape_ref_->data[i];
    }
    for (size_t i = 0; i < index_shape_ref->size; ++i) {
      shape_[out_idx++] = index_shape_ref->data[i];
      gather_size_ *= static_cast<uint64_t>(index_shape_ref->data[i]);
    }
    for (size_t i = axis_ + 1; i < src_shape_ref_->size; ++i) {
      shape_[out_idx++] = src_shape_ref_->data[i];
      inner_size_ *= static_cast<uint64_t>(src_shape_ref_->data[i]);
    }
  }
  NDLoad::Normalize(run_ops);
}

uint64_t NDGatherLoad::Emit(VectorKernel &k) {
  ASSERT(k.shard_ == nullptr);
  uint64_t rounds[2];
  DimArray round_tile;
  BuildRoundTile(k, ndd_, (flags_ & OBJ_FLAG_LOAD_SHARD_ROUND) != 0, round_tile);
  if (!round_tile.empty()) {
    BuildDimRounds(round_tile, rounds);
  }
  const uint64_t lead_align = static_cast<uint64_t>(ndd_.lead_stride());
  const uint64_t lead_dim = static_cast<uint64_t>(ndd_.lead_dim());
  int64_t tail_size = k.GetTailSize(&ndd_);
  vGatherLoad op;
  op.xn = xbuf_;
  op.from = reinterpret_cast<uint64_t>(addr_.gm);
  op.index = reinterpret_cast<uint64_t>(index_->addr_.gm);
  op.inner_size = inner_size_;
  op.gather_size = gather_size_;
  op.gather_dim_size = gather_dim_size_;
  op.gather_mode = gather_mode_;
  op.body_iter = static_cast<uint64_t>(ndd_.stride_back()) / lead_align;
  op.iter_size = lead_dim;
  op.pad_size = lead_align - lead_dim;
  op.round_rank = round_tile.size();
  op.tail_iter =
    tail_size == 0
      ? (op.body_iter == 1 ? op.iter_size : op.body_iter)
      : (op.body_iter == 1 ? static_cast<uint64_t>(tail_size)
                           : op.body_iter / static_cast<uint64_t>(ndd_[k.GetTailDim()]) * static_cast<uint64_t>(tail_size));
  addr_.Update(insn_ + vGatherLoad::RELOC_OFFSET);
  index_->addr_.Update(insn_ + vGatherLoad::INDEX_RELOC_OFFSET);
  const auto insn_id = ITEM_SIZE[type_id_] == 2 ? vAccInsnID::V_LOAD_GATHER_B16 : vAccInsnID::V_LOAD_GATHER_B32;
  return vGatherLoad::Encode(insn_, insn_id, op, rounds);
}

NDObject *NDGatherLoad::Clone(CloneHelper &h) {
  auto src_shape_ref = h.GetClone(src_shape_ref_);
  auto index = static_cast<NDAccess *>(h.GetClone(index_));
  return new NDGatherLoad(addr_.gm, src_shape_ref, index, axis_, type_id_, gather_mode_);
}

void NDGatherLoad::Dump(bool verbose, std::ostringstream &oss) {
  oss << "GatherLoad";
  if (verbose) {
    oss << "<axis=" << axis_ << ", gather_mode=" << gather_mode_ << ">";
  }
}

void NDViewLoad::DimChanged(NDObject *op) {
  auto view = static_cast<NDViewLoad *>(op);
  auto ref_stride = view->src_stride_ref_->data;
  auto ref_shape = view->shape_ref_->data;
  int ref_idx = view->shape_ref_->size - 1;
  auto &stride = view->src_stride_;
  auto &dims = view->ndd_.dims;
  size_t dim_size = dims.size();
  stride.resize(dim_size);
  int64_t acc_nd = 0;
  int64_t acc_ref = 0;
  int64_t acc_stride = ITEM_SIZE[view->type_id_];
  for (size_t i = 0; i < dim_size; ++i) {
    if (dims[i] == 1) {
      if (i > 0) {
        acc_stride *= dims[i - 1];
      }
    } else if (!acc_nd) {
      acc_stride = ref_stride[ref_idx] * ITEM_SIZE[view->type_id_];
      if (dims[i] != ref_shape[ref_idx]) {
        acc_nd = dims[i];
        acc_ref = ref_shape[ref_idx];
      }
      ref_idx--;
    } else {
      acc_stride *= dims[i - 1];
      acc_nd *= dims[i];
      while (acc_ref < acc_nd) {
        acc_ref *= ref_shape[ref_idx--];
      }
      if (acc_nd == acc_ref) {
        acc_nd = 0;
      }
    }
    stride[i] = acc_stride;
  }
}

void NDViewLoad::Normalize(std::vector<NDObject *> &run_ops) {
  auto dim_size = shape_ref_->size;
  ndd_.Reset(dim_size);
  src_stride_.resize(dim_size);
  if (dim_size > 0) {
    auto item_size = ITEM_SIZE[type_id_];
    ndd_.dims[0] = shape_ref_->data[dim_size - 1];
    src_stride_[0] = ndd_.dims[0] == 1 ? item_size : src_stride_ref_->data[dim_size - 1] * item_size;
    for (size_t i = 1; i < dim_size; i++) {
      ndd_.dims[i] = shape_ref_->data[dim_size - i - 1];
      src_stride_[i] = ndd_.dims[i] == 1 ? src_stride_[i - 1] * ndd_.dims[i - 1] : src_stride_ref_->data[dim_size - i - 1] * item_size;
    }
  }
  offset_bytes_ = 0;
}

NDObject *NDViewLoad::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  auto src_stride_ref = h.GetClone(src_stride_ref_);
  return new NDViewLoad(addr_.gm, shape_ref, src_stride_ref, type_id_);
}

void NDViewLoad::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ViewLoad";
  if (verbose) {
    int64_t offset = static_cast<int64_t>(offset_bytes_ / ITEM_SIZE[type_id_]);
    oss << "<" << offset << ", " << *src_stride_ref_ << ">";
  }
}

void NDViewStore::DimChanged(NDObject *op) {
  auto store = static_cast<NDViewStore *>(op);
  auto ref_stride = store->dst_stride_ref_;
  auto ref_shape = store->shape_ref_;
  auto &stride = store->dst_stride_;
  auto &dims = store->nd_;
  size_t dim_size = dims.size();
  stride.resize(dim_size);
  int64_t acc_nd = 0;
  int64_t acc_ref = 0;
  int64_t acc_stride = ITEM_SIZE[store->type_id_];
  int ref_idx = ref_shape->size - 1;
  for (size_t i = 0; i < dim_size; ++i) {
    if (dims[i] == 1) {
      if (i > 0) {
        acc_stride *= dims[i - 1];
      }
    } else if (!acc_nd) {
      acc_stride = ref_stride->data[ref_idx] * ITEM_SIZE[store->type_id_];
      stride[i] = acc_stride;
      if (dims[i] != ref_shape->data[ref_idx]) {
        acc_nd = dims[i];
        acc_ref = ref_shape->data[ref_idx];
      }
      ref_idx--;
    } else {
      acc_stride *= dims[i - 1];
      acc_nd *= dims[i];
      while (acc_ref < acc_nd) {
        acc_ref *= ref_shape->data[ref_idx--];
      }
      if (acc_nd == acc_ref) {
        acc_nd = 0;
      }
    }
    stride[i] = acc_stride;
  }
}

void NDViewStore::Normalize(std::vector<NDObject *> &run_ops) {
  nd_ = lhs_->nd_;
  auto dim_size = dst_stride_ref_->size;
  ASSERT(dim_size <= nd_.size());
  dst_stride_.resize(dim_size);
  if (dim_size > 0) {
    auto item_size = ITEM_SIZE[type_id_];
    dst_stride_[0] = nd_[0] == 1 ? item_size : dst_stride_ref_->data[dim_size - 1] * item_size;
    for (size_t i = 1; i < dim_size; i++) {
      dst_stride_[i] = nd_[i] == 1 ? dst_stride_[i - 1] * nd_[i - 1] : dst_stride_ref_->data[dim_size - i - 1] * ITEM_SIZE[type_id_];
    }
  }
  offset_bytes_ = 0;
}

NDObject *NDViewStore::Clone(CloneHelper &h) {
  auto src = h.GetClone(lhs_);
  auto stride_ref = h.GetClone(dst_stride_ref_);
  return new NDViewStore(addr_.gm, src, stride_ref);
}

void NDViewStore::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ViewStore";
  if (verbose) {
    int64_t offset = static_cast<int64_t>(offset_bytes_ / ITEM_SIZE[type_id_]);
    oss << "<" << offset << ", " << *dst_stride_ref_ << ">";
  }
}

void NDPadStore::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < size; i++) {
    shape_[i] = lhs_->shape_ref_->data[i];
  }
  shape_[size - 1] += pad_size_;
  dst_stride_data_.Resize(size);
  dst_stride_data_[size - 1] = 1;
  for (size_t i = size - 1; i > 0; --i) {
    dst_stride_data_[i - 1] = dst_stride_data_[i] * shape_[i];
  }
  NDViewStore::Normalize(run_ops);
}

NDObject *NDPadStore::Clone(CloneHelper &h) { return new NDPadStore(h.GetClone(lhs_), pad_size_); }

void NDPadStore::Dump(bool verbose, std::ostringstream &oss) {
  oss << "PadStore";
  if (verbose) {
    oss << "<" << pad_size_ << ">";
  }
}

void NDStore::Normalize(std::vector<NDObject *> &run_ops) {
  nd_ = lhs_->nd_;
}

void NDStore::Shard(const ShardParam &sp) {
  if (sp.sink) {
    flags_ |= OBJ_FLAG_STORE_SHARD_ROUND;
    if (nd_[sp.base + 1] == 1 && sp.tile[sp.base + 1] > 1) {
      flags_ |= OBJ_FLAG_STORE_SHARD_BCAST1;
    }
    if (nd_[sp.base] == 1 && sp.tile[sp.base] > 1) {
      flags_ |= OBJ_FLAG_STORE_SHARD_BCAST0;
    }
  }
  NDObject::Shard(sp);
}

uint64_t NDStore::Emit(VectorKernel &k) {
  if (stride_) {
    if (auto red_op = GetStoreReduceOp(lhs_); red_op != nullptr && !red_op->RoundTile().empty()) {
      return EmitAtomicView(k, red_op);
    }
    return EmitView(k, insn_);
  }
  int64_t lead_align = nd_.lead_stride();
  int64_t lead_dim = nd_.lead_dim();
  ASSERT(lead_align == lhs_->nd_.lead_stride());
  uint64_t dst_tile_stride_ = nd_.stride_back() / lead_align * lead_dim;
  if (lhs_->obj_id_ == kElementAny) {
    vStoreCond op;
    op.xn = lhs_->xbuf_;
    op.to = addr_.data;
    op.tile_stride = 0;
    op.dtype_shift = 2;
    op.pad_size = 0;
    op.iter_size = 0;
    op.round_rank = 0;
    op.cond_offset = insn_ - lhs_->tail_insn_ - vElementAny::STORE_COND_OFFSET;
    addr_.Update(insn_ + vStoreCond::RELOC_OFFSET);
    return vStoreCond::Encode(insn_, vAccInsnID::V_STORE_COND, op, nullptr);
  }
  int64_t tail_size = 0;
  if (!(flags_ & OBJ_FLAG_STORE_SHARD_ROUND)) {
    tail_size = k.GetTailSize(nd_.data);
  }
  uint64_t rounds[2];
  DimArray round_tile;
  BuildRoundTile(k, *nd_.data, (flags_ & OBJ_FLAG_STORE_SHARD_ROUND) != 0, round_tile);
  if (!round_tile.empty()) {
    BuildDimRounds(round_tile, rounds);
    if (k.comm_op_ && k.comm_op_->GetObjectType() == kReduceScatter) {
      // ReduceScatter round store accroding to rank_id
      vStoreRS op;
      op.to = addr_.data;
      op.xn = lhs_->xbuf_;
      op.iter_size = lead_dim * ITEM_SIZE[type_id_];
      op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
      op.iter_num = nd_.stride_back() / lead_align;
      op.rank_id = k.comm_op_->comm_->GetRankId();
      if (tail_size == 0) {
        op.iter_tail = op.iter_num;
      } else {
        ASSERT(op.iter_num > 1);
        op.iter_tail = op.iter_num / nd_[k.GetTailDim()] * tail_size;
      }
      op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
      op.round_rank = round_tile.size();
      addr_.Update(insn_ + vStoreRS::RELOC_OFFSET);
      return vStoreRS::Encode(insn_, V_STORE_RS, op, rounds);
    }
    if (auto red_op = GetStoreReduceOp(lhs_); red_op != nullptr) {
      if (red_op->visit_ != nullptr) {
        ASSERT(tail_size == 0);
        vStoreCond op;
        op.xn = lhs_->xbuf_;
        op.to = addr_.data;
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.dtype_shift = ITEM_SIZE[type_id_] == 4 ? 2 : 1;
        if (lead_align == lead_dim || nd_.stride_back() == lead_align) {
          op.pad_size = 0;
          op.iter_size = 0;
        } else {
          op.iter_size = lead_dim * ITEM_SIZE[type_id_];
          op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        }
        op.cond_offset = insn_ - red_op->tail_insn_ - vReduceJoin::STORE_COND_OFFSET;
        op.round_rank = round_tile.size();
        addr_.Update(insn_ + vStoreCond::RELOC_OFFSET);
        return vStoreCond::Encode(insn_, vAccInsnID::V_STORE_COND, op, rounds);
      } else {
        int code_size;
        Code &code = k.code_;
        vStoreAtomic op;
        op.to = addr_.data;
        op.xn = lhs_->xbuf_;
        op.cum_flag = red_op->CheckFlag(OBJ_FLAG_REDUCE_EMIT_CUM);
        op.iter_size = lead_dim * ITEM_SIZE[type_id_];
        op.pad_size =
          red_op->CheckFlag(OBJ_FLAG_REDUCE_EMIT_RMPAD) ? 0 : lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        op.iter_num = nd_.stride_back() / lead_align;
        if (tail_size == 0 || !red_op->nd_.data->PointwiseTile(k.GetTailDim())) {
          op.iter_tail = op.iter_num;
        } else {
          ASSERT(op.iter_num > 1);  // inner reduce is divided. outer reduce is not lead
          op.iter_tail = op.iter_num / nd_[k.GetTailDim()] * tail_size;
        }
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.round_rank = round_tile.size();
        op.red_op = red_op->red_op_;
        op.atomic_type = GetAtomicStoreType(type_id_);
        addr_.Update(insn_ + vStoreAtomic::RELOC_OFFSET);
        code_size = vStoreAtomic::Encode(insn_, V_STORE_ATOMIC, op, rounds);
        auto clean_wrap = red_op->clean_wrap_;
        if (clean_wrap == nullptr) {
          red_op->clean_wrap_ = clean_wrap = new AtomicCleanWrap(this, red_op->red_op_);
        }
        clean_wrap->CodeGen(code, this);
        code.InsertWrap(clean_wrap);
        return code_size;
      }
    }
  }
  if (lhs_->GetObjectType() == kAllGatherV2) {
    vStoreAG op;
    op.to = addr_.data;
    op.xn = k.comm_op_->xbufs_[0];
    op.iter_size = lead_dim * ITEM_SIZE[type_id_];
    op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
    op.iter_num = nd_.stride_back() / lead_align;
    if (op.iter_num == 1) {
      op.iter_tail = tail_size == 0 ? op.iter_size : tail_size * ITEM_SIZE[type_id_];
    } else {
      op.iter_tail = tail_size == 0 ? op.iter_num : op.iter_num / nd_[k.GetTailDim()] * tail_size;
    }
    // op.iter_tail = tail_size == 0 ? op.iter_num : op.iter_num / nd_[k.GetTailDim()] * tail_size;
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.rank_id = k.comm_op_->comm_->GetRankId();
    op.rank_size = k.comm_op_->comm_->GetRankSize();
    op.xbuf_size =
      reinterpret_cast<uint64_t>(k.comm_op_->xbufs_[1]) - reinterpret_cast<uint64_t>(k.comm_op_->xbufs_[0]);
    op.shard_stride = std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) *
                      ITEM_SIZE[type_id_] / op.rank_size;
    addr_.Update(insn_ + vStoreAG::RELOC_OFFSET);
    return vStoreAG::Encode(insn_, V_STORE_AG, op);
  }
  if (k.shard_) {
    vSStore op;
    op.gm = addr_.gm;
    op.xn = lhs_->xbuf_;
    op.tile_stride = dst_tile_stride_;
    op.pad_size = lead_align - lead_dim;
    op.broadcast_m = flags_ & OBJ_FLAG_STORE_SHARD_BCAST1;
    op.broadcast_n = flags_ & OBJ_FLAG_STORE_SHARD_BCAST0;
    op.round_rank = round_tile.size();
    op.type_size = ITEM_SIZE[type_id_];
    addr_.Update(insn_ + vSStore::RELOC_OFFSET);
    k.GetVisitor<MixVisitCoder>()->AddReloc(insn_, vSStore::SHARD_OFFSET);
    return vSStore::Encode(insn_, vAccInsnID::V_SSTORE, op, rounds);
  }
  vStore op;
  uint64_t iter_size = lead_dim * ITEM_SIZE[type_id_];
  bool rm_pad = lhs_->obj_id_ == kReduce ? lhs_->CheckFlag(OBJ_FLAG_REDUCE_EMIT_RMPAD) : lhs_->obj_id_ == kRemovePad;
  uint64_t pad_size = rm_pad ? 0 : lead_align * ITEM_SIZE[type_id_] - iter_size;
  uint64_t body_iter = nd_.stride_back() / lead_align;
  uint64_t tail_iter;
  if (body_iter == 1) {
    tail_iter = tail_size == 0 ? iter_size : tail_size * ITEM_SIZE[type_id_];
  } else {
    tail_iter = tail_size == 0 ? body_iter : body_iter / nd_[k.GetTailDim()] * tail_size;
    if (pad_size == 0) {
      tail_iter *= iter_size;
      iter_size *= body_iter;
      body_iter = 1;
    }
  }
  op.xn = lhs_->xbuf_;
  op.to = addr_.data;
  op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
  op.iter_num = body_iter;
  op.iter_tail = tail_iter;
  op.iter_size = iter_size;
  op.pad_size = pad_size;
  op.round_rank = round_tile.size();
  addr_.Update(insn_ + vStore::RELOC_OFFSET);
  return vStore::Encode(insn_, vAccInsnID::V_STORE, op, rounds);
}

uint64_t NDStore::EmitView(VectorKernel &k, uint64_t *insn) {
  DimArray &dst_stride_ = *stride_;
  int tail_dim = k.tile_region_.depth ? k.tile_region_.starts[k.tile_region_.depth - 1] : nd_.size() - 1;
  int64_t tail_size = k.GetTailSize(nd_.data);
  DimArray fold_dim, fold_stride;
  auto lead_idx = std::min(nd_.lead_idx(), tail_dim);
  fold_dim[0] = nd_[lead_idx];
  fold_stride[0] = dst_stride_[lead_idx];
  int last_dim = lead_idx;
  if (nd_.stride(lead_idx) == nd_[lead_idx]) {
    for (int i = last_dim + 1; i <= tail_dim; ++i) {
      if (nd_[i] == 1) continue;
      if (dst_stride_[last_dim] * nd_[last_dim] != dst_stride_[i]) {
        break;
      }
      fold_dim[0] *= nd_[i];
      last_dim = i;
    }
    lead_idx = last_dim;
  }
  int fold_idx = 0;
  for (int i = lead_idx + 1; i <= tail_dim; ++i) {
    if (nd_[i] == 1) continue;
    if (fold_idx == 0 || dst_stride_[last_dim] * nd_[last_dim] != dst_stride_[i]) {
      fold_idx++;
      fold_dim[fold_idx] = nd_[i];
      fold_stride[fold_idx] = dst_stride_[i];
    } else {
      fold_dim[fold_idx] *= nd_[i];
    }
    last_dim = i;
  }
  int tile_start = fold_idx + 1;
  if (auto &tr = k.tile_region_; tr.depth > 0) {
    auto tile_end = [this, &tr](int d) { return d >= 1 ? tr.starts[d - 1] - 1 : nd_.size(); };
    fold_idx++;
    fold_dim[fold_idx] = tr.last_num;
    if (nd_.data->PointwiseTile(tail_dim, tile_end(tr.depth - 1))) {
      fold_stride[fold_idx] = dst_stride_[tail_dim] * nd_[tail_dim];
    } else {
      fold_stride[fold_idx] = 0;
    }
    for (int d = tr.depth - 2; d >= 0; --d) {
      auto start = tr.starts[d];
      auto stride = nd_.data->PointwiseTile(start, tile_end(d)) ? dst_stride_[start] : 0;
      if (fold_stride[fold_idx] * fold_dim[fold_idx] != stride) {
        fold_idx++;
        fold_dim[fold_idx] = tr.spaces[d];
        fold_stride[fold_idx] = stride;
      } else {
        fold_dim[fold_idx] *= tr.spaces[d];
      }
    }
  }
  fold_dim.resize(fold_idx + 1);
  fold_stride.resize(fold_idx + 1);
  int64_t item_size = ITEM_SIZE[type_id_];
  if (fold_stride[0] != item_size) {
    EXCEPTION_IF(item_size != 2 && item_size != 4, "view store x only support itemsize 2\4");
    ASSERT(fold_stride[0] != 0);
    vViewStoreX op;
    op.xn = lhs_->xbuf_;
    op.to = addr_.data;
    op.offset = offset_bytes_;
    op.type_size = item_size;
    if (g_system.Arch() == AiCoreArch::kAiCore_C220) {
      op.ws = GetDynXBufBegin(k, lhs_);
      auto ws_block_num = (g_system.LocalMemSize() - op.ws) / (SIMD_BLOCK_SIZE * 2);
      op.ws_size = RoundDown(ws_block_num, SIMD_BLOCK_SIZE / item_size);
    } else {
      op.ws = 0;
      op.ws_size = 0;
    }
    op.iter_size = fold_dim[0];
    op.iter_stride = fold_stride[0];
    op.loop_depth = tile_start - 1;
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size) {
      op.tail_size = op.tail_size / nd_[tail_dim] * tail_size;
    }
    uint64_t *var_insn = insn + vViewStoreX::VAR_OFFSET;
    uint64_t src_stride = nd_.stride(lead_idx) * item_size;
    for (int i = 1; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewStore::EncodeLoop(fold_dim[i], fold_stride[i], src_stride);
      src_stride *= fold_dim[i];
    }
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewStore::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn + vViewStoreX::RELOC_OFFSET);
    return vViewStoreX::Encode(insn, V_STORE_VIEW_X, op);
  }
  vViewStore op;
  uint64_t *var_insn = insn + vViewStore::VAR_OFFSET;
  op.xn = lhs_->xbuf_;
  op.to = addr_.data;
  op.offset = offset_bytes_;
  op.iter_size = fold_dim[0] * ITEM_SIZE[type_id_];
  int loop_start;
  uint64_t src_stride = nd_.stride(lead_idx) * ITEM_SIZE[type_id_];
  if (tile_start > 1 && fold_stride[0] * fold_dim[0] <= fold_stride[1]) {
    op.src_gap = (nd_.stride(lead_idx) * ITEM_SIZE[type_id_] - op.iter_size) >> 5;
    op.dst_gap = fold_stride[1] - fold_stride[0] * fold_dim[0];
    op.iter_num = fold_dim[1];
    loop_start = 2;
    src_stride *= fold_dim[1];
  } else {
    op.src_gap = 0;
    op.dst_gap = 0;
    op.iter_num = 1;
    loop_start = 1;
  }
  op.loop_depth = tile_start - loop_start;
  for (int i = loop_start; i < tile_start; ++i, ++var_insn) {
    *var_insn = vViewStore::EncodeLoop(fold_dim[i], fold_stride[i], src_stride);
    src_stride *= fold_dim[i];
  }
  op.tail_size = fold_dim[tile_start - 1];
  if (tail_size) {
    op.tail_size = op.tail_size / nd_[tail_dim] * tail_size;
  }
  if (op.loop_depth == 0 && loop_start == 1) {
    op.tail_size *= ITEM_SIZE[type_id_];
  }
  uint64_t space = 1;
  while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
    space *= fold_dim[tile_start++];
  }
  op.tile_depth = fold_dim.size() - tile_start;
  for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
    *var_insn = vViewStore::EncodeTile(space, fold_stride[i]);
    space *= fold_dim[i];
  }
  addr_.Update(insn + vViewStore::RELOC_OFFSET);
  return vViewStore::Encode(insn, V_STORE_VIEW, op);
}

uint64_t NDStore::EmitAtomicView(VectorKernel &k, ReduceOp *red) {
  ASSERT(red->visit_ == nullptr);
  vStoreAtomicW op;
  op.red_op = red->red_op_;
  op.atomic_type = GetAtomicStoreType(type_id_);
  auto &round_tile = red->RoundTile();
  uint64_t rounds[2];
  op.round_rank = 0;
  if (red->CheckFlag(OBJ_FLAG_REDUCE_EMIT_CUM)) {
    op.round_rank = round_tile.size();
    BuildDimRounds(round_tile, rounds);
  }
  uint64_t size = vStoreAtomicW::Encode(insn_, V_STORE_ATOMIC_WRAP, op, rounds);
  size += EmitView(k, insn_ + size);
  vStoreAtomicW::EncodeSize(insn_, size);
  Code &code = k.code_;
  auto clean_wrap = red->clean_wrap_;
  if (clean_wrap == nullptr) {
    red->clean_wrap_ = clean_wrap = new AtomicCleanWrap(this, op.red_op);
  } else {
    for (auto w = code.wrap_;  w != &code; w = w->next_) {
      if (w == clean_wrap) {
        clean_wrap = nullptr;
        break;
      }
    }
  }
  if (clean_wrap) {
    clean_wrap->CodeGenView(code, this);
    code.InsertWrap(clean_wrap);
  }
  return size;
}

NDObject *NDStore::Clone(CloneHelper &h) { return new NDStore(h.GetClone(lhs_)); }

void NDStore::Dump(bool verbose, std::ostringstream &oss) { oss << "Store"; }

uint64_t CopyOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, nd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *CopyOp::Clone(CloneHelper &h) { return new CopyOp(h.GetClone(lhs_)); }

void CopyOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Copy"; }

void ReshapeOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update nd_/shape_
  auto dims = dst_shape_ref_->size;
  ndd_.Reset(dims);
  shape_.Resize(dims);
  int64_t sz = 1;
  size_t update_axis = dims;
  for (size_t i = 0; i < dims; ++i) {
    auto sh = dst_shape_ref_->data[i];
    shape_[i] = sh;
    auto nd_i = dims - i - 1;
    if (sh == -1) {
      update_axis = nd_i;
    } else {
      sz *= sh;
      ndd_.dims[nd_i] = sh;
    }
  }
  if (update_axis != dims) {
    int64_t input_sz = 1;
    auto &nd = lhs_->nd_;
    for (size_t i = 0; i < nd.size(); ++i) {
      input_sz *= nd[i];
    }
    auto v = input_sz / sz;
    ndd_.dims[update_axis] = v;
    shape_[dims - 1 - update_axis] = v;
  }
}

uint64_t ReshapeOp::Emit(VectorKernel &k) {
  ASSERT(lhs_->nd_.lead_dim() == ndd_.lead_dim());
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *ReshapeOp::Clone(CloneHelper &h) {
  auto dst_shape_ref = h.GetClone(dst_shape_ref_);
  return new ReshapeOp(h.GetClone(lhs_), dst_shape_ref);
}

void ReshapeOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Reshape"; }

bool ReshapeOp::VisitChangeRange(ChangeRange &range) {
  int out_size = ndd_.size();
  int in_size = lhs_->nd_.size();
  while (range.begin < out_size && range.begin < in_size && ndd_[range.begin] == lhs_->nd_[range.begin]) {
    range.begin++;
  }
  int out_idx = range.begin;
  int in_idx = range.begin;
  if (out_idx < out_size && in_idx < in_size) {
    size_t in_prod = lhs_->nd_[in_idx++];
    size_t out_prod = ndd_[out_idx++];
    while (in_prod != out_prod) {
      if (in_prod < out_prod) {
        in_prod *= lhs_->nd_[in_idx++];
      } else {
        out_prod *= ndd_[out_idx++];
      }
    }
  }
  if (in_idx == range.begin || out_idx == range.begin) {
    return false;
  }
  int in_one = 0;
  int out_one = 0;
  while ((in_idx + in_one) < in_size && lhs_->nd_[in_idx + in_one] == 1) in_one++;
  while ((out_idx + out_one) < out_size && ndd_[out_idx + out_one] == 1) out_one++;
  if (in_one != out_one) {
    if (in_one > out_one) {
      in_idx += in_one - out_one;
    } else {
      out_idx += out_one - in_one;
    }
  }
  range.size = out_idx - range.begin;
  range.in_size = in_idx - range.begin;
  return range.size || range.in_size;
}

void ReshapeOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ReshapeOp *>(op);
  auto &shape = self->shape_;
  auto dst_shape = self->dst_shape_ref_;
  shape.Resize(dst_shape->size);
  for (size_t i = 0; i < dst_shape->size; ++i) {
    shape[i] = dst_shape->data[i];
  }
}

ConcatOp::ConcatOp(NDObject **inputs, size_t input_num, int cat_axis)
    : FlexOp(inputs[0], inputs[1], inputs[0]->type_id_, ObjectType::kConcat), cat_axis_ref_(cat_axis) {
  shape_ref_ = &shape_;
  nd_.data = &ndd_;
  slices_.reserve(input_num);
  slices_.emplace_back(inputs[0]);
  slices_.emplace_back(inputs[1]);
  if (auto xhs_num = input_num - 2; xhs_num > 0) {
    auto xhs = reinterpret_cast<Xhs *>(new char[sizeof(Xhs) + sizeof(NDObject *) * xhs_num]);
    xhs->in_num = static_cast<int>(xhs_num);
    xhs->free_mask = 0;
    for (size_t i = 0; i < xhs_num; ++i) {
      xhs->data[i] = inputs[i + 2];
      slices_.emplace_back(inputs[i + 2]);
    }
    SetXhs(xhs);
  }
}

ConcatOp::~ConcatOp() { delete[] reinterpret_cast<char *>(xhs_); }

void ConcatOp::Normalize(std::vector<NDObject *> &run_ops) {
  int dim_size = static_cast<int>(lhs_->shape_ref_->size);
  int axis = cat_axis_ref_ >= 0 ? cat_axis_ref_ : cat_axis_ref_ + dim_size;
  int64_t cat_size = lhs_->shape_ref_->data[axis] + rhs_->shape_ref_->data[axis];
  if (xhs_) {
    for (int i = 0; i < xhs_->in_num; ++i) {
      cat_size += xhs_->data[i]->shape_ref_->data[axis];
    }
  }
  shape_ = *lhs_->shape_ref_;
  shape_[axis] = cat_size;
  ndd_.Reset(lhs_->nd_.dims());
  cat_dim_ = dim_size - 1 - axis;
  ndd_.dims[cat_dim_] = cat_size;
}

uint64_t ConcatOp::Emit(VectorKernel &k) {
  ASSERT(rhs_ == nullptr);
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *ConcatOp::Clone(CloneHelper &h) {
  std::vector<NDObject *> cloned;
  cloned.push_back(h.GetClone(lhs_));
  cloned.push_back(h.GetClone(rhs_));
  if (xhs_) {
    cloned.reserve(2 + xhs_->in_num);
    for (int i = 0; i < xhs_->in_num; ++i) {
      cloned.push_back(h.GetClone(xhs_->data[i]));
    }
  }
  return new ConcatOp(cloned.data(), cloned.size(), cat_axis_ref_);
}

void ConcatOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Concat";
  if (verbose) {
    oss << '<' << cat_axis_ref_  << '>';
  }
}

void ConcatOp::TileCollect(NDObject *op, TileInfo &info) {
  auto cat_depth = static_cast<ConcatOp *>(op)->cat_dim_ + 1;
  if (info.lead_depth > cat_depth && op->rhs_ != nullptr) {
    info.lead_depth = cat_depth;
  }
}

void ConcatOp::FoldProp(NDObject *op, PropRange &range) {
  auto cat_dim = static_cast<ConcatOp *>(op)->cat_dim_;
  if (op->rhs_ == nullptr) return;
  if (range.base > cat_dim) {
    if (auto depth_len = range.base - cat_dim; depth_len < range.depth) {
      range.depth = depth_len;
    }
  } else if (range.base == cat_dim) {
    range.depth = 0; // only for reshape prop
  }
}

void ConcatOp::DimChanged(NDObject *op) {
  for (size_t i = 0; i < op->nd_.size(); ++i) {
    if (i >= op->lhs_->nd_.size() || op->lhs_->nd_[i] != op->nd_[i]) {
      static_cast<ConcatOp *>(op)->cat_dim_ = i;
      break;
    }
  }
}

void ConcatOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ConcatOp *>(op);
  int axis = self->cat_axis_ref_;
  if (axis < 0) {
    axis += static_cast<int>(self->lhs_->shape_ref_->size);
  }
  int64_t cat_size = 0;
  auto lhs_slice = self->lhs_->shape_ref_->data[axis];
  auto rhs_slice = self->rhs_->shape_ref_->data[axis];
  if (lhs_slice > 0 && rhs_slice > 0) {
    cat_size = lhs_slice + rhs_slice;
    if (auto xhs = self->xhs_) {
      for (int i = 0; i < xhs->in_num; ++i) {
        auto size = xhs->data[i]->shape_ref_->data[axis];
        if (size < 0) {
          cat_size = 0;
          break;
        }
        cat_size += size;
      }
    }
  }
  self->shape_ = *(self->lhs_->shape_ref_);
  self->shape_[axis] = cat_size ? cat_size : sym_dim_next--;
}

uint64_t SplitOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *SplitOp::Clone(CloneHelper &h) {
  auto main = static_cast<SplitOpM *>(h.GetClone(main_));
  return main->AddSibling();
}

void SplitOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Split";
  if (verbose) {
    oss << '<' << slice_idx_ << '>';
  }
}

void SplitOp::TileCollect(NDObject *op, TileInfo &info) {
  auto split = static_cast<SplitOp *>(op);
  if (split->slice_idx_ > 0) {
    return;
  }
  auto split_dim = static_cast<SplitOpM *>(op)->split_dim_;
  if (split->ndd_[split_dim] != split->lhs_->nd_[split_dim]) {
    if (auto split_depth = split_dim + 1; info.lead_depth > split_depth) {
      info.lead_depth = split_depth;
    }
  }
}

void SplitOp::FoldProp(NDObject *op, PropRange &range) {
  auto split = static_cast<SplitOp *>(op);
  if (split->slice_idx_ > 0) {
    return;
  }
  auto split_dim = static_cast<SplitOpM *>(op)->split_dim_;
  if (split->ndd_[split_dim] != split->lhs_->nd_[split_dim]) {
    if (range.base > split_dim) {
      if (auto depth_len = range.base - split_dim; depth_len < range.depth) {
        range.depth = depth_len;
      }
    } else if (range.base == split_dim) {
      range.depth = 0; // only for reshape prop
    }
  }
}

void SplitOp::DimChanged(NDObject *op) {
  if (static_cast<SplitOp *>(op)->slice_idx_ == 0) {
    auto &dims = op->nd_.dims();
    auto &in_dims = op->lhs_->nd_.dims();
    for (size_t i = 0; i < op->nd_.size(); ++i) {
      if (i >= in_dims.size() || in_dims[i] != dims[i]) {
        static_cast<SplitOpM *>(op)->split_dim_ = i;
        break;
      }
    }
  }
}

void SplitOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  if (static_cast<SplitOp *>(op)->slice_idx_ == 0) {
    auto *main = static_cast<SplitOpM *>(op);
    int dim_size = static_cast<int>(main->lhs_->shape_ref_->size);
    int axis = main->split_axis_ref_ >= 0 ? main->split_axis_ref_ : main->split_axis_ref_ + dim_size;
    main->split_dim_ = dim_size - 1 - axis;
    auto input_shape = main->lhs_->shape_ref_;
    if (int64_t input_dim = input_shape->data[axis]; input_dim > 0) {
      for (size_t i = 0; i < main->siblings_.size(); ++i) {
        auto op = main->siblings_[i];
        op->shape_ = *input_shape;
        op->shape_[axis] = main->split_size_;
      }
      auto op = main->siblings_.back();
      op->shape_[axis] = input_dim - main->split_size_ * (main->siblings_.size() - 1);
    } else {
      auto main_sym = sym_dim_next--;
      for (size_t i = 0; i < main->siblings_.size(); ++i) {
        auto op = main->siblings_[i];
        op->shape_ = *input_shape;
        op->shape_[axis] = main_sym;
      }
      auto op = main->siblings_.back();
      op->shape_[axis] = sym_dim_next--;
    }
  }
}

void SplitOp::Normalize(std::vector<NDObject *> &run_ops) {
  auto sync = norm_sync_;
  norm_sync_ += 2;
  if (sync == main_->norm_sync_) {
    main_->norm_sync_++;
    main_->DoNormalize(run_ops);
  }
}

void SplitOpM::Normalize(std::vector<NDObject *> &run_ops) {
  if ((norm_sync_ & 1) == 0) {
    norm_sync_ += 2;
    DoNormalize(run_ops);
  } else {
    norm_sync_++;
  }
}

void SplitOpM::DoNormalize(std::vector<NDObject *> &run_ops) {
  int dim_size = static_cast<int>(lhs_->shape_ref_->size);
  int axis = split_axis_ref_ >= 0 ? split_axis_ref_ : split_axis_ref_ + dim_size;
  split_dim_ = dim_size - 1 - axis;
  for (size_t i = 0; i < siblings_.size(); ++i) {
    auto op = siblings_[i];
    op->shape_ = *lhs_->shape_ref_;
    op->shape_[axis] = split_size_;
    op->ndd_.Reset(lhs_->nd_.dims());
    op->ndd_.dims[split_dim_] = split_size_;
  }
  auto tail_size = lhs_->shape_ref_->data[axis] - split_size_ * (siblings_.size() - 1);
  auto op = siblings_.back();
  op->shape_[axis] = tail_size;
  op->ndd_.dims[split_dim_] = tail_size;
}

NDObject *SplitOpM::Clone(CloneHelper &h) {
  return new SplitOpM(h.GetClone(lhs_), split_axis_ref_, split_size_, siblings_.size());
}

uint64_t SliceOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

void SliceOp::TileCollect(NDObject *op, TileInfo &info) {
  auto &sdims = static_cast<SliceOp *>(op)->sdims_;
  ASSERT(!sdims.empty());
  if (int i = sdims.front().index; i + 1 < info.lead_depth && op->lhs_->nd_[i] != op->nd_[i]) {
    info.lead_depth = i + 1;
  }
}

void SliceOp::FoldProp(NDObject *op, PropRange &range) {
  auto &sdims = static_cast<SliceOp *>(op)->sdims_;
  ASSERT(!sdims.empty());
  size_t sidx = sdims.size() - 1;
  int index = sdims[sidx].index;
  while (sidx > 0 && range.base < index) {
    index = sdims[--sidx].index;
  }
  if (range.base >= index && op->lhs_->nd_[index] != op->nd_[index]) {
    auto new_depth = range.base > index ? range.base - index : 1;
    if (new_depth < range.depth) {
      range.depth = new_depth;
    }
  }
}

void SliceOp::DimChanged(NDObject *op) {
  auto &sdims = static_cast<SliceOp *>(op)->sdims_;
  auto &nd = static_cast<SliceOp *>(op)->ndd_.dims;
  int index = 0;
  int offset = 0;
  for (auto &sd : sdims) {
    while (offset < sd.offset) {
      if (index == 0) {
        offset = nd[0];
        index = 1;
      } else {
        offset *= nd[index];
        index++;
      }
    }
    ASSERT(offset == sd.offset);
    sd.index = index;
  }
}

void SliceOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) { static_cast<SliceOp *>(op)->DoShapeProp(sym_dim_next); }

void PermuteOp::Normalize(std::vector<NDObject *> &run_ops) {
  size_t dim_size = lhs_->shape_ref_->size;
  for (size_t k = 0; k < dim_size; ++k) {
    perm_[k] = dim_size - 1 - dims_ref_->data[dim_size - 1 - k];
  }
  perm_.resize(dim_size);
  ndd_.Reset(dim_size);
  shape_.Resize(dim_size);
  for (size_t k = 0; k < dim_size; ++k) {
    ndd_.dims[k] = lhs_->nd_.data->dims[perm_[k]];
  }
  for (size_t k = 0; k < dim_size; ++k) {
    shape_[dim_size - 1 - k] = ndd_.dims[k];
  }
}

uint64_t PermuteOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *PermuteOp::Clone(CloneHelper &h) { return new PermuteOp(h.GetClone(lhs_), h.GetClone(dims_ref_)); }

void PermuteOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Permute"; }

void PermuteOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<PermuteOp *>(op);
  auto &shape = self->shape_;
  auto dims_ref = self->dims_ref_;
  auto input_shape = self->lhs_->shape_ref_;
  shape.Resize(dims_ref->size);
  for (size_t i = 0; i < dims_ref->size; ++i) {
    shape[i] = input_shape->data[dims_ref->data[i]];
  }
}

uint64_t UnaryOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  ASSERT(size_t(op_type_) < sizeof(unary_id_list) / sizeof(InsnIdTable));
  auto id = unary_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

NDObject *UnaryOp::Clone(CloneHelper &h) { return new UnaryOp(op_type_, h.GetClone(lhs_)); }

void UnaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << unary_id_list[op_type_].name; }

void UnaryOp::Inspect(Inspector &sp) {
  if (auto id = unary_id_list[op_type_].ids[type_id_]; id == V_NONE) {
    sp.Error("NoInsn") << "unary map to none insn: op=" << op_type_ << ", dtype=" << type_id_ << ".\n";
  }
}

uint64_t RemovePadOp::Emit(VectorKernel &k) {
  if (nd_.lead_dim() == nd_.lead_stride() || nd_.stride_back() == nd_.lead_stride()) {
    return CopyOp::Emit(k);
  }
  vRemovePad op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = nd_.stride_back() / nd_.lead_stride();
  op.iter_num = nd_.lead_dim();
  op.rs = GetBlocks(nd_.lead_stride());
  return vRemovePad::Encode(insn_, removepad_id_list[type_id_], op);
}

NDObject *RemovePadOp::Clone(CloneHelper &h) { return new RemovePadOp(h.GetClone(lhs_)); }

void RemovePadOp::Dump(bool verbose, std::ostringstream &oss) { oss << "RemovePad"; }

void ElementAnyOp::Normalize(std::vector<NDObject *> &run_ops) {
  ndd_.Reset();
  ndd_.dims.resize(lhs_->nd_.size(), 1);
}

uint64_t ElementAnyOp::Emit(VectorKernel &k) {
  uint64_t clr_tail = 0;
  vElementAny op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.simd_width = SelectSimdWidth(lhs_->nd_.stride_back(), type_id_);
  op.repeat = lhs_->nd_.stride_back() / op.simd_width;
  int64_t tail_size = k.GetTailSize(lhs_->nd_.data);
  if (tail_size == 0) {
    op.repeat_tail = op.repeat;
  } else if (k.GetTailDim() <= ndd_.lead_idx()) {
    uint64_t tail_size_u = static_cast<uint64_t>(tail_size);
    op.repeat_tail = CeilDiv(tail_size_u, op.simd_width);
    if (op.repeat_tail * op.simd_width != tail_size_u) {
      clr_tail = tail_size_u;
    }
  } else {
    op.repeat_tail = op.repeat / lhs_->nd_[k.GetTailDim()] * tail_size;
  }
  uint32_t insn_num = 1;
  uint64_t size = 0;
  if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail) {
    size = EmitClearPad(insn_, lhs_, clr_tail, op.simd_width);
    tail_insn_ = insn_ + size;
    insn_num++;
  }
  size += vElementAny::Encode(tail_insn_, V_ELEMENT_ANY, op);

  if (insn_num > 1) {
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return size;
}

NDObject *ElementAnyOp::Clone(CloneHelper &h) { return new ElementAnyOp(h.GetClone(lhs_)); }

void ElementAnyOp::Dump(bool verbose, std::ostringstream &oss) { oss << "ElementAny"; }

uint64_t CastOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  auto id = cast_id_list[lhs_->type_id_][type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

NDObject *CastOp::Clone(CloneHelper &h) { return new CastOp(h.GetClone(lhs_), type_id_); }

void CastOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Cast"; }

void ExtractOp::Normalize(std::vector<NDObject *> &run_ops) { nd_ = lhs_->nd_; }

uint64_t ExtractOp::Emit(VectorKernel &k) {
  vExtract op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  op.slot = slot_;
  ASSERT(type_id_ == DataType::kInt32 || type_id_ == DataType::kFloat32);
  return vExtract::Encode(insn_, V_EXTRACT_B32, op);
}

NDObject *ExtractOp::Clone(CloneHelper &h) { return new ExtractOp(h.GetClone(lhs_), slot_, type_id_); }

void ExtractOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Extract<" << slot_ << '>'; }

uint64_t PackOp::Emit(VectorKernel &k) {
  vPack op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back() * 2;
  op.ws = wss_[0];
  return vPack::Encode(insn_, V_PACK_B32, op);
}

NDObject *PackOp::Clone(CloneHelper &h) { return new PackOp(h.GetClone(lhs_), h.GetClone(rhs_)); }

void PackOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Pack"; }

uint64_t BinaryScalarOp::Emit(VectorKernel &k) {
  vBinaryS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = scalar_;
  ASSERT(size_t(op_type_) < sizeof(binarys_id_list) / sizeof(InsnIdTable));
  auto id = binarys_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vBinaryS::Encode(insn_, id, op);
}

NDObject *BinaryScalarOp::Clone(CloneHelper &h) { return new BinaryScalarOp(op_type_, h.GetClone(lhs_), scalar_); }

void BinaryScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << binarys_id_list[op_type_].name;
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

void BinaryScalarOp::Inspect(Inspector &sp) {
  if (auto id = binarys_id_list[op_type_].ids[type_id_]; id == V_NONE) {
    sp.Error("NoInsn") << "binarys map to none insn: op=" << op_type_ << ", dtype=" << type_id_ << ".\n";
  }
}

uint64_t CompareScalarOp::Emit(VectorKernel &k) {
  vCompareS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = scalar_;
  op.ws = wss_[0];
  op.type = cmp_op_;
  auto id = binarys_id_list[cmp_op_].ids[type_id_];
  return vCompareS::Encode(insn_, id, op);
}

NDObject *CompareScalarOp::Clone(CloneHelper &h) { return new CompareScalarOp(cmp_op_, h.GetClone(lhs_), scalar_); }

void CompareScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "CompareS";
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

_BinaryNormalizer::~_BinaryNormalizer() {
  delete lhs_stuff_;
  delete rhs_stuff_;
}

void _BinaryNormalizer::Normalize(NDObject *self, std::vector<NDObject *> &run_ops) {
  // recover original input
  if (lhs_stuff_) {
    self->lhs_ = lhs_stuff_->lhs_;
  }
  if (rhs_stuff_) {
    self->rhs_ = rhs_stuff_->lhs_;
  }
  // update shape_ref_
  auto lhs_data = self->lhs_->shape_ref_->data;
  auto rhs_data = self->rhs_->shape_ref_->data;
  auto lhs_sz = self->lhs_->shape_ref_->size;
  auto rhs_sz = self->rhs_->shape_ref_->size;
  bool lhs_broadcast = false;
  bool rhs_broadcast = false;
  if (lhs_sz >= rhs_sz) {
    shape_.Resize(lhs_sz);
    std::memcpy(shape_.shape, lhs_data, sizeof(int64_t) * lhs_sz);
    auto diff = lhs_sz - rhs_sz;
    for (size_t i = 0; i < lhs_sz; ++i) {
      if (int64_t r_val = i >= diff ? rhs_data[i - diff] : 1; r_val != shape_[i]) {
        if (r_val != 1) {
          shape_[i] = r_val;
          lhs_broadcast = true;
        } else {
          rhs_broadcast = true;
        }
      }
    }
  } else {
    shape_.Resize(rhs_sz);
    std::memcpy(shape_.shape, rhs_data, sizeof(int64_t) * rhs_sz);
    auto diff = rhs_sz - lhs_sz;
    for (size_t i = 0; i < rhs_sz; ++i) {
      if (int64_t l_val = i >= diff ? lhs_data[i - diff] : 1; l_val != shape_[i]) {
        if (l_val != 1) {
          shape_[i] = l_val;
          rhs_broadcast = true;
        } else {
          lhs_broadcast = true;
        }
      }
    }
  }
  // update nd_
  auto insert_stuff = [this, self, &run_ops](NDObject *input, NDObject *&stuff) {
    NDObject *op;
    if (self->flags_ & OBJ_FLAG_EAGER) {
      op = new BroadcastOp(input, &shape_);
    } else {
      op = stuff;
      if (op == nullptr) {
        stuff = op = new BroadcastOp(input, &shape_);
      }
    }
    op->Normalize(run_ops);
    run_ops.push_back(op);
    return op;
  };
  if (lhs_broadcast) {
    self->lhs_ = insert_stuff(self->lhs_, lhs_stuff_);
  }
  if (rhs_broadcast) {
    self->rhs_ = insert_stuff(self->rhs_, rhs_stuff_);
  }
  self->nd_ = self->lhs_->nd_;
}

uint64_t BinaryOp::Emit(VectorKernel &k) {
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back();
  ASSERT(size_t(op_type_) < sizeof(binary_id_list) / sizeof(InsnIdTable));
  auto id = binary_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vBinary::Encode(insn_, id, op);
}

NDObject *BinaryOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ == nullptr ? lhs_ : norm_.lhs_stuff_->lhs_;
  auto rhs = norm_.rhs_stuff_ == nullptr ? rhs_ : norm_.rhs_stuff_->lhs_;
  return new BinaryOp(op_type_, h.GetClone(lhs), h.GetClone(rhs));
}

void BinaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << binary_id_list[op_type_].name; }

void BinaryOp::Inspect(Inspector &sp) {
  if (auto id = binary_id_list[op_type_].ids[type_id_]; id == V_NONE) {
    sp.Error("NoInsn") << "binary map to none insn: op=" << op_type_ << ", dtype=" << type_id_ << ".\n";
  }
}

static void BinaryShapeProp(const IntArrayRef *lhs, const IntArrayRef *rhs, ShapeWithRef &shape,
                            int64_t &sym_dim_next) {
  if (lhs->size < rhs->size) {
    auto tmp = lhs;
    lhs = rhs;
    rhs = tmp;
  }
  shape.Resize(lhs->size);
  auto diff = lhs->size - rhs->size;
  for (size_t i = 0; i < diff; ++i) {
    shape[i] = lhs->data[i];
  }
  for (size_t i = diff; i < lhs->size; ++i) {
    auto l_dim = lhs->data[i];
    auto r_dim = rhs->data[i - diff];
    if (l_dim == r_dim || r_dim == 1) {
      shape[i] = l_dim;
    } else if (l_dim == 1) {
      shape[i] = r_dim;
    } else {
      shape[i] = sym_dim_next--;
    }
  }
}

void BinaryOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<BinaryOp *>(op)->norm_.shape_,
                         sym_dim_next);
}

uint64_t CompareOp::Emit(VectorKernel &k) {
  vCompare op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.type = cmp_op_;
  op.ws = wss_[0];
  op.count = nd_.stride_back();
  auto id = binary_id_list[cmp_op_].ids[lhs_->type_id_];
  ASSERT(id != V_NONE);
  return vCompare::Encode(insn_, id, op);
}

void CompareOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Compare";
  if (verbose) {
    oss << "<" << cmp_op_ << ">";
  }
}

NDObject *CompareOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ == nullptr ? lhs_ : norm_.lhs_stuff_->lhs_;
  auto rhs = norm_.rhs_stuff_ == nullptr ? rhs_ : norm_.rhs_stuff_->lhs_;
  return new CompareOp(cmp_op_, h.GetClone(lhs), h.GetClone(rhs));
}

void CompareOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<CompareOp *>(op)->norm_.shape_,
                         sym_dim_next);
}

uint64_t PowerOp::Emit(VectorKernel &k) {
  ASSERT(type_id_ == dvm::kFloat32);
  vBinaryWS op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back();
  op.ws0 = wss_[0];
  op.ws1 = wss_[1];
  return vBinaryWS::Encode(insn_, V_POW, op);
}

NDObject *PowerOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ == nullptr ? lhs_ : norm_.lhs_stuff_->lhs_;
  auto rhs = norm_.rhs_stuff_ == nullptr ? rhs_ : norm_.rhs_stuff_->lhs_;
  return new PowerOp(h.GetClone(lhs), h.GetClone(rhs));
}

void PowerOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Power"; }

void PowerOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<PowerOp *>(op)->norm_.shape_,
                         sym_dim_next);
}

void SelectOp::Normalize(std::vector<NDObject *> &run_ops) {
  // recover original input
  if (stuff_ops_[0]) lhs_ = stuff_ops_[0]->lhs_;
  if (stuff_ops_[1]) rhs_ = stuff_ops_[1]->lhs_;
  if (stuff_ops_[2]) xhs_->data[0] = stuff_ops_[2]->lhs_;
  auto l_shape = lhs_->shape_ref_;
  auto r_shape = rhs_->shape_ref_;
  auto x_shape = xhs_->data[0]->shape_ref_;
  auto max_size = std::max({l_shape->size, r_shape->size, x_shape->size});
  shape_.Resize(max_size);
  bool lhs_bc = false;
  bool rhs_bc = false;
  bool xhs_bc = false;
  for (size_t i = 0; i < max_size; ++i) {
    int64_t l_val = i < l_shape->size ? l_shape->data[l_shape->size - 1 - i] : 1;
    int64_t r_val = i < r_shape->size ? r_shape->data[r_shape->size - 1 - i] : 1;
    int64_t x_val = i < x_shape->size ? x_shape->data[x_shape->size - 1 - i] : 1;
    int64_t out_val = std::max({l_val, r_val, x_val});
    if (out_val != 1) {
      if (l_val != out_val) lhs_bc = true;
      if (r_val != out_val) rhs_bc = true;
      if (x_val != out_val) xhs_bc = true;
    }
    shape_[max_size - 1 - i] = out_val;
  }
  // update nd_
  auto insert_stuff = [this, &run_ops](NDObject *input, int stuff_idx) {
    NDObject *op;
    if (flags_ & OBJ_FLAG_EAGER) {
      op = new BroadcastOp(input, &shape_);
    } else {
      op = stuff_ops_[stuff_idx];
      if (op == nullptr) {
        stuff_ops_[stuff_idx] = op = new BroadcastOp(input, &shape_);
      }
    }
    op->Normalize(run_ops);
    run_ops.push_back(op);
    return op;
  };
  if (lhs_bc) lhs_ = insert_stuff(lhs_, 0);
  if (rhs_bc) rhs_ = insert_stuff(rhs_, 1);
  if (xhs_bc) xhs_->data[0] = insert_stuff(xhs_->data[0], 2);
  nd_ = lhs_->nd_;
}

void SelectOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *select = static_cast<SelectOp *>(op);
  const IntArrayRef *inputs[] = {
      select->xhs_->data[0]->shape_ref_,
      select->lhs_->shape_ref_,
      select->rhs_->shape_ref_,
  };
  size_t max_size = 0;
  for (auto input : inputs) {
    max_size = std::max(max_size, input->size);
  }
  select->shape_.Resize(max_size);
  for (size_t i = 0; i < max_size; ++i) {
    int64_t result = 1;
    for (auto input : inputs) {
      size_t offset = max_size - input->size;
      int64_t dim = i < offset ? 1 : input->data[i - offset];
      if (dim == 1 || result == dim) {
        continue;
      }
      if (result == 1) {
        result = dim;
      } else {
        result = sym_dim_next--;
      }
    }
    select->shape_[i] = result;
  }
}

SelectOp::~SelectOp() {
  delete stuff_ops_[0];
  delete stuff_ops_[1];
  delete stuff_ops_[2];
}

uint64_t SelectOp::Emit(VectorKernel &k) {
  vSelect op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  const static vSimdInsnID id_list[DataType::kDataTypeEnd] = {V_NONE, V_SEL_FP16,  V_SEL_BF16,
                                                              V_SEL,  V_SEL_INT32, V_SEL_INT64};
  op.count = nd_.stride_back();
  op.xm = rhs_->xbuf_;
  op.cond = xhs_->data[0]->xbuf_;
  op.ws = wss_[0];
  ASSERT(id_list[type_id_] != V_NONE);
  return vSelect::Encode(insn_, id_list[type_id_], op);
}

NDObject *SelectOp::Clone(CloneHelper &h) {
  auto lhs = stuff_ops_[0] == nullptr ? lhs_ : stuff_ops_[0]->lhs_;
  auto rhs = stuff_ops_[1] == nullptr ? rhs_ : stuff_ops_[1]->lhs_;
  auto xhs = stuff_ops_[2] == nullptr ? xhs_->data[0] : stuff_ops_[2]->lhs_;
  return new SelectOp(h.GetClone(xhs), h.GetClone(lhs), h.GetClone(rhs));
}

void SelectOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Select"; }

void BroadcastOp::Normalize(std::vector<NDObject *> &run_ops) {
  auto dims = dst_shape_ref_->size;
  ndd_.Reset(dims);
  shape_.Resize(dims);
  for (size_t i = 0; i < dims; ++i) {
    shape_[i] = dst_shape_ref_->data[i];
    if (shape_[i] == -1) {
      // e.g. x_shape (4, 1), dst_shape (2, -1, 1) --> dst_shape (2, 4, 1)
      shape_[i] = lhs_->shape_ref_->data[lhs_->shape_ref_->size + i - dims];
    }
    ndd_.dims[dims - 1 - i] = shape_[i];
  }
}

uint64_t BroadcastOp::Emit(VectorKernel &k) {
  DimArray ranges;
  int ridx = 0;
  for (size_t i = ndd_.lead_idx(); i < ndd_.size(); ++i) {
    if ((ridx & 1) == 0) {
      if (ndd_[i] != lhs_->nd_[i]) {
        ranges[ridx++] = i;
      }
    } else if (lhs_->nd_[i] > 1) {
      ranges[ridx++] = i - 1;
    }
  }
  if (ridx == 0) {
    return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
  }
  if (ridx & 1) {
    ranges[ridx++] = ndd_.size() - 1;
  }
  const static vSimdInsnID x_id_list[DataType::kDataTypeEnd] = {
    V_BROADCAST_X_B8, V_BROADCAST_X_B16, V_BROADCAST_X_B16, V_BROADCAST_X_B32, V_BROADCAST_X_B32, V_BROADCAST_X_B64};
  ASSERT(x_id_list[type_id_] != V_NONE);
  auto emit_broadcast_y = [this](uint64_t *insn, int64_t start_dim, int64_t end_dim, uint64_t xn, uint64_t xd) {
    vBroadcastY op;
    op.xn = xn;
    op.xd = xd;
    op.iter_num = ndd_.stride_back() / ndd_.stride(end_dim);
    op.dup_num = ndd_.stride(end_dim) / ndd_.stride(start_dim - 1);
    op.dup_stride = ndd_.stride(start_dim - 1) * ITEM_SIZE[type_id_] / SIMD_BLOCK_SIZE;
    return vBroadcastY::Encode(insn, V_BROADCAST_Y, op);
  };
  uint64_t xout = ridx & 2 ? xbuf_ : wss_[0];
  uint64_t size = 0;
  if (ranges[0] == ndd_.lead_idx()) {
    int64_t end_dim = ranges[1];
    int64_t rank_size = static_cast<int64_t>(ndd_.size());
    int64_t lead_num = end_dim + 1 < rank_size ? ndd_[end_dim + 1] : 1;
    int64_t iter_num = end_dim + 2 < rank_size ? ndd_.stride_back() / ndd_.stride(end_dim + 1) : 1;
    int64_t lead_pad = lhs_->nd_.lead_stride() - lhs_->nd_.lead_dim();
    if (lead_pad == 0) {
      lead_num *= iter_num;
      iter_num = 1;
    }
    if (g_system.Arch() == kAiCore_C220 && ws_num_ > 0 && iter_num > 1 && lead_pad > 0) {
      uint64_t pad_out = wss_[0];
      if (!(ridx & 2)) {
        std::swap(xout, pad_out);
      }
      vRemovePad compact;
      compact.xd = pad_out;
      compact.xn = lhs_->xbuf_;
      compact.repeat = iter_num;
      compact.iter_num = lead_num;
      compact.rs = GetBlocks(lhs_->nd_.lead_stride());
      size = vRemovePad::Encode(insn_, removepad_id_list[type_id_], compact);
      vBroadcastX op;
      op.xd = xout;
      op.xn = pad_out;
      op.count = ndd_.stride(end_dim);
      op.lead_num = lead_num * iter_num;
      op.iter_num = 1;
      op.lead_pad = 0;
      tail_insn_ = insn_ + size;
      size += vBroadcastX::Encode(tail_insn_, x_id_list[type_id_], op);
    } else {
      uint64_t xin = lhs_->xbuf_;
      if (xout == xin) {
        size = EmitCopy(insn_, xbuf_, xin, lhs_->nd_.stride_back() * ITEM_SIZE[type_id_]);
        tail_insn_ = insn_ + size;
        xin = xbuf_;
      }
      vBroadcastX op;
      op.xd = xout;
      op.xn = xin;
      op.count = ndd_.stride(end_dim);
      op.lead_num = lead_num;
      op.iter_num = iter_num;
      op.lead_pad = lead_pad;
      size += vBroadcastX::Encode(tail_insn_, x_id_list[type_id_], op);
    }
  } else {
    uint64_t xin = lhs_->xbuf_;
    if (xout == xin) {
      size = EmitCopy(insn_, xbuf_, xin, lhs_->nd_.stride_back() * ITEM_SIZE[type_id_]);
      tail_insn_ = insn_ + size;
      xin = xbuf_;
    }
    size += emit_broadcast_y(tail_insn_, ranges[0], ranges[1], xin, xout);
  }
  if (insn_ != tail_insn_) {
    tail_insn_[0] |= 1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  if (ridx > 2) {
    uint64_t xin = xout == xbuf_ ? wss_[0] : xbuf_;
    for (int64_t i = 2; i < ridx; i += 2) {
      std::swap(xout, xin);
      tail_insn_ = insn_ + size;
      size += emit_broadcast_y(tail_insn_, ranges[i], ranges[i + 1], xin, xout);
      tail_insn_[0] |= 1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
  }
  ASSERT(insn_ == tail_insn_ || ws_num_ > 0);
  return size;
}

void BroadcastOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Broadcast"; }

NDObject *BroadcastOp::Clone(CloneHelper &h) {
  return new BroadcastOp(h.GetClone(lhs_), h.GetClone(dst_shape_ref_));
}

void BroadcastOp::TileCollect(NDObject *op, TileInfo &info) {
  auto self = static_cast<BroadcastOp *>(op);
  auto &lhs_nd = op->lhs_->nd_.dims();
  auto &ndd = self->ndd_.dims;
  if (info.IsGen()) {
    uint64_t range_num = 0;
    bool in_broadcast = false;
    for (size_t i = 0; i < ndd.size(); ++i) {
      if (!in_broadcast) {
        if (ndd[i] != lhs_nd[i]) {
          in_broadcast = true;
          range_num++;
        }
      } else if (lhs_nd[i] != 1) {
        in_broadcast = false;
      }
    }
    if (range_num > 1) {
      info.code_reserve += sizeof(uint64_t) * 2 * (range_num - 1);
    }
    auto broadcast_x = [&lhs_nd, &ndd]() {
      auto lead = ndd.lead_dim();
      return lead < lhs_nd.size() && ndd[lead] != lhs_nd[lead];
    };
    if (range_num > 1 || (g_system.Arch() == kAiCore_C220 && broadcast_x())) {
      self->SetWs(1, true);
    } else {
      self->UnsetWs();
    }
  }
  BroadReduceTileCollect<PropRange::BROADCAST>(lhs_nd, ndd, info);
}

void BroadcastOp::FoldProp(NDObject *op, PropRange &range) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<BroadcastOp *>(op)->ndd_;
  BroadReduceFoldProp<PropRange::BROADCAST>(lhs_nd.data->dims, ndd.dims, range);
}

void BroadcastOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<BroadcastOp *>(op);
  auto &shape = self->shape_;
  auto dst_shape = self->dst_shape_ref_;
  shape.Resize(dst_shape->size);
  for (size_t i = 0; i < dst_shape->size; ++i) {
    shape[i] = dst_shape->data[i];
  }
}

BroadcastScalarOp::~BroadcastScalarOp() {
  if (dummy_load_) {
    delete dummy_load_;
  }
}

void BroadcastScalarOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update nd_ from shape_ref_
  auto dims = shape_ref_->size;
  ndd_.Reset(dims);
  for (size_t i = 0; i < dims; ++i) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
  }
  if (flags_ & OBJ_FLAG_EAGER) {
    lhs_ = new NDLoadDummy(type_id_);
    lhs_->SetFlag(OBJ_FLAG_EAGER);
    run_ops.push_back(lhs_);
  } else if (run_ops.empty()) {
    if (dummy_load_ == nullptr) {
      dummy_load_ = new NDLoadDummy(type_id_);
    }
    lhs_ = dummy_load_;
    run_ops.push_back(dummy_load_);
  } else {
    lhs_ = nullptr;
  }
}

uint64_t BroadcastScalarOp::Emit(VectorKernel &k) {
  vBroadcastS op;
  op.scalar = scalar_;
  op.xd = xbuf_;
  op.count = ndd_.stride_back();
  const static vSimdInsnID id_list[SIMD_DTYPE_END] = {V_BROADCAST_S_B8, V_BROADCAST_S_B16, V_BROADCAST_S_B16,
                                                      V_BROADCAST_S, V_BROADCAST_S};
  return vBroadcastS::Encode(insn_, id_list[type_id_], op);
}

NDObject *BroadcastScalarOp::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  return new BroadcastScalarOp(scalar_, shape_ref, type_id_);
}

void BroadcastScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "BroadcastS";
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

BroadcastInt64ScalarOp::BroadcastInt64ScalarOp(uint64_t scalar, IntArrayRef *shape_ref)
    : BroadcastScalarOp(static_cast<scode_t>(scalar & 0xfffffffful), shape_ref, DataType::kInt64),
      high_(static_cast<scode_t>((scalar >> 32) & 0xfffffffful)) {}

uint64_t BroadcastInt64ScalarOp::Emit(VectorKernel &k) {
  vBroadcastS_B64 op;
  op.scalar = scalar_;
  op.high = high_;
  op.xd = xbuf_;
  op.count = ndd_.stride_back();
  return vBroadcastS_B64::Encode(insn_, V_BROADCAST_S_B64, op);
}

NDObject *BroadcastInt64ScalarOp::Clone(CloneHelper &h) {
  uint64_t scalar = (static_cast<uint64_t>(high_) << 32) | scalar_;
  return new BroadcastInt64ScalarOp(scalar, h.GetClone(shape_ref_));
}

void BroadcastInt64ScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "BroadcastS";
  if (verbose) {
    uint64_t scalar = (static_cast<uint64_t>(high_) << 32) | scalar_;
    oss << "<" << static_cast<int64_t>(scalar) << ">";
  }
}

void ReduceOp::FoldProp(NDObject *op, PropRange &range) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<ReduceOp *>(op)->ndd_;
  BroadReduceFoldProp<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, range);
}

void ReduceOp::TileCollect(NDObject *op, TileInfo &info) {
  auto self = static_cast<ReduceOp *>(op);
  auto &lhs_nd = self->lhs_->nd_;
  auto &ndd = self->ndd_;
  if (info.IsGen()) {
    int range_num = 0;
    bool in_red = false;
    for (size_t i = 0; i < ndd.size(); ++i) {
      if (!in_red) {
        if (ndd[i] != lhs_nd[i]) {
          in_red = true;
          range_num++;
        }
      } else if (ndd[i] != 1) {
        in_red = false;
      }
    }
    if (range_num > 1) {
      info.code_reserve += sizeof(uint64_t) * 3 * range_num;
    }
    if (self->ws_num_ == 2) {
      info.code_reserve += RedVisitCoder::BCODE_MAX + vReduceJoin::CODE_SIZE * sizeof(uint64_t);
      if (info.event_reserve < 2) {
        info.event_reserve = 2;
      }
    }
    info.flags |= ObjectMeta::kSimdDim;
  }
  BroadReduceTileCollect<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, info);
}

uint64_t ReduceOp::EmitBody(VectorKernel &k, uint64_t xbuf, uint64_t xws) {
  DimArray ranges;
  int ridx = 0;
  for (size_t i = 0; i < ndd_.size(); ++i) {
    if ((ridx & 1) == 0) {
      if (ndd_[i] != lhs_->nd_[i]) {
        ranges[ridx++] = i;
      }
    } else if (ndd_[i] > 1) {
      ranges[ridx++] = i - 1;
    }
  }
  if (ridx == 0) {
    return EmitCopy(insn_, xbuf, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
  }
  if (ridx & 1) {
    ranges[ridx++] = ndd_.size() - 1;
  }
  uint64_t tail_size = k.GetTailSize(lhs_->nd_.data);
  int tail_dim = k.GetTailDim();
  auto emit_reduce_y = [this, &ranges, tail_size, tail_dim](uint64_t *insn, int64_t eidx, uint64_t xd, uint64_t xn) {
    auto start_dim = ranges[eidx];
    auto end_dim = ranges[eidx + 1];
    vReduceY op;
    op.xd = xd;
    op.xn = xn;
    op.iter_size = ndd_.stride(start_dim);
    op.red_size = lhs_->nd_.stride(end_dim) / lhs_->nd_.stride(start_dim - 1);
    ASSERT(op.red_size > 1);
    if (tail_size > 0 && tail_dim >= start_dim && tail_dim <= end_dim) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim] * tail_size;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = lhs_->nd_.stride_back() / lhs_->nd_.stride(end_dim);
    return vReduceY::Encode(insn, reduce_y_list[type_id_][red_op_], op);
  };
  uint64_t xout = ridx & 2 ? xbuf : xws;
  uint64_t size;
  if (auto start_dim = ranges[0]; start_dim == lhs_->nd_.lead_idx()) {  // reduce x
    auto end_dim = ranges[1];
    vReduceX op;
    op.xd = xout;
    op.xn = lhs_->xbuf_;
    op.red_size = lhs_->nd_.stride(end_dim);
    uint64_t clr_tail = 0;
    if (tail_size == 0 || tail_dim > end_dim) {
      op.simd_width = SelectSimdWidth(op.red_size, type_id_);
      op.red_tail = op.red_size;
    } else if (tail_dim > lhs_->nd_.lead_idx()) {
      op.simd_width = SelectSimdWidth(lhs_->nd_.stride(tail_dim - 1), type_id_);
      op.red_tail = op.red_size / lhs_->nd_[tail_dim] * tail_size;
    } else {
      op.simd_width = SelectSimdWidth(lhs_->nd_.lead_stride(), type_id_);
      op.red_tail = RoundUp(tail_size, op.simd_width);
      if (op.red_tail != tail_size) {
        clr_tail = tail_size;
      }
    }
    op.dup_size = lhs_->nd_.stride_back() / lhs_->nd_.stride(end_dim);
    if (auto lead_dim = ndd_.lead_dim(); lead_dim > 1) {
      op.dup_block = lead_dim;
      op.dup_pad = ndd_.lead_stride() - lead_dim;
    } else {
      op.dup_block = op.dup_size;
      op.dup_pad = 0;
    }
    if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail > 0) {
      size = EmitClearPad(insn_, lhs_, clr_tail, op.simd_width, GetRedInitScalarEncode(red_op_, type_id_));
      tail_insn_ = insn_ + size;
      size += vReduceX::Encode(tail_insn_, reduce_x_list[type_id_][red_op_], op);
      *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    } else {
      size = vReduceX::Encode(tail_insn_, reduce_x_list[type_id_][red_op_], op);
    }
  } else {
    size = emit_reduce_y(insn_, 0, xout, lhs_->xbuf_);
  }
  if (ridx > 2) {
    uint64_t xin = xout == xbuf ? xws : xbuf;
    for (int64_t i = 2; i < ridx; i += 2) {
      std::swap(xout, xin);
      tail_insn_ = insn_ + size;
      size += emit_reduce_y(tail_insn_, i, xout, xin);
      *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
  }
  return size;
}

void ReduceOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Reduce";
  if (verbose) {
    oss << "<";
    oss << *dims_ref_ << ", ";
    oss << std::boolalpha << keepdims_;
    oss << ">";
  }
}

ReduceOp::~ReduceOp() {
  if (clean_wrap_ != nullptr) {
    delete clean_wrap_;
  }
  if (visit_) {
    delete visit_;
  }
}

void ReduceOp::Normalize(std::vector<NDObject *> &run_ops) {
  auto in_size = lhs_->shape_ref_->size;
  size_t nd_size = lhs_->nd_.size();
  ndd_.Reset(nd_size);
  if (in_size == 0) {
    shape_.Resize(0);
    if (keepdims_) {
      for (size_t i = 0; i < in_size; ++i) {
        shape_[i] = 1;
      }
      shape_.Resize(in_size);
    }
    for (size_t i = 0; i < nd_size; ++i) {
      ndd_.dims[i] = 1;
    }
  } else {
    std::memcpy(shape_.shape, lhs_->shape_ref_->data, sizeof(int64_t) * in_size);
    for (size_t i = 0; i < dims_ref_->size; i++) {
      auto dim = dims_ref_->data[i];
      shape_[dim >= 0 ? dim : dim + in_size] = 1;
    }
    shape_.Resize(in_size);
    for (size_t i = 0; i < nd_size; ++i) {
      ndd_.dims[i] = shape_[in_size - 1 - i];
    }
    if (!keepdims_) {
      for (size_t i = 0; i < dims_ref_->size; i++) {
        auto dim = dims_ref_->data[i];
        shape_[dim >= 0 ? dim : dim + in_size] = -1;
      }
      size_t out_size = 0;
      for (size_t i = 0; i < in_size; i++) {
        if (shape_[i] != -1) {
          shape_[out_size++] = shape_[i];
        }
      }
      shape_.Resize(out_size);
    }
  }
  if (visit_) {
    visit_->Clear();
  }
  round_tile_.resize(0);
}

void ReduceOp::Tile(const TileParam &tp) {
  CollectRoundTile(ndd_.dims, tp, round_tile_);
  NDObject::Tile(tp);
}

uint64_t ReduceOp::Emit(VectorKernel &k) {
  if (ws_num_ == 2) {
    return round_tile_.empty() ? EmitBody(k, xbuf_, wss_[0]) : EmitDeterm(k);
  }
  uint64_t iter_size = ndd_.lead_dim() * ITEM_SIZE[type_id_];
  bool rm_pad = CheckFlag(OBJ_FLAG_REDUCE_RMPAD_EN) && iter_size % SIMD_BLOCK_SIZE && iter_size < SIMD_REPEAT_SIZE &&
                ndd_.stride_back() > ndd_.lead_stride();
  bool cum = (round_tile_.size() & 1) && !g_system.deterministic_;
  if (ws_num_ == 0 || (!rm_pad && !cum)) {
    return EmitBody(k, xbuf_, xbuf_);
  }
  uint64_t size = EmitBody(k, wss_[0], xbuf_);
  int64_t stride_count = ndd_.stride_back();
  if (rm_pad) {
    vRemovePad op;
    op.xd = cum ? wss_[0] : xbuf_;
    op.xn = wss_[0];
    op.repeat = stride_count / ndd_.lead_stride();
    op.iter_num = ndd_.lead_dim();
    op.rs = GetBlocks(ndd_.lead_stride());
    tail_insn_ = insn_ + size;
    size += vRemovePad::Encode(tail_insn_, removepad_id_list[type_id_], op);
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    flags_ |= OBJ_FLAG_REDUCE_EMIT_RMPAD;
    stride_count = op.repeat * op.iter_num;
  }
  if (cum) {
    vAtomicCum op;
    uint64_t rounds[2];
    BuildDimRounds(round_tile_, rounds);
    op.xd = xbuf_;
    op.xn = wss_[0];
    op.count = stride_count;
    op.round_rank = round_tile_.size();
    op.red_op = red_op_;
    tail_insn_ = insn_ + size;
    size += vAtomicCum::Encode(tail_insn_, type_id_ == kFloat32 ? V_ATOMICCUM : V_ATOMICCUM_FP16, op, rounds);
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    flags_ |= OBJ_FLAG_REDUCE_EMIT_CUM;
  }
  return size;
}

NDObject *ReduceOp::Clone(CloneHelper &h) {
  auto dims_ref = h.GetClone(dims_ref_);
  return new ReduceOp(h.GetClone(lhs_), red_op_, dims_ref, keepdims_);
}

void ReduceOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ReduceOp *>(op);
  auto &shape = self->shape_;
  auto input_shape = self->lhs_->shape_ref_;
  auto keepdims = self->keepdims_;
  if (auto dims_ref = self->dims_ref_; dims_ref->size) {
    auto in_size = input_shape->size;
    for (size_t i = 0; i < in_size; ++i) {
      shape[i] = input_shape->data[i];
    }
    int64_t red_dim = keepdims ? 1 : 0;
    for (size_t i = 0; i < dims_ref->size; i++) {
      auto dim = dims_ref->data[i];
      if (dim < 0) {
        dim += in_size;
      }
      shape[dim] = red_dim;
    }
    if (keepdims) {
      shape.Resize(in_size);
    } else {
      size_t out_size = 0;
      for (size_t i = 0; i < in_size; ++i) {
        if (shape[i] != 0) {
          shape[out_size++] = shape[i];
        }
      }
      shape.Resize(out_size);
    }
  } else if (keepdims) {
    for (size_t i = 0; i < input_shape->size; ++i) {
      shape[i] = 1;
    }
    shape.Resize(input_shape->size);
  } else {
    shape.Resize(0);
  }
}

namespace {
void GenTileVisit(VectorKernel &k, const DimArray &round_tile, RedVisitCoder &coder) {
  auto round_depth = round_tile.size();
  uint32_t core_limit = k.code_.block_dim_;
  if (round_depth == 1) {
    auto v = reinterpret_cast<vVisitRed1 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    v->head = 0;
    v->r1 = r1;
    v->e = (k.tile_num_ / r1) << 32;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed1);
    coder.visit_id_ = V_VISIT_RED_1;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_);
  } else if (round_depth == 2) {
    auto v = reinterpret_cast<vVisitRed2 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    v->head = 0;
    v->e = (k.tile_num_ / r1) << 32;
    v->e1_r1 = e1 << 32 | r1;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed2);
    coder.visit_id_ = V_VISIT_RED_2;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_);
  } else if (round_depth == 3) {
    auto v = reinterpret_cast<vVisitRed3 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r2 = e1 << 32 | r2;
    v->r1 = r1;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed3);
    coder.visit_id_ = V_VISIT_RED_3;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_ / r2);
  } else {
    ASSERT(round_depth == 4);
    auto v = reinterpret_cast<vVisitRed4 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    uint64_t e2 = round_tile[3];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r1 = e1 << 32 | r1;
    v->e2_r2 = e2 << 32 | r2;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed4);
    coder.visit_id_ = V_VISIT_RED_4;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_ / r2);
  }
}

void AddTileVisit(size_t round_depth, RedVisitCoder *coder) {
  switch (round_depth) {
    case 1: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_1);
      reinterpret_cast<vVisitRed1 *>(coder->code_)->user_cnt++;
      break;
    }
    case 2: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_2);
      reinterpret_cast<vVisitRed2 *>(coder->code_)->user_cnt++;
      break;
    }
    case 3: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_3);
      reinterpret_cast<vVisitRed3 *>(coder->code_)->user_cnt++;
      break;
    }
    case 4: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_4);
      reinterpret_cast<vVisitRed4 *>(coder->code_)->user_cnt++;
      break;
    }
    default: {
      ASSERT(0);
      break;
    }
  }
}
}  // namespace

uint64_t ReduceOp::EmitDeterm(VectorKernel &k) {
  auto coder = k.GetVisitor<RedVisitCoder>();
  uint64_t ws_offset;
  if (likely(coder == nullptr)) {
    GenTileVisit(k, round_tile_, *visit_);
    k.AddVisitor(visit_);
    ws_offset = 0;  // TODO: multi workspace
    visit_->ws_size_ = ndd_.stride_back() * sizeof(float) * visit_->block_num_;
  } else {
    AddTileVisit(round_tile_.size(), coder);
    ws_offset = coder->ws_size_;
    coder->ws_size_ += ndd_.stride_back() * sizeof(float) * coder->block_num_;
  }
  uint64_t size = EmitBody(k, wss_[0], wss_[1]);
  vReduceJoin op;
  op.xd = xbuf_;
  op.xn = wss_[0];
  op.xs = wss_[1];
  op.capacity = k.tile_size_ * ITEM_SIZE[k.MaxType()];
  if (ndd_.lead_stride() == ndd_.lead_dim()) {
    op.iter_num = 1;
    op.iter_stride = ndd_.stride_back();
  } else if (ndd_.lead_stride() == ndd_.stride_back()) {
    op.iter_num = 1;
    op.iter_stride = ndd_.lead_dim();
  } else {
    op.iter_stride = ndd_.lead_stride();
    op.iter_num = ndd_.stride_back() / op.iter_stride;
  }
  op.ws = 0;
  tail_insn_ = insn_ + size;
  size += vReduceJoin::Encode(tail_insn_, op);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  k.GetVisitor<RedVisitCoder>()->AddReloc(tail_insn_, 0);
  ws_reloc_.ws = 0;
  ws_reloc_.Update(tail_insn_ + vReduceJoin::RELOC_OFFSET);
  k.code_.BindWorkspace(ws_reloc_, ws_offset);
  return size;
}

void OneHotOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update shape_ref
  shape_.Resize(lhs_->shape_ref_->size);
  int64_t depth = depth_->data[0];
  size_t axis = axis_ >= 0 ? axis_ : shape_.size + axis_;
  for (size_t i = 0; i < shape_.size; ++i) {
    shape_[i] = i == axis ? depth : lhs_->shape_ref_->data[i];
  }
  depth_dim_ = shape_.size - axis - 1;
  // update ndd
  ndd_.Reset(lhs_->nd_.size());
  for (size_t i = 0; i < ndd_.size(); ++i) {
    ndd_.dims[i] = i == static_cast<size_t>(depth_dim_) ? depth : lhs_->nd_[i];
  }
  tile_dim_ = ndd_.size();
}

void OneHotOp::FoldProp(NDObject *op, PropRange &range) {
  auto depth_dim = static_cast<OneHotOp *>(op)->depth_dim_;
  if (range.base > depth_dim) {
    if (auto depth_len = range.base - depth_dim; depth_len < range.depth) {
      range.depth = depth_len;
    }
  } else if (range.base == depth_dim) {
    range.depth = 1;
  }
}

void OneHotOp::TileCollect(NDObject *op, TileInfo &info) {
  auto depth_dim = static_cast<OneHotOp *>(op)->depth_dim_;
  if (depth_dim == 0) {
    info.lead_depth = 1;
  } else if (info.lead_depth > depth_dim) {
    info.lead_depth = depth_dim;
  }
}

void OneHotOp::Tile(const TileParam &tp) {
  if (tp.start <= tile_dim_) {
    tile_dim_ = tp.start;
    if (depth_dim_ == tp.end) {
      depth_tile_ = tp.num;
    } else {
      depth_tile_ *= tp.num;
    }
  }
  NDObject::Tile(tp);
}

uint64_t OneHotOp::Emit(VectorKernel &k) {
  vOneHot op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.data_size = ndd_.stride_back();
  if (depth_dim_ == 0) {
    if (tile_dim_ > 0) {
      op.depth = ndd_.lead_stride();
      op.iter_num = ndd_.stride_back() / op.depth;
      op.dup_round = 0;
      op.mode = vOneHot::MODE_X;
    } else {
      op.depth = ndd_.lead_dim();
      op.dup_round = depth_tile_;
      op.iter_num = 0;
      op.mode = vOneHot::MODE_X_TILE;
    }
  } else {
    if (tile_dim_ > depth_dim_) {
      op.dup_round = ndd_.stride_back() / ndd_.stride(depth_dim_);
      op.iter_num = ndd_.stride(depth_dim_ - 1);
      op.depth = ndd_[depth_dim_];
      op.mode = vOneHot::MODE_Y;
    } else if (tile_dim_ == depth_dim_) {
      op.dup_round = depth_tile_;
      op.iter_num = ndd_.stride(depth_dim_ - 1);
      op.depth = ndd_[depth_dim_];
      op.mode = vOneHot::MODE_Y_TILE;
    } else {
      op.dup_round = depth_tile_;
      op.iter_num = ndd_.stride_back();
      op.depth = depth_tile_ / shape_[shape_.size - depth_dim_ - 1];
      op.mode = vOneHot::MODE_Y_TILE_2;
    }
  }
  return vOneHot::Encode(insn_, ITEM_SIZE[type_id_] == 2 ? V_ONE_HOT_B16 : V_ONE_HOT, on_value_, off_value_, op);
}

NDObject *OneHotOp::Clone(CloneHelper &h) {
  auto depth = h.GetClone(depth_);
  return new OneHotOp(h.GetClone(lhs_), depth, axis_, on_value_, off_value_, type_id_);
}

void OneHotOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "OneHot";
  if (verbose) {
    oss << "<" << axis_ << "," << depth_->data[0] << ",";
    DumpScalarCode(oss, on_value_, type_id_);
    oss << ",";
    DumpScalarCode(oss, off_value_, type_id_);
    oss << ">";
  }
}

void OneHotOp::DimChanged(NDObject *op) {
  for (size_t i = 0; i < op->nd_.size(); ++i) {
    if (i >= op->lhs_->nd_.size() || op->lhs_->nd_[i] != op->nd_[i]) {
      static_cast<OneHotOp *>(op)->depth_dim_ = i;
      break;
    }
  }
}

void OneHotOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<OneHotOp *>(op);
  auto &shape = self->shape_;
  auto depth = self->depth_->data[0];
  auto in_shape = op->lhs_->shape_ref_;
  auto axis = self->axis_;
  if (axis < 0) {
    axis += in_shape->size;
  }
  shape.Resize(in_shape->size);
  for (int i = 0; i < static_cast<int>(shape.size); ++i) {
    shape[i] = i == axis ? depth : in_shape->data[i];
  }
}

uint64_t ExtOutOp::Emit(VectorKernel &k) { return vNop::Encode(insn_); }

void ExtOutOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ExtOut";
  if (verbose) {
    int index = -1;
    auto xout = static_cast<CustomOp *>(lhs_)->xout_;
    ASSERT(xout != nullptr);
    for (int i = 0; i < xout->out_num; ++i) {
      if (xout->data[i] == this) {
        index = i;
        break;
      }
    }
    oss << '<' << index << '>';
  }
}

NDObject *ExtOutOp::Clone(CloneHelper &h) { return new ExtOutOp(h.GetClone(lhs_)); }

uint64_t CustomOp::Emit(VectorKernel &k) {
  uint64_t payload_size = EmitEx(insn_ + vCustom::PAYLOAD_OFFSET);
  vCustom op;
  op.func_addr = func_id_;
  return vCustom::Encode(insn_, V_CUSTOM, payload_size, op);
}

void CustomOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Custom"; }
void CustomOp::DimChanged() {}
void CustomOp::FoldProp(PropRange &) {}
void CustomOp::TileCollect(TileInfo &) {}
void CustomOp::ShapeProp(int64_t &) {}
uint64_t CustomOp::GetFunction(const std::string &full_name) { return g_system.GetCustomFunc(full_name); }
}  // namespace dvm
