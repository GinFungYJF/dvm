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

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "acl/acl_rt.h"
#include "code.h"
#include "ops.h"

namespace dvm {
namespace {
std::unordered_map<std::string, vCompareType> cmp_insn_id = {
  {"Greater", V_CMP_GT},   {"Less", V_CMP_LT},  {"GreaterEqual", V_CMP_GE},
  {"LessEqual", V_CMP_LE}, {"Equal", V_CMP_EQ}, {"NotEqual", V_CMP_NE},
};

size_t DumpInsn(uint64_t *insn, std::ostringstream &oss, uint64_t &pipe);

template <typename T>
void DumpVal(const std::string &name, T val, std::ostringstream &oss) {
  oss << name << "(" << val << ")";
}

template <typename T>
void DumpValues(const std::string &name, const std::vector<T> &values, std::ostringstream &oss) {
  oss << name << "(";
  for (size_t i = 0; i < values.size(); ++i) {
    oss << values[i];
    if (i + 1 < values.size()) {
      oss << ",";
    }
  }
  oss << ")";
}

struct DumpInfo {
  uint64_t *insn = nullptr;
  uint64_t ext = 0;
  DumpInfo(uint64_t *insn_in, uint64_t ext_in) : insn(insn_in), ext(ext_in) {}
};

void DumpRounds(uint64_t rank, uint64_t *rounds, std::ostringstream &oss) {
  oss << "rounds(";
  for (uint64_t i = 0; i < (rank - 1) / 2; ++i) {
    uint64_t round = *rounds++;
    oss << (round & 0xfffffffful) << "," << (round >> 32) << ",";
  }
  uint64_t round = *rounds;
  oss << (round & 0xfffffffful);
  if ((rank & 1ul) == 0) {
    oss << "," << (round >> 32);
  }
  oss << ")";
}

void DumpSLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSLoad op;
  vSLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sload " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", "
      << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("broadcast_m", op.broadcast_m, oss);
  oss << ", ";
  DumpVal("broadcast_n", op.broadcast_n, oss);
  vShard2D shard;
  vVisitMix::DecodeShard(dump_info.insn + op.shard_rel, shard);
  oss << ", shard.slice(" << shard.slice_m << "," << shard.slice_n << ")";
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vSLoad::ROUND_OFFSET, oss);
  }
}

void DumpSStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSStore op;
  vSStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sstore " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.gm) << ", "
      << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("broadcast_m", op.broadcast_m, oss);
  oss << ", ";
  DumpVal("broadcast_n", op.broadcast_n, oss);
  vShard2D shard;
  vVisitMix::DecodeShard(dump_info.insn + op.shard_rel, shard);
  oss << ", shard.slice(" << shard.slice_m << "," << shard.slice_n << ")";
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vSStore::ROUND_OFFSET, oss);
  }
}

void DumpAtomicCum(const DumpInfo &dump_info, std::ostringstream &oss) {
  vAtomicCum op;
  vAtomicCum::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("red_op", op.red_op, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vAtomicCum::ROUND_OFFSET, oss);
  }
}

void DumpLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vLoad op;
  vLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "load.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vLoad::ROUND_OFFSET, oss);
  }
}

void DumpMultiLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vMultiLoad op;
  vMultiLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "multi_load.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tile_stride2", op.tile_stride2, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("gap", op.gap, oss);
  oss << ", ";
  DumpVal("xbuf_size", op.xbuf_size, oss);
  oss << ", ";
  DumpVal("peer_mem", reinterpret_cast<void *>(op.peer_mem), oss);
  oss << ", ";
  DumpVal("multi_size", op.multi_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vLoad::ROUND_OFFSET, oss);
  }
}

void DumpPingPongLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPingPongLoad op;
  vPingPongLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "PingPongLoad.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("pingpong_stride", op.pingpong_stride, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vLoad::ROUND_OFFSET, oss);
  }
}

void DumpPingpongPeerLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPingPongPeerLoad p_load;
  vPingPongPeerLoad::Decode(dump_info.insn, *dump_info.insn, p_load);
  vPingPongLoad &op = p_load.base;
  oss << "PingPongPeerLoad.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("pingpong_stride", op.pingpong_stride, oss);
  oss << ", ";
  DumpVal("offset", p_load.peer_mem_offset, oss);
  oss << ", ";
  DumpVal("set_flag", reinterpret_cast<void *>(p_load.set_flag), oss);
  oss << ", ";
  DumpVal("wait_flag", reinterpret_cast<void *>(p_load.wait_flag), oss);
  oss << ", ";
  DumpVal("event_id", reinterpret_cast<void *>(p_load.event_id), oss);
  vShard2D shard;
  vVisitMix::DecodeShard(dump_info.insn + p_load.shard_rel, shard);
  oss << ", shard.slice(" << shard.slice_m << "," << shard.slice_n << ")";
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vPingPongPeerLoad::ROUND_OFFSET, oss);
  }
}

void DumpStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStore op;
  vStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStore::ROUND_OFFSET, oss);
  }
}

void DumpStoreCond(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreCond op;
  vStoreCond::Decode<false>(dump_info.insn, *dump_info.insn, op);
  oss << "store_cond.u8." << op.tile_stride;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("iter_size", op.iter_size, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("cond_offset", op.cond_offset, oss);
  oss << ", ";
  DumpVal("dtype_shift", op.dtype_shift, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreCond::ROUND_OFFSET, oss);
  }
}

void DumpStoreAtomic(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAtomic op;
  vStoreAtomic::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_atomic.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("red_op", op.red_op, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreAtomic::ROUND_OFFSET, oss);
  }
}

void DumpStoreAtomicW(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAtomicW op;
  vStoreAtomicW::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "@atomic(red_type=" << op.red_op << ", dtype=" << op.atomic_type;
  if (op.round_rank > 0) {
    oss << ", cum=";
    DumpRounds(op.round_rank, dump_info.insn + vStoreAtomicW::ROUND_OFFSET, oss);
  }
  oss << ")\n     ";
  uint64_t pipe;
  DumpInsn(vStoreAtomicW::GetBody(dump_info.insn, op), oss, pipe);
}

void DumpLoadDummy(const DumpInfo &dump_info, std::ostringstream &oss) { oss << "dummy_load.u8.0"; }

void _DumpGatherLoad(const DumpInfo &dump_info, const vGatherLoad &op, std::ostringstream &oss) {
  oss << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from) << ", "
      << reinterpret_cast<void *>(op.index);
  oss << " //";
  DumpVal("inner_size", op.inner_size, oss);
  oss << ", ";
  DumpVal("gather_size", op.gather_size, oss);
  oss << ", ";
  DumpVal("gather_dim_size", op.gather_dim_size, oss);
  oss << ", ";
  DumpVal("gather_mode", op.gather_mode, oss);
  oss << ", ";
  DumpVal("tail_iter", op.tail_iter, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vGatherLoad::ROUND_OFFSET, oss);
  }
}

template <typename T>
void DumpGatherLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vGatherLoad op;
  vGatherLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "gather_load.b" << sizeof(T) * 8 << ".";
  _DumpGatherLoad(dump_info, op, oss);
}

void DumpLoadView(const DumpInfo &dump_info, std::ostringstream &oss) {
  vViewLoad op;
  vViewLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "view_load.u8." << op.iter_size << "x" << op.iter_num;
  std::vector<uint64_t> dst_strides, src_strides;
  bcodeptr_t var_pc = dump_info.insn + vViewLoad::VAR_OFFSET;
  if (op.loop_depth > 0) {
    oss << ".";
    for (uint64_t i = 0; i < op.loop_depth; ++i) {
      uint64_t loop_size, dst_stride, src_stride;
      vViewLoad::DecodeLoop(*var_pc++, loop_size, dst_stride, src_stride);
      dst_strides.push_back(dst_stride);
      src_strides.push_back(src_stride);
      oss << loop_size;
      if (i + 1 < op.loop_depth) {
        oss << "x";
      }
    }
  }
  std::vector<uint64_t> tile_spaces;
  std::vector<uint64_t> tile_strides;
  for (uint64_t i = 0; i < op.tile_depth; ++i) {
    uint64_t space, stride;
    vViewLoad::DecodeTile(*var_pc++, space, stride);
    tile_spaces.push_back(space);
    tile_strides.push_back(stride);
  }
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.from) << " // ";
  DumpValues("dst_stride", dst_strides, oss);
  oss << ", ";
  DumpValues("src_stride", src_strides, oss);
  oss << ", ";
  DumpValues("tile_stride", tile_strides, oss);
  oss << ", ";
  DumpValues("tile_space", tile_spaces, oss);
  oss << ", ";
  DumpVal("src_gap", op.src_gap, oss);
  oss << ", ";
  DumpVal("dst_gap", op.dst_gap, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
}

void DumpLoadViewX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vViewLoadX op;
  vViewLoadX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "view_load_x." << op.type_size << '.' << op.iter_size;
  std::vector<uint64_t> dst_strides, src_strides;
  bcodeptr_t var_pc = dump_info.insn + vViewLoadX::VAR_OFFSET;
  if (op.loop_depth > 0) {
    oss << '.';
    for (uint64_t i = 0; i < op.loop_depth; ++i) {
      uint64_t loop_size, dst_stride, src_stride;
      vViewLoad::DecodeLoop(*var_pc++, loop_size, dst_stride, src_stride);
      dst_strides.push_back(dst_stride);
      src_strides.push_back(src_stride);
      oss << loop_size;
      if (i + 1 < op.loop_depth) {
        oss << 'x';
      }
    }
  }
  std::vector<uint64_t> tile_spaces;
  std::vector<uint64_t> tile_strides;
  for (uint64_t i = 0; i < op.tile_depth; ++i) {
    uint64_t space, stride;
    vViewLoad::DecodeTile(*var_pc++, space, stride);
    tile_spaces.push_back(space);
    tile_strides.push_back(stride);
  }
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.from) << " // ";
  DumpValues("dst_stride", dst_strides, oss);
  oss << ", ";
  DumpValues("src_stride", src_strides, oss);
  oss << ", ";
  DumpValues("tile_stride", tile_strides, oss);
  oss << ", ";
  DumpValues("tile_space", tile_spaces, oss);
  oss << ", ";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
  oss << ", ";
  DumpVal("ws_size", op.ws_size, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
}

void DumpLoadViewT(const DumpInfo &dump_info, std::ostringstream &oss) {
  vViewLoadT op;
  vViewLoadT::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "view_load_t." << op.item_size << '.' << op.w_fractal << 'x' << op.h_fractal << '.' << op.iter_size;
  std::vector<uint64_t> dst_strides, src_strides;
  bcodeptr_t var_pc = dump_info.insn + vViewLoadT::VAR_OFFSET;
  if (op.loop_depth > 0) {
    oss << '.';
    for (uint64_t i = 0; i < op.loop_depth; ++i) {
      uint64_t loop_size, dst_stride, src_stride;
      vViewLoad::DecodeLoop(*var_pc++, loop_size, dst_stride, src_stride);
      dst_strides.push_back(dst_stride);
      src_strides.push_back(src_stride);
      oss << loop_size;
      if (i + 1 < op.loop_depth) {
        oss << 'x';
      }
    }
  }
  std::vector<uint64_t> tile_spaces;
  std::vector<uint64_t> tile_strides;
  for (uint64_t i = 0; i < op.tile_depth; ++i) {
    uint64_t space, stride;
    vViewLoad::DecodeTile(*var_pc++, space, stride);
    tile_spaces.push_back(space);
    tile_strides.push_back(stride);
  }
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.from) << " // ";
  DumpVal("w_gap", op.w_gap, oss);
  oss << ", ";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpValues("dst_stride", dst_strides, oss);
  oss << ", ";
  DumpValues("src_stride", src_strides, oss);
  oss << ", ";
  DumpValues("tile_stride", tile_strides, oss);
  oss << ", ";
  DumpValues("tile_space", tile_spaces, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
  oss << ", ";
  DumpVal("ws_size", op.ws_size, oss);
}

void DumpStoreView(const DumpInfo &dump_info, std::ostringstream &oss) {
  vViewStore op;
  vViewStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "view_store.u8." << op.iter_size << "x" << op.iter_num;
  std::vector<uint64_t> src_strides, dst_strides;
  bcodeptr_t var_pc = dump_info.insn + vViewStore::VAR_OFFSET;
  if (op.loop_depth > 0) {
    oss << ".";
    for (uint64_t i = 0; i < op.loop_depth; ++i) {
      uint64_t loop_size, src_stride, dst_stride;
      vViewStore::DecodeLoop(*var_pc++, loop_size, dst_stride, src_stride);
      src_strides.push_back(src_stride);
      dst_strides.push_back(dst_stride);
      oss << loop_size;
      if (i + 1 < op.loop_depth) {
        oss << "x";
      }
    }
  }
  std::vector<uint64_t> tile_spaces;
  std::vector<uint64_t> tile_strides;
  for (uint64_t i = 0; i < op.tile_depth; ++i) {
    uint64_t space, stride;
    vViewStore::DecodeTile(*var_pc++, space, stride);
    tile_spaces.push_back(space);
    tile_strides.push_back(stride);
  }
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpValues("src_stride", src_strides, oss);
  oss << ", ";
  DumpValues("dst_stride", dst_strides, oss);
  oss << ", ";
  DumpValues("tile_stride", tile_strides, oss);
  oss << ", ";
  DumpValues("tile_space", tile_spaces, oss);
  oss << ", ";
  DumpVal("src_gap", op.src_gap, oss);
  oss << ", ";
  DumpVal("dst_gap", op.dst_gap, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
}

void DumpStoreViewX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vViewStoreX op;
  vViewStoreX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "view_store_x." << op.type_size << '.' << op.iter_size;
  std::vector<uint64_t> src_strides, dst_strides;
  bcodeptr_t var_pc = dump_info.insn + vViewStoreX::VAR_OFFSET;
  if (op.loop_depth > 0) {
    oss << '.';
    for (uint64_t i = 0; i < op.loop_depth; ++i) {
      uint64_t loop_size, dst_stride, src_stride;
      vViewStore::DecodeLoop(*var_pc++, loop_size, dst_stride, src_stride);
      src_strides.push_back(src_stride);
      dst_strides.push_back(dst_stride);
      oss << loop_size;
      if (i + 1 < op.loop_depth) {
        oss << 'x';
      }
    }
  }
  std::vector<uint64_t> tile_spaces;
  std::vector<uint64_t> tile_strides;
  for (uint64_t i = 0; i < op.tile_depth; ++i) {
    uint64_t space, stride;
    vViewStore::DecodeTile(*var_pc++, space, stride);
    tile_spaces.push_back(space);
    tile_strides.push_back(stride);
  }
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpValues("src_stride", src_strides, oss);
  oss << ", ";
  DumpValues("dst_stride", dst_strides, oss);
  oss << ", ";
  DumpValues("tile_stride", tile_strides, oss);
  oss << ", ";
  DumpValues("tile_space", tile_spaces, oss);
  oss << ", ";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
  oss << ", ";
  DumpVal("ws_size", op.ws_size, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
}

void DumpExtract(const DumpInfo &dump_info, std::ostringstream &oss) {
  vExtract op;
  vExtract::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " // ";
  DumpVal("slot", op.slot, oss);
}

void DumpUnary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vUnary op;
  vUnary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpUnaryWS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpVal("ws", reinterpret_cast<void *>(op.xm), oss);
}

void DumpRemovePad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vRemovePad op;
  vRemovePad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_num << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpVal("rs", op.rs, oss);
}

template <typename T = float>
void DumpBinaryS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinaryS op;
  vBinaryS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << ", "
      << vBinaryS::GetScalar<T>(dump_info.insn);
}

void DumpBinary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm);
}

void DumpBinaryWS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinaryWS op;
  vBinaryWS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("ws0", reinterpret_cast<void *>(op.ws0), oss);
  oss << ", ";
  DumpVal("ws1", reinterpret_cast<void *>(op.ws1), oss);
}

void DumpPack(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPack op;
  vPack::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

void DumpCompare(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCompare op;
  vCompare::Decode(dump_info.insn, *dump_info.insn, op);
  std::string cmp_op("Unknown");
  for (auto it = cmp_insn_id.begin(); it != cmp_insn_id.end(); ++it) {
    if (it->second == op.type) {
      cmp_op = it->first;
    }
  }
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("cmp_type", cmp_op, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

template <typename T>
void DumpCompareS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCompareS op;
  vCompareS::Decode(dump_info.insn, *dump_info.insn, op);
  std::string cmp_op("Unknown");
  for (auto it = cmp_insn_id.begin(); it != cmp_insn_id.end(); ++it) {
    if (it->second == op.type) {
      cmp_op = it->first;
    }
  }
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << vCompareS::GetScalar<T>(dump_info.insn) << " //";
  DumpVal("cmp_type", cmp_op, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

void DumpBroadcastS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastS op;
  vBroadcastS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << op.scalar;
}

void DumpBroadcastS_B64(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastS_B64 op;
  vBroadcastS_B64::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  uint64_t scalar = (op.high << 32) | op.scalar;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << static_cast<int64_t>(scalar);
}

void DumpSelect(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSelect op;
  vSelect::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.count;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.cond) << ", "
      << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

void DumpBroadcastX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastX op;
  vBroadcastX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "[" << op.count << "]x" << op.lead_num << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("lead_pad", op.lead_pad, oss);
}

void DumpBroadcastY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastY op;
  vBroadcastY::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "32x" << op.dup_stride << "x[" << op.dup_num << "]x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpReduceX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceX op;
  auto head = *dump_info.insn;
  vReduceX::Decode(dump_info.insn, head, op);
  vReduceX::DecodeBlock(dump_info.insn, op);
  oss << "[" << op.red_size << "]x" << op.dup_size;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
  oss << ", ";
  DumpVal("dup_block", op.dup_block, oss);
  oss << ", ";
  DumpVal("dup_pad", op.dup_pad, oss);
  oss << ", ";
  DumpVal("simd_width", op.simd_width, oss);
}

void DumpReduceY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceY op;
  auto head = *dump_info.insn;
  vReduceY::Decode(dump_info.insn, head, op);
  oss << op.iter_size << "x[" << op.red_size << "]x" << op.dup_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
}

void DumpReduceJoin(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceJoin op;
  vReduceJoin::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_stride << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("xs", reinterpret_cast<void *>(op.xs), oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
  oss << ", ";
  DumpVal("seg_tile_rel", op.seg_tile_rel, oss);
  oss << ", ";
  DumpVal("capacity", op.capacity, oss);
}

void DumpCopy(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCopy op;
  auto head = *dump_info.insn;
  vCopy::Decode(dump_info.insn, head, op);
  oss << "32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpCubeCopy(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCubeCopy op;
  auto head = *dump_info.insn;
  vCubeCopy::Decode(dump_info.insn, head, op);
  oss << "32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("tile_offset", op.tile_offset, oss);
}

void DumpNop(const DumpInfo &dump_info, std::ostringstream &oss) { oss << "0"; }

void DumpClearPad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vClearPad op;
  vClearPad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_size << "x" << op.iter_num << " " << reinterpret_cast<void *>(op.xd) << ", " << op.scalar << " //";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpVal("simd_width", op.simd_width, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
}

void DumpElementAny(const DumpInfo &dump_info, std::ostringstream &oss) {
  vElementAny op;
  vElementAny::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.simd_width << "x" << op.repeat << " " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("repeat_tail", op.repeat_tail, oss);
}

void DumpOneHot(const DumpInfo &dump_info, std::ostringstream &oss) {
  vOneHot op;
  vOneHot::Decode(dump_info.insn, op);
  oss << op.data_size << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("mode", op.mode, oss);
  oss << ", ";
  DumpVal("iter_num", op.iter_num, oss);
  oss << ", ";
  DumpVal("depth", op.depth, oss);
  oss << ", ";
  DumpVal("dup_round", op.dup_round, oss);
}

void DumpCustom(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCustom op;
  vCustom::Decode(dump_info.insn, op);
  oss << '[' << g_system.GetCustomFuncName(op.func_addr) << "] ";
  uint64_t size = (*dump_info.insn >> V_HEAD_SIZE_OFFSET) & V_HEAD_SIZE_MASK;
  for (size_t i = vCustom::PAYLOAD_OFFSET; i < size; ++i) {
    oss << reinterpret_cast<void *>(dump_info.insn[i]);
    if (i < size - 1) {
      oss << ", ";
    }
  }
}

void DumpStoreAG(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAG op;
  vStoreAG::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_ag.u8.32x" << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_num", op.iter_num, oss);
  oss << ", ";
  DumpVal("iter_size", op.iter_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
  oss << ", ";
  DumpVal("rank_size", op.rank_size, oss);
  oss << ", ";
  DumpVal("shard_stride", op.shard_stride, oss);
}

void DumpStoreRS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreRS op;
  vStoreRS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_rs.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreRS::ROUND_OFFSET, oss);
  }
}

const char name_peer_load[] = "peer_load";
const char name_peer_load_mix[] = "peer_load_mix";
const char name_peer_store[] = "peer_store";
const char name_peer_store_mix[] = "peer_store_mix";

template <char const *name>
void DumpPeerDMA(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPeerDMA op;
  vPeerDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << name;
  oss << ".u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.peer_mem);
  oss << " //";
  DumpVal("rank_id", op.rank_id, oss);
  oss << ", ";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
  oss << ", ";
  DumpVal("flag_mem", reinterpret_cast<void *>(op.flag_mem), oss);
  oss << ", ";
  DumpVal("set_flag", op.set_flag, oss);
  oss << ", ";
  DumpVal("wait_flag", op.wait_flag, oss);
  oss << ", ";
  DumpVal("set_event_id", op.event_id, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vPeerDMA::ROUND_OFFSET, oss);
  }
}

using DumpFunc = void(const DumpInfo &, std::ostringstream &oss);

std::unordered_map<uint64_t, DumpFunc *> acc_dump_func_table = {
  {V_LOAD, &DumpLoad},
  {V_LOAD_DUMMY, &DumpLoadDummy},
  {V_LOAD_GATHER_B16, &DumpGatherLoad<uint16_t>},
  {V_LOAD_GATHER_B32, &DumpGatherLoad<uint32_t>},
  {V_LOAD_VIEW, &DumpLoadView},
  {V_LOAD_VIEW_X, &DumpLoadViewX},
  {V_LOAD_VIEW_TRANS, &DumpLoadViewT},
  {V_SLOAD, &DumpSLoad},
  {V_MULTI_LOAD, &DumpMultiLoad},
  {V_PINGPONG_LOAD, &DumpPingPongLoad},
  {V_PINGPONG_PEER_LOAD, &DumpPingpongPeerLoad},
  {V_PEER_LOAD, &DumpPeerDMA<name_peer_load>},
  {V_PEER_LOAD_MIX, &DumpPeerDMA<name_peer_load_mix>},
  {V_STORE, &DumpStore},
  {V_STORE_ATOMIC, &DumpStoreAtomic},
  {V_STORE_ATOMIC_WRAP, &DumpStoreAtomicW},
  {V_STORE_COND, &DumpStoreCond},
  {V_SSTORE, &DumpSStore},
  {V_STORE_AG, &DumpStoreAG},
  {V_STORE_RS, &DumpStoreRS},
  {V_STORE_VIEW, &DumpStoreView},
  {V_STORE_VIEW_X, &DumpStoreViewX},
  {V_PEER_STORE, &DumpPeerDMA<name_peer_store>},
  {V_PEER_STORE_MIX, &DumpPeerDMA<name_peer_store_mix>},
};

std::unordered_map<uint64_t, std::tuple<DumpFunc *, std::string, std::string>> op_dump_info_table = {
  {V_COPY, {&DumpCopy, "Copy", "u8"}},
  {V_COPY_CUBE_TILE, {&DumpCubeCopy, "CopyCubeTile", "u8"}},
  {V_NOP, {&DumpNop, "Nop", "u8"}},
  {V_BROADCAST_X_B32, {&DumpBroadcastX, "BroadcastX", "b32"}},
  {V_BROADCAST_X_B64, {&DumpBroadcastX, "BroadcastX", "b64"}},
  {V_BROADCAST_X_B16, {&DumpBroadcastX, "BroadcastX", "b16"}},
  {V_BROADCAST_X_B8, {&DumpBroadcastX, "BroadcastX", "b8"}},
  {V_BROADCAST_Y, {&DumpBroadcastY, "BroadcastY", "u8"}},
  {V_BROADCAST_S, {&DumpBroadcastS, "BroadcastS", "b32"}},
  {V_BROADCAST_S_B16, {&DumpBroadcastS, "BroadcastS", "b16"}},
  {V_BROADCAST_S_B8, {&DumpBroadcastS, "BroadcastS", "b8"}},
  {V_SQRT, {&DumpUnary, "Sqrt", "fp32"}},
  {V_SQRT_FP16, {&DumpUnary, "Sqrt", "fp16"}},
  {V_ABS, {&DumpUnary, "Abs", "fp32"}},
  {V_ABS_FP16, {&DumpUnary, "Abs", "fp16"}},
  {V_ABS_INT32, {&DumpUnary, "Abs", "int32"}},
  {V_LOG, {&DumpUnary, "Log", "fp32"}},
  {V_LOG_FP16, {&DumpUnary, "Log", "fp16"}},
  {V_EXP, {&DumpUnary, "Exp", "fp32"}},
  {V_EXP_FP16, {&DumpUnary, "Exp", "fp16"}},
  {V_ROUND, {&DumpUnary, "Round", "fp32"}},
  {V_FLOOR, {&DumpUnary, "Floor", "fp32"}},
  {V_CEIL, {&DumpUnary, "Ceil", "fp32"}},
  {V_TRUNC, {&DumpUnary, "Trunc", "fp32"}},
  {V_ISFINITE, {&DumpUnary, "IsFinite", "fp32"}},
  {V_ISFINITE_FP16, {&DumpUnaryWS, "IsFinite", "fp16"}},
  {V_ISFINITE_BF16, {&DumpUnaryWS, "IsFinite", "bf16"}},
  {V_CAST_FP16_TO_FP32, {&DumpUnary, "CastFP16", "fp32"}},
  {V_CAST_BOOL_TO_FP16, {&DumpUnary, "CastS8", "fp16"}},
  {V_CAST_FP16_TO_BOOL, {&DumpUnary, "CastFP16", "bool"}},
  {V_CAST_FP16_TO_INT32, {&DumpUnary, "CastFP16", "int32"}},
  {V_CAST_FP32_TO_INT32, {&DumpUnary, "CastFP32", "int32"}},
  {V_CAST_FP32_TO_FP16, {&DumpUnary, "CastFP32", "fp16"}},
  {V_CAST_FP32_TO_BF16, {&DumpUnary, "CastFP32", "bf16"}},
  {V_CAST_INT32_TO_FP32, {&DumpUnary, "CastS32", "fp32"}},
  {V_CAST_INT32_TO_FP16, {&DumpUnary, "CastS32", "fp16"}},
  {V_CAST_BF16_TO_FP32, {&DumpUnary, "CastBF16", "fp32"}},
  {V_CAST_BF16_TO_INT32, {&DumpUnary, "CastBF16", "int32"}},
  {V_CAST_INT32_TO_INT64, {&DumpUnary, "CastS32", "int64"}},
  {V_CAST_INT64_TO_INT32, {&DumpUnary, "CastS64", "int32"}},
  {V_CAST_FP32_TO_INT64, {&DumpUnary, "CastFP32", "int64"}},
  {V_CAST_INT64_TO_FP32, {&DumpUnary, "CastS64", "fp32"}},
  {V_ADDS, {&DumpBinaryS, "Adds", "fp32"}},
  {V_ADDS_FP16, {&DumpBinaryS<Float16>, "Adds", "fp16"}},
  {V_ADDS_BF16, {&DumpBinaryS<dvm::BFloat16>, "Adds", "bf16"}},
  {V_ADDS_INT32, {&DumpBinaryS<int32_t>, "Adds", "int32"}},
  {V_MULS, {&DumpBinaryS, "Muls", "fp32"}},
  {V_MULS_FP16, {&DumpBinaryS<Float16>, "Muls", "fp16"}},
  {V_MULS_BF16, {&DumpBinaryS<dvm::BFloat16>, "Muls", "bf16"}},
  {V_MULS_INT32, {&DumpBinaryS<int32_t>, "Muls", "int32"}},
  {V_SDIV, {&DumpBinaryS, "sDiv", "fp32"}},
  {V_SDIV_FP16, {&DumpBinaryS<Float16>, "sDiv", "fp16"}},
  {V_DIVS, {&DumpBinaryS, "Divs", "fp32"}},
  {V_DIVS_FP16, {&DumpBinaryS<Float16>, "Divs", "fp16"}},
  {V_MAXS, {&DumpBinaryS, "Maximums", "fp32"}},
  {V_MAXS_FP16, {&DumpBinaryS<Float16>, "Maximums", "fp16"}},
  {V_MAXS_BF16, {&DumpBinaryS<dvm::BFloat16>, "Maximums", "bf16"}},
  {V_MAXS_INT32, {&DumpBinaryS<int32_t>, "Maximums", "int32"}},
  {V_MINS, {&DumpBinaryS, "Minimums", "fp32"}},
  {V_MINS_FP16, {&DumpBinaryS<Float16>, "Maximums", "fp16"}},
  {V_MINS_BF16, {&DumpBinaryS<dvm::BFloat16>, "Maximums", "bf16"}},
  {V_MINS_INT32, {&DumpBinaryS<int32_t>, "Maximums", "int32"}},
  {V_ADD, {&DumpBinary, "Add", "fp32"}},
  {V_ADD_FP16, {&DumpBinary, "Add", "fp16"}},
  {V_ADD_BF16, {&DumpBinary, "Add", "bf16"}},
  {V_ADD_INT32, {&DumpBinary, "Add", "int32"}},
  {V_ADD_INT64, {&DumpBinary, "Add", "int64"}},
  {V_SUB, {&DumpBinary, "Sub", "fp32"}},
  {V_SUB_FP16, {&DumpBinary, "Sub", "fp16"}},
  {V_SUB_BF16, {&DumpBinary, "Sub", "bf16"}},
  {V_SUB_INT32, {&DumpBinary, "Sub", "int32"}},
  {V_SUB_INT64, {&DumpBinary, "Sub", "int64"}},
  {V_MUL, {&DumpBinary, "Mul", "fp32"}},
  {V_MUL_FP16, {&DumpBinary, "Mul", "fp16"}},
  {V_MUL_BF16, {&DumpBinary, "Mul", "bf16"}},
  {V_MUL_INT32, {&DumpBinary, "Mul", "int32"}},
  {V_DIV, {&DumpBinary, "Div", "fp32"}},
  {V_DIV_FP16, {&DumpBinary, "Div", "fp16"}},
  {V_MAX, {&DumpBinary, "Maximum", "fp32"}},
  {V_MAX_FP16, {&DumpBinary, "Maximum", "fp16"}},
  {V_MAX_BF16, {&DumpBinary, "Maximum", "bf16"}},
  {V_MAX_INT32, {&DumpBinary, "Maximum", "int32"}},
  {V_MIN, {&DumpBinary, "Minimum", "fp32"}},
  {V_MIN_FP16, {&DumpBinary, "Minimum", "fp16"}},
  {V_MIN_BF16, {&DumpBinary, "Minimum", "bf16"}},
  {V_MIN_INT32, {&DumpBinary, "Minimum", "int32"}},
  {V_LOGICAL_NOT_BOOL, {&DumpUnary, "LogicalNot", "bool"}},
  {V_LOGICAL_AND_BOOL, {&DumpBinary, "LogicalAnd", "bool"}},
  {V_LOGICAL_OR_BOOL, {&DumpBinary, "LogicalOr", "bool"}},
  {V_CMP_BOOL, {&DumpCompare, "Compare", "bool"}},
  {V_CMPS_BOOL, {&DumpCompareS<bool>, "CompareS", "bool"}},
  {V_POW, {&DumpBinaryWS, "Pow", "fp32"}},
  {V_CMP, {&DumpCompare, "Compare", "fp32"}},
  {V_CMP_FP16, {&DumpCompare, "Compare", "fp16"}},
  {V_CMP_BF16, {&DumpCompare, "Compare", "bf16"}},
  {V_CMP_INT32, {&DumpCompare, "Compare", "int32"}},
  {V_CMP_INT64, {&DumpCompare, "Compare", "int64"}},
  {V_EXTRACT_B32, {&DumpExtract, "Extract", "b32"}},
  {V_PACK_B32, {&DumpPack, "Pack", "b32"}},
  {V_BROADCAST_S_B64, {&DumpBroadcastS_B64, "BroadcastS", "b64"}},
  {V_CMPS, {&DumpCompareS<float>, "CompareS", "fp32"}},
  {V_CMPS_FP16, {&DumpCompareS<Float16>, "CompareS", "fp16"}},
  {V_CMPS_BF16, {&DumpCompareS<dvm::BFloat16>, "CompareS", "bf16"}},
  {V_CMPS_INT32, {&DumpCompareS<int32_t>, "CompareS", "int32"}},
  {V_SEL, {&DumpSelect, "Select", "fp32"}},
  {V_SEL_FP16, {&DumpSelect, "Select", "fp16"}},
  {V_SEL_BF16, {&DumpSelect, "Select", "bf16"}},
  {V_SEL_INT32, {&DumpSelect, "Select", "int32"}},
  {V_SEL_INT64, {&DumpSelect, "Select", "int64"}},
  {V_RSUM_X, {&DumpReduceX, "SumX", "fp32"}},
  {V_RSUM_Y, {&DumpReduceY, "SumY", "fp32"}},
  {V_RMAX_X, {&DumpReduceX, "MaxX", "fp32"}},
  {V_RMAX_Y, {&DumpReduceY, "MaxY", "fp32"}},
  {V_RMIN_X, {&DumpReduceX, "MinX", "fp32"}},
  {V_RMIN_Y, {&DumpReduceY, "MinY", "fp32"}},
  {V_RMAX_X_FP16, {&DumpReduceX, "MaxX", "fp16"}},
  {V_RMAX_Y_FP16, {&DumpReduceY, "MaxY", "fp16"}},
  {V_RMIN_X_FP16, {&DumpReduceX, "MinX", "fp16"}},
  {V_RMIN_Y_FP16, {&DumpReduceY, "MinY", "fp16"}},
  {V_RSUM_JOIN, {&DumpReduceJoin, "SumJoin", "fp32"}},
  {V_CLR_PAD, {&DumpClearPad, "ClrPad", "b32"}},
  {V_CLR_PAD_B16, {&DumpClearPad, "ClrPad", "b16"}},
  {V_ELEMENT_ANY, {&DumpElementAny, "ElementAny", "fp32"}},
  {V_REMOVEPAD, {&DumpRemovePad, "RemovePad", "u32"}},
  {V_REMOVEPAD_U16, {&DumpRemovePad, "RemovePad", "u16"}},
  {V_ATOMICCUM, {&DumpAtomicCum, "AtomicCum", "fp32"}},
  {V_ATOMICCUM_FP16, {&DumpAtomicCum, "AtomicCum", "fp16"}},
  {V_ONE_HOT, {&DumpOneHot, "OneHot", "b32"}},
  {V_ONE_HOT_B16, {&DumpOneHot, "OneHot", "b16"}},
  {V_CUSTOM, {&DumpCustom, "Custom", "x"}},
};

enum vPipe {
  V_PIPE_LOAD = 0,
  V_PIPE_STORE,
  V_PIPE_SIMD,
  V_PIPE_ALL,
};

size_t DumpInsn(uint64_t *insn, std::ostringstream &oss, uint64_t &pipe) {
  auto convert_id = [](const uint64_t offsets[], uint64_t none_idx, uint64_t id) -> uint64_t {
    for (uint64_t i = 0; i <= none_idx; ++i) {
      if (offsets[i] == id) {
        return i;
      }
    }
    ASSERT(0);
    return 0;
  };
  uint64_t head = *insn;
  uint64_t id = (head >> V_HEAD_ID_OFFSET) & V_HEAD_ID_MASK;
  uint64_t offset = 0;
  if (head & (1ul << V_HEAD_SIMD_FLAG_OFFSET)) {
    uint64_t ext = (head >> V_HEAD_EXT_OFFSET) & V_HEAD_EXT_MASK;
    offset = (head >> V_HEAD_SIZE_OFFSET) & V_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext};
    id = convert_id(g_system.g_simd_func_offset_, V_NONE, id);
    if (op_dump_info_table.find(id) != op_dump_info_table.end()) {
      auto [dump_func, name, dtype_str] = op_dump_info_table[id];
      oss << name << "." << dtype_str << ".";
      dump_func(info, oss);
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
    pipe = V_PIPE_SIMD;
  } else {
    uint64_t ext = (head >> V_M_HEAD_EXT_OFFSET) & V_M_HEAD_EXT_MASK;
    offset = (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext};
    id = convert_id(g_system.g_access_func_offset_, V_ACCESS_NONE, id);
    if (acc_dump_func_table.find(id) != acc_dump_func_table.end()) {
      acc_dump_func_table[id](info, oss);
      pipe = id >= V_STORE ? V_PIPE_STORE : V_PIPE_LOAD;
#ifdef DEBUG
      if (pipe == V_PIPE_STORE) {
        uint64_t debug_size = *(insn + (((head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK) - 1));
        oss << ", ";
        DumpVal("dbg_size", debug_size, oss);
      }
#endif
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  }
  return offset;
}

void DasBody(std::ostringstream &oss, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
  bcodeptr_t insn = reinterpret_cast<bcodeptr_t>(bcode);
  bcodeptr_t insn_end = reinterpret_cast<bcodeptr_t>(bcode + bcode_size);
  uint64_t insn_idx = 0;
  uint64_t pipe = 0;
  while (insn < insn_end) {
    auto head = *insn;
    if (head == 0) break;
    oss << indent << insn_idx << ": ";
    insn_idx++;
    auto offset = DumpInsn(insn, oss, pipe);
    oss << "\n";
    insn = insn + offset;
    oss << indent << "    {";
    if (pipe == V_PIPE_SIMD) {
      oss << "simd";
      if (head & (0x1ul << V_HEAD_BAR_FLAG_OFFSET)) {
        oss << ", bar(1)";
      }
      if (head & (0x1ul << V_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", load_simd_sync(wait, " << int((head >> V_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_BACK_WAIT_OFFSET)) {
        oss << ", store_simd_sync(wait, " << int((head >> V_HEAD_B_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_SET_FLAG_OFFSET)) {
        oss << ", simd_store_sync(set, " << int((head >> V_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_BACK_SET_OFFSET)) {
        oss << ", simd_load_sync(set, " << int((head >> V_HEAD_B_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    } else if (pipe == V_PIPE_LOAD) {
      oss << "load";
      if (head & (0x1ul << V_M_HEAD_SET_FLAG_OFFSET)) {
        oss << ", load_simd_sync(set, " << int((head >> V_M_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", simd_load_sync(wait, " << int((head >> V_M_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    } else {
      oss << "store";
      if (head & (0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", simd_store_sync(wait, " << int((head >> V_M_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_M_HEAD_SET_FLAG_OFFSET)) {
        oss << ", store_simd_sync(set, " << int((head >> V_M_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    }
    oss << " }" << std::endl;
  }
}
}  // namespace

class DisAssembler {
 public:
  explicit DisAssembler(std::ostringstream &oss_) : oss(oss_) {}

  void Run(Code *code, const char *prefix) {
    void *ffts = *reinterpret_cast<void **>(code->data_);
    uint64_t entry = *reinterpret_cast<uint64_t *>(code->data_ + sizeof(uint64_t));
    uint8_t *bcode = code->data_ + code->HeadSize();
    uint64_t bcode_size = code->data_size_ - code->HeadSize();
    oss << "// target=" << code->target_ << ", block_dim=" << code->block_dim_ << ", ffts_addr=" << ffts << std::endl;
    oss << prefix << ".";
    auto ktype = entry & V_ENTRY_MASK_TYPE;
    if (ktype == V_ENTRY_TYPE_V) {
      DasVec(entry, bcode, bcode_size, "");
    } else if (ktype == V_ENTRY_TYPE_VE) {
      if (entry & V_ENTRY_FLAG_CUBE_MIX) {
        DasMix(entry, bcode, bcode_size, "");
      } else {
        DasVecEx(entry, bcode, bcode_size, "");
      }
    } else if (ktype == V_ENTRY_TYPE_P) {
      DasParallel(entry, bcode, bcode_size, code->target_, code->block_dim_, "");
    } else if (ktype == V_ENTRY_TYPE_C) {
      DasCube(entry, bcode, bcode_size, "");
    } else {
      ASSERT(0);  // removed
      DasStages(entry, bcode, bcode_size, code->block_dim_, "");
    }
  }

  void DasVecBody(uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    DasBody(oss, bcode, bcode_size, indent);
  }

  void DasCubeBody(vCubeOp *op, const std::string &indent) {
    oss << indent << "MatMul." << op->m_real << "x" << op->k_real << "x" << op->n_real << " "
        << reinterpret_cast<void *>(op->gm_c) << " " << reinterpret_cast<void *>(op->gm_a) << " "
        << reinterpret_cast<void *>(op->gm_b);
    oss << "// align(" << op->m_align << "," << op->k_align << "," << op->n_align << "), loop(" << op->m_loop << ","
        << op->k_loop << "," << op->n_loop << "), offset=(" << op->offset_a << "," << op->offset_b << "), ";
    DumpVal("m0", op->m0, oss);
    oss << ", ";
    DumpVal("k0", op->k0, oss);
    oss << ", ";
    DumpVal("n0", op->n0, oss);
    oss << ", ";
    DumpVal("trans_a", bool(op->flags & V_CUBE_FLAG_TRANS_A), oss);
    oss << ", ";
    DumpVal("trans_b", bool(op->flags & V_CUBE_FLAG_TRANS_B), oss);
    oss << ", ";
    DumpVal("input_a_nz", bool(op->flags & V_CUBE_FLAG_INPUT_A_NZ), oss);
    oss << ", ";
    DumpVal("input_b_nz", bool(op->flags & V_CUBE_FLAG_INPUT_B_NZ), oss);
    oss << ", ";
    DumpVal("swizzle", op->swizzle, oss);
    oss << ", ";
    DumpVal("group_num", op->group_num, oss);
    oss << ", ";
    DumpVal("batch_cast", op->batch_cast, oss);

    if (op->flags & V_CUBE_FLAG_GROUPED_LIST) {
      oss << ", ";
      DumpVal("group_list", reinterpret_cast<void *>(op->gm_group_list), oss);
      oss << ", ";
      DumpVal("group_list_size", op->group_list_size, oss);
    }

    if (op->flags & V_CUBE_FLAG_WITH_BIAS) {
      oss << ", ";
      DumpVal("gm_bias", reinterpret_cast<void *>(op->gm_bias), oss);
      if (op->flags & V_CUBE_FLAG_BIAS_FP32) {
        oss << ", ";
        DumpVal("bias_fp32", "", oss);
      }
    }
  }

  void DasVec(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto tile_body = vGetBitRange(entry, V_ENTRY_V_TILE_BODY_OFFSET, V_ENTRY_V_TILE_BODY_BITS);
    auto tile_tail = vGetBitRange(entry, V_ENTRY_V_TILE_TAIL_OFFSET, V_ENTRY_V_TILE_TAIL_BITS);
    oss << indent << "aiv(body_tile=" << tile_body << ", tail_tile_diff=" << tile_tail;
    oss << ") {" << std::endl;
    DasVecBody(bcode, bcode_size, indent + "  ");
    oss << indent << "}";
  }

  class VisitDumper {
   public:
    VisitDumper(std::ostringstream &oss, const std::string &indent) : oss_(oss), indent_(indent) {}
    void Dump(uint64_t visit_id, uint64_t *visit_addr) {
      for (uint64_t i = 0; i < V_VISIT_NONE; ++i) {
        if (g_system.g_visit_func_offset_[i] == visit_id) {
          visit_id = i;
          break;
        }
      }
      constexpr uint64_t MASK_32 = 0xfffffffful;
      oss_ << indent_ << "// ";
      switch (visit_id) {
        case V_VISIT_RED_1: {
          auto v = reinterpret_cast<vVisitRed1 *>(visit_addr);
          oss_ << ".visit: red_1, user=" << v->user_cnt << ", e=" << (v->e >> 32) << ", r1=" << (v->r1) << std::endl;
          break;
        }
        case V_VISIT_RED_2: {
          auto v = reinterpret_cast<vVisitRed2 *>(visit_addr);
          oss_ << ".visit: red_2, user=" << v->user_cnt << ", e=" << (v->e >> 32) << ", r1=" << (v->e1_r1 & MASK_32) << ", e1="
               << (v->e1_r1 >> 32) << std::endl;
          break;
        }
        case V_VISIT_RED_3: {
          auto v = reinterpret_cast<vVisitRed3 *>(visit_addr);
          oss_ << ".visit: red_3, user=" << v->user_cnt << ", e=" << (v->e >> 32) << ", r1=" << (v->r1) << ", e1=" << (v->e1_r2 >> 32)
              << ", r2=" << (v->e1_r2 & MASK_32) << std::endl;
          break;
        }
        case V_VISIT_RED_4: {
          auto v = reinterpret_cast<vVisitRed4 *>(visit_addr);
          oss_ << ".visit: red_4, user=" << v->user_cnt << ", e=" << (v->e >> 32) << ", r1=" << (v->e1_r1 & MASK_32) << ", e1=" << (v->e1_r1 >> 32)
              << ", r2=" << (v->e2_r2 & MASK_32) << ", e2=" << (v->e2_r2 >> 32) << std::endl;
          break;
        }
        case V_VISIT_REORDER: {
          vVisitReorder op;
          vVisitReorder::Decode(visit_addr, op);
          oss_ << ".visit: reorder, offset=" << op.wave_tidx << ", parallel=[:" << op.parallel_extent << ":" << op.parallel_stride
               << "], loop=[:" << op.loop_num << '(' << op.loop_tail << "):" << op.loop_stride << "], iter_coord=(";
          for (uint64_t i = 0; i < op.coord_num; ++i) {
            uint64_t stride, extent, coord_val;
            vVisitReorder::DecodeCoord(visit_addr, i, stride, extent, coord_val);
            oss_ << "[:" << extent << ':' << stride << ']';
            if (i < op.coord_num - 1) {
              oss_ << ", ";
            }
          }
          oss_ << ')' << std::endl;
          break;
        }
        case V_VISIT_PIPE_SET: {
          DumpPipeSet(visit_addr);
          break;
        }
        case V_VISIT_PIPE_WAIT: {
          DumpPipeWait(visit_addr);
          break;
        }
        default:
          break;
      }
    }
    void DumpPipeSet(uint64_t *visit_addr) {
      vVisitPipeSet op;
      vVisitPipeSet::Decode(visit_addr, op);
      oss_ << ".visit: pipe_set, step=" << op.step << ", uid=" << op.uid << ", addr=" << reinterpret_cast<void *>(op.gm) << std::endl;
      Dump(op.visit_id, visit_addr + vVisitPipeSet::NEST_VISIT_OFFSET);
    }
    void DumpPipeWait(uint64_t *visit_addr) {
      vVisitPipeWait op;
      vVisitPipeWait::Decode(visit_addr, op);
      oss_ << ".visit: pipe_wait, step=" << op.step << ", uid=" << op.uid << ", prod_num=" << op.prod_num
           << ", addr=" << reinterpret_cast<void *>(op.gm) << std::endl;
      Dump(op.visit_id, visit_addr + vVisitPipeWait::NEST_VISIT_OFFSET);
    }

   private:
    std::ostringstream &oss_;
    const std::string &indent_;
  };

  void DasVecEx(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto visit_id = vGetBitRange(entry, V_ENTRY_VE_VISIT_ID_OFFSET, V_ENTRY_VE_VISIT_ID_BITS);
    auto offset = vGetBitRange(entry, V_ENTRY_VE_VISIT_OFFSET_OFFSET, V_ENTRY_VE_VISIT_OFFSET_BITS);
    VisitDumper dumper(oss, indent);
    dumper.Dump(visit_id, reinterpret_cast<uint64_t *>(bcode) + offset);
    oss << indent << "aiv() {" << std::endl;
    DasVecBody(bcode, bcode_size, indent + "  ");
    oss << indent << "}";
  }

  void DasCube(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    vCubeOp *cube = reinterpret_cast<vCubeOp *>(bcode);
    if (cube->flags & V_CUBE_FLAG_PIPELINE) {
      vPipeCubeOp *pipe = reinterpret_cast<vPipeCubeOp *>(bcode);
      uint64_t set_step, wait_step, wait_prod_num, uid;
      vPipeCubeOp::DecodeStep(pipe->step, set_step, wait_step, wait_prod_num, uid);
      if (set_step) {
        oss << indent << "// .visit: pipe_set, step=" << set_step << ", uid=" << uid
            << ", addr=" << reinterpret_cast<void *>(pipe->set_gm) << std::endl;
      }
      if (wait_step) {
        oss << indent << "// .visit: pipe_wait, step=" << wait_step << ", prod_num=" << wait_prod_num << ", uid=" << uid
            << ", addr=" << reinterpret_cast<void *>(pipe->wait_gm) << std::endl;
      }
    }
    oss << indent << "aic() {" << std::endl;
    DasCubeBody(cube, indent + "  ");
    oss << std::endl << indent << "}";
  }

  void DasMix(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    oss << indent << "mix() {" << std::endl;
    auto sub_indent = indent + "  ";
    vCubeOp *cube = reinterpret_cast<vCubeOp *>(bcode);
    if (cube->flags & V_CUBE_FLAG_GROUP_SET) {
      oss << sub_indent << "aic(group_set=1";
    } else if (cube->flags & V_CUBE_FLAG_STORE_UB) {
      oss << sub_indent << "aic(cc_ub_set=1";
    } else {
      ASSERT(false);
    }
    if (cube->flags & V_CUBE_FLAG_PINGPONG_STORE) {
      oss << ", pingpong_store=1";
    }
    if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
      oss << ", peer_store=1";
    }
    oss << ", pos=" << reinterpret_cast<void *>(cube->gm_pos);
    oss << ") {" << std::endl;
    DasCubeBody(cube, sub_indent + "  ");
    oss << std::endl << sub_indent << "}" << std::endl;
    bcode += sizeof(vCubeOp);
    bcode_size -= sizeof(vCubeOp);
    auto offset = vGetBitRange(entry, V_ENTRY_VE_VISIT_OFFSET_OFFSET, V_ENTRY_VE_VISIT_OFFSET_BITS);
    auto visit_code = reinterpret_cast<bcodeptr_t>(bcode + offset * sizeof(uint64_t));
    vVisitMix visit;
    vVisitMix::Decode(visit_code, visit);
    vShard2D shard;
    vVisitMix::DecodeShard(visit_code, shard);
    oss << sub_indent << "aiv(sub_tile_num=[" << visit.subtile0 << ", " << visit.subtile1 << "], "
        << "shard.slice=[" << shard.slice_m << ", " << shard.slice_n << "], "
        << "shard.tail=[" << shard.tail_m << ", " << shard.tail_n << "], "
        << "shard.stride=[" << shard.stride_m << ", " << shard.stride_n << "]) {" << std::endl;
    DasVecBody(bcode, bcode_size, sub_indent + "  ");
    oss << sub_indent << "}" << std::endl;
    oss << indent << "}";
  }

  void DasParallel(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, int target, uint64_t block_num, const std::string &indent) {
    struct ProgInfo {
      vProgEntry prog;
      uint32_t core_begin;
      uint32_t core_end;
    };
    auto get_programs = [bcode](uint64_t core_num, uint64_t lookup, std::vector<ProgInfo> &programs) {
      uint64_t last_offset = 0;
      for (uint64_t i = 0; i < core_num; ++i) {
        uint64_t prog_offset = *(bcode + lookup + i);
        if (prog_offset == V_ENTRY_P_LKUP_INVALID) continue;
        if (prog_offset != last_offset) {
          auto &data = programs.emplace_back();
          vProgEntry::Decode(reinterpret_cast<uint64_t *>(bcode) + prog_offset, data.prog);
          data.core_begin = i;
          data.core_end = i;
          last_offset = prog_offset;
        } else {
          programs.back().core_end = i;
        }
      }
    };
    std::vector<ProgInfo> aic_programs, aiv_programs;
    if (!Code::IsVector(target)) {
      auto lookup = (entry >> V_ENTRY_P_AIC_LKUP_OFFSET) & V_ENTRY_P_LKUP_MASK;
      get_programs(block_num, lookup, aic_programs);
    }
    if (target != Code::kTargetCube) {
      auto lookup = (entry >> V_ENTRY_P_AIV_LKUP_OFFSET) & V_ENTRY_P_LKUP_MASK;
      get_programs(Code::IsVector(target) ? block_num : block_num * 2, lookup, aiv_programs);
    }
    oss << indent << "parallel() {" << std::endl;
    auto child_indent = indent + "  ";
    for (auto &[prog, core_begin, core_end] : aic_programs) {
      ASSERT(core_begin == prog.b_begin && core_end + 1 == prog.b_begin + prog.b_num);
      oss << child_indent << "// aic[" << core_begin << ", " << core_end << "]";
      uint64_t prog_size =
        vGetBitRange(prog.entry, V_ENTRY_CODE_SIZE_OFFSET, V_ENTRY_CODE_SIZE_BITS) * sizeof(uint64_t);
      if (auto ktype = prog.entry & V_ENTRY_MASK_TYPE; ktype == V_ENTRY_TYPE_C) {
        oss << std::endl;
        DasCube(prog.entry, bcode + prog.offset, prog_size, child_indent);
      } else {
        auto it = std::find_if(aiv_programs.begin(), aiv_programs.end(),
                               [&prog](const ProgInfo &info) { return info.prog.offset == prog.offset; });
        ASSERT(it != aiv_programs.end());
        oss << ", aiv[" << it->core_begin << ", " << it->core_end << "]" << std::endl;
        DasMix(prog.entry, bcode + prog.offset, prog_size, child_indent);
      }
      oss << std::endl;
    }
    for (auto &[prog, core_begin, core_end] : aiv_programs) {
      ASSERT(core_begin == prog.b_begin && core_end + 1 == prog.b_begin + prog.b_num);
      if (prog.entry & V_ENTRY_FLAG_CUBE_MIX) {
        continue;
      }
      oss << child_indent << "// aiv[" << core_begin << ", " << core_end << "]" << std::endl;
      uint64_t prog_size =
        vGetBitRange(prog.entry, V_ENTRY_CODE_SIZE_OFFSET, V_ENTRY_CODE_SIZE_BITS) * sizeof(uint64_t);
      uint64_t ktype = prog.entry & V_ENTRY_MASK_TYPE;
      if (ktype == V_ENTRY_TYPE_V) {
        DasVec(prog.entry, bcode + prog.offset, prog_size, child_indent);
      } else {
        ASSERT(ktype == V_ENTRY_TYPE_VE);
        DasVecEx(prog.entry, bcode + prog.offset, prog_size, child_indent);
      }
      oss << std::endl;
    }
    oss << indent << "}";
  }

  void DasStages(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, uint64_t block_num, const std::string &indent) {
    oss << indent << "stages() {" << std::endl;
    int stage_idx = 0;
    uint8_t *bcode_end = bcode + bcode_size;
    while (bcode < bcode_end) {
      uint64_t stage_size = vGetBitRange(entry, V_ENTRY_CODE_SIZE_OFFSET, V_ENTRY_CODE_SIZE_BITS) * sizeof(uint64_t);
      oss << "  // stage " << stage_idx << std::endl;
      stage_idx++;
      auto ktype = entry & V_ENTRY_MASK_TYPE;
      if (ktype == V_ENTRY_TYPE_V) {
        DasVec(entry, bcode, stage_size, indent + "  ");
      } else if (ktype == V_ENTRY_TYPE_VE) {
        if (entry & V_ENTRY_FLAG_CUBE_MIX) {
          DasMix(entry, bcode, stage_size, indent + "  ");
        } else {
          DasVecEx(entry, bcode, stage_size, indent + "  ");
        }
      } else if (ktype == V_ENTRY_TYPE_P) {
        DasParallel(entry, bcode, stage_size, Code::kTargetVec, block_num, indent + "  ");
      } else {
        ASSERT(ktype == V_ENTRY_TYPE_C);
        DasCube(entry, bcode, stage_size, indent + "  ");
      }
      oss << std::endl;
      bcode += stage_size;
      entry = *reinterpret_cast<uint64_t *>(bcode);
      bcode += sizeof(uint64_t);
    }
    oss << indent << "}";
  }

 private:
  std::ostringstream &oss;
};

int CodeWrap::LaunchWrap(void *workspace, void *stream) { return next_->LaunchWrap(workspace, stream); }

void CodeWrap::CombineWrap(Code *to, uint64_t ws_base) {
  auto next = next_;
  to->InsertWrap(this);
  next->CombineWrap(to, ws_base);
}

void CodeWrap::DasWrap(std::ostringstream &oss) { next_->DasWrap(oss); }
void CodeWrap::CollectWrap(std::vector<Code *> &codes) { return next_->CollectWrap(codes); }

Code::~Code() { CheckFree(); }

Code &Code::operator=(Code &&other) {
  if (this != &other) {
    MoveCode(other);
    bind_wss_ = other.bind_wss_;
    bind_ops_ = other.bind_ops_;
    wrap_ = other.wrap_;
  }
  return *this;
}

void Code::MoveCode(Code &other) {
  CheckFree();
  data_ = other.data_;
  data_size_ = other.data_size_;
  block_dim_ = other.block_dim_;
  target_ = other.target_;
  mem_size_ = other.mem_size_;

  other.data_ = nullptr;
  other.data_size_ = 0;
  other.mem_size_ = 0;
}

void Code::Alloc(size_t size) {
  if (size <= mem_size_) {
    return;
  }
  CheckFree();
  if (size <= PARAM_TABLE_LIMIT) {
    constexpr size_t MEM_ALGIN = 1024ul;
    size = RoundUp(size, MEM_ALGIN);
    data_ = static_cast<unsigned char *>(std::malloc(size));
  } else {
    constexpr uint32_t RT_MEM_ALIGN = 1024 * 1024;
    size = (size + RT_MEM_ALIGN - 1) / RT_MEM_ALIGN * RT_MEM_ALIGN;
    auto ret = aclrtMallocHost(reinterpret_cast<void **>(&data_), size);
    EXCEPTION_IF(ret != 0, "Alloc aclrtMallocHost error");
  }
  mem_size_ = size;
}

void Code::Free() {
  if (mem_size_ <= PARAM_TABLE_LIMIT) {
    std::free(data_);
  } else {
    auto ret = aclrtFreeHost(data_);
    EXCEPTION_IF(ret != 0, "aclrtFreeHost error");
  }
}

void Code::DisAssemble(std::ostringstream &oss) {
  if (wrap_) {
    wrap_->DasWrap(oss);
  } else {
    Code::DasWrap(oss);
  }
}

void Code::CombineBind(const Code &code, uint64_t ws_base) {
  if (auto op = code.bind_wss_) {
    for (auto next = op->bind_list_; next != nullptr; next = next->bind_list_) {
      BindWorkspace(*op, op->ws + ws_base);
      op = next;
    }
    BindWorkspace(*op, op->ws + ws_base);
  }
  if (auto op = code.bind_ops_) {
    for (auto next = op->bind_list_; next != nullptr; next = next->bind_list_) {
      BindOp(*op, *(op->op));
      op = next;
    }
    BindOp(*op, *(op->op));
  }
}

void Code::Combine(const Code &code, uint64_t ws_base) {
  CombineBind(code, ws_base);
  if (code.wrap_) {
    code.wrap_->CombineWrap(this, ws_base);
  }
}

void Code::BindOp(RelocAddr &op, const RelocAddr &target) {
  op.op = &target;
  for (auto x = bind_ops_; x != nullptr; x = x->bind_list_) {
    if (x == op.op) {
      InsertBind(x->bind_list_, op);
      return;
    }
  }
  InsertBind(bind_ops_, op);
}

uint64_t Code::ReserveCodeSpace(uint64_t workspace_size) {
  uint64_t *head = reinterpret_cast<uint64_t *>(data_);
  head[1] |= V_ENTRY_FLAG_EXTERN_CODE;
  auto offset = RoundUp<uint64_t>(data_size_, 512);
  for (auto op = bind_wss_; op != nullptr; op = op->bind_list_) {
    op->ws += offset;
  }
  return workspace_size + offset;
}

int Code::LaunchWrap(void *workspace, void *stream) { return DoLaunch(workspace, stream); }

void Code::CombineWrap(Code *code, uint64_t ws_base) {}
void Code::DasWrap(std::ostringstream &oss) {
  if (data_ != nullptr) {
    DisAssembler(oss).Run(this, "vmain");
  }
}
void Code::CollectWrap(std::vector<Code *> &codes) { codes.push_back(this); }

CodeLaunchGuard::CodeLaunchGuard(Code &root) {
  root.Collect(codes_);
  for (auto code : codes_) {
    if (code->wrap_ == nullptr) {
      code->wrap_ = this;
    } else {
      for (auto prev = code->wrap_; prev; prev = prev->next_) {
        if (prev->next_ == code) {
          prev->next_ = this;
          break;
        }
      }
    }
  }
}

CodeLaunchGuard::~CodeLaunchGuard() {
  for (auto code : codes_) {
    if (code->wrap_ == this) {
      code->wrap_ = nullptr;
    } else {
      for (auto prev = code->wrap_; prev; prev = prev->next_) {
        if (prev->next_ == this) {
          prev->next_ = code;
          break;
        }
      }
    }
  }
}

int CodeLaunchGuard::LaunchWrap(void *workspace, void *stream) {
  ASSERT(launch_idx_ < codes_.size());
  return CodeLaunch(codes_[launch_idx_++], workspace, stream);
}

void PCodeEncoder::Reset(Code *code, int target, int max_prog_num, uint64_t code_reserve, uint64_t core_reserve) {
  uint64_t aic_num, aiv_num;
  if (Code::IsVector(target)) {
    aic_num = 0;
    aiv_num = core_reserve > 0 ? core_reserve : g_system.CoreNum(CoreType::kAIV);
  } else {
    aic_num = core_reserve > 0 ? core_reserve : g_system.CoreNum(CoreType::kAIC);
    aiv_num = target == Code::kTargetCube ? 0 : aic_num * 2;
  }
  uint64_t reserve_size = Code::HeadSize() + aic_num + aiv_num +
                          vProgEntry::CODE_SIZE * sizeof(uint64_t) * max_prog_num + code_reserve;
  code->Alloc(reserve_size);
  code->target_ = target;
  code_ = code;
  aic_lookup_ = nullptr;
  aiv_lookup_ = nullptr;
  prog_entry_ = nullptr;
  auto code_body = code->data_ + Code::HeadSize();
  prog_data_ = code_body;
  uint64_t head_data = 0;
  if (aic_num > 0) {
    aic_lookup_ = prog_data_;
    prog_data_ += aic_num;
    head_data |= (aic_lookup_ - code_body) << V_ENTRY_P_AIC_LKUP_OFFSET;
  }
  if (aiv_num > 0) {
    aiv_lookup_ = prog_data_;
    prog_data_ += aiv_num;
    head_data |= (aiv_lookup_ - code_body) << V_ENTRY_P_AIV_LKUP_OFFSET;
  }
  constexpr uint64_t align_size = 8;
  if (uint64_t unalign = uint64_t(prog_data_ - code_body) & (align_size - 1); unalign > 0) {
    prog_data_ += align_size - unalign;
  }
  prog_entry_ = reinterpret_cast<uint64_t *>(prog_data_);
  for (uint64_t *lookup = reinterpret_cast<uint64_t *>(code_body); lookup < prog_entry_; ++lookup) {
    *lookup = std::numeric_limits<uint64_t>::max();
  }
  prog_data_ += vProgEntry::CODE_SIZE * sizeof(uint64_t) * max_prog_num;
  code->UpdateHead(Code::GenEntry(head_data, V_ENTRY_TYPE_P, prog_data_ - code_body));
}
}  // namespace dvm
