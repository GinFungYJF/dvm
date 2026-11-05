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
#ifndef _VM_KERNEL_
#error "vm_aiv.h must be included from a VM kernel translation unit"
#endif
#ifndef __VM_ARCH__
#error "vm_aiv.h requires __VM_ARCH__"
#endif
#include "isa.h"

#define PC_BASE 32
#define VBRCB_NUM 8

#define IS_FLOAT(x) (sizeof(*x) == 4)
#define SIMD_WIDTH_SHIFT(x) (IS_FLOAT(x) ? 3 : 4)
#define TYPE_SHIFT(x) (32 / sizeof(x))
#define BlockDataShift(x) (5 - sizeof(x) / 2)  // 1B - 5, 2B - 4, 4B - 3, 8B - not support
#define BlockNum(x) ((sizeof(x) == 8) ? 64 : (256 / sizeof(x)))
#define MaxRepeat(num, x) ((num + BlockNum(x) - 1) / BlockNum(x))
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

#if __VM_ARCH__ == 220
inline constexpr uint64_t UB_WORKSPACE_SIZE = 512;
inline constexpr uint64_t WORKSPACE = 192ul * 1024ul - UB_WORKSPACE_SIZE;
#endif

#define VREG_TABLE_BASE 0

#define MASK_32 0xfffffffful
#define INVALID_INT uint64_t(-1)

// tile format:
//  tile_idx(32) << 32 | tile_count(28) << 4 | skip_stride(1) << 3 | tail_flag(1) << 2 | end_flag(1) << 1 |
//  not_begin_flag(1)
#define V_TILE_N_BEGIN_FLAG_OFFSET 0
#define V_TILE_END_FLAG_OFFSET 1
#define V_TILE_TAIL_FLAG_OFFSET 2
#define V_TILE_SKIP_FLAG_OFFSET 3
#define V_TILE_COUNT_OFFSET 4
#define V_TILE_CUR_IDX_OFFSET 32
#define V_TILE_COUNT_BITS 28

#define TileIdx(tile) ((tile) >> V_TILE_CUR_IDX_OFFSET)
#define IsLastTile(tile) (((tile) & 6ul) == 6ul)
#define IsBeginTile(tile) ((tile & (1ul << V_TILE_N_BEGIN_FLAG_OFFSET)) == 0)
#define IsEndTile(tile) ((tile) & (1ul << V_TILE_END_FLAG_OFFSET))
#define IsSkipTile(tile) ((tile) & (1ul << V_TILE_SKIP_FLAG_OFFSET))

typedef uint64_t (*VisitFunc)(bcodeptr_t pc);

struct vRegTable {
  uint32_t blockidx;
  uint32_t blocknum;
  uint64_t blockgroup;
  uint8_t *vm_code_base;
  uint64_t skip_stride;
};

__aicore_inline__ __ubuf__ vRegTable *__restrict__ RegTable() {
  return reinterpret_cast<__ubuf__ vRegTable *>(get_imm(VREG_TABLE_BASE));
}

__aicore_inline__ void vBlockInit(uint64_t blockidx, uint64_t blocknum) {
  auto reg = (__ubuf__ uint64_t *)get_imm(VREG_TABLE_BASE);
  *reg = blocknum << 32 | blockidx;
}

__aicore_inline__ uint64_t vBlockIdx() { return RegTable()->blockidx; }
__aicore_inline__ uint64_t vBlockNum() { return RegTable()->blocknum; }

__aicore_inline__ void vBlockGroupInit(uint64_t groupidx, uint64_t groupnum) {
  auto reg = (__ubuf__ uint64_t *)get_imm(VREG_TABLE_BASE + 8);
  *reg = groupnum << 16 | groupidx;
}
__aicore_inline__ uint64_t vBlockGroup() { return RegTable()->blockgroup; }
#define vGroupIdx(blockgroup) ((blockgroup) & 0xfffful)
#define vGroupNum(blockgroup) (((blockgroup) >> 16) & 0xfffful)

__aicore_inline__ void PushSkipStride(uint64_t skip_stride) { RegTable()->skip_stride = skip_stride; }
__aicore_inline__ uint64_t PopSkipStride() { return RegTable()->skip_stride; }

__aicore_inline__ uint64_t GetBlock(uint64_t stride, uint64_t type_size) { return (stride * type_size + 31) >> 5; }

template <typename T>
__aicore_inline__ void SetInvalidMask(uint64_t simd_width, uint64_t simd_valid) {
  uint64_t lmask = 0, rmask = 0;
  if (sizeof(T) == 4) {
    uint64_t simd_mask = 0xffffffffffffffffllu >> (64 - simd_width);
    rmask = simd_mask & (0xffffffffffffffffllu << simd_valid);
  } else if (sizeof(T) == 2) {
    if (simd_width > 64) {
      lmask = 0xffffffffffffffffllu >> (128 - simd_width);
      rmask = 0xffffffffffffffffllu;
    } else {
      rmask = 0xffffffffffffffffllu >> (64 - simd_width);
    }
    if (simd_valid > 64) {
      lmask ^= 0xffffffffffffffffllu >> (128 - simd_valid);
      rmask ^= 0xffffffffffffffffllu;
    } else {
      rmask ^= 0xffffffffffffffffllu >> (64 - simd_valid);
    }
  }
  set_vector_mask(lmask, rmask);
}

__aicore_inline__ void SetVectorMask(uint64_t simd_width) {
  if (simd_width > 64) {
    set_vector_mask(0xffffffffffffffffllu >> (128 - simd_width), 0xffffffffffffffffllu);
  } else {
    set_vector_mask(0, 0xffffffffffffffffllu >> (64 - simd_width));
  }
}

#ifdef DEBUG
__aicore_inline__ void OVER_WRITE_CHECK(bcodeptr_t pc, uint64_t head, uint64_t write_end) {
  uint64_t debug_size = *(pc + (((head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK) - 1));
  if (write_end > debug_size) {
    trap();
  }
}
#else
#define OVER_WRITE_CHECK(pc, head, write_end)
#endif

#define VA_UB_OFFSET 0x80000ul
#define VA_VMAIN_BASE ((get_pc() - VMAIN_OFFSET) & (~0xffful))
typedef void (*OpFunc)(__bcode__ uint64_t *pc, uint64_t head, uint64_t tile);

#ifndef __VM_DRY_RUN__
#define VA_BCODE_BASE_UB (get_sys_va_base() + VA_UB_OFFSET + PC_BASE)
#define DEF_DRY_FUNC(offset, op, func)
__aicore_inline__ uint8_t *GetFunc(uint64_t id, uint8_t *base_addr) { return base_addr + (id << 2); }
#endif

enum RoundCacheMode { kRoundCacheLoad, kRoundCacheStore, kRoundCacheStoreAtomic, kRoundCacheNone };

#define V_RED_HEAD_REPEAT_OFFSET 0
#define V_RED_HEAD_REPEAT_MASK 0xffffffu
#define V_RED_HEAD_THREAD_OFFSET 24
#define V_RED_HEAD_THREAD_MASK 0x7fu
#define V_RED_HEAD_INC_FLAG_OFFSET 31

template <RoundCacheMode mode>
__aicore_inline__ uint64_t RoundUpTileIdx(uint64_t tile, uint64_t tile_idx, uint64_t round_rank, bcodeptr_t rounds) {
  if (round_rank == 1) {
    uint64_t r1 = rounds[0];
    uint64_t r_tile_idx = tile_idx / r1;
    if (mode == kRoundCacheLoad) {
      return r_tile_idx * r1 == tile_idx || IsBeginTile(tile) || IsSkipTile(tile) ? r_tile_idx : INVALID_INT;
    } else if (mode == kRoundCacheStore) {
      return r_tile_idx * r1 == tile_idx ? r_tile_idx : INVALID_INT;
    } else if (mode == kRoundCacheStoreAtomic) {
      return r_tile_idx * r1 + r1 - 1 == tile_idx || IsEndTile(tile) ? r_tile_idx : INVALID_INT;
    } else {
      return r_tile_idx;
    }
  } else if (round_rank == 2) {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & MASK_32;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t p1 = tile_idx / r1;
    if (mode == kRoundCacheStore) {
      uint64_t align = p1 * r1;
      return tile_idx < align + r2 ? p1 * r2 + tile_idx - align : INVALID_INT;
    }
    return p1 * r2 + tile_idx % r2;
  } else if (round_rank == 3) {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & MASK_32;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t r3 = rounds[1];
    uint64_t p0 = tile_idx / r1;
    uint64_t p1 = p0 * r2;
    uint64_t p2 = tile_idx % (r2 * r3);
    uint64_t p3 = p2 / r3;
    if (mode == kRoundCacheLoad) {
      return p3 * r3 == p2 || IsBeginTile(tile) || IsSkipTile(tile) ? p1 + p3 : INVALID_INT;
    } else if (mode == kRoundCacheStore) {
      return (tile_idx < p0 * r1 + r2 * r3) && (p3 * r3 == p2) ? p1 + p3 : INVALID_INT;
    } else if (mode == kRoundCacheStoreAtomic) {
      return p3 * r3 + r3 - 1 == p2 || IsEndTile(tile) ? p1 + p3 : INVALID_INT;
    } else {
      return p1 + p3;
    }
  } else {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & MASK_32;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t r4_r3 = rounds[1];
    uint64_t r3 = r4_r3 & MASK_32;
    uint64_t r4 = r4_r3 >> 32;
    uint64_t p0 = tile_idx / r1;
    uint64_t p1 = tile_idx % (r2 * r3);
    uint64_t p2 = p1 / r3;
    if (mode == kRoundCacheStore) {
      return (tile_idx < p0 * r1 + r2 * r3) && (p1 < p2 * r3 + r4) ? (p0 * r2 + p2) * r4 + p1 - p2 * r3 : INVALID_INT;
    }
    return (p0 * r2 + p2) * r4 + tile_idx % r4;
  }
}

__aicore_inline__ uint64_t RoundUpTileIdxRS(uint64_t tile_idx, uint64_t round_rank, uint64_t rank_id,
                                            bcodeptr_t rounds) {
  if (round_rank == 1) {
    if (tile_idx != rank_id) {
      return INVALID_INT;
    }
    return 0;
  }
  uint64_t r2_r1 = rounds[0];
  uint64_t r1 = r2_r1 & MASK_32;
  uint64_t r2 = r2_r1 >> 32;
  uint64_t p2 = tile_idx / r2;
  if (p2 != rank_id) {
    return INVALID_INT;
  }
  uint64_t p1 = tile_idx / r1;
  return p1 * r2 + tile_idx % r2;
}

__aicore_inline__ uint64_t RoundUpTileIdxAG(uint64_t tile_idx, uint64_t round_rank, uint64_t rank_id,
                                            bcodeptr_t rounds) {
  (void)round_rank;
  uint64_t r2_r1 = rounds[0];
  uint64_t r2 = r2_r1 >> 32;
  uint64_t p2 = tile_idx / r2;
  if (p2 != rank_id) {
    return INVALID_INT;
  }
  return tile_idx;
}

template <typename T, bool inc = false>
__aicore_inline__ T MakeRedHead(T repeat, T thread_size) {
  T head = repeat << V_RED_HEAD_REPEAT_OFFSET | thread_size << V_RED_HEAD_THREAD_OFFSET;
  if constexpr (inc) {
    head |= 1ul << V_RED_HEAD_INC_FLAG_OFFSET;
  }
  return head;
}

__aicore_inline__ void PadRedSync(uint32_t cnt) {
  for (uint32_t i = 0; i < cnt; ++i) {
    ffts_cross_core_sync(PIPE_MTE3, vFftsSyncConfig(0, 0));
#if __VM_ARCH__ == 220
    wait_flag_dev(0);
#else
    wait_flag_dev(PIPE_S, 0);
#endif
  }
}

__aicore_inline__ uint64_t VisitRed_1(bcodeptr_t pc) {
  __bcode__ vVisitRed1 *__restrict__ op = reinterpret_cast<__bcode__ vVisitRed1 *>(pc);
  uint64_t blocknum = vBlockNum();
  uint64_t blockidx = vBlockIdx();
  uint64_t e_idx = op->e & MASK_32;
  uint64_t e_num = op->e >> 32;
  if (e_idx >= e_num) {
    return 0;
  }
  uint64_t r1 = op->r1;
  if (e_idx + blocknum / 2 < e_num) {
    op->e += blocknum;
    uint64_t e_cur = e_idx + blockidx;
    if (e_cur >= e_num) {
      return 0;
    }
    op->head = MakeRedHead<uint32_t>(0, 1);
    uint64_t tile = (e_cur * r1) << V_TILE_CUR_IDX_OFFSET | r1 << V_TILE_COUNT_OFFSET;
    if (e_cur == e_num - 1) {
      tile |= 1ul << V_TILE_TAIL_FLAG_OFFSET;
    }
    return tile;
  }
  uint64_t group_num = e_num - e_idx;
  uint64_t thread_size = min(r1, blocknum / group_num);
  op->e += group_num;
  if (blockidx >= thread_size * group_num) {
    PadRedSync(op->user_cnt);
    return 0;
  }
  uint64_t thread_idx = blockidx % thread_size;
  uint64_t group_idx = blockidx / thread_size;
  uint64_t e_cur = e_idx + group_idx;
  uint64_t tidx = e_cur * r1 + thread_idx;
  uint64_t loop = (r1 - thread_idx - 1) / thread_size + 1;
  op->head = MakeRedHead<uint32_t>(0, thread_size);
  uint64_t tile = tidx << V_TILE_CUR_IDX_OFFSET | loop << V_TILE_COUNT_OFFSET | 1ul << V_TILE_SKIP_FLAG_OFFSET;
  if (e_cur == e_num - 1 && thread_size * (loop - 1) + thread_idx == r1 - 1) {
    tile |= 1ul << V_TILE_TAIL_FLAG_OFFSET;
  }
  PushSkipStride(thread_size);
  return tile;
}

__aicore_inline__ uint64_t VisitRed_2(bcodeptr_t pc) {
  __bcode__ vVisitRed2 *__restrict__ op = reinterpret_cast<__bcode__ vVisitRed2 *>(pc);
  uint64_t blocknum = vBlockNum();
  uint64_t blockidx = vBlockIdx();
  uint64_t e_idx = op->e & MASK_32;
  uint64_t e_num = op->e >> 32;
  if (e_idx >= e_num) {
    return 0;
  }
  uint64_t tidx;
  uint64_t count;
  uint64_t stride;
  uint64_t e1 = op->e1_r1 >> 32;
  uint64_t r1 = op->e1_r1 & MASK_32;
  if (e_idx + blocknum / 2 < e_num) {
    op->e += blocknum;
    uint64_t e_cur = e_idx + blockidx;
    if (e_cur >= e_num) {
      return 0;
    }
    op->head = MakeRedHead<uint32_t>(0, 1);
    tidx = r1 * e1 * (e_cur / e1) + e_cur % e1;
    count = r1;
    stride = e1;
  } else {
    uint64_t group_num = e_num - e_idx;
    uint64_t thread_size = min(r1, blocknum / group_num);
    op->e += group_num;
    if (blockidx >= thread_size * group_num) {
      PadRedSync(op->user_cnt);
      return 0;
    }
    uint64_t thread_idx = blockidx % thread_size;
    uint64_t group_idx = blockidx / thread_size;
    uint64_t e_cur = e_idx + group_idx;
    uint64_t group_tile = r1 * e1 * (e_cur / e1) + e_cur % e1;
    op->head = MakeRedHead<uint32_t>(0, thread_size);
    tidx = group_tile + thread_idx * e1;
    count = (r1 - thread_idx - 1) / thread_size + 1;
    stride = e1 * thread_size;
  }
  PushSkipStride(stride);
  return tidx << V_TILE_CUR_IDX_OFFSET | count << V_TILE_COUNT_OFFSET | 1ul << V_TILE_SKIP_FLAG_OFFSET;
}

__aicore_inline__ uint64_t VisitRed_3(bcodeptr_t pc) {
  __bcode__ vVisitRed3 *__restrict__ op = reinterpret_cast<__bcode__ vVisitRed3 *>(pc);
  uint64_t tidx_head = op->tidx_head;
  uint64_t e1 = op->e1_r2 >> 32;
  uint64_t r2 = op->e1_r2 & MASK_32;
  if (tidx_head & V_RED_HEAD_REPEAT_MASK) {
    uint64_t thread_size = (tidx_head >> V_RED_HEAD_THREAD_OFFSET) & V_RED_HEAD_THREAD_MASK;
    tidx_head = (tidx_head | 1ul << V_RED_HEAD_INC_FLAG_OFFSET) + ((r2 * e1 * thread_size) << 32) - 1ul;
    op->tidx_head = tidx_head;
    return (tidx_head >> 32) << V_TILE_CUR_IDX_OFFSET | r2 << V_TILE_COUNT_OFFSET;
  }
  uint64_t blocknum = vBlockNum();
  uint64_t blockidx = vBlockIdx();
  uint64_t e_idx = op->e & MASK_32;
  uint64_t e_num = op->e >> 32;
  if (e_idx >= e_num) {
    return 0;
  }
  uint64_t tidx;
  uint64_t r1 = op->r1;
  if (e_idx + blocknum / 2 < e_num) {
    op->e += blocknum;
    uint64_t e_cur = e_idx + blockidx;
    if (e_cur >= e_num) {
      return 0;
    }
    tidx = (r1 * e1 * (e_cur / e1) + e_cur % e1) * r2;
    op->tidx_head = tidx << 32 | MakeRedHead<uint64_t>(r1 - 1, 1);
  } else {
    uint64_t group_num = e_num - e_idx;
    uint64_t thread_size = min(r1, blocknum / group_num);
    op->e += group_num;
    if (blockidx >= thread_size * group_num) {
      PadRedSync(op->user_cnt);
      return 0;
    }
    uint64_t thread_idx = blockidx % thread_size;
    uint64_t group_idx = blockidx / thread_size;
    uint64_t e_cur = e_idx + group_idx;
    uint64_t repeat = (r1 - thread_idx - 1) / thread_size + 1;
    tidx = (r1 * e1 * (e_cur / e1) + e_cur % e1 + thread_idx * e1) * r2;
    op->tidx_head = tidx << 32 | MakeRedHead<uint64_t>(repeat - 1, thread_size);
  }
  return tidx << V_TILE_CUR_IDX_OFFSET | r2 << V_TILE_COUNT_OFFSET;
}

__aicore_inline__ uint64_t VisitRed_4(bcodeptr_t pc) {
  __bcode__ vVisitRed4 *__restrict__ op = reinterpret_cast<__bcode__ vVisitRed4 *>(pc);
  uint64_t tidx_head = op->tidx_head;
  uint64_t e1 = op->e1_r1 >> 32;
  uint64_t r1 = op->e1_r1 & MASK_32;
  uint64_t e2 = op->e2_r2 >> 32;
  uint64_t r2 = op->e2_r2 & MASK_32;
  if (tidx_head & V_RED_HEAD_REPEAT_MASK) {
    uint64_t thread_size = (tidx_head >> V_RED_HEAD_THREAD_OFFSET) & V_RED_HEAD_THREAD_MASK;
    tidx_head = (tidx_head | 1ul << V_RED_HEAD_INC_FLAG_OFFSET) + ((r2 * e1 * e2 * thread_size) << 32) - 1ul;
    op->tidx_head = tidx_head;
    PushSkipStride(e2);
    return (tidx_head >> 32) << V_TILE_CUR_IDX_OFFSET | 1ul << V_TILE_SKIP_FLAG_OFFSET | r2 << V_TILE_COUNT_OFFSET;
  }
  uint64_t blocknum = vBlockNum();
  uint64_t blockidx = vBlockIdx();
  uint64_t e_idx = op->e & MASK_32;
  uint64_t e_num = op->e >> 32;
  if (e_idx >= e_num) {
    return 0;
  }
  uint64_t tidx;
  if (e_idx + blocknum / 2 < e_num) {
    op->e += blocknum;
    uint64_t e_cur = e_idx + blockidx;
    if (e_cur >= e_num) {
      return 0;
    }
    uint64_t e0_idx = e_cur / (e1 * e2);
    uint64_t e0_offset = e_cur % (e1 * e2);
    uint64_t e1_idx = e0_offset / e2;
    uint64_t e2_idx = e0_offset % e2;
    tidx = e0_idx * (r1 * e1 * r2 * e2) + e1_idx * (r2 * e2) + e2_idx;
    op->tidx_head = tidx << 32 | MakeRedHead<uint64_t>(r1 - 1, 1);
  } else {
    uint64_t group_num = e_num - e_idx;
    uint64_t thread_size = min(r1, blocknum / group_num);
    op->e += group_num;
    if (blockidx >= thread_size * group_num) {
      PadRedSync(op->user_cnt);
      return 0;
    }
    uint64_t thread_idx = blockidx % thread_size;
    uint64_t group_idx = blockidx / thread_size;
    uint64_t e_cur = e_idx + group_idx;
    uint64_t e0_idx = e_cur / (e1 * e2);
    uint64_t e0_offset = e_cur % (e1 * e2);
    uint64_t e1_idx = e0_offset / e2;
    uint64_t e2_idx = e0_offset % e2;
    uint64_t group_tile = e0_idx * (r1 * e1 * r2 * e2) + e1_idx * (r2 * e2) + e2_idx;
    uint64_t repeat = (r1 - thread_idx - 1) / thread_size + 1;
    tidx = group_tile + r2 * e2 * e1 * thread_idx;
    op->tidx_head = tidx << 32 | MakeRedHead<uint64_t>(repeat - 1, thread_size);
  }
  PushSkipStride(e2);
  return tidx << V_TILE_CUR_IDX_OFFSET | r2 << V_TILE_COUNT_OFFSET | 1ul << V_TILE_SKIP_FLAG_OFFSET;
}

#if __VM_ARCH__ == 220
__aicore_inline__ uint64_t VisitMix(bcodeptr_t pc) {
  vVisitMix op;
  vVisitMix::Decode(pc, op);
  uint64_t blockgroup = vBlockGroup();
  uint64_t vgroupidx = vGroupIdx(blockgroup);
  uint64_t vgroupnum = vGroupNum(blockgroup);
  __gm__ vCubeOp *__restrict__ cube = op.cube;
  uint64_t group_idx = op.group_idx > 0 ? op.group_idx : vgroupidx;
  if (group_idx >= cube->group_num) {
    return 0;
  }
  if (group_idx >= vgroupnum) {
    if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
      int32_t subblocknum = op.subtile1 == 0 ? 1 : 2;
      int32_t expect = ((group_idx - vgroupidx) / vgroupnum + 1) / 2 * subblocknum * (cube->rank_size - 1);
      uint64_t pingpong_offset = op.pingpong ? vgroupidx : vgroupidx + vgroupnum;
      __gm__ int32_t *peer_mem =
        (__gm__ int32_t *)((__gm__ uint8_t *)cube->gm_c + PEERMEM_ATOMIC_OFFSET + pingpong_offset * CACHE_LINE_SIZE);
      while (1) {
        if (*peer_mem == expect) {
          break;
        }
        dcci(peer_mem, 0);
        for (int64_t i = 0; i < 10; ++i) {
          __asm__ __volatile__("nop");
        }
      }
    }
    ffts_cross_core_sync(PIPE_MTE3, vFftsSyncConfig(2, 1));
  }
  uint64_t tile;
  if (get_subblockid() == 0) {
    tile = op.subtile0 << V_TILE_COUNT_OFFSET;
    if (op.subtile1 == 0) {
      tile |= 1ul << V_TILE_TAIL_FLAG_OFFSET;
    }
  } else {
    tile = op.subtile0 << V_TILE_CUR_IDX_OFFSET | 1ul << V_TILE_TAIL_FLAG_OFFSET | op.subtile1 << V_TILE_COUNT_OFFSET;
  }
  wait_flag_dev(0);
  __gm__ vMixGroupMsg *msg = reinterpret_cast<__gm__ vMixGroupMsg *>(cube->gm_pos + sizeof(vMixGroupMsg) * vgroupidx);
  dcci(msg, 0);
  uint64_t pos = msg->pos[op.pingpong];
  uint64_t cidx = vGetBitRange(pos, V_MM_POS_C_OFFSET, V_MM_POS_C_BITS);
  uint64_t midx = vGetBitRange(pos, V_MM_POS_M_OFFSET, V_MM_POS_M_BITS);
  uint64_t nidx = vGetBitRange(pos, V_MM_POS_N_OFFSET, V_MM_POS_N_BITS);
  bool last_m = (midx + 1) * cube->m0 > cube->m_real;
  bool last_n = (nidx + 1) * cube->n0 > cube->n_real;
  uint64_t offset = (cidx * cube->m_real + midx * cube->m0) * cube->n_real + nidx * cube->n0;
  vVisitMix::UpdateShard(pc, pos, offset, last_m, last_n);
  vVisitMix::Update(pc, group_idx + vgroupnum, op.pingpong ^ 1ul);
  if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
    uint64_t pingpong_offset = op.pingpong ? vgroupnum + vgroupidx : vgroupidx;
    __gm__ uint64_t *peer_flag_mem =
      (__gm__ uint64_t *)((__gm__ uint8_t *)cube->gm_c + PEERMEM_FLAG_OFFSET + pingpong_offset * CACHE_LINE_SIZE);
    *peer_flag_mem = (static_cast<uint64_t>(cube->unique_id) << 32) | (offset + 1);
    dcci(peer_flag_mem, 0);
  }
  return tile;
}

#else
__aicore_inline__ uint64_t VisitMix(bcodeptr_t pc) {
  vVisitMix op;
  vVisitMix::Decode(pc, op);
  uint64_t blockgroup = vBlockGroup();
  uint64_t vgroupidx = vGroupIdx(blockgroup);
  uint64_t vgroupnum = vGroupNum(blockgroup);
  __gm__ vCubeOp *__restrict__ cube = op.cube;
  uint64_t group_idx = op.group_idx > 0 ? op.group_idx : vgroupidx;
  if (group_idx >= cube->group_num) {
    return 0;
  }
  if (cube->flags & V_CUBE_FLAG_GROUP_SET) {
    if (group_idx >= vgroupnum) {
      if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
        // TODO: cube peer_store backsync
      }
      set_intra_block(PIPE_MTE3, V_INTRA_GM_MIX_BACKWARD_ID);
    }
    wait_intra_block(PIPE_MTE2, V_INTRA_GM_MIX_FORWARD_ID);
  } else if (cube->flags & V_CUBE_FLAG_STORE_UB) {
    set_intra_block(PIPE_V, V_INTRA_UB_MIX_BACKWARD_ID);
    wait_intra_block(PIPE_V, V_INTRA_UB_MIX_FORWARD_ID);
  }
  uint64_t tile;
  if (get_subblockid() == 0) {
    tile = op.subtile0 << V_TILE_COUNT_OFFSET;
    if (op.subtile1 == 0) {
      tile |= 1ul << V_TILE_TAIL_FLAG_OFFSET;
    }
  } else {
    tile = op.subtile0 << V_TILE_CUR_IDX_OFFSET | 1ul << V_TILE_TAIL_FLAG_OFFSET | op.subtile1 << V_TILE_COUNT_OFFSET;
  }
  uint64_t midx;
  uint64_t nidx;
  uint64_t cidx;
  vCubeOp::IndexCompute(group_idx, cube->swizzle, cube->m_loop, cube->n_loop, midx, nidx, cidx);
  uint64_t pos = cidx << V_MM_POS_C_OFFSET | midx << V_MM_POS_M_OFFSET | nidx << V_MM_POS_N_OFFSET;
  bool last_m = (midx + 1) * cube->m0 > cube->m_real;
  bool last_n = (nidx + 1) * cube->n0 > cube->n_real;
  uint64_t offset = (cidx * cube->m_real + midx * cube->m0) * cube->n_real + nidx * cube->n0;
  vVisitMix::UpdateShard(pc, pos, offset, last_m, last_n);
  vVisitMix::Update(pc, group_idx + vgroupnum, op.pingpong ^ 1ul);
  if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
    // Todo: cube peer_store backsync
  }
  return tile;
}
#endif

__aicore_inline__ uint64_t VisitReorder(bcodeptr_t pc) {
  vVisitReorder op;
  vVisitReorder::Decode(pc, op);
  if (op.wave_block >= op.parallel_extent) {
    if (op.coord_num == 0) {
      return 0;
    }
    op.wave_tidx -= op.parallel_stride * op.wave_block;
    op.wave_block = 0;
    for (uint64_t cidx = 0; cidx < op.coord_num; ++cidx) {
      uint64_t stride;
      uint64_t extent;
      uint64_t value;
      vVisitReorder::DecodeCoord(pc, cidx, stride, extent, value);
      if (uint64_t n_value = value + 1; n_value < extent) {
        op.wave_tidx += stride;
        vVisitReorder::UpdateCoord(pc, cidx, n_value);
        if (cidx == op.coord_num - 1 && n_value == extent - 1) {
          vVisitReorder::ClearCoord(pc);
        }
        break;
      }
      op.wave_tidx -= stride * value;
      vVisitReorder::UpdateCoord(pc, cidx, 0);
    }
  }
  uint64_t blocknum = vBlockNum();
  uint64_t blockidx = vBlockIdx();
  vVisitReorder::UpdateWave(pc, op.wave_block + blocknum, op.wave_tidx + op.parallel_stride * blocknum);
  uint64_t parallel_idx = op.wave_block + blockidx;
  if (parallel_idx >= op.parallel_extent) {
    return 1ul << V_TILE_CUR_IDX_OFFSET;
  }
  uint64_t tidx = op.wave_tidx + op.parallel_stride * blockidx;
  bool last_block = parallel_idx == op.parallel_extent - 1;
  uint64_t tile = tidx << V_TILE_CUR_IDX_OFFSET | (last_block ? op.loop_tail : op.loop_num) << V_TILE_COUNT_OFFSET;
  if (last_block && op.coord_num == 0) {
    tile |= 1ul << V_TILE_TAIL_FLAG_OFFSET;
  }
  if (op.loop_stride > 1) {
    PushSkipStride(op.loop_stride);
    tile |= 1ul << V_TILE_SKIP_FLAG_OFFSET;
  }
  return tile;
}

__aicore_inline__ uint64_t VisitPipeSet(bcodeptr_t pc) {
  vVisitPipeSet op;
  vVisitPipeSet::Decode(pc, op);
  uint64_t end_cnt = op.idx & 0xfffful;
  uint64_t set_cnt = (op.idx >> 16) & 0xfffful;
  if (uint64_t next_set = set_cnt + 1; op.step * next_set == end_cnt) {
    auto msg = (__gm__ vPipeMsg *)(op.gm + vBlockIdx() * CACHE_LINE_SIZE);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    msg->pipe_cnt = next_set;
    if (next_set == 1) {
      msg->magic_id = vPipeMsg::MagicID(op.uid);
    }
    dcci(msg, 0);
    op.idx += 1ul << 16;
  }
  vVisitPipeSet::UpdateIdx(pc, op.idx + 1);
  VisitFunc nest_func = (VisitFunc)GetFunc(op.visit_id, RegTable()->vm_code_base);
  return nest_func(pc + vVisitPipeSet::NEST_VISIT_OFFSET);
}

__aicore_inline__ uint64_t VisitPipeWait(bcodeptr_t pc) {
#if __VM_ARCH__ == 220
  vVisitPipeWait op;
  vVisitPipeWait::Decode(pc, op);
  VisitFunc nest_func = (VisitFunc)GetFunc(op.visit_id, RegTable()->vm_code_base);
  uint64_t tile = nest_func(pc + vVisitPipeWait::NEST_VISIT_OFFSET);
  if (tile) {
    uint64_t start_idx = op.idx & 0xfffful;
    uint64_t wait_cnt = (op.idx >> 16) & 0xfffful;
    if (start_idx >= op.step * wait_cnt) {
      uint64_t mask = (1ul << op.prod_num) - 1;
      uint64_t min_set_cnt = INVALID_INT;
      while (mask) {
        uint64_t msg_addr = op.gm;
        for (uint64_t i = 0; i < op.prod_num; ++i) {
          if (mask & (1ul << i)) {
            auto msg = (__gm__ volatile vPipeMsg *)(msg_addr);
            dcci(msg, 0);
            uint64_t pipe_cnt = msg->pipe_cnt;
            if (pipe_cnt > wait_cnt && (wait_cnt > 0 || msg->magic_id == vPipeMsg::MagicID(op.uid))) {
              mask ^= 1ul << i;
              if (pipe_cnt < min_set_cnt) {
                min_set_cnt = pipe_cnt;
              }
            }
          }
          msg_addr += CACHE_LINE_SIZE;
        }
      }
      op.idx += (min_set_cnt - wait_cnt) << 16;
    }
    vVisitPipeWait::UpdateIdx(pc, op.idx + 1);
  }
  return tile;
#endif
  return -1;
}
