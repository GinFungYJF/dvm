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

#ifndef _DVM_ISA_H_
#define _DVM_ISA_H_

#include <stdint.h>

#ifdef _CCE_KERNEL_
#define __aicore_inline__ static[aicore] __attribute__((always_inline))
#define __bcode__ __gm__
#define bcodeptr_t __bcode__ uint64_t *__restrict__
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#else
#define __gm__
#define __bcode__
#define __aicore_inline__ static inline
#define bcodeptr_t uint64_t *
#define __aicore__
#endif
#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif 

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

enum vAccInsnID {
  V_LOAD = 0,
  V_LOAD_DUMMY,
  V_LOAD_GATHER_B16, // [c310]
  V_LOAD_GATHER_B32, // [c310]
  V_LOAD_VIEW,
  V_LOAD_VIEW_X,
  V_LOAD_VIEW_TRANS,
  V_SLOAD,
  V_MULTI_LOAD, // [c220]
  V_PINGPONG_LOAD,
  V_PINGPONG_PEER_LOAD, // [c220]
  V_PEER_LOAD, // [c220]
  V_PEER_LOAD_MIX, // [c220]
  V_STORE,
  V_STORE_ATOMIC,
  V_STORE_ATOMIC_WRAP,
  V_STORE_COND,
  V_STORE_VIEW,
  V_STORE_VIEW_X,
  V_SSTORE,
  V_STORE_AG,  // [c220] For AllGather
  V_STORE_RS,  // [c220] For ReduceScatter
  V_PEER_STORE, // [c220]
  V_PEER_STORE_MIX, // [c220]
  V_ACCESS_NONE,
};

enum vSimdInsnID {
  V_COPY = 0,
  V_NOP,
  V_BROADCAST_S,
  V_BROADCAST_S_B8,  // [c310]
  V_BROADCAST_S_B16,
  V_BROADCAST_S_B64,
  V_BROADCAST_X_B8,  // [c310]
  V_BROADCAST_X_B16,
  V_BROADCAST_X_B32,
  V_BROADCAST_X_B64,  // [c310]
  V_BROADCAST_Y,

  V_CAST_FP32_TO_FP16,
  V_CAST_FP32_TO_BF16,
  V_CAST_FP32_TO_INT32,
  V_CAST_FP32_TO_INT64,
  V_CAST_FP16_TO_BOOL,
  V_CAST_FP16_TO_FP32,
  V_CAST_FP16_TO_INT32,
  V_CAST_BF16_TO_FP32,
  V_CAST_BF16_TO_INT32,
  V_CAST_INT32_TO_FP16,
  V_CAST_INT32_TO_FP32,
  V_CAST_INT32_TO_INT64,
  V_CAST_INT64_TO_FP32,
  V_CAST_INT64_TO_INT32,
  V_CAST_BOOL_TO_FP16,

  V_SQRT,
  V_SQRT_FP16,
  V_ABS,
  V_ABS_FP16,
  V_ABS_INT32,  // [c310]
  V_LOG,
  V_LOG_FP16,
  V_EXP,
  V_EXP_FP16,
  V_POW,
  V_ROUND,
  V_FLOOR,
  V_CEIL,
  V_TRUNC,
  V_ISFINITE,
  V_ISFINITE_FP16,
  V_ISFINITE_BF16,  // [c310]

  V_ADDS,
  V_ADDS_FP16,
  V_ADDS_BF16,  // [c310]
  V_ADDS_INT32,
  V_MULS,
  V_MULS_FP16,
  V_MULS_BF16,  // [c310]
  V_MULS_INT32,
  V_DIVS,
  V_DIVS_FP16,
  V_SDIV,
  V_SDIV_FP16,
  V_CMPS,
  V_CMPS_BOOL,  // [c310]
  V_CMPS_FP16,
  V_CMPS_BF16,   // [c310]
  V_CMPS_INT32,  // [c310]
  V_MAXS,
  V_MAXS_FP16,
  V_MAXS_BF16,  // [c310]
  V_MAXS_INT32,
  V_MINS,
  V_MINS_FP16,
  V_MINS_BF16,  // [c310]
  V_MINS_INT32,

  V_ADD,
  V_ADD_FP16,
  V_ADD_BF16,  // [c310]
  V_ADD_INT32,
  V_ADD_INT64,  // [c310]
  V_SUB,
  V_SUB_FP16,
  V_SUB_BF16,  // [c310]
  V_SUB_INT32,
  V_SUB_INT64,  // [c310]
  V_MUL,
  V_MUL_FP16,
  V_MUL_BF16,  // [c310]
  V_MUL_INT32,
  V_DIV,
  V_DIV_FP16,
  V_MIN,
  V_MIN_FP16,
  V_MIN_BF16,  // [c310]
  V_MIN_INT32,
  V_MAX,
  V_MAX_FP16,
  V_MAX_BF16,  // [c310]
  V_MAX_INT32,

  V_CMP,
  V_CMP_BOOL,  // [c310]
  V_CMP_FP16,
  V_CMP_BF16,  // [c310]
  V_CMP_INT32,
  V_CMP_INT64,  // [c310]
  V_SEL,
  V_SEL_FP16,
  V_SEL_BF16,  // [c310]
  V_SEL_INT32,
  V_SEL_INT64,         // [c310]
  V_LOGICAL_NOT_BOOL,  // [c310]
  V_LOGICAL_AND_BOOL,  // [c310]
  V_LOGICAL_OR_BOOL,   // [c310]

  V_RSUM_X,
  V_RSUM_Y,
  V_RSUM_JOIN,
  V_RMAX_X,
  V_RMAX_X_FP16,
  V_RMAX_Y,
  V_RMAX_Y_FP16,
  V_RMIN_X,
  V_RMIN_X_FP16,
  V_RMIN_Y,
  V_RMIN_Y_FP16,
  V_ATOMICCUM,
  V_ATOMICCUM_FP16,
  V_ELEMENT_ANY,

  V_ONE_HOT,
  V_ONE_HOT_B16,

  V_CLR_PAD,
  V_CLR_PAD_B16,
  V_REMOVEPAD,
  V_REMOVEPAD_U16,
  V_EXTRACT_B32,  // [c220]
  V_PACK_B32,     // [c220]

  V_CUSTOM,
  V_COPY_CUBE_TILE,  // [c310]
  V_NONE,
};

enum vVisitID {
  V_VISIT_RED_1 = 0,
  V_VISIT_RED_2,
  V_VISIT_RED_3,
  V_VISIT_RED_4,
  V_VISIT_MIX,
  V_VISIT_REORDER, // [c220]
  V_VISIT_PIPE_SET, // [c220]
  V_VISIT_PIPE_WAIT, // [c220]
  V_VISIT_NONE,
};

enum CommType {
  kCommAllReduce = 0,
  kCommReduceScatter,
  kCommAllGather,
};

enum vReduceOpType {
  V_RED_SUM = 0,
  V_RED_MAX,
  V_RED_MIN,
  V_END,
};

enum vAtomicType {
  V_ATOMIC_FP32 = 0,
  V_ATOMIC_FP16,
  V_ATOMIC_BF16,
};

// head(simd):
//  ID(16) << 48 | ext(26) << 22 | b_wait_event(3) << 19 | b_set_event(3) << 16 | wait_event(3) << 13 | set_event(3) <<
//  10 | len(3) << 7 | back_wait(1) << 6 | back_set(1) << 5 | wait_flag(1) << 4 | set_flag(1) << 3 | bar_flag(1) << 2 |
//  reserved(1) << 1 | SIMD_FLAG(1)
// head(load/store):
//  ID(16) << 48 | ext(34) << 14 | wait_event(3) << 11 | set_event(3) << 8 | len(4) << 4 |
//  wait_flag(1) << 3 | set_flag(1) << 2 | reserved(1) << 1 | SIMD_FLAG(1)

// common area
#define V_HEAD_SIMD_FLAG_OFFSET 0
#define V_HEAD_ID_OFFSET 48
#define V_HEAD_ID_MASK 0xfffful
#define V_HEAD_EVENT_MASK 0x7ul

// simd
#define V_HEAD_BAR_FLAG_OFFSET 2
#define V_HEAD_SET_FLAG_OFFSET 3
#define V_HEAD_WAIT_FLAG_OFFSET 4
#define V_HEAD_BACK_SET_OFFSET 5
#define V_HEAD_BACK_WAIT_OFFSET 6
#define V_HEAD_SIZE_OFFSET 7
#define V_HEAD_SET_EVENT_OFFSET 10
#define V_HEAD_WAIT_EVENT_OFFSET 13
#define V_HEAD_B_SET_EVENT_OFFSET 16
#define V_HEAD_B_WAIT_EVENT_OFFSET 19
#define V_HEAD_EXT_OFFSET 22
#define V_HEAD_EXT_MASK 0x3fffffful
#define V_HEAD_SIZE_MASK 0x7ul

// load/store
#define V_M_HEAD_SET_FLAG_OFFSET 2
#define V_M_HEAD_WAIT_FLAG_OFFSET 3
#define V_M_HEAD_SIZE_OFFSET 4
#define V_M_HEAD_SET_EVENT_OFFSET 8
#define V_M_HEAD_WAIT_EVENT_OFFSET 11
#define V_M_HEAD_EXT_OFFSET 14
#define V_M_HEAD_EXT_MASK 0x3fffffffful
#define V_M_HEAD_SIZE_MASK 0xful

#define V_FFTS_GM_MIX_FORWARD_ID 0
#define V_FFTS_GM_MIX_BACKWARD_ID 1
#define V_INTRA_GM_MIX_FORWARD_ID 2
#define V_INTRA_GM_MIX_BACKWARD_ID 3
#define V_INTRA_UB_MIX_FORWARD_ID 4
#define V_INTRA_UB_MIX_BACKWARD_ID 5
#define V_INTRA_BLOCK_AIV_OFFSET 16

// Comm related
#define PEERMEM_FLAG_OFFSET (200 * 1024 * 1024)          // 200MB
#define PEERMEM_TWOSHOT_OFFSET (100 * 1024 * 1024)       // 100MB
#define PEERMEM_TWOSHOT_FLAG_OFFSET (202 * 1024 * 1024)  // 202MB
#define CACHE_LINE_SIZE 512                              // 512 Byte
#define PEERMEM_ATOMIC_OFFSET (PEERMEM_FLAG_OFFSET + 1024 * 512 + 32)

// common mask
// ub address, loop ext, stride should not extent ub size limit
#define V_X_BITS 18
#define V_C_X_BITS 13
#define V_X_MASK 0x3fffful
#define V_RS_MASK 0xful  // repeat stride
#define V_GROUP_OFFSET_SIZE 32
#define vCompactX(x) ((x) >> 5)
#define vDeCompactX(x) ((x) << 5)

#ifndef _CCE_KERNEL_
#include "system.h"
__aicore_inline__ uint64_t vMakeAccHead(uint64_t id, uint64_t ext, uint64_t len) {
  return ext << V_M_HEAD_EXT_OFFSET | len << V_M_HEAD_SIZE_OFFSET |
         dvm::g_system.g_access_func_offset_[id] << V_HEAD_ID_OFFSET;
}
__aicore_inline__ uint64_t vMakeSimdHead(uint64_t id, uint64_t ext, uint64_t len) {
  return ext << V_HEAD_EXT_OFFSET | len << V_HEAD_SIZE_OFFSET |
         dvm::g_system.g_simd_func_offset_[id] << V_HEAD_ID_OFFSET | 1 << V_HEAD_SIMD_FLAG_OFFSET;
}
template <typename T>
__aicore_inline__ T min(T a, T b) {
  return a < b ? a : b;
}
#else
__aicore_inline__ uint64_t vMakeAccHead(uint64_t id, uint64_t ext, uint64_t len) { return 0; }
__aicore_inline__ uint64_t vMakeSimdHead(uint64_t id, uint64_t ext, uint64_t len) { return 0; }
#endif

__aicore_inline__ uint64_t vGetBitRange(uint64_t x, uint64_t offset, uint64_t len) {
  return x << (64 - len - offset) >> (64 - len);
}

__aicore_inline__ void vClrBitRange(uint64_t &x, uint64_t offset, uint64_t len) {
  x &= (~(((1ul << len) - 1) << offset));
}

template <typename T>
__aicore_inline__ T DecodeScalar(__bcode__ uint32_t *scalr_offset) {
  return *(__bcode__ T *)(scalr_offset);
}

template <typename T>
__aicore_inline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__aicore_inline__ T RoundUp(T num, T rnd) {
  return (num + rnd - 1) / rnd * rnd;
}

template <typename T>
__aicore_inline__ T RoundDown(T num, T rnd) {
  return num / rnd * rnd;
}

template <typename T>
__aicore_inline__ T RoundUpSafe(T num, T rnd) {
  return rnd == 0 ? 0 : RoundUp(num, rnd);
}

template <typename T>
__aicore_inline__ T RoundDownSafe(T num, T rnd) {
  return rnd == 0 ? 0 : RoundDown(num, rnd);
}

template <typename T>
__aicore_inline__ T Gcd(T a, T b) {
  while (b != 0) {
    T c = b;
    b = a % b;
    a = c;
  }
  return a;
}

template <typename T>
__aicore_inline__ T Lcm(T a, T b) {
  return a * b / Gcd(a, b);
}

struct vUnary {
  uint64_t xd;
  uint64_t xn;
  uint64_t count;
  // pc[0]: xd
  // pc[1]: xn(18) << 32 | count(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vUnary &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.count = data & 0xfffful;
    op.xn = data >> 32;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vUnary &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.xn << 32 | op.count;
    return size;
  }
};

struct vAtomicCum {
  enum { ROUND_OFFSET = 2 };
  uint64_t xd;
  uint64_t xn;
  uint64_t count;
  uint64_t round_rank;
  uint64_t red_op;
  // pc[0]: xd
  // pc[1]: red_op(4) << 60 | xn(18) << 32 | count(16) | round_rank
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vAtomicCum &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.red_op = (data >> 60) & 0xful;
    op.count = (data >> 16) & 0xfffful;
    op.xn = (data >> 32) & V_X_MASK;
    op.round_rank = data & 0xful;
  }

  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vAtomicCum &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vAtomicCum::ROUND_OFFSET + round_size;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.red_op << 60 | op.xn << 32 | op.count << 16 | op.round_rank;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vAtomicCum::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vRemovePad {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  uint64_t iter_num;
  uint64_t rs;
  // pc[0]: xd
  // pc[1]: xn(18) << 46 | rs(8) << 32 | iter_num(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vRemovePad &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.repeat = data & 0xfffful;
    op.iter_num = (data >> 16) & 0xfffful;
    op.xn = data >> 46;
    op.rs = (data >> 32) & 0xfful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vRemovePad &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.xn << 46 | op.rs << 32 | op.iter_num << 16 | op.repeat;
    return size;
  }
};

struct vBinaryS {
  uint64_t xn;
  uint64_t xd;
  uint64_t count;
  uint64_t scalar;
  // pc[0]: xn
  // pc[1]: scalar(32) << 32 | c_xd(13) << 16 | count(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinaryS &op) {
    uint32_t data = *((__bcode__ uint32_t *)pc + 2);
    op.count = data & 0xffffu;
    op.xd = vDeCompactX(data >> 16);
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  template <typename T>
  __aicore_inline__ T GetScalar(bcodeptr_t pc) {
    return DecodeScalar<T>((__bcode__ uint32_t *)pc + 3);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBinaryS &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.scalar << 32 | vCompactX(op.xd) << 16 | op.count;
    return size;
  }
};

struct vBinary {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t count;
  // pc[0]: xn(18)
  // pc[1]: count(16) << 48 | xd(18) << 18 | xm(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinary &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xm = data & V_X_MASK;
    op.xd = (data >> 18) & V_X_MASK;
    op.count = data >> 48;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBinary &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.count << 48 | op.xd << 18 | op.xm;
    return size;
  }
};

struct vBinaryWS {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t ws0;
  uint64_t ws1;
  uint64_t count;
  // pc[0]: xn(18)
  // pc[1]: count(16) << 48 | xd(18) << 18 | xm(18)
  // pc[2]: ws1(18) << 18 | ws0(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinaryWS &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xm = data & V_X_MASK;
    op.xd = (data >> 18) & V_X_MASK;
    op.count = data >> 48;
    data = pc[2];
    op.ws0 = data & V_X_MASK;
    op.ws1 = (data >> 18) & V_X_MASK;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBinaryWS &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.count << 48 | op.xd << 18 | op.xm;
    pc[2] = op.ws1 << 18 | op.ws0;
    return size;
  }
};

struct vPack {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t ws;
  uint64_t count;
  // pc[0]: xn(18)
  // pc[1]: count(16) << 48 | xd(18) << 18 | xm(18)
  // pc[2]: ws(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vPack &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xm = data & V_X_MASK;
    op.xd = (data >> 18) & V_X_MASK;
    op.count = data >> 48;
    data = pc[2];
    op.ws = data & V_X_MASK;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vPack &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.count << 48 | op.xd << 18 | op.xm;
    pc[2] = op.ws;
    return size;
  }
};

enum vCompareType {
  V_CMP_EQ = 0,
  V_CMP_NE,
  V_CMP_GT,
  V_CMP_GE,
  V_CMP_LT,
  V_CMP_LE,
  V_CMP_ALL,
};

struct vCompare {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t type;
  uint64_t count;
  uint64_t ws;
  // pc[0]: op(4) << 18 | xn(18)
  // pc[1]: count(16) << 48 | ws(13) << 35 | c_xd(13) << 18 | xm(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCompare &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.type = (head >> (V_HEAD_EXT_OFFSET + 18)) & 0xful;
    uint64_t data = pc[1];
    op.xm = data & V_X_MASK;
    op.xd = vDeCompactX(vGetBitRange(data, 18, 13));
    op.ws = vDeCompactX(vGetBitRange(data, 35, 13));
    op.count = data >> 48;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vCompare &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.type << 18 | op.xn, size);
    pc[1] = op.count << 48 | vCompactX(op.ws) << 35 | vCompactX(op.xd) << 18 | op.xm;
    return size;
  }
};

struct vCompareS {
  uint64_t xd;
  uint64_t xn;
  uint64_t ws;
  uint64_t type;
  uint64_t count;
  uint64_t scalar;
  // pc[0]: op(4) << 18 | xn(18)
  // pc[1]: count(16) << 48 | ws(18) << 18 | xd(18)
  // pc[2]: scalar
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCompareS &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.type = (head >> (V_HEAD_EXT_OFFSET + 18)) & 0xful;
    uint64_t data = pc[1];
    op.xd = data & V_X_MASK;
    op.ws = (data >> 18) & V_X_MASK;
    op.count = data >> 48;
  }
  template <typename T>
  __aicore_inline__ T GetScalar(bcodeptr_t pc) {
    return DecodeScalar<T>((__bcode__ uint32_t *)pc + 4);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vCompareS &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.type << 18 | op.xn, size);
    pc[1] = op.count << 48 | op.ws << 18 | op.xd;
    pc[2] = op.scalar;
    return size;
  }
};

struct vExtract {
  uint64_t xd;
  uint64_t xn;
  uint64_t count;
  uint64_t slot;
  // pc[0]:
  // pc[1]: slot(4) << 60 | reserve(8) << 52 | xd(18) << 34 | xn(18) << 16 | count(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vExtract &op) {
    uint64_t data = pc[1];
    op.count = data & 0xfffful;
    op.xn = (data >> 16)  & V_X_MASK;
    op.xd = (data >> 34) & V_X_MASK;
    op.slot = data >> 60;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vExtract &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, 0, size);
    pc[1] = op.slot << 60 | op.xd << 34 | op.xn << 16 | op.count;
    return size;
  }
};

struct vBroadcastS {
  uint64_t xd;
  uint64_t count;
  uint64_t scalar;
  // pc[0]: xd
  // pc[1]: val << 32 | repeat
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastS &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.count = data & 0xfffffffful;
    op.scalar = (data >> 32) & 0xfffffffful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastS &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.scalar << 32 | op.count;
    return size;
  }
};

struct vBroadcastS_B64 {
  uint64_t xd;
  uint64_t count;
  uint64_t scalar;
  uint64_t high;
  // pc[0]: xd
  // pc[1]: count
  // pc[2]: high(32) << 32 | scalar(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastS_B64 &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.count = pc[1];
    op.scalar = pc[2] & 0xfffffffful;
    op.high = pc[2] >> 32;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastS_B64 &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.count;
    pc[2] = op.high << 32 | op.scalar;
    return size;
  }
};

struct vSelect {  // 24B
  uint64_t xn;
  uint64_t xd;
  uint64_t count;
  uint64_t xm;
  uint64_t cond;
  uint64_t ws;
  // pc[0]: xn
  // pc[1]: count(16) << 48 | xm(18) << 18 | cond(18)
  // pc[2]: xd(18) << 32 | ws(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSelect &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.count = data >> 48;
    op.xm = (data >> 18) & V_X_MASK;
    op.cond = data & V_X_MASK;
    data = pc[2];
    op.xd = (data >> 32) & V_X_MASK;
    op.ws = data & V_X_MASK;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vSelect &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.count << 48 | op.xm << 18 | op.cond;
    pc[2] = op.xd << 32 | op.ws;
    return size;
  }
};

// [iter_num, lead_num, count] = Broadcast([iter_num, lead_num + pad, 1])
struct vBroadcastX {
  uint64_t xd;
  uint64_t xn;
  uint64_t count;
  uint64_t lead_num;
  uint64_t iter_num;
  uint64_t lead_pad;
  // pc[0]:  lead_pad(8) << 18 | xn(18)
  // pc[1]:  c_xd(16) << 48 | iter_num(16) << 32 | lead_num(16) << 16 | count(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastX &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.lead_pad = (head >> (V_HEAD_EXT_OFFSET + 18)) & 0xfful;
    uint64_t data = pc[1];
    op.count = data & 0xfffful;
    op.lead_num = (data >> 16) & 0xfffful;
    op.iter_num = (data >> 32) & 0xfffful;
    op.xd = vDeCompactX(data >> 48);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastX &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.lead_pad << 18 | op.xn, size);
    pc[1] = vCompactX(op.xd) << 48 | op.iter_num << 32 | op.lead_num << 16 | op.count;
    return size;
  }
};

// [iter_num, dup_num, dup_stride] = Broadcast([iter_num, 1, dup_stride])
struct vBroadcastY {
  uint64_t xd;
  uint64_t xn;
  uint64_t iter_num;
  uint64_t dup_num;
  uint64_t dup_stride;
  // pc[0]:  xn(18)
  // pc[1]:  c_xd(16) << 48 | iter_num(16) << 32 | dup_num(16) << 16 | dup_stride(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastY &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.dup_stride = data & 0xfffful;
    op.dup_num = (data >> 16) & 0xfffful;
    op.iter_num = (data >> 32) & 0xfffful;
    op.xd = vDeCompactX(data >> 48);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastY &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = vCompactX(op.xd) << 48 | op.iter_num << 32 | op.dup_num << 16 | op.dup_stride;
    return size;
  }
};

// [dup_size/dup_block, dup_block/dup_pad, 1] = reduce([dup_size, red_size/red_tail])
struct vReduceX {
  uint64_t xd;
  uint64_t xn;
  uint64_t red_size;
  uint64_t red_tail;
  uint64_t dup_size;
  uint64_t dup_pad;
  uint64_t dup_block;
  uint64_t simd_width;
  // pc[0]: xd
  // pc[1]: dup_size(16) << 32 | red_tail(16) << 16 | red_size(16)
  // pc[2]: simd_width(8) << 56 | xn << 32 | dup_pad << 16 | dup_block
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceX &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data1 = pc[1];
    op.red_size = data1 & 0xfffful;
    op.red_tail = (data1 >> 16) & 0xfffful;
    op.dup_size = data1 >> 32;
    op.xn = (pc[2] >> 32) & V_X_MASK;
    op.simd_width = (pc[2] >> 56) & 0xfful;
  }
  __aicore_inline__ void DecodeBlock(bcodeptr_t pc, vReduceX &op) {
    uint64_t data2 = pc[2];
    op.dup_block = data2 & 0xfffful;
    op.dup_pad = (data2 >> 16) & 0xfffful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vReduceX &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.dup_size << 32 | op.red_tail << 16 | op.red_size;
    pc[2] = op.simd_width << 56 | op.xn << 32 | op.dup_pad << 16 | op.dup_block;
    return size;
  }
};

// [dup_num, 1, iter_size] = reduce([dup_num, red_size/red_tail, iter_size])
struct vReduceY {
  uint64_t xd;
  uint64_t xn;
  uint64_t iter_size;
  uint64_t red_size;
  uint64_t red_tail;
  uint64_t dup_num;
  // pc[0]: xd(18)
  // pc[1]: dup_num(16) << 48 | red_tail(16) << 32 | red_size(16) << 16 | iter_size(16)
  // pc[2]: xn(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceY &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.iter_size = data & 0xfffful;
    op.red_size = (data >> 16) & 0xfffful;
    op.red_tail = (data >> 32) & 0xfffful;
    op.dup_num = data >> 48;
    op.xn = pc[2] & V_X_MASK;
  }
  __aicore_inline__ uint64_t GetXd(bcodeptr_t pc, uint64_t head) { return (head >> V_HEAD_EXT_OFFSET) & V_X_MASK; }

  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vReduceY &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.xd, size);
    pc[1] = op.dup_num << 48 | op.red_tail << 32 | op.red_size << 16 | op.iter_size;
    pc[2] = op.xn;
    return size;
  }
};

struct vReduceJoin {
  enum { STORE_COND_OFFSET = 3 };
  enum { RELOC_OFFSET = 2 };
  uint64_t iter_num;
  uint64_t iter_stride;
  uint64_t xd;
  uint64_t xn;
  uint64_t xs;
  uint64_t ws;
  uint64_t seg_tile_rel;
  uint64_t capacity;
  // pc[0]: c_xd(13) << 13 | seg_tile_rel(12)
  // pc[1]: iter_num(16) << 48 | iter_stride(17) << 31 | c_xs(13) << 18 | xn(18)
  // pc[2]: ws
  // pc[3]: store_cond
  // pc[4]: capacity
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceJoin &op) {
    op.seg_tile_rel = vGetBitRange(head, V_HEAD_EXT_OFFSET, 12);
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    uint64_t data = pc[1];
    op.iter_num = data >> 48;
    op.iter_stride = vGetBitRange(data, 31, 17);
    op.xs = vDeCompactX(vGetBitRange(data, 18, 13));
    op.xn = vGetBitRange(data, 0, 18);
    op.ws = pc[2];
    op.capacity = pc[4];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, const vReduceJoin &op) {
    constexpr uint64_t size = 5;
    pc[0] = vMakeSimdHead(V_RSUM_JOIN, vCompactX(op.xd) << V_C_X_BITS, size);
    pc[1] = op.iter_num << 48 | op.iter_stride << 31 | vCompactX(op.xs) << 18 | op.xn;
    pc[2] = op.ws;
    pc[3] = 0;
    pc[4] = op.capacity;
    return size;
  }
};

struct vCopy {
  uint64_t xd;
  uint64_t xn;
  uint64_t lenburst;
  // pc[0]: xn(18)
  // pc[1]: lenburst(16) << 32 | xd(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCopy &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xd = data & V_X_MASK;
    op.lenburst = (data >> 32) & 0xfffful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vCopy &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.lenburst << 32 | op.xd;
    return size;
  }
};

struct vCubeCopy {
  uint64_t xd;
  uint64_t xn;
  uint64_t lenburst;
  uint64_t tile_offset;
  // pc[0]: xn(18)
  // pc[1]: tile_offset(16) << 48 | lenburst(16) << 32 | xd(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCubeCopy &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xd = data & V_X_MASK;
    op.lenburst = (data >> 32) & 0xfffful;
    op.tile_offset = data >> 48;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vCubeCopy &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, op.xn, size);
    pc[1] = op.tile_offset << 48 | op.lenburst << 32 | op.xd;
    return size;
  }
};

struct vNop {
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc) {
    uint64_t size = 1;
    pc[0] = vMakeSimdHead(V_NOP, 0, size);
    return size;
  }
};

// [iter_num, iter_size/iter_stride]
// [iter_num, iter_size/iter_stride]
struct vClearPad {
  uint64_t xd;
  uint64_t iter_num;
  uint64_t iter_size;
  uint64_t iter_stride;
  uint64_t simd_width;
  uint64_t iter_tail;
  uint64_t scalar;
  // pc[0]: simd_width(8) << 18 | xd(18)
  // pc[1]: iter_tail(16) << 48 | iter_stride(16) << 32 | iter_size(16) << 16  | iter_num(16)
  // pc[2]: scalar
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vClearPad &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.simd_width = vGetBitRange(head, V_HEAD_EXT_OFFSET + V_X_BITS, 8);
    uint64_t data = pc[1];
    op.iter_tail = data >> 48;
    op.iter_stride = (data >> 32) & 0xfffful;
    op.iter_size = (data >> 16) & 0xfffful;
    op.iter_num = data & 0xfffful;
    op.scalar = pc[2];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vClearPad &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.simd_width << V_X_BITS | op.xd, size);
    pc[1] = op.iter_tail << 48 | op.iter_stride << 32 | op.iter_size << 16 | op.iter_num;
    pc[2] = op.scalar;
    return size;
  }
};

struct vElementAny {
  enum { STORE_COND_OFFSET = 2 };
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  uint64_t repeat_tail;
  uint64_t simd_width;
  // pc[0]: simd_width(8) << 18 | xn(18)
  // pc[1]: xd(18) << 32 | repeat_tail(16) << 16 | repeat(16)
  // pc[2]: store_cond
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vElementAny &op) {
    op.xn = vGetBitRange(head, V_HEAD_EXT_OFFSET, V_X_BITS);
    op.simd_width = vGetBitRange(head, V_HEAD_EXT_OFFSET + V_X_BITS, 8);
    uint64_t data = pc[1];
    op.xd = data >> 32;
    op.repeat_tail = (data >> 16) & 0xfffful;
    op.repeat = data & 0xfffful;
  }

  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vElementAny &op) {
    uint64_t size = 3;
    pc[0] = vMakeSimdHead(id, op.simd_width << V_X_BITS | op.xn, size);
    pc[1] = op.xd << 32 | op.repeat_tail << 16 | op.repeat;
    pc[2] = 0;
    return size;
  }
};

struct vOneHot {
  enum {
    MODE_X = 0,
    MODE_X_TILE,
    MODE_Y,
    MODE_Y_TILE,
    MODE_Y_TILE_2,
  };
  uint64_t xn;
  uint64_t xd;
  uint64_t data_size;
  uint64_t iter_num;
  uint64_t depth;
  uint64_t dup_round;  // Y: dup_num. TILE_xx: tile_round
  uint64_t mode;
  // pc[0]: reserved(26)
  // pc[1]: data_size(18) << 36 | xd(18) << 18 | xn(18)
  // pc[2]: mode(8) << 48 | dup_round(16) << 32 | depth(16) << 16 | iter_num(16)
  // pc[3]: off_value(32) << 32 | on_value(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, vOneHot &op) {
    uint64_t data = pc[1];
    op.data_size = data >> 36;
    op.xd = vGetBitRange(data, 18, 18);
    op.xn = vGetBitRange(data, 0, 18);
    data = pc[2];
    op.mode = data >> 48;
    op.dup_round = (data >> 32) & 0xfffful;
    op.depth = (data >> 16) & 0xfffful;
    op.iter_num = data & 0xfffful;
  }

  template <typename T>
  __aicore_inline__ void DecodeValue(bcodeptr_t pc, T &on_value, T &off_value) {
    uint64_t data = pc[3];
    off_value = data >> 32;
    on_value = (data << 32) >> 32;
  }

  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, uint32_t on_value, uint32_t off_value,
                                    const vOneHot &op) {
    uint64_t size = 4;
    pc[0] = vMakeSimdHead(id, 0, size);
    pc[1] = op.data_size << 36 | op.xd << 18 | op.xn;
    pc[2] = op.mode << 48 | op.dup_round << 32 | op.depth << 16 | op.iter_num;
    pc[3] = uint64_t(off_value) << 32 | on_value;
    return size;
  }
};

// [dup_size, xd_lead+xd_pad] = Reshape([*, xn_lead+xn_pad])
struct vReshape {
  uint64_t xd;
  uint64_t xn;
  uint64_t xn_lead;
  uint64_t xd_lead;
  uint64_t xn_pad;
  uint64_t xd_pad;
  uint64_t dup_size;
  // pc[0]: c_xd(13) << 13 | c_xn(13)
  // pc[1]: xd_pad(8) << 56 | xn_pad(8) << 48 | dup_size(16) << 32 | xd_lead(16) << 16 | xn_lead(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReshape &op) {
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    op.xn = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.xn_lead = data & 0xfffful;
    op.xd_lead = (data >> 16) & 0xfffful;
    op.dup_size = (data >> 32) & 0xfffful;
    op.xn_pad = (data >> 48) & 0xfful;
    op.xd_pad = data >> 56;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vReshape &op) {
    uint64_t size = 2;
    pc[0] = vMakeSimdHead(id, vCompactX(op.xd) << V_C_X_BITS | vCompactX(op.xn), size);
    pc[1] = op.xd_pad << 56 | op.xn_pad << 48 | op.dup_size << 32 | op.xd_lead << 16 | op.xn_lead;
    return size;
  }
};

struct vCustom {
  enum { PAYLOAD_OFFSET = 2 };
  uint64_t func_addr;
  // pc[0]: reserve(26)
  // pc[1]: func_addr(64)
  // pc[2]+: payload
  __aicore_inline__ void Decode(bcodeptr_t pc, vCustom &op) {
    op.func_addr = pc[1];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, uint64_t payload_size, const vCustom &op) {
    uint64_t size = PAYLOAD_OFFSET + payload_size;
    pc[0] = vMakeSimdHead(id, 0, size);
    pc[1] = op.func_addr;
    return size;
  }
};

struct vSLoad {
  enum { SHARD_OFFSET = 1 };
  enum { RELOC_OFFSET = 2 };
  enum { ROUND_OFFSET = 3 };
  __gm__ void *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t pad_size;
  bool broadcast_m;
  bool broadcast_n;
  uint64_t round_rank;
  uint64_t shard_rel;
  uint64_t type_size;
  // pc[0]: xn(18)
  // pc[1]: tile_stride(24) << 40 | reserved(8) << 32 | type_size(4) << 28 | pad_size(8) << 20 | round_rank(4) << 14 |
  //        broadcast_m(1) << 13 | broadcast_n(1) << 12 | shard_rel(12)
  // pc[2]: gm
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSLoad &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.tile_stride = data >> 40;
    op.type_size = (data >> 28) & 0xful;
    op.pad_size = (data >> 20) & 0xfful;
    op.round_rank = (data >> 14) & 0xful;
    op.broadcast_m = (data >> 13) & 0x1ul;
    op.broadcast_n = (data >> 12) & 0x1ul;
    op.shard_rel = data & 0xffful;
    op.gm = reinterpret_cast<__gm__ void *>(pc[2]);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vSLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vSLoad::ROUND_OFFSET + round_size;
    pc[0] = vMakeAccHead(id, op.xn, size);
    pc[1] = op.tile_stride << 40 | op.type_size << 28 | op.pad_size << 20 | op.round_rank << 14 |
            uint64_t(op.broadcast_m) << 13 | uint64_t(op.broadcast_n) << 12;
    pc[2] = reinterpret_cast<uint64_t>(op.gm);
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vSLoad::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

using vSStore = vSLoad;

struct vLoad {
  enum { ROUND_OFFSET = 3 };
  enum { RELOC_OFFSET = 1 };
  __gm__ void *from;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t body_iter;
  uint64_t tail_iter;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: from
  // pc[2]: round_rank(4) << 60 | pad_size(8) << 52 | iter_size(18) << 34 | tail_iter(18) << 16 | body_iter(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vLoad &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + 13, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13));
    op.from = reinterpret_cast<__gm__ void *>(pc[1]);
    uint64_t data = pc[2];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 52) & 0xfful;
    op.iter_size = (data >> 34) & 0x3fffful;
    op.tail_iter = (data >> 16) & 0x3fffful;
    op.body_iter = data & 0xfffful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vLoad::ROUND_OFFSET + round_size;
    pc[0] = vMakeAccHead(id, op.tile_stride << 13 | vCompactX(op.xn), size);
    pc[1] = reinterpret_cast<uint64_t>(op.from);
    pc[2] = op.round_rank << 60 | op.pad_size << 52 | op.iter_size << 34 | op.tail_iter << 16 | op.body_iter;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vLoad::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vViewLoad {
  enum { RELOC_OFFSET = 3 };
  enum { VAR_OFFSET = 4 };
  uint64_t xd;
  uint64_t from;
  uint64_t offset;
  uint64_t iter_size;
  uint64_t iter_num;
  uint64_t src_gap;
  uint64_t dst_gap;
  uint64_t tail_size;
  uint64_t loop_depth;
  uint64_t tile_depth;
  // pc[0]: reserved(16) << 18 | xd(18)
  // pc[1]: dst_gap(4) << 60 | loop_depth(4) << 56 | tile_depth(4) << 52 | tail_size(18) << 34 | iter_size(18) << 16 |
  //        iter_num(16)
  // pc[2]: src_gap(32) << 32 | offset(32)
  // pc[3]: from(64)
  // pc[VAR::loop_depth]: loop_size(16) << 48 | dst_stride(16) << 32 | src_stride(32)
  // pc[VAR+loop_depth::tile_depth]: tile_space(32) << 32 | tile_stride(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vViewLoad &op) {
    op.xd = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data1 = pc[1];
    op.iter_num = data1 & 0xfffful;
    op.iter_size = vGetBitRange(data1, 16, 18);
    op.tail_size = vGetBitRange(data1, 34, 18);
    op.tile_depth = vGetBitRange(data1, 52, 4);
    op.loop_depth = vGetBitRange(data1, 56, 4);
    op.dst_gap = data1 >> 60;
    uint64_t data2 = pc[2];
    op.src_gap = data2 >> 32;
    op.offset = vGetBitRange(data2, 0, 32);
    op.from = pc[3];
  }
  __aicore_inline__ void DecodeLoop(uint64_t data, uint64_t &loop_size, uint64_t &dst_stride, uint64_t &src_stride) {
    loop_size = data >> 48;
    dst_stride = (data >> 32) & 0xfffful;
    src_stride = vGetBitRange(data, 0, 32);
  }
  __aicore_inline__ void DecodeTile(uint64_t data, uint64_t &tile_space, uint64_t &tile_stride) {
    tile_space = data >> 32;
    tile_stride = vGetBitRange(data, 0, 32);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vViewLoad &op) {
    uint64_t size = VAR_OFFSET + op.loop_depth + op.tile_depth;
    pc[0] = vMakeAccHead(id, op.xd, size);
    pc[1] = op.dst_gap << 60 | op.loop_depth << 56 | op.tile_depth << 52 | op.tail_size << 34 | op.iter_size << 16 |
            op.iter_num;
    pc[2] = op.src_gap << 32 | op.offset;
    pc[3] = op.from;
    return size;
  }
  __aicore_inline__ uint64_t EncodeLoop(uint64_t loop_size, uint64_t dst_stride, uint64_t src_stride) {
    return loop_size << 48 | dst_stride << 32 | src_stride;
  }
  __aicore_inline__ uint64_t EncodeTile(uint64_t space, uint64_t stride) { return space << 32 | stride; }
};

struct vViewLoadX {
  enum { RELOC_OFFSET = 3 };
  enum { VAR_OFFSET = 4 };
  uint64_t xd;
  uint64_t from;
  uint64_t offset;
  uint64_t ws;
  uint64_t ws_size;
  uint64_t iter_size;
  uint64_t iter_stride;
  uint64_t tail_size;
  uint64_t loop_depth;
  uint64_t tile_depth;
  uint64_t type_size;
  // pc[0]: iter_stride(32)
  // pc[1]: tail_size(16) << 48 | iter_size(16) << 32 | offset(32)
  // pc[2]: ws_size(16) << 48 | type_size(4) << 44 | loop_depth(4) << 40 | tile_depth(4) << 36 | xd(18) << 18 | ws(18)
  // pc[3]: from(64)
  // pc[VAR::loop_depth]: loop_size(16) << 48 | dst_stride(16) << 32 | src_stride(32)
  // pc[VAR+loop_depth::tile_depth]: tile_space(32) << 32 | tile_stride(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vViewLoadX &op) {
    op.iter_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 32);
    uint64_t data1 = pc[1];
    op.offset = vGetBitRange(data1, 0, 32);
    op.iter_size = (data1 >> 32) & 0xfffful;
    op.tail_size = (data1 >> 48);
    uint64_t data2 = pc[2];
    op.ws = data2 & 0x3fffful;
    op.xd = (data2 >> 18) & 0x3fffful;
    op.tile_depth = (data2 >> 36) & 0xful;
    op.loop_depth = (data2 >> 40) & 0xful;
    op.type_size = (data2 >> 44) & 0xful;
    op.ws_size = data2 >> 48;
    op.from = pc[3];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vViewLoadX &op) {
    uint64_t size = VAR_OFFSET + op.loop_depth + op.tile_depth;
    pc[0] = vMakeAccHead(id, op.iter_stride, size);
    pc[1] = op.tail_size << 48 | op.iter_size << 32 | op.offset;
    pc[2] = op.ws_size << 48 | op.type_size << 44 | op.loop_depth << 40 | op.tile_depth << 36 | op.xd << 18 | op.ws;
    pc[3] = op.from;
    return size;
  }
};

// [loopn, ..., loop0, iter_size, h_factor, w_factor] =  load[loopn, ..., loop0, iter_size, w_factor(w_gap), h_factor]
struct vViewLoadT {
  enum { RELOC_OFFSET = 4 };
  enum { VAR_OFFSET = 5 };
  uint64_t xd;
  uint64_t from;
  uint64_t offset;
  uint64_t item_size;
  uint64_t w_fractal;
  uint64_t h_fractal;
  uint64_t w_gap;
  uint64_t iter_size;
  uint64_t iter_stride;
  uint64_t ws;
  uint64_t ws_size;
  uint64_t row_stride_blocks;  // UB row stride in 32B blocks.
  uint64_t is_width_iter;
  uint64_t tail_size;
  uint64_t loop_depth;
  uint64_t tile_depth;
  // pc[0]: w_gap(32)
  // pc[1]: iter_stride(32) << 32 | offset(32)
  // pc[2]: xd(18) << 46 | item_size(4) << 40 | loop_depth(4) << 36 | tile_depth(4) << 32 | tail_size(16) << 16 | h_fractal(8) << 8 | w_fractal(8)
  // pc[3]: ws(18) << 46 | is_width_iter(1) << 45 | row_stride_blocks(13) << 32 | ws_size(16) << 16 |
  //        iter_size(16)
  // pc[4]: from(64)
  // pc[VAR::loop_depth]: loop_size(16) << 48 | dst_stride(16) << 32 | src_stride(32)
  // pc[VAR+loop_depth::tile_depth]: tile_space(32) << 32 | tile_stride(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vViewLoadT &op) {
    op.w_gap = vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 32);
    uint64_t data1 = pc[1];
    op.offset = vGetBitRange(data1, 0, 32);
    op.iter_stride = data1 >> 32;
    uint64_t data2 = pc[2];
    op.w_fractal = data2 & 0xfful;
    op.h_fractal = (data2 >> 8) & 0xfful;
    op.tail_size = (data2 >> 16) & 0xfffful;
    op.tile_depth = (data2 >> 32) & 0xful;
    op.loop_depth = (data2 >> 36) & 0xful;
    op.item_size = (data2 >> 40) & 0xful;
    op.xd = data2 >> 46;
    uint64_t data3 = pc[3];
    op.iter_size = data3 & 0xfffful;
    op.ws_size = (data3 >> 16) & 0xfffful;
    op.row_stride_blocks = (data3 >> 32) & 0x1ffful;
    op.is_width_iter = (data3 >> 45) & 0x1ul;
    op.ws = data3 >> 46;
    op.from = pc[4];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vViewLoadT &op) {
    uint64_t size = VAR_OFFSET + op.loop_depth + op.tile_depth;
    pc[0] = vMakeAccHead(id, op.w_gap, size);
    pc[1] = op.iter_stride << 32 | op.offset;
    pc[2] = op.xd << 46 | op.item_size << 40 | op.loop_depth << 36 | op.tile_depth << 32 | op.tail_size << 16 | op.h_fractal << 8 | op.w_fractal;
    pc[3] = op.ws << 46 | op.is_width_iter << 45 | op.row_stride_blocks << 32 | op.ws_size << 16 |
            op.iter_size;
    pc[4] = op.from;
    return size;
  }
};

struct vViewStore {
  enum { RELOC_OFFSET = 3 };
  enum { VAR_OFFSET = 4 };
  uint64_t xn;
  uint64_t to;
  uint64_t offset;
  uint64_t iter_size;
  uint64_t iter_num;
  uint64_t dst_gap;
  uint64_t src_gap;
  uint64_t tail_size;
  uint64_t loop_depth;
  uint64_t tile_depth;
  // pc[0]: reserved(16) << 18 | xn(18)
  // pc[1]: src_gap(4) << 60 | loop_depth(4) << 56 | tile_depth(4) << 52 | tail_size(18) << 34 | iter_size(18) << 16 |
  //        iter_num(16)
  // pc[2]: dst_gap(32) << 32 | offset(32)
  // pc[3]: to(64)
  // pc[VAR::loop_depth]: loop_size(16) << 48 | src_stride(16) << 32 | dst_stride(32)
  // pc[VAR+loop_depth::tile_depth]: tile_space(32) << 32 | tile_stride(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vViewStore &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data1 = pc[1];
    op.iter_num = data1 & 0xfffful;
    op.iter_size = vGetBitRange(data1, 16, 18);
    op.tail_size = vGetBitRange(data1, 34, 18);
    op.tile_depth = vGetBitRange(data1, 52, 4);
    op.loop_depth = vGetBitRange(data1, 56, 4);
    op.src_gap = data1 >> 60;
    uint64_t data2 = pc[2];
    op.dst_gap = data2 >> 32;
    op.offset = vGetBitRange(data2, 0, 32);
    op.to = pc[3];
  }
  __aicore_inline__ void DecodeLoop(uint64_t data, uint64_t &loop_size, uint64_t &dst_stride, uint64_t &src_stride) {
    loop_size = data >> 48;
    dst_stride = vGetBitRange(data, 0, 32);
    src_stride = (data >> 32) & 0xfffful;
  }
  __aicore_inline__ void DecodeTile(uint64_t data, uint64_t &tile_space, uint64_t &tile_stride) {
    tile_space = data >> 32;
    tile_stride = vGetBitRange(data, 0, 32);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vViewStore &op) {
    uint64_t size = VAR_OFFSET + op.loop_depth + op.tile_depth;
    pc[0] = vMakeAccHead(id, op.xn, size);
    pc[1] = op.src_gap << 60 | op.loop_depth << 56 | op.tile_depth << 52 | op.tail_size << 34 | op.iter_size << 16 |
            op.iter_num;
    pc[2] = op.dst_gap << 32 | op.offset;
    pc[3] = op.to;
    return size;
  }
  __aicore_inline__ uint64_t EncodeLoop(uint64_t loop_size, uint64_t dst_stride, uint64_t src_stride) {
    return loop_size << 48 | src_stride << 32 | dst_stride;
  }
  __aicore_inline__ uint64_t EncodeTile(uint64_t space, uint64_t stride) { return space << 32 | stride; }
};

struct vViewStoreX {
  enum { RELOC_OFFSET = 3 };
  enum { VAR_OFFSET = 4 };
  uint64_t xn;
  uint64_t to;
  uint64_t offset;
  uint64_t ws;
  uint64_t ws_size;
  uint64_t iter_size;
  uint64_t iter_stride;
  uint64_t tail_size;
  uint64_t loop_depth;
  uint64_t tile_depth;
  uint64_t type_size;
  // pc[0]: iter_stride(32)
  // pc[1]: tail_size(16) << 48 | iter_size(16) << 32 | offset(32)
  // pc[2]: ws_size(16) << 48 | type_size(4) << 44 | loop_depth(4) << 40 | tile_depth(4) << 36 | xn(18) << 18 | ws(18)
  // pc[3]: to(64)
  // pc[VAR::loop_depth]: loop_size(16) << 48 | src_stride(16) << 32 | dst_stride(32)  [vViewStore format]
  // pc[VAR+loop_depth::tile_depth]: tile_space(32) << 32 | tile_stride(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vViewStoreX &op) {
    op.iter_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 32);
    uint64_t data1 = pc[1];
    op.offset = vGetBitRange(data1, 0, 32);
    op.iter_size = (data1 >> 32) & 0xfffful;
    op.tail_size = (data1 >> 48);
    uint64_t data2 = pc[2];
    op.ws = data2 & 0x3fffful;
    op.xn = (data2 >> 18) & 0x3fffful;
    op.tile_depth = (data2 >> 36) & 0xful;
    op.loop_depth = (data2 >> 40) & 0xful;
    op.type_size = (data2 >> 44) & 0xful;
    op.ws_size = data2 >> 48;
    op.to = pc[3];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vViewStoreX &op) {
    uint64_t size = VAR_OFFSET + op.loop_depth + op.tile_depth;
    pc[0] = vMakeAccHead(id, op.iter_stride, size);
    pc[1] = op.tail_size << 48 | op.iter_size << 32 | op.offset;
    pc[2] = op.ws_size << 48 | op.type_size << 44 | op.loop_depth << 40 | op.tile_depth << 36 | op.xn << 18 | op.ws;
    pc[3] = op.to;
    return size;
  }
};

struct vGatherLoad {
  enum { ROUND_OFFSET = 5 };
  enum { RELOC_OFFSET = 1 };
  enum { INDEX_RELOC_OFFSET = 2 };
  enum GatherMode { kElementGather = 0, kSliceGather = 1 };
  uint64_t xn;
  uint64_t from;
  uint64_t index;
  uint64_t inner_size;
  uint64_t gather_size;
  uint64_t gather_dim_size;
  uint64_t gather_mode;
  uint64_t body_iter;
  uint64_t tail_iter;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  // pc[0]: c_xn(13)
  // pc[1]: from
  // pc[2]: index
  // pc[3]: round_rank(4) << 60 | pad_size(8) << 48 | iter_size(16) << 32 | tail_iter(16) << 16 | body_iter(16)
  // pc[4]: gather_mode(4) << 48 | gather_dim_size(16) << 32 | gather_size(16) << 16 | inner_size(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vGatherLoad &op) {
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
    op.from = pc[1];
    op.index = pc[2];
    uint64_t data = pc[3];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 48) & 0xfful;
    op.iter_size = (data >> 32) & 0xfffful;
    op.tail_iter = (data >> 16) & 0xfffful;
    op.body_iter = data & 0xfffful;
    data = pc[4];
    op.inner_size = data & 0xfffful;
    op.gather_size = (data >> 16) & 0xfffful;
    op.gather_dim_size = (data >> 32) & 0xfffful;
    op.gather_mode = (data >> 48) & 0xful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vGatherLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vGatherLoad::ROUND_OFFSET + round_size;
    uint64_t ext = vCompactX(op.xn);
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.from;
    pc[2] = op.index;
    pc[3] = op.round_rank << 60 | op.pad_size << 48 | op.iter_size << 32 | op.tail_iter << 16 | op.body_iter;
    pc[4] = op.gather_mode << 48 | op.gather_dim_size << 32 | op.gather_size << 16 | op.inner_size;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vGatherLoad::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vMultiLoad {
  enum { ROUND_OFFSET = 7 };
  enum { RELOC_OFFSET = 1 };
  enum { UNIQUEID_OFFSET = 6 };
  __gm__ void *from;
  uint64_t xn;
  uint64_t multi_size{1};
  uint64_t xbuf_size;
  uint64_t gap{0};
  uint64_t tile_stride;
  uint64_t body_iter;
  uint64_t tail_iter;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  __gm__ uint8_t *peer_mem;
  __gm__ uint8_t *flag_mem;
  uint64_t rank_id;
  uint64_t pingpong{0};
  uint64_t unique_id;
  uint64_t tile_stride2;  // for PeerStore
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: from
  // pc[2]: round_rank(4) << 60 | pad_size(8) << 50 | iter_size(18) << 32 | tail_iter(16) << 16 | body_iter(16)
  // pc[3]: gap(38) << 26 | xbuf_size(16) << 10 | multi_size(10)
  // pc[4]: peer_mem
  // pc[5]: flag_mem
  // pc[6]: tile_stride2(18) << 38 | rank_id(4) << 34 | pingpong(2) << 32 | unique_id(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vMultiLoad &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + 13, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13));
    op.from = reinterpret_cast<__gm__ void *>(pc[1]);
    uint64_t data = pc[2];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 50) & 0xfful;
    op.iter_size = (data >> 32) & 0x3fffful;
    op.tail_iter = (data >> 16) & 0xfffful;
    op.body_iter = data & 0xfffful;
    op.multi_size = pc[3] & 0x3fful;
    op.xbuf_size = (pc[3] >> 10) & 0xfffful;
    op.gap = pc[3] >> 26;
    op.peer_mem = reinterpret_cast<__gm__ uint8_t *>(pc[4]);
    op.flag_mem = reinterpret_cast<__gm__ uint8_t *>(pc[5]);
    data = pc[6];
    op.rank_id = (data >> 34) & 0xful;
    op.pingpong = (data >> 32) & 0x3ul;
    op.unique_id = data & 0xfffffffful;
    op.tile_stride2 = data >> 38;
  }
  __aicore_inline__ void PingPongSwitch(bcodeptr_t pc) {
    auto &data = pc[6];
    data ^= ((data >> 32) & 1ul) << 33;
    data ^= 1ul << 32;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vMultiLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vMultiLoad::ROUND_OFFSET + round_size;
    pc[0] = vMakeAccHead(id, op.tile_stride << 13 | vCompactX(op.xn), size);
    pc[1] = reinterpret_cast<uint64_t>(op.from);
    pc[2] = op.round_rank << 60 | op.pad_size << 50 | op.iter_size << 32 | op.tail_iter << 16 | op.body_iter;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vLoad::ROUND_OFFSET + i] = rounds[i];
    }
    pc[3] = op.gap << 26 | op.xbuf_size << 10 | op.multi_size;
    pc[4] = reinterpret_cast<uint64_t>(op.peer_mem);
    pc[5] = reinterpret_cast<uint64_t>(op.flag_mem);
    pc[6] = op.tile_stride2 << 38 | op.rank_id << 34 | 0x1ul;
    return size;
  }
};

struct vPingPongLoad {
  enum { ROUND_OFFSET = 4 };
  enum { RELOC_OFFSET = 1 };
  __gm__ void *from;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t body_iter;  // nburst
  uint64_t tail_iter;
  uint64_t iter_size;  // lenburst
  uint64_t pad_size;
  uint64_t round_rank;
  uint64_t pingpong;
  uint64_t pingpong_stride;
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: from
  // pc[2]: round_rank(4) << 60 | pad_size(8) << 50 | iter_size(18) << 32 | tail_iter(16) << 16 | body_iter(16)
  // pc[3]: pingpong_stride(32) << 32 | pingpong(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vPingPongLoad &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + 13, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13));
    op.from = reinterpret_cast<__gm__ void *>(pc[1]);
    uint64_t data = pc[2];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 50) & 0xfful;
    op.iter_size = (data >> 32) & 0x3fffful;
    op.tail_iter = (data >> 16) & 0xfffful;
    op.body_iter = data & 0xfffful;
    data = pc[3];
    op.pingpong_stride = data >> 32;
    op.pingpong = data & 0xfffful;
  }
  __aicore_inline__ void PingPongSwitch(bcodeptr_t pc) { pc[3] ^= 0x1ul; }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vPingPongLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vPingPongLoad::ROUND_OFFSET + round_size;
    pc[0] = vMakeAccHead(id, op.tile_stride << 13 | vCompactX(op.xn), size);
    pc[1] = reinterpret_cast<uint64_t>(op.from);
    pc[2] = op.round_rank << 60 | op.pad_size << 50 | op.iter_size << 32 | op.tail_iter << 16 | op.body_iter;
    pc[3] = op.pingpong_stride << 32 | op.pingpong;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vPingPongLoad::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vPingPongPeerLoad {
  enum { ROUND_OFFSET = 6 };
  enum { UNIQUEID_OFFSET = 5 };
  enum { SHARD_OFFSET = 4 };
  vPingPongLoad base;
  uint64_t peer_mem_offset;
  uint64_t unique_id;
  uint64_t event_id{0};
  bool set_flag{false};
  bool wait_flag{false};
  uint64_t shard_rel;
  __bcode__ int32_t *__restrict__ step_addr;
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: from
  // pc[2]: round_rank(4) << 60 | pad_size(8) << 50 | iter_size(18) << 32 | tail_iter(16) << 16 | body_iter(16)
  // pc[3]: pingpong_stride(32) << 32 | pingpong(16)
  // pc[4]: peer_mem_offset(32) << 32 | wait_flag(1) << 16 | set_flag(1) << 15 | event_id(3) << 12 | shard_rel(12)
  // pc[5]: step(32) << 32 | unique_id(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vPingPongPeerLoad &op) {
    vPingPongLoad::Decode(pc, head, op.base);
    uint64_t data = pc[4];
    op.peer_mem_offset = data >> 32;
    op.event_id = (data >> 12) & 0x7ul;
    op.set_flag = (data >> 15) & 0x1ul;
    op.wait_flag = (data >> 16) & 0x1ul;
    op.shard_rel = data & 0xffful;
    op.step_addr = reinterpret_cast<__bcode__ int32_t *>(pc) + 11;
    op.unique_id = pc[5] & 0xfffffffful;
  }
  __aicore_inline__ void PingPongSwitch(bcodeptr_t pc) { pc[3] ^= 0x1ul; }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vPingPongPeerLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.base.round_rank + 1) / 2;
    uint64_t size = vPingPongPeerLoad::ROUND_OFFSET + round_size;  // Do we need round?
    pc[0] = vMakeAccHead(id, op.base.tile_stride << 13 | vCompactX(op.base.xn), size);
    pc[1] = reinterpret_cast<uint64_t>(op.base.from);
    pc[2] = op.base.round_rank << 60 | op.base.pad_size << 50 | op.base.iter_size << 32 | op.base.tail_iter << 16 |
            op.base.body_iter;
    pc[3] = op.base.pingpong_stride << 32 | (op.base.pingpong & 0xfffful);
    pc[4] = (op.peer_mem_offset & 0xfffffffful) << 32 | uint64_t(op.wait_flag) << 16 | uint64_t(op.set_flag) << 15 |
            op.event_id << 12;
    pc[5] = 0x1ul << 32 | 0x1ul;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vPingPongPeerLoad::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

// complete lead_dim: [iter_num/iter_tail, iter_size+pad_size]
// tiling lead_dim:   [iter_size/iter_tail+pad_size]
struct vStore {
  enum { RELOC_OFFSET = 2 };
  enum { ROUND_OFFSET = 3 };
  uint64_t xn;
  uint64_t to;
  uint64_t tile_stride;
  uint64_t iter_num;
  uint64_t iter_tail;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: round_rank(4) << 60 | pad_size(8) << 52 | iter_size(18) << 34 | iter_tail(18) << 16 | iter_num(16)
  // pc[2]: to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStore &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.iter_num = data & 0xfffful;
    op.iter_tail = (data >> 16) & 0x3fffful;
    op.iter_size = (data >> 34) & 0x3fffful;
    op.pad_size = (data >> 52) & 0xfful;
    op.round_rank = data >> 60;
    op.to = pc[2];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStore &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vStore::ROUND_OFFSET + round_size;
    uint64_t ext = op.tile_stride << 13 | vCompactX(op.xn);
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.round_rank << 60 | op.pad_size << 52 | op.iter_size << 34 | op.iter_tail << 16 | op.iter_num;
    pc[2] = op.to;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vStore::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vStoreCond {
  enum { RELOC_OFFSET = 2 };
  enum { ROUND_OFFSET = 3 };
  uint64_t xn;
  uint64_t to;
  uint64_t tile_stride;
  uint64_t iter_offset;
  uint64_t iter_range;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  uint64_t cond_offset;
  uint64_t dtype_shift;
  // pc[0]: tile_stride(18) << 13 | cond_offset(13)
  // pc[1]: round_rank(4) << 60 | dtype_shift(8) << 52 | iter_size(18) << 34 | xn(18) << 16 | pad_size(16)
  // pc[2]: to
  template <bool check>
  __aicore_inline__ bool Decode(bcodeptr_t pc, uint64_t head, vStoreCond &op) {
    bcodeptr_t cond = pc - vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13);
    uint64_t cond_data = *cond;
    op.iter_range = cond_data & 0x3fffful;
    if (check) {
      if (op.iter_range == 0) return false;
      *cond = 0;
    }
    op.iter_offset = cond_data >> 32;
    op.cond_offset = vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13);
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS, 18);
    uint64_t data = pc[1];
    op.pad_size = data & 0xfful;
    op.xn = (data >> 16) & 0x3fffful;
    op.iter_size = (data >> 34) & 0x3fffful;
    op.dtype_shift = (data >> 52) & 0xfful;
    op.round_rank = data >> 60;
    op.to = pc[2];
    return true;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStoreCond &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vStore::ROUND_OFFSET + round_size;
    uint64_t ext = op.tile_stride << 13 | op.cond_offset;
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.round_rank << 60 | op.dtype_shift << 52 | op.iter_size << 34 | op.xn << 16 | op.pad_size;
    pc[2] = op.to;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vStore::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
  __aicore_inline__ void EncodeCond(bcodeptr_t cond, uint64_t iter_offset, uint64_t iter_range) {
    cond[0] = iter_offset << 32 | iter_range;
  }
};

// [iter_num/iter_tail, iter_size+pad_size]
struct vStoreAtomic {
  enum { RELOC_OFFSET = 2 };
  enum { ROUND_OFFSET = 3 };
  uint64_t to;
  uint64_t xn;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t iter_num;
  uint64_t iter_tail;
  uint64_t tile_stride;
  uint64_t round_rank;
  uint64_t cum_flag;
  uint64_t red_op;
  uint64_t atomic_type;
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: round_rank(4) << 60 | cum_flag(2) << 58 | pad_size(8) << 50 | iter_size(18) << 32 | iter_tail(16) << 16 |
  // atomic_type(2) << 14| red_op(2) << 12| iter_num(12)
  // pc[2]: to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreAtomic &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.round_rank = data >> 60;
    op.cum_flag = (data >> 58) & 0x3ul;
    op.pad_size = (data >> 50) & 0xfful;
    op.iter_size = (data >> 32) & 0x3fffful;
    op.iter_tail = (data >> 16) & 0xfffful;
    op.atomic_type = (data >> 14) & 0x3ul;
    op.red_op = (data >> 12) & 0x3ul;
    op.iter_num = data & 0xffful;
    op.to = pc[2];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStoreAtomic &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vStoreAtomic::ROUND_OFFSET + round_size;
    uint64_t ext = op.tile_stride << 13 | vCompactX(op.xn);
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.round_rank << 60 | op.cum_flag << 58 | op.pad_size << 50 | op.iter_size << 32 | op.iter_tail << 16 |
            op.atomic_type << 14 | op.red_op << 12 | op.iter_num;
    pc[2] = op.to;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vStoreAtomic::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vStoreAtomicW {
  enum { ROUND_OFFSET = 1 };
  uint64_t round_rank;
  uint64_t red_op;
  uint64_t atomic_type;
  // pc[0]: atomic_type(2) << 4 | red_op(2) << 2 | round_rank(2)
  // pc[1 ~ (round_rank+1)/2]: round
  // pc[(round_rank+1)/2]: body
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreAtomicW &op) {
    op.round_rank = (head >> V_M_HEAD_EXT_OFFSET) & 0x3ul;
    op.red_op = (head >> (V_M_HEAD_EXT_OFFSET + 2)) & 0x3ul;
    op.atomic_type = (head >> (V_M_HEAD_EXT_OFFSET + 4)) & 0x3ul;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStoreAtomicW &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t ext = op.atomic_type << 4 | op.red_op << 2 | op.round_rank;
    pc[0] = vMakeAccHead(id, ext, 0);
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vStoreAtomicW::ROUND_OFFSET + i] = rounds[i];
    }
    return vStoreAtomicW::ROUND_OFFSET + round_size;
  }
  __aicore_inline__ void EncodeSize(bcodeptr_t pc, uint64_t size) { pc[0] |= size << V_M_HEAD_SIZE_OFFSET; }
  __aicore_inline__ bcodeptr_t GetBody(bcodeptr_t pc, const vStoreAtomicW &op) {
    return pc + (vStoreAtomicW::ROUND_OFFSET + (op.round_rank + 1) / 2);
  }
};

// [iter_num/iter_tail, iter_size+pad_size]
struct vStoreRS {
  enum { RELOC_OFFSET = 2 };
  enum { ROUND_OFFSET = 3 };
  uint64_t to;
  uint64_t xn;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t iter_num;
  uint64_t iter_tail;
  uint64_t tile_stride;
  uint64_t round_rank;
  uint64_t rank_id;
  // pc[0]: rank_id(4) << 29 | iter_tail(16) << 13 | c_xn(13)
  // pc[1]: round_rank(4) << 60 | pad_size(8) << 52 | iter_size(18) << 34 | tile_stride(18) << 16 |
  //        iter_num(16)
  // pc[2]: to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreRS &op) {
    op.iter_tail = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS, 16);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
    op.rank_id = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS + 16, 4);
    uint64_t data = pc[1];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 52) & 0xfful;
    op.iter_size = (data >> 34) & 0x3fffful;
    op.tile_stride = (data >> 16) & 0x3fffful;
    op.iter_num = data & 0xfffful;
    op.to = pc[2];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStoreRS &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vStoreAtomic::ROUND_OFFSET + round_size;
    uint64_t ext = op.rank_id << 29 | op.iter_tail << 13 | vCompactX(op.xn);
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.round_rank << 60 | op.pad_size << 52 | op.iter_size << 34 | op.tile_stride << 16 | op.iter_num;
    pc[2] = op.to;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vStoreAtomic::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vStoreAG {
  enum { RELOC_OFFSET = 1 };
  uint64_t to;
  uint64_t xn;  // the first ub addr
  uint64_t xbuf_size;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t iter_num;
  uint64_t iter_tail;
  uint64_t tile_stride;
  uint64_t rank_id;
  uint64_t rank_size;
  uint64_t shard_stride;  // all the data of one rank is a shard
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreAG &op) {
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
    op.rank_size = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS + 11, 5);
    op.rank_id = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS + 16, 5);
    op.to = pc[1];
    op.pad_size = (pc[2] >> 52) & 0xfful;
    op.iter_size = (pc[2] >> 34) & 0x3fffful;
    op.tile_stride = (pc[2] >> 16) & 0x3fffful;
    op.iter_num = pc[2] & 0xfffful;
    op.iter_tail = pc[3] >> 32;
    op.xbuf_size = pc[3] & 0xfffffffful;
    op.shard_stride = pc[4];
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vStoreAG &op) {
    uint64_t size = 5;
    uint64_t ext = op.rank_id << 29 | op.rank_size << 24 | vCompactX(op.xn);
    pc[0] = vMakeAccHead(id, ext, size);
    pc[1] = op.to;
    pc[2] = op.pad_size << 52 | op.iter_size << 34 | op.tile_stride << 16 | op.iter_num;
    pc[3] = op.iter_tail << 32 | op.xbuf_size;
    pc[4] = op.shard_stride;
    return size;
  }
};

#define V_INSN_SIZE_MAX (4 * sizeof(uint64_t))

#define V_CUBE_FLAG_TRANS_A (1ul << 4)
#define V_CUBE_FLAG_TRANS_B (1ul << 5)
#define V_CUBE_FLAG_GROUP_SET (1ul << 6)
#define V_CUBE_FLAG_PINGPONG_STORE (1ul << 7)
#define V_CUBE_FLAG_OUT_FP32 (1ul << 8)
#define V_CUBE_FLAG_ATOMIC_ADD (1ul << 9)
#define V_CUBE_FLAG_WITH_BIAS (1ul << 10)
#define V_CUBE_FLAG_BIAS_FP32 (1ul << 11)
#define V_CUBE_FLAG_PEER_STORE (1ul << 12)
#define V_CUBE_FLAG_GROUPED_LIST (1ul << 13)
#define V_CUBE_FLAG_GROUP_K (1ul << 14)
#define V_CUBE_FLAG_DIS_UNITFLAG (1ul << 16)
#define V_CUBE_FLAG_STORE_UB (1ul << 17)
#define V_CUBE_FLAG_GMM_SIZE_MODE (1ul << 18)
#define V_CUBE_FLAG_PIPELINE (1ul << 19)
#define V_CUBE_FLAG_INPUT_A_NZ (1ul << 20)
#define V_CUBE_FLAG_INPUT_B_NZ (1ul << 21)
#define V_CUBE_FLAG_DTYPE_OFFSET 30  // [30, 31]

#define V_CUBE_SWIZ_VISIT_nZ 0
#define V_CUBE_SWIZ_VISIT_zN 1
#define V_CUBE_SWIZ_VISIT_DIAGONAL_Z 2
#define V_CUBE_SWIZ_VISIT_DIAGONAL_N 3
#define V_CUBE_DIAGONAL_MIN_DIM 6

#define V_CUBE_BCAST_FLAG_BCAST_A0 (1u << 0)
#define V_CUBE_BCAST_FLAG_BCAST_A1 (1u << 1)
#define V_CUBE_BCAST_FLAG_BCAST_B0 (1u << 2)
#define V_CUBE_BCAST_FLAG_BCAST_B1 (1u << 3)
#define V_CUBE_BCAST_C1_OFFSET 8
#define V_CUBE_BCAST_C1_BITS 24

struct vCubeOp {
  enum { FP16, BF16 };
  union {
    uint64_t gm_c;
    uint64_t ub_c;
  };
  uint64_t group_num;
  uint64_t gm_a;
  uint64_t gm_b;
  uint64_t gm_bias;
  uint64_t gm_group_list;

  uint32_t flags;
  uint32_t m_real, n_real, k_real;
  uint32_t m_align, n_align, k_align;
  uint32_t ka_align, kb_align;
  uint32_t m_loop, n_loop, k_loop;
  // shape_a: [batch_a0, batch_a1, m, k], shape_b: [batch_b0, batch_b1, k, n]
  uint32_t batch_cast;
  uint32_t m0, n0, k0;
  // format: visit_type(16) << 16 | data(16)
  // data:
  //  nZ: swizzle_cnt(16)
  //  zN: swizzle_cnt(16)
  uint32_t swizzle;
  uint32_t group_list_size{0};
  uint64_t offset_a, offset_b;
  // for aiv
  uint64_t gm_pos;
  uint32_t unique_id;
  uint32_t rank_size;

  __aicore_inline__ uint64_t UbInfoEncode(uint64_t xbuf, uint64_t subtile0, uint64_t subtile1) {
    return xbuf << 32 | subtile0 << 16 | subtile1;
  }

  __aicore_inline__ void UbInfoDecode(uint64_t data, uint64_t &xbuf, uint64_t &subtile0, uint64_t &subtile1) {
    subtile1 = data & 0xfffful;
    subtile0 = (data >> 16) & 0xfffful;
    xbuf = data >> 32;
  }

  __aicore_inline__ uint32_t SwizzleEncode(uint32_t visit_type, uint32_t data) { return visit_type << 16 | data; }

  __aicore_inline__ void SwizzleDiagonalZ(uint64_t tile_idx, uint64_t swiz_cnt, uint64_t m_loop, uint64_t n_loop,
                                          uint64_t &midx, uint64_t &nidx) {
    uint64_t m_block_idx = tile_idx / (swiz_cnt * n_loop);
    tile_idx -= m_block_idx * (swiz_cnt * n_loop);
    uint64_t m_row = min(swiz_cnt, m_loop - m_block_idx * swiz_cnt);

    uint64_t mn_block_idx = tile_idx / (m_row * swiz_cnt);
    tile_idx -= mn_block_idx * (m_row * swiz_cnt);
    uint64_t n_col = min(swiz_cnt, n_loop - mn_block_idx * swiz_cnt);

    uint64_t diagonal_size = m_row * n_col;
    uint64_t in_tile_block_idx = tile_idx % diagonal_size;

    midx = in_tile_block_idx % m_row + m_block_idx * swiz_cnt;
    nidx = (in_tile_block_idx + in_tile_block_idx / Lcm(m_row, n_col)) % n_col + mn_block_idx * swiz_cnt;
  }

  __aicore_inline__ void SwizzlezN(uint64_t tile_idx, uint64_t swiz_cnt, uint64_t m_loop, uint64_t n_loop,
                                   uint64_t &midx, uint64_t &nidx) {
    uint64_t tile_block_idx = tile_idx / (swiz_cnt * m_loop);
    uint64_t in_tile_block_idx = tile_idx - tile_block_idx * (swiz_cnt * m_loop);
    uint64_t n_col = min(swiz_cnt, n_loop - swiz_cnt * tile_block_idx);
    midx = in_tile_block_idx / n_col;
    nidx = tile_block_idx * swiz_cnt + in_tile_block_idx - n_col * midx;
    if (tile_block_idx & 1) {
      midx = m_loop - midx - 1;
    }
  }

  __aicore_inline__ void IndexCompute(uint64_t tile_idx, uint64_t swizzle, uint64_t m_loop, uint64_t n_loop,
                                      uint64_t &midx, uint64_t &nidx, uint64_t &cidx) {
    uint64_t core_loop = m_loop * n_loop;
    uint64_t swiz_cnt = swizzle & 0xffff;
    uint64_t swiz_type = swizzle >> 16;
    cidx = tile_idx / core_loop;
    tile_idx -= cidx * core_loop;
    if ((swiz_type == V_CUBE_SWIZ_VISIT_DIAGONAL_Z || V_CUBE_SWIZ_VISIT_DIAGONAL_N) &&
        (m_loop < V_CUBE_DIAGONAL_MIN_DIM || n_loop < V_CUBE_DIAGONAL_MIN_DIM)) {
      swiz_type = m_loop < n_loop ? V_CUBE_SWIZ_VISIT_zN : V_CUBE_SWIZ_VISIT_nZ;
    }
    switch (swiz_type) {
      case V_CUBE_SWIZ_VISIT_nZ:
        vCubeOp::SwizzlezN(tile_idx, swiz_cnt, n_loop, m_loop, nidx, midx);
        break;
      case V_CUBE_SWIZ_VISIT_zN:
        vCubeOp::SwizzlezN(tile_idx, swiz_cnt, m_loop, n_loop, midx, nidx);
        break;
      case V_CUBE_SWIZ_VISIT_DIAGONAL_Z:
        vCubeOp::SwizzleDiagonalZ(tile_idx, swiz_cnt, m_loop, n_loop, midx, nidx);
        break;
      case V_CUBE_SWIZ_VISIT_DIAGONAL_N:
        vCubeOp::SwizzleDiagonalZ(tile_idx, swiz_cnt, n_loop, m_loop, nidx, midx);
        break;
      default:
        break;
    }
  }
};

struct vPipeCubeOp {
  vCubeOp base;
  uint64_t step; // uid(16) << 48 | wait_prod_num(16) << 32 | wait_step(16) << 16 | set_step(16)
  uint64_t set_gm;
  uint64_t wait_gm;
  __aicore_inline__ uint64_t EncodeStep(uint64_t set_step, uint64_t wait_step, uint64_t wait_prod_num, uint64_t uid) {
    return uid << 48 | wait_prod_num << 32 | wait_step << 16 | set_step;
  }
  __aicore_inline__ void DecodeStep(uint64_t step, uint64_t &set_step, uint64_t &wait_step, uint64_t &wait_prod_num, uint64_t &uid) {
    set_step = step & 0xfffful;
    wait_step = (step >> 16) & 0xfffful;
    wait_prod_num = (step >> 32) & 0xfffful;
    uid = step >> 48;
  }
};

// peer memory <-> ub
struct vPeerDMA {
  enum { ROUND_OFFSET = 5 };
  enum { UNIQUEID_OFFSET = 4 };
  __gm__ uint8_t *peer_mem;
  __gm__ uint8_t *flag_mem;  // used to do softsync
  uint64_t rank_id{0};       // used to skip execute
  uint64_t comm_type{0};     // CommType
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t lenburst;
  uint64_t tail_lenburst;
  uint64_t round_rank{0};
  uint64_t unique_id;

  uint64_t pingpong;
  uint64_t event_id{0};
  bool set_flag{false};
  bool wait_flag{false};
  // pc[0]: round_rank(4) << 20 | xn(18)
  // pc[1]: peer_mem
  // pc[2]: tile_stride(32) << 32 | tail_lenburst(16) << 16 | lenburst(16)
  // pc[3]: flag_mem
  // pc[4]: comm_type(2) << 43 | rank_id(4) << 39 | pingpong(2) << 37 | wait_flag(1) << 36 | set_flag(1) << 35 |
  //        event_id(3) << 32 | unique_id(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vPeerDMA &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.round_rank = (head >> (V_M_HEAD_EXT_OFFSET + 20)) & 0xful;
    op.peer_mem = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    uint64_t data = pc[2];
    op.lenburst = data & 0xfffful;
    op.tail_lenburst = (data >> 16) & 0xfffful;
    op.tile_stride = data >> 32;
    op.flag_mem = reinterpret_cast<__gm__ uint8_t *>(pc[3]);
    op.unique_id = pc[4] & 0xfffffffful;
    op.event_id = (pc[4] >> 32) & 0x7ul;
    op.set_flag = (pc[4] >> 35) & 0x1ul;
    op.wait_flag = (pc[4] >> 36) & 0x1ul;
    op.pingpong = (pc[4] >> 37) & 0x3ul;
    op.rank_id = (pc[4] >> 39) & 0xful;
    op.comm_type = (pc[4] >> 43) & 0x3ul;
  }
  __aicore_inline__ void PingPongSwitch(bcodeptr_t pc) {
    auto &data = pc[4];
    data ^= ((data >> 37) & 1ul) << 38;
    data ^= 1ul << 37;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t id, const vPeerDMA &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vPeerDMA::ROUND_OFFSET + round_size;
    pc[0] = vMakeAccHead(id, op.round_rank << 20 | op.xn, size);
    pc[1] = reinterpret_cast<uint64_t>(op.peer_mem);
    pc[2] = op.tile_stride << 32 | op.tail_lenburst << 16 | op.lenburst;
    pc[3] = reinterpret_cast<uint64_t>(op.flag_mem);
    pc[4] = op.comm_type << 43 | op.rank_id << 39 | op.event_id << 32 | 0x1ul;
    if (op.set_flag) {
      pc[4] |= 0x1ul << 35;
    }
    if (op.wait_flag) {
      pc[4] |= 0x1ul << 36;
    }
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vPeerDMA::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vVisitReorder {
  enum { COORD_OFFSET = 3 };
  uint64_t loop_stride;
  uint64_t loop_num;
  uint64_t loop_tail;
  uint64_t parallel_stride;
  uint64_t parallel_extent;
  uint64_t coord_num;
  uint64_t wave_tidx;
  uint64_t wave_block;
  // pc[0]: wave_block(32) << 32 | wave_tidx(32)
  // pc[1]: loop_stride(32) << 32 | loop_tail(16) << 16 | loop_num(16)
  // pc[2]: coord_num(4) << 60 | parallel_extent(28) << 32 | parallel_stride(32)
  // pc[3:]: stride(32) << 32 | extent(16) << 16 | coord_value(16)
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, const vVisitReorder &op) {
    pc[0] = op.wave_tidx;
    pc[1] = op.loop_stride << 32 | op.loop_tail << 16 | op.loop_num;
    pc[2] = op.coord_num << 60 | op.parallel_extent << 32 | op.parallel_stride;
    return op.coord_num + COORD_OFFSET;
  }
  __aicore_inline__ void EncodeCoord(bcodeptr_t pc, int idx, uint64_t stride, uint64_t extent) {
    pc[idx + COORD_OFFSET] = stride << 32 | extent << 16;
  }
  __aicore_inline__ void Decode(bcodeptr_t pc, vVisitReorder &op) {
    uint64_t data = pc[0];
    op.wave_tidx = vGetBitRange(data, 0, 32);
    op.wave_block = data >> 32;
    data = pc[1];
    op.loop_stride = data >> 32;
    op.loop_tail = (data >> 16) & 0xfffful;
    op.loop_num = data & 0xfffful;
    data = pc[2];
    op.coord_num = data >> 60;
    op.parallel_extent = vGetBitRange(data, 32, 28);
    op.parallel_stride = vGetBitRange(data, 0, 32);
  }
  __aicore_inline__ void DecodeCoord(bcodeptr_t pc, int idx, uint64_t &stride, uint64_t &extent, uint64_t &value) {
    uint64_t data = pc[idx + COORD_OFFSET];
    stride = data >> 32;
    extent = (data >> 16) & 0xfffful;
    value = data & 0xfffful;
  }
  __aicore_inline__ void UpdateWave(bcodeptr_t pc, uint64_t wave_block, uint64_t wave_tidx) { pc[0] = wave_block << 32 | wave_tidx; }
  __aicore_inline__ void UpdateCoord(bcodeptr_t pc, int idx, uint16_t value) {
    *reinterpret_cast<__bcode__ uint16_t *>(pc + (idx + COORD_OFFSET)) = value;
  }
  __aicore_inline__ void ClearCoord(bcodeptr_t pc) { pc[2] = (pc[2] << 4) >> 4; }
};

struct vPipeMsg {
  uint64_t pipe_cnt;
  uint64_t magic_id;
  __aicore_inline__ uint64_t MagicID(uint64_t uid) { return 0xc25f90ae4761ul << 16 | uid; }
};

struct vVisitPipeSet {
  enum { RELOC_OFFSET = 1 };
  enum { NEST_VISIT_OFFSET = 3 };
  uint32_t idx;
  uint64_t visit_id;
  uint64_t step;
  uint64_t gm;
  uint64_t uid;
  // pc[0]: visit_id(16) << 48 | step(16) << 32 | idx(32)
  // pc[1]: gm
  // pc[2]: uid(16)
  // pc[3:]: visit code
  __aicore_inline__ void Decode(bcodeptr_t pc, vVisitPipeSet &op) {
    uint64_t data = pc[0];
    op.idx = vGetBitRange(data, 0, 32);
    op.visit_id = data >> 48;
    op.step = (data >> 32) & 0xfffful;
    op.gm = pc[1];
    op.uid = pc[2];
  }
  __aicore_inline__ void UpdateIdx(bcodeptr_t pc, uint32_t idx) {
    *reinterpret_cast<__bcode__ uint32_t *>(pc) = idx;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, const vVisitPipeSet &op) {
    pc[0] = op.visit_id << 48 | op.step << 32;
    pc[1] = op.gm;
    pc[2] = op.uid;
    return NEST_VISIT_OFFSET;
  }
};

struct vVisitPipeWait {
  enum { RELOC_OFFSET = 2 };
  enum { NEST_VISIT_OFFSET = 3 };
  uint32_t idx;
  uint64_t visit_id;
  uint64_t prod_num;
  uint64_t step;
  uint64_t gm;
  uint64_t uid;
  // pc[0]: visit_id(16) << 48 | step(16) << 32 | idx(32)
  // pc[1]: uid(16) << 16 | prod_num(16)
  // pc[2]: gm
  // pc[3:]: visit code
  __aicore_inline__ void Decode(bcodeptr_t pc, vVisitPipeWait &op) {
    uint64_t data = pc[0];
    op.idx = vGetBitRange(data, 0, 32);
    op.visit_id = data >> 48;
    op.step = (data >> 32) & 0xfffful;
    data = pc[1];
    op.prod_num = data & 0xfffful;
    op.uid = data >> 16;
    op.gm = pc[2];
  }
  __aicore_inline__ void UpdateIdx(bcodeptr_t pc, uint32_t idx) {
    *reinterpret_cast<__bcode__ uint32_t *>(pc) = idx;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, const vVisitPipeWait &op) {
    pc[0] = op.visit_id << 48 | op.step << 32;
    pc[1] = op.uid << 16 | op.prod_num;
    pc[2] = op.gm;
    return NEST_VISIT_OFFSET;
  }
};

// head(32b): inc(1) << 31 | thread_size(7) << 24 | repeat(24) << 8
// e(64b):    e_num << 32 | e_idx
struct vVisitRed1 {
  uint32_t head;
  uint32_t r1;
  uint64_t e;
  uint32_t user_cnt;
  uint32_t reserved;
};

struct vVisitRed2 {
  uint32_t head;
  uint32_t user_cnt;
  uint64_t e;
  uint64_t e1_r1;
};

struct vVisitRed3 {
  uint64_t tidx_head;
  uint64_t e;
  uint64_t e1_r2;
  uint32_t r1;
  uint32_t user_cnt;
};

struct vVisitRed4 {
  uint64_t tidx_head;
  uint64_t e;
  uint64_t e1_r1;
  uint64_t e2_r2;
  uint32_t user_cnt;
  uint32_t reserved;
};

#define V_MM_POS_M_OFFSET 0
#define V_MM_POS_M_BITS 22
#define V_MM_POS_N_OFFSET 22
#define V_MM_POS_N_BITS 22
#define V_MM_POS_C_OFFSET 44
#define V_MM_POS_C_BITS 20

struct vMixGroupMsg {
  uint64_t pos[2];
  uint64_t reserved[62];  //  align to 512 cache line
};

struct vShard2D {
  enum { CODE_SIZE = 4 };
  uint64_t slice_m, slice_n;
  uint64_t tail_m, tail_n;
  uint64_t stride_m, stride_n;
  uint64_t pos;         // device only
  uint64_t offset;      // device only
  bool last_m, last_n;  // device only
  // pc[0]: pos
  // pc[1]: offset(32) << 32 | last_m(1) << 1 | last_n(1)
  // pc[2]: slice_m(16) << 48 | slice_n(16) << 32 | tail_m(16) << 16 | tail_n(16)
  // pc[3]: stride_m(32) << 32 | stride_n(32)
  __aicore_inline__ void Decode(bcodeptr_t pc, vShard2D &op) {
    op.pos = pc[0];
    uint64_t data = pc[1];
    op.offset = data >> 32;
    op.last_m = (data >> 1) & 0x1ul;
    op.last_n = data & 0x1ul;
    uint64_t data1 = pc[2];
    op.slice_m = data1 >> 48;
    op.slice_n = (data1 >> 32) & 0xfffful;
    op.tail_m = (data1 >> 16) & 0xfffful;
    op.tail_n = data1 & 0xfffful;
    uint64_t data2 = pc[3];
    op.stride_m = data2 >> 32;
    op.stride_n = vGetBitRange(data2, 0, 32);
  }
  __aicore_inline__ uint64_t DecodeOffset(bcodeptr_t pc) { return pc[1] >> 32; }
  __aicore_inline__ void Update(bcodeptr_t pc, uint64_t pos, uint64_t offset, bool last_m, bool last_n) {
    pc[0] = pos;
    pc[1] = offset << 32 | uint64_t(last_m) << 1 | uint64_t(last_n);
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t slice[], uint64_t tail[], uint64_t stride[]) {
    pc[0] = 0;
    pc[1] = 0;
    pc[2] = slice[1] << 48 | slice[0] << 32 | tail[1] << 16 | tail[0];
    pc[3] = stride[1] << 32 | stride[0];
    return CODE_SIZE;
  }
};

struct vVisitMix {
  enum { CODE_SIZE = 6 };
  uint64_t subtile0;
  uint64_t subtile1;
  __gm__ vCubeOp *cube;
  uint64_t group_idx;
  uint64_t pingpong;
  // pc[0-3]: shard2d
  // pc[4]: suttile0(16) << 48 | suttile1(16) << 32 | group_idx(31) << 1 | pingpong(1)
  // pc[5]: cube
  __aicore_inline__ void Decode(bcodeptr_t pc, vVisitMix &op) {
    uint64_t data1 = pc[4];
    op.subtile0 = (data1 >> 48) & 0xfffful;
    op.subtile1 = (data1 >> 32) & 0xfffful;
    op.group_idx = vGetBitRange(data1, 1, 31);
    op.pingpong = data1 & 0x1ul;
    op.cube = reinterpret_cast<__gm__ vCubeOp *>(pc[5]);
  }
  __aicore_inline__ void Update(bcodeptr_t pc, uint64_t group_idx, uint64_t pingpong) {
    *reinterpret_cast<__bcode__ uint32_t *>(pc + 4) = static_cast<uint32_t>(group_idx << 1 | pingpong);
  }
  __aicore_inline__ void UpdateCube(bcodeptr_t pc, __gm__ void *cube) { pc[5] = reinterpret_cast<uint64_t>(cube); }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t slice[], uint64_t tail[], uint64_t stride[],
                                    uint64_t subtile0, uint64_t subtile1) {
    vShard2D::Encode(pc, slice, tail, stride);
    pc[4] = subtile0 << 48 | subtile1 << 32;
    pc[5] = 0;
    return CODE_SIZE;
  }
  __aicore_inline__ void DecodeShard(bcodeptr_t pc, vShard2D &op) { vShard2D::Decode(pc, op); }
  __aicore_inline__ void UpdateShard(bcodeptr_t pc, uint64_t pos, uint64_t offset, bool last_m, bool last_n) {
    vShard2D::Update(pc, pos, offset, last_m, last_n);
  }
};

// [entry]
// common:
//  data(32) << 32 | reserved(12) << 20 | code_size_8B(12) << 8 | next_stage(reserve: 1) << 5 | extern_code(1) << 4 |
//  cube_mix(1) << 3 | type(3)
// data:
//  cube: reserved(32) << 32 | common(32)
//  vector:   tile_body(24) << 40 | tail_tail(8) << 32 | common(32)
//  vectorEx: visit_offset_8B(16) << 48 | visit_id(16) << 32 | common(32)
//  parallel: aiv_lookup(16) << 48 | aic_lookup(16) << 32 | common(32)

#define V_ENTRY_TYPE_V 0
#define V_ENTRY_TYPE_VE 1  // vector ext
#define V_ENTRY_TYPE_C 2
#define V_ENTRY_TYPE_P 3

#define V_ENTRY_MASK_TYPE 7ul
#define V_ENTRY_FLAG_CUBE_MIX 8
#define V_ENTRY_FLAG_EXTERN_CODE 16
#define V_ENTRY_CODE_SIZE_OFFSET 8
#define V_ENTRY_CODE_SIZE_BITS 12

// vector
#define V_ENTRY_V_TILE_TAIL_OFFSET 32
#define V_ENTRY_V_TILE_TAIL_BITS 8
#define V_ENTRY_V_TILE_BODY_OFFSET 40
#define V_ENTRY_V_TILE_BODY_BITS 24

// vector ext
#define V_ENTRY_VE_VISIT_ID_OFFSET 32
#define V_ENTRY_VE_VISIT_ID_BITS 16
#define V_ENTRY_VE_VISIT_OFFSET_OFFSET 48
#define V_ENTRY_VE_VISIT_OFFSET_BITS 16

// parallel
#define V_ENTRY_P_AIC_LKUP_OFFSET 32
#define V_ENTRY_P_AIV_LKUP_OFFSET 48
#define V_ENTRY_P_LKUP_MASK 0xfffful
#define V_ENTRY_P_LKUP_INVALID 0xff
struct vProgEntry {
  enum { CODE_SIZE = 2 };
  uint64_t entry;
  uint64_t b_begin;
  uint64_t b_num;
  uint64_t offset;
  // pc[0]: entry
  // pc[1]: reserve(16) << 48 | offset(16) << 32 | b_num(16) << 16 | b_begin(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, vProgEntry &e) {
    e.entry = pc[0];
    uint64_t data = pc[1];
    e.offset = (data >> 32) & 0xfffful;
    e.b_num = (data >> 16) & 0xfffful;
    e.b_begin = data & 0xfffful;
  }
  __aicore_inline__ uint64_t Encode(bcodeptr_t pc, uint64_t entry, uint64_t offset) {
    pc[0] = entry;
    pc[1] = offset << 32;
    return CODE_SIZE;
  }
  __aicore_inline__ void UpdateBlock(bcodeptr_t pc, uint64_t b_begin, uint64_t b_num) {
    pc[1] |= b_num << 16 | b_begin;
  }
};

__aicore_inline__ uint64_t vFftsSyncConfig(uint64_t mode, uint64_t event_id) { return 1ul | mode << 4 | event_id << 8; }
#endif  // _DVM_ISA_H_
