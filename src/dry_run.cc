/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <cstring>
#include "isa.h"
#include "system.h"
#include "code.h"

namespace dvm {
/************* cce definition ***************/
typedef enum {
  EVENT_ID0 = 0,
  EVENT_ID1,
  EVENT_ID2,
  EVENT_ID3,
  EVENT_ID4,
  EVENT_ID5,
  EVENT_ID6,
  EVENT_ID7,
} event_t;

typedef enum {
  PIPE_S = 0,  // Scalar Pipe
  PIPE_V,      // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
  PIPE_M,      // Matrix Pipe, including{}
  PIPE_MTE1,   // L1->L0{A,B}
  PIPE_MTE2,   // OUT ->{L1, L0{A,B}, UB}
  PIPE_MTE3,   // UB ->{OUT,L1}
  PIPE_FIX,
  PIPE_ALL,
} pipe_t;

typedef enum {
  NoQuant= 0,
  F322F16 = 1,
  F322BF16 = 16,
  VREQ4 = 21,
  REQ4 = 22,
  VQF322B8_PRE = 23,
  QF322B8_PRE = 24,
  VQF322S4_PRE = 25,
  QF322S4_PRE = 26,
} QuantMode_t;

const int ONLY_VALUE = 2;
const int PAD_NONE = 0;
const int inc = 0;

typedef enum {
  VA0 = 0,
  VA1,
  VA2,
  VA3,
  VA4,
  VA5,
  VA6,
  VA7,
} ub_addr8_t;

/************* cce intrinsic ***************/
struct CallTracer {
  CallTracer(const char *func) : func_(func) {}
  void Print() {}
  template <typename T>
  void Print(T head) {
    std::cout << head;
  }
  template <typename T>
  void Print(T* head) {
    std::cout << (void*)(head);
  }
  template <typename T, class ...Args>
  void Print(T head, Args... rest) {
    Print(head);
    std::cout << ", ";
    Print(rest...);
  }
  template <typename ...Args>
  void operator()(const Args... args) {
    std::cout << func_ << "(";
    Print(args...);
    std::cout << ")" << std::endl;
  }
  const char *func_;
};
#define CCE_CALL(func)  CallTracer(#func)

void pipe_barrier(pipe_t) {}
void set_flag(int, int, int) {}
void wait_flag(int, int, int) {}
uint64_t get_ctrl() { return 0; }
uint64_t sbitset1(uint64_t value, uint64_t bit) { return value | (1ul << bit); }
uint64_t sbitset0(uint64_t value, uint64_t bit) { return value & ~(1ul << bit); }
void set_ctrl(uint64_t) {}

#define set_atomic_none CCE_CALL(set_atomic_none)
#define set_mask_norm CCE_CALL(set_mask_norm)
#define set_vector_mask CCE_CALL(set_vector_mask)
#define set_mask_count CCE_CALL(set_mask_count)
#define set_cmpmask CCE_CALL(set_cmpmask)
#define dcci CCE_CALL(dcci)
#define set_atomic_f32 CCE_CALL(set_atomic_f32)
#define set_atomic_f16 CCE_CALL(set_atomic_f16)
#define set_atomic_add CCE_CALL(set_atomic_add)
#define set_atomic_max CCE_CALL(set_atomic_max)
#define set_atomic_min CCE_CALL(set_atomic_min)
#define trap CCE_CALL(trap)
#define set_deqscale CCE_CALL(set_deqscale)
#define set_ffts_base_addr CCE_CALL(set_ffts_base_addr)
#define ffts_cross_core_sync CCE_CALL(ffts_cross_core_sync)
#define wait_flag_dev CCE_CALL(wait_flag_dev)

#define copy_ubuf_to_ubuf CCE_CALL(copy_ubuf_to_ubuf)
#define copy_ubuf_to_gm CCE_CALL(copy_ubuf_to_gm)
#define copy_ubuf_to_gm_align_b8 CCE_CALL(copy_ubuf_to_gm_align_b8)
#define copy_ubuf_to_gm_align_b16 CCE_CALL(copy_ubuf_to_gm_align_b16)
#define copy_ubuf_to_gm_align_b32 CCE_CALL(copy_ubuf_to_gm_align_b32)

#define vabs CCE_CALL(vabs)
#define vln CCE_CALL(vln)
#define vexp CCE_CALL(vexp)
#define vconv_f322s32c CCE_CALL(vconv_f322s32c)
#define vconv_s322f32 CCE_CALL(vconv_s322f32)
#define vmuls CCE_CALL(vmuls)
#define vadds CCE_CALL(vadds)
#define vadd CCE_CALL(vadd)
#define vsub CCE_CALL(vsub)
#define vmul CCE_CALL(vmul)
#define vdiv CCE_CALL(vdiv)
#define vmax CCE_CALL(vmax)
#define vreducev2 CCE_CALL(vreducev2)
#define vgather CCE_CALL(vgather)
#define vcadd CCE_CALL(vcadd)
#define vcmax CCE_CALL(vcmax)
#define vcmin CCE_CALL(vcmin)
#define vector_dup CCE_CALL(vector_dup)
#define vsel CCE_CALL(vsel)
#define vcmpvs_ne CCE_CALL(vcmpvs_ne)
#define vcmpvs_eq CCE_CALL(vcmpvs_eq)
#define vcmpvs_le CCE_CALL(vcmpvs_le)
#define vcmpvs_lt CCE_CALL(vcmpvs_lt)
#define vcmpvs_ge CCE_CALL(vcmpvs_ge)
#define vcmpvs_gt CCE_CALL(vcmpvs_gt)
#define vcmpv_eq CCE_CALL(vcmpv_eq)
#define vcmpv_le CCE_CALL(vcmpv_le)
#define vcmpv_lt CCE_CALL(vcmpv_lt)
#define vcmpv_ge CCE_CALL(vcmpv_ge)
#define vcmpv_gt CCE_CALL(vcmpv_gt)
#define vconv_f322s32z CCE_CALL(vconv_f322s32z)
#define vconv_f322s64z CCE_CALL(vconv_f322s64z)
#define vconv_s322s64 CCE_CALL(vconv_s322s64)
#define vconv_s642f32z CCE_CALL(vconv_s642f32z)
#define vconv_s642s32 CCE_CALL(vconv_s642s32)
#define vshl CCE_CALL(vshl)
#define vshr CCE_CALL(vshr)
#define vmins CCE_CALL(vmins)
#define vconv_s162f16 CCE_CALL(vconv_s162f16)
#define vsqrt CCE_CALL(vsqrt)
#define vconv_f322f32r CCE_CALL(vconv_f322f32r)
#define vconv_f322f32f CCE_CALL(vconv_f322f32f)
#define vconv_f322f32c CCE_CALL(vconv_f322f32c)
#define vconv_f322f32z CCE_CALL(vconv_f322f32z)
#define vmin CCE_CALL(vmin)
#define vconv_f322f16 CCE_CALL(vconv_f322f16)
#define vconv_f162s8 CCE_CALL(vconv_f162s8)
#define vconv_f162f32 CCE_CALL(vconv_f162f32)
#define vconv_f162s32z CCE_CALL(vconv_f162s32z)
#define vconv_s82f16 CCE_CALL(vconv_s82f16)
#define vconv_deq CCE_CALL(vconv_deq)
#define vmaxs CCE_CALL(vmaxs)
#define vconv_bf162f32 CCE_CALL(vconv_bf162f32)
#define vconv_bf162s32z CCE_CALL(vconv_bf162s32z)
#define vconv_f322bf16r CCE_CALL(vconv_f322bf16r)
#define vconv_f162s16c CCE_CALL(vconv_f162s16c)
#define vbrcb CCE_CALL(vbrcb)
#define vcopy CCE_CALL(vcopy)
#define scatter_vnchwconv_b32 CCE_CALL(scatter_vnchwconv_b32)
#define scatter_vnchwconv_b16 CCE_CALL(scatter_vnchwconv_b16)
#define set_va_reg_sb CCE_CALL(set_va_reg_sb)


#define copy_gm_to_cbuf CCE_CALL(copy_gm_to_cbuf)
#define copy_cbuf_to_bt CCE_CALL(copy_cbuf_to_bt)
#define copy_matrix_cc_to_gm CCE_CALL(copy_matrix_cc_to_gm)
#define set_padding CCE_CALL(set_padding)
#define set_nd_para CCE_CALL(set_nd_para)
#define copy_gm_to_cbuf_multi_nd2nz_b16 CCE_CALL(copy_gm_to_cbuf_multi_nd2nz_b16)
#define load_cbuf_to_ca CCE_CALL(load_cbuf_to_ca)
#define load_cbuf_to_cb CCE_CALL(load_cbuf_to_cb)
#define mad CCE_CALL(mad)
#define create_ca_matrix CCE_CALL(create_ca_matrix)
#define create_cb_matrix CCE_CALL(create_ca_matrix)

/************* vm code stub ***************/
#define __VM_DRY_RUN__
#define __global__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define half int16_t
#define bfloat16_t int32_t
#define VMAIN_OFFSET  0
#define VA_BCODE_BASE_UB  (g_ubuf_mem + PC_BASE)
#define get_pc()  VMAIN_OFFSET
#define min(x,y) (x > y ? y : x)
#define max(x,y) (x < y ? y : x)

namespace {
uint64_t block_idx{0};
uint64_t block_num{0};
uint64_t g_subblockid{0};
uint64_t g_subblocknum{1};
bool g_cube_core{false};
void *g_bytecode{nullptr};
uint8_t *g_ubuf_mem{nullptr};

struct FuncEntry {
  const uint64_t **offsets;
  int idx;
  const char *name;
  void *func;
};
std::vector<FuncEntry> g_functable;

uint64_t get_subblockdim() { return g_subblocknum; }
uint64_t get_subblockid() { return g_subblockid; }
void *get_para_base() { return g_bytecode; }
uint8_t *GetFunc(uint64_t id, uint8_t *base_addr) {
  for (auto &e: g_functable) {
    if ((*e.offsets)[e.idx] == reinterpret_cast<uint64_t>(base_addr + id)) {
      std::cout << "[" << e.name << "]" << std::endl;
      return static_cast<uint8_t *>(e.func);
    }
  }
  std::cout << "GetFunc failed!" << std::endl;
  return nullptr;
}

uint64_t get_imm(uint64_t imm) { return reinterpret_cast<uint64_t>(g_ubuf_mem) + imm; }
void copy_gm_to_ubuf_align_b8(void *dst, void *src, uint8_t sid, uint16_t nBurst, uint32_t lenBurst,
                              uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap);

struct FuncRegister {
  FuncRegister(const uint64_t **offsets, int op_id, const char *name, void *func) {
    g_functable.emplace_back(FuncEntry{offsets, op_id, name, func});
  }
};
#define DEF_DRY_FUNC(offsets, OP, func) FuncRegister g_dryfunc_##OP(&dvm::g_system.offsets##_, OP, #OP, (void *)(&func));

#include "vm_aiv.cce"
#include "vm_aic.cce"

void copy_gm_to_ubuf_align_b8(void *dst, void *src, uint8_t sid, uint16_t nBurst, uint32_t lenBurst,
                              uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap) {
  if (dst == reinterpret_cast<void *>(PC_BASE)) {
    std::memcpy(g_ubuf_mem + PC_BASE, src, nBurst * lenBurst);
  }
  CallTracer("copy_gm_to_ubuf_align_b8")(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap,
                                          dstGap);
}
} // end namespace

void DryLaunch(Code *code, void *workspace, void *stream, uint64_t core_idx, bool is_cube) {
  if (g_system.Arch() != kAiCore_C220) {
    DvmException("dry run only supports C220");
  }
  g_cube_core = is_cube;
  g_subblocknum = Code::IsVector(code->target_) ? 1 : 2;
  if (is_cube) {
    block_idx = core_idx;
  } else {
    block_idx = core_idx / g_subblocknum;
    g_subblockid = core_idx & 1;
  }
  block_num = code->block_dim_;
  size_t size = code->data_size_;
  g_bytecode = std::malloc(size);
  std::memcpy(g_bytecode, code->data_, size);
  uint64_t ffts_addr = *(reinterpret_cast<uint64_t*>(g_bytecode));
  uint64_t entry = *(reinterpret_cast<uint64_t*>(g_bytecode) + 1);
  if (!g_cube_core) {
    g_ubuf_mem = reinterpret_cast<uint8_t *>(std::malloc(size + PC_BASE));
    dvm_mix_aiv(ffts_addr, entry);
    std::free(g_ubuf_mem);
  } else {
    dvm_mix_aic(ffts_addr, entry);
  }
  std::free(g_bytecode);
}
}  // namespace dvm
