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

#include <cstdint>
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define _VM_KERNEL_ 1
#include "isa.h"
#include "vm_gpu.h"

namespace dvm::gpu {
namespace {

constexpr uint64_t kGpuThreads = 256;
constexpr uint64_t kInvalidTile = ~0ull;
constexpr uint64_t kTileStride = (1ull << 32) - (1ull << 4);

uint64_t g_local_mem_size = 0;

// Tile format (kept in sync with src/vm_aiv.h):
//  tile_idx(32) << 32 | tile_count(28) << 4 | skip_stride(1) << 3 |
//  tail_flag(1) << 2 | end_flag(1) << 1 | not_begin_flag(1)
#define GPU_TILE_N_BEGIN_FLAG_OFFSET 0
#define GPU_TILE_END_FLAG_OFFSET 1
#define GPU_TILE_TAIL_FLAG_OFFSET 2
#define GPU_TILE_SKIP_FLAG_OFFSET 3
#define GPU_TILE_COUNT_OFFSET 4
#define GPU_TILE_CUR_IDX_OFFSET 32
#define GPU_TILE_COUNT_BITS 28

__device__ __forceinline__ uint64_t TileIdx(uint64_t tile) { return tile >> GPU_TILE_CUR_IDX_OFFSET; }
__device__ __forceinline__ bool IsLastTile(uint64_t tile) { return (tile & 6ull) == 6ull; }

__device__ __forceinline__ uint64_t AlignUp32(uint64_t x) { return (x + 31ull) & ~31ull; }

enum GpuRoundMode { kGpuRoundCacheLoad, kGpuRoundCacheStore, kGpuRoundCacheNone };

// Kept in sync with RoundUpTileIdx() in src/vm_aiv.h.
template <GpuRoundMode mode>
__device__ __forceinline__ uint64_t RoundUpTileIdxGpu(uint64_t tile, uint64_t tile_idx, uint64_t round_rank,
                                                      const uint64_t *rounds) {
  if (round_rank == 1) {
    uint64_t r1 = rounds[0];
    uint64_t r_tile_idx = tile_idx / r1;
    if (mode == kGpuRoundCacheLoad) {
      return r_tile_idx;
    } else if (mode == kGpuRoundCacheStore) {
      return r_tile_idx * r1 == tile_idx ? r_tile_idx : kInvalidTile;
    }
    return r_tile_idx;
  } else if (round_rank == 2) {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & 0xffffffffull;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t p1 = tile_idx / r1;
    if (mode == kGpuRoundCacheStore) {
      uint64_t align = p1 * r1;
      return tile_idx < align + r2 ? p1 * r2 + tile_idx - align : kInvalidTile;
    }
    return p1 * r2 + tile_idx % r2;
  } else if (round_rank == 3) {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & 0xffffffffull;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t r3 = rounds[1];
    uint64_t p0 = tile_idx / r1;
    uint64_t p1 = p0 * r2;
    uint64_t p2 = tile_idx % (r2 * r3);
    uint64_t p3 = p2 / r3;
    if (mode == kGpuRoundCacheLoad) {
      return p1 + p3;
    } else if (mode == kGpuRoundCacheStore) {
      return (tile_idx < p0 * r1 + r2 * r3) && (p3 * r3 == p2) ? p1 + p3 : kInvalidTile;
    }
    return p1 + p3;
  } else {
    uint64_t r2_r1 = rounds[0];
    uint64_t r1 = r2_r1 & 0xffffffffull;
    uint64_t r2 = r2_r1 >> 32;
    uint64_t r4_r3 = rounds[1];
    uint64_t r3 = r4_r3 & 0xffffffffull;
    uint64_t r4 = r4_r3 >> 32;
    uint64_t p0 = tile_idx / r1;
    uint64_t p1 = tile_idx % (r2 * r3);
    uint64_t p2 = p1 / r3;
    if (mode == kGpuRoundCacheStore) {
      return (tile_idx < p0 * r1 + r2 * r3) && (p1 < p2 * r3 + r4)
               ? (p0 * r2 + p2) * r4 + p1 - p2 * r3
               : kInvalidTile;
    }
    return (p0 * r2 + p2) * r4 + tile_idx % r4;
  }
}

struct VmCtx {
  uint8_t *ub;
};

template <typename F>
__device__ __forceinline__ void ParallelFor(uint64_t count, F f) {
  for (uint64_t i = threadIdx.x; i < count; i += kGpuThreads) {
    f(i);
  }
}

template <typename T>
__device__ __forceinline__ T *UBPtr(VmCtx &ctx, uint64_t off) {
  return reinterpret_cast<T *>(ctx.ub + off);
}

template <typename T>
__device__ __forceinline__ const T *UBCPtr(const VmCtx &ctx, uint64_t off) {
  return reinterpret_cast<const T *>(ctx.ub + off);
}

__device__ __forceinline__ float F16ToF32(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t F32ToF16(float v) { return __half_as_ushort(__float2half_rn(v)); }

__device__ __forceinline__ float BF16ToF32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t F32ToBF16(float v) {
  uint32_t u = __float_as_uint(v);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

struct F32Tag {
  using S = float;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint64_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint64_t i, float v) { p[i] = v; }
};

struct F16Tag {
  using S = uint16_t;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint64_t i) { return F16ToF32(p[i]); }
  __device__ __forceinline__ static void set(S *p, uint64_t i, float v) { p[i] = F32ToF16(v); }
};

struct BF16Tag {
  using S = uint16_t;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint64_t i) { return BF16ToF32(p[i]); }
  __device__ __forceinline__ static void set(S *p, uint64_t i, float v) { p[i] = F32ToBF16(v); }
};

struct I32Tag {
  using S = int32_t;
  using scalar_t = int32_t;
  __device__ __forceinline__ static int32_t get(const S *p, uint64_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint64_t i, int32_t v) { p[i] = v; }
};

struct BoolTag {
  using S = uint8_t;
  using scalar_t = uint8_t;
  __device__ __forceinline__ static uint8_t get(const S *p, uint64_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint64_t i, uint8_t v) { p[i] = v; }
};

// Binary functors.  F16/BF16 tags expose float to the functor.
struct AddOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};
struct SubOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a - b;
  }
};
struct MulOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a * b;
  }
};
struct DivOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a / b;
  }
};
struct MinOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? a : b;
  }
};
struct MaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a > b ? a : b;
  }
};
struct DivLeftOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return b / a;
  }
};
struct AndOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return static_cast<T>((a != T(0)) && (b != T(0)));
  }
};
struct OrOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return static_cast<T>((a != T(0)) || (b != T(0)));
  }
};
struct PowOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return powf(a, b); }
};

struct SqrtOp {
  __device__ __forceinline__ float operator()(float a) const { return sqrtf(a); }
};
struct AbsFOp {
  __device__ __forceinline__ float operator()(float a) const { return fabsf(a); }
};
struct AbsIOp {
  __device__ __forceinline__ int32_t operator()(int32_t a) const {
    return a < 0 ? static_cast<int32_t>(0u - static_cast<uint32_t>(a)) : a;
  }
};
struct LogOp {
  __device__ __forceinline__ float operator()(float a) const { return logf(a); }
};
struct ExpOp {
  __device__ __forceinline__ float operator()(float a) const { return expf(a); }
};
struct RoundOp {
  __device__ __forceinline__ float operator()(float a) const { return nearbyintf(a); }
};
struct FloorOp {
  __device__ __forceinline__ float operator()(float a) const { return floorf(a); }
};
struct CeilOp {
  __device__ __forceinline__ float operator()(float a) const { return ceilf(a); }
};
struct TruncOp {
  __device__ __forceinline__ float operator()(float a) const { return truncf(a); }
};
struct IsFiniteOp {
  __device__ __forceinline__ float operator()(float a) const { return isfinite(a) ? 1.0f : 0.0f; }
};
struct NotBoolOp {
  __device__ __forceinline__ uint8_t operator()(uint8_t a) const { return a == 0 ? 1 : 0; }
};

// Cast functors.
struct CvtF32ToF16 {
  __device__ __forceinline__ uint16_t operator()(float x) const { return F32ToF16(x); }
};
struct CvtF32ToBF16 {
  __device__ __forceinline__ uint16_t operator()(float x) const { return F32ToBF16(x); }
};
struct CvtF32ToI32 {
  __device__ __forceinline__ int32_t operator()(float x) const { return static_cast<int32_t>(x); }
};
struct CvtF16ToBool {
  __device__ __forceinline__ uint8_t operator()(uint16_t x) const { return F16ToF32(x) != 0.0f ? 1 : 0; }
};
struct CvtF16ToF32 {
  __device__ __forceinline__ float operator()(uint16_t x) const { return F16ToF32(x); }
};
struct CvtF16ToI32 {
  __device__ __forceinline__ int32_t operator()(uint16_t x) const { return static_cast<int32_t>(F16ToF32(x)); }
};
struct CvtBF16ToF32 {
  __device__ __forceinline__ float operator()(uint16_t x) const { return BF16ToF32(x); }
};
struct CvtBF16ToI32 {
  __device__ __forceinline__ int32_t operator()(uint16_t x) const { return static_cast<int32_t>(BF16ToF32(x)); }
};
struct CvtI32ToF16 {
  __device__ __forceinline__ uint16_t operator()(int32_t x) const {
    return F32ToF16(static_cast<float>(x));
  }
};
struct CvtI32ToF32 {
  __device__ __forceinline__ float operator()(int32_t x) const { return static_cast<float>(x); }
};
struct CvtBoolToF16 {
  __device__ __forceinline__ uint16_t operator()(uint8_t x) const {
    return F32ToF16(x != 0 ? 1.0f : 0.0f);
  }
};

template <typename T>
__device__ __forceinline__ bool CompareValue(T a, T b, int type) {
  switch (type) {
    case V_CMP_EQ:
      return a == b;
    case V_CMP_NE:
      return a != b;
    case V_CMP_GT:
      return a > b;
    case V_CMP_GE:
      return a >= b;
    case V_CMP_LT:
      return a < b;
    case V_CMP_LE:
      return a <= b;
    default:
      return false;
  }
}

template <typename Tag, typename F>
__device__ __forceinline__ void BinaryOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t xm, uint64_t count, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  const S *m = UBCPtr<S>(ctx, xm);
  ParallelFor(count, [&](uint64_t i) { Tag::set(d, i, f(Tag::get(n, i), Tag::get(m, i))); });
}

template <typename Tag, typename F>
__device__ __forceinline__ void BinarySOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t count,
                                          typename Tag::scalar_t scalar, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint64_t i) { Tag::set(d, i, f(Tag::get(n, i), scalar)); });
}

template <typename Tag, typename F>
__device__ __forceinline__ void UnaryOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t count, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint64_t i) { Tag::set(d, i, f(Tag::get(n, i))); });
}

template <typename Tag>
__device__ __forceinline__ void CmpOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t xm, uint64_t count, int type) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  const S *m = UBCPtr<S>(ctx, xm);
  ParallelFor(count, [&](uint64_t i) {
    bool r = CompareValue(Tag::get(n, i), Tag::get(m, i), type);
    Tag::set(d, i, r ? 1 : 0);
  });
}

template <typename Tag>
__device__ __forceinline__ void CmpSOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t count,
                                       typename Tag::scalar_t scalar, int type) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint64_t i) {
    bool r = CompareValue(Tag::get(n, i), scalar, type);
    Tag::set(d, i, r ? 1 : 0);
  });
}

template <typename Tag>
__device__ __forceinline__ void SelectOp(VmCtx &ctx, const vSelect &op) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, op.xd);
  const S *n = UBCPtr<S>(ctx, op.xn);
  const S *m = UBCPtr<S>(ctx, op.xm);
  const S *c = UBCPtr<S>(ctx, op.cond);
  ParallelFor(op.count, [&](uint64_t i) { d[i] = Tag::get(c, i) != 0 ? n[i] : m[i]; });
}

__device__ __forceinline__ void CopyBytesParallel(uint8_t *dst, const uint8_t *src, uint64_t bytes);

template <typename DstTag, typename SrcTag, typename Conv>
__device__ __forceinline__ void CastOp(VmCtx &ctx, uint64_t xd, uint64_t xn, uint64_t count, Conv conv) {
  using DS = typename DstTag::S;
  using SS = typename SrcTag::S;
  DS *d = UBPtr<DS>(ctx, xd);
  const SS *s = UBCPtr<SS>(ctx, xn);
  ParallelFor(count, [&](uint64_t i) { d[i] = conv(s[i]); });
}

__device__ __forceinline__ void TrapUnsupported(uint64_t id, uint64_t head, const void *pc) {
  if (threadIdx.x == 0) {
    printf("DVM GPU: unsupported opcode %llu (head=0x%llx) at %p\n", static_cast<unsigned long long>(id),
           static_cast<unsigned long long>(head), pc);
  }
  __trap();
}

__device__ __forceinline__ void CopyBytesSerial(uint8_t *dst, const uint8_t *src, uint64_t bytes) {
  uint64_t off = 0;
  auto align = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  if ((align & 15ull) == 0) {
    for (; off + 16 <= bytes; off += 16) {
      *reinterpret_cast<uint4 *>(dst + off) = *reinterpret_cast<const uint4 *>(src + off);
    }
  } else if ((align & 7ull) == 0) {
    for (; off + 8 <= bytes; off += 8) {
      *reinterpret_cast<uint64_t *>(dst + off) = *reinterpret_cast<const uint64_t *>(src + off);
    }
  } else if ((align & 3ull) == 0) {
    for (; off + 4 <= bytes; off += 4) {
      *reinterpret_cast<uint32_t *>(dst + off) = *reinterpret_cast<const uint32_t *>(src + off);
    }
  }
  for (; off < bytes; ++off) {
    dst[off] = src[off];
  }
}

__device__ __forceinline__ void CopyBytesParallel(uint8_t *dst, const uint8_t *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  uint64_t off = 0;
  auto align = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  if ((align & 15ull) == 0) {
    uint64_t count = bytes >> 4;
    ParallelFor(count, [&](uint64_t i) {
      *reinterpret_cast<uint4 *>(dst + (i << 4)) = *reinterpret_cast<const uint4 *>(src + (i << 4));
    });
    off = count << 4;
  } else if ((align & 7ull) == 0) {
    uint64_t count = bytes >> 3;
    ParallelFor(count, [&](uint64_t i) {
      *reinterpret_cast<uint64_t *>(dst + (i << 3)) = *reinterpret_cast<const uint64_t *>(src + (i << 3));
    });
    off = count << 3;
  } else if ((align & 3ull) == 0) {
    uint64_t count = bytes >> 2;
    ParallelFor(count, [&](uint64_t i) {
      *reinterpret_cast<uint32_t *>(dst + (i << 2)) = *reinterpret_cast<const uint32_t *>(src + (i << 2));
    });
    off = count << 2;
  }
  ParallelFor(bytes - off, [&](uint64_t i) { dst[off + i] = src[off + i]; });
}

__device__ __forceinline__ void ZeroBytesSerial(uint8_t *dst, uint64_t bytes) {
  uint64_t off = 0;
  auto align = reinterpret_cast<uintptr_t>(dst);
  if ((align & 15ull) == 0) {
    for (; off + 16 <= bytes; off += 16) {
      *reinterpret_cast<uint4 *>(dst + off) = make_uint4(0, 0, 0, 0);
    }
  } else if ((align & 7ull) == 0) {
    for (; off + 8 <= bytes; off += 8) {
      *reinterpret_cast<uint64_t *>(dst + off) = 0;
    }
  } else if ((align & 3ull) == 0) {
    for (; off + 4 <= bytes; off += 4) {
      *reinterpret_cast<uint32_t *>(dst + off) = 0;
    }
  }
  for (; off < bytes; ++off) {
    dst[off] = 0;
  }
}

__device__ __forceinline__ void ZeroBytesParallel(uint8_t *dst, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  uint64_t off = 0;
  auto align = reinterpret_cast<uintptr_t>(dst);
  if ((align & 15ull) == 0) {
    uint64_t count = bytes >> 4;
    ParallelFor(count, [&](uint64_t i) { *reinterpret_cast<uint4 *>(dst + (i << 4)) = make_uint4(0, 0, 0, 0); });
    off = count << 4;
  } else if ((align & 7ull) == 0) {
    uint64_t count = bytes >> 3;
    ParallelFor(count, [&](uint64_t i) { *reinterpret_cast<uint64_t *>(dst + (i << 3)) = 0; });
    off = count << 3;
  } else if ((align & 3ull) == 0) {
    uint64_t count = bytes >> 2;
    ParallelFor(count, [&](uint64_t i) { *reinterpret_cast<uint32_t *>(dst + (i << 2)) = 0; });
    off = count << 2;
  }
  ParallelFor(bytes - off, [&](uint64_t i) { dst[off + i] = 0; });
}

__device__ __forceinline__ void CopyRows(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t len,
                                         uint64_t src_stride, uint64_t dst_stride) {
  if (n == 0 || len == 0) {
    return;
  }
  if (n == 1 || (src_stride == len && dst_stride == len)) {
    CopyBytesParallel(dst, src, n * len);
    return;
  }
  if (n >= kGpuThreads && len <= 16 * kGpuThreads) {
    ParallelFor(n, [&](uint64_t row) {
      CopyBytesSerial(dst + row * dst_stride, src + row * src_stride, len);
    });
    return;
  }
  for (uint64_t row = 0; row < n; ++row) {
    CopyBytesParallel(dst + row * dst_stride, src + row * src_stride, len);
  }
}

__device__ __forceinline__ void ZeroRows(uint8_t *dst, uint64_t n, uint64_t len, uint64_t dst_stride) {
  if (n == 0 || len == 0) {
    return;
  }
  if (n == 1 || dst_stride == len) {
    ZeroBytesParallel(dst, n * len);
    return;
  }
  if (n >= kGpuThreads && len <= 16 * kGpuThreads) {
    ParallelFor(n, [&](uint64_t row) { ZeroBytesSerial(dst + row * dst_stride, len); });
    return;
  }
  for (uint64_t row = 0; row < n; ++row) {
    ZeroBytesParallel(dst + row * dst_stride, len);
  }
}

__device__ __forceinline__ void CopyGmToUb(VmCtx &ctx, uint64_t xd, const uint8_t *src, uint64_t n, uint64_t len,
                                           uint64_t src_gap, uint64_t dst_gap32) {
  if (n == 0 || len == 0) {
    return;
  }
  uint8_t *dst = ctx.ub + xd;
  uint64_t dst_row = AlignUp32(len) + (dst_gap32 << 5);
  auto align = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  if (n == 1 && (align & 15ull) == 0) {
    uint64_t count = len >> 4;
    ParallelFor(count, [&](uint64_t i) {
      uint4 v = __ldg(reinterpret_cast<const uint4 *>(src + (i << 4)));
      *reinterpret_cast<uint4 *>(dst + (i << 4)) = v;
    });
    uint64_t off = count << 4;
    if (off < len) {
      CopyBytesSerial(dst + off, src + off, len - off);
    }
  } else {
    CopyRows(dst, src, n, len, len + src_gap, dst_row);
  }
  uint64_t pad = AlignUp32(len) - len;
  if (pad != 0) {
    ZeroRows(dst + len, n, pad, dst_row);
  }
}

__device__ __forceinline__ void CopyUbToGm(VmCtx &ctx, uint8_t *dst, uint64_t xn, uint64_t n, uint64_t len,
                                           uint64_t src_gap32, uint64_t dst_gap) {
  if (n == 0 || len == 0) {
    return;
  }
  const uint8_t *src = ctx.ub + xn;
  uint64_t src_row = AlignUp32(len) + (src_gap32 << 5);
  uint64_t dst_row = len + dst_gap;
  CopyRows(dst, src, n, len, src_row, dst_row);
}

template <typename T>
__device__ __forceinline__ void BroadcastS(VmCtx &ctx, uint64_t xd, uint64_t count, T scalar) {
  T *d = UBPtr<T>(ctx, xd);
  ParallelFor(count, [&](uint64_t i) { d[i] = scalar; });
}

template <typename T>
__device__ __forceinline__ void FillSerial(T *d, uint64_t count, T value) {
  for (uint64_t i = 0; i < count; ++i) {
    d[i] = value;
  }
}

template <typename T>
__device__ __forceinline__ void BroadcastX(VmCtx &ctx, const vBroadcastX &op) {
  T *d = UBPtr<T>(ctx, op.xd);
  const T *s = UBCPtr<T>(ctx, op.xn);
  uint64_t src_stride = op.lead_num + op.lead_pad;
  uint64_t dst_elems = op.iter_num * op.lead_num * op.count;
  if (dst_elems == 0) {
    return;
  }
  uint64_t groups = op.iter_num * op.lead_num;
  if (op.count >= kGpuThreads) {
    for (uint64_t i = 0; i < op.iter_num; ++i) {
      const T *src_row = s + i * src_stride;
      T *dst_row = d + i * op.lead_num * op.count;
      for (uint64_t j = 0; j < op.lead_num; ++j) {
        T *dst = dst_row + j * op.count;
        T value = src_row[j];
        ParallelFor(op.count, [&](uint64_t k) { dst[k] = value; });
      }
    }
  } else if (groups >= kGpuThreads) {
    ParallelFor(groups, [&](uint64_t g) {
      uint64_t i = g / op.lead_num;
      uint64_t j = g - i * op.lead_num;
      FillSerial(d + (i * op.lead_num + j) * op.count, op.count, s[i * src_stride + j]);
    });
  } else {
    uint64_t row = op.lead_num * op.count;
    ParallelFor(dst_elems, [&](uint64_t idx) {
      uint64_t i = idx / row;
      uint64_t rem = idx - i * row;
      uint64_t j = rem / op.count;
      d[idx] = s[i * src_stride + j];
    });
  }
}

__device__ __forceinline__ void BroadcastY(VmCtx &ctx, const vBroadcastY &op) {
  uint64_t block = op.dup_stride * 32;
  uint64_t dst_bytes = op.iter_num * op.dup_num * block;
  if (dst_bytes == 0) {
    return;
  }
  uint8_t *d = ctx.ub + op.xd;
  const uint8_t *s = ctx.ub + op.xn;
  uint64_t per_iter = op.dup_num * block;
  uint64_t groups = op.iter_num * op.dup_num;
  if (block >= kGpuThreads) {
    for (uint64_t i = 0; i < op.iter_num; ++i) {
      const uint8_t *src_row = s + i * block;
      uint8_t *dst_row = d + i * per_iter;
      for (uint64_t j = 0; j < op.dup_num; ++j) {
        CopyBytesParallel(dst_row + j * block, src_row, block);
      }
    }
  } else if (groups >= kGpuThreads) {
    ParallelFor(groups, [&](uint64_t g) {
      uint64_t i = g / op.dup_num;
      uint64_t j = g - i * op.dup_num;
      CopyBytesSerial(d + i * per_iter + j * block, s + i * block, block);
    });
  } else {
    for (uint64_t i = 0; i < op.iter_num; ++i) {
      const uint8_t *src_row = s + i * block;
      uint8_t *dst_row = d + i * per_iter;
      for (uint64_t j = 0; j < op.dup_num; ++j) {
        CopyBytesParallel(dst_row + j * block, src_row, block);
      }
    }
  }
}

__device__ __forceinline__ void CopyOp(VmCtx &ctx, const vCopy &op) {
  uint64_t bytes = op.lenburst * 32;
  if (bytes == 0) {
    return;
  }
  uint8_t *d = ctx.ub + op.xd;
  uint8_t *s = ctx.ub + op.xn;
  CopyBytesParallel(d, s, bytes);
}

__device__ __forceinline__ void LoadOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint64_t tile) {
  vLoad op;
  vLoad::Decode(pc, head, op);
  uint64_t tile_idx = TileIdx(tile);
  if (op.round_rank > 0) {
    tile_idx = RoundUpTileIdxGpu<kGpuRoundCacheLoad>(tile, tile_idx, op.round_rank, pc + vLoad::ROUND_OFFSET);
  }
  uint64_t body_iter = op.body_iter;
  uint64_t iter_size = op.iter_size;
  if (IsLastTile(tile)) {
    if (body_iter == 1) {
      iter_size = op.tail_iter;
    } else {
      body_iter = op.tail_iter;
    }
  }
  const uint8_t *src = reinterpret_cast<const uint8_t *>(op.from) + op.tile_stride * tile_idx;
  CopyGmToUb(ctx, op.xn, src, body_iter, iter_size, 0, op.pad_size >> 5);
}

__device__ __forceinline__ void StoreOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint64_t tile) {
  vStore op;
  vStore::Decode(pc, head, op);
  uint64_t tile_idx = TileIdx(tile);
  if (op.round_rank > 0) {
    tile_idx = RoundUpTileIdxGpu<kGpuRoundCacheStore>(tile, tile_idx, op.round_rank, pc + vStore::ROUND_OFFSET);
    if (tile_idx == kInvalidTile) {
      return;
    }
  }
  uint64_t iter_num = op.iter_num;
  uint64_t iter_size = op.iter_size;
  if (IsLastTile(tile)) {
    if (iter_num == 1) {
      iter_size = op.iter_tail;
    } else {
      iter_num = op.iter_tail;
    }
  }
  uint8_t *dst = reinterpret_cast<uint8_t *>(op.to) + op.tile_stride * tile_idx;
  CopyUbToGm(ctx, dst, op.xn, iter_num, iter_size, op.pad_size >> 5, 0);
}

__device__ __forceinline__ void RunOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint64_t tile) {
  uint32_t id = head >> V_HEAD_ID_OFFSET;
  switch (id) {
    case F_LOAD:
      LoadOp(ctx, pc, head, tile);
      break;
    case F_STORE:
      StoreOp(ctx, pc, head, tile);
      break;
    case F_LOAD_DUMMY:
      break;
    case F_NOP:
      break;
    case F_COPY: {
      vCopy op;
      vCopy::Decode(pc, head, op);
      CopyOp(ctx, op);
      break;
    }
    case F_BROADCAST_S: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint32_t>(ctx, op.xd, op.count, static_cast<uint32_t>(op.scalar));
      break;
    }
    case F_BROADCAST_S_B8: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint8_t>(ctx, op.xd, op.count, static_cast<uint8_t>(op.scalar));
      break;
    }
    case F_BROADCAST_S_B16: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint16_t>(ctx, op.xd, op.count, static_cast<uint16_t>(op.scalar));
      break;
    }
    case F_BROADCAST_X_B8: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint8_t>(ctx, op);
      break;
    }
    case F_BROADCAST_X_B16: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint16_t>(ctx, op);
      break;
    }
    case F_BROADCAST_X_B32: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint32_t>(ctx, op);
      break;
    }
    case F_BROADCAST_Y: {
      vBroadcastY op;
      vBroadcastY::Decode(pc, head, op);
      BroadcastY(ctx, op);
      break;
    }
    case F_CAST_FP32_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, F32Tag>(ctx, op.xd, op.xn, op.count, CvtF32ToF16{});
      break;
    }
    case F_CAST_FP32_TO_BF16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<BF16Tag, F32Tag>(ctx, op.xd, op.xn, op.count, CvtF32ToBF16{});
      break;
    }
    case F_CAST_FP32_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, F32Tag>(ctx, op.xd, op.xn, op.count, CvtF32ToI32{});
      break;
    }
    case F_CAST_FP16_TO_BOOL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<BoolTag, F16Tag>(ctx, op.xd, op.xn, op.count, CvtF16ToBool{});
      break;
    }
    case F_CAST_FP16_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, F16Tag>(ctx, op.xd, op.xn, op.count, CvtF16ToF32{});
      break;
    }
    case F_CAST_FP16_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, F16Tag>(ctx, op.xd, op.xn, op.count, CvtF16ToI32{});
      break;
    }
    case F_CAST_BF16_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, BF16Tag>(ctx, op.xd, op.xn, op.count, CvtBF16ToF32{});
      break;
    }
    case F_CAST_BF16_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, BF16Tag>(ctx, op.xd, op.xn, op.count, CvtBF16ToI32{});
      break;
    }
    case F_CAST_INT32_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, I32Tag>(ctx, op.xd, op.xn, op.count, CvtI32ToF16{});
      break;
    }
    case F_CAST_INT32_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, I32Tag>(ctx, op.xd, op.xn, op.count, CvtI32ToF32{});
      break;
    }
    case F_CAST_BOOL_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, BoolTag>(ctx, op.xd, op.xn, op.count, CvtBoolToF16{});
      break;
    }
    case F_SQRT: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, SqrtOp{});
      break;
    }
    case F_SQRT_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, op.xd, op.xn, op.count, SqrtOp{});
      break;
    }
    case F_ABS: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, AbsFOp{});
      break;
    }
    case F_ABS_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, op.xd, op.xn, op.count, AbsFOp{});
      break;
    }
    case F_ABS_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<I32Tag>(ctx, op.xd, op.xn, op.count, AbsIOp{});
      break;
    }
    case F_LOG: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, LogOp{});
      break;
    }
    case F_LOG_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, op.xd, op.xn, op.count, LogOp{});
      break;
    }
    case F_EXP: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, ExpOp{});
      break;
    }
    case F_EXP_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, op.xd, op.xn, op.count, ExpOp{});
      break;
    }
    case F_ROUND: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, RoundOp{});
      break;
    }
    case F_FLOOR: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, FloorOp{});
      break;
    }
    case F_CEIL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, CeilOp{});
      break;
    }
    case F_TRUNC: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, TruncOp{});
      break;
    }
    case F_ISFINITE: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, op.xd, op.xn, op.count, IsFiniteOp{});
      break;
    }
    case F_ISFINITE_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, op.xd, op.xn, op.count, IsFiniteOp{});
      break;
    }
    case F_ISFINITE_BF16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<BF16Tag>(ctx, op.xd, op.xn, op.count, IsFiniteOp{});
      break;
    }
    case F_LOGICAL_NOT_BOOL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<BoolTag>(ctx, op.xd, op.xn, op.count, NotBoolOp{});
      break;
    }
    case F_ADDS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), AddOp{});
      break;
    }
    case F_ADDS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), AddOp{});
      break;
    }
    case F_ADDS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, op.xd, op.xn, op.count, BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), AddOp{});
      break;
    }
    case F_ADDS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<int32_t>(pc), AddOp{});
      break;
    }
    case F_MULS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), MulOp{});
      break;
    }
    case F_MULS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MulOp{});
      break;
    }
    case F_MULS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, op.xd, op.xn, op.count, BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MulOp{});
      break;
    }
    case F_MULS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<int32_t>(pc), MulOp{});
      break;
    }
    case F_DIVS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), DivOp{});
      break;
    }
    case F_DIVS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), DivOp{});
      break;
    }
    case F_SDIV: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), DivLeftOp{});
      break;
    }
    case F_SDIV_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), DivLeftOp{});
      break;
    }
    case F_MAXS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), MaxOp{});
      break;
    }
    case F_MAXS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MaxOp{});
      break;
    }
    case F_MAXS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, op.xd, op.xn, op.count, BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MaxOp{});
      break;
    }
    case F_MAXS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<int32_t>(pc), MaxOp{});
      break;
    }
    case F_MINS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<float>(pc), MinOp{});
      break;
    }
    case F_MINS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MinOp{});
      break;
    }
    case F_MINS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, op.xd, op.xn, op.count, BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MinOp{});
      break;
    }
    case F_MINS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, op.xd, op.xn, op.count, vBinaryS::GetScalar<int32_t>(pc), MinOp{});
      break;
    }
    case F_CMPS: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<F32Tag>(ctx, op.xd, op.xn, op.count, vCompareS::GetScalar<float>(pc), op.type);
      break;
    }
    case F_CMPS_FP16: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<F16Tag>(ctx, op.xd, op.xn, op.count, F16ToF32(vCompareS::GetScalar<uint16_t>(pc)), op.type);
      break;
    }
    case F_CMPS_BF16: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<BF16Tag>(ctx, op.xd, op.xn, op.count, BF16ToF32(vCompareS::GetScalar<uint16_t>(pc)), op.type);
      break;
    }
    case F_CMPS_INT32: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<I32Tag>(ctx, op.xd, op.xn, op.count, vCompareS::GetScalar<int32_t>(pc), op.type);
      break;
    }
    case F_CMPS_BOOL: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<BoolTag>(ctx, op.xd, op.xn, op.count, vCompareS::GetScalar<uint8_t>(pc), op.type);
      break;
    }
    case F_ADD: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, AddOp{});
      break;
    }
    case F_ADD_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, AddOp{});
      break;
    }
    case F_ADD_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, AddOp{});
      break;
    }
    case F_ADD_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, AddOp{});
      break;
    }
    case F_SUB: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, SubOp{});
      break;
    }
    case F_SUB_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, SubOp{});
      break;
    }
    case F_SUB_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, SubOp{});
      break;
    }
    case F_SUB_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, SubOp{});
      break;
    }
    case F_MUL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MulOp{});
      break;
    }
    case F_MUL_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MulOp{});
      break;
    }
    case F_MUL_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MulOp{});
      break;
    }
    case F_MUL_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MulOp{});
      break;
    }
    case F_DIV: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, DivOp{});
      break;
    }
    case F_DIV_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, DivOp{});
      break;
    }
    case F_MIN: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MinOp{});
      break;
    }
    case F_MIN_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MinOp{});
      break;
    }
    case F_MIN_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MinOp{});
      break;
    }
    case F_MIN_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MinOp{});
      break;
    }
    case F_MAX: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MaxOp{});
      break;
    }
    case F_MAX_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MaxOp{});
      break;
    }
    case F_MAX_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, MaxOp{});
      break;
    }
    case F_MAX_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, MaxOp{});
      break;
    }
    case F_CMP: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, static_cast<int>(op.type));
      break;
    }
    case F_CMP_FP16: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<F16Tag>(ctx, op.xd, op.xn, op.xm, op.count, static_cast<int>(op.type));
      break;
    }
    case F_CMP_BF16: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<BF16Tag>(ctx, op.xd, op.xn, op.xm, op.count, static_cast<int>(op.type));
      break;
    }
    case F_CMP_INT32: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<I32Tag>(ctx, op.xd, op.xn, op.xm, op.count, static_cast<int>(op.type));
      break;
    }
    case F_CMP_BOOL: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<BoolTag>(ctx, op.xd, op.xn, op.xm, op.count, static_cast<int>(op.type));
      break;
    }
    case F_SEL: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<F32Tag>(ctx, op);
      break;
    }
    case F_SEL_FP16: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<F16Tag>(ctx, op);
      break;
    }
    case F_SEL_BF16: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<BF16Tag>(ctx, op);
      break;
    }
    case F_SEL_INT32: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<I32Tag>(ctx, op);
      break;
    }
    case F_POW: {
      vBinaryWS op;
      vBinaryWS::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, op.xd, op.xn, op.xm, op.count, PowOp{});
      break;
    }
    case F_LOGICAL_AND_BOOL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BoolTag>(ctx, op.xd, op.xn, op.xm, op.count, AndOp{});
      break;
    }
    case F_LOGICAL_OR_BOOL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BoolTag>(ctx, op.xd, op.xn, op.xm, op.count, OrOp{});
      break;
    }
    default:
      TrapUnsupported(id, head, pc);
      break;
  }
}

}  // namespace

__global__ void dvm_vm_gpu(__grid_constant__ const VmParam param) {
  extern __shared__ uint8_t smem[];
  uint64_t *code = const_cast<uint64_t *>(param.code);
  uint64_t entry = code[1];
  if (entry & V_ENTRY_FLAG_EXTERN_CODE) {
    code = reinterpret_cast<uint64_t *>(code[0]);
  }
  VmCtx ctx;
  ctx.ub = smem;

  uint8_t *base_body = reinterpret_cast<uint8_t *>(code + 2);
  uint64_t bid = blockIdx.x;
  uint64_t bnum = gridDim.x;
  uint64_t ktype = entry & V_ENTRY_MASK_TYPE;

  if (ktype == V_ENTRY_TYPE_P) {
    uint64_t lookup_offset = (entry >> V_ENTRY_P_AIV_LKUP_OFFSET) & V_ENTRY_P_LKUP_MASK;
    uint8_t prog_offset = base_body[lookup_offset + bid];
    if (prog_offset == V_ENTRY_P_LKUP_INVALID) {
      return;
    }
    const uint64_t *prog = reinterpret_cast<const uint64_t *>(base_body) + prog_offset;
    entry = prog[0];
    uint64_t data = prog[1];
    uint64_t b_begin = data & 0xffffull;
    uint64_t b_num = (data >> 16) & 0xffffull;
    uint64_t body_offset = (data >> 32) & 0xffffull;
    bid -= b_begin;
    bnum = b_num;
    base_body += body_offset;
    ktype = entry & V_ENTRY_MASK_TYPE;
  }

  uint64_t *body = reinterpret_cast<uint64_t *>(base_body);
  uint64_t body_tile = (entry >> V_ENTRY_V_TILE_BODY_OFFSET) & ((1ull << V_ENTRY_V_TILE_BODY_BITS) - 1ull);
  uint64_t tail_diff = (entry >> V_ENTRY_V_TILE_TAIL_OFFSET) & ((1ull << V_ENTRY_V_TILE_TAIL_BITS) - 1ull);
  uint64_t count = body_tile;
  uint64_t tile = (bid * body_tile) << GPU_TILE_CUR_IDX_OFFSET;
  if (bid == bnum - 1) {
    count -= tail_diff;
    tile |= 1ull << GPU_TILE_TAIL_FLAG_OFFSET;
  }
  tile |= count << GPU_TILE_COUNT_OFFSET;

  while (((tile >> GPU_TILE_COUNT_OFFSET) & ((1ull << GPU_TILE_COUNT_BITS) - 1ull)) != 0) {
    uint64_t cur_count = (tile >> GPU_TILE_COUNT_OFFSET) & ((1ull << GPU_TILE_COUNT_BITS) - 1ull);
    if (cur_count == 1) {
      tile |= 1ull << GPU_TILE_END_FLAG_OFFSET;
    }
    uint64_t *pc = body;
    for (uint64_t head = *pc; head != 0; head = *pc) {
      bcodeptr_t op_pc = pc;
      if (head & (1ull << V_HEAD_SIMD_FLAG_OFFSET)) {
        pc += (head >> V_HEAD_SIZE_OFFSET) & V_HEAD_SIZE_MASK;
      } else {
        pc += (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
      }
      RunOp(ctx, op_pc, head, tile);
      __syncthreads();
    }
    tile |= 1ull << GPU_TILE_N_BEGIN_FLAG_OFFSET;
    tile += kTileStride;
  }
}

int InitVM(uint64_t local_mem_size) {
  if (local_mem_size == 0 || g_local_mem_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto err = cudaFuncSetAttribute(dvm_vm_gpu, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(local_mem_size));
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaFuncSetAttribute(dvm_vm_gpu, cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  g_local_mem_size = local_mem_size;
  return 0;
}

int LaunchVM(const VmParam &param, uint32_t block_dim, void *stream) {
  if (block_dim == 0) {
    return 0;
  }
  if (g_local_mem_size == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  auto shared_bytes = static_cast<size_t>(g_local_mem_size);
  dvm_vm_gpu<<<block_dim, kGpuThreads, shared_bytes, cuda_stream>>>(param);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace dvm::gpu
