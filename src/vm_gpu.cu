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
#include "gpu_pointwise.cuh"

namespace dvm::gpu {
namespace {

constexpr uint32_t kInvalidTile = 0xffffffffu;

__device__ __forceinline__ uint32_t AlignUp32(uint32_t x) { return (x + 31u) & ~31u; }

enum GpuRoundMode { kGpuRoundCacheLoad, kGpuRoundCacheStore, kGpuRoundStoreAtomic, kGpuRoundCacheNone };

// Kept in sync with RoundUpTileIdx() in src/vm_aiv.h.
template <GpuRoundMode mode>
__device__ __forceinline__ uint32_t RoundUpTileIdxGpu(uint32_t tile_idx, uint32_t round_rank,
                                                      const uint64_t *rounds) {
  if (round_rank == 1) {
    uint32_t r1 = static_cast<uint32_t>(rounds[0]);
    uint32_t r_tile_idx = tile_idx / r1;
    if (mode == kGpuRoundCacheLoad) {
      return r_tile_idx;
    } else if (mode == kGpuRoundCacheStore) {
      return r_tile_idx * r1 == tile_idx ? r_tile_idx : kInvalidTile;
    } else if (mode == kGpuRoundStoreAtomic) {
      return r_tile_idx * r1 + r1 - 1 == tile_idx ? r_tile_idx : kInvalidTile;
    }
    return r_tile_idx;
  } else if (round_rank == 2) {
    uint2 r2_r1 = *reinterpret_cast<const uint2 *>(rounds);
    uint32_t r1 = r2_r1.x;
    uint32_t r2 = r2_r1.y;
    uint32_t p1 = tile_idx / r1;
    if (mode == kGpuRoundCacheStore) {
      uint32_t align = p1 * r1;
      return tile_idx < align + r2 ? p1 * r2 + tile_idx - align : kInvalidTile;
    }
    return p1 * r2 + tile_idx % r2;
  } else if (round_rank == 3) {
    uint2 r2_r1 = *reinterpret_cast<const uint2 *>(rounds);
    uint32_t r1 = r2_r1.x;
    uint32_t r2 = r2_r1.y;
    uint32_t r3 = static_cast<uint32_t>(rounds[1]);
    uint32_t p0 = tile_idx / r1;
    uint32_t p1 = p0 * r2;
    uint32_t p2 = tile_idx % (r2 * r3);
    uint32_t p3 = p2 / r3;
    if (mode == kGpuRoundCacheLoad) {
      return p1 + p3;
    } else if (mode == kGpuRoundCacheStore) {
      return (tile_idx < p0 * r1 + r2 * r3) && (p3 * r3 == p2) ? p1 + p3 : kInvalidTile;
    } else if (mode == kGpuRoundStoreAtomic) {
      return p3 * r3 + r3 - 1 == p2 ? p1 + p3 : kInvalidTile;
    }
    return p1 + p3;
  } else {
    uint2 r2_r1 = *reinterpret_cast<const uint2 *>(rounds);
    uint2 r4_r3 = *reinterpret_cast<const uint2 *>(rounds + 1);
    uint32_t r1 = r2_r1.x;
    uint32_t r2 = r2_r1.y;
    uint32_t r3 = r4_r3.x;
    uint32_t r4 = r4_r3.y;
    uint32_t p0 = tile_idx / r1;
    uint32_t p1 = tile_idx % (r2 * r3);
    uint32_t p2 = p1 / r3;
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

__device__ __forceinline__ uint32_t NarrowU32(uint64_t v) { return static_cast<uint32_t>(v); }

template <typename F>
__device__ __forceinline__ void ParallelFor(uint32_t count, F f) {
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    f(i);
  }
}

__device__ __forceinline__ double WarpReduceSum(double value) {
  const unsigned mask = __activemask();
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ double BlockReduceSum(double value, double *scratch) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warp_count = (blockDim.x + 31u) >> 5;
  value = WarpReduceSum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < warp_count ? scratch[threadIdx.x] : 0.0;
  __syncthreads();
  if (warp == 0) {
    value = WarpReduceSum(value);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T *UBPtr(VmCtx &ctx, uint32_t off) {
  return reinterpret_cast<T *>(ctx.ub + off);
}

template <typename T>
__device__ __forceinline__ const T *UBCPtr(const VmCtx &ctx, uint32_t off) {
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
  // Rounding a NaN payload as an integer can carry into the sign bit or truncate it to infinity.
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fc0u;
  }
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

struct F32Tag {
  using S = float;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint32_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint32_t i, float v) { p[i] = v; }
};

struct F16Tag {
  using S = uint16_t;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint32_t i) { return F16ToF32(p[i]); }
  __device__ __forceinline__ static void set(S *p, uint32_t i, float v) { p[i] = F32ToF16(v); }
};

struct BF16Tag {
  using S = uint16_t;
  using scalar_t = float;
  __device__ __forceinline__ static float get(const S *p, uint32_t i) { return BF16ToF32(p[i]); }
  __device__ __forceinline__ static void set(S *p, uint32_t i, float v) { p[i] = F32ToBF16(v); }
};

struct I32Tag {
  using S = int32_t;
  using scalar_t = int32_t;
  __device__ __forceinline__ static int32_t get(const S *p, uint32_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint32_t i, int32_t v) { p[i] = v; }
};

struct BoolTag {
  using S = uint8_t;
  using scalar_t = uint8_t;
  __device__ __forceinline__ static uint8_t get(const S *p, uint32_t i) { return p[i]; }
  __device__ __forceinline__ static void set(S *p, uint32_t i, uint8_t v) { p[i] = v; }
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
struct TanhOp {
  __device__ __forceinline__ float operator()(float a) const { return tanhf(a); }
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

__device__ __forceinline__ void FmaOp(VmCtx &ctx, const vFma &op) {
  float *dst = UBPtr<float>(ctx, NarrowU32(op.xd));
  const float *lhs = UBCPtr<float>(ctx, NarrowU32(op.xn));
  const float *rhs = UBCPtr<float>(ctx, NarrowU32(op.xm));
  const auto count = NarrowU32(op.count);
  const auto fma = [negate = op.negate_product](float a, float b, float c) {
    return __fmaf_rn(negate ? -a : a, b, c);
  };
  if (op.mode == vFma::kTensors) {
    const float *addend = UBCPtr<float>(ctx, op.operand);
    ParallelFor(count, [&](uint32_t i) { dst[i] = fma(lhs[i], rhs[i], addend[i]); });
  } else {
    const float scalar = __uint_as_float(op.operand);
    if (op.mode == vFma::kScalarAddend) {
      ParallelFor(count, [&](uint32_t i) { dst[i] = fma(lhs[i], rhs[i], scalar); });
    } else {
      ParallelFor(count, [&](uint32_t i) { dst[i] = fma(lhs[i], scalar, rhs[i]); });
    }
  }
}

template <typename Tag, typename F>
__device__ __forceinline__ void BinaryOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t xm, uint32_t count, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  const S *m = UBCPtr<S>(ctx, xm);
  ParallelFor(count, [&](uint32_t i) { Tag::set(d, i, f(Tag::get(n, i), Tag::get(m, i))); });
}

template <typename Tag, typename F>
__device__ __forceinline__ void BinarySOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t count,
                                          typename Tag::scalar_t scalar, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint32_t i) { Tag::set(d, i, f(Tag::get(n, i), scalar)); });
}

template <typename Tag, typename F>
__device__ __forceinline__ void UnaryOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t count, F f) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint32_t i) { Tag::set(d, i, f(Tag::get(n, i))); });
}

template <typename Tag>
__device__ __forceinline__ void CmpOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t xm, uint32_t count, int type) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  const S *m = UBCPtr<S>(ctx, xm);
  ParallelFor(count, [&](uint32_t i) {
    bool r = CompareValue(Tag::get(n, i), Tag::get(m, i), type);
    Tag::set(d, i, r ? 1 : 0);
  });
}

template <typename Tag>
__device__ __forceinline__ void CmpSOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t count,
                                       typename Tag::scalar_t scalar, int type) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  ParallelFor(count, [&](uint32_t i) {
    bool r = CompareValue(Tag::get(n, i), scalar, type);
    Tag::set(d, i, r ? 1 : 0);
  });
}

template <typename Tag>
__device__ __forceinline__ void SelectOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t xm, uint32_t cond,
                                         uint32_t count) {
  using S = typename Tag::S;
  S *d = UBPtr<S>(ctx, xd);
  const S *n = UBCPtr<S>(ctx, xn);
  const S *m = UBCPtr<S>(ctx, xm);
  const S *c = UBCPtr<S>(ctx, cond);
  ParallelFor(count, [&](uint32_t i) { d[i] = Tag::get(c, i) != 0 ? n[i] : m[i]; });
}

__device__ __forceinline__ void CopyBytesParallel(uint8_t *dst, const uint8_t *src, uint32_t bytes);

template <typename DstTag, typename SrcTag, typename Conv>
__device__ __forceinline__ void CastOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t count, Conv conv) {
  using DS = typename DstTag::S;
  using SS = typename SrcTag::S;
  DS *d = UBPtr<DS>(ctx, xd);
  const SS *s = UBCPtr<SS>(ctx, xn);
  ParallelFor(count, [&](uint32_t i) { d[i] = conv(s[i]); });
}

__device__ __forceinline__ void TrapUnsupported(uint32_t id, uint64_t head, const void *pc) {
  if (threadIdx.x == 0) {
    printf("DVM GPU: unsupported opcode %llu (head=0x%llx) at %p\n", static_cast<unsigned long long>(id),
           static_cast<unsigned long long>(head), pc);
  }
  __trap();
}

__device__ __forceinline__ void CopyBytesSerial(uint8_t *dst, const uint8_t *src, uint32_t bytes) {
  uint32_t off = 0;
  uint32_t align = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src));
  if ((align & 15u) == 0) {
    for (; off + 16 <= bytes; off += 16) {
      *reinterpret_cast<uint4 *>(dst + off) = *reinterpret_cast<const uint4 *>(src + off);
    }
  } else if ((align & 7u) == 0) {
    for (; off + 8 <= bytes; off += 8) {
      *reinterpret_cast<uint64_t *>(dst + off) = *reinterpret_cast<const uint64_t *>(src + off);
    }
  } else if ((align & 3u) == 0) {
    for (; off + 4 <= bytes; off += 4) {
      *reinterpret_cast<uint32_t *>(dst + off) = *reinterpret_cast<const uint32_t *>(src + off);
    }
  }
  for (; off < bytes; ++off) {
    dst[off] = src[off];
  }
}

__device__ __forceinline__ void CopyBytesParallel(uint8_t *dst, const uint8_t *src, uint32_t bytes) {
  if (bytes == 0) {
    return;
  }
  uint32_t off = 0;
  uint32_t align = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src));
  if ((align & 15u) == 0) {
    uint32_t count = bytes >> 4;
    ParallelFor(count, [&](uint32_t i) {
      *reinterpret_cast<uint4 *>(dst + (i << 4)) = *reinterpret_cast<const uint4 *>(src + (i << 4));
    });
    off = count << 4;
  } else if ((align & 7u) == 0) {
    uint32_t count = bytes >> 3;
    ParallelFor(count, [&](uint32_t i) {
      *reinterpret_cast<uint64_t *>(dst + (i << 3)) = *reinterpret_cast<const uint64_t *>(src + (i << 3));
    });
    off = count << 3;
  } else if ((align & 3u) == 0) {
    uint32_t count = bytes >> 2;
    ParallelFor(count, [&](uint32_t i) {
      *reinterpret_cast<uint32_t *>(dst + (i << 2)) = *reinterpret_cast<const uint32_t *>(src + (i << 2));
    });
    off = count << 2;
  }
  ParallelFor(bytes - off, [&](uint32_t i) { dst[off + i] = src[off + i]; });
}

__device__ __forceinline__ void ZeroBytesSerial(uint8_t *dst, uint32_t bytes) {
  uint32_t off = 0;
  uint32_t align = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
  if ((align & 15u) == 0) {
    for (; off + 16 <= bytes; off += 16) {
      *reinterpret_cast<uint4 *>(dst + off) = make_uint4(0, 0, 0, 0);
    }
  } else if ((align & 7u) == 0) {
    for (; off + 8 <= bytes; off += 8) {
      *reinterpret_cast<uint64_t *>(dst + off) = 0;
    }
  } else if ((align & 3u) == 0) {
    for (; off + 4 <= bytes; off += 4) {
      *reinterpret_cast<uint32_t *>(dst + off) = 0;
    }
  }
  for (; off < bytes; ++off) {
    dst[off] = 0;
  }
}

__device__ __forceinline__ void ZeroBytesParallel(uint8_t *dst, uint32_t bytes) {
  if (bytes == 0) {
    return;
  }
  uint32_t off = 0;
  uint32_t align = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
  if ((align & 15u) == 0) {
    uint32_t count = bytes >> 4;
    ParallelFor(count, [&](uint32_t i) { *reinterpret_cast<uint4 *>(dst + (i << 4)) = make_uint4(0, 0, 0, 0); });
    off = count << 4;
  } else if ((align & 7u) == 0) {
    uint32_t count = bytes >> 3;
    ParallelFor(count, [&](uint32_t i) { *reinterpret_cast<uint64_t *>(dst + (i << 3)) = 0; });
    off = count << 3;
  } else if ((align & 3u) == 0) {
    uint32_t count = bytes >> 2;
    ParallelFor(count, [&](uint32_t i) { *reinterpret_cast<uint32_t *>(dst + (i << 2)) = 0; });
    off = count << 2;
  }
  ParallelFor(bytes - off, [&](uint32_t i) { dst[off + i] = 0; });
}

__device__ __forceinline__ void CopyRows(uint8_t *dst, const uint8_t *src, uint32_t n, uint32_t len,
                                         uint32_t src_stride, uint32_t dst_stride) {
  if (n == 0 || len == 0) {
    return;
  }
  if (n == 1 || (src_stride == len && dst_stride == len)) {
    CopyBytesParallel(dst, src, n * len);
    return;
  }
  if (n >= blockDim.x && len <= 16 * blockDim.x) {
    ParallelFor(n, [&](uint32_t row) {
      CopyBytesSerial(dst + row * dst_stride, src + row * src_stride, len);
    });
    return;
  }
  for (uint32_t row = 0; row < n; ++row) {
    CopyBytesParallel(dst + row * dst_stride, src + row * src_stride, len);
  }
}

__device__ __forceinline__ void ZeroRows(uint8_t *dst, uint32_t n, uint32_t len, uint32_t dst_stride) {
  if (n == 0 || len == 0) {
    return;
  }
  if (n == 1 || dst_stride == len) {
    ZeroBytesParallel(dst, n * len);
    return;
  }
  if (n >= blockDim.x && len <= 16 * blockDim.x) {
    ParallelFor(n, [&](uint32_t row) { ZeroBytesSerial(dst + row * dst_stride, len); });
    return;
  }
  for (uint32_t row = 0; row < n; ++row) {
    ZeroBytesParallel(dst + row * dst_stride, len);
  }
}

__device__ __forceinline__ void CopyGmToUb(VmCtx &ctx, uint32_t xd, const uint8_t *src, uint32_t n, uint32_t len,
                                           uint32_t src_gap, uint32_t dst_gap32) {
  if (n == 0 || len == 0) {
    return;
  }
  uint8_t *dst = ctx.ub + xd;
  uint32_t dst_row = AlignUp32(len) + (dst_gap32 << 5);
  uint32_t align = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src));
  if (n == 1 && (align & 15u) == 0) {
    uint32_t count = len >> 4;
    ParallelFor(count, [&](uint32_t i) {
      uint4 v = __ldg(reinterpret_cast<const uint4 *>(src + (i << 4)));
      *reinterpret_cast<uint4 *>(dst + (i << 4)) = v;
    });
    uint32_t off = count << 4;
    if (off < len) {
      CopyBytesSerial(dst + off, src + off, len - off);
    }
  } else {
    CopyRows(dst, src, n, len, len + src_gap, dst_row);
  }
  uint32_t pad = AlignUp32(len) - len;
  if (pad != 0) {
    ZeroRows(dst + len, n, pad, dst_row);
  }
}

__device__ __forceinline__ void CopyUbToGm(VmCtx &ctx, uint8_t *dst, uint32_t xn, uint32_t n, uint32_t len,
                                           uint32_t src_gap32, uint32_t dst_gap) {
  if (n == 0 || len == 0) {
    return;
  }
  const uint8_t *src = ctx.ub + xn;
  uint32_t src_row = AlignUp32(len) + (src_gap32 << 5);
  uint32_t dst_row = len + dst_gap;
  CopyRows(dst, src, n, len, src_row, dst_row);
}

template <typename T>
__device__ __forceinline__ void BroadcastS(VmCtx &ctx, uint32_t xd, uint32_t count, T scalar) {
  T *d = UBPtr<T>(ctx, xd);
  ParallelFor(count, [&](uint32_t i) { d[i] = scalar; });
}

template <typename T>
__device__ __forceinline__ void FillSerial(T *d, uint32_t count, T value) {
  for (uint32_t i = 0; i < count; ++i) {
    d[i] = value;
  }
}

template <typename T>
__device__ __forceinline__ void BroadcastX(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t count,
                                           uint32_t lead_num, uint32_t iter_num, uint32_t lead_pad) {
  T *d = UBPtr<T>(ctx, xd);
  const T *s = UBCPtr<T>(ctx, xn);
  uint32_t src_stride = lead_num + lead_pad;
  uint32_t dst_elems = iter_num * lead_num * count;
  if (dst_elems == 0) {
    return;
  }
  uint32_t groups = iter_num * lead_num;
  if (count >= blockDim.x) {
    for (uint32_t i = 0; i < iter_num; ++i) {
      const T *src_row = s + i * src_stride;
      T *dst_row = d + i * lead_num * count;
      for (uint32_t j = 0; j < lead_num; ++j) {
        T *dst = dst_row + j * count;
        T value = src_row[j];
        ParallelFor(count, [&](uint32_t k) { dst[k] = value; });
      }
    }
  } else if (groups >= blockDim.x) {
    ParallelFor(groups, [&](uint32_t g) {
      uint32_t i = g / lead_num;
      uint32_t j = g - i * lead_num;
      FillSerial(d + (i * lead_num + j) * count, count, s[i * src_stride + j]);
    });
  } else {
    uint32_t row = lead_num * count;
    ParallelFor(dst_elems, [&](uint32_t idx) {
      uint32_t i = idx / row;
      uint32_t rem = idx - i * row;
      uint32_t j = rem / count;
      d[idx] = s[i * src_stride + j];
    });
  }
}

__device__ __forceinline__ void BroadcastY(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t iter_num,
                                           uint32_t dup_num, uint32_t dup_stride) {
  uint32_t block = dup_stride * 32;
  uint32_t dst_bytes = iter_num * dup_num * block;
  if (dst_bytes == 0) {
    return;
  }
  uint8_t *d = ctx.ub + xd;
  const uint8_t *s = ctx.ub + xn;
  uint32_t per_iter = dup_num * block;
  uint32_t groups = iter_num * dup_num;
  if (block >= blockDim.x) {
    for (uint32_t i = 0; i < iter_num; ++i) {
      const uint8_t *src_row = s + i * block;
      uint8_t *dst_row = d + i * per_iter;
      for (uint32_t j = 0; j < dup_num; ++j) {
        CopyBytesParallel(dst_row + j * block, src_row, block);
      }
    }
  } else if (groups >= blockDim.x) {
    ParallelFor(groups, [&](uint32_t g) {
      uint32_t i = g / dup_num;
      uint32_t j = g - i * dup_num;
      CopyBytesSerial(d + i * per_iter + j * block, s + i * block, block);
    });
  } else {
    for (uint32_t i = 0; i < iter_num; ++i) {
      const uint8_t *src_row = s + i * block;
      uint8_t *dst_row = d + i * per_iter;
      for (uint32_t j = 0; j < dup_num; ++j) {
        CopyBytesParallel(dst_row + j * block, src_row, block);
      }
    }
  }
}

__device__ __forceinline__ void CopyOp(VmCtx &ctx, uint32_t xd, uint32_t xn, uint32_t lenburst) {
  uint32_t bytes = lenburst * 32;
  if (bytes == 0) {
    return;
  }
  uint8_t *d = ctx.ub + xd;
  uint8_t *s = ctx.ub + xn;
  CopyBytesParallel(d, s, bytes);
}

__device__ __forceinline__ void LoadOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint32_t tile_idx, bool is_last) {
  vLoad op;
  vLoad::Decode(pc, head, op);
  const uint8_t *from = reinterpret_cast<const uint8_t *>(op.from);
  uint32_t round_rank = NarrowU32(op.round_rank);
  uint32_t body_iter = NarrowU32(op.body_iter);
  uint32_t iter_size = NarrowU32(op.iter_size);
  uint32_t tail_iter = NarrowU32(op.tail_iter);
  uint32_t tile_stride = NarrowU32(op.tile_stride);
  uint32_t xn = NarrowU32(op.xn);
  uint32_t pad_size = NarrowU32(op.pad_size);
  if (round_rank > 0) {
    tile_idx = RoundUpTileIdxGpu<kGpuRoundCacheLoad>(tile_idx, round_rank, pc + vLoad::ROUND_OFFSET);
  }
  if (is_last) {
    if (body_iter == 1) {
      iter_size = tail_iter;
    } else {
      body_iter = tail_iter;
    }
  }
  uint64_t tile_off = static_cast<uint64_t>(tile_stride) * tile_idx;
  const uint8_t *src = from + tile_off;
  CopyGmToUb(ctx, xn, src, body_iter, iter_size, 0, pad_size >> 5);
}

__device__ __forceinline__ void StoreOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint32_t tile_idx, bool is_last) {
  vStore op;
  vStore::Decode(pc, head, op);
  uint8_t *to = reinterpret_cast<uint8_t *>(op.to);
  uint32_t round_rank = NarrowU32(op.round_rank);
  uint32_t iter_num = NarrowU32(op.iter_num);
  uint32_t iter_size = NarrowU32(op.iter_size);
  uint32_t iter_tail = NarrowU32(op.iter_tail);
  uint32_t tile_stride = NarrowU32(op.tile_stride);
  uint32_t xn = NarrowU32(op.xn);
  uint32_t pad_size = NarrowU32(op.pad_size);
  if (round_rank > 0) {
    tile_idx = RoundUpTileIdxGpu<kGpuRoundCacheStore>(tile_idx, round_rank, pc + vStore::ROUND_OFFSET);
    if (tile_idx == kInvalidTile) {
      return;
    }
  }
  if (is_last) {
    if (iter_num == 1) {
      iter_size = iter_tail;
    } else {
      iter_num = iter_tail;
    }
  }
  uint64_t tile_off = static_cast<uint64_t>(tile_stride) * tile_idx;
  uint8_t *dst = to + tile_off;
  CopyUbToGm(ctx, dst, xn, iter_num, iter_size, pad_size >> 5, 0);
}

__device__ __forceinline__ void ClearPadOp(VmCtx &ctx, uint64_t *pc, uint64_t head, bool is_last) {
  vClearPad op;
  vClearPad::Decode(pc, head, op);
  uint32_t iter_num = NarrowU32(op.iter_num);
  uint32_t iter_size = NarrowU32(op.iter_size);
  uint32_t iter_stride = NarrowU32(op.iter_stride);
  uint32_t simd_width = NarrowU32(op.simd_width);
  uint32_t iter_tail = NarrowU32(op.iter_tail);
  if (is_last && iter_tail != 0) {
    iter_size = iter_tail;
    iter_stride = iter_tail + simd_width - (iter_tail % simd_width);
  }
  if (iter_num == 0 || iter_size >= iter_stride) {
    return;
  }
  union {
    uint32_t u;
    float f;
  } scalar;
  scalar.u = static_cast<uint32_t>(op.scalar);
  float *xd = UBPtr<float>(ctx, NarrowU32(op.xd));
  uint32_t pad = iter_stride - iter_size;
  ParallelFor(iter_num * pad, [&](uint32_t idx) {
    uint32_t row = idx / pad;
    uint32_t col = idx - row * pad;
    xd[row * iter_stride + iter_size + col] = scalar.f;
  });
}

__device__ __forceinline__ void ReduceSumXOp(VmCtx &ctx, uint64_t *pc, uint64_t head, bool is_last) {
  vReduceX op;
  vReduceX::Decode(pc, head, op);
  vReduceX::DecodeBlock(pc, op);
  uint32_t red_size = NarrowU32(is_last ? op.red_tail : op.red_size);
  uint32_t dup_size = NarrowU32(op.dup_size);
  uint32_t dup_block = NarrowU32(op.dup_block);
  uint32_t dup_pad = NarrowU32(op.dup_pad);
  if (red_size == 0 || dup_size == 0 || dup_block == 0) {
    return;
  }
  float *xd = UBPtr<float>(ctx, NarrowU32(op.xd));
  const float *xn = UBCPtr<float>(ctx, NarrowU32(op.xn));
  uint32_t dst_stride = dup_block + dup_pad;
  uint32_t src_stride = red_size * dup_block;
  uint32_t iter_count = dup_size / dup_block;
  uint32_t group_count = iter_count * dup_block;
  if (group_count == 1 && red_size >= 64) {
    double sum = 0.0;
    for (uint32_t j = threadIdx.x; j < red_size; j += blockDim.x) {
      sum += static_cast<double>(xn[j]);
    }
    sum = BlockReduceSum(sum, reinterpret_cast<double *>(xd + 2));
    if (threadIdx.x == 0) {
      xd[0] = static_cast<float>(sum);
    }
    __syncthreads();
    return;
  }
  uint32_t warp_count = (blockDim.x + 31u) >> 5;
  if (group_count > 1 && red_size >= 64) {
    uint32_t warp = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    for (uint32_t idx = warp; idx < group_count; idx += warp_count) {
      uint32_t k = idx / dup_block;
      uint32_t i = idx - k * dup_block;
      const float *src = xn + k * src_stride + i * red_size;
      double sum = 0.0;
      for (uint32_t j = lane; j < red_size; j += 32u) {
        sum += static_cast<double>(src[j]);
      }
      sum = WarpReduceSum(sum);
      if (lane == 0) {
        xd[k * dst_stride + i] = static_cast<float>(sum);
      }
    }
    return;
  }
  ParallelFor(group_count, [&](uint32_t idx) {
    uint32_t k = idx / dup_block;
    uint32_t i = idx - k * dup_block;
    const float *src = xn + k * src_stride + i * red_size;
    double sum = 0.0;
    for (uint32_t j = 0; j < red_size; ++j) {
      sum += static_cast<double>(src[j]);
    }
    xd[k * dst_stride + i] = static_cast<float>(sum);
  });
}

__device__ __forceinline__ void ReduceSumYOp(VmCtx &ctx, uint64_t *pc, uint64_t head, bool is_last) {
  vReduceY op;
  vReduceY::Decode(pc, head, op);
  uint32_t red_size = NarrowU32(is_last ? op.red_tail : op.red_size);
  uint32_t iter_size = NarrowU32(op.iter_size);
  uint32_t dup_num = NarrowU32(op.dup_num);
  if (red_size == 0 || iter_size == 0 || dup_num == 0) {
    return;
  }
  float *xd = UBPtr<float>(ctx, NarrowU32(op.xd));
  const float *xn = UBCPtr<float>(ctx, NarrowU32(op.xn));
  uint32_t count = dup_num * iter_size;
  if (red_size == 1) {
    ParallelFor(count, [&](uint32_t idx) { xd[idx] = xn[idx]; });
    return;
  }
  uint32_t src_stride = red_size * iter_size;
  ParallelFor(count, [&](uint32_t idx) {
    uint32_t d = idx / iter_size;
    uint32_t t = idx - d * iter_size;
    const float *src = xn + d * src_stride + t;
    double sum = 0.0;
    for (uint32_t k = 0; k < red_size; ++k) {
      sum += static_cast<double>(src[k * iter_size]);
    }
    xd[idx] = static_cast<float>(sum);
  });
}

__device__ __forceinline__ bool AtomicCumInRound(uint32_t tile_idx, uint32_t round_rank, const uint64_t *rounds,
                                                 bool is_begin) {
  if (round_rank == 1) {
    uint32_t r1 = static_cast<uint32_t>(rounds[0]);
    return (tile_idx % r1) == 0 || is_begin;
  }
  if (round_rank == 3) {
    uint32_t r2 = static_cast<uint32_t>(rounds[0] >> 32);
    uint32_t r3 = static_cast<uint32_t>(rounds[1]);
    uint32_t p2 = tile_idx % (r2 * r3);
    uint32_t p3 = p2 / r3;
    return p3 * r3 == p2 || is_begin;
  }
  return true;
}

__device__ __forceinline__ void AtomicCumOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint32_t tile_idx,
                                            bool is_begin) {
  vAtomicCum op;
  vAtomicCum::Decode(pc, head, op);
  uint32_t count = NarrowU32(op.count);
  if (count == 0) {
    return;
  }
  float *xd = UBPtr<float>(ctx, NarrowU32(op.xd));
  const float *xn = UBCPtr<float>(ctx, NarrowU32(op.xn));
  uint32_t round_rank = NarrowU32(op.round_rank);
  if (round_rank != 0 && !AtomicCumInRound(tile_idx, round_rank, pc + vAtomicCum::ROUND_OFFSET, is_begin)) {
    ParallelFor(count, [&](uint32_t i) { xd[i] += xn[i]; });
  } else {
    ParallelFor(count, [&](uint32_t i) { xd[i] = xn[i]; });
  }
}

__device__ __forceinline__ void StoreAtomicOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint32_t tile_idx, bool is_end,
                                              bool is_last) {
  vStoreAtomic op;
  vStoreAtomic::Decode(pc, head, op);
  uint32_t round_rank = NarrowU32(op.round_rank);
  if (round_rank != 0) {
    uint32_t mapped = RoundUpTileIdxGpu<kGpuRoundStoreAtomic>(tile_idx, round_rank, pc + vStoreAtomic::ROUND_OFFSET);
    if (mapped == kInvalidTile) {
      if (!is_end) {
        return;
      }
      mapped = RoundUpTileIdxGpu<kGpuRoundCacheNone>(tile_idx, round_rank, pc + vStoreAtomic::ROUND_OFFSET);
    }
    tile_idx = mapped;
  }
  if (op.red_op != V_RED_SUM) {
    return;
  }
  if (is_last) {
    op.iter_num = op.iter_tail;
  }
  uint32_t iter_num = NarrowU32(op.iter_num);
  uint32_t iter_size = NarrowU32(op.iter_size);
  uint32_t pad_size = NarrowU32(op.pad_size);
  if (iter_num == 0 || iter_size < sizeof(float)) {
    return;
  }
  uint32_t elem_count = iter_size / sizeof(float);
  uint32_t src_row = iter_size + pad_size;
  const uint8_t *src = ctx.ub + NarrowU32(op.xn);
  uint8_t *dst = reinterpret_cast<uint8_t *>(op.to) + static_cast<uint64_t>(tile_idx) * op.tile_stride;
  ParallelFor(iter_num * elem_count, [&](uint32_t idx) {
    uint32_t row = idx / elem_count;
    uint32_t col = idx - row * elem_count;
    float value = reinterpret_cast<const float *>(src + row * src_row)[col];
    atomicAdd(reinterpret_cast<float *>(dst + row * iter_size) + col, value);
  });
}

// Keep the packed arithmetic temporaries out of the general interpreter's live register set.
__device__ __noinline__ void PointwiseOp(const uint64_t *pc, uint64_t head, uint32_t tile_idx) {
  const auto table_offset = (head >> V_HEAD_EXT_OFFSET) & V_HEAD_EXT_MASK;
  const auto &program = *reinterpret_cast<const PointwiseProgram *>(pc + table_offset);
  for (uint32_t lane = threadIdx.x; lane < kPointwiseThreads; lane += blockDim.x) {
    RunPointwise(program, tile_idx * kPointwiseThreads + lane);
  }
}

__device__ __forceinline__ void RunOp(VmCtx &ctx, uint64_t *pc, uint64_t head, uint32_t tile_idx, bool is_begin,
                                      bool is_end, bool is_last) {
  uint32_t id = head >> V_HEAD_ID_OFFSET;
  switch (id) {
    case F_LOAD:
      LoadOp(ctx, pc, head, tile_idx, is_last);
      break;
    case F_STORE:
      StoreOp(ctx, pc, head, tile_idx, is_last);
      break;
    case F_STORE_ATOMIC:
      StoreAtomicOp(ctx, pc, head, tile_idx, is_end, is_last);
      break;
    case F_LOAD_DUMMY:
      break;
    case F_NOP:
      break;
    case F_COPY: {
      vCopy op;
      vCopy::Decode(pc, head, op);
      CopyOp(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.lenburst));
      break;
    }
    case F_BROADCAST_S: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint32_t>(ctx, NarrowU32(op.xd), NarrowU32(op.count), static_cast<uint32_t>(op.scalar));
      break;
    }
    case F_BROADCAST_S_B8: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint8_t>(ctx, NarrowU32(op.xd), NarrowU32(op.count), static_cast<uint8_t>(op.scalar));
      break;
    }
    case F_BROADCAST_S_B16: {
      vBroadcastS op;
      vBroadcastS::Decode(pc, head, op);
      BroadcastS<uint16_t>(ctx, NarrowU32(op.xd), NarrowU32(op.count), static_cast<uint16_t>(op.scalar));
      break;
    }
    case F_BROADCAST_X_B8: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint8_t>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count),
                          NarrowU32(op.lead_num), NarrowU32(op.iter_num), NarrowU32(op.lead_pad));
      break;
    }
    case F_BROADCAST_X_B16: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint16_t>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count),
                           NarrowU32(op.lead_num), NarrowU32(op.iter_num), NarrowU32(op.lead_pad));
      break;
    }
    case F_BROADCAST_X_B32: {
      vBroadcastX op;
      vBroadcastX::Decode(pc, head, op);
      BroadcastX<uint32_t>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count),
                           NarrowU32(op.lead_num), NarrowU32(op.iter_num), NarrowU32(op.lead_pad));
      break;
    }
    case F_BROADCAST_Y: {
      vBroadcastY op;
      vBroadcastY::Decode(pc, head, op);
      BroadcastY(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.iter_num),
                  NarrowU32(op.dup_num), NarrowU32(op.dup_stride));
      break;
    }
    case F_CAST_FP32_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF32ToF16{});
      break;
    }
    case F_CAST_FP32_TO_BF16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<BF16Tag, F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF32ToBF16{});
      break;
    }
    case F_CAST_FP32_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF32ToI32{});
      break;
    }
    case F_CAST_FP16_TO_BOOL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<BoolTag, F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF16ToBool{});
      break;
    }
    case F_CAST_FP16_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF16ToF32{});
      break;
    }
    case F_CAST_FP16_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtF16ToI32{});
      break;
    }
    case F_CAST_BF16_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtBF16ToF32{});
      break;
    }
    case F_CAST_BF16_TO_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<I32Tag, BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtBF16ToI32{});
      break;
    }
    case F_CAST_INT32_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtI32ToF16{});
      break;
    }
    case F_CAST_INT32_TO_FP32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F32Tag, I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtI32ToF32{});
      break;
    }
    case F_CAST_BOOL_TO_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      CastOp<F16Tag, BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CvtBoolToF16{});
      break;
    }
    case F_SQRT: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), SqrtOp{});
      break;
    }
    case F_SQRT_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), SqrtOp{});
      break;
    }
    case F_ABS: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), AbsFOp{});
      break;
    }
    case F_ABS_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), AbsFOp{});
      break;
    }
    case F_ABS_INT32: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), AbsIOp{});
      break;
    }
    case F_LOG: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), LogOp{});
      break;
    }
    case F_LOG_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), LogOp{});
      break;
    }
    case F_EXP: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), ExpOp{});
      break;
    }
    case F_EXP_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), ExpOp{});
      break;
    }
    case F_ROUND: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), RoundOp{});
      break;
    }
    case F_FLOOR: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), FloorOp{});
      break;
    }
    case F_CEIL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), CeilOp{});
      break;
    }
    case F_TRUNC: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), TruncOp{});
      break;
    }
    case F_ISFINITE: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), IsFiniteOp{});
      break;
    }
    case F_ISFINITE_FP16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), IsFiniteOp{});
      break;
    }
    case F_ISFINITE_BF16: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), IsFiniteOp{});
      break;
    }
    case F_LOGICAL_NOT_BOOL: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), NotBoolOp{});
      break;
    }
    case F_ADDS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), AddOp{});
      break;
    }
    case F_ADDS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), AddOp{});
      break;
    }
    case F_ADDS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), AddOp{});
      break;
    }
    case F_ADDS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<int32_t>(pc), AddOp{});
      break;
    }
    case F_MULS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), MulOp{});
      break;
    }
    case F_MULS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MulOp{});
      break;
    }
    case F_MULS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MulOp{});
      break;
    }
    case F_MULS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<int32_t>(pc), MulOp{});
      break;
    }
    case F_DIVS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), DivOp{});
      break;
    }
    case F_DIVS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), DivOp{});
      break;
    }
    case F_SDIV: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), DivLeftOp{});
      break;
    }
    case F_SDIV_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), DivLeftOp{});
      break;
    }
    case F_MAXS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), MaxOp{});
      break;
    }
    case F_MAXS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MaxOp{});
      break;
    }
    case F_MAXS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MaxOp{});
      break;
    }
    case F_MAXS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<int32_t>(pc), MaxOp{});
      break;
    }
    case F_MINS: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<float>(pc), MinOp{});
      break;
    }
    case F_MINS_FP16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MinOp{});
      break;
    }
    case F_MINS_BF16: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), BF16ToF32(vBinaryS::GetScalar<uint16_t>(pc)), MinOp{});
      break;
    }
    case F_MINS_INT32: {
      vBinaryS op;
      vBinaryS::Decode(pc, head, op);
      BinarySOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vBinaryS::GetScalar<int32_t>(pc), MinOp{});
      break;
    }
    case F_CMPS: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vCompareS::GetScalar<float>(pc), NarrowU32(op.type));
      break;
    }
    case F_CMPS_FP16: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), F16ToF32(vCompareS::GetScalar<uint16_t>(pc)), NarrowU32(op.type));
      break;
    }
    case F_CMPS_BF16: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), BF16ToF32(vCompareS::GetScalar<uint16_t>(pc)), NarrowU32(op.type));
      break;
    }
    case F_CMPS_INT32: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vCompareS::GetScalar<int32_t>(pc), NarrowU32(op.type));
      break;
    }
    case F_CMPS_BOOL: {
      vCompareS op;
      vCompareS::Decode(pc, head, op);
      CmpSOp<BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), vCompareS::GetScalar<uint8_t>(pc), NarrowU32(op.type));
      break;
    }
    case F_ADD: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), AddOp{});
      break;
    }
    case F_ADD_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), AddOp{});
      break;
    }
    case F_ADD_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), AddOp{});
      break;
    }
    case F_ADD_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), AddOp{});
      break;
    }
    case F_SUB: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), SubOp{});
      break;
    }
    case F_SUB_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), SubOp{});
      break;
    }
    case F_SUB_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), SubOp{});
      break;
    }
    case F_SUB_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), SubOp{});
      break;
    }
    case F_MUL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MulOp{});
      break;
    }
    case F_MUL_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MulOp{});
      break;
    }
    case F_MUL_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MulOp{});
      break;
    }
    case F_MUL_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MulOp{});
      break;
    }
    case F_DIV: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), DivOp{});
      break;
    }
    case F_DIV_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), DivOp{});
      break;
    }
    case F_MIN: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MinOp{});
      break;
    }
    case F_MIN_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MinOp{});
      break;
    }
    case F_MIN_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MinOp{});
      break;
    }
    case F_MIN_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MinOp{});
      break;
    }
    case F_MAX: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MaxOp{});
      break;
    }
    case F_MAX_FP16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MaxOp{});
      break;
    }
    case F_MAX_BF16: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MaxOp{});
      break;
    }
    case F_MAX_INT32: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), MaxOp{});
      break;
    }
    case F_CMP: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), static_cast<int>(NarrowU32(op.type)));
      break;
    }
    case F_CMP_FP16: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), static_cast<int>(NarrowU32(op.type)));
      break;
    }
    case F_CMP_BF16: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), static_cast<int>(NarrowU32(op.type)));
      break;
    }
    case F_CMP_INT32: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), static_cast<int>(NarrowU32(op.type)));
      break;
    }
    case F_CMP_BOOL: {
      vCompare op;
      vCompare::Decode(pc, head, op);
      CmpOp<BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), static_cast<int>(NarrowU32(op.type)));
      break;
    }
    case F_SEL: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.cond), NarrowU32(op.count));
      break;
    }
    case F_SEL_FP16: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<F16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.cond), NarrowU32(op.count));
      break;
    }
    case F_SEL_BF16: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<BF16Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.cond), NarrowU32(op.count));
      break;
    }
    case F_SEL_INT32: {
      vSelect op;
      vSelect::Decode(pc, head, op);
      SelectOp<I32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.cond), NarrowU32(op.count));
      break;
    }
    case F_POW: {
      vBinaryWS op;
      vBinaryWS::Decode(pc, head, op);
      BinaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), PowOp{});
      break;
    }
    case F_LOGICAL_AND_BOOL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), AndOp{});
      break;
    }
    case F_LOGICAL_OR_BOOL: {
      vBinary op;
      vBinary::Decode(pc, head, op);
      BinaryOp<BoolTag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.xm), NarrowU32(op.count), OrOp{});
      break;
    }
    case F_CLR_PAD:
      ClearPadOp(ctx, pc, head, is_last);
      break;
    case F_RSUM_X:
      ReduceSumXOp(ctx, pc, head, is_last);
      break;
    case F_RSUM_Y:
      ReduceSumYOp(ctx, pc, head, is_last);
      break;
    case F_ATOMICCUM:
      AtomicCumOp(ctx, pc, head, tile_idx, is_begin);
      break;
    case F_POINTWISE:
      PointwiseOp(pc, head, tile_idx);
      break;
    case F_FMA: {
      vFma op;
      vFma::Decode(pc, head, op);
      FmaOp(ctx, op);
      break;
    }
    case F_TANH: {
      vUnary op;
      vUnary::Decode(pc, head, op);
      UnaryOp<F32Tag>(ctx, NarrowU32(op.xd), NarrowU32(op.xn), NarrowU32(op.count), TanhOp{});
      break;
    }
    default:
      TrapUnsupported(id, head, pc);
      break;
  }
}

}  // namespace

__global__ void dvm_vm_gpu_packed_code(__grid_constant__ const VmParam param) {
  const auto &program = *reinterpret_cast<const PointwiseProgram *>(param.code + PointwiseCode::kProgram);
  RunPointwise(program, blockIdx.x * kPointwiseThreads + threadIdx.x);
}

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
  uint32_t bid = blockIdx.x;
  uint32_t bnum = gridDim.x;
  uint32_t ktype = static_cast<uint32_t>(entry & V_ENTRY_MASK_TYPE);

  if (ktype == V_ENTRY_TYPE_P) {
    uint32_t lookup_offset = static_cast<uint32_t>((entry >> V_ENTRY_P_AIV_LKUP_OFFSET) & V_ENTRY_P_LKUP_MASK);
    uint8_t prog_offset = base_body[lookup_offset + bid];
    if (prog_offset == V_ENTRY_P_LKUP_INVALID) {
      return;
    }
    const uint64_t *prog = reinterpret_cast<const uint64_t *>(base_body) + prog_offset;
    entry = prog[0];
    uint64_t data = prog[1];
    uint32_t b_begin = static_cast<uint32_t>(data & 0xffffu);
    uint32_t b_num = static_cast<uint32_t>((data >> 16) & 0xffffu);
    uint32_t body_offset = static_cast<uint32_t>((data >> 32) & 0xffffu);
    bid -= b_begin;
    bnum = b_num;
    base_body += body_offset;
    ktype = static_cast<uint32_t>(entry & V_ENTRY_MASK_TYPE);
  }

  uint64_t *body = reinterpret_cast<uint64_t *>(base_body);
  uint32_t body_tile =
    static_cast<uint32_t>((entry >> V_ENTRY_V_TILE_BODY_OFFSET) & ((1ull << V_ENTRY_V_TILE_BODY_BITS) - 1ull));
  uint32_t tail_blocks =
    static_cast<uint32_t>((entry >> V_ENTRY_V_TILE_TAIL_OFFSET) & ((1ull << V_ENTRY_V_TILE_TAIL_BITS) - 1ull));
  // The first bnum - tail_blocks blocks handle body_tile tiles, the rest handle body_tile - 1.
  uint32_t full_blocks = bnum - tail_blocks;
  uint32_t count = body_tile;
  uint32_t tile_begin = bid * body_tile;
  if (bid >= full_blocks) {
    count = body_tile - 1;
    tile_begin = full_blocks * body_tile + (bid - full_blocks) * (body_tile - 1);
  }
  uint32_t tile_idx = tile_begin;
  bool tail = (bid == bnum - 1);

  while (count != 0) {
    bool is_begin = tile_idx == tile_begin;
    bool is_end = (count == 1);
    bool is_last = tail && (count == 1);
    uint64_t *pc = body;
    for (uint64_t head = *pc; head != 0; head = *pc) {
      bcodeptr_t op_pc = pc;
      pc += (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
      if ((head >> V_HEAD_BAR_FLAG_OFFSET) & 0x1ull) {
        __syncthreads();
      }
      RunOp(ctx, op_pc, head, tile_idx, is_begin, is_end, is_last);
    }
    ++tile_idx;
    --count;
  }
}


inline constexpr uint32_t THREAD_ORDER_STEP = 64;
inline constexpr uint32_t MAX_BLOCK_PER_SM = 8;

struct BlockBudget {
  uint32_t shared_limit;
  uint32_t max_threads;
};

struct GpuSystem {
  uint32_t sm_num;
  uint32_t max_block_per_sm;
  BlockBudget budgets[MAX_BLOCK_PER_SM];
};

static GpuSystem g_gpu;

int LaunchVM(const VmParam &param, uint32_t block_dim, uint32_t threads, uint32_t shared_bytes, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (PointwiseCode::Matches(param, threads, shared_bytes)) {
    dvm_vm_gpu_packed_code<<<block_dim, threads, 0, cuda_stream>>>(param);
  } else {
    dvm_vm_gpu<<<block_dim, threads, shared_bytes, cuda_stream>>>(param);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

uint64_t SelectBlockThread(uint64_t op_num, uint64_t tile_num, uint64_t tile_size, uint32_t shared_bytes) {
  auto ceil_div = [](auto n, auto r) { return (n - 1) / r + 1; };
  constexpr uint64_t kDispatchCost = 1;
  constexpr uint64_t kThreadPenalty = 1;
  constexpr uint64_t kWarpSize = 32;
  const uint32_t sm_num = g_gpu.sm_num;
  const uint64_t tile_size_align = ceil_div(tile_size, THREAD_ORDER_STEP) * THREAD_ORDER_STEP;
  uint64_t best_cost = static_cast<uint64_t>(-1);
  uint64_t best_block_num = 0;
  uint64_t best_thread = 0;
  uint32_t max_block_per_sm = std::min(g_gpu.max_block_per_sm, ceil_div(static_cast<uint32_t>(tile_num), sm_num));
  for (uint32_t i = 0; i < max_block_per_sm && shared_bytes <= g_gpu.budgets[i].shared_limit; ++i) {
    uint64_t block_num = std::min(static_cast<uint64_t>(sm_num) * (i + 1), tile_num);
    uint64_t tile_per_block = ceil_div(tile_num, block_num);
    uint64_t max_threads = std::min(static_cast<uint64_t>(g_gpu.budgets[i].max_threads), tile_size_align);
    for (uint64_t threads = THREAD_ORDER_STEP; threads <= max_threads; threads *= 2) {
      uint64_t cost = (op_num * (ceil_div(tile_size, threads) + kDispatchCost) + kThreadPenalty * threads / kWarpSize) * tile_per_block;
      if (cost < best_cost || (cost == best_cost && block_num > best_block_num)) {
        best_cost = cost;
        best_block_num = block_num;
        best_thread = threads;
      }
    }
  }
  return best_block_num << 32 | best_thread;
}

int InitVM(int dev_id, SysConfig &conf) {
  cudaDeviceProp prop;
  auto err = cudaGetDeviceProperties(&prop, dev_id);
  if (err != 0) {
    return static_cast<int>(err);
  }
  g_gpu.sm_num = static_cast<uint64_t>(prop.multiProcessorCount);
  uint32_t shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
  uint32_t shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  uint32_t max_threads_per_block = prop.maxThreadsPerBlock;
  uint32_t max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  uint32_t max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
  err = cudaFuncSetAttribute(dvm_vm_gpu, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_mem_per_block_optin));
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  g_gpu.max_block_per_sm = std::min(max_blocks_per_sm, MAX_BLOCK_PER_SM);
  for (uint32_t i = 0; i < g_gpu.max_block_per_sm; ++i) {
    uint32_t blocks_per_sm = i + 1;
    uint32_t budget = std::min(shared_mem_per_sm / blocks_per_sm, shared_mem_per_block_optin) & ~0x3ffull;
    int max_threads = std::min(max_threads_per_block, max_threads_per_sm / blocks_per_sm);
    g_gpu.budgets[i].shared_limit = budget;
    g_gpu.budgets[i].max_threads = max_threads;
    for (int threads = THREAD_ORDER_STEP; threads <= max_threads; threads += THREAD_ORDER_STEP) {
      int occ = 0;
      err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dvm_vm_gpu, threads, budget);
      if (err != cudaSuccess || occ != blocks_per_sm) {
        g_gpu.budgets[i].max_threads = threads - THREAD_ORDER_STEP;
        break;
      }
    }
  }
  uint32_t default_block_per_sm = std::min(4u, g_gpu.max_block_per_sm);
  conf.core_num = g_gpu.sm_num * default_block_per_sm ;
  conf.local_mem_size = g_gpu.budgets[default_block_per_sm - 1].shared_limit;
  conf.default_threads = g_gpu.budgets[default_block_per_sm - 1].max_threads;
  return 0;
}
}  // namespace dvm::gpu
