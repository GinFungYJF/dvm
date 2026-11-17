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
#include "fusion/emit/emit.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Context.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <cmath>
#include <utility>

#include "adapter/adapter.h"

namespace dvm_torch {
namespace {

bool SupportedDtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
    case at::kInt:
    case at::kLong:
      return true;
    default:
      return false;
  }
}

at::Tensor AllocateLike(const std::vector<int64_t> &shape, at::ScalarType dtype,
                        const at::Tensor &device_from) {
  auto opts = at::TensorOptions().dtype(dtype).device(device_from.device());
  // Allocation dispatches through aten like any other op. Flushing here would
  // free the graph nodes the caller is still holding.
  GraphManager::NoFlushScope no_flush;
  return GetAdapter().Allocate(at::IntArrayRef(shape), opts);
}

bool CanBroadcast(const at::Tensor &a, const at::Tensor &b) {
  try {
    (void)at::infer_size(a.sizes(), b.sizes());
    return true;
  } catch (const c10::Error &) {
    return false;
  }
}

/**
 * A Python scalar keeps the tensor dtype for floating tensors, so `fp16 * 2.0`
 * stays fp16. Integers are excluded: the VM materialises the constant as a
 * float, and torch promotes `int32 / 2` to float even for integral values, so
 * mixing an integer tensor with a scalar cannot be fused without lying about
 * the result.
 */
bool ScalarTypeIsFusable(const at::Tensor &t) { return IsFloatDtype(t); }

}  // namespace

bool IsWrappedScalar(const at::Tensor &t) {
  // Python scalars arrive as 0-dim CPU tensors ("wrapped numbers").
  return t.defined() && t.dim() == 0 && t.device().is_cpu();
}

bool Eligible(const at::Tensor &t) {
  if (!t.defined()) {
    return false;
  }
  if (!t.is_cuda() || !t.is_contiguous()) {
    return false;
  }
  // dvm::DimArray and ShapeWithRef store at most ten dimensions.
  if (t.dim() > 10) {
    return false;
  }
  if (t.scalar_type() == at::kLong) {
    // GPU VM has no int64 arithmetic path; only loads/stores are defined.
    return false;
  }
  return SupportedDtype(t.scalar_type());
}

bool IsFloatDtype(const at::Tensor &t) {
  switch (t.scalar_type()) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
      return true;
    default:
      return false;
  }
}

bool IsIntegerDtype(const at::Tensor &t) { return t.scalar_type() == at::kInt; }

std::vector<int64_t> BroadcastShape(const at::Tensor &a, const at::Tensor &b) {
  auto sizes = at::infer_size(a.sizes(), b.sizes());
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

EmitResult EmitBinary(EmitCtx &ctx, dvm::BinaryType op, const at::Tensor &a,
                      const at::Tensor &b) {
  if (!Eligible(a) || !Eligible(b)) {
    return std::nullopt;
  }
  if (a.scalar_type() != b.scalar_type()) {
    // DVM arithmetic does not promote; let ATen do the promotion.
    return std::nullopt;
  }
  if (op == dvm::kDiv && IsIntegerDtype(a)) {
    // ATen true division on integers produces float32; the VM would truncate.
    return std::nullopt;
  }
  if (!CanBroadcast(a, b)) {
    return std::nullopt;
  }
  std::vector<int64_t> out_shape = BroadcastShape(a, b);

  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *lb = ctx.k->Read(b);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kAdd: return ctx.k->GetKernel().Binary<dvm::kAdd>(la, lb);
      case dvm::kSub: return ctx.k->GetKernel().Binary<dvm::kSub>(la, lb);
      case dvm::kMul: return ctx.k->GetKernel().Binary<dvm::kMul>(la, lb);
      case dvm::kDiv: return ctx.k->GetKernel().Binary<dvm::kDiv>(la, lb);
      case dvm::kMaximum: return ctx.k->GetKernel().Binary<dvm::kMaximum>(la, lb);
      case dvm::kMinimum: return ctx.k->GetKernel().Binary<dvm::kMinimum>(la, lb);
      case dvm::kPow: return ctx.k->GetKernel().Binary<dvm::kPow>(la, lb);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }

  if ((op == dvm::kMaximum || op == dvm::kMinimum) && IsFloatDtype(a)) {
    // The VM comparison selects b when a is NaN. ATen propagates either NaN.
    auto &kernel = ctx.k->GetKernel();
    auto nan = kernel.Binary<dvm::kNotEqual>(la, la);
    res = kernel.Select(nan, la, res);
  }

  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitBinaryScalar(EmitCtx &ctx, dvm::BinaryType op, const at::Tensor &a,
                            const at::Scalar &b) {
  if (!Eligible(a) || !ScalarTypeIsFusable(a)) {
    return std::nullopt;
  }
  // A Python scalar does not participate in dtype promotion: `fp16 * 2.0` stays
  // fp16, so the constant is materialised at the tensor's own dtype.
  const float constant = b.to<float>();
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kAdd: return ctx.k->GetKernel().Binary<dvm::kAdd>(la, constant);
      case dvm::kSub: return ctx.k->GetKernel().Binary<dvm::kSub>(la, constant);
      case dvm::kMul: return ctx.k->GetKernel().Binary<dvm::kMul>(la, constant);
      case dvm::kDiv: return ctx.k->GetKernel().Binary<dvm::kDiv>(la, constant);
      case dvm::kMaximum: return ctx.k->GetKernel().Binary<dvm::kMaximum>(la, constant);
      case dvm::kMinimum: return ctx.k->GetKernel().Binary<dvm::kMinimum>(la, constant);
      case dvm::kPow: return ctx.k->GetKernel().Binary<dvm::kPow>(la, constant);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitBinaryScalarLhs(EmitCtx &ctx, dvm::BinaryType op, const at::Scalar &a,
                               const at::Tensor &b) {
  if (!Eligible(b) || !ScalarTypeIsFusable(b)) {
    return std::nullopt;
  }
  const float constant = a.to<float>();
  dvm::NDObject *lb = ctx.k->Read(b);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kAdd: return ctx.k->GetKernel().Binary<dvm::kAdd>(constant, lb);
      case dvm::kSub: return ctx.k->GetKernel().Binary<dvm::kSub>(constant, lb);
      case dvm::kMul: return ctx.k->GetKernel().Binary<dvm::kMul>(constant, lb);
      case dvm::kDiv: return ctx.k->GetKernel().Binary<dvm::kDiv>(constant, lb);
      case dvm::kPow: return ctx.k->GetKernel().Binary<dvm::kPow>(constant, lb);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }
  std::vector<int64_t> out_shape(b.sizes().begin(), b.sizes().end());
  at::Tensor out = AllocateLike(out_shape, b.scalar_type(), b);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitUnary(EmitCtx &ctx, dvm::UnaryType op, const at::Tensor &a) {
  if (!Eligible(a)) {
    return std::nullopt;
  }
  // Two separate constraints on integers:
  //  * sqrt/log/exp/reciprocal promote an int input to float in ATen, so fusing
  //    them would return the wrong dtype.
  //  * round/floor/ceil/trunc keep int32 in ATen, but the GPU VM has no INT32
  //    form of those opcodes and traps at runtime. Only kAbs has one.
  if (!IsFloatDtype(a) && op != dvm::kAbs) {
    return std::nullopt;
  }
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kSqrt: return ctx.k->GetKernel().Unary<dvm::kSqrt>(la);
      case dvm::kAbs: return ctx.k->GetKernel().Unary<dvm::kAbs>(la);
      case dvm::kLog: return ctx.k->GetKernel().Unary<dvm::kLog>(la);
      case dvm::kExp: return ctx.k->GetKernel().Unary<dvm::kExp>(la);
      case dvm::kReciprocal: return ctx.k->GetKernel().Unary<dvm::kReciprocal>(la);
      case dvm::kRound: return ctx.k->GetKernel().Unary<dvm::kRound>(la);
      case dvm::kFloor: return ctx.k->GetKernel().Unary<dvm::kFloor>(la);
      case dvm::kCeil: return ctx.k->GetKernel().Unary<dvm::kCeil>(la);
      case dvm::kTrunc: return ctx.k->GetKernel().Unary<dvm::kTrunc>(la);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitNeg(EmitCtx &ctx, const at::Tensor &a) {
  if (!Eligible(a)) {
    return std::nullopt;
  }
  // There is no unary minus in the VM; express it as 0 - x.
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *res = ctx.k->GetKernel().Binary<dvm::kSub>(0.0f, la);
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitRelu(EmitCtx &ctx, const at::Tensor &a) {
  if (!Eligible(a)) {
    return std::nullopt;
  }
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *res = ctx.k->GetKernel().Binary<dvm::kMaximum>(la, 0.0f);
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitRsqrt(EmitCtx &ctx, const at::Tensor &a) {
  if (!Eligible(a) || !IsFloatDtype(a)) {
    return std::nullopt;
  }
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *sq = ctx.k->GetKernel().Unary<dvm::kSqrt>(la);
  dvm::NDObject *res = ctx.k->GetKernel().Unary<dvm::kReciprocal>(sq);
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitSigmoid(EmitCtx &ctx, const at::Tensor &a) {
  if (!Eligible(a) || !IsFloatDtype(a)) {
    return std::nullopt;
  }
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *neg = ctx.k->GetKernel().Binary<dvm::kSub>(0.0f, la);
  dvm::NDObject *e = ctx.k->GetKernel().Unary<dvm::kExp>(neg);
  dvm::NDObject *den = ctx.k->GetKernel().Binary<dvm::kAdd>(e, 1.0f);
  dvm::NDObject *res = ctx.k->GetKernel().Unary<dvm::kReciprocal>(den);
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitTanh(EmitCtx &ctx, const at::Tensor &a) {
  if (!Eligible(a) || !IsFloatDtype(a)) {
    return std::nullopt;
  }
  // tanh(x) = 2 / (1 + exp(-2x)) - 1. Not bit-exact against ATen, so the caller
  // must check Caps::strict_accuracy before enabling it.
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *t2 = ctx.k->GetKernel().Binary<dvm::kMul>(la, 2.0f);
  dvm::NDObject *neg = ctx.k->GetKernel().Binary<dvm::kSub>(0.0f, t2);
  dvm::NDObject *e = ctx.k->GetKernel().Unary<dvm::kExp>(neg);
  dvm::NDObject *den = ctx.k->GetKernel().Binary<dvm::kAdd>(e, 1.0f);
  dvm::NDObject *r = ctx.k->GetKernel().Binary<dvm::kDiv>(2.0f, den);
  dvm::NDObject *res = ctx.k->GetKernel().Binary<dvm::kSub>(r, 1.0f);
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitPowScalar(EmitCtx &ctx, const at::Tensor &a, double exponent) {
  if (!Eligible(a)) {
    return std::nullopt;
  }
  if (!IsFloatDtype(a) && exponent < 0) {
    // A negative exponent on an integer base is an error in ATen; let it report.
    return std::nullopt;
  }
  dvm::NDObject *la = ctx.k->Read(a);
  dvm::NDObject *res = ctx.k->GetKernel().Binary<dvm::kPow>(la, static_cast<float>(exponent));
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, res);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitClamp(EmitCtx &ctx, const at::Tensor &a, const std::optional<c10::Scalar> &min,
                     const std::optional<c10::Scalar> &max) {
  if (!Eligible(a)) {
    return std::nullopt;
  }
  if (!min.has_value() && !max.has_value()) {
    return std::nullopt;
  }
  if (a.scalar_type() != at::kFloat) {
    // Only the float path is verified on the GPU VM.
    return std::nullopt;
  }
  dvm::NDObject *node = ctx.k->Read(a);
  if (min.has_value()) {
    node = ctx.k->GetKernel().Binary<dvm::kMaximum>(node, min->to<float>());
  }
  if (max.has_value()) {
    node = ctx.k->GetKernel().Binary<dvm::kMinimum>(node, max->to<float>());
  }
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  at::Tensor out = AllocateLike(out_shape, a.scalar_type(), a);
  ctx.k->Write(out, node);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitReduction(EmitCtx &ctx, const at::Tensor &a, at::OptionalIntArrayRef dims,
                         bool keepdim, std::optional<at::ScalarType> dtype, bool mean) {
  if (!Eligible(a) || !IsFloatDtype(a) || a.numel() == 0 ||
      at::globalContext().deterministicAlgorithms()) {
    return std::nullopt;
  }
  const at::ScalarType out_dtype = dtype.value_or(a.scalar_type());
  if (out_dtype != at::kFloat && out_dtype != at::kHalf && out_dtype != at::kBFloat16) {
    return std::nullopt;
  }

  // ATen treats both an omitted dim and an empty dim list as a full reduction.
  if (dims.has_value() && dims->empty()) {
    dims = std::nullopt;
  }
  std::bitset<at::dim_bitset_size> mask;
  try {
    mask = at::dim_list_to_bitset(dims, a.dim());
  } catch (const c10::Error &) {
    return std::nullopt;
  }
  std::vector<int64_t> axes;
  std::vector<int64_t> out_shape;
  int64_t count = 1;
  for (int64_t dim = 0; dim < a.dim(); ++dim) {
    if (mask[dim]) {
      axes.push_back(dim);
      count *= a.size(dim);
      if (keepdim) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(a.size(dim));
    }
  }

  auto &kernel = ctx.k->GetKernel();
  dvm::NDObject *node = ctx.k->Read(a);
  // An explicit dtype casts each input before reduction, including narrowing.
  node = kernel.Cast(node, ToDvmDtype(out_dtype));
  node = kernel.Cast(node, dvm::kFloat32);
  if (!axes.empty()) {
    node = ctx.k->ReduceSum(node, axes, keepdim);
  }
  if (mean && count != 1) {
    node = kernel.Binary<dvm::kDiv>(node, static_cast<float>(count));
  }
  node = kernel.Cast(node, ToDvmDtype(out_dtype));
  at::Tensor out = AllocateLike(out_shape, out_dtype, a);
  ctx.k->Write(out, node);
  ctx.k->AppendSignature(ctx.op_name);
  return out;
}

EmitResult EmitBinaryInplace(EmitCtx &ctx, dvm::BinaryType op, at::Tensor &self,
                             const at::Tensor &other) {
  if (!Eligible(self) || !Eligible(other)) {
    return std::nullopt;
  }
  if (self.scalar_type() != other.scalar_type()) {
    return std::nullopt;
  }
  if (!CanBroadcast(self, other)) {
    return std::nullopt;
  }
  // An in-place result must keep the destination shape: broadcasting self up
  // would change it.
  if (BroadcastShape(self, other) != std::vector<int64_t>(self.sizes().begin(), self.sizes().end())) {
    return std::nullopt;
  }

  dvm::NDObject *ls = ctx.k->Read(self);
  dvm::NDObject *lo = ctx.k->Read(other);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kAdd: return ctx.k->GetKernel().Binary<dvm::kAdd>(ls, lo);
      case dvm::kSub: return ctx.k->GetKernel().Binary<dvm::kSub>(ls, lo);
      case dvm::kMul: return ctx.k->GetKernel().Binary<dvm::kMul>(ls, lo);
      case dvm::kDiv: return ctx.k->GetKernel().Binary<dvm::kDiv>(ls, lo);
      case dvm::kMaximum: return ctx.k->GetKernel().Binary<dvm::kMaximum>(ls, lo);
      case dvm::kMinimum: return ctx.k->GetKernel().Binary<dvm::kMinimum>(ls, lo);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }
  ctx.k->WriteInplace(self, res);
  ctx.k->AppendSignature(std::string(ctx.op_name) + "!");
  return self;
}

EmitResult EmitBinaryInplaceScalar(EmitCtx &ctx, dvm::BinaryType op, at::Tensor &self,
                                   const at::Scalar &other) {
  if (!Eligible(self) || !ScalarTypeIsFusable(self)) {
    return std::nullopt;
  }
  const float constant = other.to<float>();
  dvm::NDObject *ls = ctx.k->Read(self);
  dvm::NDObject *res = [&]() -> dvm::NDObject * {
    switch (op) {
      case dvm::kAdd: return ctx.k->GetKernel().Binary<dvm::kAdd>(ls, constant);
      case dvm::kSub: return ctx.k->GetKernel().Binary<dvm::kSub>(ls, constant);
      case dvm::kMul: return ctx.k->GetKernel().Binary<dvm::kMul>(ls, constant);
      case dvm::kDiv: return ctx.k->GetKernel().Binary<dvm::kDiv>(ls, constant);
      default: return nullptr;
    }
  }();
  if (res == nullptr) {
    return std::nullopt;
  }
  ctx.k->WriteInplace(self, res);
  ctx.k->AppendSignature(std::string(ctx.op_name) + "!");
  return self;
}

EmitResult EmitReluInplace(EmitCtx &ctx, at::Tensor &self) {
  if (!Eligible(self)) {
    return std::nullopt;
  }
  dvm::NDObject *ls = ctx.k->Read(self);
  dvm::NDObject *res = ctx.k->GetKernel().Binary<dvm::kMaximum>(ls, 0.0f);
  ctx.k->WriteInplace(self, res);
  ctx.k->AppendSignature("relu_");
  return self;
}

}  // namespace dvm_torch
