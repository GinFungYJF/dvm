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

#ifndef _DVM_TORCH_FUSION_EMIT_EMIT_H_
#define _DVM_TORCH_FUSION_EMIT_EMIT_H_

#include <ATen/core/Tensor.h>
#include <dvm.h>

#include <optional>
#include <string>

#include "fusion/core/graph.h"

namespace dvm_torch {

/**
 * @brief Emitters for aten ops.
 *
 * An emitter returns the Pending output tensor on success. An empty `nullopt`
 * means "not fusible" and MUST be side-effect free: the caller then falls back
 * to the original kernel.
 */
using EmitResult = std::optional<at::Tensor>;

struct EmitCtx {
  GraphKernel *k{nullptr};
  const char *op_name{nullptr};
};

/** Shape/dtype checks shared by every emitter. */
bool Eligible(const at::Tensor &t);
bool IsFloatDtype(const at::Tensor &t);
bool IsIntegerDtype(const at::Tensor &t);
/** True for a 0-dim CPU tensor, which is how Python scalars reach the dispatcher. */
bool IsWrappedScalar(const at::Tensor &t);

/** Broadcasting output shape of two tensors, mirroring torch semantics. */
std::vector<int64_t> BroadcastShape(const at::Tensor &a, const at::Tensor &b);

EmitResult EmitBinary(EmitCtx &ctx, dvm::BinaryType op, const at::Tensor &a, const at::Tensor &b);
/** Scalar operand form: `x * 2.0`, `x + 1`, ... */
EmitResult EmitBinaryScalar(EmitCtx &ctx, dvm::BinaryType op, const at::Tensor &a,
                            const at::Scalar &b);
EmitResult EmitBinaryScalarLhs(EmitCtx &ctx, dvm::BinaryType op, const at::Scalar &a,
                               const at::Tensor &b);
EmitResult EmitUnary(EmitCtx &ctx, dvm::UnaryType op, const at::Tensor &a);
EmitResult EmitNeg(EmitCtx &ctx, const at::Tensor &a);
EmitResult EmitRelu(EmitCtx &ctx, const at::Tensor &a);
EmitResult EmitRsqrt(EmitCtx &ctx, const at::Tensor &a);
EmitResult EmitSigmoid(EmitCtx &ctx, const at::Tensor &a);
EmitResult EmitTanh(EmitCtx &ctx, const at::Tensor &a);
EmitResult EmitPowScalar(EmitCtx &ctx, const at::Tensor &a, double exponent);
EmitResult EmitClamp(EmitCtx &ctx, const at::Tensor &a, const std::optional<c10::Scalar> &min,
                     const std::optional<c10::Scalar> &max);
/** FP32 accumulation for sum and mean, with ATen dimension and dtype semantics. */
EmitResult EmitReduction(EmitCtx &ctx, const at::Tensor &a, at::OptionalIntArrayRef dims,
                         bool keepdim, std::optional<at::ScalarType> dtype, bool mean);

/** In-place variants: the result is `self` with its storage updated. */
EmitResult EmitBinaryInplace(EmitCtx &ctx, dvm::BinaryType op, at::Tensor &self,
                             const at::Tensor &other);
EmitResult EmitBinaryInplaceScalar(EmitCtx &ctx, dvm::BinaryType op, at::Tensor &self,
                                   const at::Scalar &other);
EmitResult EmitReluInplace(EmitCtx &ctx, at::Tensor &self);

}  // namespace dvm_torch

#endif  // _DVM_TORCH_FUSION_EMIT_EMIT_H_
