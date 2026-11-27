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

#ifndef _DVM_TORCH_DISPATCH_OVERRIDES_H_
#define _DVM_TORCH_DISPATCH_OVERRIDES_H_

#include <ATen/core/Tensor.h>
#include <dvm.h>

#include <optional>

#include "dispatch/registry.h"
#include "fusion/core/graph.h"
#include "fusion/emit/emit.h"

namespace dvm_torch {

/**
 * @brief Convergence point for every capture path.
 *
 * Returns the deferred output when the op joined the graph. `nullopt` means the
 * op was refused, and refusing is side-effect free: the caller then flushes and
 * runs the captured original kernel.
 */
EmitResult DispatchBinary(const OpKey &key, dvm::BinaryType op, const at::Tensor &a,
                          const at::Tensor &b, const at::Scalar &alpha = 1);
EmitResult DispatchBinaryScalar(const OpKey &key, dvm::BinaryType op, const at::Tensor &a,
                                const at::Scalar &b, const at::Scalar &alpha = 1);
EmitResult DispatchExpandedDiv(const OpKey &key, const at::Tensor &input, const at::Scalar &divisor);
EmitResult DispatchDropoutBackward(const OpKey &key, const at::Tensor &grad, const at::Tensor &mask,
                                   double scale);
EmitResult DispatchTanhBackward(const OpKey &key, const at::Tensor &grad, const at::Tensor &output);
EmitResult DispatchEq(const OpKey &key, const at::Tensor &a, const at::Tensor &b);
EmitResult DispatchEqScalar(const OpKey &key, const at::Tensor &a, const at::Scalar &b);
EmitResult DispatchMaskedFill(const OpKey &key, const at::Tensor &self, const at::Tensor &mask,
                              const at::Scalar &value, bool inplace);
EmitResult DispatchBinaryInplace(const OpKey &key, dvm::BinaryType op, at::Tensor &self,
                                 const at::Tensor &other, const at::Scalar &alpha = 1);
EmitResult DispatchUnary(const OpKey &key, dvm::UnaryType op, const at::Tensor &a);
EmitResult DispatchNeg(const OpKey &key, const at::Tensor &a);
EmitResult DispatchRelu(const OpKey &key, const at::Tensor &a);
EmitResult DispatchRsqrt(const OpKey &key, const at::Tensor &a);
EmitResult DispatchSigmoid(const OpKey &key, const at::Tensor &a);
EmitResult DispatchPowScalar(const OpKey &key, const at::Tensor &a, double exponent);
EmitResult DispatchClamp(const OpKey &key, const at::Tensor &a,
                         const std::optional<c10::Scalar> &min,
                         const std::optional<c10::Scalar> &max);
EmitResult DispatchReluInplace(const OpKey &key, at::Tensor &self);
EmitResult DispatchReduction(const OpKey &key, const at::Tensor &a, at::OptionalIntArrayRef dims,
                             bool keepdim, std::optional<at::ScalarType> dtype, bool mean);

/** True when fusion is enabled and the op is allowed by level / disable_ops. */
bool OpEnabled(const OpKey &key);

/** Materialise the graph before an op that cannot join it. */
void FlushForFallback(FlushReason reason);

/** Install the unboxed wrappers plus the coverage guards. */
void InstallOverrides();

}  // namespace dvm_torch

#endif  // _DVM_TORCH_DISPATCH_OVERRIDES_H_
