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
#include "dispatch/overrides.h"

#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Exception.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapter/adapter.h"
#include "dispatch/registry.h"

namespace dvm_torch {
namespace {

constexpr c10::DispatchKey kBackend = c10::DispatchKey::CUDA;

c10::DispatchKeySet BackendKeySet() { return c10::DispatchKeySet(kBackend); }

thread_local bool in_native_kernel = false;

/**
 * Native kernels may dispatch ATen ops and then read their outputs directly.
 * For example, softmax backward multiplies grad by output before launching a
 * CUDA kernel over the product. Those nested ops must finish producing their
 * values on the stream instead of returning deferred tensors.
 */
class NativeKernelScope {
 public:
  NativeKernelScope() : previous_(in_native_kernel) { in_native_kernel = true; }
  ~NativeKernelScope() { in_native_kernel = previous_; }

  NativeKernelScope(const NativeKernelScope &) = delete;
  NativeKernelScope &operator=(const NativeKernelScope &) = delete;

 private:
  bool previous_;
};

/**
 * @brief One captured original kernel.
 *
 * Stored by value: installing an override replaces the OperatorEntry, so a
 * reference into the dispatcher dangles. This was observed as a segfault.
 */
struct Slot {
  std::optional<c10::OperatorHandle> handle;
  c10::KernelFunction original;
  bool captured{false};

  bool Callable() const { return captured && original.isValid(); }
};

std::unordered_map<std::string, Slot> &Slots() {
  static auto *slots = new std::unordered_map<std::string, Slot>();
  return *slots;
}

Slot &SlotFor(const OpKey &key) { return Slots()[ToString(key)]; }

/**
 * Capture the CUDA kernel for `key`. Must run before the override is installed;
 * afterwards this returns our own wrapper and calling it recurses forever.
 */
bool CaptureInto(const OpKey &key, Slot *out) {
  auto found = c10::Dispatcher::singleton().findOp(c10::OperatorName(key.name, key.overload));
  if (!found.has_value()) {
    return false;
  }
  out->handle = *found;
  out->original = found->lookup(BackendKeySet());
  out->captured = out->original.isValid();
  return out->captured;
}

/**
 * Call the captured original unboxed. `KernelFunction::call` skips the c10::Stack
 * round trip.
 */
template <typename Ret, typename... Args>
Ret CallOriginal(const OpKey &key, Args &&...args) {
  Slot &slot = SlotFor(key);
  NativeKernelScope native_scope;
  return slot.original.call<Ret, Args...>(*slot.handle, BackendKeySet(), std::forward<Args>(args)...);
}

/**
 * In-place kernels return a reference, which the unboxed call cannot express:
 * boxing binds by value. Push the handle instead and take the value back out.
 */
void CallOriginalBoxed(const OpKey &key, c10::Stack *stack) {
  Slot &slot = SlotFor(key);
  NativeKernelScope native_scope;
  slot.original.callBoxed(*slot.handle, BackendKeySet(), stack);
}

bool FusionActive() {
  if (in_native_kernel) {
    return false;
  }
  const Flags &f = GetFlags();
  return f.enabled && f.mode != FusionMode::kOff;
}

/**
 * Shared prologue. Resolve execution and alias boundaries before emission;
 * emitters check tensor eligibility before constructing any graph nodes.
 */
bool Prepare(const OpKey &key, at::TensorList inputs, GraphKernel **out, bool writable = false) {
  if (in_native_kernel || !OpEnabled(key)) {
    return false;
  }
  GraphManager &gm = GraphManager::Instance();
  if (GetAdapter().CapturingNow()) {
    // Launching here would record the DVM kernel into the CUDA graph and replay
    // it on every launch(), multiplying the work. The graph cannot be flushed
    // once capture has started, so refuse the op; pending values are materialised
    // by the first op that runs outside the capture window.
    return false;
  }
  // A view over a pending storage cannot reuse the node, and a write to any
  // storage the graph has touched has to be resolved against real memory.
  // In-place ops check every operand: the read operand can also be pending.
  if (gm.NeedFlushForInputs(inputs, writable)) {
    gm.Flush(FlushReason::kWriteAlias);
  }
  GraphKernel *k = gm.Get();
  // Bound graph growth by op count and the runtime's dependency-mask capacity.
  if (k->PendingCount() >= kMaxGraphOps || k->ShouldTrim()) {
    gm.Flush(FlushReason::kGraphTooLarge);
    k = gm.Get();
  }
  *out = k;
  return true;
}

/** Users write `mul`, not `aten::mul`; accept both forms. */
bool NameMatches(const std::unordered_set<std::string> &set, const std::string &qualified) {
  if (set.empty()) {
    return false;
  }
  if (set.count(qualified) != 0) {
    return true;
  }
  const size_t sep = qualified.rfind("::");
  return sep != std::string::npos && set.count(qualified.substr(sep + 2)) != 0;
}

/**
 * Full-coverage guard.
 *
 * Any non-fusible op must materialise the graph before it runs, otherwise it
 * reads uncomputed memory. 
 */
void GenericGuard(const c10::OperatorHandle &op, c10::Stack *stack) {
  const auto &name = op.schema().operator_name();
  std::string key = name.overload_name.empty() ? name.name : name.name + "." + name.overload_name;
  auto &slots = Slots();
  auto it = slots.find(key);
  TORCH_CHECK(it != slots.end() && it->second.Callable(), "dvm-torch: no captured kernel for ", key);
  // Unconditional flush. A collective whose boundary depended on local graph
  // state would desynchronise the ranks, so this must not test "has pending".
  if (FusionActive()) {
    GraphManager::Instance().Flush(FlushReason::kNonFusibleOp);
  }
  NativeKernelScope native_scope;
  it->second.original.callBoxed(op, BackendKeySet(), stack);
}

}  // namespace

bool OpEnabled(const OpKey &key) {
  const Flags &f = GetFlags();
  if (!f.enabled || f.mode == FusionMode::kOff) {
    return false;
  }
  if (!f.enable_ops.empty() && !NameMatches(f.enable_ops, key.name)) {
    return false;
  }
  return !NameMatches(f.disable_ops, key.name);
}

void FlushForFallback(FlushReason reason) {
  if (FusionActive()) {
    GraphManager::Instance().Flush(reason);
  }
}

EmitResult DispatchBinary(const OpKey &key, dvm::BinaryType op, const at::Tensor &a,
                          const at::Tensor &b) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  // A Python scalar reaches the dispatcher as a 0-dim CPU tensor. It cannot be
  // loaded from device memory, but it folds into an immediate operand.
  if (IsWrappedScalar(b)) {
    if (!Prepare(key, {a}, &k)) {
      return std::nullopt;
    }
    EmitCtx ctx{k, key.name.c_str()};
    return EmitBinaryScalar(ctx, op, a, b.item());
  }
  if (IsWrappedScalar(a)) {
    if (!Prepare(key, {b}, &k)) {
      return std::nullopt;
    }
    EmitCtx ctx{k, key.name.c_str()};
    return EmitBinaryScalarLhs(ctx, op, a.item(), b);
  }
  if (!Prepare(key, {a, b}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitBinary(ctx, op, a, b);
}

EmitResult DispatchBinaryInplace(const OpKey &key, dvm::BinaryType op, at::Tensor &self,
                                 const at::Tensor &other) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (IsWrappedScalar(other)) {
    if (!Prepare(key, {self}, &k, true)) {
      return std::nullopt;
    }
    EmitCtx ctx{k, key.name.c_str()};
    return EmitBinaryInplaceScalar(ctx, op, self, other.item());
  }
  if (!Prepare(key, {self, other}, &k, true)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitBinaryInplace(ctx, op, self, other);
}

EmitResult DispatchBinaryScalar(const OpKey &key, dvm::BinaryType op, const at::Tensor &a,
                                const at::Scalar &b) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k) || b.isComplex()) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitBinaryScalar(ctx, op, a, b);
}

EmitResult DispatchEqScalar(const OpKey &key, const at::Tensor &a, const at::Scalar &b) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitEqScalar(ctx, a, b);
}

EmitResult DispatchEq(const OpKey &key, const at::Tensor &a, const at::Tensor &b) {
  if (IsWrappedScalar(b)) {
    return DispatchEqScalar(key, a, b.item());
  }
  if (IsWrappedScalar(a)) {
    return DispatchEqScalar(key, b, a.item());
  }
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a, b}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitEq(ctx, a, b);
}

EmitResult DispatchMaskedFill(const OpKey &key, const at::Tensor &self, const at::Tensor &mask,
                              const at::Scalar &value, bool inplace) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {self, mask}, &k, inplace)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitMaskedFill(ctx, self, mask, value, inplace);
}

EmitResult DispatchUnary(const OpKey &key, dvm::UnaryType op, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitUnary(ctx, op, a);
}

EmitResult DispatchNeg(const OpKey &key, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitNeg(ctx, a);
}

EmitResult DispatchRelu(const OpKey &key, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitRelu(ctx, a);
}

EmitResult DispatchRsqrt(const OpKey &key, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitRsqrt(ctx, a);
}

EmitResult DispatchSigmoid(const OpKey &key, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitSigmoid(ctx, a);
}

EmitResult DispatchTanh(const OpKey &key, const at::Tensor &a) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  // tanh is synthesised from exp, so it is not bit-exact against ATen, and the
  // GPU VM has no tanh instruction. Off unless the backend accepts a tolerance.
  if (GetAdapter().Capabilities().strict_accuracy && !GetFlags().compile_fast) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitTanh(ctx, a);
}

EmitResult DispatchPowScalar(const OpKey &key, const at::Tensor &a, double exponent) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitPowScalar(ctx, a, exponent);
}

EmitResult DispatchClamp(const OpKey &key, const at::Tensor &a,
                         const std::optional<c10::Scalar> &min,
                         const std::optional<c10::Scalar> &max) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitClamp(ctx, a, min, max);
}

EmitResult DispatchReluInplace(const OpKey &key, at::Tensor &self) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {self}, &k, true)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitReluInplace(ctx, self);
}

EmitResult DispatchReduction(const OpKey &key, const at::Tensor &a, at::OptionalIntArrayRef dims,
                             bool keepdim, std::optional<at::ScalarType> dtype, bool mean) {
  GraphManager::ScopedLock guard = GraphManager::Instance().Lock();
  GraphKernel *k = nullptr;
  if (!Prepare(key, {a}, &k)) {
    return std::nullopt;
  }
  // Sum only joins a graph that produces its input. Check after Prepare, since
  // alias, stream, and size boundaries may have materialised that producer.
  if (!mean && !k->HasPendingProducer(a)) {
    return std::nullopt;
  }
  EmitCtx ctx{k, key.name.c_str()};
  return EmitReduction(ctx, a, dims, keepdim, dtype, mean);
}

// ---------------------------------------------------------------------------
// Unboxed wrappers for the fusible set.
//
// Every fallback branch calls the captured kernel directly. Calling at::add()
// here would re-enter this registration recursively.
// ---------------------------------------------------------------------------

namespace {

at::Tensor WrapAddTensor(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha) {
  static const OpKey kKey{"aten::add", "Tensor"};
  if (alpha.to<double>() == 1.0) {
    if (auto out = DispatchBinary(kKey, dvm::kAdd, a, b); out.has_value()) {
      return *out;
    }
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b, alpha);
}

at::Tensor WrapSubTensor(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha) {
  static const OpKey kKey{"aten::sub", "Tensor"};
  if (alpha.to<double>() == 1.0) {
    if (auto out = DispatchBinary(kKey, dvm::kSub, a, b); out.has_value()) {
      return *out;
    }
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b, alpha);
}

at::Tensor WrapMulTensor(const at::Tensor &a, const at::Tensor &b) {
  static const OpKey kKey{"aten::mul", "Tensor"};
  if (auto out = DispatchBinary(kKey, dvm::kMul, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapMulScalar(const at::Tensor &a, const at::Scalar &b) {
  static const OpKey kKey{"aten::mul", "Scalar"};
  if (auto out = DispatchBinaryScalar(kKey, dvm::kMul, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapDetach(const at::Tensor &a) {
  static const OpKey kKey{"aten::detach", ""};
  if (!OpEnabled(kKey)) {
    FlushForFallback(FlushReason::kNonFusibleOp);
  }
  // Detach reads metadata only. Keep the native storage alias and shared version
  // counter; storage-based lookup and liveness also cover the detached owner.
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapEqTensor(const at::Tensor &a, const at::Tensor &b) {
  static const OpKey kKey{"aten::eq", "Tensor"};
  if (auto out = DispatchEq(kKey, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapEqScalar(const at::Tensor &a, const at::Scalar &b) {
  static const OpKey kKey{"aten::eq", "Scalar"};
  if (auto out = DispatchEqScalar(kKey, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapMaskedFill(const at::Tensor &self, const at::Tensor &mask, const at::Scalar &value) {
  static const OpKey kKey{"aten::masked_fill", "Scalar"};
  if (auto out = DispatchMaskedFill(kKey, self, mask, value, false); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, self, mask, value);
}

at::Tensor WrapDivTensor(const at::Tensor &a, const at::Tensor &b) {
  static const OpKey kKey{"aten::div", "Tensor"};
  if (auto out = DispatchBinary(kKey, dvm::kDiv, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapMaximum(const at::Tensor &a, const at::Tensor &b) {
  static const OpKey kKey{"aten::maximum", ""};
  if (auto out = DispatchBinary(kKey, dvm::kMaximum, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapMinimum(const at::Tensor &a, const at::Tensor &b) {
  static const OpKey kKey{"aten::minimum", ""};
  if (auto out = DispatchBinary(kKey, dvm::kMinimum, a, b); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, b);
}

at::Tensor WrapPowTensorScalar(const at::Tensor &a, const at::Scalar &exponent) {
  static const OpKey kKey{"aten::pow", "Tensor_Scalar"};
  if (auto out = DispatchPowScalar(kKey, a, exponent.to<double>()); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, exponent);
}

at::Tensor WrapNeg(const at::Tensor &a) {
  static const OpKey kKey{"aten::neg", ""};
  if (auto out = DispatchNeg(kKey, a); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapRelu(const at::Tensor &a) {
  static const OpKey kKey{"aten::relu", ""};
  if (auto out = DispatchRelu(kKey, a); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapRsqrt(const at::Tensor &a) {
  static const OpKey kKey{"aten::rsqrt", ""};
  if (auto out = DispatchRsqrt(kKey, a); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapSigmoid(const at::Tensor &a) {
  static const OpKey kKey{"aten::sigmoid", ""};
  if (auto out = DispatchSigmoid(kKey, a); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapTanh(const at::Tensor &a) {
  static const OpKey kKey{"aten::tanh", ""};
  if (auto out = DispatchTanh(kKey, a); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a);
}

at::Tensor WrapClamp(const at::Tensor &a, const std::optional<at::Scalar> &min,
                     const std::optional<at::Scalar> &max) {
  static const OpKey kKey{"aten::clamp", ""};
  if (auto out = DispatchClamp(kKey, a, min, max); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, min, max);
}

#define DVM_TORCH_WRAP_UNARY(fn_name, op_name, op_enum)                \
  at::Tensor fn_name(const at::Tensor &a) {                            \
    static const OpKey kKey{op_name, ""};                              \
    if (auto out = DispatchUnary(kKey, op_enum, a); out.has_value()) { \
      return *out;                                                     \
    }                                                                  \
    FlushForFallback(FlushReason::kNonFusibleOp);                      \
    return CallOriginal<at::Tensor>(kKey, a);                          \
  }

DVM_TORCH_WRAP_UNARY(WrapSqrt, "aten::sqrt", dvm::kSqrt)
DVM_TORCH_WRAP_UNARY(WrapAbs, "aten::abs", dvm::kAbs)
DVM_TORCH_WRAP_UNARY(WrapExp, "aten::exp", dvm::kExp)
DVM_TORCH_WRAP_UNARY(WrapLog, "aten::log", dvm::kLog)
DVM_TORCH_WRAP_UNARY(WrapReciprocal, "aten::reciprocal", dvm::kReciprocal)
DVM_TORCH_WRAP_UNARY(WrapRound, "aten::round", dvm::kRound)
DVM_TORCH_WRAP_UNARY(WrapFloor, "aten::floor", dvm::kFloor)
DVM_TORCH_WRAP_UNARY(WrapCeil, "aten::ceil", dvm::kCeil)
DVM_TORCH_WRAP_UNARY(WrapTrunc, "aten::trunc", dvm::kTrunc)

#undef DVM_TORCH_WRAP_UNARY

at::Tensor WrapSum(const at::Tensor &a, std::optional<at::ScalarType> dtype) {
  static const OpKey kKey{"aten::sum", ""};
  if (auto out = DispatchReduction(kKey, a, std::nullopt, false, dtype, false); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, std::move(dtype));
}

at::Tensor WrapSumDim(const at::Tensor &a, at::OptionalIntArrayRef dims, bool keepdim,
                      std::optional<at::ScalarType> dtype) {
  static const OpKey kKey{"aten::sum", "dim_IntList"};
  if (auto out = DispatchReduction(kKey, a, dims, keepdim, dtype, false); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, std::move(dims), bool{keepdim}, std::move(dtype));
}

at::Tensor WrapMean(const at::Tensor &a, std::optional<at::ScalarType> dtype) {
  static const OpKey kKey{"aten::mean", ""};
  if (auto out = DispatchReduction(kKey, a, std::nullopt, false, dtype, true); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, std::move(dtype));
}

at::Tensor WrapMeanDim(const at::Tensor &a, at::OptionalIntArrayRef dims, bool keepdim,
                       std::optional<at::ScalarType> dtype) {
  static const OpKey kKey{"aten::mean", "dim"};
  if (auto out = DispatchReduction(kKey, a, dims, keepdim, dtype, true); out.has_value()) {
    return *out;
  }
  FlushForFallback(FlushReason::kNonFusibleOp);
  return CallOriginal<at::Tensor>(kKey, a, std::move(dims), bool{keepdim}, std::move(dtype));
}

// ---- in-place -----------------------------------------------------------------

at::Tensor &WrapMaskedFillInplace(at::Tensor &self, const at::Tensor &mask, const at::Scalar &value) {
  static const OpKey kKey{"aten::masked_fill_", "Scalar"};
  if (auto out = DispatchMaskedFill(kKey, self, mask, value, true); out.has_value()) {
    return self;
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  stack.push_back(mask);
  stack.push_back(value);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

at::Tensor &WrapAddInplace(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  static const OpKey kKey{"aten::add_", "Tensor"};
  if (alpha.to<double>() == 1.0) {
    if (auto out = DispatchBinaryInplace(kKey, dvm::kAdd, self, other); out.has_value()) {
      return self;
    }
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  stack.push_back(other);
  stack.push_back(alpha);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

at::Tensor &WrapSubInplace(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  static const OpKey kKey{"aten::sub_", "Tensor"};
  if (alpha.to<double>() == 1.0) {
    if (auto out = DispatchBinaryInplace(kKey, dvm::kSub, self, other); out.has_value()) {
      return self;
    }
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  stack.push_back(other);
  stack.push_back(alpha);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

at::Tensor &WrapMulInplace(at::Tensor &self, const at::Tensor &other) {
  static const OpKey kKey{"aten::mul_", "Tensor"};
  if (auto out = DispatchBinaryInplace(kKey, dvm::kMul, self, other); out.has_value()) {
    return self;
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  stack.push_back(other);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

at::Tensor &WrapDivInplace(at::Tensor &self, const at::Tensor &other) {
  static const OpKey kKey{"aten::div_", "Tensor"};
  if (auto out = DispatchBinaryInplace(kKey, dvm::kDiv, self, other); out.has_value()) {
    return self;
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  stack.push_back(other);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

at::Tensor &WrapReluInplace(at::Tensor &self) {
  static const OpKey kKey{"aten::relu_", ""};
  if (auto out = DispatchReluInplace(kKey, self); out.has_value()) {
    return self;
  }
  FlushForFallback(FlushReason::kWriteAlias);
  c10::Stack stack;
  stack.push_back(self);
  CallOriginalBoxed(kKey, &stack);
  return self;
}

/** Registered as an override after the original has been captured. */
template <typename Fn>
bool RegisterWrapper(const OpKey &key, Fn fn) {
  Slot &slot = SlotFor(key);
  if (!CaptureInto(key, &slot)) {
    return false;
  }
  c10::OperatorName name{key.name, key.overload};
  GetRegistry().handles.push_back(c10::Dispatcher::singleton().registerImpl(
      name, kBackend, c10::KernelFunction::makeFromUnboxedRuntimeFunction(fn),
      c10::impl::CppSignature::make<Fn>(), nullptr, "dvm-torch"));
  ++GetRegistry().fusible;
  return true;
}

void InstallFusibleWrappers() {
  RegisterWrapper(OpKey{"aten::add", "Tensor"}, &WrapAddTensor);
  RegisterWrapper(OpKey{"aten::sub", "Tensor"}, &WrapSubTensor);
  RegisterWrapper(OpKey{"aten::mul", "Tensor"}, &WrapMulTensor);
  RegisterWrapper(OpKey{"aten::mul", "Scalar"}, &WrapMulScalar);
  RegisterWrapper(OpKey{"aten::detach", ""}, &WrapDetach);
  RegisterWrapper(OpKey{"aten::eq", "Tensor"}, &WrapEqTensor);
  RegisterWrapper(OpKey{"aten::eq", "Scalar"}, &WrapEqScalar);
  RegisterWrapper(OpKey{"aten::masked_fill", "Scalar"}, &WrapMaskedFill);
  RegisterWrapper(OpKey{"aten::masked_fill_", "Scalar"}, &WrapMaskedFillInplace);
  RegisterWrapper(OpKey{"aten::div", "Tensor"}, &WrapDivTensor);
  RegisterWrapper(OpKey{"aten::maximum", ""}, &WrapMaximum);
  RegisterWrapper(OpKey{"aten::minimum", ""}, &WrapMinimum);
  RegisterWrapper(OpKey{"aten::pow", "Tensor_Scalar"}, &WrapPowTensorScalar);
  RegisterWrapper(OpKey{"aten::neg", ""}, &WrapNeg);
  RegisterWrapper(OpKey{"aten::relu", ""}, &WrapRelu);
  RegisterWrapper(OpKey{"aten::rsqrt", ""}, &WrapRsqrt);
  RegisterWrapper(OpKey{"aten::sigmoid", ""}, &WrapSigmoid);
  RegisterWrapper(OpKey{"aten::tanh", ""}, &WrapTanh);
  RegisterWrapper(OpKey{"aten::clamp", ""}, &WrapClamp);
  RegisterWrapper(OpKey{"aten::sqrt", ""}, &WrapSqrt);
  RegisterWrapper(OpKey{"aten::abs", ""}, &WrapAbs);
  RegisterWrapper(OpKey{"aten::exp", ""}, &WrapExp);
  RegisterWrapper(OpKey{"aten::log", ""}, &WrapLog);
  RegisterWrapper(OpKey{"aten::reciprocal", ""}, &WrapReciprocal);
  RegisterWrapper(OpKey{"aten::round", ""}, &WrapRound);
  RegisterWrapper(OpKey{"aten::floor", ""}, &WrapFloor);
  RegisterWrapper(OpKey{"aten::ceil", ""}, &WrapCeil);
  RegisterWrapper(OpKey{"aten::trunc", ""}, &WrapTrunc);
  RegisterWrapper(OpKey{"aten::sum", ""}, &WrapSum);
  RegisterWrapper(OpKey{"aten::sum", "dim_IntList"}, &WrapSumDim);
  RegisterWrapper(OpKey{"aten::mean", ""}, &WrapMean);
  RegisterWrapper(OpKey{"aten::mean", "dim"}, &WrapMeanDim);
  RegisterWrapper(OpKey{"aten::add_", "Tensor"}, &WrapAddInplace);
  RegisterWrapper(OpKey{"aten::sub_", "Tensor"}, &WrapSubInplace);
  RegisterWrapper(OpKey{"aten::mul_", "Tensor"}, &WrapMulInplace);
  RegisterWrapper(OpKey{"aten::div_", "Tensor"}, &WrapDivInplace);
  RegisterWrapper(OpKey{"aten::relu_", ""}, &WrapReluInplace);
}

/**
 * Register a guard on every remaining CUDA op.
 *
 * Enumerating at runtime instead of generating a table at build time keeps the
 * coverage assertion true for whatever torch version is actually installed.
 */
void InstallCoverageGuards() {
  auto &dispatcher = c10::Dispatcher::singleton();
  // CUDA plus CompositeExplicitAutograd.
  //
  // An op like aten::copy_ has no CUDA kernel at all: it lives only on the
  // CompositeExplicitAutograd alias key. Enumerating the CUDA key alone misses
  // it, and a missed op silently reads uncomputed memory. Registering our guard
  // on the CUDA key still wins, because a backend key outranks an alias key.
  auto names = dispatcher.getRegistrationsForDispatchKey(kBackend);
  auto composite =
      dispatcher.getRegistrationsForDispatchKey(c10::DispatchKey::CompositeExplicitAutograd);
  names.insert(names.end(), composite.begin(), composite.end());

  auto &slots = Slots();
  for (const auto &name : names) {
    OpKey key{name.name, name.overload_name};
    // aten covers elementwise work; the two collective namespaces carry tensors
    // across ranks, where a pending value would poison every process. Custom
    // namespaces remain out of scope (see the coverage note in the README).
    const bool covered = key.name.rfind("aten::", 0) == 0 ||
                         key.name.rfind("c10d::", 0) == 0 ||
                         key.name.rfind("_c10d_functional::", 0) == 0;
    if (!covered) {
      continue;
    }
    std::string text = ToString(key);
    if (slots.count(text) != 0) {
      continue;  // already has a hand-written wrapper
    }
    Slot slot;
    if (!CaptureInto(key, &slot)) {
      continue;
    }
    auto found = dispatcher.findOp(name);
    if (!found.has_value()) {
      continue;
    }
    // An op with no Tensor argument cannot consume a pending value.
    bool has_tensor_arg = false;
    for (const auto &arg : found->schema().arguments()) {
      const auto &type = arg.type();
      if (type->kind() == c10::TypeKind::TensorType) {
        has_tensor_arg = true;
        break;
      }
      // A ListType may be int[]/float[]; only Tensor[] can carry a Pending value.
      if (type->kind() == c10::TypeKind::ListType &&
          type->castRaw<c10::ListType>()->getElementType()->kind() ==
              c10::TypeKind::TensorType) {
        has_tensor_arg = true;
        break;
      }
    }
    if (!has_tensor_arg) {
      continue;
    }
    slots[text] = slot;
    GetRegistry().handles.push_back(dispatcher.registerImpl(
        name, kBackend, c10::KernelFunction::makeFromBoxedFunction<&GenericGuard>(), std::nullopt,
        nullptr, "dvm-torch-guard"));
    ++GetRegistry().guarded;
  }
}

}  // namespace

void InstallOverrides() {
  if (OverridesInstalled()) {
    return;
  }
  InstallFusibleWrappers();
  if (GetFlags().guards) {
    InstallCoverageGuards();
  }
}

void UninstallOverrides() {
  auto &reg = GetRegistry();
  // Dropping the RAII handles restores the previous registrations.
  reg.handles.clear();
  Slots().clear();
  reg.fusible = 0;
  reg.guarded = 0;
}

}  // namespace dvm_torch
