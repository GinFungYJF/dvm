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

#ifndef _DVM_TORCH_FUSION_CORE_GRAPH_H_
#define _DVM_TORCH_FUSION_CORE_GRAPH_H_

#include <ATen/core/Tensor.h>
#include <dvm.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "flags.h"
#include "fusion/core/side_table.h"
#include "fusion/core/tensor_meta.h"

namespace dvm_torch {

/** Hard upper bound on ops in one graph, independent of TrimPolicy tuning. */
constexpr size_t kMaxGraphOps = 4096;

struct Stats {
  size_t emitted{0};
  size_t flushes{0};
  size_t codegen_us{0};
  size_t launch_us{0};
  size_t bails{0};
  size_t stores_skipped{0};
  std::array<size_t, static_cast<size_t>(FlushReason::kNumReasons)> by_reason{};
  std::map<size_t, size_t> graph_size_histogram;

  void Reset();
  std::string ToString() const;
};

/**
 * @brief One deferred graph: a dvm::Kernel plus the storage-to-node side table.
 *
 * Tensors returned to users are real at::Tensor instances allocated by
 * at::empty; only their contents are deferred. Stores are issued at flush time
 * so an in-place chain writes each storage exactly once.
 */
class GraphKernel {
 public:
  GraphKernel();
  ~GraphKernel();

  GraphKernel(const GraphKernel &) = delete;
  GraphKernel &operator=(const GraphKernel &) = delete;

  dvm::Kernel &GetKernel() {
    has_state_ = true;
    return kernel_;
  }

  /** Node holding this storage's current value; loads from GM if absent. */
  dvm::NDObject *Read(const at::Tensor &t);

  struct BoolPredicate {
    dvm::NDObject *value;
    bool inverted{false};
  };

  /** Equivalent Boolean condition, optionally with the Select branches swapped. */
  BoolPredicate ResolveBoolPredicate(dvm::NDObject *value) const;
  void RecordBoolPredicate(dvm::NDObject *result, dvm::NDObject *input, bool inverted);

  /** Keep reduction dimensions alive until the deferred kernel is destroyed. */
  dvm::NDObject *ReduceSum(dvm::NDObject *input, const std::vector<int64_t> &dims,
                           bool keepdim);

  /** Register a freshly produced value; the store is deferred. */
  void Write(const at::Tensor &out, dvm::NDObject *value);

  /** Register an in-place update of an existing storage. */
  void WriteInplace(const at::Tensor &self, dvm::NDObject *value);

  bool Empty() const { return !has_pending_; }
  size_t PendingCount() const { return pending_stores_.size(); }
  bool HasStorage(const at::Tensor &t) const { return table_.ContainsStorage(t); }
  bool IsExactPending(const at::Tensor &t) const { return table_.FindExact(t) != nullptr; }
  bool HasPendingProducer(const at::Tensor &t) const {
    const TensorEntry *entry = table_.FindExact(t);
    return entry != nullptr && entry->needs_store;
  }
  bool NeedFlushForInputs(at::TensorList inputs, bool writable) const {
    return has_pending_ && table_.NeedFlushForInputs(inputs, writable);
  }
  bool ShouldTrim() const {
    // EagerArea uses a 64-bit dependency mask. Each result can introduce a
    // store area, and each reduction can introduce one additional area.
    // Reserve space for the next emitter's reduction and result.
    return PendingCount() + reduction_ops_ >= 62 || trim_.ShouldTrim();
  }

  bool Flush(void *stream, Stats *stats);

  void Reset();

  const std::string &Signature() const { return signature_; }
  void AppendSignature(const std::string &op);

  TrimPolicy &Trim() { return trim_; }
  const TrimPolicy &Trim() const { return trim_; }

 private:
  struct PendingStore {
    c10::StorageImpl *storage;
    dvm::NDObject *value;
    bool inplace;
  };

  dvm::NDObject *LoadStorage(const at::Tensor &t, TensorEntry &e);
  void RecordStore(const at::Tensor &t, dvm::NDObject *value, bool inplace);
  dvm::IntArrayRef *HoldShape(const std::vector<int64_t> &shape);

  SideTable table_;
  // Immutable node identities keep predicate origins separate from later writes
  // or reloads of the same tensor storage. All origins expire with the graph.
  std::unordered_map<dvm::NDObject *, BoolPredicate> bool_predicates_;
  // A failed emitter can leave loads or IR without producing a pending store.
  bool has_state_{false};
  bool has_pending_{false};
  bool in_flush_{false};
  size_t emitted_ops_{0};
  size_t reduction_ops_{0};
  std::string signature_;
  TrimPolicy trim_;
  std::vector<PendingStore> pending_stores_;

  /// DVM keeps IntArrayRef pointers and never copies: these must outlive the kernel.
  std::vector<std::unique_ptr<std::vector<int64_t>>> shape_storage_;
  std::vector<std::unique_ptr<dvm::IntArrayRef>> shape_refs_;
  dvm::Kernel kernel_;
};

/**
 * @brief Owns the current graph, the flush policy and the counters.
 *
 * A graph belongs to one stream and one thread; either changing forces a flush
 * because a launch must stay on the stream that built it.
 */
class GraphManager {
 public:
  static GraphManager &Instance();

  /** Serialise emit against a concurrent flush on another thread. */
  using ScopedLock = std::unique_lock<std::recursive_mutex>;
  ScopedLock Lock();

  GraphKernel *Get();
  /** Forget the current graph without launching (used by capture and shutdown). */
  void Discard();
  void Flush(FlushReason reason);

  bool NeedFlushForInputs(at::TensorList inputs, bool writable = false) const;
  bool HasPending() const;

  Stats &GetStats() { return stats_; }
  std::map<FlushReason, size_t> ReasonTotals() const;
  void ResetStats();

  void SetEnabled(bool on);

  /**
   * @brief Suspend flushing while the graph allocates its own outputs.
   *
   * at::empty dispatches through the same guards as real ops. Flushing there
   * would free nodes the emitter is still holding.
   */
  class NoFlushScope {
   public:
    NoFlushScope();
    ~NoFlushScope();

   private:
    bool previous_;
  };

 private:
  GraphManager() = default;
  ~GraphManager();

  mutable std::recursive_mutex mu_;
  GraphKernel *current_{nullptr};
  void *current_stream_{nullptr};
  std::thread::id owner_;
  Stats stats_;
  bool in_capture_{false};
  size_t no_flush_depth_{0};
};

extern "C" __attribute__((visibility("default"))) void dvm_torch_flush_if_pending(void);

}  // namespace dvm_torch

#endif  // _DVM_TORCH_FUSION_CORE_GRAPH_H_
