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
#include "fusion/core/graph.h"

#include <chrono>
#include <sstream>
#include <unordered_set>

#include "adapter/adapter.h"

namespace dvm_torch {

void Stats::Reset() {
  emitted = 0;
  flushes = 0;
  codegen_us = 0;
  launch_us = 0;
  bails = 0;
  stores_skipped = 0;
  by_reason.fill(0);
  graph_size_histogram.clear();
}

std::string Stats::ToString() const {
  std::ostringstream os;
  os << "emit=" << emitted << " flush=" << flushes << " bail=" << bails
     << " stores_skipped=" << stores_skipped
     << " codegen=" << codegen_us << "us launch=" << launch_us << "us sizes={";
  bool first = true;
  for (const auto &kv : graph_size_histogram) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << kv.first << "op*" << kv.second;
  }
  os << "}";
  return os.str();
}

GraphKernel::GraphKernel() { kernel_.Reset(dvm::kEager, dvm::kUnifyWS); }

GraphKernel::~GraphKernel() = default;

void GraphKernel::AppendSignature(const std::string &op) {
  if (!signature_.empty()) {
    signature_.push_back(';');
  }
  signature_ += op;
}

dvm::IntArrayRef *GraphKernel::HoldShape(const std::vector<int64_t> &shape) {
  auto data = std::make_unique<std::vector<int64_t>>(shape);
  auto ref = std::make_unique<dvm::IntArrayRef>(data->data(), data->size());
  dvm::IntArrayRef *raw = ref.get();
  shape_storage_.push_back(std::move(data));
  shape_refs_.push_back(std::move(ref));
  return raw;
}

dvm::NDObject *GraphKernel::LoadStorage(const at::Tensor &t, TensorEntry &e) {
  const auto dtype = ToDvmDtype(t.scalar_type());
  std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
  dvm::NDObject *node = kernel_.Load(t.data_ptr(), HoldShape(shape), dtype);

  e.value = node;
  e.loaded_from_gm = true;
  e.needs_store = false;
  e.offset = t.storage_offset();
  e.numel = t.numel();
  e.shape = std::move(shape);
  e.dtype = t.scalar_type();
  e.addr = t.data_ptr();
  e.anchor.emplace(t.getIntrusivePtr());
  e.keep_alive = t.storage();
  return node;
}

dvm::NDObject *GraphKernel::Read(const at::Tensor &t) {
  if (const TensorEntry *exact = table_.FindExact(t)) {
    if (exact->value != nullptr) {
      return exact->value;
    }
  }
  // Storage is known but this is a view, or the value was already stored back:
  // reload from global memory so the graph sees what is actually there.
  TensorEntry &e = table_.Upsert(t);
  return LoadStorage(t, e);
}

dvm::NDObject *GraphKernel::ReduceSum(dvm::NDObject *input, const std::vector<int64_t> &dims,
                                     bool keepdim) {
  ++reduction_ops_;
  return kernel_.Reduce<dvm::kSum>(input, HoldShape(dims), keepdim);
}

void GraphKernel::RecordStore(const at::Tensor &t, dvm::NDObject *value, bool inplace) {
  TensorEntry &e = table_.Upsert(t);
  e.value = value;
  e.needs_store = true;
  e.loaded_from_gm = false;
  e.offset = t.storage_offset();
  e.numel = t.numel();
  e.shape.assign(t.sizes().begin(), t.sizes().end());
  e.dtype = t.scalar_type();
  e.addr = t.data_ptr();
  e.anchor.emplace(t.getIntrusivePtr());
  // Pin the storage: without it the allocator may hand this address to a later
  // tensor in the same graph, corrupting the load map. TrimPolicy bounds how
  // many can accumulate before a flush.
  e.keep_alive = t.storage();
  pending_stores_.push_back(PendingStore{SideTable::StorageOf(t), value, inplace});
  has_pending_ = true;
  ++emitted_ops_;
  trim_.OnEmit();
}

void GraphKernel::Write(const at::Tensor &out, dvm::NDObject *value) {
  RecordStore(out, value, false);
}

void GraphKernel::WriteInplace(const at::Tensor &self, dvm::NDObject *value) {
  RecordStore(self, value, true);
}

bool GraphKernel::Flush(void *stream, Stats *stats) {
  if (!has_pending_ || pending_stores_.empty()) {
    Reset();
    return false;
  }
  if (in_flush_) {
    // Re-entrant flush from the allocator: leave the graph alone.
    return false;
  }
  in_flush_ = true;
  // Release the guard and the workspace however we leave this function: a
  // DvmException during codegen would otherwise wedge the graph permanently.
  struct FlushGuard : dvm::WsAllocator {
    explicit FlushGuard(GraphKernel *graph) : self(graph) {}
    GraphKernel *self;
    void *ws{nullptr};
    void *Alloc(size_t size) override {
      // kUnifyWS requests one allocation for all split kernels.
      TORCH_CHECK(ws == nullptr, "dvm-torch: multiple unified workspace allocations");
      ws = size != 0 ? GetAdapter().WsAlloc(size) : nullptr;
      return ws;
    }
    ~FlushGuard() {
      self->in_flush_ = false;
      if (ws != nullptr) {
        GetAdapter().WsFree(ws);
      }
    }
  } guard{this};

  const size_t graph_ops = pending_stores_.size();

  // One store per storage. A value whose owner is already gone is a dead store
  // and need not reach global memory; live aliases still require the last value.
  std::unordered_set<c10::StorageImpl *> emitted;
  size_t stores_issued = 0;
  for (auto it = pending_stores_.rbegin(); it != pending_stores_.rend(); ++it) {
    if (!emitted.insert(it->storage).second) {
      // An earlier store of the same storage was superseded by this one.
      continue;
    }
    TensorEntry *entry = table_.GetByStorage(it->storage);
    if (entry == nullptr) {
      continue;
    }
    // Dead-store elimination. The anchor is the TensorImpl that produced the
    // value; when it is gone we still have to write if any other tensor shares
    // the storage -- a live view reads through the same buffer.
    if (entry->anchor.has_value() && entry->anchor->expired() &&
        entry->keep_alive.use_count() <= 1) {
      ++stats->stores_skipped;
      continue;
    }
    // Two-argument Store is the contiguous form; the three-argument overload
    // takes a *stride*, not a shape, and lands in NDViewStore.
    dvm::NDObject *st = kernel_.Store(entry->addr, it->value);
    kernel_.SetStoreInplace(st);
    ++stores_issued;
  }

  if (stores_issued == 0) {
    // Every value was transient and DVM cannot codegen a graph with no store.
    // Nothing observable escapes, so dropping the graph is correct.
    Reset();
    return true;
  }

  auto t0 = std::chrono::steady_clock::now();
  kernel_.Normalize();
  kernel_.CodeGen(nullptr, 0, &guard);
  auto t1 = std::chrono::steady_clock::now();
  int ret = kernel_.Launch(stream);
  auto t2 = std::chrono::steady_clock::now();

  stats->codegen_us += static_cast<size_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  stats->launch_us += static_cast<size_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
  ++stats->flushes;
  stats->emitted += emitted_ops_;
  stats->graph_size_histogram[graph_ops] += 1;
  if (ret != 0) {
    GetAdapter().CheckLastError();
  }

  Reset();
  return ret == 0;
}

void GraphKernel::Reset() {
  kernel_.Reset(dvm::kEager, dvm::kUnifyWS);
  pending_stores_.clear();
  shape_storage_.clear();
  shape_refs_.clear();
  table_.Clear();
  signature_.clear();
  has_pending_ = false;
  in_flush_ = false;
  emitted_ops_ = 0;
  reduction_ops_ = 0;
  trim_.Reset();
}

GraphManager &GraphManager::Instance() {
  static GraphManager gm;
  return gm;
}

GraphManager::~GraphManager() {
  // Never let a process exit with a half-built graph and a dangling kernel.
  try {
    Flush(FlushReason::kProcessExit);
  } catch (...) {
  }
}

GraphManager::ScopedLock GraphManager::Lock() { return ScopedLock(mu_); }

GraphKernel *GraphManager::Get() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  void *stream = GetAdapter().CurrentStream();
  if (current_ != nullptr &&
      (stream != current_stream_ || owner_ != std::this_thread::get_id())) {
    Flush(FlushReason::kStreamSwitch);
  }
  // Rebind unconditionally: the branch above leaves current_ non-null, so
  // relying on the null check alone would keep launching on the old stream.
  if (current_ == nullptr) {
    current_ = new GraphKernel();
  }
  current_stream_ = stream;
  owner_ = std::this_thread::get_id();
  return current_;
}

void GraphManager::Discard() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (current_ != nullptr) {
    delete current_;
    current_ = nullptr;
  }
}

void GraphManager::Flush(FlushReason reason) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (current_ == nullptr) {
    return;
  }
  if (in_capture_ || no_flush_depth_ > 0) {
    return;
  }
  current_->Flush(current_stream_, &stats_);
  stats_.by_reason[static_cast<size_t>(reason)] += 1;
}

bool GraphManager::HasPending() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return current_ != nullptr && !current_->Empty();
}

bool GraphManager::NeedFlushForInput(const at::Tensor &t) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (current_ == nullptr || current_->Empty()) {
    return false;
  }
  if (!current_->HasStorage(t)) {
    return false;
  }
  // Same storage but not the whole tensor: a view, which cannot reuse the node.
  return !current_->IsExactPending(t);
}

bool GraphManager::NeedFlushForWritable(const at::Tensor &t) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (current_ == nullptr || current_->Empty()) {
    return false;
  }
  return current_->HasStorage(t);
}

std::map<FlushReason, size_t> GraphManager::ReasonTotals() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  std::map<FlushReason, size_t> out;
  const auto &arr = stats_.by_reason;
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] != 0) {
      out[static_cast<FlushReason>(i)] = arr[i];
    }
  }
  return out;
}

void GraphManager::ResetStats() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  stats_.Reset();
}

void GraphManager::SetEnabled(bool on) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Flags f = GetFlags();
  f.enabled = on;
  if (on && f.mode == FusionMode::kOff) {
    f.mode = FusionMode::kOverride;
  }
  if (!on) {
    f.mode = FusionMode::kOff;
  }
  SetFlags(f);
}

GraphManager::NoFlushScope::NoFlushScope() {
  GraphManager &gm = GraphManager::Instance();
  std::lock_guard<std::recursive_mutex> lock(gm.mu_);
  previous_ = gm.no_flush_depth_ > 0;
  ++gm.no_flush_depth_;
}

GraphManager::NoFlushScope::~NoFlushScope() {
  GraphManager &gm = GraphManager::Instance();
  std::lock_guard<std::recursive_mutex> lock(gm.mu_);
  if (!previous_) {
    gm.no_flush_depth_ = 0;
  } else {
    --gm.no_flush_depth_;
  }
}

extern "C" __attribute__((visibility("default"))) void dvm_torch_flush_if_pending(void) {
  GraphManager::Instance().Flush(FlushReason::kEventRecord);
}

}  // namespace dvm_torch
