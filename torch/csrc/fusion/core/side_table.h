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

#ifndef _DVM_TORCH_FUSION_CORE_SIDE_TABLE_H_
#define _DVM_TORCH_FUSION_CORE_SIDE_TABLE_H_

#include <ATen/core/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/util/intrusive_ptr.h>
#include <dvm.h>

#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "fusion/core/tensor_meta.h"

namespace dvm_torch {

/**
 * @brief One live graph node per storage.
 *
 * Keyed by StorageImpl* on purpose. Views and in-place updates rebuild the
 * TensorImpl, and a view's data_ptr is offset from its base: keying by
 * data_ptr makes a view look external and reads uninitialised memory.
 */
struct TensorEntry {
  dvm::NDObject *value{nullptr}; ///< Current value for this storage in the graph.
  bool needs_store{false};       ///< Graph produced this value; flush must write it.
  bool loaded_from_gm{false};    ///< value came straight from a Load node.
  int64_t offset{0};
  int64_t numel{0};
  std::vector<int64_t> shape;
  at::ScalarType dtype{at::ScalarType::Undefined};
  void *addr{nullptr};

  /// Weak liveness: a value with no live owner is a dead store.
  std::optional<c10::weak_intrusive_ptr<c10::TensorImpl>> anchor;
  /// Strong storage reference: without it the allocator may reuse this address
  /// for a different tensor inside the same graph and corrupt the load map.
  c10::Storage keep_alive;
};

class SideTable {
 public:
  static c10::StorageImpl *StorageOf(const at::Tensor &t);

  /** Exact match: same storage, offset 0, identical numel. */
  const TensorEntry *FindExact(const at::Tensor &t) const;
  /** Storage-only membership: used for write-alias detection. */
  bool ContainsStorage(const at::Tensor &t) const;

  TensorEntry *Get(const at::Tensor &t);
  TensorEntry *GetByStorage(c10::StorageImpl *s);
  TensorEntry &Upsert(const at::Tensor &t);
  void Erase(const at::Tensor &t);
  void Clear();
  size_t Size() const;

  template <typename Fn>
  void ForEach(Fn &&fn) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    for (const auto &kv : map_) {
      fn(kv.first, kv.second);
    }
  }

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<c10::StorageImpl *, TensorEntry> map_;
};

}  // namespace dvm_torch

#endif  // _DVM_TORCH_FUSION_CORE_SIDE_TABLE_H_
