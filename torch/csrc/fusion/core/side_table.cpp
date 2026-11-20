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
#include "fusion/core/side_table.h"

namespace dvm_torch {
namespace {

bool Matches(const at::Tensor &t, const TensorEntry &e) {
  return t.storage_offset() == e.offset && t.numel() == e.numel &&
         t.scalar_type() == e.dtype && t.sizes() == at::IntArrayRef(e.shape);
}

}  // namespace

c10::StorageImpl *SideTable::StorageOf(const at::Tensor &t) {
  return t.storage().unsafeGetStorageImpl();
}

const TensorEntry *SideTable::FindExact(const at::Tensor &t) const {
  if (!t.defined() || !t.has_storage()) {
    return nullptr;
  }
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = map_.find(StorageOf(t));
  if (it == map_.end()) {
    return nullptr;
  }
  const TensorEntry &e = it->second;
  // Partial overlap (view) can neither reuse the node nor be loaded directly;
  // the caller has to flush instead.
  if (!Matches(t, e)) {
    return nullptr;
  }
  return &e;
}

bool SideTable::ContainsStorage(const at::Tensor &t) const {
  if (!t.defined() || !t.has_storage()) {
    return false;
  }
  std::shared_lock<std::shared_mutex> lock(mu_);
  return map_.find(StorageOf(t)) != map_.end();
}

bool SideTable::NeedFlushForInputs(at::TensorList inputs, bool writable) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  for (const auto &t : inputs) {
    if (!t.defined() || !t.has_storage()) {
      continue;
    }
    auto it = map_.find(StorageOf(t));
    if (it != map_.end() && (writable || !Matches(t, it->second))) {
      return true;
    }
  }
  return false;
}

TensorEntry *SideTable::Get(const at::Tensor &t) {
  if (!t.defined() || !t.has_storage()) {
    return nullptr;
  }
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = map_.find(StorageOf(t));
  return it == map_.end() ? nullptr : &it->second;
}

TensorEntry *SideTable::GetByStorage(c10::StorageImpl *s) {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = map_.find(s);
  return it == map_.end() ? nullptr : &it->second;
}

TensorEntry &SideTable::Upsert(const at::Tensor &t) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  return map_[StorageOf(t)];
}

void SideTable::Erase(const at::Tensor &t) {
  if (!t.defined() || !t.has_storage()) {
    return;
  }
  std::unique_lock<std::shared_mutex> lock(mu_);
  map_.erase(StorageOf(t));
}

void SideTable::Clear() {
  std::unique_lock<std::shared_mutex> lock(mu_);
  map_.clear();
}

size_t SideTable::Size() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return map_.size();
}

}  // namespace dvm_torch
