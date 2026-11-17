/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef _DVM_PASS_H_
#define _DVM_PASS_H_

#include <vector>
#include "ops.h"

namespace dvm::pass {
class ObjectList {
 public:
  ObjectList() : sentinel_(kDataTypeEnd) {}
  void Build(const std::vector<NDObject *> &objects, bool reindex);

  void Insert(NDObject *pos, NDObject *obj) {
    auto prev = Prev(pos);
    SetPrev(obj, prev);
    SetNext(obj, pos);
    SetPrev(pos, obj);
    SetNext(prev, obj);
    size_++;
    obj->index_ = capacity_++;
  }

  void Erase(NDObject *obj) {
    auto prev = Prev(obj);
    auto next = Next(obj);
    SetNext(prev, next);
    SetPrev(next, prev);
    size_--;
  }

  NDObject *Begin() { return Next(&sentinel_); }
  NDObject *End() { return &sentinel_; }
  NDObject *ReverseBegin() { return Prev(&sentinel_); }
  NDObject *ReverseEnd() { return &sentinel_; }

  static NDObject *Next(NDObject *obj) { return reinterpret_cast<NDObject *>(obj->insn_); }
  static NDObject *Prev(NDObject *obj) { return reinterpret_cast<NDObject *>(obj->tail_insn_); }
  static void SetNext(NDObject *obj, NDObject *next) { obj->insn_ = reinterpret_cast<uint64_t *>(next); }
  static void SetPrev(NDObject *obj, NDObject *prev) { obj->tail_insn_ = reinterpret_cast<uint64_t *>(prev); }

 protected:
  NDLoadDummy sentinel_;
  size_t size_;
  size_t capacity_;
};

class BasicBlock : public ObjectList {
 public:
  struct Edge {
    int64_t next;
    NDObject *user;
  };
  BasicBlock(const std::vector<NDObject *> &objects, GraphTracker *tracker = nullptr, bool is_dynamic = false);

  inline size_t size() const { return size_; }
  inline size_t capacity() const { return capacity_; }

  NDObject *Insert(NDObject *pos, NDObject *object);
  void Erase(NDObject *object);
  void RemoveEdge(NDObject *pred, NDObject *user);
  NDObject *Move(NDObject *pos, NDObject *object);
  void UpdateInput(NDObject *obj, NDObject *old, NDObject *update);

  void PushFront(NDObject *ptr) { Insert(Begin(), ptr); }
  void PushBack(NDObject *ptr) { Insert(End(), ptr); }

  template <bool if_update_index = true>
  std::vector<NDObject *> ToVector();

  void Export(std::vector<NDObject *> &objects);

  // Will fail when object is not exist in the context.
  // And users may duplicate
  std::vector<NDObject *> GetUsers(NDObject *object) const {
    std::vector<NDObject *> res;
    for (auto idx = GetHead(object); idx != -1; idx = edges_[idx].next) {
      res.push_back(edges_[idx].user);
    }
    return res;
  }

  size_t GetUserNum(NDObject *object) const {
    size_t num = 0;
    for (auto idx = GetHead(object); idx != -1; idx = edges_[idx].next) {
      num++;
    }
    return num;
  }

  bool IsMultiUsers(NDObject *object) const {
    auto idx = GetHead(object);
    return idx != -1 && edges_[idx].next != -1;
  }

  inline void AddUser(NDObject *obj, NDObject *new_user) {
    edges_.push_back({GetHead(obj), new_user});
    SetHead(obj, static_cast<int>(edges_.size() - 1));
  }

  GraphTracker *Tracker() { return tracker_; }

  std::vector<NDObject *> dels_;
  std::vector<NDObject *> news_;
  bool is_dynamic_{false};

 protected:
  static int GetHead(NDObject *obj) { return obj->xbuf_; }
  static void SetHead(NDObject *obj, int head) { obj->xbuf_ = head; }

  std::vector<Edge> edges_;
  GraphTracker *tracker_;
};

// ------ Introducing Pass--------

/// @brief Reorder Store, to reduce maximum num of live variables
/// @details Check the variable used by Store, if the Store operator is
/// the only consumer of the variable, move the Store operator
/// @example
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Binary(%0, %1)
///   %3 = Binary(%0, %1)
///   %4 = Store(%3)
///   %5 = Store(%2)
/// }
/// ---------->
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Binary(%0, %1)
///   %5 = Store(%2)
///   %3 = Binary(%0, %1)
///   %4 = Store(%3)
/// }
/// ----------
DVM_API void ReorderStore(BasicBlock &block);

/// @brief Reorder Load, to bring the that were used earlier forward
/// @example
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Unary(%1)
///   %3 = Binary(%0, %2)
///   %4 = Load()
///   %5 = Binary(%3, %4)
///   %6 = Store(%5)
/// }
/// ---------->
/// vkernel.graph() {
///   %1 = Load()
///   %0 = Load()
///   %4 = Load()
///   %2 = Unary(%1)
///   %3 = Binary(%0, %2)
///   %5 = Binary(%3, %4)
///   %6 = Store(%5)
/// }
/// ----------
DVM_API void ReorderLoad(BasicBlock &block);

// For debug
DVM_API void PrintPeakLive(BasicBlock &bb);

/// @brief Reorder objects, to minimum peak live variables
/// @details there are two methods now.
/// The heuristic version is quick but sometimes give suboptimal results.
/// The DP version is slow but give optimal results.
/// Currently we use the heuristic method.
DVM_API void CompactPeakLiveness(BasicBlock &bb);

/// @brief Eliminate Reshape Objects if possible
/// @example
/// vkernel.graph() {
///   %0::(4,8) = Load()
///   %1::(8,4) = Load()
///   %2::(4,8) = Reshape(%1)
///   %3::(4,8) = Binary(%0, %2)
///   %4::(4,8) = Store(%3)
/// }
/// ---------->
/// vkernel.graph() {
///   %0::(8,4) = Load()
///   %1::(8,4) = Load()
///   %2::(8,4) = Binary(%0, %1)
///   %3::(8,4) = Store(%2)
/// }
/// ----------
DVM_API void EliminateReshape(BasicBlock &bb);

/// @brief Fuse supported C310 pointwise subgraphs into a JIT Custom op.
/// The pass is gated by the default-off Config VF fusion switch.
DVM_API void VfFusion(BasicBlock &bb);

/// @brief Optimize memory transfer operations, especially from UB (Unified Buffer) to GM (Global Memory), by
/// reorganizing non-continuous memory segments within the UB into a continuous memory layout. This significantly speeds
/// up the data transfer process to the GM.
DVM_API void InsertRemovePad(BasicBlock &block);

DVM_API void VectorDoubleBuffer(BasicBlock &block);

DVM_API void DeadCodeEliminate(BasicBlock &bb);

DVM_API void EliminateDiamondView(BasicBlock &bb);

using Pass = void (*)(BasicBlock &);

class PassOptimizer {
 public:
  PassOptimizer() = default;
  virtual ~PassOptimizer() {}
  void Run(std::vector<NDObject *> &objects, std::vector<NDObject *> &mng, GraphTracker *tracker) {
    auto bb = BasicBlock(objects, tracker);
    RunPass(bb, false);
    bb.Export(objects);
    if (!bb.news_.empty()) {
      mng.insert(mng.end(), bb.news_.begin(), bb.news_.end());
    }
  }
  void RunD(std::vector<NDObject *> &objects, std::vector<NDObject *> &mng, GraphTracker *tracker) {
    auto bb = BasicBlock(objects, tracker, true);
    RunPass(bb, true);
    bb.Export(objects);
    if (!bb.dels_.empty()) {
      mng.insert(mng.end(), bb.dels_.begin(), bb.dels_.end());
    }
  }
  virtual void RunPass(BasicBlock &bb, bool dyn_shape) = 0;
};

PassOptimizer *CreateOptimizer(AiCoreArch arch);
}  // namespace dvm::pass

#endif  // _DVM_PASS_H_
