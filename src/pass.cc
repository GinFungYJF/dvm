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

#include <algorithm>
#include <queue>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "pass.h"
#include "kernel.h"

namespace dvm::pass {

namespace {
inline std::vector<NDObject *> GetPreds(NDObject *obj) {
  std::vector<NDObject *> res;
  obj->ForInput([&res](NDObject *op) { res.push_back(op); });
  return res;
}

size_t GetInputsNum(NDObject *obj) {
  if (obj->lhs_ == nullptr) {
    return 0;
  }
  if (obj->rhs_ == nullptr) {
    return 1;
  }
  return 2 + (obj->CheckFlag(OBJ_FLAG_XHS) ? static_cast<FlexOp *>(obj)->xhs_->in_num : 0);
}

constexpr int STATIC_LIVE = 2;
constexpr int DYN_LIVE = 1;

size_t LivenessAnalyze(BasicBlock &bb, std::unordered_map<NDObject *, int> &lives) {
  for (NDObject *obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    if (obj->IsLoad()) {
      lives[obj] = STATIC_LIVE;
    } else if (obj->IsStore()) {
      lives[obj->lhs_] = STATIC_LIVE;
    }
  }
  size_t current = lives.size();
  size_t peak = current;
  for (auto obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    if (!lives.count(obj)) {
      continue;
    }
    obj->ForInput([&lives, &current](NDObject *in) {
      if (auto it = lives.find(in); it == lives.end()) {
        current++;
        lives[in] = DYN_LIVE;
      }
    });
    if (current > peak) {
      peak = current;
    }
    if (lives[obj] == DYN_LIVE) {
      current--;
    }
  }
  return peak;
}

std::vector<NDObject *> ReorderObjectsHeuristic(BasicBlock &bb, std::unordered_map<NDObject *, int> &lives) {
  // Preprocess, calculate points of all objects
  std::unordered_map<NDObject *, int64_t> points;
  std::unordered_map<NDObject *, int64_t> heights;
  constexpr int64_t B1 = -1;
  constexpr int64_t B2 = 1000;
  constexpr int64_t B3 = 1000000;
  for (NDObject *obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    int64_t height = 0;
    for (auto user : bb.GetUsers(obj)) {
      if (auto h = heights[user] + 1; h > height) {
        height = h;
      }
    }
    heights[obj] = height;
    int64_t point = height * B1;
    if (!obj->IsSimd()) {
      point += B3;
    }
    obj->ForInput([&lives, &point](NDObject *in) {
      if (lives[in] == DYN_LIVE) {
        point += B2;
      }
    });
    points[obj] = point;
  }
  auto compare = [&points](NDObject *a, NDObject *b) { return points[a] < points[b]; };
  std::priority_queue<NDObject *, std::vector<NDObject *>, decltype(compare)> pq(compare);
  std::unordered_map<NDObject *, uint32_t> in_degrees;
  for (NDObject *obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    in_degrees[obj] = GetInputsNum(obj);
    if (in_degrees[obj] == 0) {
      pq.push(obj);
    }
  }
  std::vector<NDObject *> res;
  res.reserve(bb.size());
  while (!pq.empty()) {
    auto obj = pq.top();
    pq.pop();
    res.emplace_back(obj);
    for (auto user : bb.GetUsers(obj)) {
      --in_degrees[user];
      if (in_degrees[user] == 0) {
        pq.push(user);
      }
    }
  }
  return res;
}
}  // namespace

void ObjectList::Build(const std::vector<NDObject *> &objects, bool reindex) {
  // build linked list from objects
  NDObject *last_object = &sentinel_;
  int index = 0;
  for (auto object : objects) {
    SetNext(last_object, object);
    SetPrev(object, last_object);
    last_object = object;
    if (reindex) {
      last_object->index_ = index;
      ++index;
    }
  }
  SetNext(last_object, &sentinel_);
  SetPrev(&sentinel_, last_object);
  if (reindex) {
    size_ = capacity_ = objects.size();
  }
}

BasicBlock::BasicBlock(const std::vector<NDObject *> &objects, GraphTracker *tracker, bool is_dynamic)
    : is_dynamic_(is_dynamic), tracker_(tracker) {
  // build linked list from objects
  ObjectList::Build(objects, true);
  for (auto obj : objects) {
    SetHead(obj, -1);
  }
  edges_.reserve(objects.size() * 2);
  for (auto obj : objects) {
    obj->ForInput([this, obj](NDObject *pred) { this->AddUser(pred, obj); });
  }
}

template <bool if_update_index>
std::vector<NDObject *> BasicBlock::ToVector() {
  std::vector<NDObject *> res;
  res.reserve(size());
  int i = 0;
  for (NDObject *iter = Begin(); iter != End(); iter = Next(iter)) {
    if constexpr (if_update_index) {
      iter->index_ = i++;
    }
    res.push_back(iter);
  }
  return res;
}

// ToVector is defined in this translation unit.  Keep the non-reindexing
// specialization available to pass implementations in other translation
// units as well.
template std::vector<NDObject *> BasicBlock::ToVector<false>();

void BasicBlock::Export(std::vector<NDObject *> &objects) {
  objects.clear();
  objects.reserve(size());
  int i = 0;
  for (NDObject *iter = Begin(); iter != End(); iter = Next(iter)) {
    iter->index_ = i++;
    ObjectList::Prev(iter)->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    objects.push_back(iter);
  }
  ObjectList::Prev(End())->insn_ = nullptr;
}

NDObject *BasicBlock::Insert(NDObject *pos, NDObject *object) {
  if (pos == object) {
    return object;
  }
  ObjectList::Insert(pos, object);
  SetHead(object, -1);
  for (auto pred : GetPreds(object)) {
    AddUser(pred, object);
  }
  // object should be deleted by owner
  news_.push_back(object);
  return object;
}

void BasicBlock::Erase(NDObject *object) {
  ObjectList::Erase(object);
  SetHead(object, -1);
  for (auto pred : GetPreds(object)) {
    RemoveEdge(pred, object);
  }
  dels_.push_back(object);
}

void BasicBlock::RemoveEdge(NDObject *pred, NDObject *user) {
  auto idx = GetHead(pred);
  auto last = idx;
  while (idx != -1) {
    if (edges_[idx].user == user) {
      if (idx == GetHead(pred)) {
        SetHead(pred, edges_[idx].next);
      } else {
        edges_[last].next = edges_[idx].next;
      }
      return;
    }
    last = idx;
    idx = edges_[idx].next;
  }
}

NDObject *BasicBlock::Move(NDObject *pos, NDObject *obj) {
  if (pos == obj || Prev(pos) == obj) {
    return obj;
  }
  ObjectList::Erase(obj);
  ObjectList::Insert(pos, obj);
  return obj;
}

void BasicBlock::UpdateInput(NDObject *obj, NDObject *old, NDObject *update) {
  auto Rebind = [this, old, update, obj](NDObject *&slot) {
    if (tracker_) {
      tracker_->Record(&slot);
    }
    slot = update;
    RemoveEdge(old, obj);
    AddUser(update, obj);
  };
  if (obj->GetObjectType() == kConcat) {
    for (auto &s : static_cast<ConcatOp *>(obj)->slices_) {
      if (s.input == old) {
        s.input = update;
      }
    }
  }
  if (obj->lhs_ == old) {
    Rebind(obj->lhs_);
    if (obj->SharedNdd()) {
      auto old_ndd = obj->nd_.data;
      auto new_ndd = update->nd_.data;
      if (old_ndd != new_ndd) {
        std::vector<NDObject *> stack = {obj};
        while (!stack.empty()) {
          auto top = stack.back();
          stack.pop_back();
          top->nd_.data = new_ndd;
          auto users = GetUsers(top);
          for (auto op : users) {
            if (op->nd_.data == old_ndd) {
              stack.push_back(op);
            }
          }
        }
      }
    }
    return;
  } else if (obj->rhs_ == old) {
    Rebind(obj->rhs_);
    return;
  } else {
    // ops with extended inputs (xhs_)
    ASSERT(obj->flags_ & OBJ_FLAG_XHS);
    auto xhs = static_cast<FlexOp *>(obj)->xhs_;
    if (xhs != nullptr) {
      for (int i = 0; i < xhs->in_num; ++i) {
        if (xhs->data[i] == old) {
          Rebind(xhs->data[i]);
          return;
        }
      }
    }
  }
  RemoveEdge(old, obj);
  AddUser(update, obj);
}

void ReorderStore(BasicBlock &block) {
  for (NDObject *iter = block.Begin(); iter != block.End();) {
    if (iter->IsStore()) {
      auto obj = iter->lhs_;
      auto store = iter;
      iter = block.Next(iter);
      // If store is in right position, Move will do nothing
      block.Move(block.Next(obj), store);
    } else {
      iter = block.Next(iter);
    }
  }
}

void ReorderLoad(BasicBlock &block) {
  uint32_t idx = 0;
  std::unordered_map<NDObject *, std::pair<uint32_t, uint32_t>> user_idx;
  // Get order of Load by usage
  for (NDObject *iter = block.Begin(); iter != block.End(); iter = block.Next(iter)) {
    iter->ForInput([&user_idx, idx](NDObject *in) {
      if (in->IsLoad()) {
        auto it = user_idx.find(in);
        if (it == user_idx.end()) {
          user_idx[in] = {idx, idx};
        } else {
          it->second.second = idx;
        }
      }
    });
    idx++;
  }
  using LoadOrder = std::pair<NDObject *, uint32_t>;
  std::vector<LoadOrder> loads;
  loads.reserve(user_idx.size());
  for (const auto &kv : user_idx) {
    loads.push_back({kv.first, kv.second.first << 16 | kv.second.second});
  }
  std::sort(loads.begin(), loads.end(), [](const LoadOrder &a, const LoadOrder &b) { return a.second > b.second; });
  for (auto &load : loads) {
    block.Move(block.Begin(), load.first);
  }
}

namespace {
class RmPadStore final : public NDStore {
 public:
  RmPadStore(NDStore *store) : NDStore(store->addr_.gm, store->lhs_), store_(store) {
    rm_pad_ = new RemovePadOp(store->lhs_);
  }
  ~RmPadStore() override { delete rm_pad_; }
  void Normalize(std::vector<NDObject *> &run_ops) override {
    lhs_ = rm_pad_->lhs_;
    if (NeedRemovePad(lhs_)) {
      rm_pad_->Normalize(run_ops);
      run_ops.push_back(rm_pad_);
      lhs_ = rm_pad_;
    }
    NDStore::Normalize(run_ops);
  }
  uint64_t Emit(VectorKernel &k) override {
    auto size = NDStore::Emit(k);
    store_->addr_.Update(addr_);
    return size;
  }
  bool NeedRemovePad(NDObject *input) {
    constexpr uint64_t kMinSizeLimit = 2048;
    if (auto &dims = input->nd_.dims(); !dims.empty()) {
      for (size_t i = 0; i < dims.size() - 1; ++i) {
        if (dims[i] != 1) {
          uint64_t lead_size = dims[i] * ITEM_SIZE[type_id_];
          if (lead_size % SIMD_BLOCK_SIZE && lead_size < SIMD_REPEAT_SIZE) {
            uint64_t total_size = lead_size;
            for (size_t j = i + 1; j < dims.size(); ++j) {
              total_size *= dims[j];
            }
            if (total_size > g_system.CoreNum() * kMinSizeLimit) {
              return true;
            }
          }
          break;
        }
      }
    }
    return false;
  }
  NDStore *store_;
  RemovePadOp *rm_pad_;
};
} // namespace

void InsertRemovePad(BasicBlock &block) {
  if (block.is_dynamic_) {
    bool fold_reject = false;
    for (auto op = block.Begin(); op != block.End(); op = block.Next(op)) {
      auto obj_type = op->GetObjectType();
      if (obj_type == kConcat || obj_type == kSplitOp) {
        return;
      }
      if (obj_type == kBroadcastTo || obj_type == kReduce) {
        fold_reject = true;
      }
    }
    if (fold_reject) {
      for (auto op = block.Begin(); op != block.End(); op = block.Next(op)) {
        if (op->GetObjectType() != kStore) continue;
        auto input = op->lhs_;
        if (uint64_t item_size = ITEM_SIZE[input->type_id_]; item_size != 2 && item_size != 4) {
          continue;
        }
        if (input->obj_id_ == kReduce) {
          if (!g_system.deterministic_) {
            input->SetFlag(OBJ_FLAG_REDUCE_RMPAD_EN);
          }
        } else if (input->obj_id_ != kElementAny) {
          auto store = new RmPadStore(static_cast<NDStore *>(op));
          block.Insert(op, store);
          block.Erase(op);
          op = store;
        }
      }
    }
    return;
  }
  size_t max_depth = 1;
  auto min_type_id = kDataTypeEnd;
  for (NDObject *op = block.Begin(); op != block.End(); op = block.Next(op)) {
    auto obj_type = op->GetObjectType();
    if (obj_type == kConcat || obj_type == kSplitOp) {
      return;
    }
    max_depth = std::max(max_depth, op->nd_.size());
    min_type_id = std::min(min_type_id, op->type_id_);
  }
  TileInfo info;
  info.Reset(max_depth, true);
  for (NDObject *op = block.Begin(); op != block.End(); op = block.Next(op)) {
    if (auto ndd = op->Ndd(); ndd != nullptr && ndd->dims.size() != max_depth) {
      ndd->dims.resize(max_depth, 1);
    }
    op->TileCollect(info);
  }
  for (NDObject *iter = block.Begin(); iter != block.End(); iter = block.Next(iter)) {
    if (iter->GetObjectType() == kStore) {
      auto obj_id = iter->lhs_->obj_id_;
      if (obj_id == kElementAny || static_cast<int>(iter->nd_.size()) == info.lead_depth) {
        continue;
      }
      if (obj_id == kReduce && g_system.deterministic_) {
        continue;
      }
      uint64_t item_size = ITEM_SIZE[iter->lhs_->type_id_];
      if (item_size != 2 && item_size != 4) {
        continue;
      }
      uint64_t iter_size = ITEM_SIZE[min_type_id];
      for (int i = 0; i < info.lead_depth; i++) {
        iter_size *= iter->nd_[i];
      }
      const uint64_t store_threshold = g_system.CoreNum() * iter_size;
      if (static_cast<uint64_t>(iter->Size()) <= store_threshold) {  // Below this threshold, the store can be split without requiring repeat.
        continue;
      }
      if (iter_size % SIMD_BLOCK_SIZE && iter_size < SIMD_REPEAT_SIZE) {
        if (obj_id == kReduce) {
          iter->lhs_->SetFlag(OBJ_FLAG_REDUCE_RMPAD_EN);
        } else {
          auto remove_pad = new RemovePadOp(iter->lhs_);
          remove_pad->nd_ = iter->lhs_->nd_;
          if (auto tracker = block.Tracker()) {
            tracker->Record(&iter->lhs_);
          }
          iter->lhs_ = remove_pad;
          block.Insert(iter, remove_pad);
        }
      }
    }
  }
}

void PrintPeakLive(BasicBlock &bb) {
  std::unordered_map<NDObject *, int> lives;
  auto maxlive = LivenessAnalyze(bb, lives);
  std::cout << "peak live: " << maxlive << std::endl;
}

void CompactPeakLiveness(BasicBlock &bb) {
  std::vector<NDObject *> backup = bb.ToVector<false>();
  std::unordered_map<NDObject *, int> lives;
  auto old_peak = LivenessAnalyze(bb, lives);
  auto new_order = ReorderObjectsHeuristic(bb, lives);
  bb.Build(new_order, false);
  lives.clear();
  auto new_peak = LivenessAnalyze(bb, lives);
  if (new_peak >= old_peak) {
    // Reorder cause a bad result, rollback
    bb.Build(backup, false);
  }
}

void EliminateReshape(BasicBlock &bb) {
  for (NDObject *it = bb.Begin(); it != bb.End(); it = bb.Next(it)) {
    auto op = it;
    if (op->GetObjectType() != kReshape) continue;
    auto lhs = op->lhs_;
    auto small = &lhs->nd_.dims();
    auto big = &op->nd_.dims();
    if (small->size() > big->size()) {
      std::swap(small, big);
    }
    for (size_t i = 0; i < small->size(); ++i) {
      if (small->operator[](i) != big->operator[](i)) continue;
    }
    for (size_t i = small->size(); i < big->size(); ++i) {
      if (big->operator[](i) != 1) continue;
    }
    if (lhs->IsLoad()) {
      for (auto succ : bb.GetUsers(op)) {
        if (succ->IsStore()) {
          auto copy = new CopyOp(lhs);
          std::vector<NDObject *> stuff_ops;
          copy->Normalize(stuff_ops);
          ASSERT(stuff_ops.empty());
          bb.Insert(op, copy);
          lhs = copy;
          break;
        }
      }
    }
    for (auto succ : bb.GetUsers(op)) {
      bb.UpdateInput(succ, op, lhs);
    }
    bb.Erase(op);
  }
}

void VectorDoubleBuffer(BasicBlock &bb) {
  constexpr int LOAD_SPAN = 6;
  constexpr int STORE_SPAN = 3;
  constexpr int MIN_DB_SPAN = 2;
  constexpr int64_t MIN_DB_SIZE = 1024 * 1024;
  static const std::unordered_map<int, int> long_simd_spans = {
    {kReduce, 3}, {kBroadcastTo, 2}, {kOneHot, 3}, {kCustom, 4}};
  auto get_simd_span = [](NDObject *op) -> int {
    auto it = long_simd_spans.find(op->obj_id_);
    return it != long_simd_spans.end() ? it->second : 1;
  };
  struct LoadInfo {
    NDObject *load;
    NDObject *start_ref_op;
    int start_ref;
    int end_ref;
  };
  std::vector<LoadInfo> loads;
  std::vector<NDObject *> stores;
  size_t load_num = 0;
  for (auto op = bb.Begin(); op != bb.End(); op = bb.Next(op)) {
    if (op->IsLoad()) {
      load_num++;
    } else if (op->IsStore()) {
      auto id = op->lhs_->obj_id_;
      if (id != kReduce && id != kRemovePad && (bb.is_dynamic_ || op->nd_.dims().prod() > MIN_DB_SIZE)) {
        stores.push_back(op);
      }
    }
  }
  loads.reserve(load_num);
  int cur_tm = 0;
  for (auto op = bb.Begin(); op != bb.End(); op = bb.Next(op)) {
    if (op->IsSimd()) {
      int end_tm = cur_tm + get_simd_span(op);
      op->ForInput([&bb, &loads, cur_tm, end_tm, op](NDObject *in) {
        if (in->IsLoad()) {
          auto it = std::find_if(loads.begin(), loads.end(), [in](LoadInfo &i) { return i.load == in; });
          if (it != loads.end()) {
            it->end_ref = end_tm;
          } else if (bb.is_dynamic_ || in->nd_.dims().prod() > MIN_DB_SIZE) {
            loads.emplace_back(LoadInfo{in, op, cur_tm, end_tm});
          }
        }
      });
      cur_tm = end_tm;
    }
  }
  auto &stuff_ops = stores;
  int total_tm = cur_tm;
  while (true) {
    int end_ref = 0;
    LoadInfo *info = nullptr;
    for (auto &i : loads) {
      if (i.end_ref > end_ref) {
        end_ref = i.end_ref;
      }
      auto span = end_ref - i.start_ref;
      if (span > MIN_DB_SPAN && span + LOAD_SPAN > total_tm) {
        info = &i;
        break;
      }
    }
    if (info == nullptr) {
      break;
    }
    auto load = info->load;
    auto copy = new CopyOp(load);
    copy->Normalize(stuff_ops);
    for (auto succ : bb.GetUsers(load)) {
      bb.UpdateInput(succ, load, copy);
    }
    bb.Insert(info->start_ref_op, copy);
    auto end_tm = info->start_ref + 1;
    info->end_ref = end_tm;
    for (auto &i : loads) {
      if (i.start_ref > end_tm) {
        i.start_ref++;
      }
      if (i.end_ref > end_tm) {
        i.end_ref++;
      }
    }
    total_tm++;
  }
  for (auto store : stores) {
    auto input = store->lhs_;
    if (int span = get_simd_span(input); span > MIN_DB_SPAN && span + STORE_SPAN > total_tm) {
      auto copy = new CopyOp(input);
      copy->Normalize(stuff_ops);
      bb.UpdateInput(store, input, copy);
      bb.Insert(bb.Next(input), copy);
      total_tm++;
    }
  }
}

void DeadCodeEliminate(BasicBlock &bb) {
  for (auto obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    obj->reuse_dep_ = obj->IsStore() ? 1 : 0;
  }
  for (auto obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    if (obj->reuse_dep_) {
      obj->ForInput([](NDObject *in) { in->reuse_dep_ = 1; });
    } else if (obj->GetObjectType() == kSplitOp && static_cast<SplitOp *>(obj)->slice_idx_ == 0) {
      auto main = static_cast<SplitOpM *>(obj);
      for (auto sib : main->siblings_) {
        if (sib->reuse_dep_) {
          main->reuse_dep_ = 1;
          break;
        }
      }
    }
  }
  for (auto obj = bb.Begin(); obj != bb.End();) {
    auto next = bb.Next(obj);
    if (obj->reuse_dep_ == 0) {
      bb.Erase(obj);
    }
    obj = next;
  }
}

namespace {

class NDBindLoad : public NDLoad {
 public:
  NDBindLoad(NDAccess *input) : NDLoad(nullptr, input->shape_ref_, input->type_id_) {
    SetFlag(OBJ_FLAG_LOAD_BIND);
    addr_.gm = input;
  }
  ~NDBindLoad() override {
    for (auto *bind : bind_addrs_) delete bind;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override {
    NDLoad::Normalize(run_ops);
    addr_used_ = 0;
  }
  uint64_t Emit(VectorKernel &k) override {
    auto &input = *LoadBind();
    if (!input.addr_.reloc_) input.addr_.Update(&input.addr_.data);
    auto size = NDLoad::Emit(k);
    RelocAddr *bind;
    if (addr_used_ < static_cast<int>(bind_addrs_.size())) {
      bind = bind_addrs_[addr_used_++];
    } else {
      bind = new RelocAddr();
      bind_addrs_.push_back(bind);
      addr_used_++;
    }
    bind->Update(addr_);
    k.code_.BindOpFast(*bind, input.addr_);
    return size;
  }
  NDObject *Clone(CloneHelper &h) override { return new NDBindLoad(static_cast<NDAccess *>(h.GetClone(LoadBind()))); }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "BindLoad"; }
  std::vector<RelocAddr *> bind_addrs_;
  int addr_used_{0};
};

class NDViewBindLoad : public NDViewLoad {
 public:
  NDViewBindLoad(NDViewLoad *input)
      : NDViewLoad(nullptr, input->shape_ref_, input->src_stride_ref_, input->type_id_), input_(input) {
    SetFlag(OBJ_FLAG_LOAD_BIND);
  }
  ~NDViewBindLoad() override {
    for (auto *bind : bind_addrs_) delete bind;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override {
    NDViewLoad::Normalize(run_ops);
    addr_used_ = 0;
  }
  uint64_t Emit(VectorKernel &k) override {
    auto &input = *input_;
    if (!input.addr_.reloc_) input.addr_.Update(&input.addr_.data);
    auto size = NDViewLoad::Emit(k);
    RelocAddr *bind;
    if (addr_used_ < static_cast<int>(bind_addrs_.size())) {
      bind = bind_addrs_[addr_used_++];
    } else {
      bind = new RelocAddr();
      bind_addrs_.push_back(bind);
      addr_used_++;
    }
    bind->Update(addr_);
    k.code_.BindOpFast(*bind, input.addr_);
    return size;
  }
  NDObject *Clone(CloneHelper &h) override {
    auto input = static_cast<NDViewLoad *>(h.GetClone(static_cast<NDObject *>(input_)));
    return new NDViewBindLoad(input);
  }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "ViewBindLoad"; }
 private:
  NDViewLoad *input_;
  std::vector<RelocAddr *> bind_addrs_;
  int addr_used_{0};
};

struct VisitState {
  uint32_t cnt{0};
  std::vector<uint64_t> mask;
  std::vector<uint32_t> visit;
  std::unordered_map<NDObject *, uint64_t> view_index;
  std::vector<NDObject *> joins;
  std::vector<std::pair<NDObject *, uint64_t>> stack;
  std::unordered_set<NDObject *> join_seen;
  std::vector<NDObject *> objects;
};

void CollectJoins(NDObject *root, VisitState &st) {
  st.cnt++;
  st.stack.clear();
  st.join_seen.clear();
  st.joins.clear();
  auto visit = [&](NDObject *obj, uint64_t mask) {
    auto &seen = st.visit[obj->index_];
    if (seen != st.cnt) {
      seen = st.cnt;
      st.mask[obj->index_] = mask;
      st.stack.push_back({obj, mask});
      return;
    }
    auto merged = st.mask[obj->index_] | mask;
    if (merged != st.mask[obj->index_]) {
      st.mask[obj->index_] = merged;
      st.stack.push_back({obj, merged});
      if ((!obj->IsLoad() || obj->GetObjectType() == kViewLoad) && st.join_seen.insert(obj).second) st.joins.push_back(obj);
    }
  };
  visit(root, 0);
  while (!st.stack.empty()) {
    auto [top, top_mask] = st.stack.back();
    st.stack.pop_back();
    auto mask = top_mask;
    if (top->IsViewOp()) {
      NDObject *key = top;
      if (top->GetObjectType() == kSplitOp && static_cast<SplitOp *>(top)->slice_idx_ != 0)
        key = static_cast<SplitOp *>(top)->main_;
      mask |= st.view_index.try_emplace(key, 1ULL << st.view_index.size()).first->second;
    }
    top->ForInput([&](NDObject *in) { visit(in, mask); });
  }
}

struct DiamondCloneHelper final : public CloneHelper {
  std::unordered_map<NDObject *, NDObject *> map_;
  BasicBlock *bb_{nullptr};
  NDObject *pos_{nullptr};
  explicit DiamondCloneHelper(BasicBlock *bb, NDObject *pos) : bb_(bb), pos_(pos) {}
  NDObject *Insert(NDObject *clone) {
    std::vector<NDObject *> stuff_ops;
    clone->Normalize(stuff_ops);
    ASSERT(stuff_ops.empty());
    bb_->Insert(pos_, clone);
    return clone;
  }
  NDObject *GetClone(NDObject *op) override {
    auto it = map_.find(op);
    if (it != map_.end()) return it->second;
    if (op->IsLoad()) {
      auto input = static_cast<NDAccess *>(op);
      while (input->CheckFlag(OBJ_FLAG_LOAD_BIND)) input = input->LoadBind();
      NDObject *bind;
      if (input->GetObjectType() == kViewLoad)
        bind = Insert(new NDViewBindLoad(static_cast<NDViewLoad *>(input)));
      else
        bind = Insert(new NDBindLoad(input));
      bb_->AddUser(input, bind);
      return bind;
    }
    if (op->GetObjectType() == kSplitOp) {
      auto sib = static_cast<SplitOp *>(op);
      auto main = sib->main_;
      if (!map_.count(main)) {
        auto mc = new SplitOpM(GetClone(main->lhs_), main->split_axis_ref_, main->split_size_,
                               main->siblings_.size());
        bb_->Insert(pos_, mc);
        map_[main] = mc;
        for (size_t j = 1; j < main->siblings_.size(); ++j) {
          auto cj = mc->AddSibling();
          bb_->Insert(pos_, cj);
          map_[main->siblings_[j]] = cj;
        }
        std::vector<NDObject *> stuff_ops;
        mc->Normalize(stuff_ops);
        ASSERT(stuff_ops.empty());
      }
      return map_[op];
    }
    map_[op] = Insert(op->CloneUpdate(*this));
    return map_[op];
  }
  IntArrayRef *GetClone(IntArrayRef *shape) override { return shape; }
  ScalarRef *GetClone(ScalarRef *scalar) override { return scalar; }
  void SetClone(NDObject *op, NDObject *clone) override { map_[op] = clone; }
};

bool RepairFork(BasicBlock &bb, NDObject *fork) {
  std::vector<NDObject *> uniq;
  std::unordered_set<NDObject *> seen;
  for (auto u : bb.GetUsers(fork)) {
    auto rep = u->GetObjectType() == kSplitOp ? static_cast<NDObject *>(static_cast<SplitOp *>(u)->main_) : u;
    if (seen.insert(rep).second) uniq.push_back(rep);
  }
  bool split_only = false;
  if (uniq.size() < 2) {
    if (uniq.size() == 1 && uniq[0]->GetObjectType() == kSplitOp) {
      auto main = static_cast<SplitOp *>(uniq[0])->main_;
      split_only = std::count_if(main->siblings_.begin(), main->siblings_.end(),
                                 [&](SplitOp *s) { return s->lhs_ == fork; }) >= 2;
    }
    if (!split_only) return false;
  }

  for (size_t u = split_only ? 0 : 1; u < uniq.size(); ++u) {
    auto user = uniq[u];
    if (user->IsStore()) continue;
    DiamondCloneHelper helper(&bb, fork);
    auto fork_clone = helper.GetClone(fork);
    if (user->GetObjectType() == kSplitOp) {
      for (auto sib : static_cast<SplitOp *>(user)->main_->siblings_)
        if (sib->lhs_ == fork) { bb.UpdateInput(sib, fork, fork_clone); break; }
    } else {
      bb.UpdateInput(user, fork, fork_clone);
    }
    return true;
  }
  return false;
}

bool RepairOneRound(BasicBlock &bb, VisitState &st) {
  st.objects.clear();
  int idx = 0;
  for (NDObject *iter = bb.Begin(); iter != bb.End(); iter = bb.Next(iter)) {
    iter->index_ = idx++;
    st.objects.push_back(iter);
  }
  st.mask.assign(st.objects.size(), 0);
  st.visit.assign(st.objects.size(), 0);
  st.view_index.clear();
  for (auto store : st.objects) {
    if (!store->IsStore() || !bb.GetUserNum(store->lhs_)) continue;
    CollectJoins(store->lhs_, st);
    for (auto fork : st.joins) {
      if (RepairFork(bb, fork)) return true;
    }
  }
  return false;
}

}  // namespace

void EliminateDiamondView(BasicBlock &bb) {
  int view_cnt = 0;
  for (NDObject *iter = bb.Begin(); iter != bb.End(); iter = bb.Next(iter)) {
    if (iter->IsViewOp() && (iter->obj_id_ != kSplitOp || static_cast<SplitOp *>(iter)->slice_idx_ == 0)) {
      view_cnt++;
    }
  }
  if (view_cnt < 2) return;
  VisitState st;
  for (int i = 0; i < 8; ++i) {
    if (!RepairOneRound(bb, st)) return;
  }
}

#ifdef DVM_ARCH_GPU
class PassOptimizerGpu : public PassOptimizer {
 public:
  void RunPass(BasicBlock &bb, bool dyn_shape) override {
    DeadCodeEliminate(bb);
    if (!dyn_shape) {
      EliminateReshape(bb);
    }
    EliminateDiamondView(bb);
    CompactPeakLiveness(bb);
    ReorderLoad(bb);
    ReorderStore(bb);
    VectorDoubleBuffer(bb);
  }
};
PassOptimizer *CreateOptimizer(AiCoreArch arch) {
  (void)arch;
  return new PassOptimizerGpu();
}
#else
class PassOptimizerC220 : public PassOptimizer {
 public:
  void RunPass(BasicBlock &bb, bool dyn_shape) override {
    DeadCodeEliminate(bb);
    if (!dyn_shape) {
      EliminateReshape(bb);
    }
    EliminateDiamondView(bb);
    CompactPeakLiveness(bb);
    ReorderLoad(bb);
    ReorderStore(bb);
    InsertRemovePad(bb);
    VectorDoubleBuffer(bb);
  }
};

class PassOptimizerC310 : public PassOptimizer {
 public:
  void RunPass(BasicBlock &bb, bool dyn_shape) override {
    DeadCodeEliminate(bb);
    if (!dyn_shape) {
      EliminateReshape(bb);
    }
    EliminateDiamondView(bb);
    CompactPeakLiveness(bb);
    VfFusion(bb);
    ReorderLoad(bb);
    ReorderStore(bb);
    VectorDoubleBuffer(bb);
  }
};

PassOptimizer *CreateOptimizer(AiCoreArch arch) {
  if (arch == kAiCore_C220) {
    return new PassOptimizerC220();
  }
  return new PassOptimizerC310();
}
#endif // DVM_ARCH_GPU
}  // namespace dvm::pass
