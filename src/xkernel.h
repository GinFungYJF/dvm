/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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

#ifndef _DVM_X_KERNEL_H_
#define _DVM_X_KERNEL_H_

#include <string>
#include <vector>
#include "ops_m.h"
#include "kernel.h"

namespace dvm {
class CubeKernel : public VKernel {
 public:
  CubeKernel(KernelType ktype = KernelType::kCube, uint32_t flags = 0) : VKernel(ktype, flags), tuner_(g_system.online_tuner_) {}
  ~CubeKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;

  void SetTuner(CubeTuner *tuner) { tuner_ = tuner; }
  size_t ReserveCodeSize() { return sizeof(vCubeOp); }
  void NormalizeCube() {
    if (cube_op_->output_ == nullptr) {
      auto output = new NDStore(nullptr, cube_op_);
      output->SetFlag(OBJ_FLAG_STAGE_IO);
      cube_op_->output_ = output;
    }
    std::vector<NDObject *> empty_run_ops;
    cube_op_->lhs_->Normalize(empty_run_ops);
    cube_op_->rhs_->Normalize(empty_run_ops);
    cube_op_->NormalizeCube();
    cube_op_->output_->Normalize(empty_run_ops);
    cube_op_->TryBatchFold();
  }
  uint8_t *DoCodeGen(uint8_t *code_ptr, uint64_t core_limit);
  uint64_t DoCodeGen() {
    code_.Alloc(code_.HeadSize() + ReserveCodeSize());
    code_.data_size_ =  DoCodeGen(code_.data_ + code_.HeadSize(), g_system.CoreNum(CoreType::kAIC)) - code_.data_;
    code_.UpdateC();
    return 0;
  }
  void Clear() {
    code_.Clear();
    cube_op_->Clear();
  }
  CubeOp *GetCube() const { return cube_op_; }

 protected:
  CubeOp *cube_op_{nullptr};
  CubeTuner *tuner_;
  bool reload_rhs_{false};
};

class MixKernelBase : public CubeKernel {
 public:
  MixKernelBase(KernelType ktype = KernelType::kMix, uint32_t flags = 0) : CubeKernel(ktype, flags) {}
  ~MixKernelBase() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void Inspect(Inspector &sp) override;

  void NormalizePost() {
    if (post_fusion_ && !post_fusion_->NormBuild()) {
      DvmException("MixKernel broker affine failed");
    }
  }
  size_t ReserveCodeSize() {
    size_t size = CubeKernel::ReserveCodeSize();
    if (post_fusion_) size += post_fusion_->ReserveCodeSize();
    return size;
  }

  struct GenOut {
    GenOut(uint8_t *c, uint64_t w, uint64_t e) : code_end(c), ws_size(w), entry(e) {}
    uint8_t *code_end;
    uint64_t ws_size;
    uint64_t entry;
  };
  GenOut DoCodeGen(uint8_t *code_ptr, uint64_t core_limit, size_t code_reserve);
  uint64_t DoCodeGen() {
    if (post_fusion_ == nullptr) {
      return CubeKernel::DoCodeGen();
    }
    size_t code_reserve = ReserveCodeSize();
    code_.Alloc(code_.HeadSize() + code_reserve);
    auto out = DoCodeGen(code_.data_ + code_.HeadSize(), g_system.CoreNum(CoreType::kAIC), code_reserve);
    code_.data_size_ = out.code_end - code_.data_;
    code_.UpdateMix(out.entry);
    return out.ws_size;
  }
  void Clear() {
    CubeKernel::Clear();
    if (post_fusion_) {
      post_fusion_->Clear();
    }
  }
 
 protected:
  VKernelS *post_fusion_{nullptr};
  NDAccess *sload_{nullptr};
  RelocAddr gm_pos_;
  std::vector<std::pair<NDAccess *, NDAccess *>> reloads_;
};

class StagesKernel;
class MixKernel : public MixKernelBase {
 public:
  MixKernel(uint32_t flags = 0) : MixKernelBase(KernelType::kMix, flags) {}
  ~MixKernel() override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;

 protected:
  uint64_t StageCodeGen(const CubeOp::Tactics &tactics);
  StagesKernel *stage_kernel_{nullptr};
  std::vector<NDObject *> mng_;
};

class DynMixKernel : public MixKernel {
 public:
  DynMixKernel() : MixKernel(KernelFlag::kDynamic) { tuner_ = g_system.dyn_lazy_tuner_; }
  uint64_t CodeGen() override;

 protected:
  GraphTracker tracker_;
};

class ParallelKernel : public VKernel {
 public:
  explicit ParallelKernel(uint32_t flags) : VKernel(KernelType::kParallel, flags) {}
  ~ParallelKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void AddKernel(KernelType type, uint32_t flags, size_t thread_limit);

  struct Node {
    Node() = default;
    Node(VKernel *k, uint32_t limit) : kernel(k), core_limit(limit) {}
    VKernel *kernel;
    uint32_t core_limit;
    uint32_t code_reserve;
    uint64_t wload;
  };

 protected:
  uint64_t CodeGenVE(VKernelS *kernel, RedVisitCoder *visit, uint8_t *code_begin, uint64_t code_size, uint64_t ws_size);

  class _IsolateWrap : public CodeWrap {
   public:
    int LaunchWrap(void *workspace, void *stream) override;
    void DasWrap(std::ostringstream &oss) override;
    void CollectWrap(std::vector<Code *> &codes) override;
    std::vector<Code *> codes_;
    bool term_{false};
  };
  std::vector<Node> mixes_;
  std::vector<Node> cubes_;
  std::vector<Node> vectors_;
  VKernel *current_{nullptr};
  _IsolateWrap *wrap_{nullptr};
};

class StageCodeWrap : public CodeWrap {
 public:
  explicit StageCodeWrap(StagesKernel *kernel) : kernel_(kernel) {}
  int LaunchWrap(void *workspace, void *stream) override;
  void DasWrap(std::ostringstream &oss) override;
  void CollectWrap(std::vector<Code *> &codes) override;

 private:
  StagesKernel *kernel_;
};

class StagesKernel : public VKernel {
 public:
  StagesKernel(uint32_t flags = 0) : VKernel(KernelType::kSequence, flags), code_wrap_(this) {}
  ~StagesKernel() override;

  struct SLoadInfo {
    SLoadInfo(NDAccess *l, NDAccess *s) : load(l), store(s) {}
    NDAccess *load;
    NDAccess *store;
  };

  struct SStoreInfo {
    SStoreInfo(NDAccess *s, bool o) : store(s), is_out(o) {}
    NDAccess *store;
    bool is_out;
  };

  struct Stage {
    explicit Stage(VKernel *k) : kernel(k) {}
    virtual ~Stage() { delete kernel; }
    VKernel *kernel;
    int64_t ws_size{-1};
    union {
      int64_t ws_offset{-1};
      Stage *group_next;
    };
    std::vector<SLoadInfo> sloads_;
    std::vector<SStoreInfo> sstores_;

    void StageStore(NDAccess *store, bool is_out = false) { sstores_.emplace_back(store, is_out); }
    void StageLoad(NDAccess *load, NDAccess *store) { sloads_.emplace_back(load, store); }
    void Reset() {
      ws_size = -1;
      ws_offset = -1;
      sloads_.clear();
      sstores_.clear();
    }
  };
  void Reset() {
    stages_.clear();
    code_.Clear();
  }
  void AppendStage(Stage *stage) { stages_.push_back(stage); }
  Stage *StageAt(size_t idx) const { return stages_[idx]; }
  void AddStage(VKernel *k) { stages_.push_back(new Stage(k)); }

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;

 protected:
  static void SetWorkspace(NDAccess *op, int64_t offset) { op->addr_.ws = offset; }
  static int64_t GetWorkspace(NDAccess *op) { return op->addr_.ws; }
  static void SetOutputReuse(NDAccess *op, NDAccess *store) { op->addr_.gm = static_cast<void *>(store); }
  static NDAccess *GetOutputReuse(NDAccess *op) { return static_cast<NDAccess *>(op->addr_.gm); }

  uint64_t AllocWorkspace();

  std::vector<Stage *> stages_;
  StageCodeWrap code_wrap_;
  friend StageCodeWrap;
};

class SequenceKernel : public StagesKernel {
 public:
  SequenceKernel(uint32_t flags) : StagesKernel(flags) {}
  void Append(NDObject *obj) override;
 protected:
  static int GetStage(NDObject *obj) { return obj->prop_id_; }
  static void SetStage(NDObject *obj, int area_id) { obj->prop_id_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }
};

class SplitContext {
 public:
  enum { MEM_BLOCK_SIZE = 32 };
  struct MemNode {
    size_t size;
    void *addr;
    MemNode *next;
  };

  SplitContext();
  ~SplitContext();

  MemNode *Alloc(size_t size) { return mem_head_ && size <= mem_tail_->size ? DoAlloc(size) : nullptr; }
  MemNode *DoAlloc(size_t size);
  void Free(void *addr, size_t size);
  void MemReset() {
    mem_head_ = nullptr;
    mem_pos_ = 0;
  }

  std::vector<NDAccess *> gen_;
  std::vector<std::pair<NDAccess *, size_t>> kill_;
  std::vector<NDObject *> build_;

  using SlotWorkspace = std::vector<std::pair<size_t, RelocAddr *>>;
  SlotWorkspace slot_ws_;

 protected:
  std::vector<MemNode *> mem_blocks_;
  MemNode *mem_head_;
  MemNode *mem_tail_;
  uint64_t mem_pos_;
};

class EagerVector;
class EagerArea;
class _SplitKernel : public VKernel {
 public:
  _SplitKernel(KernelType type, uint32_t flags);
  ~_SplitKernel() override;

  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  std::string &DisAssemble() override;
  virtual void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc);
  int Launch(void *stream) override;

  NDObject *AppendCube(CubeOp *mm);
  void SlotCodeGen(const RelocEntry *relocs, size_t reloc_size);

  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }
  static void SetStoreInplace(NDObject *store, int flag) { store->reuse_dep_ = flag; }
  static CubeOp *GetCubeOp(EagerVector *kernel);

 protected:
  static int GetArea(NDObject *obj) { return obj->prop_id_; }
  static void SetArea(NDObject *obj, int area_id) { obj->prop_id_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static int GetStoreInplace(NDObject *store) { return store->reuse_dep_; }
  static void SetStoreSize(NDObject *store, uint64_t size) { store->xbuf_ = size; }
  static uint64_t GetStoreSize(NDObject *store) { return store->xbuf_; }
  static NDAccess *GetRecentLoad(NDAccess *store) { return reinterpret_cast<NDAccess *>(store->addr_.reloc_); }
  static void SetRecentLoad(NDAccess *store, NDAccess *load) { store->addr_.reloc_ = reinterpret_cast<uint64_t *>(load); }

  void Reset() {
    objects_.clear();
    area_used_ = 0;
    kernel_used_ = 0;
  }

  void InitObjInfo(NDObject *obj, int area = -1) {
    SetArea(obj, area);
    SetStore(obj, nullptr);
  }

  void InitStoreInfo(NDObject *store, int area = -1) {
    InitObjInfo(store, area);
    SetStoreSize(store, store->Size());
    SetStoreInplace(store, 0);
    SetRecentLoad(static_cast<NDAccess *>(store), nullptr);
  }

  void *AllocWS(NDAccess *store, WsAllocator *alloc) {
    auto size = GetStoreSize(store);
    if (unlikely(size == 0)) {
      store->addr_.data = uint64_t(-1);
    } else if (auto mem = ctx_->Alloc(size); mem != nullptr) {
      store->addr_.gm = mem->addr;
      SetStoreSize(store, mem->size);
    } else {
      store->addr_.gm = alloc->Alloc(size);
    }
    return store->addr_.gm;
  }

  void Split(NDObject *root, NDObject *dom, uint64_t pattern = 0);
  NDObject *Exchange(NDObject *input, int to_aid);
  void BuildKernel(EagerVector *kernel, const EagerArea *area, WsAllocator *alloc);
  void RelocBinds();

  std::vector<std::pair<EagerArea *, EagerArea *>> areas_;
  std::vector<EagerVector *> kernels_;
  int area_used_{0};
  int kernel_used_{0};
  union {
    uint64_t extern_code_data_{0};
    void *extern_code_;
  };
  std::vector<NDObject *> objects_;
  std::vector<NDObject *> temp_vec_;
  std::vector<NDObject **> exchange_cache_;
  SplitContext *__restrict__ ctx_;
  friend EagerArea;
};

class VKernelE : public _SplitKernel {
 public:
  VKernelE();
  ~VKernelE() override;
  void Normalize() override;
  void Clear() {
    for (auto op : objects_) {
      op->~NDObject();
      NDObject::mem_pool_.Put(op);
    }
    Reset();
  }
};

class _SplitGraph : public _SplitKernel {
 public:
  _SplitGraph(uint32_t flags);
  ~_SplitGraph() override;
  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Normalize() override;

 protected:
  std::vector<NDObject *> build_ops_;
};

class SplitGraphD : public _SplitGraph {
 public:
  SplitGraphD(uint32_t flags = 0) : _SplitGraph(flags | KernelFlag::kDynamic) {}
  void Append(NDObject *obj) override;
  void Normalize() override;
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
  void Clone(VKernel *base, CloneHelper &helper) override;

 protected:
  GraphTracker tracker_;
};

class SplitGraphS : public _SplitGraph {
 public:
  SplitGraphS(uint32_t flags) : _SplitGraph(flags) {}
  void Normalize() override;
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;

 protected:
  SplitContext::SlotWorkspace slot_ws_;
  uint64_t ws_size_{0};
};

class SplitEagerW : public VKernelE {
 public:
  SplitEagerW() :  VKernelE() { flags_ |= KernelFlag::kUnifyWS; }
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
};

class SplitGraphDW : public SplitGraphD {
 public:
  SplitGraphDW() : SplitGraphD(KernelFlag::kUnifyWS) {}
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
};

class SplitEagerLazy : public VKernelE {
 public:
  SplitEagerLazy() : VKernelE() {}
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
  int Launch(void *stream) override;

 protected:
  uint64_t AllocArea(const EagerArea *area, WsAllocator *ws_alloc);
  void GenKernel(EagerVector *kernel);
  void *ws_mem_{nullptr};
  void (*kernel_init_func_)(SplitEagerLazy *, EagerVector *, NDObject *){nullptr};
};

class SplitEagerLazyW : public SplitEagerLazy {
 public:
  SplitEagerLazyW() : SplitEagerLazy() {}
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
  static void RelocKernel(SplitEagerLazy *self, EagerVector *kernel, NDObject *dom);

 protected:
  struct _SlotWs : public WsAllocator {
    void *Alloc(uint64_t size) override;
    void Reloc(RelocAddr &addr) {
      if (uint64_t idx = addr.data - 1; idx < slots_.size()) {
        addr.gm = slots_[idx];
      }
    }
    void *Reloc(void *addr) {
      uint64_t idx = reinterpret_cast<uint64_t>(addr) - 1;
      return idx < slots_.size() ? slots_[idx]: addr;
    }
    std::vector<void *> slots_;
    size_t acc_size_;
  };
  _SlotWs slot_ws_;
};
}  // namespace dvm
#endif  // _DVM_X_KERNEL_H_
