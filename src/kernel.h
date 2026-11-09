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

#ifndef _DVM_KERNEL_H_
#define _DVM_KERNEL_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "code.h"
#include "ops.h"
#include "ops_c.h"
#include "pass.h"

namespace dvm {
class MsprofHelper;
class IdleCleanWrap;

class Inspector {
 public:
  std::ostringstream &Warn(const char *type) {
    oss_ << "[W][" << type << "] ";
    return oss_;
  }
  std::ostringstream &Error(const char *type) {
    oss_ << "[E][" << type << "] ";
    return oss_;
  }
  std::ostringstream oss_;
};

#define K_FLAG_BEGIN ((uint32_t)KernelFlag::kSpeculate)
#define K_FLAG_SIMT (K_FLAG_BEGIN << 1)
#define K_FLAG_DIS_DUP_TILING (K_FLAG_BEGIN << 2)

class VKernel {
 public:
  VKernel(KernelType ktype, uint32_t flags) : ktype_(ktype), flags_(flags) {}
  virtual ~VKernel();

  virtual void Append(NDObject *obj);
  virtual void Normalize();
  virtual void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc);
  virtual int Launch(void *stream);
  virtual uint64_t CodeGen();
  virtual void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) = 0;
  virtual void Inspect(Inspector &sp);
  virtual void Clone(VKernel *base, CloneHelper &helper);
  std::string &DumpGraph(bool rgraph = false) {
    std::ostringstream oss;
    Dump(oss, "", rgraph);
    dump_str_ = oss.str();
    return dump_str_;
  }
  std::string &Inspect() {
    Inspector sp;
    Inspect(sp);
    dump_str_ = sp.oss_.str();
    return dump_str_;
  }
  virtual std::string &DisAssemble();
  KernelType KType() const { return ktype_; }
  uint32_t Flags() const { return flags_; }
  bool IsSplit() const { return ktype_ == KernelType::kSplit ||  ktype_ == KernelType::kEager; }
  bool IsDynamic() const { return flags_ & KernelFlag::kDynamic; }

  void UpdateIdle(const std::vector<NDObject *> &cleans);
  void UpdatePreWS(void *mem) { pre_ws_mem_ = mem; }
  void SetNameHint(const char *name, const char *fullname) {
    op_name_ = name;
    op_fullname_ = fullname;
  }

  Code code_;

 protected:
  KernelType ktype_;
  uint32_t flags_;
  union {
    size_t pre_ws_size_{0};
    void *pre_ws_mem_;
  };
  MsprofHelper *msprof_{nullptr};
  const char *op_name_{nullptr};
  const char *op_fullname_{nullptr};
  IdleCleanWrap *idle_clean_wrap_{nullptr};
  std::string dump_str_;
};

class VectorKernel : public VKernel {
 public:
  explicit VectorKernel(KernelType ktype, uint32_t flags) : VKernel(ktype, flags) {
    MESS(max_type_, 100);
    MESS(min_type_, 200);
    MESS(visit_, reinterpret_cast<VisitCoder *>(100));
    // Reserve the zero instruction terminator before Mix calls TileCollect.
    tile_info_.code_reserve = sizeof(uint64_t);
  }
  ~VectorKernel() override = default;

  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;

  void BuildDomain();
  void PrepareTiling();

  uint8_t *DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, uint64_t code_reserve);
  uint64_t DoCodeGenInner(uint64_t core_limit) {
    auto code_reserve = ReserveCodeSize();
    code_.Alloc(code_reserve + code_.HeadSize());
    auto code_end = DoCodeGen(core_limit, code_.data_ + code_.HeadSize(), code_reserve);
    code_.data_size_ = code_end - code_.data_;
    if (auto visit = GetVisitor<RedVisitCoder>(); visit != nullptr) {
      code_.block_dim_ = CeilDiv<uint32_t>(visit->block_num_, 2);
      code_.UpdateVE(visit);
      return visit->ws_size_;
    }
    code_.block_dim_ = CompactBlockDim(core_limit);
    code_.UpdateV(tile_num_, flags_ & K_FLAG_SIMT);
    return 0;
  }
  uint64_t DoCodeGen(uint64_t core_limit) {
    PrepareTiling();
    if (unlikely(!tile_size_)) {
      ProcessIdle();
      return 0;
    }
    return DoCodeGenInner(core_limit);
  }

  uint8_t *DoTileGen(int64_t live_peak, uint8_t *code_ptr, uint64_t code_reserve);
  uint64_t TileGen(int64_t live_peak, uint64_t core_limit) {
    auto code_reserve = ReserveCodeSize();
    code_.Alloc(code_reserve + code_.HeadSize());
    code_.block_dim_ = core_limit;
    auto code_end = DoTileGen(live_peak, code_.data_ + code_.HeadSize(), code_reserve);
    code_.data_size_ = code_end - code_.data_;
    if (auto visit = GetVisitor<RedVisitCoder>(); visit != nullptr) {
      code_.block_dim_ = CeilDiv<uint32_t>(visit->block_num_, 2);
      code_.UpdateVE(visit);
      return visit->ws_size_;
    }
    code_.block_dim_ = CompactBlockDim(core_limit);
    code_.UpdateV(tile_num_, flags_ & K_FLAG_SIMT);
    return 0;
  }

  void Shard(const ShardParam &sp) {
    for (auto op : objects_) {
      op->Shard(sp);
    }
    shard_ = &sp;
  }
  void ClearShard() { shard_ = nullptr; }

  int MaxType() const { return max_type_; }
  int MinType() const { return min_type_; }
  uint64_t LeadAlign() const { return lead_align_; }
  const DimArray &DimSpace() const { return dom_->nd_.dims(); }
  int64_t AnalyzeTileSizeLimit() { return TileSizeLimit(Analyze()); }

  uint64_t ReserveCodeSize() const {
    auto res = objects_.size() * V_INSN_SIZE_MAX + tile_info_.code_reserve;
    return (res + 511ul) & ~511ul;  // 512B align
  }

  uint32_t CompactBlockDim(uint64_t core_limit) {
    auto tile_per_block = (tile_num_ + core_limit - 1) / core_limit;
    return (tile_num_ + tile_per_block - 1) / tile_per_block;
  }

  void InOutReusePlan(const DimArray *dom = nullptr);
  template <typename T>
  NDAccess *InOutReuseFind(NDAccess *load, const T &check) {
    if (auto index = load->index_; index < 64) {
      for (size_t i = load_num_; i < static_ops_.size(); ++i) {
        auto store = static_cast<NDAccess *>(static_ops_[i]);
        if ((store->io_reuse_mask_ & (1ull << index)) && store->type_id_ == load->type_id_ && check(store)) {
          return store;
        }
      }
    }
    return nullptr;
  }

  void CollectIdle(std::vector<NDObject *> &cleans);
  void ProcessIdle();

  template <typename T>
  T *GetVisitor() {
    // TODO(multi visitor):
    //   if (type_id_ != T::ID) return next_->GetCoder<T>();
    return static_cast<T *>(visit_);
  }

  void AddVisitor(VisitCoder *visit) {
    ASSERT(visit_ == nullptr);
    visit_ = visit;
  }

  std::vector<NDObject *> objects_;
  CommOp *comm_op_{nullptr};

  int64_t tile_num_{0};
  int64_t tile_size_;  // shape size of object tile size
  int64_t align_space_;  // space of align depth [0, lead_depth-1]
  TileInfo tile_info_;
  NDObject *dom_;
  const ShardParam *shard_{nullptr};

  VisitCoder *visit_;

  uint64_t local_mem_size_;
  union {
    uint64_t block_align_;  // tiling
    uint64_t lead_align_;   // codegen
  };

  size_t load_num_{0};
  std::vector<NDObject *> static_ops_;

  struct TileUpdate {
    int64_t tile_num;
    int64_t tile_size;
  };
  struct TileRegion {
    void Add(int start, int64_t space, int64_t num, int64_t tile) {
      last_num = num;
      last_tile = tile;
      starts[depth] = start;
      spaces[depth] = space;
      depth++;
    }
    void Reset() {
      depth = 0;
      last_num = 0;
      last_tile = 0;
      tail_size = 0;
      tail_dim = -1;
    }
    int64_t last_num;
    int64_t last_tile;
    int64_t tail_size;
    int tail_dim;
    int depth;
    int starts[DimArray::kMaxDimSize];
    int64_t spaces[DimArray::kMaxDimSize];
  };

  TileRegion tile_region_;

  int64_t GetTailSize(const NDSpaceData *ndd) const {
    auto tail_dim = tile_region_.tail_dim;
    return tail_dim >= 0 && (ndd->pointwise_tile_mask >> tail_dim) ? tile_region_.tail_size : 0;
  }
  int GetTailDim() const { return tile_region_.tail_dim; }

  int64_t Analyze();
  int64_t TileSizeLimit(int64_t live_peak);
  void ShapeTiling(int64_t size_limit, int64_t core_limit, TileUpdate &update);
  template <bool dyn_shape>
  void Optimize(std::vector<NDObject *> &objects, std::vector<NDObject *> &mng) {
    if (auto opt = g_system.pass_opt_) {
      if (dyn_shape) {
        opt->RunD(objects, mng, nullptr);
      } else {
        opt->Run(objects, mng, nullptr);
      }
    }
  }
  void ApplyTiling(const TileUpdate &update);
  void AlignSimd(int64_t tile_size_limit);

 protected:
  void TileProp(const TileParam &tp) {
    for (auto op : objects_) {
      op->Tile(tp);
    }
  }

  void StaticAppend(NDObject *obj) {
    int type = obj->type_id_;
    if (type > max_type_) {
      max_type_ = type;
    } else if (type < min_type_) {
      min_type_ = type;
    }
    if (obj->IsLoad()) {
      if (load_num_ == static_ops_.size()) {
        static_ops_.push_back(obj);
      } else {
        static_ops_.insert(static_ops_.begin() + load_num_, obj);
      }
      load_num_++;
    } else if (obj->IsStore()) {
      static_ops_.push_back(obj);
    }
  }

  int max_type_;
  int min_type_;
  friend class CodeGenHelper;
};

class SchGenHelper;
class DupTilingSchGen;

class VKernelS : public VectorKernel {
 public:
  VKernelS(uint32_t flags = 0) : VectorKernel(KernelType::kVector, flags) {}
  ~VKernelS() override;
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void Inspect(Inspector &sp) override;

  bool NormBuild();

  bool Normalize(bool broker_norm) {
    if (broker_norm && broker_num_ == -1) {
      BrokerInit();
    }
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      objects_.emplace_back(op);
    }
    return !broker_norm || BrokerAffine();
  }

  void StaticInit(const std::vector<NDObject *> &objects);
  void SchInit(const std::vector<NDObject *> &objects);
  void BrokerInit();
  bool BrokerAffine();
  uint64_t BrokerCodeGen(VKernel **hold_kernel);
  int64_t DupTilingGen(const TileRegion &region, int64_t tile_size, int64_t tile_size_limit);

  void Clear() {
    code_.Clear();
    objects_.clear();
  }

  void SetTile(int start, int end, int64_t num, int64_t factor) {
    tiles_.emplace_back(DimTile{start, end, num, factor});
  }

  std::vector<NDObject *> build_ops_;

 protected:
  void ManualTiling();

  int broker_num_{-1};
  int last_broker_;
  VKernel *stage_kernel_{nullptr};
  SchGenHelper *sch_gen_{nullptr};
  DupTilingSchGen *dup_gen_{nullptr};

  struct DimTile {
    int start;
    int end;
    int64_t num;
    int64_t factor;
  };
  std::vector<DimTile> tiles_;
  std::vector<NDObject *> adopt_ops_;
};

class VKernelD : public VKernelS {  // TODO: remove VKernelD
 public:
  VKernelD(uint32_t flags = KernelFlag::kDynamic) : VKernelS(flags) {}
};

class _SpecVector : public VKernelD {
 public:
  _SpecVector(uint32_t flags) : VKernelD(flags | KernelFlag::kSpeculate) {}
  ~_SpecVector() override;
  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void Next() { last_stage_++; }

 protected:
  void BuildFallKernel(bool dyn_shape);
  bool use_fall_{false};
  int last_stage_{0};
  std::vector<int> stage_ids_;
  std::vector<NDObject *> post_reduces_;
  VKernel *fall_kernel_{nullptr};
};

template <bool dyn_shape>
class SpecVector : public _SpecVector {
 public:
  SpecVector() : _SpecVector(dyn_shape ? KernelFlag::kDynamic : 0) {}
  uint64_t CodeGen() override;
  uint64_t FallCodeGen();
};

class StagesKernel;
class SpecVecStage;
class SpecVecContext {
 public:
  SpecVecContext() = default;
  ~SpecVecContext();

  void Reset() {
    stage_size_ = 0;
    for (size_t i = spec_begin_; i < spec_ops_.size(); ++i) {
      delete spec_ops_[i];
    }
    spec_ops_.resize(spec_begin_);
    tracker_.RecoverClear();
  }
  void ResetSpec() { area_size_ = 0; }

  int AssignArea() {
    if (int size = static_cast<int>(areas_.size()); area_size_ == size) {
      areas_.resize(size + 8);
    }
    auto aid = area_size_++;
    areas_[aid].parent = aid;
    areas_[aid].child_mask = 0;
    areas_[aid].u64 = 0;
    return aid;
  }
  void MergeArea(int aid, int src_aid) {
    uint64_t src_group = (1ull << src_aid) | areas_[src_aid].child_mask;
    areas_[aid].child_mask |= src_group;
    while (src_group) {
      auto b = __builtin_ctzll(src_group);
      src_group &= src_group - 1;
      areas_[b].parent = aid;
    }
    areas_[src_aid].child_mask = 0;
  }
  int RootArea(int aid) const {
    ASSERT(aid >= 0);
    return areas_[aid].parent;
  }

  struct Area {
    int parent;
    union {
      uint64_t child_mask;
      SpecVecStage *stage;
    };
    union {
      uint64_t u64;
      struct {
        uint32_t ext_opt;
        uint32_t u32;
      };
    };
    uint32_t &MergeMask() { return ext_opt; }
    uint32_t &UnMergeMask() { return u32; }
    void ClearMask() { u64 = 0; }
  };

  std::vector<Area> areas_;
  int area_size_{0};
  size_t stage_size_{0};
  StagesKernel *stage_k_{nullptr};
  std::vector<SpecVecStage *> stage_pool_;
  std::vector<NDObject *> spec_ops_;
  size_t spec_begin_{0};
  GraphTracker tracker_;
};

class SpecVecBase : public VKernelS {
 public:
  using Area = SpecVecContext::Area;
  SpecVecBase(uint32_t flags, SpecVecContext &ctx) : VKernelS(flags), ctx_(ctx) {}
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;

  void Reset() {
    min_type_ = kDataTypeEnd;
    max_type_ = 0;
    objects_.clear();
    code_.Clear();
    load_num_ = 0;
    static_ops_.clear();
  }

 protected:
  bool SpecBuild();
  bool BroadcastSpec();
  bool ReduceSpec();
  bool ReshapeSpec();
  bool CustomSpec();
  bool PermuteSpec();

  struct OpMeta {
    void SetCut() { cut_mark = CUT_MARK; }
    void UnCut() { cut_mark = IO_END; }
    bool IsCut() const { return cut_mark == CUT_MARK; }
    int32_t aid;
    union {
      uint16_t store;
      uint16_t cut_mark;
    };
    uint16_t recent_load;
    static constexpr uint16_t IO_END = 0xfffu;
    static constexpr uint16_t CUT_MARK = IO_END + 1;
  };
  void InitMeta(NDObject *op) { op->insn_ = reinterpret_cast<uint64_t *>(-1); }
  OpMeta *__restrict__ GetMeta(NDObject *op) const { return reinterpret_cast<OpMeta *__restrict__ >(&op->insn_); }

  int64_t LazyTileLimit() {
    if (tile_limit_ < 0) {
      live_peak_ = Analyze();
      tile_limit_ = TileSizeLimit(live_peak_);
    }
    return tile_limit_;
  }

  void SplitPlan(size_t cut_begin);
  void SplitAppend(NDObject *op) {
    op->index_ = objects_.size();
    objects_.push_back(op);
    StaticAppend(op);
  }
  void SplitBuild();

  bool PermPropCheck(int prop);
  void PermPropUpdate(int prop, const DimArray &perm);

  SpecVecContext &ctx_;
  int64_t live_peak_;
  int64_t tile_limit_;
  uint32_t fall_opt_;
  static constexpr uint32_t FALL_BROADCAST = 1;
  static constexpr uint32_t FALL_REDUCE = 2;
  static constexpr uint32_t FALL_RESHAPE = 4;
  static constexpr uint32_t FALL_CUSTOM_SPLIT = 8;
  static constexpr uint32_t FALL_PERMUTE = 16;
};

class SpecVecKernel : public SpecVecBase {
 public:
  SpecVecKernel(uint32_t flags) : SpecVecBase(flags, context_) {}

  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  uint64_t CodeGen() override;

 protected:
  void SpecInit();

  uint32_t fall_opt_init_{0};
  SpecVecContext context_;
};

class DumpRefHelper {
 public:
  explicit DumpRefHelper(std::ostringstream &oss) : oss_(oss) {}
  virtual ~DumpRefHelper() = default;
  void Dump(NDObject *op);
  virtual NDObject *GetInput(NDObject *input);
  void DumpGraph(const std::string &indent, const std::string &name, const std::vector<NDObject *> &build_ops);

 protected:
  std::ostringstream &oss_;
  int idx_{0};
  std::unordered_map<NDObject *, int> idx_map_;
};

class KernelBuilder : public Kernel {
 public:
  KernelBuilder(VKernel *k) { kernel_ = k; }
  ~KernelBuilder() { kernel_ = nullptr; }
};
}  // namespace dvm
#endif  // _DVM_KERNEL_H_
