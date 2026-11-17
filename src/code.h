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

#ifndef _DVM_CODE_H_
#define _DVM_CODE_H_
#include <sstream>
#include <cstring>
#include "isa.h"
#include "system.h"

namespace dvm {

class VisitCoder {
 public:
  void AddReloc(uint64_t *pc, uint64_t offset) { rel_relocs_.emplace_back(pc, offset); }
  void Clear() { rel_relocs_.clear(); }
  void Update(void *new_base, void *old_base) {
    for (auto &r : rel_relocs_) {
      r.first = static_cast<uint64_t *>(new_base) + (r.first - static_cast<uint64_t *>(old_base));
    }
  }
  std::vector<std::pair<uint64_t *, uint64_t>> rel_relocs_;

 protected:
  ~VisitCoder() = default;
};

struct RedVisitCoder : public VisitCoder {
 public:
  enum { BCODE_MAX = 64 };
  uint32_t block_num_;
  uint32_t ws_size_;
  uint32_t visit_id_;
  uint32_t code_size_;
  uint8_t code_[BCODE_MAX];
};

class MixVisitCoder : public VisitCoder {
 public:
  uint64_t ub_base_{0};
};

struct RelocAddr {
  void Reloc(void *dst) {
    if (reloc_) {
      *reloc_ = reinterpret_cast<uint64_t>(dst);
    }
  }
  void Update(uint64_t *insn) { reloc_ = insn; }
  void Update(const RelocAddr &share) { reloc_ = share.reloc_; }
  void Update(void *new_base, void *old_base) {
    reloc_ = static_cast<uint64_t *>(new_base) + (reloc_ - static_cast<uint64_t *>(old_base));
  }

  union {  // NOTICE: bind after codegen
    void *gm;
    uint64_t ws;
    const RelocAddr *op;
    uint64_t data;
  };
  uint64_t *reloc_{nullptr};
  RelocAddr *bind_list_{nullptr};
};

class Code;
class DVM_API CodeWrap {
 public:
  CodeWrap() = default;
  virtual ~CodeWrap() {}
  virtual int LaunchWrap(void *workspace, void *stream);
  virtual void CombineWrap(Code *to, uint64_t ws_base);
  virtual void DasWrap(std::ostringstream &oss);
  virtual void CollectWrap(std::vector<Code *> &codes);
  CodeWrap *next_{nullptr};
};

class Code : public CodeWrap {
 public:
  enum { kTargetVec = 0, kTargetSimtVec, kTargetCube, kTargetMix };
  static bool IsVector(int target) { return target <= kTargetSimtVec; }

  Code() = default;
  Code(const Code &obj) = delete;
  Code &operator=(const Code &) = delete;
  Code &operator=(Code &&other);
  ~Code() override;
  void Clear() {
    bind_wss_ = nullptr;
    bind_ops_ = nullptr;
    wrap_ = nullptr;
    ArchClear();
  }
  void Alloc(size_t size);
  void MoveCode(Code &other);

  static inline uint64_t GenEntry(uint64_t data, uint64_t ktype, uint64_t data_size) {
    return data | ktype | (data_size / sizeof(uint64_t)) << V_ENTRY_CODE_SIZE_OFFSET;
  }
  static inline uint64_t GenEntryV(uint64_t tile_num, uint64_t block_dim, uint64_t data_size) {
    uint64_t block_tile = CeilDiv<uint64_t>(tile_num, block_dim);
    uint64_t block_tail = block_dim * block_tile - tile_num;
    ASSERT(block_tail < block_dim);
    uint64_t data = block_tile << V_ENTRY_V_TILE_BODY_OFFSET | block_tail << V_ENTRY_V_TILE_TAIL_OFFSET;
    return GenEntry(data, V_ENTRY_TYPE_V, data_size);
  }
  static inline uint64_t GenEntryVE(uint64_t visit_id, uint64_t visit_offset, uint64_t data_size) {
    uint64_t data = g_system.g_visit_func_offset_[visit_id] << V_ENTRY_VE_VISIT_ID_OFFSET |
                    visit_offset << V_ENTRY_VE_VISIT_OFFSET_OFFSET;
    return GenEntry(data, V_ENTRY_TYPE_VE, data_size);
  }
  static inline uint64_t GenEntryC(uint64_t data_size) {
    return GenEntry(V_ENTRY_FLAG_CUBE_MIX, V_ENTRY_TYPE_C, data_size);
  }

  static inline uint64_t GenEntryMix(const MixVisitCoder *visit, uint64_t slice[], uint64_t tail[], uint64_t stride[],
                                     uint64_t subtile0, uint64_t subtile1, uint8_t *code_begin, uint8_t *&code_end) {
    auto visit_code = reinterpret_cast<uint64_t *>(code_end);
    uint64_t visit_offset = (code_end - code_begin - sizeof(vCubeOp)) / sizeof(uint64_t);
    for (auto &r : visit->rel_relocs_) {
      *(r.first + r.second) |= static_cast<uint64_t>(visit_code - r.first);
    }
    code_end += vVisitMix::Encode(visit_code, slice, tail, stride, subtile0, subtile1) * sizeof(uint64_t);
    return Code::GenEntryVE(V_VISIT_MIX, visit_offset, code_end  - code_begin) | V_ENTRY_FLAG_CUBE_MIX;
  }

  void UpdateHead(uint64_t entry) {
    uint64_t *head = reinterpret_cast<uint64_t *>(data_);
    head[0] = 0;
    head[1] = entry;
  }

  void UpdateV(uint64_t tile_num, bool simt) {
    target_ = simt ? kTargetSimtVec : kTargetVec;
    UpdateHead(GenEntryV(tile_num, block_dim_, data_size_ - HeadSize()));
  }

  void UpdateVE(const RedVisitCoder *visit) {
    target_ = kTargetMix;
    uint64_t *visit_code = reinterpret_cast<uint64_t *>(data_ + data_size_);
    data_size_ += visit->code_size_;
    uint64_t offset = visit_code - reinterpret_cast<uint64_t *>(data_) - 2;
    std::memcpy(visit_code, visit->code_, visit->code_size_);
    for (auto &r : visit->rel_relocs_) {
      *r.first |= static_cast<uint64_t>(visit_code - r.first) << V_HEAD_EXT_OFFSET;
    }
    UpdateHead(GenEntryVE(visit->visit_id_, offset, data_size_ - HeadSize()));
  }

  void UpdateC() {
    target_ = kTargetCube;
    UpdateHead(GenEntryC(data_size_ - HeadSize()));
  }

  void UpdateMix(uint64_t entry) {
    target_ = kTargetMix;
    UpdateHead(entry);
  }

  static constexpr uint64_t HeadSize() { return sizeof(uint64_t) * 2; }  // ffts + entry

  uint64_t ReserveWorkspace(uint64_t workspace_size) {
    return data_size_ <= PARAM_TABLE_LIMIT ? workspace_size : ReserveCodeSpace(workspace_size);
  }

  int Launch(void *workspace, void *stream) {
    if (!wrap_) {
      return DoLaunch(workspace, stream);
    }
    return wrap_->LaunchWrap(workspace, stream);
  }

  void CombineBind(const Code &code, uint64_t ws_base);
  void Combine(const Code &code, uint64_t ws_base);
  void DisAssemble(std::ostringstream &oss);
  void DasWrap(std::ostringstream &oss) override;
  int LaunchWrap(void *workspace, void *stream) override;
  void CombineWrap(Code *to, uint64_t ws_base) override;
  void CollectWrap(std::vector<Code *> &codes) override;

  void Collect(std::vector<Code *> &codes) {
    if (wrap_) {
      wrap_->CollectWrap(codes);
    } else {
      codes.push_back(this);
    }
  }

  void InsertWrap(CodeWrap *wrap) {
    wrap->next_ = wrap_ ? wrap_ : this;
    wrap_ = wrap;
  }

  void RelocWssBind(void *workspace) {
    for (auto op = bind_wss_; op != nullptr; op = op->bind_list_) {
      op->Reloc(static_cast<char *>(workspace) + op->ws);
    }
  }
  void RelocOpsBind() {
    for (auto op = bind_ops_; op != nullptr; op = op->bind_list_) {
      op->Reloc(reinterpret_cast<void *>(*op->op->reloc_));
    }
  }
  void RelocBinds(void *workspace) {
    RelocWssBind(workspace);
    RelocOpsBind();
  }
  void BindWorkspace(RelocAddr &op, uint64_t offset) {
    op.ws = offset;
    InsertBind(bind_wss_, op);
  }
  void BindOpFast(RelocAddr &op, const RelocAddr &target) {
    op.op = &target;
    InsertBind(bind_ops_, op);
  }
  void BindOp(RelocAddr &op, const RelocAddr &target);

  int DoLaunch(void *workspace, void *stream) {
    return g_system.code_launch_(g_system, this, workspace, stream);
  }

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
  RelocAddr *bind_wss_{nullptr};
  RelocAddr *bind_ops_{nullptr};
  CodeWrap *wrap_{nullptr};
  size_t mem_size_{0};

 private:
  uint64_t ReserveCodeSpace(uint64_t workspace_size);

  void Free();
  void CheckFree() {
    if (data_) Free();
  }

  void InsertBind(RelocAddr *&pos, RelocAddr &op) {
#ifdef DEBUG
    for (auto x = pos; x != nullptr; x = x->bind_list_) {
      ASSERT(x != &op);
    }
#endif
    op.bind_list_ = pos;
    pos = &op;
  }

 public:
#ifdef DVM_ARCH_GPU
  void ArchClear() {
    threads_ = 0;
    shared_bytes_ = 0;
  }
  void ArchMove(Code &other) {
    threads_ = other.threads_;
    shared_bytes_ = other.shared_bytes_;
  }
  void ArchCombine(const Code &other) {
    if (other.threads_ > threads_) {
      threads_ = other.threads_;
    }
    if (other.shared_bytes_ > shared_bytes_) {
      shared_bytes_ = other.shared_bytes_;
    }
  }
  uint32_t threads_{0};
  uint32_t shared_bytes_{0};
#else
  void ArchClear() {}
  void ArchMove(Code &) {}
  void ArchCombine(const Code &) {}
#endif
};

class DVM_API CodeLaunchGuard : public CodeWrap {
 public:
  CodeLaunchGuard(Code &root);
  virtual ~CodeLaunchGuard();
  int LaunchWrap(void *workspace, void *stream) override;
  virtual int CodeLaunch(Code *code, void *workspace, void *stream) = 0;

 protected:
  std::vector<Code *> codes_;
  size_t launch_idx_{0};
};

class PCodeEncoder {
 public:
  PCodeEncoder() = default;
  ~PCodeEncoder() = default;

  void Reset(Code *code, int target, int max_prog_num, uint64_t code_reserve, uint64_t core_reserve = 0);
  uint64_t *Append(uint64_t entry, uint64_t code_size) {
    auto prog = prog_entry_;
    prog_entry_ += vProgEntry::Encode(prog, entry, prog_data_ - (code_->data_ + Code::HeadSize()));
    prog_data_ += code_size;
    return prog;
  }
  uint64_t *CloneProg(uint64_t *prog) {
    auto clone = prog_entry_;
    for (int i = 0; i < vProgEntry::CODE_SIZE; ++i) {
      *prog_entry_++ = prog[i];
    }
    return clone;
  }
  void AssignAic(uint64_t block_begin, uint64_t block_num, uint64_t *prog) {
    Assign(aic_lookup_, block_begin, block_num, prog);
  }
  void AssignAiv(uint64_t block_begin, uint64_t block_num, uint64_t *prog) {
    Assign(aiv_lookup_, block_begin, block_num, prog);
  }
  void Submit(uint64_t block_dim) {
    code_->data_size_ = prog_data_ - code_->data_;
    code_->block_dim_ = block_dim;
  }
  uint8_t *ProgData() const { return prog_data_; }

 protected:
  void Assign(uint8_t *lookup, uint64_t block_begin, uint64_t block_num, uint64_t *prog) {
    auto prog_offset = prog - reinterpret_cast<uint64_t *>(code_->data_ + Code::HeadSize());
    for (uint64_t i = 0; i < block_num; ++i) {
      lookup[block_begin + i] = prog_offset;
    }
    vProgEntry::UpdateBlock(prog, block_begin, block_num);
  }

  Code *code_;
  uint8_t *aic_lookup_;
  uint8_t *aiv_lookup_;
  uint64_t *prog_entry_;
  uint8_t *prog_data_;
};
}  // namespace dvm
#endif  // _DVM_CODE_H_
