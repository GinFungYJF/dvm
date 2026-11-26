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

#ifndef _DVM_SCHEDULE_H_
#define _DVM_SCHEDULE_H_

#include "kernel.h"

namespace dvm {
class VectorSchedule {
 public:
  VectorSchedule(VectorKernel *kernel) : kernel_(kernel) {}
  virtual ~VectorSchedule() = default;
  void SpaceInit();
  void SpaceSplit(int dim, int64_t npart, int64_t nfactor);
  void SpaceTrans(int dim1, int dim2);
  void SaveSpace();
  void ApplySubSpace(const DimArray &size) {
    for (auto &info : space_records_) {
      info.ndd->Reset();
      auto &dims = info.ndd->dims;
      for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = (info.bcast_mask >> i) & 1ul ?  1 : size[i];
      }
    }
  }
  uint32_t GetBCast(NDObject *op) {
    for (auto &r : space_records_) {
      if (r.ndd == op->nd_.data) return r.bcast_mask;
    }
    return 0;
  }

  struct SpaceRecord {
    uint32_t bcast_mask;
    NDSpaceData *ndd;
  };
  std::vector<SpaceRecord> space_records_;
  VectorKernel *kernel_;
};

class SchGenHelper : public VectorSchedule {
 public:
  SchGenHelper(VectorKernel *kernel) : VectorSchedule(kernel) {}
  ~SchGenHelper() override;
  virtual int64_t CodeGen();
  RelocAddr *ReserveReloc(size_t size) {
    if (size > reloc_size_) {
      delete []reloc_array_;
      reloc_array_ = new RelocAddr[size];
      reloc_size_ = size;
    }
    return reloc_array_;
  }
 protected:
  void AllocStride(NDAccess *acc);
  void ResetStrides();

  struct ExtStride {
    NDAccess *acc;
    DimArray stride;
    NDObject *rm_pad{nullptr};
  };
  size_t ext_stride_used_{0};
  std::vector<ExtStride> ext_strides_;
  size_t reloc_size_{0};
  RelocAddr *reloc_array_{nullptr};
};

class FractalSchGen : public SchGenHelper {
 public:
  FractalSchGen(VectorKernel *kernel);
  int64_t CodeGen() override;
};

class ConcatSchGen : public SchGenHelper {
 public:
  ConcatSchGen(VectorKernel *kernel, ConcatOp *concat, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;
 protected:
  ConcatOp *concat_;
  struct SliceIO {
    std::vector<NDObject *> ios;
    std::vector<NDObject *> deads;
    size_t load_num{0};
    uint64_t slice_mask{0};
    uint64_t bcast_mask{0};
  };
  std::vector<SliceIO> slice_ios_;
};

class SplitSchGen : public SchGenHelper {
 public:
  SplitSchGen(VectorKernel *kernel, SplitOpM *split, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;
 protected:
  SplitOpM *split_;
  struct SliceIO {
    std::vector<NDObject *> ios;
    std::vector<NDObject *> deads;
    size_t load_num{0};
    uint64_t slice_mask{0};
    uint64_t bcast_mask{0};
  };
  std::vector<SliceIO> slice_ios_;
};

class DupTilingSchGen : public SchGenHelper {
 public:
  explicit DupTilingSchGen(VectorKernel *kernel) : SchGenHelper(kernel) {}
  void DupCodeGen(int split_dim, int64_t body_size, int64_t tail_size);
};

class SliceSchGen : public SchGenHelper {
 public:
  SliceSchGen(VectorKernel *kernel, NDObject *slice, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;

 protected:
  SliceOp *slice_;
  NDObject *slice_dom_{nullptr};
  uint64_t full_io_mask_{0};
};

class GeneralViewSchGen : public SchGenHelper {
 public:
  struct LoadInfo {
    int io_idx;
    uint64_t view_mask;
  };
  struct Group {
    std::vector<NDObject *> static_ops;
    std::vector<LoadInfo> load_infos;
    NDObject *dom{nullptr};
    uint64_t quota;
    uint64_t view_mask;
    DimArray space;
  };
  struct ConcatGroup {
    std::vector<NDObject *> static_ops;
    std::vector<LoadInfo> load_infos;
    uint64_t view_mask;
  };

  GeneralViewSchGen(VectorKernel *kernel, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;

 protected:
  std::vector<Group> groups_;
  std::vector<ConcatGroup> concat_groups_;  // ordered by slice index. only support one concat
  std::vector<uint32_t> load_bcast_mask_;
  std::vector<NDObject *> view_ops_;
  int concat_view_idx_{-1};
};

SchGenHelper *BuildViewSch(VectorKernel *kernel, const std::vector<NDObject *> &objects);
}  // namespace dvm
#endif  // _DVM_SCHEDULE_H_