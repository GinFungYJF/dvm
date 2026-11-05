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

#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <fstream>
#include "pybind11/numpy.h"
#include "runtime.h"
#include "kernel.h"
#include "pybind_api.h"
#include "msprof.h"
#include "xkernel.h"
#ifndef DVM_ARCH_GPU
#include "acl/acl_prof.h"
#endif

namespace dvm {
class ProfileMgr {
 public:
#ifdef DVM_ARCH_GPU
  ProfileMgr(const std::string &, const std::string &) {}
  ~ProfileMgr() = default;
  void ProfStart() {}
  void ProfStop() {}
#else
  ProfileMgr(const std::string &path, const std::string &metric) {
    static const std::unordered_map<std::string, aclprofAicoreMetrics> aicore_metrics = {
      {"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
      {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
      {"MemoryAccess", ACL_AICORE_MEMORY_ACCESS},
      {"L2Cache", ACL_AICORE_L2_CACHE},
    };
    auto iter = aicore_metrics.find(metric);
    auto aicore_metric = iter == aicore_metrics.end() ? ACL_AICORE_ARITHMETIC_UTILIZATION : iter->second;
    aclprofInit(path.c_str(), path.length());
    int32_t device_id;
    aclrtGetDevice(&device_id);
    uint32_t device_list[1] = {static_cast<uint32_t>(device_id)};
    uint32_t device_num = 1;
    uint64_t mask =
      ACL_PROF_ACL_API | ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME | ACL_PROF_TRAINING_TRACE | ACL_PROF_AICPU;
    acl_config_ = aclprofCreateConfig(device_list, device_num, aicore_metric, nullptr, mask);
  }
  ~ProfileMgr() {
    aclprofDestroyConfig(acl_config_);
    aclprofFinalize();
  }
  void ProfStart() { aclprofStart(acl_config_); }
  void ProfStop() { aclprofStop(acl_config_); }

 private:
  aclprofConfig *acl_config_;
#endif // DVM_ARCH_GPU
};
}

namespace dvm {
namespace {
const size_t TEST_NUM = 10;

inline void F32ToBF16(float *input, uint16_t *output, uint32_t size) {
  while (size-- != 0) {
    *output++ = BFloat16(*input++).int_value();
  }
}

inline void BF16ToF32(uint16_t *input, float *output, uint32_t size) {
  while (size-- != 0) {
    *output++ = static_cast<float>(BFloat16(*input++));
  }
}

std::string GetBufferFormat(const DataType type) {
  const std::string formats[kDataTypeEnd] = {
    py::format_descriptor<bool>::format(),     "e",
    py::format_descriptor<uint16_t>::format(), py::format_descriptor<float>::format(),
    py::format_descriptor<int32_t>::format(),  py::format_descriptor<int64_t>::format()};
  return formats[type];
}

std::vector<int64_t> GetVector(py::object shape) {
  py::list shape_list = py::cast<py::list>(shape);
  size_t size = shape_list.size();
  std::vector<int64_t> shape_vec(size);
  for (size_t i = 0; i < size; ++i) {
    shape_vec[i] = py::cast<int64_t>(shape_list[i]);
  }
  return shape_vec;
}

inline void ResetStoreMemory(const std::vector<RtKernelPy::StoreInfo> &stores) {
  for (auto &info : stores) {
    auto buf = info.host.request();
    ERROR_CHECK(rt::MemToDev(info.dev, buf.ptr, info.size));
  }
}

std::unordered_map<std::string, KernelType> kernel_type_map = {
  {"", KernelType::kVector},     {"vector", KernelType::kVector},     {"cube", KernelType::kCube},
  {"mix", KernelType::kMix},     {"parallel", KernelType::kParallel}, {"seq", KernelType::kSequence},
  {"split", KernelType::kSplit}, {"eager", KernelType::kEager}};

std::pair<KernelType, uint32_t> ParseKernelType(const std::string &ker_type) {
  auto pos = ker_type.find(':', 0);
  auto type_name = ker_type.substr(0, pos);
  uint32_t flags = 0;
  while (pos != std::string::npos) {
    pos++;
    auto end = ker_type.find(',', pos);
    auto flag_name = ker_type.substr(pos, end == std::string::npos ? end : end - pos);
    if (flag_name == "dyn") {
      flags |= KernelFlag::kDynamic;
    } else if (flag_name == "unify_ws") {
      flags |= KernelFlag::kUnifyWS;
    } else if (flag_name == "spec") {
      flags |= KernelFlag::kSpeculate;
    } else if (flag_name == "opt_fractal") {
      flags |= KernelFlag::kOptFractalTrans;
    } else if (flag_name == "lazy") {
      flags |= KernelFlag::kLazyCodeGen;
    } else if (flag_name == "priv1") {
      flags |= KernelFlag::kPrivate1;
    } else if (flag_name == "priv2") {
      flags |= KernelFlag::kPrivate2;
    } else {
      DvmException("kernel flag error");
    }
    pos = end;
  }
  auto it = kernel_type_map.find(type_name);
  if (it == kernel_type_map.end()) {
    DvmException("kernel type error");
  }
  return std::make_pair(it->second, flags);
}

template <typename T>
T &FindVectorInfo(std::vector<T> &infos, NDObject *op) {
  for (auto &info : infos) {
    if (info.op == op) {
      return info;
    }
  }
  ASSERT(0);
  return infos.front();
}

struct MpCtx {
  pid_t pids[32];
  int rank_size{1};
  int rank_id{0};
  int shmid{0};
  volatile int64_t *bars{nullptr};
  Comm comm;
};
MpCtx g_mpc;
}  // end namespace

class KernelRunner : public WsAllocator {
 public:
  using LoadInfo = RtKernelPy::LoadInfo;
  using StoreInfo = RtKernelPy::StoreInfo;

  virtual ~KernelRunner() {}
  virtual void AllocLoad(const py::buffer_info &buf, LoadInfo &load) = 0;
  virtual void AllocStore(StoreInfo &store) = 0;
  virtual void Release(void *) {}
  virtual void Reset() = 0;
  virtual int Run(Kernel &kernel, void *workspace, bool sync) = 0;
  void *Stream() const { return stream_; }

 protected:
  void *stream_{nullptr};
};

class DevRunner : public KernelRunner {
 public:
  DevRunner(int dev_id) {
#ifdef VK_SIM_MODEL
    // Since CANN 8.5, Python must preload this library before runtime-related code can run correctly.
    void *handle = dlopen("libruntime_camodel.so", RTLD_NOW | RTLD_GLOBAL);
    EXCEPTION_IF(handle == nullptr, dlerror());
    dev_id = 0;
#else
    uint32_t dev_count = 0;
    ERROR_CHECK(rt::GetDeviceCount(&dev_count));
    ASSERT(static_cast<uint32_t>(dev_id) < dev_count);
#endif
    ERROR_CHECK(rt::SetDevice(dev_id));
    ERROR_CHECK(rt::CreateStream(&stream_));
    dev_id_ = dev_id;
  }
  ~DevRunner() override {
    Reset();
    rt::DestroyStream(stream_);
  }

  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override {
    size_t size = buf.itemsize * buf.size;
    if (size == 0) return;
    ERROR_CHECK(rt::Malloc(&load.dev, size));
    ERROR_CHECK(rt::MemToDev(load.dev, buf.ptr, size));
    dev_mem_.push_back(load.dev);
  }

  void AllocStore(StoreInfo &store) override {
    if (store.size == 0) return;
    auto buf = store.host.request();
    const uint64_t reserve_mem = 512;
    ERROR_CHECK(rt::Malloc(&store.dev, store.size + reserve_mem));
    if (store.clear_mem || store.set_host) {
      std::memset(buf.ptr, 0, store.size);
      ERROR_CHECK(rt::MemToDev(store.dev, buf.ptr, store.size));
    }
    dev_mem_.push_back(store.dev);
  }

  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ERROR_CHECK(rt::Malloc(&ws, size));
      dev_mem_.push_back(ws);
    }
    return ws;
  }

  void Release(void *mem) override {
    auto iter = std::find(dev_mem_.begin(), dev_mem_.end(), mem);
    if (iter == dev_mem_.end()) return;
    rt::Free(mem);
    dev_mem_.erase(iter);
  }

  int Run(Kernel &kernel, void *workspace, bool sync) override {
    if (kernel.GetImpl()->IsSplit()) {
      kernel.Launch(stream_);
    } else {
      ERROR_CHECK(kernel.Launch(nullptr, 0, workspace, stream_));
    }
    return sync ? rt::SynchronizeStream(stream_) : 0;
  }

  void Reset() override {
    for (auto mem : dev_mem_) {
      (void)rt::Free(mem);
    }
    dev_mem_.clear();
  }

  std::vector<void *> dev_mem_;
  int dev_id_{0};
};

void DryLaunch(Code *code, void *workspace, void *stream, uint64_t core_idx, bool is_cube);
class DryRunner : public KernelRunner {
 public:
  DryRunner(int core_id) : core_id_(core_id) {}
  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override { load.dev = buf.ptr; }
  void AllocStore(StoreInfo &store) override {
    auto buf = store.host.request();
    if (store.clear_mem) {
      std::memset(buf.ptr, 0, store.size);
    }
    store.dev = buf.ptr;
  }
  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ws = std::malloc(size);
      host_mem_.push_back(ws);
    }
    return ws;
  }
  void Reset() override {
    for (auto mem : host_mem_) {
      std::free(mem);
    }
    host_mem_.clear();
  }
  int Run(Kernel &kernel, void *workspace, bool sync) override {
    DryRun(kernel, workspace, core_id_, false);
    return 0;
  }

  struct _LaunchGuard : public CodeLaunchGuard {
    _LaunchGuard(Code &code, uint64_t core_idx, bool is_cube)
        : CodeLaunchGuard(code), core_idx_(core_idx), is_cube_(is_cube) {}
    int CodeLaunch(Code *code, void *workspace, void *stream) override {
      DryLaunch(code, workspace, stream, core_idx_, is_cube_);
      return 0;
    }
    uint64_t core_idx_;
    bool is_cube_;
  };

  void DryRun(Kernel &kernel, void *workspace, int core_id, bool is_cube) {
    _LaunchGuard guard(kernel.GetImpl()->code_, core_id, is_cube);
    if (kernel.GetImpl()->IsSplit()) {
      kernel.Launch(nullptr);
    } else {
      ERROR_CHECK(kernel.Launch(nullptr, 0, workspace, nullptr));
    }
  }

 private:
  int core_id_;
  std::vector<void *> host_mem_;
};

class DasRunner : public KernelRunner {
 public:
  DasRunner() { addr_ = reinterpret_cast<uint8_t *>(0x10); }
  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override {
    load.dev = addr_;
    addr_ += buf.itemsize * buf.size;
  }
  void AllocStore(StoreInfo &store) override {
    store.dev = addr_;
    addr_ += store.size;
  }
  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ws = addr_;
      addr_ += size;
    }
    return ws;
  }
  void Reset() override {
    addr_ = reinterpret_cast<uint8_t *>(0x10);
  }
  int Run(Kernel &kernel, void *workspace, bool sync) override {
    auto &code = kernel.GetImpl()->code_;
    code.RelocBinds(workspace);
    return 0;
  }

 private:
  uint8_t *addr_;
};

class RunnerManager {
 public:
  ~RunnerManager() {
    delete dev_;
    delete dry_;
    delete das_;
  }

  static RunnerManager &Instance() {
    static RunnerManager mng;
    return mng;
  }

  void ResetDevRuner() {
    delete dev_;
    dev_ = nullptr;
  }

  KernelRunner *Get(const std::string type, int dev_id) {
    if (type == "dev") {
      if (dev_) {
        if (dev_->dev_id_ == dev_id) {
          return dev_;
        }
        delete dev_;
      }
      dev_ = new DevRunner(dev_id);
      return dev_;
    } else if (type == "dry") {
      if (!dry_) {
        dry_ = new DryRunner(dev_id);
      }
      return dry_;
    } else if (type == "das") {
      if (!das_) {
        das_ = new DasRunner();
      }
      return das_;
    }
    ASSERT(0);
    return nullptr;
  }

 private:
  DevRunner *dev_;
  DryRunner *dry_;
  DasRunner *das_;
};

RtKernelPy::RtKernelPy(const std::string &ker_type, const std::string &run_type, int dev_id) {
  runner_ = RunnerManager::Instance().Get(run_type, dev_id);
  auto [type, flags] = ParseKernelType(ker_type);
  kernel_.Reset(type, flags);
}

RtKernelPy::~RtKernelPy() {
  if (profile_mgr_ != nullptr) {
    profile_mgr_->ProfStop();
    delete profile_mgr_;
  }
  runner_->Reset();
#ifdef VK_SIM_MODEL
  RunnerManager::Instance().ResetDevRuner();
#endif
  for (auto *ref : shape_) {
    delete ref;
  }
}

IntArrayRef *RtKernelPy::GetShapeRef(py::object shape) {
  if (py::isinstance<IntArrayRefPy>(shape)) {
    auto shape_ptr = shape.cast<std::shared_ptr<IntArrayRefPy>>();
    return shape_ptr->Get();
  }
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(GetVector(shape));
  return shape_.emplace_back(new IntArrayRef(shape_vec));
}

py::object RtKernelPy::OneHot(py::object indices, int depth, int axis, py::object on_value, py::object off_value,
                              DataTypePy dtype) {
  auto indices_obj = PyToObj(indices);
  auto depth_ref = shape_.emplace_back(new IntArrayRef(shape_vec_.emplace_back(1, depth)));
  NDObject *op;
  if (dtype == kInt32) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, py::cast<int32_t>(on_value), py::cast<int32_t>(off_value));
  } else {
    auto on_float = py::cast<float>(on_value);
    auto off_float = py::cast<float>(off_value);
    if (dtype == kFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, Float16(on_float), Float16(off_float));
    } else if (dtype == kBFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, BFloat16(on_float), BFloat16(off_float));
    } else {
      ASSERT(dtype == kFloat32);
      op = kernel_.OneHot(indices_obj, depth_ref, axis, on_float, off_float);
    }
  }
  return ObjToPy(op);
}

py::object RtKernelPy::Load(py::object shape, DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto op = kernel_.Load(nullptr, shape_ref, type);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::GlobalAccess(py::object shape, DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto op = kernel_.GlobalAccess(nullptr, shape_ref, type);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::ViewLoad(py::object shape, py::object stride, DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto stride_ref = GetShapeRef(stride);
  auto op = kernel_.Load(nullptr, shape_ref, stride_ref, type);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::GatherLoad(py::object shape, py::object index, DataTypePy type, int axis, int gather_mode) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto index_op = PyToObj(index);
  auto op = kernel_.GatherLoad(nullptr, shape_ref, index_op, axis, type, static_cast<GatherMode>(gather_mode));
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::SliceLoad(py::object shape, py::object start, py::object size, DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto size_ref = GetShapeRef(size);
  auto op = kernel_.SliceLoad(nullptr, shape_ref, start_ref, size_ref, type);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::StridedSliceLoad(py::object shape, py::object start, py::object end, py::object step,
                                        DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto end_ref = GetShapeRef(end);
  auto step_ref = GetShapeRef(step);
  auto op = kernel_.StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref, step_ref, type);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::MultiLoad(py::object shape, DataTypePy type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
  auto op = kernel_.MultiLoad(nullptr, shape_ref, type, &g_mpc.comm);
  info.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::Store(py::object obj, DataTypePy type) {
  auto in_obj = PyToObj(obj);
  if (type != kDataTypeEnd) {
    in_obj = kernel_.Cast(in_obj, type);
  }
  auto op = kernel_.Store(nullptr, in_obj);
  auto &store = stores_.emplace_back();
  store.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::ViewStore(py::object obj, py::object stride, DataTypePy type) {
  auto in_obj = PyToObj(obj);
  if (type != kDataTypeEnd) {
    in_obj = kernel_.Cast(in_obj, type);
  }
  auto stride_ref = GetShapeRef(stride);
  auto op = kernel_.Store(nullptr, in_obj, stride_ref);
  auto &store = stores_.emplace_back();
  store.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::PadStore(py::object obj, int64_t pad_size) {
  auto op = kernel_.PadStore(nullptr, PyToObj(obj), pad_size);
  auto &store = stores_.emplace_back();
  store.op = op;
  return ObjToPy(op);
}

py::object RtKernelPy::AllReduce(const std::string &type, py::object input) {
  ReduceType reduce_type{ReduceType::kReduceTypeEnd};
  if (type == "sum") {
    reduce_type = ReduceType::kSum;
  } else if (type == "max") {
    reduce_type = ReduceType::kMax;
  } else {
    throw py::value_error("Unsupported AllReduce type: " + type);
  }
  return ObjToPy(kernel_.AllReduce(reduce_type, PyToObj(input), &g_mpc.comm));
}

py::object RtKernelPy::ReduceScatter(py::object input) {
  return ObjToPy(kernel_.ReduceScatter(PyToObj(input), &g_mpc.comm));
}

py::object RtKernelPy::AllGather(py::object input) { return ObjToPy(kernel_.AllGather(PyToObj(input), &g_mpc.comm)); }

py::object RtKernelPy::AllGatherV2(py::object input) {
  return ObjToPy(kernel_.AllGatherV2(PyToObj(input), &g_mpc.comm));
}

py::object RtKernelPy::ConvertToBF16(py::object input) {
  auto array = py::array(input);
  py::buffer_info buf = array.request();
  ASSERT(buf.itemsize == 4);  // input should be array of f32
  size_t size = buf.size;
  std::vector<uint16_t> bf16(size);
  F32ToBF16(reinterpret_cast<float *>(buf.ptr), bf16.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride /= 2; });
  py::buffer_info new_buf(bf16.data(), 2, py::format_descriptor<uint16_t>::format(), buf.ndim, buf.shape, buf.strides);
  bf16s_.push_back(std::move(bf16));
  return py::array(new_buf);
}

py::object RtKernelPy::ConvertFromBF16(py::object input) {
  auto array = py::array(input);
  py::buffer_info buf = array.request();
  ASSERT(buf.itemsize == 2);  // input should be array of bf16
  size_t size = buf.size;
  std::vector<float> float_data(size);
  BF16ToF32(reinterpret_cast<uint16_t *>(buf.ptr), float_data.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride *= 2; });
  py::buffer_info new_buf(float_data.data(), 4, py::format_descriptor<float>::format(), buf.ndim, buf.shape,
                          buf.strides);
  f32s_.push_back(std::move(float_data));
  return py::array(new_buf);
}

void RtKernelPy::Tile(int start, int end, int64_t num, int64_t factor) {
  static_cast<VKernelS *>(kernel_.GetImpl())->SetTile(start, end, num, factor);
}

void RtKernelPy::CodeGen(py::object pass_names) {
  class _PassOptimizer : public pass::PassOptimizer {
   public:
    _PassOptimizer(py::list pass_list) : names_(pass_list.cast<std::vector<std::string>>()) {
      // RAII: replace pass_opt_ on construction and restore it on destruction
      // (even on exception), preventing dangling-pointer crash in ~System().
      old_ = g_system.pass_opt_;
      g_system.pass_opt_ = this;
    }
    ~_PassOptimizer() { g_system.pass_opt_ = old_; }
    void RunPass(pass::BasicBlock &bb, bool dyn_shape) override {
      struct PassInfo {
        pass::Pass pass;
        bool shape_sensitive;
      };
      const static std::unordered_map<std::string, PassInfo> pass_map = {
        {"PrintPeakLive", {pass::PrintPeakLive, false}},
        {"ReorderStore", {pass::ReorderStore, false}},
        {"ReorderLoad", {pass::ReorderLoad, false}},
        {"CompactPeakLiveness", {pass::CompactPeakLiveness, false}},
        {"VectorDoubleBuffer", {pass::VectorDoubleBuffer, false}},
        {"DeadCodeEliminate", {pass::DeadCodeEliminate, false}},
        {"EliminateReshape", {pass::EliminateReshape, true}},
        {"EliminateDiamondView", {pass::EliminateDiamondView, false}},
        {"InsertRemovePad", {pass::InsertRemovePad, false}},
        {"VfFusion", {pass::VfFusion, false}}};
      for (auto &name : names_) {
        auto pass = pass_map.find(name);
        if (pass != pass_map.end() && !(dyn_shape && pass->second.shape_sensitive)) {
          pass->second.pass(bb);
        }
      }
    }
    std::vector<std::string> names_;
    pass::PassOptimizer *old_{nullptr};
  };
  if (kernel_.GetImpl()->IsSplit()) {
    std::vector<RelocEntry> relocs;
    relocs.reserve(loads_.size() + stores_.size());
    for (auto &info : loads_) {
      relocs.emplace_back(info.op, reinterpret_cast<uint8_t *>(info.dev) + info.offset);
    }
    kernel_.Normalize();
    for (auto &info : stores_) {
      PrepareStore(info);
      relocs.emplace_back(info.op, info.dev);
    }
    kernel_.CodeGen(relocs.data(), relocs.size(), runner_);
    return;
  }
  uint64_t workspace_size;
  if (py::isinstance<py::list>(pass_names)) {
    _PassOptimizer opt(py::cast<py::list>(pass_names));
    workspace_size = kernel_.CodeGen();
  } else {
    workspace_size = kernel_.CodeGen();
  }
  workspace_ = runner_->Alloc(workspace_size);
}

void RtKernelPy::InitComm(int rank_id, int rank_size, const std::string &comm_type) {
  int type_id;
  if (comm_type == "hccl") {
    type_id = Comm::kHccl;
  } else if (comm_type == "dummy") {
    type_id = Comm::kDummy;
  } else {
    type_id = Comm::kMemory;
  }
  g_mpc.rank_id = rank_id;
  g_mpc.rank_size = rank_size;
  g_mpc.comm.Init(rank_id, rank_size, type_id);
}

void RtKernelPy::Run() {
  if (!kernel_.GetImpl()->IsSplit()) {
    PrepareIO();
  }
  auto ret = runner_->Run(kernel_, workspace_, true);
  if (ret != 0) {
    std::cerr << kernel_.DumpR() << std::endl;
    std::cerr << kernel_.Dump() << std::endl;
    std::cerr << kernel_.Das() << std::endl;
    std::string err = "Kernel Execute Exception:" + std::to_string(ret);
    DvmException(err.c_str());
  }
}

void RtKernelPy::DryRun(int core_idx, bool cube_core) {
  static_cast<DryRunner *>(runner_)->DryRun(kernel_, workspace_, core_idx, cube_core);
}

void RtKernelPy::RegCustom(const std::string &nspace, const std::string &so_path, const std::string &bin_path) {
  static bool promote_scope = false;
  if (!promote_scope) {
    promote_scope = true;
    // Promote _dvm_py.so to the global symbol scope so the runtime-compiled host .so can
    // resolve dvm symbols (g_system, NDObject::mem_pool_, NDObject vtable, ...) from it./
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&RtKernelPy::RegCustom), &info) && info.dli_fname) {
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
  }
  g_system.RegCustom(nspace, so_path, bin_path);
}

py::object RtKernelPy::Custom(const std::string &full_name, py::object inputs, py::object attrs) {
  py::list input_list = py::cast<py::list>(inputs);
  std::vector<NDObject *> inputs_args;
  for (auto a : input_list) {
    inputs_args.push_back(PyToObj(a.cast<py::object>()));
  }
  std::vector<ScalarRef> attrs_args;
  if (py::isinstance<py::int_>(attrs)) {
    py::list attr_list = py::cast<py::list>(attrs);
    for (auto a : attr_list) {
      if (py::isinstance<py::int_>(a)) {
        attrs_args.emplace_back(a.cast<int64_t>());
      } else if (py::isinstance<py::float_>(a)) {
        attrs_args.emplace_back(a.cast<float>());
      } else {
        ASSERT(0);
      }
    }
  }
  auto op = g_system.CreateCustom(full_name, inputs_args, attrs_args);
  auto kernel = kernel_.GetImpl();
  kernel->Append(op);
  if (auto xout = static_cast<CustomOp *>(op)->xout_; xout != nullptr) {
    for (int i = 0; i < xout->out_num; ++i) {
      kernel->Append(xout->data[i]);
    }
  }
  return ObjToPy(op);
}

py::object RtKernelPy::Perf() {
  // warm up
  if (!kernel_.GetImpl()->IsSplit()) {
    PrepareIO();
  }
  runner_->Run(kernel_, workspace_, true);
  RepeatProfiler profiler;
  profiler.Reset();
  for (size_t i = 0; i < TEST_NUM; i++) {
    ResetStoreMemory(stores_);
    profiler.RecordStart(runner_->Stream());
    runner_->Run(kernel_, workspace_, false);
    profiler.RecordEnd(runner_->Stream());
  }
  return py::make_tuple(py::float_(profiler.min_us_), py::float_(profiler.max_us_),
                        py::float_(profiler.total_us_ / float(TEST_NUM)));
}

py::object RtKernelPy::Msprof(const std::string &path, int64_t test_num, const std::string &metric) {
  ProfileMgr mgr(path, metric);
  if (kernel_.GetImpl()->IsSplit()) {
    runner_->Run(kernel_, workspace_, true);
    mgr.ProfStart();
    while (test_num--) {
      ResetStoreMemory(stores_);
      kernel_.Launch(runner_->Stream());
    }
  } else {
    constexpr const char *kTempFusionOp = "DvmOp";
    kernel_.SetNameHint(kTempFusionOp, kTempFusionOp);
    PrepareIO();
    runner_->Run(kernel_, workspace_, true);
    mgr.ProfStart();
    std::vector<RelocEntry> relocs;
    relocs.reserve(loads_.size() + stores_.size());
    std::vector<void *> inputs_addr;
    std::vector<void *> outputs_addr;
    std::vector<NDObject *> inputs;
    std::vector<NDObject *> outputs;
    for (auto &info : loads_) {
      relocs.emplace_back(info.op, reinterpret_cast<uint8_t *>(info.dev) + info.offset);
    }
    for (auto &info : stores_) {
      relocs.emplace_back(info.op, info.dev);
    }
    while (test_num--) {
      ResetStoreMemory(stores_);
      ERROR_CHECK(kernel_.Launch(relocs.data(), relocs.size(), workspace_, runner_->Stream()));
    }
  }
  rt::SynchronizeStream(runner_->Stream());
  mgr.ProfStop();
  return py::none();
}

void RtKernelPy::MsprofStart(const std::string &path, const std::string &metric) {
  EXCEPTION_IF(profile_mgr_ != nullptr, "msprof has already started");
  profile_mgr_ = new ProfileMgr(path, metric);
  profile_mgr_->ProfStart();
}

void RtKernelPy::MsprofStop() {
  EXCEPTION_IF(profile_mgr_ == nullptr, "msprof has not started");
  profile_mgr_->ProfStop();
  delete profile_mgr_;
  profile_mgr_ = nullptr;
}

void RtKernelPy::Input(py::object obj, py::object val, size_t offset) {
  if (py::isinstance<NDObjectPy>(obj)) {
    auto op = static_cast<NDAccess *>(PyToObj(obj));
    auto &info = FindVectorInfo(loads_, op);
    auto input = py::array(val);
    py::buffer_info buf = input.request();
    runner_->AllocLoad(buf, info);
    info.offset = offset * ITEM_SIZE[op->type_id_];
    if (kernel_.GetImpl()->IsDynamic()) {
      info.shape.resize(buf.ndim);
      for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
        info.shape[i] = buf.shape[i];
      }
      *(op->shape_ref_) = info.shape;
    }
  } else if (py::isinstance<IntArrayRefPy>(obj)) {
    auto shape = obj.cast<std::shared_ptr<IntArrayRefPy>>();
    shape->Update(val);
  } else {
    ASSERT(py::isinstance<ScalarRefPy>(obj));
    auto scalar = obj.cast<std::shared_ptr<ScalarRefPy>>();
    scalar->Update(val);
  }
}

void RtKernelPy::ReleaseIO() {
  EXCEPTION_IF(kernel_.GetImpl()->IsSplit(), "release_io is not supported by split/eager kernels");
  for (auto &info : loads_) {
    runner_->Release(info.dev);
    info.dev = nullptr;
  }
  for (auto &info : stores_) {
    runner_->Release(info.dev);
    info.dev = nullptr;
  }
}

py::object RtKernelPy::Output(py::object store) {
  auto op = PyToObj(store);
  auto &info = FindVectorInfo(stores_, op);
  if (info.size > 0) {
    ASSERT(info.dev);
    auto buf = info.host.request();
    ERROR_CHECK(rt::MemFromDev(buf.ptr, info.dev, info.size));
  }
  return info.host;
}

void RtKernelPy::SetOutput(py::object store, py::object val) {
  auto &info = FindVectorInfo(stores_, PyToObj(store));
  info.host = py::array(val);
  info.size = info.host.nbytes();
  info.set_host = true;
}

void RtKernelPy::ClearStoreMemory(py::object store) {
  auto &info = FindVectorInfo(stores_, PyToObj(store));
  info.clear_mem = true;
}

void RtKernelPy::PrepareStore(StoreInfo &info) {
  auto op = info.op;
  if (op->CheckFlag(OBJ_FLAG_STORE_TEMP)) {
    return;
  }
  if (!info.set_host) {
    size_t size = ITEM_SIZE[op->type_id_];
    for (size_t i = 0; i < op->shape_ref_->size; i++) {
      size *= op->shape_ref_->data[i];
    }
    std::vector<ssize_t> shape(op->shape_ref_->size);
    for (size_t i = 0; i < op->shape_ref_->size; i++) {
      shape[i] = op->shape_ref_->data[i];
    }
    info.size = size;
    info.host = py::array(py::dtype(GetBufferFormat(op->type_id_)), shape);
  }
  runner_->AllocStore(info);
}

void RtKernelPy::PrepareIO() {
  for (auto &info : loads_) {
    static_cast<NDAccess *>(info.op)->addr_.Reloc(reinterpret_cast<uint8_t *>(info.dev) + info.offset);
  }
  for (auto &info : stores_) {
    PrepareStore(info);
    static_cast<NDAccess *>(info.op)->addr_.Reloc(info.dev);
  }
}

void RtKernelPy::Reset() {
  runner_->Reset();
  if (kernel_.GetImpl()->KType() == kEager) {
    kernel_.Clear();
    loads_.clear();
    stores_.clear();
    for (auto ref : shape_) {
      delete ref;
    }
    shape_.clear();
    shape_vec_.clear();
  }
}

py::object RtKernelPy::Clone(py::object base, py::object remap) {
  struct _CloneHelper : public CloneHelper {
    IntArrayRef *GetClone(IntArrayRef *shape) override {
      auto it = ref_map_.find(shape);
      return it != ref_map_.end() ? static_cast<IntArrayRef *>(it->second) : shape;
    }
    ScalarRef *GetClone(ScalarRef *scalar) override {
      auto it = ref_map_.find(scalar);
      return it != ref_map_.end() ? static_cast<ScalarRef *>(it->second) : scalar;
    }
    NDObject *GetClone(NDObject *op) override {
      auto it = op_map_.find(op);
      return it != op_map_.end() ? it->second : nullptr;
    }
    void SetClone(NDObject *op, NDObject *clone) override { op_map_[op] = clone; }
    std::unordered_map<NDObject *, NDObject *> op_map_;
    std::unordered_map<void *, void *> ref_map_;
  };
  _CloneHelper helper;
  RtKernelPy *other = base.cast<RtKernelPyPtr>().get();
  for (auto &load : other->loads_) {
    auto &info = loads_.emplace_back();
    info.shape = load.shape;
    auto shape_ref = shape_.emplace_back(new IntArrayRef(info.shape));
    for (auto ref : other->shape_) {
      if (ref->data == load.shape.data()) {
        helper.ref_map_[ref] = shape_ref;
        break;
      }
    }
  }
  py::list remap_list = py::cast<py::list>(remap);
  size_t remap_size = remap_list.size();
  py::tuple remap_out(remap_size);
  for (size_t i = 0; i < remap_size; ++i) {
    if (py::isinstance<IntArrayRefPy>(remap_list[i])) {
      auto base = remap_list[i].cast<std::shared_ptr<IntArrayRefPy>>()->Get();
      auto ref = std::make_shared<IntArrayRefPy>();
      helper.ref_map_[base] = ref->Get();
      remap_out[i] = py::cast(ref);
    } else if (py::isinstance<ScalarRefPy>(remap_list[i])) {
      auto base = remap_list[i].cast<std::shared_ptr<ScalarRefPy>>();
      auto ref = std::make_shared<ScalarRefPy>();
      helper.ref_map_[&base->data_] = &ref->data_;
      remap_out[i] = py::cast(ref);
    }
  }
  for (auto ref : other->shape_) {
    if (helper.ref_map_.find(ref) == helper.ref_map_.end()) {
      auto clone = new IntArrayRef(shape_vec_.emplace_back(ref->data, ref->data + ref->size));
      helper.ref_map_[ref] = clone;
    }
  }
  for (size_t i = 0; i < other->loads_.size(); ++i) {
    auto *op = other->loads_[i].op;
    if (op && op->GetObjectType() == ObjectType::kGlobalAccess) {
      loads_[i].op = op->CloneUpdate(helper);
    }
  }
  kernel_.Clone(other->kernel_, helper);
  for (size_t i = 0; i < other->loads_.size(); ++i) {
    loads_[i].op = helper.GetClone(other->loads_[i].op);
  }
  for (auto &store : other->stores_) {
    auto &clone = stores_.emplace_back();
    clone.op = helper.GetClone(store.op);
  }
  for (size_t i = 0; i < remap_size; ++i) {
    if (py::isinstance<NDObjectPy>(remap_list[i])) {
      remap_out[i] = ObjToPy(helper.GetClone(PyToObj(remap_list[i])));
    }
  }
  return remap_out;
}

void RtKernelPy::Fork(int size, const std::string &comm_type) {
  ASSERT(size <= static_cast<int>(sizeof(g_mpc.pids) / sizeof(pid_t)));
  g_mpc.rank_size = size;
  auto shm_size = 1024;
  g_mpc.shmid = ::shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0600);
  for (int i = 1; i < size; ++i) {
    g_mpc.rank_id = i;
    auto pid = ::fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
      goto INIT_COMM;
    }
    g_mpc.pids[i - 1] = pid;
  }
  g_mpc.rank_id = 0;
INIT_COMM:
  int rank_id = g_mpc.rank_id;
  g_mpc.bars = (int64_t *)shmat(g_mpc.shmid, nullptr, 0);
  g_mpc.bars[rank_id] = 0;
  ERROR_CHECK(rt::SetDevice(rank_id));
  g_mpc.comm.Init(rank_id, size);
}

void RtKernelPy::Join() {
  // TODO: free g_mpc.comm
  ::shmdt((void *)(g_mpc.bars));
  if (g_mpc.rank_id > 0) {
    ::exit(0);
  }
  for (int i = 0; i < g_mpc.rank_size - 1; ++i) {
    ::wait(nullptr);
  }
  shmctl(g_mpc.shmid, IPC_RMID, nullptr);
}

void RtKernelPy::Barrier() {
  if (g_mpc.rank_size > 1 && g_mpc.bars) {
    auto cur_cnt = g_mpc.bars[g_mpc.rank_id] + 1;
    g_mpc.bars[g_mpc.rank_id] = cur_cnt;
    for (int i = 0; i < g_mpc.rank_size; ++i) {
      while (g_mpc.bars[i] != cur_cnt) {
        sleep(1);
      }
    }
  }
}

int RtKernelPy::RankId() { return g_mpc.rank_id; }

int RtKernelPy::RankSize() { return g_mpc.rank_size; }

class DevicePy {
 public:
  static std::string Arch() {
    g_system.Init();
    return AiCoreArchName(g_system.Arch());
  }
  static int CoreNum() {
    g_system.Init();
    return g_system.CoreNum();
  }
  static std::string SocName() {
    g_system.Init();
    return SocTypeName(g_system.SocName());
  }
};

PYBIND11_MODULE(_dvm_py, m) {
  RegDvmPy(m);
  py::class_<RtKernelPy, KernelPy, std::shared_ptr<RtKernelPy>>(m, "PyKernel")
    .def(py::init<const std::string &, const std::string &, int>())
    .def("global_access", &RtKernelPy::GlobalAccess, "create global access")
    .def("gather_load", &RtKernelPy::GatherLoad, "gather load array", py::arg("shape"), py::arg("index"),
         py::arg("type"), py::arg("axis") = 0, py::arg("gather_mode") = static_cast<int>(kElementGather))
    .def("slice_load", &RtKernelPy::SliceLoad, "load array")
    .def("stridedslice_load", &RtKernelPy::StridedSliceLoad, "load array")
    .def("multi_load", &RtKernelPy::MultiLoad, "load array(for reducescatter)")
    .def("pad_store", &RtKernelPy::PadStore, "pad store array")
    .def("one_hot", &RtKernelPy::OneHot, "emit onehot op")
    .def("allreduce", &RtKernelPy::AllReduce, "emit allreduce op")
    .def("allgather", &RtKernelPy::AllGather, "emit allgather op")
    .def("allgatherv2", &RtKernelPy::AllGatherV2, "emit allgatherv2 op")
    .def("reducescatter", &RtKernelPy::ReduceScatter, "emit reducescatter op")
    .def("custom", &RtKernelPy::Custom, "custom op", py::arg("full_name"), py::arg("inputs"), py::arg("attrs") = py::none())
    .def("ext_out", &RtKernelPy::ExtOut, "get item from multi-output op")
    .def("convert_to_bf16", &RtKernelPy::ConvertToBF16, "convert f32 array to bf16 array")
    .def("convert_from_bf16", &RtKernelPy::ConvertFromBF16, "convert bf16 array to f32 array")
    .def("reset", &RtKernelPy::Reset, "reset eager")
    .def("clone", &RtKernelPy::Clone, "clone kernel")
    .def("input", &RtKernelPy::Input, "get ouput array", py::arg("op"), py::arg("val"), py::arg("offset") = 0)
    .def("release_io", &RtKernelPy::ReleaseIO, "release input and output device memory")
    .def("output", &RtKernelPy::Output, "get ouput array")
    .def("set_output", &RtKernelPy::SetOutput, "set ouput array")
    .def("clear_store_memory", &RtKernelPy::ClearStoreMemory, "clear store memory")
    .def("tile", &RtKernelPy::Tile, "set tiling", py::arg("start"), py::arg("end"), py::arg("num"),
         py::arg("factor") = 0)
    .def("codegen", &RtKernelPy::CodeGen, "generate code")
    .def("perf", &RtKernelPy::Perf, "perf test")
    .def("msprof", &RtKernelPy::Msprof, "perf test", py::arg("path"), py::arg("test_num") = 10, py::arg("metric") = "ArithmeticUtilization")
    .def("msprof_start", &RtKernelPy::MsprofStart, "start msprof", py::arg("path"), py::arg("metric") = "ArithmeticUtilization")
    .def("msprof_stop", &RtKernelPy::MsprofStop, "stop msprof")
    .def("run", &RtKernelPy::Run, "run kernel")
    .def("dry_run", &RtKernelPy::DryRun, "dry run vm")
    .def_static("reg_custom", &RtKernelPy::RegCustom, "register custom")
    .def_static("init_comm", &RtKernelPy::InitComm, "init communicatior")
    .def_static("set_cube_store_type", &RtKernelPy::SetCubeStoreType, "set sync type")
    .def_static("set_lazy_tuning", &RtKernelPy::SetLazyTuning, "set lazy tuning")
    .def_static("fork", &RtKernelPy::Fork, "fork process", py::arg("size"), py::arg("comm_type") = "")
    .def_static("join", &RtKernelPy::Join, "join process")
    .def_static("barrier", &RtKernelPy::Barrier, "barrier process")
    .def_static("rank_id", &RtKernelPy::RankId, "get current rank id")
    .def_static("rank_size", &RtKernelPy::RankSize, "get current rank size");

  (void)py::class_<DevicePy, std::shared_ptr<DevicePy>>(m, "Device")
    .def_static("arch", &DevicePy::Arch, "Get system architecture")
    .def_static("core_num", &DevicePy::CoreNum, "Get soc core number")
    .def_static("soc_name", &DevicePy::SocName, "Get soc name");
}
}  // namespace dvm
