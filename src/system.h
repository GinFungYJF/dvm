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

#ifndef _DVM_SYSTEM_H_
#define _DVM_SYSTEM_H_
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include "dvm.h"

// rts_runtime
typedef int32_t rtError_t;
typedef char char_t;
const int32_t RT_ERROR_NONE = 0;  // success
typedef void *rtStream_t;
struct tagRtSmCtrl;
typedef struct tagRtSmCtrl rtSmDesc_t;
struct MsprofApi;
typedef void *VOID_PTR;

namespace dvm {
#ifdef DEBUG
#define ASSERT(cond)                                                                                            \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      std::cerr << "[ASSERT ERROR]" << __FILE__ << ":" << __LINE__ << ": ASSERT(" << #cond << ")" << std::endl; \
      DvmException("assert");                                                                                   \
    }                                                                                                           \
  } while (0)

#define MESS(var, init) \
  do {                  \
    var = init;         \
  } while (0)
#else
#define ASSERT(cond)
#define MESS(var, init)
#endif

#define EXCEPTION_IF(cond, error_str)       \
  do {                                      \
    if (cond) dvm::DvmException(error_str); \
  } while (0)
void DvmException(const char *error_str);

#define ERROR_CHECK(func)  \
  do {                     \
    if ((func) != 0) {     \
      DvmException(#func); \
    }                      \
  } while (0)

#define DVM_EXPORT __attribute__((visibility("default")))

enum AiCoreArch {
  kAiCore_C220,
  kAiCore_C310,
  kAiCore_Gpu,
  kAiCoreArchEnd,
};

enum class CoreType {
  kAIV,
  kAIC,
};

enum SocType {
  // C220(B)
  kAscend910B1,
  kAscend910B2,
  kAscend910B3,
  kAscend910B4,
  // C220(C)
  kAscend910_9391,
  kAscend910_9392,
  kAscend910_9381,
  kAscend910_9382,
  kAscend910_9372,
  kAscend910_9362,
  kAscend910_9363,
  // C310(PR)
  kAscend950PR_9579,
  kAscend950PR_9589,
  kAscend950PR_9599,
  kAscend950PR_958b,
  kAscend950PR_957b,
  kAscend950PR_957bx,
  kAscend950PR_957c,
  kAscend950PR_957d,
  kAscend950PR_950z,
  // C310(DT)
  kAscend950DT_950x,
  kAscend950DT_950y,
  kAscend950DT_9571,
  kAscend950DT_9572,
  kAscend950DT_9573,
  kAscend950DT_9574,
  kAscend950DT_9575,
  kAscend950DT_9576,
  kAscend950DT_9577,
  kAscend950DT_9578,
  kAscend950DT_9581,
  kAscend950DT_9582,
  kAscend950DT_9582x,
  kAscend950DT_9583,
  kAscend950DT_9584,
  kAscend950DT_9585,
  kAscend950DT_9586,
  kAscend950DT_9587,
  kAscend950DT_9588,
  kAscend950DT_9591,
  kAscend950DT_9592,
  kAscend950DT_9595,
  kAscend950DT_9596,
  kAscend950DT_95A1,
  kAscend950DT_95A2,
  // GPU
  kSocGpu,
  kSocUnknow,
};

const char *AiCoreArchName(AiCoreArch arch);
const char *SocTypeName(SocType type);

enum CubeStoreType {
  kCubeStoreGM = 0,
  kCubeStoreUB,
};

enum ProfilerLevel {
  Level0 = 0,
  Level1,
  Level2,
};

class NDObject;
struct CustomDef {
  typedef NDObject *(*CreateFunc)(const std::vector<NDObject *> &, const std::vector<ScalarRef> &);
  const char *name;
  CreateFunc create_func;
};

namespace pass {
class PassOptimizer;
}
class CubeTuner;
class Code;
class System : public Config {
 public:
  System() = default;
  ~System();
  void Init() {
    if (!inited_) {
      DoInit();
    }
  }
  bool IsInitialized() const { return inited_; }

  Config &SetDeterm() override;
  Config &UnsetDeterm() override;
  Config &SetOnlineTuner() override;
  Config &UnsetOnlineTuner() override;
  Config &SetLazyTuner() override;
  Config &UnsetLazyTuner() override;
  Config &SetVfFusion(int mode) override;
  Config &UnsetVfFusion() override;

  // hardware config
  AiCoreArch Arch() const { return arch_; }
  void SetLocalMemSize(uint64_t size) { local_mem_size_ = size; }
  uint64_t LocalMemSize() const { return local_mem_size_; }
  uint64_t UbWorkspaceSize() const { return ub_workspace_size_; }
  uint64_t SimtWorkspace() const { return 32 * 1024; }
  uint64_t L2Size() const { return l2_size_; }
  uint64_t L2CacheLineSize() const { return l2_cache_line_size_; }
  float L2DdrBandwidthRatio() const { return l2_ddr_bandwidth_ratio_; }
  uint64_t L1Size() const { return l1_size_; }
  // Capacity of each L0A/L0B, in bytes.
  uint64_t L0ABSize() const { return 64 * 1024; }
  uint64_t L0CSize() const { return l0c_size_; }
  uint64_t CoreNum(CoreType core_type = CoreType::kAIV) const {
    return core_type == CoreType::kAIV ? vector_core_num_ : cube_core_num_;
  }
  uint64_t EventNum() const { return event_num_; }
  SocType SocName() const { return soc_name_; }
  uint32_t BtSize() const { return bt_size_; }
  void SetCubeStoreType(CubeStoreType type) { cube_store_type_ = type; }
  CubeStoreType GetCubeStoreType() { return cube_store_type_; }

  // features config

  bool deterministic_{false};
  int vf_fusion_{0};
  bool enable_profile_{false};
  ProfilerLevel profiler_level_{Level0};
  CubeTuner *online_tuner_{nullptr};
  CubeTuner *lazy_tuner_{nullptr};
  CubeTuner *dyn_lazy_tuner_{nullptr};
  pass::PassOptimizer *pass_opt_{nullptr};

  // runtime api
  int (*code_launch_)(const System &self, const Code *code, void *extern_ws, void *stream){nullptr};
  void *func_handles_[4];
  void *get_ffts_addr_func_{nullptr};
  void *kernel_launch_func_{nullptr};

  const uint64_t *g_simd_func_offset_;
  const uint64_t *g_access_func_offset_;
  const uint64_t *g_visit_func_offset_;

  void RegCustom(const std::string &nspace, const std::string &so_path, const std::string &bin_path);
  void RegCustom(const std::string &nspace, const std::string &bin_path);
  uint64_t GetCustomFunc(const std::string &full_name) const;
  std::string GetCustomFuncName(uint64_t func_id) const;
  NDObject *CreateCustom(const std::string &op_name, const std::vector<NDObject *> &inputs,
                         const std::vector<ScalarRef> &attrs) const;

 private:
  void DoInit();
  void GetSocConfig();

  uint64_t local_mem_size_;
  uint64_t ub_workspace_size_;
  uint64_t l2_size_;
  uint64_t l2_cache_line_size_;
  // C220 and custom DVM_SOC_NAME values use the calibrated default; C310 uses the static SoC table.
  float l2_ddr_bandwidth_ratio_{5.0f};
  uint64_t l1_size_;
  uint64_t l0c_size_;
  uint64_t event_num_;
  uint64_t vector_core_num_;
  uint64_t cube_core_num_;
  AiCoreArch arch_;
  uint32_t bt_size_;
  SocType soc_name_{kSocUnknow};
  bool inited_{false};
  CubeStoreType cube_store_type_{kCubeStoreGM};
  void *renamed_bin_{nullptr};
  void *simt_bin_{nullptr};
  void *rt_handle_{nullptr};

  std::unordered_map<std::string, uint64_t> custom_funcs_;
  std::unordered_map<std::string, CustomDef::CreateFunc> custom_def_;

  std::vector<void *> bins_;   // TODO: simt_bin_ && renamed_bin_ add to here

#ifdef DVM_ARCH_GPU
  static int CodeLaunchGPU(const System &self, const Code *code, void *extern_ws, void *stream);
#else
  static int CodeLaunchRT(const System &self, const Code *code, void *extern_ws, void *stream);
  static int CodeLaunchACL_C220(const System &self, const Code *code, void *extern_ws, void *stream);
  static int CodeLaunchACL_C310(const System &self, const Code *code, void *extern_ws, void *stream);
#endif
};

extern System g_system;

inline constexpr uint64_t SIMD_BLOCK_SIZE = 32;
inline constexpr uint64_t SIMD_REPEAT_SIZE = 256;
inline constexpr uint64_t PARAM_TABLE_LIMIT = 4096;
inline constexpr uint32_t MATMUL_ALIGN_MAX = 1024;

inline constexpr uint32_t SIMD_DTYPE_END = DataType::kInt64;

extern const uint64_t ITEM_SIZE[DataType::kDataTypeEnd];
extern const char *DTYPE_NAMES[DataType::kDataTypeEnd];
extern const uint64_t ITEM_SIMD_WIDTH_MAX[DataType::kDataTypeEnd];
}  // namespace dvm
#endif  // _DVM_SYSTEM_H_
