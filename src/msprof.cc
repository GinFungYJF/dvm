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

#include <map>
#include <algorithm>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "kernel.h"
#include "msprof.h"

using ShapeVector = std::vector<int64_t>;

// GE task info task_type
enum class TaskInfoTaskType {
  TASK_TYPE_AI_CORE = 0,
  TASK_TYPE_AI_CPU = 1,
  TASK_TYPE_AIV = 2,
  TASK_TYPE_WRITE_BACK = 3,
  TASK_TYPE_MIX_AIC = 4,
  TASK_TYPE_MIX_AIV = 5,
  TASK_TYPE_FFTS_PLUS = 6,
  TASK_TYPE_DSA = 7,
  TASK_TYPE_DVPP = 8,
  TASK_TYPE_HCCL = 9,
  MSPROF_RTS = 11,
  MSPROF_UNKNOWN_TYPE = 1000,
};

constexpr uint32_t kTensorInfoBytes = 44UL;
constexpr uint32_t kTensorInfoBytesWithCap = 56U;

namespace dvm {
const TensorDtypeMs MAP_DTYPE_TO_MSDTYPE[DataType::kDataTypeEnd + 1] = {BOOL_,  FLOAT16_, DT_BF16,         FLOAT32_,
                                                                        INT32_, INT64_,   NUMBER_TYPE_END_};

void InitLaunchApi(const uint64_t name_hash, MsprofApi *api) {
  const auto kernel_type_hash = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  api->type = kernel_type_hash;
  api->level = MSPROF_REPORT_NODE_LEVEL;
  api->itemId = name_hash;
}
uint64_t GetMsprofHashId(const char *info) {
  uint64_t hash_id = MsprofGetHashId(info, strlen(info));
  return hash_id;
}

void MsprofHelper::BuildSingleTensorInfo(const uint64_t opName_hash_id, const size_t index_begin,
                                         const size_t index_end, TensorInfoWrapper *tensor_info_wrapper) {
  auto &tensor_info = tensor_info_wrapper->tensor_info;
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = MSPROF_REPORT_NODE_LEVEL;
  tensor_info_wrapper->tensor_num = index_end - index_begin;
  tensor_info.dataLen =
    kTensorInfoBytesWithCap + kTensorInfoBytes * (static_cast<uint32_t>(tensor_info_wrapper->tensor_num) - 1U);
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  prof_tensor_data->opName = opName_hash_id;
  prof_tensor_data->tensorNum = tensor_info_wrapper->tensor_num;
  for (size_t tensor_index = index_begin; tensor_index < index_end; tensor_index++) {
    size_t k = tensor_index - index_begin;
    prof_tensor_data->tensorData[k].tensorType =
      tensor_index < info_.input_size ? MSPROF_GE_TENSOR_TYPE_INPUT : MSPROF_GE_TENSOR_TYPE_OUTPUT;
    prof_tensor_data->tensorData[k].format = FormatMs::ND_;
    prof_tensor_data->tensorData[k].dataType = info_.data_types[tensor_index];
    uint64_t shape_size = info_.shapes[tensor_index]->size;
    for (uint64_t i = 0; i < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++i) {
      prof_tensor_data->tensorData[k].shape[i] = i < shape_size ? info_.shapes[tensor_index]->data[i] : 0;
    }
  }
}

void MsprofHelper::UpdateTensorShape(const size_t index_begin, const size_t index_end,
                                     TensorInfoWrapper *tensor_info_wrapper) {
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info_wrapper->tensor_info.data);
  for (size_t tensor_index = index_begin; tensor_index < index_end; tensor_index++) {
    size_t k = tensor_index - index_begin;
    uint64_t shape_size = info_.shapes[tensor_index]->size;
    for (uint64_t i = 0; i < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++i) {
      prof_tensor_data->tensorData[k].shape[i] = i < shape_size ? info_.shapes[tensor_index]->data[i] : 0;
    }
  }
}

void MsprofHelper::InitReportNode() {
  MsprofCompactInfo &basic_info = addition_info_.node_basic_info;
  basic_info.level = MSPROF_REPORT_NODE_LEVEL;
  basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  auto &prof_node_basic_info = basic_info.data.nodeBasicInfo;
  uint64_t opName_hash_id = GetMsprofHashId(info_.op_fullname);
  prof_node_basic_info.opName = opName_hash_id;
  prof_node_basic_info.blockDim = info_.block_dim;
  prof_node_basic_info.opType = GetMsprofHashId(info_.op_name);
  MsprofAdditionalInfo &context_id_info = addition_info_.context_id_info;
  context_id_info.level = MSPROF_REPORT_NODE_LEVEL;
  context_id_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  auto ctx_id = reinterpret_cast<MsprofContextIdInfo *>(context_id_info.data);
  ctx_id->opName = prof_node_basic_info.opName;
  ctx_id->ctxIdNum = 1;
  ctx_id->ctxIds[0] = 0;
  size_t total_size = info_.input_size + info_.output_size;
  for (size_t i = 0U; i < total_size; i += MSPROF_GE_TENSOR_DATA_NUM) {
    TensorInfoWrapper tensor_info_wrapper;
    BuildSingleTensorInfo(opName_hash_id, i, std::min(total_size, (i + MSPROF_GE_TENSOR_DATA_NUM)),
                          &tensor_info_wrapper);
    addition_info_.tensor_info_wrappers.emplace_back(tensor_info_wrapper);
  }
  InitLaunchApi(opName_hash_id, &addition_info_.api);
}

void MsprofHelper::UpdateReportNode(uint32_t block_dim) {
  addition_info_.node_basic_info.data.nodeBasicInfo.blockDim = block_dim;
  size_t total_size = info_.input_size + info_.output_size;
  for (size_t i = 0U; i < total_size; i += MSPROF_GE_TENSOR_DATA_NUM) {
    UpdateTensorShape(i, std::min(total_size, (i + MSPROF_GE_TENSOR_DATA_NUM)),
                      &addition_info_.tensor_info_wrappers[i / MSPROF_GE_TENSOR_DATA_NUM]);
  }
}

void MsprofHelper::Update(uint32_t kernel_target) {
  addition_info_.api.beginTime = MsprofSysCycleTime();
  auto &prof_node_basic_info = addition_info_.node_basic_info.data.nodeBasicInfo;
  if (kernel_target == Code::kTargetCube) {
    prof_node_basic_info.taskType = static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_AI_CORE);
  } else if (kernel_target == Code::kTargetMix) {
    prof_node_basic_info.taskType = static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_MIX_AIC);
  } else {
    prof_node_basic_info.taskType = static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_AIV);
  }
}

void MsprofHelper::ReportTask() {
  const uint64_t prof_time = MsprofSysCycleTime();
  auto tid = syscall(SYS_gettid);
  if (g_system.profiler_level_ >= Level0) {
    if (addition_info_.node_basic_info.data.nodeBasicInfo.taskType ==
        static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_MIX_AIC)) {
      addition_info_.context_id_info.threadId = static_cast<uint32_t>(tid);
      addition_info_.context_id_info.timeStamp = prof_time;
      MsprofReportAdditionalInfo(false, &addition_info_.context_id_info, sizeof(MsprofAdditionalInfo));
    }
    addition_info_.api.endTime = prof_time;
    addition_info_.api.threadId = static_cast<uint32_t>(tid);
    MsprofReportApi(false, &addition_info_.api);
  }
  if (g_system.profiler_level_ >= Level1) {
    addition_info_.node_basic_info.timeStamp = prof_time;
    addition_info_.node_basic_info.threadId = static_cast<uint32_t>(tid);
    MsprofReportCompactInfo(false, &addition_info_.node_basic_info, sizeof(MsprofCompactInfo));
    for (auto &tensor_info_wrapper : addition_info_.tensor_info_wrappers) {
      tensor_info_wrapper.tensor_info.timeStamp = prof_time;
      tensor_info_wrapper.tensor_info.threadId = static_cast<uint32_t>(tid);
      MsprofReportAdditionalInfo(false, &tensor_info_wrapper.tensor_info, sizeof(MsprofAdditionalInfo));
    }
  }
}
}  // namespace dvm
