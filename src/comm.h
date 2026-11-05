/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef _DVM_COMMUNICATOR_H_
#define _DVM_COMMUNICATOR_H_

#include <cstdint>
#include <vector>

#include "isa.h"
#include "runtime.h"

namespace dvm {
inline constexpr size_t MAX_RANK_SIZE = 8;
inline constexpr size_t MAX_BUFFER_BYTES = 204 * 1024 * 1024;  // 204MB
inline constexpr size_t IPC_NAME_SIZE = 65;
class SocketChannel;

class Communicator {
 public:
  virtual ~Communicator() {}

  uint8_t *GetPeerMemPtr(size_t i) const { return peer_mem_[i % rank_size_]; };
  const int GetRankId() const { return rank_id_; }
  const int GetRankSize() const { return rank_size_; }

 protected:
  int rank_id_;         // global rank id
  int rank_size_;       // global rank size
  uint8_t **peer_mem_;  // virtual memory addr of peer memory
};

class DummyComm : public Communicator {
 public:
  DummyComm(int rank_id, int rank_size) {
    rank_id_ = rank_id;
    rank_size_ = rank_size;
    peer_mem_ = peer_mem_data_;
  }
  uint8_t *peer_mem_data_[MAX_RANK_SIZE] = {};
};

class MemoryComm : public Communicator {
 public:
  MemoryComm(int rank_id, int rank_size, const uint32_t *group_ranks);
  ~MemoryComm() override;
  bool Init();

 private:
  /**
   * @brief Collect all the device ids
   */
  void CollectDev();

  /**
   * @brief Collect all the process ids
   * @param pids a reference to the vector which stores pid. After calling this method, this variable in each process
   * will get all pids of the current task
   */
  void CollectPid(std::vector<int32_t> &pids);
  /**
   * @brief Collect all the shareable handles
   */
  void CollectShareableHandle();
  /**
   * @brief set white list for shareable handle. Only processes that have pids on the white list can access this peer
   * mem
   */
  void SetPidToShareableHandle(std::vector<int32_t> &pids);
  /**
   * @brief Enable each process to visit all the peer memories allocated by other process
   * @param names names of all the peer memories
   */
  void OpenIpcMem();
  /**
   * @brief Malloc peer memory and clear it. Each process only process the peer memory of npu which it occupies
   */
  void InitMem();
  /**
   * @brief Initialize the connection between any two npus
   */
  void InitCommon();
  /**
   * @brief Initialize peer memory
   */
  void InitCommMem();
  /**
   * @brief Free peer memory. Each process only processes the peer memory of npu which it occupies
   */
  void FreeCommMem();

  bool inited_;
  static int communicator_id_;
  int dev_id_;                                    // local device id
  aclrtDrvMemHandle physical_mem_handle_;         // physical memory handle of current device
  uint8_t *peer_mem_data_[MAX_RANK_SIZE] = {};    // virtual memory addr of peer memory
  uint64_t peer_mem_handle_[MAX_RANK_SIZE] = {};  // shareable memory handle of peer memory
  SocketChannel *socket_channel_;
  int dev_list_[MAX_RANK_SIZE];
  std::vector<uint32_t> group_rank_list_;
};

class HcclComm : public Communicator {
 public:
  HcclComm(void *hccl);

  void *Hccl() const { return hccl_; }

 protected:
  void *hccl_;
};

}  // namespace dvm

#endif  // _DVM_COMMUNICATOR_H_
