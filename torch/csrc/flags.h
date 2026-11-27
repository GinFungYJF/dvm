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

#ifndef _DVM_TORCH_FLAGS_H_
#define _DVM_TORCH_FLAGS_H_

#include <string>
#include <unordered_set>

namespace dvm_torch {

/** Capture mode selected by TORCH_DVM_FUSION / torch_dvm.enable(mode). */
enum class FusionMode : int {
  kOff = 0,      ///< No overrides registered; behaviour is stock PyTorch.
  kOverride = 1, ///< Per-operator C++ wrappers.
};

struct Flags {
  bool enabled = false;
  FusionMode mode = FusionMode::kOff;
  int level = 2;      ///< 1 = conservative op set, 2 = full op set.
  bool dump_as_text = false;
  std::string dump_dir = "./dvm_dump";
  bool guards = true; ///< Register a flush guard on every CUDA op.
  std::unordered_set<std::string> disable_ops;
  std::unordered_set<std::string> enable_ops;
};

const Flags &GetFlags();
void SetFlags(const Flags &f);
/** Parse a TORCH_DVM_FUSION style string. Returns false on malformed input. */
bool ParseFlags(const std::string &text, Flags *out);
/** Re-read TORCH_DVM_FUSION from the environment (process-wide). */
void ReloadFlagsFromEnv();

}  // namespace dvm_torch

#endif  // _DVM_TORCH_FLAGS_H_
