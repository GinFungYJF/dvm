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

#ifndef _DVM_TORCH_DISPATCH_REGISTRY_H_
#define _DVM_TORCH_DISPATCH_REGISTRY_H_

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>

#include <string>
#include <vector>

namespace dvm_torch {

/** Op identity: name plus overload, exactly as the dispatcher keys them. */
struct OpKey {
  std::string name;
  std::string overload;

  bool operator==(const OpKey &other) const {
    return name == other.name && overload == other.overload;
  }
};

std::string ToString(const OpKey &k);

/** Every override installed so far, so they can all be removed again. */
struct Registry {
  std::vector<c10::RegistrationHandleRAII> handles;
  size_t fusible{0};
  size_t guarded{0};
};

Registry &GetRegistry();

/** Install every override. Idempotent. */
void InstallOverrides();
/** Remove every override and forget captured kernels. */
void UninstallOverrides();
bool OverridesInstalled();

}  // namespace dvm_torch

#endif  // _DVM_TORCH_DISPATCH_REGISTRY_H_
