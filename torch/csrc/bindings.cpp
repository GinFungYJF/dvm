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
#include <torch/extension.h>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <sstream>
#include <string>

#include "dispatch/overrides.h"
#include "dispatch/registry.h"
#include "flags.h"
#include "fusion/core/graph.h"

namespace {

using dvm_torch::FlushReason;
using dvm_torch::FusionMode;
using dvm_torch::GraphManager;
using dvm_torch::OpKey;

FusionMode ParseMode(const std::string &mode) {
  if (mode.empty() || mode == "override" || mode == "cpp") {
    return FusionMode::kOverride;
  }
  TORCH_CHECK(false, "torch_dvm: unknown mode ", mode, " (only \"override\" is supported)");
  return FusionMode::kOverride;
}

void ParseOpList(const std::string &text, std::unordered_set<std::string> *out) {
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out->insert(item);
    }
  }
}

void Enable(const std::string &mode, int64_t level, bool guards,
            const std::string &dump_dir, bool dump_as_text,
            const std::string &disable_ops, const std::string &enable_ops) {
  dvm_torch::Flags f = dvm_torch::GetFlags();
  f.enabled = true;
  f.mode = ParseMode(mode);
  f.level = static_cast<int>(level);
  f.guards = guards;
  f.dump_dir = dump_dir;
  f.dump_as_text = dump_as_text;
  f.disable_ops.clear();
  f.enable_ops.clear();
  ParseOpList(disable_ops, &f.disable_ops);
  ParseOpList(enable_ops, &f.enable_ops);
  dvm_torch::SetFlags(f);
  dvm_torch::InstallOverrides();
}

void EnableFromEnv() {
  dvm_torch::ReloadFlagsFromEnv();
  if (dvm_torch::GetFlags().enabled) {
    dvm_torch::InstallOverrides();
  }
}

void Disable() {
  GraphManager::Instance().Flush(FlushReason::kManual);
  dvm_torch::UninstallOverrides();
  dvm_torch::Flags f = dvm_torch::GetFlags();
  f.enabled = false;
  f.mode = FusionMode::kOff;
  dvm_torch::SetFlags(f);
}

py::dict StatsDict() {
  const auto &s = GraphManager::Instance().GetStats();
  py::dict d;
  d["emitted"] = s.emitted;
  d["flushes"] = s.flushes;
  d["bails"] = s.bails;
  d["stores_skipped"] = s.stores_skipped;
  d["codegen_us"] = s.codegen_us;
  d["launch_us"] = s.launch_us;

  py::dict reasons;
  for (size_t i = 0; i < s.by_reason.size(); ++i) {
    if (s.by_reason[i] != 0) {
      reasons[py::str(dvm_torch::FlushReasonName(static_cast<FlushReason>(i)))] = s.by_reason[i];
    }
  }
  d["by_reason"] = reasons;

  py::dict hist;
  for (const auto &kv : s.graph_size_histogram) {
    hist[py::int_(kv.first)] = kv.second;
  }
  d["graph_sizes"] = hist;
  d["summary"] = s.ToString();

  const auto &reg = dvm_torch::GetRegistry();
  d["registered_fusible"] = reg.fusible;
  d["registered_guards"] = reg.guarded;
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "DVM eager fusion for native PyTorch";

  m.def("enable", &Enable, py::arg("mode") = "", py::arg("level") = 2,
        py::kw_only(), py::arg("guards") = true,
        py::arg("dump_dir") = "./dvm_dump", py::arg("dump_as_text") = false,
        py::arg("disable_ops") = "", py::arg("enable_ops") = "");
  m.def("enable_from_env", &EnableFromEnv);
  m.def("disable", &Disable);
  m.def("is_enabled", []() { return dvm_torch::GetFlags().enabled; });
  m.def("flush", [](const std::string &reason) {
    FlushReason r = FlushReason::kManual;
    if (reason == "HostRead") {
      r = FlushReason::kHostRead;
    } else if (reason == "NonFusibleOp") {
      r = FlushReason::kNonFusibleOp;
    }
    GraphManager::Instance().Flush(r);
  }, py::arg("reason") = "Manual");
  m.def("stats", &StatsDict);
  m.def("reset_stats", []() { GraphManager::Instance().ResetStats(); });
  m.def("flush_if_pending", []() { dvm_torch::dvm_torch_flush_if_pending(); });
}
