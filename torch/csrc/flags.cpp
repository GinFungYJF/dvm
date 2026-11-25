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
#include "flags.h"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace dvm_torch {
namespace {

Flags &MutableFlags() {
  static Flags f;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    const char *text = std::getenv("TORCH_DVM_FUSION");
    if (text != nullptr) {
      Flags parsed;
      if (ParseFlags(text, &parsed)) {
        f = parsed;
      }
    }
  }
  return f;
}

std::string Trim(const std::string &s) {
  size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::string ToLower(const std::string &s) {
  std::string out = s;
  for (auto &c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

bool IsTrue(const std::string &v) {
  std::string s = ToLower(v);
  return s == "true" || s == "1" || s == "on" || s == "yes";
}

bool IsFalse(const std::string &v) {
  std::string s = ToLower(v);
  return s == "false" || s == "0" || s == "off" || s == "no" || s.empty();
}

void SplitSet(const std::string &value, std::unordered_set<std::string> *out) {
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = Trim(item);
    if (!item.empty()) {
      out->insert(item);
    }
  }
}

}  // namespace

const Flags &GetFlags() { return MutableFlags(); }

void SetFlags(const Flags &f) { MutableFlags() = f; }

void ReloadFlagsFromEnv() {
  Flags f;
  const char *text = std::getenv("TORCH_DVM_FUSION");
  if (text != nullptr) {
    ParseFlags(text, &f);
  }
  MutableFlags() = f;
}

bool ParseFlags(const std::string &text, Flags *out) {
  // Accepted forms:
  //   ""              -> disabled
  //   "True" / "1"    -> enabled, O2, C++ override mode
  //   "level=O1"      -> enabled at level 1
  //   "True disable_ops=where,sum dump_as_text dump_dir=/tmp/dvm"
  std::string s = Trim(text);
  if (s.empty() || IsFalse(s)) {
    out->enabled = false;
    out->mode = FusionMode::kOff;
    return true;
  }

  Flags f;
  f.enabled = true;
  f.mode = FusionMode::kOverride;

  std::stringstream ss(s);
  std::string token;
  while (ss >> token) {
    auto eq = token.find('=');
    if (eq == std::string::npos) {
      if (IsTrue(token)) {
        f.enabled = true;
      } else if (IsFalse(token)) {
        f.enabled = false;
        f.mode = FusionMode::kOff;
      } else if (token == "dump_as_text") {
        f.dump_as_text = true;
      } else if (token == "compile=fast") {
        f.compile_fast = true;
      } else if (token.rfind("level=", 0) == 0) {
        std::string v = token.substr(6);
        f.level = (v == "O1" || v == "1") ? 1 : 2;
      } else {
        return false;
      }
      continue;
    }
    std::string key = token.substr(0, eq);
    std::string value = token.substr(eq + 1);
    if (key == "level") {
      f.level = (value == "O1" || value == "1") ? 1 : 2;
    } else if (key == "disable_ops") {
      SplitSet(value, &f.disable_ops);
    } else if (key == "enable_ops") {
      SplitSet(value, &f.enable_ops);
    } else if (key == "dump_dir") {
      f.dump_dir = value;
    } else if (key == "dump_as_text") {
      f.dump_as_text = IsTrue(value);
    } else if (key == "compile") {
      f.compile_fast = value == "fast";
    } else if (key == "mode") {
      if (value == "override" || value == "cpp") {
        f.mode = FusionMode::kOverride;
      } else if (value == "off") {
        f.enabled = false;
        f.mode = FusionMode::kOff;
      }
    } else {
      return false;
    }
  }
  *out = f;
  return true;
}

}  // namespace dvm_torch
