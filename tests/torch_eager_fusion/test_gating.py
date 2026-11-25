# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gating: the package must be invisible until it is explicitly turned on."""

import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import torch_dvm
from conftest import CUDA, assert_same, eager, fused


def test_installed_fusion_package_works_on_its_own(tmp_path):
    try:
        requirements = metadata.requires("torch_dvm")
    except metadata.PackageNotFoundError:
        pytest.skip("requires an installed torch_dvm distribution")
    assert "dvm-py" not in {canonicalize_name(Requirement(req).name) for req in requirements or ()}

    # Relocate only the fusion package so a sibling SDK cannot satisfy its ELF dependencies.
    shutil.copytree(Path(torch_dvm.__file__).resolve().parent, tmp_path / "torch_dvm")
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.modules["dvm_py"] = sys.modules["dvm"] = None
import torch_dvm
assert Path(torch_dvm._C.__file__).parent == Path(sys.argv[1]) / "torch_dvm"
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], check=True)


@CUDA
def test_disabled_records_no_graph():
    torch_dvm.disable()
    a = torch.arange(512, device="cuda", dtype=torch.float32)
    expected = a * 32 + 31
    torch_dvm.reset_stats()
    actual = a
    for _ in range(5):
        actual = actual * 2 + 1
    stats = torch_dvm.stats()
    assert stats["flushes"] == 0
    assert stats["emitted"] == 0
    assert_same(actual, expected)


@CUDA
def test_disable_restores_dispatch():
    """After disable(), the numeric behaviour must be unchanged again."""
    a = torch.randn(256, device="cuda")
    torch_dvm.enable()
    _ = a * 2
    torch_dvm.disable()
    expected = a * 2
    assert_same(expected, a.clone() * 2.0)


@CUDA
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"disable_ops": "mul"}, id="disable-mul"),
        pytest.param({"enable_ops": "add"}, id="enable-add"),
    ],
)
def test_op_filter(options):
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    with eager():
        expected_product = a * b
        expected = (expected_product + 1) + b
    with fused(**options):
        torch_dvm.reset_stats()
        product = a * b
        assert torch_dvm.stats()["emitted"] == 0
        actual = (product + 1) + b
    assert torch_dvm.stats()["graph_sizes"] == {2: 1}
    assert_same(product, expected_product)
    assert_same(actual, expected)


def test_env_gate_is_process_wide():
    """TORCH_DVM_FUSION set before import installs the overrides."""
    env = dict(os.environ)
    env["TORCH_DVM_FUSION"] = "True"
    code = (
        "import torch;"
        "original = torch.cuda.Event.record;"
        "import torch_dvm;"
        "assert torch.cuda.Event.record is not original;"
        "assert not torch.cuda.is_initialized();"
        "torch.cuda.init() if torch.cuda.is_available() else None;"
        "print(torch_dvm.is_enabled())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("True")


def test_no_env_means_not_enabled():
    env = dict(os.environ)
    env.pop("TORCH_DVM_FUSION", None)
    code = (
        "import torch;"
        "original = (torch.cuda.Event.record, torch.cuda.Stream.query, "
        "torch.cuda.synchronize, torch.Stream, torch.Event);"
        "import torch_dvm;"
        "assert original == (torch.cuda.Event.record, torch.cuda.Stream.query, "
        "torch.cuda.synchronize, torch.Stream, torch.Event);"
        "assert not torch.cuda.is_initialized();"
        "print(torch_dvm.is_enabled())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("False")


@CUDA
def test_auto_enable_then_disable_repeatedly():
    a = torch.randn(128, device="cuda")
    for _ in range(3):
        torch_dvm.enable()
        out = a * 2
        torch_dvm.flush()
        torch_dvm.disable()
        assert_same(out, a.clone() * 2.0)
