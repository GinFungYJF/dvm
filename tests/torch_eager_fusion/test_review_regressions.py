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
"""Regressions for defects found in code review.

Each test targets a case that produced silently wrong numbers or a stuck graph,
so they assert on dtype and on counters as well as on values.
"""

import gc

import pytest
import torch

import torch_dvm
from conftest import CUDA, assert_same, eager, fused


@CUDA
@pytest.mark.parametrize("name", ["sqrt", "exp", "log", "reciprocal", "rsqrt", "sigmoid"])
def test_float_math_on_int_input_promotes(name):
    """ATen promotes an integer input to float; the fused path must not keep int."""
    fn = getattr(torch, name)
    t = torch.tensor([4, 9, 16], device="cuda", dtype=torch.int32)
    with eager():
        expected = fn(t)
    with fused():
        actual = fn(t)
    assert actual.dtype == expected.dtype == torch.float32, (
        "{} returned {} instead of float32".format(name, actual.dtype)
    )
    assert_same(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["relu", "neg", "abs"])
def test_integer_preserving_unary_keeps_dtype(name):
    """These keep int32 in ATen and the VM has an INT32 form, so fuse them."""
    fn = getattr(torch, name)
    t = torch.tensor([-4, 0, 9], device="cuda", dtype=torch.int32)
    with eager():
        expected = fn(t)
    with fused():
        actual = fn(t)
    assert actual.dtype == expected.dtype == torch.int32
    assert_same(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["round", "floor", "ceil", "trunc"])
def test_no_int32_opcode_falls_back(name):
    """The VM has no INT32 round/floor/ceil/trunc; these must not trap."""
    fn = getattr(torch, name)
    t = torch.tensor([-4, 0, 9], device="cuda", dtype=torch.int32)
    with eager():
        expected = fn(t)
    with fused():
        actual = fn(t)
    assert actual.dtype == expected.dtype == torch.int32
    assert_same(actual, expected)


@CUDA
def test_int_tensor_true_division_promotes():
    a = torch.tensor([5, 7, 9], device="cuda", dtype=torch.int32)
    b = torch.tensor([2, 2, 4], device="cuda", dtype=torch.int32)
    with eager():
        expected = a / b
    with fused():
        actual = a / b
    assert actual.dtype == expected.dtype == torch.float32
    assert_same(actual, expected)


@CUDA
def test_single_op_graph_is_compiled():
    """A live single-op graph also executes through DVM."""
    a = torch.randn(4096, device="cuda")
    b = torch.randn(4096, device="cuda")
    with eager():
        expected = a * b
    torch_dvm.enable()
    try:
        torch_dvm.reset_stats()
        out = a * b
        torch_dvm.flush()
    finally:
        torch_dvm.disable()
    stats = torch_dvm.stats()
    assert stats["flushes"] == 1
    assert stats["emitted"] == 1
    assert stats["graph_sizes"] == {1: 1}
    assert_same(out, expected)


@CUDA
def test_stream_switch_rebinds_stream():
    """After a switch the next graph must launch on the new stream."""
    s2 = torch.cuda.Stream()
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    with fused():
        with torch.cuda.stream(s2):
            x = a * b
        # Back on the default stream: the deferred work must be visible here.
        y = x + 1
    s2.synchronize()
    torch.cuda.synchronize()
    assert_same(y, a.clone() * b.clone() + 1)



@CUDA
def test_host_read_of_pending_scalar():
    """Reading a pending scalar on the host must materialise the graph first."""
    a = torch.randn(64, device="cuda")
    torch_dvm.enable()
    try:
        torch_dvm.reset_stats()
        pending = a * 2  # deferred
        value = pending[0].item()
    finally:
        torch_dvm.disable()
    expected = (a.clone() * 2)[0].item()
    assert value == expected, "host read returned {} not {}".format(value, expected)
    assert torch_dvm.stats()["flushes"] >= 1


@CUDA
def test_empty_flush_releases_loads_from_failed_emitter():
    """A failed scalar conversion leaves a load but no output to materialise."""
    torch_dvm.disable()
    gc.collect()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    x = torch.ones(1 << 20, device="cuda")
    input_bytes = x.nbytes
    with fused():
        with pytest.raises(RuntimeError, match="cannot be converted"):
            torch.clamp(x, min=1 + 2j)
        del x
        gc.collect()
        assert torch.cuda.memory_allocated() >= allocated_before + input_bytes
        torch_dvm.flush()
        assert torch.cuda.memory_allocated() == allocated_before
        # A second empty flush must also allow a fresh graph to execute.
        torch_dvm.flush()
        actual = torch.ones(32, device="cuda") * 3
    assert_same(actual, torch.full_like(actual, 3))


@CUDA
def test_empty_flush_boundaries_preserve_next_graph():
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    expected_first = (x + 1) * 2
    with fused():
        torch_dvm.reset_stats()
        first = (x + 1) * 2
        torch_dvm.flush()
        for _ in range(8):
            torch_dvm.flush()
        stats = torch_dvm.stats()
        assert stats["flushes"] == 1
        assert stats["emitted"] == 2
        x.fill_(7)
        second = (x + 1) * 2
        torch_dvm.flush()
        assert torch_dvm.stats()["graph_sizes"] == {2: 2}
    assert_same(first, expected_first)
    assert_same(second, torch.full_like(second, 16))


@CUDA
@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_", "div_"])
def test_inplace_pending_read_operand_boundary(operation):
    x = torch.arange(1, 129, device="cuda", dtype=torch.float32)
    destination = torch.ones_like(x)
    expected = getattr(destination.clone(), operation)(x * 2)
    with fused():
        torch_dvm.reset_stats()
        pending = x * 2
        getattr(destination, operation)(pending)
        assert torch_dvm.stats()["flushes"] == (0 if operation == "add_" else 1)
        torch_dvm.flush()
        assert torch_dvm.stats()["graph_sizes"] == ({2: 1} if operation == "add_" else {1: 2})
    assert_same(destination, expected)
