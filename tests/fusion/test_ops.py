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
"""Op equivalence: every fusible op must match eager exactly."""

import pytest
import torch

import dvm_torch
from conftest import CUDA, assert_same, eager, fused

SHAPES = [(1,), (1024,), (64, 64), (3, 5, 7), (4097,)]

BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}

UNARY = {
    "neg": torch.neg,
    "relu": torch.relu,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "reciprocal": torch.reciprocal,
    "round": torch.round,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "trunc": torch.trunc,
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", sorted(BINARY))
@CUDA
def test_binary(shape, name):
    fn = BINARY[name]
    a = torch.randn(*shape, device="cuda")
    b = torch.randn(*shape, device="cuda").abs() + 0.5
    with eager():
        expected = fn(a, b)
    with fused():
        actual = fn(a, b)
    assert_same(actual, expected)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", sorted(UNARY))
@CUDA
def test_unary(shape, name):
    fn = UNARY[name]
    # Positive input so log/sqrt/reciprocal stay finite on both paths.
    a = torch.randn(*shape, device="cuda").abs() + 0.5
    with eager():
        expected = fn(a)
    with fused():
        actual = fn(a)
    assert_same(actual, expected)


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [((4, 1), (1, 5)), ((3,), (2, 4, 3)), ((1,), (1024,)), ((2, 1, 6), (1, 4, 1))],
)
@CUDA
def test_broadcast(shape_a, shape_b):
    a = torch.randn(*shape_a, device="cuda")
    b = torch.randn(*shape_b, device="cuda")
    with eager():
        expected = a * b
    with fused():
        actual = a * b
    assert_same(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@CUDA
def test_dtype(dtype):
    a = torch.randn(512, device="cuda", dtype=dtype)
    b = torch.randn(512, device="cuda", dtype=dtype)
    with eager():
        expected = a * b + 1.0
    with fused():
        actual = a * b + 1.0
    assert_same(actual, expected, tol=0.0 if dtype is torch.float32 else 1e-2)


@CUDA
def test_int32_tensor_ops():
    a = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    b = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    for fn in (lambda x, y: x + y, lambda x, y: x * y, torch.maximum):
        with eager():
            expected = fn(a, b)
        with fused():
            actual = fn(a, b)
        assert_same(actual, expected)


@pytest.mark.parametrize("name", ["maximum", "minimum"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@CUDA
def test_minimum_maximum_capture_and_nan(name, dtype):
    a = torch.tensor([float("nan"), 1, -2, float("inf")], device="cuda", dtype=dtype)
    b = torch.tensor([1, float("nan"), 3, float("-inf")], device="cuda", dtype=dtype)
    fn = getattr(torch, name)
    with eager():
        expected = fn(a, b)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(a, b)
    assert dvm_torch.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected, equal_nan=True)


@CUDA
def test_int_scalar_promotion():
    """Integer true division promotes to float; the fused path must not lie."""
    a = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    with eager():
        expected = a / 2
    with fused():
        actual = a / 2
    assert actual.dtype == expected.dtype == torch.float32
    assert_same(actual, expected)


@CUDA
def test_long_chain_reduces_to_one_kernel():
    x = torch.randn(65536, device="cuda")
    keep = []  # keep the result alive so the graph is not dead-code eliminated

    def chain(t):
        y = t
        for _ in range(10):
            y = y * 1.0001 + 0.001
        keep.append(y)
        if len(keep) > 2:
            keep.pop(0)
        return y

    with eager():
        expected = chain(x)
    with fused():
        actual = chain(x)
    assert_same(actual, expected)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with fused():
            chain(x)
        torch.cuda.synchronize()
    kernels = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1, "expected one fused launch, saw {}".format(len(kernels))
