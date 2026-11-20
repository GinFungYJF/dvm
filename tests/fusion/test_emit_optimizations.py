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
"""Numerical boundaries and graph recovery for optimized emit expressions."""

import pytest
import torch

import dvm_torch
from conftest import CUDA, eager, fused


def assert_exact_with_zero_sign(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    torch.testing.assert_close(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_neg_preserves_special_values_and_zero_sign(dtype):
    info = torch.finfo(dtype)
    x = torch.tensor([0., -0., 1., -1., info.tiny, -info.tiny,
                      info.tiny * info.eps, -info.tiny * info.eps,
                      info.max, -info.max, float("inf"), -float("inf")],
                     device="cuda", dtype=dtype)
    with eager():
        expected = -x
    with fused():
        dvm_torch.reset_stats()
        actual = -x
    assert dvm_torch.stats()["emitted"] == 1
    assert_exact_with_zero_sign(actual, expected)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16,
    pytest.param(torch.bfloat16, marks=pytest.mark.xfail(
        strict=True, reason="Existing GPU VM BF16 neg(NaN) produces -0, also before the emit rewrite"))])
def test_neg_preserves_nan(dtype):
    x = torch.tensor([float("nan")], device="cuda", dtype=dtype)
    with eager():
        expected = -x
    with fused():
        actual = -x
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@CUDA
def test_neg_preserves_int32_extremes():
    info = torch.iinfo(torch.int32)
    x = torch.tensor([info.min, info.max, -(2**24 + 1), 2**24 + 1, -1, 0, 1],
                     device="cuda", dtype=torch.int32)
    with eager():
        expected = -x
    with fused():
        actual = -x
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("divisor", [8., -8., .5, -.5, 1., -1., 2.**126, 2.**-126])
@pytest.mark.parametrize("inplace", [False, True])
def test_fp32_power_of_two_division_boundaries(divisor, inplace):
    info = torch.finfo(torch.float32)
    x = torch.tensor([0., -0., 1., -1., 7., -9., info.tiny, -info.tiny,
                      info.tiny * info.eps, -info.tiny * info.eps,
                      info.max, -info.max, float("inf"), -float("inf"), float("nan")],
                     device="cuda")
    destination = x.clone()
    alias = destination.detach()
    with eager():
        expected = x / divisor
    with fused():
        dvm_torch.reset_stats()
        actual = destination.div_(divisor) if inplace else x / divisor
    assert dvm_torch.stats()["emitted"] == 1
    assert_exact_with_zero_sign(actual, expected)
    if inplace:
        assert actual is destination
        assert_exact_with_zero_sign(alias, expected)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("divisor", [8., -8., .5])
def test_low_precision_scalar_division_preserves_results(dtype, divisor):
    x = torch.tensor([0., -0., 1., -1., 7., -9., 64., -128.], device="cuda", dtype=dtype)
    with eager():
        expected = x / divisor
    with fused():
        actual = x / divisor
    assert_exact_with_zero_sign(actual, expected)


@CUDA
def test_neg_and_scalar_division_backward():
    torch.manual_seed(20)
    x = torch.randn(5, 37, device="cuda", requires_grad=True)
    weight = torch.randn_like(x)
    gradient = torch.randn(5, device="cuda")

    def run():
        y = (-(x / 8.) * weight).sum(-1)
        return y, torch.autograd.grad(y, x, gradient)[0]

    with eager():
        expected, expected_grad = run()
    with fused():
        actual, actual_grad = run()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("operation", ["add", "eq", "masked_fill", "add_"])
def test_invalid_broadcast_preserves_pending_graph(operation):
    x = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    other = torch.ones(4, 3, device="cuda")
    if operation == "masked_fill":
        other = other.bool()

    def invalid(pending):
        if operation == "masked_fill":
            return pending.masked_fill(other, 0.)
        return getattr(pending, operation)(other)

    with eager():
        expected = x * 2 + 1
        with pytest.raises(RuntimeError):
            invalid(x)
    with fused():
        pending = x * 2
        with pytest.raises(RuntimeError):
            invalid(pending)
        actual = pending + 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
