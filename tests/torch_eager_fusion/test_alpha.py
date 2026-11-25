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
"""Scaled add/sub preserve dtype, alpha validation, and mutation boundaries."""

import pytest
import torch

import dvm_torch
from conftest import CUDA, FLOAT_DTYPES, eager, fused, pairwise_cases, run_eager_and_fused

pytestmark = CUDA


@pytest.mark.parametrize(
    "op,alpha,dtype",
    pairwise_cases(("add", "sub", "add_", "sub_"), (0, -2, 0.125, 0.1), FLOAT_DTYPES),
)
def test_tensor_alpha_broadcast_and_alias(op, dtype, alpha):
    torch.manual_seed(19)
    initial = torch.randn(8, 17, device="cuda", dtype=dtype)
    source = torch.randn(17, device="cuda", dtype=dtype)

    def run(target):
        rhs = source * 2
        return getattr(target, op)(rhs, alpha=alpha), rhs

    with eager():
        expected, expected_rhs = run(initial.clone())
    target = initial.clone()
    alias = target.detach()
    version = target._version
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        actual, rhs = run(target)
    assert dvm_torch.stats()["emitted"] == 2
    torch.testing.assert_close(actual, expected, rtol=1e-5 if dtype == torch.float32 else 1e-2,
                               atol=1e-6 if dtype == torch.float32 else 1e-3)
    torch.testing.assert_close(rhs, expected_rhs, rtol=0, atol=0)
    if op.endswith("_"):
        assert actual is target
        assert alias._version == version + 1
        torch.testing.assert_close(alias, actual, rtol=0, atol=0)
    else:
        torch.testing.assert_close(target, initial, rtol=0, atol=0)
    if op in ("add", "add_", "sub"):
        assert dvm_torch.stats()["graph_sizes"] == {2: 1}


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
@pytest.mark.parametrize("direct_scalar", [False, True])
def test_scalar_alpha_entrypoints(op, direct_scalar):
    initial = torch.arange(17, device="cuda", dtype=torch.float32)

    def run(value):
        if direct_scalar:
            return getattr(torch.ops.aten, op).Scalar(value, 1.25, -2)
        return getattr(value, op)(1.25, alpha=-2)

    with eager():
        expected = run(initial.clone())
    value = initial.clone()
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        actual = run(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 1


@pytest.mark.parametrize(
    "op,alpha,dtype",
    pairwise_cases(("add", "sub", "sub_", "add_"), (1, 1.0003), (torch.float16, torch.bfloat16)),
)
def test_scalar_alpha_preserves_opmath_precision(op, dtype, alpha):
    initial = torch.tensor([-1.0 if op.startswith("add") else 1.0], device="cuda", dtype=dtype)
    call = getattr(torch.ops.aten, op).Scalar
    with eager():
        expected = call(initial.clone(), 1.0003, alpha)
    value = initial.clone()
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        actual = call(value, 1.0003, alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-7)
    assert dvm_torch.stats()["emitted"] == 1


@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar_side", [None, "left", "right"])
def test_alpha_does_not_round_product_to_low_precision(op, dtype, scalar_side):
    left = torch.tensor([-1.0], device="cuda", dtype=dtype)
    right = torch.tensor([1.0], device="cuda", dtype=dtype)
    if op == "sub":
        left.neg_()
    if scalar_side == "left":
        left = left.cpu()
    if scalar_side == "right":
        right = right.cpu()

    def run():
        # A producer admits Tensor add without changing the exact input value.
        rhs = right * 1 if scalar_side is None else right.reshape(())
        lhs = left.reshape(()) if scalar_side == "left" else left
        return getattr(torch.ops.aten, op).Tensor(lhs, rhs, alpha=1.0003)

    actual, expected = run_eager_and_fused(run, compile_fast=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] >= 1
    assert actual.abs().item() > 0


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
@pytest.mark.parametrize("alpha", [0, -3, 16777217, 2147483647, -2147483648])
def test_int32_alpha_stays_integer(op, alpha):
    if op.startswith("sub") and alpha == -2147483648:
        alpha = 2147483648  # Native sub converts -alpha to int32.
    initial = torch.tensor([1, -2, 16777217, 2147483647], device="cuda", dtype=torch.int32)
    source = torch.tensor([1, 2, 3, -1], device="cuda", dtype=torch.int32)

    def run(value):
        return getattr(value, op)(source * source, alpha=alpha)

    with eager():
        expected = run(initial.clone())
    value = initial.clone()
    with fused():
        dvm_torch.reset_stats()
        actual = run(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 2


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
@pytest.mark.parametrize("dtype,alpha", [
    (torch.float32, True), (torch.float32, 1 + 0j), (torch.float32, 1e40),
    (torch.int32, 1.0), (torch.int32, 2147483649),
])
def test_invalid_alpha_uses_native_error_after_flushing(op, dtype, alpha):
    x = torch.ones(17, device="cuda", dtype=dtype)
    y = torch.full_like(x, 2)
    with eager(), pytest.raises(RuntimeError) as error:
        getattr(x.clone(), op)(y, alpha=alpha)
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        pending = y * y
        with pytest.raises(RuntimeError) as actual_error:
            getattr(x, op)(pending, alpha=alpha)
    assert str(actual_error.value) == str(error.value)
    torch.testing.assert_close(pending, torch.full_like(y, 4), rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 1


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
def test_strict_alpha_retains_native_fma(op):
    sign = -1 if op.startswith("add") else 1
    initial = torch.tensor([sign * 3e38], device="cuda")
    source = torch.tensor([3e38], device="cuda")

    def run(target):
        return getattr(target, op)(source * 1, alpha=2)

    with eager():
        expected = run(initial.clone())
    target = initial.clone()
    with fused():
        dvm_torch.reset_stats()
        actual = run(target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert dvm_torch.stats()["emitted"] == 1


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
def test_scaled_add_sub_backward(op):
    torch.manual_seed(23)
    initial = [torch.randn(8, 17, device="cuda"), torch.randn(17, device="cuda")]
    upstream = torch.randn_like(initial[0])

    def run(inputs):
        left, right = inputs
        if op.endswith("_"):
            out = getattr(left.clone(), op)(right * 2, alpha=0.1)
        else:
            out = getattr(left * 2, op)(right, alpha=0.1)
        return out, torch.autograd.grad(out, inputs, upstream)

    with eager():
        expected, expected_grads = run([x.clone().requires_grad_() for x in initial])
    inputs = [x.clone().requires_grad_() for x in initial]
    with fused(compile_fast=True):
        actual, grads = run(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for got, want in zip(grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "dtype,alpha,op",
    pairwise_cases(FLOAT_DTYPES, (0.0, float("inf"), float("nan")), ("add", "sub")),
)
def test_scaled_add_sub_special_values(op, alpha, dtype):
    x = torch.tensor([0., -0., 1., -1., 0., 1.], device="cuda", dtype=dtype)
    y = torch.tensor([float("inf"), float("nan"), -0., 0., 1., 2.], device="cuda", dtype=dtype)

    def run():
        return getattr(x * 1, op)(y, alpha=alpha)

    actual, expected = run_eager_and_fused(run, compile_fast=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert dvm_torch.stats()["emitted"] == 2


@pytest.mark.parametrize("op", ["add", "add_"])
def test_alpha_does_not_bypass_producer_filter(op):
    x, y = [torch.randn(17, device="cuda") for _ in range(2)]
    with eager():
        expected = getattr(x.clone(), op)(y, alpha=2)
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        actual = getattr(x, op)(y, alpha=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 0


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
@pytest.mark.parametrize("fallback", ["dtype", "stride"])
def test_scaled_alpha_fallback_materializes_pending_rhs(op, fallback):
    initial = torch.randn(8, 34, device="cuda")
    target = initial[:, ::2] if fallback == "stride" else initial[:, :17].contiguous()
    source = torch.randn(8, 17, device="cuda",
                         dtype=torch.float16 if fallback == "dtype" else torch.float32)
    with eager():
        expected = getattr(target.clone(), op)(source * 2, alpha=0.1)
    with fused(compile_fast=True):
        dvm_torch.reset_stats()
        pending = source * 2
        actual = getattr(target, op)(pending, alpha=0.1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 1


@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("dtype,scalar", [(torch.int32, 0.5), (torch.float32, 1 + 2j)])
def test_scaled_scalar_preserves_native_promotion(op, dtype, scalar):
    x = torch.arange(17, device="cuda", dtype=dtype)
    call = getattr(torch.ops.aten, op).Scalar
    actual, expected = run_eager_and_fused(call, x, scalar, 2, compile_fast=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype
    assert dvm_torch.stats()["emitted"] == 0


@pytest.mark.parametrize("op", ["add", "sub", "add_", "sub_"])
def test_scaled_scalar_respects_disable_ops(op):
    x = torch.arange(17, device="cuda", dtype=torch.float32)
    call = getattr(torch.ops.aten, op).Scalar
    actual, expected = run_eager_and_fused(
        lambda: call(x * 2, 0.1, 0.2), compile_fast=True, disable_ops=op
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["emitted"] == 1
