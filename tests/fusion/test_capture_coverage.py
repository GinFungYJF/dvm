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
"""Overload coverage and metadata aliases across deferred computation."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import dvm_torch
from conftest import CUDA, eager, fused


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mul_scalar_overload_joins_chain(dtype):
    x = torch.randn(8, 17, device="cuda", dtype=dtype)
    with eager():
        expected = torch.ops.aten.mul.Scalar(x + 1, 2.0) - 3
    with fused():
        dvm_torch.reset_stats()
        actual = torch.ops.aten.mul.Scalar(x + 1, 2.0) - 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dvm_torch.stats()["graph_sizes"] == {3: 1}


@CUDA
@pytest.mark.parametrize("dtype,scalar", [(torch.int32, 0.5), (torch.float32, 2 + 3j)])
def test_mul_scalar_overload_preserves_native_promotion(dtype, scalar):
    x = torch.arange(17, device="cuda", dtype=dtype)
    with eager():
        expected = torch.ops.aten.mul.Scalar(x, scalar)
    with fused():
        actual = torch.ops.aten.mul.Scalar(x, scalar)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype


@CUDA
def test_mul_scalar_overflow_uses_native_floating_semantics():
    x = torch.tensor([0., 1., -1., float("inf")], device="cuda")
    with eager():
        expected = torch.ops.aten.mul.Scalar(x, 1e300)
    with fused():
        actual = torch.ops.aten.mul.Scalar(x, 1e300)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@CUDA
def test_scalar_and_detach_opt_out_preserves_boundary():
    x = torch.randn(17, device="cuda")
    with fused(disable_ops="mul,detach"):
        dvm_torch.reset_stats()
        y = (x + 1).detach()
        assert dvm_torch.stats()["flushes"] == 1
        y = torch.ops.aten.mul.Scalar(y, 2)
        actual = y - 3
    assert dvm_torch.stats()["emitted"] == 2
    with eager():
        torch.testing.assert_close(actual, (x + 1) * 2 - 3)


@CUDA
def test_detach_keeps_only_surviving_alias_and_chain():
    x = torch.randn(1024, device="cuda")
    with eager():
        expected_alias = x + 1
        expected = expected_alias * 2 - 3
    with fused():
        dvm_torch.reset_stats()
        pending = x + 1
        alias = pending.detach()
        del pending
        assert dvm_torch.stats()["flushes"] == 0
        actual = alias * 2 - 3
    assert dvm_torch.stats()["graph_sizes"] == {3: 1}
    torch.testing.assert_close(alias, expected_alias)
    torch.testing.assert_close(actual, expected)


@CUDA
def test_detach_stops_gradient_without_flushing():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    with eager():
        y = x * 2
        expected = torch.autograd.grad((y * y.detach()).sum(), x)[0]
    with fused():
        dvm_torch.reset_stats()
        y = x * 2
        alias = y.detach()
        assert not alias.requires_grad and alias.grad_fn is None
        assert dvm_torch.stats()["flushes"] == 0
        actual = torch.autograd.grad((y * alias).sum(), x)[0]
    torch.testing.assert_close(actual, expected)


@CUDA
def test_detached_alias_mutation_resolves_pending_reads():
    x = torch.randn(1024, device="cuda")
    with eager():
        expected_original = x * 2
        expected_read = expected_original * 3
        expected_original.add_(1)
    with fused():
        pending = x * 2
        alias = pending.detach()
        actual_read = pending * 3
        version = pending._version
        alias.add_(1)
        assert pending._version == version + 1 == alias._version
    torch.testing.assert_close(actual_read, expected_read)
    torch.testing.assert_close(pending, expected_original)
    torch.testing.assert_close(alias, expected_original)


@CUDA
def test_detached_alias_preserves_saved_tensor_version_check():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    with fused():
        y = x * 2
        loss = y.square().sum()
        y.detach().add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()


@CUDA
def test_detach_alias_native_and_strided_consumers():
    x = torch.randn(8, 32, device="cuda")
    with eager():
        expected = (x + 1)[:, ::2].softmax(-1)
    with fused():
        pending = x + 1
        alias = pending.detach()
        del pending
        actual = alias[:, ::2].softmax(-1)
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("checkpointed", [False, True])
def test_layernorm_saved_values_and_scalar_backward(checkpointed):
    torch.manual_seed(17)
    base = [torch.randn(4, 17, 32, device="cuda"),
            torch.randn(32, device="cuda"), torch.randn(32, device="cuda")]
    upstream = torch.randn_like(base[0])

    def norm(x, weight, bias):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return weight * (x - mean) / (std + 1e-6) + bias

    def run():
        values = [t.clone().requires_grad_() for t in base]
        result = checkpoint(norm, *values, use_reentrant=False) if checkpointed else norm(*values)
        grads = torch.autograd.grad(result, values, upstream)
        return result, grads

    with eager():
        expected, expected_grads = run()
    with fused():
        actual, actual_grads = run()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)
