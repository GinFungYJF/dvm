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
"""Producer admission preserves pending dependencies and native boundaries."""

import pytest
import torch

import torch_dvm
from conftest import CUDA, FLOAT_DTYPES, eager, fused, pairwise_cases, run_eager_and_fused


@CUDA
@pytest.mark.parametrize(
    "dtype,shape,inplace", pairwise_cases(FLOAT_DTYPES, ((), (8, 17)), (False, True))
)
def test_tensor_add_without_producer_is_native(dtype, shape, inplace):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    other = torch.randn(shape[-1:] if shape else (), device="cuda", dtype=dtype)
    alias = x.detach()
    version = x._version
    with eager():
        expected = x + other
    with fused():
        torch_dvm.reset_stats()
        actual = x.add_(other) if inplace else x + other
    assert torch_dvm.stats()["emitted"] == 0
    assert torch_dvm.stats()["flushes"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if inplace:
        assert actual is x
        assert alias._version == version + 1
        torch.testing.assert_close(alias, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("producer", ["left", "right", "both"])
def test_tensor_add_consumes_pending_broadcast_inputs(dtype, producer):
    x = torch.randn(8, 17, device="cuda", dtype=dtype)
    bias = torch.randn(17, device="cuda", dtype=dtype)

    def run():
        left = x * 2 if producer in ("left", "both") else x
        right = bias * 2 if producer in ("right", "both") else bias
        return left + right

    actual, expected = run_eager_and_fused(run)
    assert torch_dvm.stats()["graph_sizes"] == {3 if producer == "both" else 2: 1}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("cached_input", [False, True])
def test_tensor_add_does_not_consume_a_load_as_a_producer(cached_input):
    x, y, z = [torch.randn(8, 17, device="cuda") for _ in range(3)]
    source = x if cached_input else z
    with eager():
        expected = x + y
        product_expected = source * 2
    with fused():
        torch_dvm.reset_stats()
        product = source * 2
        actual = x + y
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    torch.testing.assert_close(product, product_expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_native_add_followed_by_mean_remains_native():
    x, y = [torch.randn(8, 17, device="cuda") for _ in range(2)]
    actual, expected = run_eager_and_fused(lambda: (x + y).mean(-1))
    assert torch_dvm.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_tensor_add_rechecks_producer_after_graph_trim():
    x = torch.ones(8, 17, device="cuda")
    y = torch.full_like(x, 2)
    with fused():
        torch_dvm.reset_stats()
        pending = x
        for _ in range(62):
            pending = pending + 1
        actual = pending + y
    assert torch_dvm.stats()["emitted"] == 62
    assert torch_dvm.stats()["by_reason"]["GraphTooLarge"] == 1
    torch.testing.assert_close(actual, torch.full_like(x, 65), rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("operation", ["add", "mean"])
def test_producer_alias_boundary_before_admission(operation):
    x = torch.arange(136, device="cuda", dtype=torch.float32).reshape(8, 17)
    view = x[1:]
    other = torch.ones_like(view)
    with eager():
        expected = view * 2 + other if operation == "add" else (view * 2).mean(-1)
    with fused():
        torch_dvm.reset_stats()
        x.mul_(2)
        actual = view + other if operation == "add" else view.mean(-1)
    assert torch_dvm.stats()["emitted"] == 1
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_detached_producer_still_feeds_tensor_add_and_backward():
    base = torch.randn(8, 17, device="cuda")
    bias = torch.randn(17, device="cuda")
    upstream = torch.randn_like(base)

    def run(x, b):
        pending = x * 2
        result = pending.detach() + b
        grad = torch.autograd.grad(result, b, upstream)[0]
        return result, grad

    with eager():
        expected = run(base.clone().requires_grad_(), bias.clone().requires_grad_())
    x, b = base.clone().requires_grad_(), bias.clone().requires_grad_()
    with fused():
        actual = run(x, b)
    assert x.grad is None
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@CUDA
def test_std_backward_splits_at_native_mean_and_preserves_inplace_output():
    x = torch.randn(4, 17, 768, device="cuda")
    grad = torch.randn(4, 17, 1, device="cuda")
    mask = torch.rand_like(grad) < .25

    def run(value):
        value.masked_fill_(mask, 0)
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        factor = value * (2. / 767.)
        return factor * centered

    with eager():
        expected_value = grad.clone()
        expected = run(expected_value)
    value = grad.clone()
    alias = value.detach()
    version = value._version
    with fused():
        torch_dvm.reset_stats()
        actual = run(value)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1, 3: 1}
    assert alias._version == version + 1
    torch.testing.assert_close(alias, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@CUDA
@pytest.mark.parametrize(
    "dtype,broadcast,retain_rhs", pairwise_cases(FLOAT_DTYPES, (False, True), (False, True))
)
def test_add_inplace_consumes_pending_read_operand(dtype, broadcast, retain_rhs):
    target = torch.randn(8, 17, device="cuda", dtype=dtype)
    source = torch.randn((17,) if broadcast else (8, 17), device="cuda", dtype=dtype)
    alias = target.detach()
    version = target._version
    with eager():
        expected_rhs = source * 2
        expected = target + expected_rhs
    with fused():
        torch_dvm.reset_stats()
        rhs = source * 2
        actual = target.add_(rhs)
        if not retain_rhs:
            del rhs
    stats = torch_dvm.stats()
    assert stats["graph_sizes"] == {2: 1}
    assert stats["stores_skipped"] == (0 if retain_rhs else 1)
    assert actual is target
    assert alias._version == version + 1
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)
    if retain_rhs:
        torch.testing.assert_close(rhs, expected_rhs, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("destination", ["loaded", "pending", "same_as_rhs"])
def test_add_inplace_keeps_write_boundary_for_graph_storage(destination):
    x, y = [torch.randn(8, 17, device="cuda") for _ in range(2)]

    def run(value):
        if destination == "loaded":
            rhs = value * 2
            target = value
        else:
            target = value * 2
            rhs = target if destination == "same_as_rhs" else y * 2
        target.add_(rhs)
        return target, rhs

    with eager():
        expected = run(x.clone())
    value = x.clone()
    with fused():
        torch_dvm.reset_stats()
        actual = run(value)
    assert torch_dvm.stats()["graph_sizes"] == {2 if destination == "pending" else 1: 1}
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_add_inplace_pending_read_view_still_flushes():
    source = torch.randn(8, 17, device="cuda")
    view = source[1:]
    target = torch.randn_like(view)
    with eager():
        expected = target + view * 2
    with fused():
        torch_dvm.reset_stats()
        source.mul_(2)
        target.add_(view)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("fallback", ["dtype", "stride", "alpha"])
def test_add_inplace_materializes_read_operand_when_emission_is_rejected(fallback):
    target = torch.randn(8, 34 if fallback == "stride" else 17, device="cuda")
    if fallback == "stride":
        target = target[:, ::2]
    source = torch.randn(8, 17, device="cuda",
                         dtype=torch.float16 if fallback == "dtype" else torch.float32)
    alpha = 2 if fallback == "alpha" else 1
    with eager():
        expected = target.clone().add_(source * 2, alpha=alpha)
    with fused():
        torch_dvm.reset_stats()
        rhs = source * 2
        target.add_(rhs, alpha=alpha)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@CUDA
def test_add_inplace_rechecks_read_producer_after_trim():
    source = torch.ones(8, 17, device="cuda")
    target = torch.full_like(source, 2)
    with fused():
        torch_dvm.reset_stats()
        rhs = source
        for _ in range(62):
            rhs = rhs + 1
        target.add_(rhs)
    assert torch_dvm.stats()["emitted"] == 62
    assert torch_dvm.stats()["by_reason"]["GraphTooLarge"] == 1
    torch.testing.assert_close(target, torch.full_like(target, 65), rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("mutate", ["source", "rhs", "target"])
def test_add_inplace_preserves_reads_before_later_writes(mutate):
    x, y = [torch.randn(8, 17, device="cuda") for _ in range(2)]

    def run(source, target):
        rhs = source * 2
        target.add_(rhs)
        {"source": source, "rhs": rhs, "target": target}[mutate].mul_(3)
        return source, rhs, target

    with eager():
        expected = run(x.clone(), y.clone())
    source, target = x.clone(), y.clone()
    with fused():
        torch_dvm.reset_stats()
        actual = run(source, target)
    assert torch_dvm.stats()["graph_sizes"] == {2: 1, 1: 1}
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_add_inplace_pending_read_preserves_autograd():
    left, right, upstream = [torch.randn(8, 17, device="cuda") for _ in range(3)]

    def run(a, b):
        target = a.clone()
        rhs = b * 2
        target.add_(rhs)
        gradients = torch.autograd.grad(target, (a, b), upstream)
        return target, *gradients

    with eager():
        expected = run(left.clone().requires_grad_(), right.clone().requires_grad_())
    a, b = left.clone().requires_grad_(), right.clone().requires_grad_()
    with fused():
        actual = run(a, b)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_add_inplace_still_invalidates_saved_autograd_value():
    leaf = torch.randn(8, 17, device="cuda", requires_grad=True)
    target = leaf.clone()
    saved = (target * target).sum()
    source = torch.randn_like(target)
    with fused():
        target.add_(source * 2)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()
