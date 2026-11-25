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
"""Compact broadcast reads feeding accumulation and dropout backward."""

import math

import pytest
import torch

from conftest import CUDA, eager, fused, torch_dvm

pytestmark = CUDA


def divide(input, divisor, direct_scalar):
    if direct_scalar:
        return torch.ops.aten.div.Scalar(input, divisor)
    return input / divisor


def chain(expanded, accumulator, mask, divisor, direct_scalar=False, retain=False):
    delta = divide(expanded, divisor, direct_scalar)
    accumulator.add_(delta)
    output = torch.ops.aten.native_dropout_backward(accumulator, mask, 1 / .9)
    return (output, accumulator, delta) if retain else (output,)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6, equal_nan=True)


@pytest.mark.parametrize(
    "shape,offset,direct_scalar,retain",
    [
        ((32, 128, 768), 0, False, False),
        ((17,), 0, False, True),
        ((17,), 0, True, False),
        ((4, 5, 128), 0, True, True),
        ((17,), 7, False, False),
        ((4, 5, 128), 7, False, True),
        ((4, 5, 128), 7, True, False),
        ((32, 128, 768), 7, True, True),
    ],
)
def test_broadcast_div_accumulate_dropout(shape, offset, direct_scalar, retain):
    with eager():
        compact_shape = shape[:-1] + (1,)
        count = math.prod(compact_shape)
        backing = torch.randn(count + offset, device="cuda")
        compact = backing[offset:].view(compact_shape)
        expanded = compact.expand(shape)
        initial = torch.randn(shape, device="cuda")
        mask = torch.rand(shape, device="cuda") > .1
        expected = chain(expanded, initial.clone(), mask, shape[-1], direct_scalar, retain)
        value = initial.clone()
        alias = value.detach()
        version = value._version
    with fused():
        torch_dvm.reset_stats()
        actual = chain(expanded, value, mask, shape[-1], direct_scalar, retain)
    stats = torch_dvm.stats()
    assert stats["graph_sizes"] == {3: 1}
    assert stats["stores_skipped"] == (0 if retain else 1)
    assert value._version == alias._version == version + 1
    close(alias, initial + expanded / shape[-1])
    close(actual, expected)


def test_dead_accumulator_store_is_eliminated():
    with eager():
        expanded = torch.randn(4, 5, 1, device="cuda").expand(4, 5, 128)
        initial = torch.randn(4, 5, 128, device="cuda")
        mask = torch.rand_like(initial) > .1
        expected = chain(expanded, initial.clone(), mask, 128)
        value = initial.clone()
    with fused():
        torch_dvm.reset_stats()
        actual = chain(expanded, value, mask, 128)
        del value
    assert torch_dvm.stats()["graph_sizes"] == {3: 1}
    assert torch_dvm.stats()["stores_skipped"] == 2
    close(actual, expected)


def test_consecutive_dropout_backwards_keep_live_intermediate():
    with eager():
        expanded = torch.randn(4, 5, 1, device="cuda").expand(4, 5, 128)
        initial = torch.randn(4, 5, 128, device="cuda")
        mask1 = torch.rand_like(initial) > .1
        mask2 = torch.rand_like(initial) > .2
        expected1 = chain(expanded, initial.clone(), mask1, 128)[0]
        expected2 = torch.ops.aten.native_dropout_backward(expected1, mask2, .5)
        value = initial.clone()
    with fused():
        torch_dvm.reset_stats()
        first = chain(expanded, value, mask1, 128)[0]
        second = torch.ops.aten.native_dropout_backward(first, mask2, .5)
        del value
    assert torch_dvm.stats()["graph_sizes"] == {4: 1}
    assert torch_dvm.stats()["stores_skipped"] == 2
    close(first, expected1)
    close(second, expected2)


def test_old_accumulator_consumer_is_preserved():
    with eager():
        expanded = torch.randn(4, 5, 1, device="cuda").expand(4, 5, 128)
        initial = torch.randn(4, 5, 128, device="cuda")
        mask = torch.rand_like(initial) > .1
        expected = chain(expanded, initial.clone(), mask, 128)
        value = initial.clone()
    with fused():
        torch_dvm.reset_stats()
        old = value * 2
        actual = chain(expanded, value, mask, 128)
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    close(old, initial * 2)
    close(actual, expected)


@pytest.mark.parametrize("direct_scalar", [False, True])
def test_broadcast_view_of_pending_dense_storage_flushes(direct_scalar):
    with eager():
        initial = torch.randn(2, 3, 17, device="cuda")
        value = initial.clone()
        expanded = value.as_strided((2, 3, 17), (3, 1, 0))
        expected = (initial * 2).as_strided((2, 3, 17), (3, 1, 0)) / 17
    with fused():
        torch_dvm.reset_stats()
        value.mul_(2)
        actual = divide(expanded, 17, direct_scalar)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    close(value, initial * 2)
    close(actual, expected)


@pytest.mark.parametrize("write_first", [False, True])
def test_compact_storage_write_obeys_read_order(write_first):
    with eager():
        initial = torch.randn(4, 5, 1, device="cuda")
        compact = initial.clone()
        expanded = compact.expand(4, 5, 128)
        expected = (initial * (2 if write_first else 1)).expand_as(expanded) / 128
    with fused():
        torch_dvm.reset_stats()
        if write_first:
            compact.mul_(2)
        actual = expanded / 128
        if not write_first:
            compact.mul_(2)
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    close(actual, expected)
    close(compact, initial * 2)


@pytest.mark.parametrize("direct_scalar", [False, True])
@pytest.mark.parametrize("divisor", [-3.0, .25, 7.5, 0.0, float("inf"), float("nan")])
def test_broadcast_divisor_values(direct_scalar, divisor):
    with eager():
        compact = torch.tensor([0.0, -0.0, 1.0, -2.0, float("inf"), float("nan")], device="cuda")
        expanded = compact[:, None].expand(6, 33)
        expected = divide(expanded, divisor, direct_scalar)
    with fused():
        torch_dvm.reset_stats()
        actual = divide(expanded, divisor, direct_scalar)
    assert torch_dvm.stats()["emitted"] == int(math.isfinite(divisor) and divisor != 0)
    close(actual, expected)


@pytest.mark.parametrize("layout", ["prefix_gap", "multiple_expansions", "nonlast_expansion", "rank11"])
def test_unsupported_broadcast_layouts_fall_back(layout):
    with eager():
        if layout == "prefix_gap":
            value = torch.randn(4, 10, 1, device="cuda")[:, ::2].expand(4, 5, 128)
        elif layout == "multiple_expansions":
            value = torch.randn(1, 5, 1, device="cuda").expand(4, 5, 128)
        elif layout == "nonlast_expansion":
            value = torch.randn(4, 1, 128, device="cuda").expand(4, 5, 128)
        else:
            value = torch.randn((1,) * 11, device="cuda").expand((1,) * 10 + (17,))
        expected = value / 7
    with fused():
        torch_dvm.reset_stats()
        actual = value / 7
    assert torch_dvm.stats()["emitted"] == 0
    close(actual, expected)


def test_empty_direct_scalar_div_remains_native():
    with eager():
        value = torch.empty(0, 5, 1, device="cuda").expand(0, 5, 128)
        expected = torch.ops.aten.div.Scalar(value, 7)
    with fused():
        torch_dvm.reset_stats()
        actual = torch.ops.aten.div.Scalar(value, 7)
    assert torch_dvm.stats()["emitted"] == 0
    close(actual, expected)


@pytest.mark.parametrize("fallback", ["half", "bfloat16", "double", "strided_mask", "disabled_div", "disabled_dropout"])
def test_chain_fallbacks(fallback):
    with eager():
        dtype = {"half": torch.float16, "bfloat16": torch.bfloat16, "double": torch.float64}.get(fallback, torch.float32)
        expanded = torch.randn(4, 5, 1, device="cuda", dtype=dtype).expand(4, 5, 128)
        initial = torch.randn(4, 5, 128, device="cuda", dtype=dtype)
        if fallback == "strided_mask":
            mask = (torch.rand(4, 128, 5, device="cuda") > .1).transpose(1, 2)
        else:
            mask = torch.rand(4, 5, 128, device="cuda") > .1
        expected = chain(expanded, initial.clone(), mask, 128, direct_scalar=True)
        value = initial.clone()
    options = {"disable_ops": {"disabled_div": "div", "disabled_dropout": "native_dropout_backward"}[fallback]} if fallback.startswith("disabled_") else {}
    with fused(**options):
        torch_dvm.reset_stats()
        actual = chain(expanded, value, mask, 128, direct_scalar=True)
    assert 3 not in torch_dvm.stats()["graph_sizes"]
    close(actual, expected)


@pytest.mark.parametrize("scale", [1 / .9, 0.0, -2.0, float("inf"), float("nan")])
def test_dropout_special_values(scale):
    with eager():
        grad = torch.randn(2, 3, 17, device="cuda")
        mask = torch.rand_like(grad) > .1
        grad[0, 0, :6] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0], device="cuda")
        mask[0, 0, :6] = False
        expected = torch.ops.aten.native_dropout_backward(grad * 2, mask, scale)
    with fused():
        torch_dvm.reset_stats()
        actual = torch.ops.aten.native_dropout_backward(grad * 2, mask, scale)
    assert torch_dvm.stats()["graph_sizes"] == ({2: 1} if math.isfinite(scale) else {1: 1})
    close(actual, expected)
    zero = expected == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def test_dropout_requires_pending_gradient():
    with eager():
        grad = torch.randn(4, 5, 128, device="cuda")
        mask = torch.rand_like(grad) > .1
        expected = torch.ops.aten.native_dropout_backward(grad, mask, 1 / .9)
    with fused():
        torch_dvm.reset_stats()
        actual = torch.ops.aten.native_dropout_backward(grad, mask, 1 / .9)
    assert torch_dvm.stats()["emitted"] == 0
    close(actual, expected)


def test_autograd_mean_and_residual_gradient():
    with eager():
        initial = torch.randn(4, 5, 128, device="cuda")
        full_grad = torch.randn_like(initial)
        mean_grad = torch.randn(4, 5, 1, device="cuda")
        state = torch.cuda.get_rng_state()

        def backward(x):
            value = torch.nn.functional.dropout(x, .1, training=True)
            outputs = (value.square(), value.mean(-1, keepdim=True))
            return torch.autograd.grad(outputs, x, (full_grad, mean_grad))[0]

        expected = backward(initial.clone().requires_grad_())
        actual_input = initial.clone().requires_grad_()
        torch.cuda.set_rng_state(state)
    with fused():
        actual = backward(actual_input)
    close(actual, expected)
