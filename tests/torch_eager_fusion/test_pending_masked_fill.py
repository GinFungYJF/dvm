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
"""Safe masked writes to produced values, with old values and aliases still live."""

import pytest
import torch

import torch_dvm
from conftest import CUDA, eager, fused, run_eager_and_fused


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("retain_intermediates", [False, True])
def test_std_gradient_preparation_and_mask_share_a_graph(dtype, retain_intermediates):
    std = torch.tensor([0., .5, 1., 2.], device="cuda", dtype=dtype).repeat(128).reshape(32, 16, 1)
    grad = torch.full_like(std, 3)

    def run():
        denominator = std * 2
        scaled = grad / denominator
        mask = std == 0
        alias = scaled.detach()
        version = scaled._version
        result = scaled.masked_fill_(mask, 0)
        assert result is scaled and alias._version == version + 1
        return (result, alias, denominator, mask) if retain_intermediates else (result, alias)

    actual, expected = run_eager_and_fused(run)
    stats = torch_dvm.stats()
    assert stats["graph_sizes"] == {4: 1}
    assert stats["stores_skipped"] == (0 if retain_intermediates else 2)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
def test_masked_write_preserves_old_consumers_and_updates_detached_alias(dtype, broadcast):
    x = torch.arange(32 * 16, device="cuda", dtype=dtype).reshape(32, 16)
    mask = (x[:1] if broadcast else x) == 2

    def run():
        produced = x * 2
        alias = produced.detach()
        earlier = produced * 3
        alias.masked_fill_(mask, -7)
        later = produced * 5
        return produced, alias, earlier, later

    actual, expected = run_eager_and_fused(run)
    assert torch_dvm.stats()["graph_sizes"] == {4: 1}
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_repeated_masked_writes_keep_each_predicate_value():
    x = torch.arange(32, device="cuda", dtype=torch.float32)

    def run():
        value = x * 2
        old_mask = value == 2
        version = value._version
        value.masked_fill_(old_mask, 5)
        new_mask = value == 5
        value.masked_fill_(new_mask, -3)
        assert value._version == version + 2
        return value, old_mask, new_mask

    actual, expected = run_eager_and_fused(run)
    assert torch_dvm.stats()["graph_sizes"] == {5: 1}
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("first_inplace", [False, True])
def test_masked_write_keeps_boundary_for_any_earlier_memory_load(first_inplace):
    x = torch.arange(32 * 16, device="cuda", dtype=torch.float32).reshape(32, 16)
    mask = x == 2

    def run(value):
        if first_inplace:
            value.mul_(2)
        earlier = value * 3
        value.masked_fill_(mask, -7)
        return earlier, value

    with eager():
        expected = run(x.clone())
    value = x.clone()
    with fused():
        torch_dvm.reset_stats()
        actual = run(value)
    assert torch_dvm.stats()["graph_sizes"] == ({2: 1, 1: 1} if first_inplace else {1: 2})
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_masked_write_preserves_read_mask_view_boundary():
    x = torch.arange(32 * 16, device="cuda", dtype=torch.float32).reshape(32, 1, 16)
    mask = torch.tensor([True, False] * 8, device="cuda").reshape(1, 1, 16)
    view = mask[0]

    def run():
        target = x * 2
        earlier = target.masked_fill(mask, 1)
        target.masked_fill_(view, -3)
        return earlier, target

    actual, expected = run_eager_and_fused(run)
    assert torch_dvm.stats()["graph_sizes"] == {2: 1, 1: 1}
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@CUDA
def test_masked_write_to_view_of_pending_inplace_result_flushes():
    x = torch.arange(32 * 16, device="cuda", dtype=torch.float32).reshape(32, 16)
    view = x[1:]
    mask = torch.ones_like(view, dtype=torch.bool)
    with eager():
        expected = x * 2
        expected[1:].masked_fill_(mask, -3)
    with fused():
        torch_dvm.reset_stats()
        x.mul_(2)
        view.masked_fill_(mask, -3)
    assert torch_dvm.stats()["graph_sizes"] == {1: 2}
    assert torch_dvm.stats()["by_reason"]["WriteAlias"] >= 1
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("retain_alias", [False, True])
def test_masked_write_dse_preserves_live_alias(retain_alias):
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    mask = x == 2
    expected = (x * 2).masked_fill(mask, -3)
    with fused():
        torch_dvm.reset_stats()
        target = x * 2
        if retain_alias:
            alias = target.detach()
        target.masked_fill_(mask, -3)
        del target
    if retain_alias:
        assert torch_dvm.stats()["graph_sizes"] == {2: 1}
        torch.testing.assert_close(alias, expected, rtol=0, atol=0)
    else:
        assert torch_dvm.stats()["flushes"] == 0
        assert torch_dvm.stats()["stores_skipped"] == 1


@CUDA
def test_masked_write_after_trim_uses_materialized_value():
    x = torch.ones(32, device="cuda")
    mask = torch.arange(32, device="cuda") < 8
    with fused():
        torch_dvm.reset_stats()
        value = x
        for _ in range(62):
            value = value + 1
        value.masked_fill_(mask, -3)
    assert torch_dvm.stats()["graph_sizes"] == {62: 1, 1: 1}
    assert torch_dvm.stats()["by_reason"]["GraphTooLarge"] == 1
    torch.testing.assert_close(value, torch.full_like(x, 63).masked_fill(mask, -3), rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("failure", ["overflow", "mask_dtype", "shape"])
def test_failed_pending_masked_write_preserves_value_and_graph_recovery(failure):
    x = torch.ones(32, 16, device="cuda", dtype=torch.float16)
    mask = torch.ones((2, 32, 16) if failure == "shape" else x.shape,
                      device="cuda", dtype=torch.int32 if failure == "mask_dtype" else torch.bool)
    with fused():
        value = x * 2
        with pytest.raises(RuntimeError):
            value.masked_fill_(mask, 1e9 if failure == "overflow" else 0)
        later = value * 3
    torch.testing.assert_close(value, torch.full_like(x, 2), rtol=0, atol=0)
    torch.testing.assert_close(later, torch.full_like(x, 6), rtol=0, atol=0)


@CUDA
def test_pending_masked_write_autograd_and_saved_version():
    base = torch.randn(32, 16, device="cuda")
    mask = base == 0
    mask[0] = True
    upstream = torch.randn_like(base)

    def run(x):
        value = x * x
        value.masked_fill_(mask, -3)
        return value, torch.autograd.grad(value, x, upstream)[0]

    with eager():
        expected = run(base.clone().requires_grad_())
    x = base.clone().requires_grad_()
    with fused():
        actual = run(x)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)

    x = base.clone().requires_grad_()
    with fused():
        torch_dvm.reset_stats()
        value = x * 2
        saved = (value * value).sum()
        value.masked_fill_(mask, -3)
    assert torch_dvm.stats()["graph_sizes"] == {4: 1}
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()


@CUDA
def test_std_backward_with_zero_variance_and_nan_rows():
    base = torch.randn(32, 128, 768, device="cuda")
    base[0, 0] = 1
    base[0, 1] = float("nan")
    upstream = torch.randn(32, 128, 1, device="cuda")

    def run(x):
        result = x.std(-1, keepdim=True)
        return torch.autograd.grad(result, x, upstream)[0]

    with eager():
        expected = run(base.clone().requires_grad_())
    x = base.clone().requires_grad_()
    with fused():
        torch_dvm.reset_stats()
        actual = run(x)
    assert torch_dvm.stats()["graph_sizes"] == {4: 1, 3: 1}
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, equal_nan=True)
    assert torch.count_nonzero(actual[0, 0]) == 0
    assert torch.isnan(actual[0, 1]).all()
