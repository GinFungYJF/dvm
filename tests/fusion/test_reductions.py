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
"""GPU sum/mean capture, reduction dependencies and native fallback semantics."""

import pytest
import torch

import dvm_torch
from conftest import CUDA, eager, fused


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape,dims",
    [
        ((251,), None),
        ((10, 251), (1,)),
        ((64, 1001), (0,)),
        ((3, 4, 5), (-1, 0)),
        ((3, 4, 5), ()),
        ((3, 4, 5), None),
        ((2, 1, 7), (1,)),
        ((), (0,)),
        ((), (-1,)),
        ((), None),
    ],
)
def test_reduction_dimensions(name, keepdim, shape, dims):
    x = torch.randn(shape, device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=dims, keepdim=keepdim)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x * 1, dim=dims, keepdim=keepdim)
    assert dvm_torch.stats()["emitted"] == 2
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32, torch.float16, torch.bfloat16])
def test_reduction_dtype(name, dtype, out_dtype):
    x = torch.randn((8, 257), device="cuda", dtype=dtype)
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=1, dtype=out_dtype)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x * 1, dim=1, dtype=out_dtype)
    assert dvm_torch.stats()["emitted"] == 2
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_explicit_dtype_casts_before_accumulating(name):
    x = torch.tensor([[1.0004, -1.0], [65520.0, -65504.0]], device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=1, dtype=torch.float16)
    with fused():
        actual = fn(x * 1, dim=1, dtype=torch.float16)
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("dims", [None, (0,)])
def test_standalone_reduction_gating(name, dims):
    x = torch.randn((32, 257), device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x) if dims is None else fn(x, dim=dims)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x) if dims is None else fn(x, dim=dims)
    assert dvm_torch.stats()["emitted"] == 0
    assert dvm_torch.stats()["flushes"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("cached_input", [False, True])
def test_reduction_requires_a_producer_not_an_unrelated_pending_op(name, cached_input):
    x = torch.randn((32, 257), device="cuda")
    other = torch.randn_like(x)
    source = x if cached_input else other
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=0)
        product_expected = source * 2
    with fused():
        dvm_torch.reset_stats()
        product = source * 2
        actual = fn(x, dim=0)
    assert dvm_torch.stats()["emitted"] == 1
    torch.testing.assert_close(product, product_expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_after_materialising_producer_uses_native_kernel(name):
    x = torch.randn((32, 257), device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x * 2, dim=0)
    with fused():
        product = x * 2
        dvm_torch.flush()
        dvm_torch.reset_stats()
        actual = fn(product, dim=0)
    assert dvm_torch.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_rechecks_producer_after_graph_trim(name):
    x = torch.ones((8, 33), device="cuda")
    fn = getattr(torch, name)
    with fused():
        dvm_torch.reset_stats()
        pending = x
        for _ in range(62):
            pending = pending + 1
        actual = fn(pending, dim=0)
    assert dvm_torch.stats()["emitted"] == 62
    assert dvm_torch.stats()["by_reason"]["GraphTooLarge"] == 1
    torch.testing.assert_close(actual, fn(x + 62, dim=0), rtol=0, atol=0)


@CUDA
def test_sum_of_inplace_result_fuses():
    x = torch.randn((32, 257), device="cuda")
    with eager():
        expected = (x + 1).sum(dim=0)
    with fused():
        dvm_torch.reset_stats()
        x.add_(1)
        actual = x.sum(dim=0)
    assert dvm_torch.stats()["emitted"] == 2
    assert dvm_torch.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_elementwise_reduction_chain(name):
    x = torch.randn((32, 257), device="cuda")
    fn = getattr(torch, name)

    def chain(t):
        return fn(t * 2 + 1, dim=-1, keepdim=True) + 3

    with eager():
        expected = chain(x)
    with fused():
        dvm_torch.reset_stats()
        actual = chain(x)
    assert dvm_torch.stats()["emitted"] == 4
    assert dvm_torch.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_reduction_keeps_live_intermediate():
    x = torch.randn((32, 257), device="cuda")
    y = torch.randn_like(x)
    with eager():
        product_expected = x * y
        expected = product_expected.sum()
    with fused():
        dvm_torch.reset_stats()
        product = x * y
        actual = product.sum()
    assert dvm_torch.stats()["emitted"] == 2
    assert dvm_torch.stats()["flushes"] == 1
    assert dvm_torch.stats()["stores_skipped"] == 0
    torch.testing.assert_close(product, product_expected)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_reduce_broadcast_and_multiple_reductions():
    x = torch.randn((8, 33, 65), device="cuda")

    def chain(t):
        mean = t.mean(dim=(0, 2), keepdim=True)
        centered = t - mean
        variance = (centered * centered).mean(dim=(0, 2))
        total = centered.sum(dim=1)
        return mean, centered, variance, total

    with eager():
        expected = chain(x)
    with fused():
        dvm_torch.reset_stats()
        actual = chain(x)
    assert dvm_torch.stats()["emitted"] == 4
    assert dvm_torch.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_empty_reduction(name, shape):
    x = torch.empty(shape, device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=0, keepdim=True)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x, dim=0, keepdim=True)
    assert dvm_torch.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected, equal_nan=True)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float64])
def test_unsupported_input_dtype(name, dtype):
    x = torch.ones((3, 5), device="cuda", dtype=dtype)
    kwargs = {"dtype": torch.float32} if name == "mean" and dtype != torch.float64 else {}
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=1, **kwargs)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x, dim=1, **kwargs)
    assert dvm_torch.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_unsupported_output_dtype_flushes_pending(name):
    x = torch.randn((3, 5), device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x * 2, dtype=torch.float64)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x * 2, dtype=torch.float64)
    assert dvm_torch.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_noncontiguous_reduction(name):
    x = torch.randn((7, 9), device="cuda").t()
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=0)
    with fused():
        dvm_torch.reset_stats()
        actual = fn(x, dim=0)
    assert dvm_torch.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("dims", [(0, 0), (0, -2), (2,), (-3,)])
def test_invalid_dimensions_flush_before_native_error(dims):
    x = torch.randn((3, 5), device="cuda")
    with fused():
        pending = x * 2
        with pytest.raises((RuntimeError, IndexError)):
            pending.sum(dim=dims)
        actual = pending.sum(dim=1)
    torch.testing.assert_close(actual, (x * 2).sum(dim=1))


@CUDA
def test_deterministic_reduction_uses_native_kernel():
    x = torch.randn((64, 1001), device="cuda")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with eager():
            expected = x.sum(dim=0)
        with fused():
            dvm_torch.reset_stats()
            actual = (x * 1).sum(dim=0)
        assert dvm_torch.stats()["emitted"] == 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_disable_ops(name):
    x = torch.randn((7, 11), device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x)
    with fused(disable_ops=name):
        dvm_torch.reset_stats()
        actual = fn(x * 1)
    assert dvm_torch.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("rank", [10, 11])
def test_reduction_rank_limit(rank):
    x = torch.ones((2,) * rank, device="cuda")
    with fused():
        dvm_torch.reset_stats()
        actual = (x * 1).sum(dim=-1)
    assert dvm_torch.stats()["emitted"] == (2 if rank == 10 else 0)
    torch.testing.assert_close(actual, x.sum(dim=-1))


@CUDA
def test_many_reduction_branches_trim_and_resume():
    x = torch.ones((8, 33), device="cuda")
    with eager():
        expected = [(x + i).sum(dim=0) * 0.5 for i in range(80)]
    with fused():
        dvm_torch.reset_stats()
        actual = [(x + i).sum(dim=0) * 0.5 for i in range(80)]
    # A trim can materialise a producer before its sum is considered for fusion.
    assert 160 < dvm_torch.stats()["emitted"] <= 240
    assert dvm_torch.stats()["by_reason"]["GraphTooLarge"] > 0
    torch.testing.assert_close(actual, expected)


@CUDA
def test_same_storage_different_shape_reduction():
    x = torch.randn((3, 5), device="cuda")
    view = x.view(5, 3)
    with eager():
        expected = x.sum(dim=0), view.sum(dim=0)
    with fused():
        actual = x.sum(dim=0), view.sum(dim=0)
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_out_and_inplace(name):
    x = torch.randn((3, 5), device="cuda")
    fn = getattr(torch, name)
    out = torch.empty((3,), device="cuda")
    with eager():
        expected = fn(x * 2, dim=1) + 1
    with fused():
        actual = fn(x * 2, dim=1, out=out)
        actual.add_(1)
    assert actual is out
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_nonfinite_values(name):
    x = torch.tensor(
        [[float("inf"), 1], [float("nan"), 2], [float("inf"), float("-inf")]],
        device="cuda",
    )
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=1)
    with fused():
        actual = fn(x * 1, dim=1)
    torch.testing.assert_close(actual, expected, equal_nan=True)
