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

import torch_dvm
from conftest import CUDA, eager, fused, run_eager_and_fused


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize(
    "shape,dims,keepdim_for",
    [
        ((251,), None, "mean"),
        ((10, 251), (1,), "sum"),
        ((64, 1001), (0,), "mean"),
        ((3, 4, 5), (-1, 0), "sum"),
        ((3, 4, 5), (), "mean"),
        ((3, 4, 5), None, "sum"),
        ((2, 1, 7), (1,), "mean"),
        ((), (0,), "sum"),
        ((), (-1,), "mean"),
        ((), None, "sum"),
    ],
)
def test_reduction_dimensions(name, shape, dims, keepdim_for):
    keepdim = name == keepdim_for
    x = torch.randn(shape, device="cuda")
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=dims, keepdim=keepdim)
    with fused():
        torch_dvm.reset_stats()
        actual = fn(x * 1, dim=dims, keepdim=keepdim)
    assert torch_dvm.stats()["emitted"] == 2
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize(
    "name,dtype,out_dtype",
    [
        ("sum", torch.float32, None),
        ("mean", torch.float32, torch.float32),
        ("sum", torch.float32, torch.float16),
        ("mean", torch.float32, torch.bfloat16),
        ("mean", torch.float16, None),
        ("sum", torch.float16, torch.float32),
        ("mean", torch.float16, torch.float16),
        ("sum", torch.float16, torch.bfloat16),
        ("sum", torch.bfloat16, None),
        ("mean", torch.bfloat16, torch.float32),
        ("mean", torch.bfloat16, torch.float16),
        ("sum", torch.bfloat16, torch.bfloat16),
    ],
)
def test_reduction_dtype(name, dtype, out_dtype):
    x = torch.randn((8, 257), device="cuda", dtype=dtype)
    fn = getattr(torch, name)
    with eager():
        expected = fn(x, dim=1, dtype=out_dtype)
    with fused():
        torch_dvm.reset_stats()
        actual = fn(x * 1, dim=1, dtype=out_dtype)
    assert torch_dvm.stats()["emitted"] == 2
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
    actual, expected = run_eager_and_fused(lambda: fn(x) if dims is None else fn(x, dim=dims))
    assert torch_dvm.stats()["emitted"] == 0
    assert torch_dvm.stats()["flushes"] == 0
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
        torch_dvm.reset_stats()
        product = source * 2
        actual = fn(x, dim=0)
    assert torch_dvm.stats()["emitted"] == 1
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
        torch_dvm.flush()
        torch_dvm.reset_stats()
        actual = fn(product, dim=0)
    assert torch_dvm.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_reduction_rechecks_producer_after_graph_trim(name):
    x = torch.ones((8, 33), device="cuda")
    fn = getattr(torch, name)
    with fused():
        torch_dvm.reset_stats()
        pending = x
        for _ in range(62):
            pending = pending + 1
        actual = fn(pending, dim=0)
    assert torch_dvm.stats()["emitted"] == 62
    assert torch_dvm.stats()["by_reason"]["GraphTooLarge"] == 1
    torch.testing.assert_close(actual, fn(x + 62, dim=0), rtol=0, atol=0)


@CUDA
def test_sum_of_inplace_result_fuses():
    x = torch.randn((32, 257), device="cuda")
    with eager():
        expected = (x + 1).sum(dim=0)
    with fused():
        torch_dvm.reset_stats()
        x.add_(1)
        actual = x.sum(dim=0)
    assert torch_dvm.stats()["emitted"] == 2
    assert torch_dvm.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_elementwise_reduction_chain(name):
    x = torch.randn((32, 257), device="cuda")
    fn = getattr(torch, name)

    def chain(t):
        return fn(t * 2 + 1, dim=-1, keepdim=True) + 3

    actual, expected = run_eager_and_fused(chain, x)
    assert torch_dvm.stats()["emitted"] == 4
    assert torch_dvm.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_reduction_keeps_live_intermediate():
    x = torch.randn((32, 257), device="cuda")
    y = torch.randn_like(x)
    with eager():
        product_expected = x * y
        expected = product_expected.sum()
    with fused():
        torch_dvm.reset_stats()
        product = x * y
        actual = product.sum()
    assert torch_dvm.stats()["emitted"] == 2
    assert torch_dvm.stats()["flushes"] == 1
    assert torch_dvm.stats()["stores_skipped"] == 0
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

    actual, expected = run_eager_and_fused(chain, x)
    assert torch_dvm.stats()["emitted"] == 4
    assert torch_dvm.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_empty_reduction(name, shape):
    x = torch.empty(shape, device="cuda")
    fn = getattr(torch, name)
    actual, expected = run_eager_and_fused(lambda: fn(x, dim=0, keepdim=True))
    assert torch_dvm.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected, equal_nan=True)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float64])
def test_unsupported_input_dtype(name, dtype):
    x = torch.ones((3, 5), device="cuda", dtype=dtype)
    kwargs = {"dtype": torch.float32} if name == "mean" and dtype != torch.float64 else {}
    fn = getattr(torch, name)
    actual, expected = run_eager_and_fused(lambda: fn(x, dim=1, **kwargs))
    assert torch_dvm.stats()["emitted"] == 0
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_unsupported_output_dtype_flushes_pending(name):
    x = torch.randn((3, 5), device="cuda")
    fn = getattr(torch, name)
    actual, expected = run_eager_and_fused(lambda: fn(x * 2, dtype=torch.float64))
    assert torch_dvm.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("name", ["sum", "mean"])
def test_noncontiguous_reduction(name):
    x = torch.randn((7, 9), device="cuda").t()
    fn = getattr(torch, name)
    actual, expected = run_eager_and_fused(lambda: fn(x, dim=0))
    assert torch_dvm.stats()["emitted"] == 0
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
            torch_dvm.reset_stats()
            actual = (x * 1).sum(dim=0)
        assert torch_dvm.stats()["emitted"] == 1
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
        torch_dvm.reset_stats()
        actual = fn(x * 1)
    assert torch_dvm.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("rank", [10, 11])
def test_reduction_rank_limit(rank):
    x = torch.ones((2,) * rank, device="cuda")
    with fused():
        torch_dvm.reset_stats()
        actual = (x * 1).sum(dim=-1)
    assert torch_dvm.stats()["emitted"] == (2 if rank == 10 else 0)
    torch.testing.assert_close(actual, x.sum(dim=-1))


@CUDA
def test_many_reduction_branches_trim_and_resume():
    x = torch.ones((8, 33), device="cuda")
    actual, expected = run_eager_and_fused(
        lambda: [(x + i).sum(dim=0) * 0.5 for i in range(80)]
    )
    # A trim can materialise a producer before its sum is considered for fusion.
    assert 160 < torch_dvm.stats()["emitted"] <= 240
    assert torch_dvm.stats()["by_reason"]["GraphTooLarge"] > 0
    torch.testing.assert_close(actual, expected)


@CUDA
def test_same_storage_different_shape_reduction():
    x = torch.arange(15, dtype=torch.float32, device="cuda").reshape(3, 5)
    view = x.view(5, 3)
    with eager():
        expected = (x * 2).sum(dim=0), (view * 2).sum(dim=0)
    with fused():
        torch_dvm.reset_stats()
        x.mul_(2)
        base_sum = x.sum(dim=0)
        assert torch_dvm.stats()["flushes"] == 0
        view_sum = view.sum(dim=0)
        assert torch_dvm.stats()["flushes"] == 1
    stats = torch_dvm.stats()
    assert stats["by_reason"]["WriteAlias"] == 1
    assert stats["graph_sizes"] == {2: 1}
    torch.testing.assert_close((base_sum, view_sum), expected, rtol=0, atol=0)


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
