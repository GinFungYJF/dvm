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
"""Boolean intermediates, broadcast masking and their autograd boundaries."""

import pytest
import torch

import torch_dvm
from conftest import CUDA, eager, fused, run_eager_and_fused


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_scale_eq_mask_chain(dtype):
    scores = torch.randn(2, 3, 17, 17, device="cuda", dtype=dtype)
    keep = torch.rand(2, 1, 17, 17, device="cuda") > 0.3
    actual, expected = run_eager_and_fused(lambda: (scores / 8).masked_fill(keep == 0, -10000))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch_dvm.stats()["graph_sizes"] == {3: 1}


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.bool])
def test_eq_tensor_broadcast(dtype):
    x = torch.tensor([[0], [1], [0]], device="cuda", dtype=dtype)
    y = torch.tensor([[0, 1, 0, 1]], device="cuda", dtype=dtype)
    values = torch.randn(3, 4, device="cuda")
    with eager():
        expected_mask = x == y
        expected = values.masked_fill(expected_mask, 2)
    with fused():
        torch_dvm.reset_stats()
        mask = x == y
        actual = values.masked_fill(mask, 2)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(mask, expected_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch_dvm.stats()["graph_sizes"] == {2: 1}


@CUDA
@pytest.mark.parametrize("scalar", [0, 1, 2, -1, 0.5, float("nan")])
def test_bool_eq_preserves_scalar_promotion(scalar):
    x = torch.tensor([False, True], device="cuda")
    with eager():
        expected = x == scalar
    with fused():
        actual = x == scalar
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [1e9, 1e300, -1e300])
def test_eq_large_float_scalar_narrows_like_native(dtype, scalar):
    x = torch.tensor([0., 1., float("inf"), -float("inf")], device="cuda", dtype=dtype)
    with eager():
        expected = x == scalar
    with fused():
        actual = x == scalar
    torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("fill", [0.0, -0.0, float("inf"), float("nan")])
def test_masked_fill_special_values(fill):
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), -0., 1.], device="cuda")
    mask = torch.tensor([True, False, True, False, True], device="cuda")
    with eager():
        expected = x.masked_fill(mask, fill)
        expected_eq = x == float("inf")
    with fused():
        actual = x.masked_fill(mask, fill)
        actual_eq = x == float("inf")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual_eq, expected_eq)
    torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected))


@CUDA
def test_masked_fill_inplace_preserves_pending_mask_and_alias():
    x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
    with eager():
        expected_before = (x + 1) * 2
        expected = expected_before.masked_fill(expected_before == 2, 9)
    with fused():
        pending = (x + 1) * 2
        alias = pending.detach()
        earlier = pending + 0
        mask = pending == 2
        version = pending._version
        returned = pending.masked_fill_(mask, 9)
        assert returned is pending and alias._version == version + 1
    torch.testing.assert_close(earlier, expected_before)
    torch.testing.assert_close(alias, expected)
    torch.testing.assert_close(mask, expected_before == 2)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_attention_backward(dtype):
    torch.manual_seed(19)
    x = torch.randn(2, 3, 17, 17, device="cuda", dtype=dtype, requires_grad=True)
    keep = torch.rand(2, 1, 17, 17, device="cuda") > .3
    upstream = torch.randn_like(x)

    def run():
        y = (x / 8).masked_fill(keep == 0, -10000).softmax(-1)
        return y, torch.autograd.grad(y, x, upstream)[0]

    with eager():
        expected, expected_grad = run()
    with fused():
        actual, actual_grad = run()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@CUDA
@pytest.mark.parametrize("case", ["int64", "float64", "strided", "empty", "tensor_fill", "mixed_eq"])
def test_masking_native_fallbacks(case):
    x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    if case == "int64": x = x.long()
    if case == "float64": x = x.double()
    if case == "strided": x = x.t()
    if case == "empty": x = x[:0]
    if case == "mixed_eq":
        y = x.double() + 0.5
        fn = lambda: x == y
    else:
        value = torch.tensor(9., device="cuda") if case == "tensor_fill" else 9
        fn = lambda: (x + 1).masked_fill(x == 0, value)
    with eager():
        expected = fn()
    with fused():
        actual = fn()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_masked_fill_broadcast_self_and_inplace_shape_error():
    x = torch.tensor([[1., 2., 3.]], device="cuda")
    mask = torch.tensor([[True, False, True], [False, True, False]], device="cuda")
    with eager():
        expected = x.masked_fill(mask, 9)
    with fused():
        actual = x.masked_fill(mask, 9)
        with pytest.raises(RuntimeError):
            x.masked_fill_(mask, 9)
    torch.testing.assert_close(actual, expected)


@CUDA
def test_masked_fill_overflow_does_not_leave_partial_ir():
    x = torch.randn(32, device="cuda", dtype=torch.float16)
    with fused():
        mask = x == 0
        with pytest.raises(RuntimeError, match="overflow"):
            x.masked_fill(mask, 1e9)
        actual = (x + 1).masked_fill(mask, -3)
    with eager():
        expected = (x + 1).masked_fill(x == 0, -3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
