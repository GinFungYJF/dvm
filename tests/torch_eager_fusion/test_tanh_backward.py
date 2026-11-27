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
"""Capture and numerical semantics of the functional CUDA tanh backward."""

import pytest
import torch

import torch_dvm
from conftest import CUDA, fused, run_eager_and_fused


pytestmark = CUDA
tanh_backward = torch.ops.aten.tanh_backward.default


def test_tanh_backward_matches_cuda_rounding_and_special_values():
    saved = torch.tensor(
        [-float("inf"), -2., -1., -.9999, -.75, -0., 0., .75, .9999, 1., 2.,
         float("inf"), float("nan")], device="cuda", dtype=torch.float32
    )
    grad = saved.repeat_interleave(saved.numel())
    saved = saved.repeat(saved.numel())
    actual, expected = run_eager_and_fused(lambda: tanh_backward(grad * 1., saved))
    assert torch_dvm.stats()["graph_sizes"] == {2: 1}
    valid = ~expected.isnan()
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual.view(torch.int32)[valid], expected.view(torch.int32)[valid])


def test_explicit_mul_and_sub_keep_separate_rounding():
    saved = torch.tensor([0.9, -0.9, 0.9999, -0.9999], device="cuda", dtype=torch.float32)
    ones = torch.ones_like(saved)

    def run():
        explicit = ones - saved * saved
        backward = tanh_backward(ones * 1., saved)
        return explicit, backward

    actual, expected = run_eager_and_fused(run)
    assert torch.all(expected[0] != expected[1])
    for got, want in zip(actual, expected):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("pending_saved", [False, True])
def test_tanh_backward_requires_pending_gradient(pending_saved):
    grad = torch.randn(5, 17, device="cuda")
    saved = torch.randn_like(grad).tanh()
    actual, expected = run_eager_and_fused(lambda: tanh_backward(grad, saved * .5 if pending_saved else saved))
    assert torch_dvm.stats()["graph_sizes"] == ({1: 1} if pending_saved else {})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["half", "bfloat16", "double", "mixed", "broadcast", "strided", "empty", "disabled"])
def test_tanh_backward_native_fallback(case):
    dtype = {"half": torch.float16, "bfloat16": torch.bfloat16, "double": torch.float64}.get(case, torch.float32)
    grad = torch.randn(0 if case == "empty" else 8, 17, device="cuda", dtype=dtype)
    saved = torch.randn_like(grad).tanh()
    if case == "mixed":
        saved = saved.half()
    elif case == "broadcast":
        saved = saved[0]
    elif case == "strided":
        saved = torch.randn(17, 8, device="cuda").t()
    options = {"disable_ops": "tanh_backward"} if case == "disabled" else {}
    actual, expected = run_eager_and_fused(lambda: tanh_backward(grad * .5, saved), **options)
    assert 2 not in torch_dvm.stats()["graph_sizes"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("retain", [False, True])
def test_tanh_backward_stores_observable_intermediates(retain):
    grad = torch.randn(3, 7, 11, device="cuda")
    saved = torch.randn_like(grad).tanh()

    def run():
        pending = grad * .5
        derivative = tanh_backward(pending, saved)
        tail = derivative * 2.
        return (pending, derivative, tail) if retain else (tail,)

    actual, expected = run_eager_and_fused(run)
    stats = torch_dvm.stats()
    assert stats["graph_sizes"] == {3: 1}
    assert stats["stores_skipped"] == (0 if retain else 2)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_tanh_backward_out_materializes_pending_grad_and_writes_original_storage():
    grad = torch.randn(5, 17, device="cuda")
    saved = torch.randn_like(grad).tanh()
    buffer = torch.empty(17, 5, device="cuda").t()
    expected = tanh_backward(grad * .5, saved)
    with fused():
        torch_dvm.reset_stats()
        actual = torch.ops.aten.tanh_backward.grad_input(grad * .5, saved, grad_input=buffer)
    assert actual is buffer
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    torch.testing.assert_close(buffer, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_tanh_backward_rejects_mixed_devices_after_materializing_grad():
    grad = torch.randn(17, device="cuda:0")
    saved = torch.randn(17, device="cuda:1").tanh()
    expected_grad = grad * .5
    with pytest.raises(RuntimeError) as native:
        tanh_backward(expected_grad, saved)
    with fused():
        torch_dvm.reset_stats()
        pending = grad * .5
        with pytest.raises(RuntimeError) as captured:
            tanh_backward(pending, saved)
    assert str(captured.value) == str(native.value)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    torch.testing.assert_close(pending, expected_grad, rtol=0, atol=0)


def test_tanh_backward_rechecks_producer_after_graph_trim():
    grad = torch.ones(17, device="cuda")
    saved = torch.full_like(grad, .75)

    def run():
        pending = grad
        for _ in range(62):
            pending = pending + 1
        return tanh_backward(pending, saved)

    actual, expected = run_eager_and_fused(run)
    stats = torch_dvm.stats()
    assert stats["by_reason"]["GraphTooLarge"] == 1
    assert stats["emitted"] == 62
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tanh_backward_higher_order_autograd():
    grad = torch.linspace(-2., 2., 257, device="cuda").requires_grad_()
    x = torch.linspace(2., -2., 257, device="cuda").requires_grad_()

    def run():
        saved = x.tanh()
        out = tanh_backward(grad * .5, saved)
        first = torch.autograd.grad(out.sum(), (grad, x), create_graph=True)
        second = torch.autograd.grad(first[0].sum() + first[1].sum(), (grad, x))
        return (out, *first, *second)

    actual, expected = run_eager_and_fused(run)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
