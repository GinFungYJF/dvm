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
"""Op equivalence: every fusible op must match eager exactly."""

import pytest
import torch

import torch_dvm
from conftest import CUDA, assert_same, eager, fused, roundtrip, run_eager_and_fused

BASE_SHAPE = (1024,)
EDGE_SHAPES = [(1,), (64, 64), (3, 5, 7), (4097,)]

BINARY = {
    "add": lambda a, b: a * 1 + b,  # Tensor add needs a pending producer.
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}

UNARY = {
    "neg": torch.neg,
    "relu": torch.relu,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "reciprocal": torch.reciprocal,
    "round": torch.round,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "trunc": torch.trunc,
}


@pytest.mark.parametrize(
    "name,shape",
    [(name, BASE_SHAPE) for name in BINARY] + [("mul", shape) for shape in EDGE_SHAPES],
)
@CUDA
def test_binary(shape, name):
    fn = BINARY[name]
    a = torch.randn(*shape, device="cuda")
    b = torch.randn(*shape, device="cuda").abs() + 0.5
    actual, expected = run_eager_and_fused(fn, a, b)
    graph_size = 2 if name == "add" else 1
    assert torch_dvm.stats()["graph_sizes"] == {graph_size: 1}
    assert_same(actual, expected)


@pytest.mark.parametrize(
    "name,shape",
    [(name, BASE_SHAPE) for name in UNARY] + [("neg", shape) for shape in EDGE_SHAPES],
)
@CUDA
def test_unary(shape, name):
    fn = UNARY[name]
    a = torch.randn(*shape, device="cuda").abs() + 0.5
    if name in ("relu", "abs"):
        flat = a.view(-1)
        flat[::3].neg_()
        flat[1::3].zero_()
    actual, expected = run_eager_and_fused(fn, a)
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    assert_same(actual, expected)


@CUDA
def test_tanh_fuses_with_its_producer_and_preserves_fp32_bits():
    edges = torch.tensor([-0.0, 0.0, -1e-40, 1e-40, -1e-8, 1e-8,
                          -float("inf"), float("inf"), float("nan")], device="cuda")
    x = torch.cat((edges, torch.linspace(-12, 12, 4097, device="cuda")))
    actual, expected = run_eager_and_fused(lambda: (x * 1.0).tanh())
    assert torch_dvm.stats()["graph_sizes"] == {2: 1}
    assert torch.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    assert torch.equal(actual.view(torch.int32)[valid], expected.view(torch.int32)[valid])


@CUDA
def test_tanh_does_not_join_a_graph_that_only_reads_its_input():
    x = torch.linspace(-2, 2, 31, device="cuda")
    actual, expected = run_eager_and_fused(lambda: (x * 2, x.tanh()))
    assert torch_dvm.stats()["graph_sizes"] == {1: 1}
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.float64])
@CUDA
def test_tanh_native_dtype_fallback(dtype):
    x = torch.arange(-17, 18, device="cuda", dtype=dtype)
    actual, expected = run_eager_and_fused(lambda: (x * x).tanh())
    assert torch_dvm.stats()["graph_sizes"] == ({} if dtype == torch.float64 else {1: 1})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_tanh_forward_saves_the_value_for_backward():
    x = torch.linspace(-3, 3, 257, device="cuda", requires_grad=True)
    grad = torch.linspace(2, -2, 257, device="cuda")

    def run():
        y = (x * 0.5).tanh()
        dx, = torch.autograd.grad(y, x, grad)
        return y, dx

    actual, expected = run_eager_and_fused(run)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [((4, 1), (1, 5)), ((3,), (2, 4, 3)), ((1,), (1024,)), ((2, 1, 6), (1, 4, 1))],
)
@CUDA
def test_broadcast(shape_a, shape_b):
    a = torch.randn(*shape_a, device="cuda")
    b = torch.randn(*shape_b, device="cuda")
    roundtrip(lambda: a * b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@CUDA
def test_dtype(dtype):
    a = torch.randn(512, device="cuda", dtype=dtype)
    b = torch.randn(512, device="cuda", dtype=dtype)
    roundtrip(lambda: a * b + 1.0, tol=0.0 if dtype is torch.float32 else 1e-2)


@CUDA
def test_int32_tensor_ops():
    a = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    b = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    for fn in (lambda x, y: x + y, lambda x, y: x * y, torch.maximum):
        with eager():
            expected = fn(a, b)
        with fused():
            actual = fn(a, b)
        assert_same(actual, expected)


@pytest.mark.parametrize("name", ["maximum", "minimum"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@CUDA
def test_minimum_maximum_capture_and_nan(name, dtype):
    a = torch.tensor([float("nan"), 1, -2, float("inf")], device="cuda", dtype=dtype)
    b = torch.tensor([1, float("nan"), 3, float("-inf")], device="cuda", dtype=dtype)
    fn = getattr(torch, name)
    actual, expected = run_eager_and_fused(fn, a, b)
    assert torch_dvm.stats()["emitted"] == 1
    torch.testing.assert_close(actual, expected, equal_nan=True)


@CUDA
def test_int_scalar_promotion():
    """Integer true division promotes to float; the fused path must not lie."""
    a = torch.randint(1, 9, (256,), device="cuda", dtype=torch.int32)
    with eager():
        expected = a / 2
    with fused():
        actual = a / 2
    assert actual.dtype == expected.dtype == torch.float32
    assert_same(actual, expected)


@CUDA
def test_long_chain_reduces_to_one_kernel():
    x = torch.randn(65536, device="cuda")
    keep = []  # keep the result alive so the graph is not dead-code eliminated

    def chain(t):
        y = t
        for _ in range(10):
            y = y * 1.0001 + 0.001
        keep.append(y)
        if len(keep) > 2:
            keep.pop(0)
        return y

    with eager():
        expected = chain(x)
    with fused():
        actual = chain(x)
    assert_same(actual, expected)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with fused():
            chain(x)
        torch.cuda.synchronize()
    kernels = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1, "expected one fused launch, saw {}".format(len(kernels))
