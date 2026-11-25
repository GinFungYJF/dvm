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
"""Autograd correctness across fused graphs and native CUDA kernels."""

import math

import pytest
import torch
import torch.nn.functional as F

import torch_dvm
from conftest import CUDA, eager, fused


@CUDA
@pytest.mark.parametrize("shape,dim", [((4, 128), -1), ((2, 3, 7), 1), ((2, 2049), -1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_backward_with_pending_gradient(shape, dim, dtype):
    """Softmax's native backward must compute its internal product eagerly."""
    torch.manual_seed(123)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(x)
    with eager():
        y = x.softmax(dim)
        expected = torch.autograd.grad(y, x, upstream * 1.25 + 0.5, retain_graph=True)[0]
        expected_tail = expected * 1.5 + 0.25

    with fused():
        torch_dvm.reset_stats()
        actual = torch.autograd.grad(y, x, upstream * 1.25 + 0.5)[0]
        tail = actual * 1.5 + 0.25

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tail, expected_tail)
    # Both outer chains still fuse after returning through nested native calls.
    assert torch_dvm.stats()["emitted"] == 4


@CUDA
@pytest.mark.parametrize("fuse_forward", [False, True])
def test_attention_forward_and_backward(fuse_forward):
    """Check query, key and value gradients with dropout and transposed heads."""
    torch.manual_seed(123)
    inputs = [torch.randn(2, 17, 4, 16, device="cuda") for _ in range(3)]
    upstream = torch.randn(2, 4, 17, 16, device="cuda")
    rng = torch.cuda.get_rng_state()

    def attention(tensors):
        query, key, value = [x.transpose(1, 2) for x in tensors]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        probabilities = F.dropout(scores.softmax(-1), p=0.1, training=True)
        return probabilities @ value

    with eager():
        reference_inputs = [x.clone().requires_grad_() for x in inputs]
        torch.cuda.set_rng_state(rng)
        expected = attention(reference_inputs)
        expected.backward(upstream)

    actual_inputs = [x.clone().requires_grad_() for x in inputs]
    torch.cuda.set_rng_state(rng)
    if fuse_forward:
        with fused():
            actual = attention(actual_inputs)
            actual.backward(upstream)
    else:
        actual = attention(actual_inputs)
        with fused():
            actual.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-4, atol=1e-5)


@CUDA
def test_fusion_resumes_after_native_exception():
    x = torch.randn(4, 8, device="cuda")
    with eager():
        expected = x * 2 + 1
    with fused():
        with pytest.raises((IndexError, RuntimeError)):
            torch.softmax(x, dim=5)
        torch_dvm.reset_stats()
        actual = x * 2 + 1
    torch.testing.assert_close(actual, expected)
    assert torch_dvm.stats()["emitted"] == 2
