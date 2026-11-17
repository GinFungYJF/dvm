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
"""Escape tests: a Pending tensor reaching an op the graph did not build.

These are the failures that matter most. A missed flush produces wrong numbers
with no error, so every case here compares against eager rather than checking
for an exception.
"""

import torch

from conftest import CUDA, assert_same, eager, fused


@CUDA
def test_escape_to_matmul():
    x = torch.randn(64, 64, device="cuda")
    w = torch.randn(64, 64, device="cuda")
    with eager():
        expected = torch.nn.functional.linear(x * 2 + 1, w)
    with fused():
        actual = torch.nn.functional.linear(x * 2 + 1, w)
    assert_same(actual, expected)


@CUDA
def test_escape_to_softmax():
    x = torch.randn(32, 128, device="cuda")
    with eager():
        expected = torch.softmax(x * 3, dim=-1)
    with fused():
        actual = torch.softmax(x * 3, dim=-1)
    assert_same(actual, expected)


@CUDA
def test_escape_to_unsupported_reduction():
    x = torch.randn(4096, device="cuda")
    with eager():
        expected = (x * 3).amax()
    with fused():
        actual = (x * 3).amax()
    assert_same(actual, expected)


@CUDA
def test_host_read_item():
    """`.item()` reads on the host; an unflushed graph returns stale memory."""
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    with eager():
        expected = (x * 3).sum().item()
    with fused():
        actual = (x * 3).sum().item()
    assert actual == expected


@CUDA
def test_host_read_cpu_copy():
    x = torch.randn(1024, device="cuda")
    with eager():
        expected = (x * 3).cpu()
    with fused():
        actual = (x * 3).cpu()
    assert_same(actual, expected)


@CUDA
def test_escape_to_conv():
    x = torch.randn(1, 3, 16, 16, device="cuda")
    w = torch.randn(4, 3, 3, 3, device="cuda")
    with eager():
        expected = torch.nn.functional.conv2d(x * 2, w)
    with fused():
        actual = torch.nn.functional.conv2d(x * 2, w)
    assert_same(actual, expected)


@CUDA
def test_escape_after_chain_resumes():
    """A flush must not break the ability to start a new graph immediately."""
    x = torch.randn(128, device="cuda")
    w = torch.randn(128, 128, device="cuda")
    with eager():
        expected = torch.nn.functional.linear(x * 2, w) + 1
    with fused():
        actual = torch.nn.functional.linear(x * 2, w) + 1
    assert_same(actual, expected)


@CUDA
def test_backward_through_fused_region():
    x = torch.randn(256, device="cuda", requires_grad=True)
    w = torch.randn(256, device="cuda", requires_grad=True)
    with eager():
        (x * w).sum().backward()
        expected = (x.grad.clone(), w.grad.clone())
        x.grad = None
        w.grad = None
    with fused():
        (x * w).sum().backward()
    assert_same(x.grad, expected[0])
    assert_same(w.grad, expected[1])
