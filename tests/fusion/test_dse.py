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
"""Dead-store elimination with live aliases and autograd saved tensors."""

import torch

import dvm_torch
from conftest import CUDA, eager, fused


@CUDA
def test_dead_intermediate_store_is_eliminated():
    x, y, z = [torch.randn(1024, device="cuda") for _ in range(3)]
    with eager():
        expected = x * y * z
    with fused():
        dvm_torch.reset_stats()
        temporary = x * y
        actual = temporary * z
        del temporary
    assert dvm_torch.stats()["stores_skipped"] == 1
    torch.testing.assert_close(actual, expected)


@CUDA
def test_graph_with_only_dead_outputs_is_discarded():
    x, y = [torch.randn(1024, device="cuda") for _ in range(2)]
    with eager():
        expected = x * y
    with fused():
        dvm_torch.reset_stats()
        unused = x * y
        del unused
        dvm_torch.flush()
        assert dvm_torch.stats()["stores_skipped"] == 1
        assert dvm_torch.stats()["flushes"] == 0
        actual = x * y
    assert dvm_torch.stats()["flushes"] == 1
    torch.testing.assert_close(actual, expected)


@CUDA
def test_preserves_detached_storage_alias():
    base = torch.randn(1024, device="cuda")
    other = torch.randn_like(base)
    alias = base.detach()
    with eager():
        expected = base * other
    with fused():
        dvm_torch.reset_stats()
        base.mul_(other)
        del base
    assert dvm_torch.stats()["stores_skipped"] == 0
    torch.testing.assert_close(alias, expected)


@CUDA
def test_preserves_autograd_saved_intermediate():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    with eager():
        expected = (x * y).square().sum()
        expected_grads = torch.autograd.grad(expected, (x, y))
    with fused(disable_ops="sum"):
        dvm_torch.reset_stats()
        actual = (x * y).square().sum()
        # The product is saved for square's backward even without a Python owner.
        assert dvm_torch.stats()["stores_skipped"] == 0
        actual_grads = torch.autograd.grad(actual, (x, y))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
