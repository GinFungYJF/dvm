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
"""Boolean predicate consumers, saved masks, and graph lifetime boundaries."""

import pytest
import torch

import dvm_torch
from conftest import CUDA, eager, fused


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0, 1])
@pytest.mark.parametrize("save_mask", [False, True])
def test_attention_predicate_preserves_live_outputs(dtype, scalar, save_mask):
    x = torch.randn(2, 3, 17, 17, device="cuda", dtype=dtype)
    keep = torch.rand(2, 1, 17, 17, device="cuda") > .3

    def run():
        scaled = x / 8
        mask = keep == scalar
        out = scaled.masked_fill(mask, -10000)
        return (out, mask) if save_mask else (out,)

    with eager():
        expected = run()
    with fused():
        dvm_torch.reset_stats()
        actual = run()
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert dvm_torch.stats()["graph_sizes"] == {3: 1}
    assert dvm_torch.stats()["stores_skipped"] == (1 if save_mask else 2)


@CUDA
@pytest.mark.parametrize("source_dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("scalars", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nested_bool_predicates_preserve_intermediate_masks(source_dtype, scalars):
    values = [0, 1, 0, 1] if source_dtype == torch.bool else [0, 2, -1, float("nan")]
    source = torch.tensor([values], device="cuda", dtype=source_dtype)
    x = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)

    def run():
        first = source == scalars[0]
        second = first == scalars[1]
        return x.masked_fill(second, -7), first, second

    with eager():
        expected = run()
    with fused():
        actual = run()
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("scalar", [False, True, -0.0, 1.0])
@pytest.mark.parametrize("fill", [-0.0, float("inf"), float("nan")])
def test_predicate_select_preserves_special_values_and_zero_sign(scalar, fill):
    x = torch.tensor([float("nan"), float("inf"), -float("inf"),
                      -0., 0., 1., -1., -0.], device="cuda")
    keep = torch.tensor([True, False, True, False, True, False, True, False], device="cuda")
    with eager():
        expected = x.masked_fill(keep == scalar, fill)
    with fused():
        actual = x.masked_fill(keep == scalar, fill)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    torch.testing.assert_close(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@CUDA
@pytest.mark.parametrize("mutation", ["source", "mask", "flush"])
def test_predicate_provenance_expires_at_materialization(mutation):
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    keep = torch.tensor([True, False] * 4, device="cuda")
    replacement = torch.tensor([True] * 4 + [False] * 4, device="cuda")

    def run(source):
        mask = source == 0
        if mutation == "source":
            source.detach().logical_not_()
        elif mutation == "mask":
            mask.detach().copy_(replacement)
        else:
            dvm_torch.flush()
        return x.masked_fill(mask, -3), mask

    source = keep.clone()
    with eager():
        expected = run(keep.clone())
    with fused():
        actual = run(source)
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("alias_kind", ["detach", "view", "offset"])
def test_predicate_aliases_use_the_correct_shape_and_offset(alias_kind):
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    keep = torch.tensor([True, False, False, True, True, False, True, False], device="cuda")
    values = x.reshape(2, 4) if alias_kind == "view" else x[1:] if alias_kind == "offset" else x

    def run():
        mask = keep == 0
        if alias_kind == "detach":
            mask = mask.detach()
        elif alias_kind == "view":
            mask = mask.view(2, 4)
        else:
            mask = mask[1:]
        return values.masked_fill(mask, -5)

    with eager():
        expected = run()
    with fused():
        actual = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_saved_predicate_backward_after_source_mutation():
    x = torch.randn(2, 3, 17, 17, device="cuda", requires_grad=True)
    keep = torch.rand(2, 1, 17, 17, device="cuda") > .3
    upstream = torch.randn_like(x)

    def run(source):
        mask = source == 0
        out = (x / 8).masked_fill(mask, -10000)
        source.logical_not_()
        grad = torch.autograd.grad(out, x, upstream)[0]
        return out, mask, grad

    source = keep.clone()
    with eager():
        expected = run(keep.clone())
    with fused():
        actual = run(source)
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@CUDA
def test_saved_predicate_mutation_still_checks_autograd_version():
    x = torch.randn(2, 3, 17, 17, device="cuda", requires_grad=True)
    keep = torch.rand(2, 1, 17, 17, device="cuda") > .3
    with fused():
        mask = keep == 0
        out = (x / 8).masked_fill(mask, -10000)
        mask.logical_not_()
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            out.sum().backward()


@CUDA
def test_predicate_inplace_fill_preserves_alias_and_version():
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    keep = torch.tensor([True, False] * 4, device="cuda")
    alias = x.detach()
    with eager():
        expected = x.masked_fill(keep == 0, -7)
    with fused():
        mask = keep == 0
        version = x._version
        actual = x.masked_fill_(mask, -7)
        assert actual is x and alias._version == version + 1
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)


@CUDA
def test_predicate_metadata_does_not_leak_to_later_graphs():
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    keep = torch.tensor([True, False] * 4, device="cuda")
    with eager():
        expected = [x.masked_fill(keep == scalar, -9) for scalar in (0, 1)]
    with fused():
        for iteration in range(16):
            scalar = iteration % 2
            actual = x.masked_fill(keep == scalar, -9)
            dvm_torch.flush()
            torch.testing.assert_close(actual, expected[scalar], rtol=0, atol=0)
