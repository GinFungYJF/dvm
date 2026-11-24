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
"""In-place, aliasing and view cases.

Each of these silently produces wrong numbers without the matching mechanism:
a deferred store overwriting a user write, a deferred load reading memory that
was changed after the op was recorded, or a view read before its base exists.
"""

import torch

from conftest import CUDA, assert_same, eager, fused, roundtrip


@CUDA
def test_inplace_write_to_pending_output():
    """`t = a*b; t.add_(c)` must keep the add_, not overwrite it."""
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    c = torch.randn(256, device="cuda")
    roundtrip(lambda x, y, z: (x * y).add_(z), a, b, c)


@CUDA
def test_inplace_chain():
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")

    def run(x, y):
        out = x.clone()
        out.add_(y)
        out.relu_()
        return out

    roundtrip(run, a, b)


@CUDA
def test_inplace_scalar_variants():
    a = torch.randn(256, device="cuda")
    roundtrip(lambda x: x.clone().add_(2.0), a)
    roundtrip(lambda x: x.clone().sub_(0.5), a)
    roundtrip(lambda x: x.clone().mul_(3.0), a)


@CUDA
def test_write_to_graph_input():
    """`t = a*b; a.add_(1)` must evaluate a*b with the original a."""
    a = torch.randn(128, device="cuda")
    b = torch.randn(128, device="cuda")

    def run(x, y):
        out = x * y
        x.add_(1.0)
        return out

    # The function mutates its input, so each path needs its own copy.
    with eager():
        expected = run(a.clone(), b)
    with fused():
        actual = run(a.clone(), b)
    assert_same(actual, expected)


@CUDA
def test_alias_write_through_view():
    """Writing through a view of a Pending tensor must land."""
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")

    def run(x, y):
        out = x * y
        out.view(-1).mul_(0)
        return out

    roundtrip(run, a, b)


@CUDA
def test_copy_into_pending():
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    src = torch.randn(256, device="cuda")

    def run(x, y, s):
        out = x * y
        out.copy_(s)
        return out

    roundtrip(run, a, b, src)


@CUDA
def test_view_input_on_pending_base():
    """A view's data_ptr is offset from its base; reading it before the base is
    computed returns uninitialised memory (observed as 0.0)."""
    n = 4096
    t = torch.randn(n, device="cuda")
    b = torch.randn(n // 2, device="cuda")

    def run(base, other):
        pending = base * 2.0
        return pending[n // 2:] * other

    roundtrip(run, t, b)


@CUDA
def test_strided_view_input():
    t = torch.randn(4096, device="cuda")
    b = torch.randn(2048, device="cuda")
    roundtrip(lambda x, y: x[::2] * y, t, b)


@CUDA
def test_resize_after_pending():
    t = torch.randn(64, device="cuda")
    with fused():
        p = t * 2
        p.resize_(8)
    assert p.numel() == 8


@CUDA
def test_partial_overlap_forces_flush():
    """Two half-views of one storage cannot both live in the graph."""
    n = 2048
    t = torch.randn(n, device="cuda")

    def run(x):
        left = x[: n // 2] * 2
        right = x[n // 2 :] * 3
        return left, right

    with eager():
        expected = run(t)
    with fused():
        actual = run(t)
    assert_same(actual[0], expected[0])
    assert_same(actual[1], expected[1])
