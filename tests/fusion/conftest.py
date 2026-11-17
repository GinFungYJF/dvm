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
"""Shared harness for the eager-fusion tests."""

import contextlib

import pytest
import torch

dvm_torch = pytest.importorskip("dvm_torch")

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@contextlib.contextmanager
def fused(**kwargs):
    """Run a block with fusion on, then materialise and turn it off again."""
    dvm_torch.enable(**kwargs)
    try:
        yield
        dvm_torch.flush()
    finally:
        dvm_torch.disable()


@contextlib.contextmanager
def eager():
    """Run a block with fusion off, as the reference."""
    dvm_torch.disable()
    yield


def assert_same(actual, expected, tol=0.0):
    """Compare against the eager reference. Default requires bit-exactness."""
    torch.cuda.synchronize()
    a = actual.float()
    b = expected.float()
    assert a.shape == b.shape, "shape {} != {}".format(tuple(a.shape), tuple(b.shape))
    diff = (a - b).abs().max().item() if a.numel() else 0.0
    assert diff <= tol, "maxdiff {:.3e} exceeds {:.3e}".format(diff, tol)


def roundtrip(fn, *args, tol=0.0):
    """Run fn eagerly, then fused, and compare. Returns the fused result."""
    with eager():
        expected = fn(*args)
    with fused():
        actual = fn(*args)
    assert_same(actual, expected, tol)
    return actual
