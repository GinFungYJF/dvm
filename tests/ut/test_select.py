# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================

import pytest
import numpy as np
import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape", [(1024, 32), (13, 131), (16, 11), (3, 3)])
@pytest.mark.parametrize('type', [np.float32, np.float16, np.int32])
@pytest.mark.gpu
def test_select(shape, type):
    t = Tester()
    a = np.random.randint(1024, size=shape).astype(type)
    b = np.random.randint(1024, size=shape).astype(type)
    c = np.random.randint(1024, size=shape).astype(type)
    d = np.random.randint(1024, size=shape).astype(type)
    e = t.load(a)
    f = t.load(b)
    g = t.load(c)
    h = t.load(d)
    i = t.greater(e, f)
    j = t.add(e, f)
    k = t.add(g, h)
    l = t.select(i, j, k)
    t.store_expect(l, np.select([np.greater(a, b), ~np.greater(a, b)], [a + b, c + d]).astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_select_fp32_compare_no_cast():
    t = Tester()
    shape = (128, 777)
    a = np.random.normal(0, 1, shape).astype(np.float32)
    b = np.random.normal(0, 1, shape).astype(np.float32)
    c = np.random.normal(0, 1, shape).astype(np.float32)
    d = np.random.normal(0, 1, shape).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.load(c)
    x3 = t.load(d)
    cond = t.greater(x0, x1)
    out = t.select(cond, x2, x3)
    t.store_expect(out, np.where(a > b, c, d).astype(np.float32))
    t.codegen()
    assert "cast" not in t.das().lower()
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float32, np.float16, np.int32])
@pytest.mark.gpu
def test_select_broadcast(type):
    t = Tester()
    a = np.random.randint(1024, size=(1, 10, 1, 1, 50)).astype(type)
    b = np.random.randint(1024, size=(1, 1, 20, 30, 1)).astype(type)
    c = np.random.randint(1024, size=(7, 1, 1, 1, 50)).astype(type)
    d = np.random.randint(1024, size=(1, 1, 1, 30, 1)).astype(type)
    e = t.load(a)
    f = t.load(b)
    g = t.load(c)
    h = t.load(d)
    i = t.greater(e, f)
    j = t.add(e, f)
    k = t.add(g, h)
    l = t.select(i, j, k)
    t.store_expect(l, np.select([np.greater(a, b), ~np.greater(a, b)], [a + b, c + d]).astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape", [(1024, 32), (13, 131), (16, 11), (3, 3)])
@pytest.mark.parametrize('type, eps', [(np.float32, 1e-5), (np.float16, 1e-3)])
@pytest.mark.gpu
def test_select_bool_input(shape, type, eps):
    t = Tester()
    a = np.random.rand(*shape).astype(type)
    b = np.random.rand(*shape).astype(type)
    c = np.random.choice([True, False], shape).astype(bool)
    x = t.load(c)
    y = t.load(a)
    z = t.load(b)
    z = t.select(x, y, z)
    t.store_expect(z, np.select([c == True, c == False], [a, b]), eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_select_scalar():
    t = Tester()
    a = np.random.randint(1024, size=(1024, 10)).astype(np.float16)
    b = np.random.randint(1024, size=()).astype(np.float16)
    c = np.array(True)
    e = t.load(a)
    f = t.load(b)
    g = t.load(c)
    l = t.select(g, e, f)
    t.store_expect(l, np.select([c, ~c], [a, b]).astype(np.float16))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_select_int64():
    t = Tester()
    shape = (13, 131)
    a = np.random.randint(-(1 << 40), 1 << 40, size=shape, dtype=np.int64)
    b = np.random.randint(-(1 << 40), 1 << 40, size=shape, dtype=np.int64)
    c = np.random.randint(-(1 << 40), 1 << 40, size=shape, dtype=np.int64)
    d = np.random.randint(-(1 << 40), 1 << 40, size=shape, dtype=np.int64)
    e = t.load(a)
    f = t.load(b)
    g = t.load(c)
    h = t.load(d)
    i = t.greater(e, f)
    l = t.select(i, g, h)
    t.store_expect(l, np.where(a > b, c, d).astype(np.int64))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_select_int64_boundary_patterns():
    t = Tester()
    lhs = np.array([
        [11, -(1 << 33), 1 << 45, -(1 << 45)],
        [((1 << 32) | 5), -((1 << 32) | 6), np.iinfo(np.int64).min, np.iinfo(np.int64).max],
        [((1 << 48) | 0x1234), -((1 << 48) | 0x5678), (1 << 40) | 0x55, -((1 << 40) | 0x66)],
    ], dtype=np.int64)
    rhs = np.array([
        [-11, 1 << 33, -(1 << 44), 1 << 44],
        [-((1 << 32) | 15), (1 << 32) | 16, 17, -18],
        [-(1 << 47), 1 << 47, -((1 << 39) | 0x77), (1 << 39) | 0x88],
    ], dtype=np.int64)
    cond = np.random.choice([True, False], lhs.shape).astype(bool)
    x = t.load(cond)
    y = t.load(lhs)
    z = t.load(rhs)
    out = t.select(x, y, z)
    t.store_expect(out, np.where(cond, lhs, rhs).astype(np.int64))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_select_int64_broadcast():
    t = Tester()
    a = np.random.randint(-(1 << 40), 1 << 40, size=(1, 10, 1, 1, 50), dtype=np.int64)
    b = np.random.randint(-(1 << 40), 1 << 40, size=(1, 1, 20, 30, 1), dtype=np.int64)
    c = np.random.randint(-(1 << 40), 1 << 40, size=(7, 1, 1, 1, 50), dtype=np.int64)
    d = np.random.randint(-(1 << 40), 1 << 40, size=(1, 1, 1, 30, 1), dtype=np.int64)
    e = t.load(a)
    f = t.load(b)
    g = t.load(c)
    h = t.load(d)
    i = t.greater(e, f)
    l = t.select(i, g, h)
    t.store_expect(l, np.where(a > b, c, d).astype(np.int64))
    assert (t.run_check())
