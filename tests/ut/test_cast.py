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
from dvm.tester import Tester
from tests.mark_utils import arg_mark


pytestmark = pytest.mark.gpu


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type1, type2, eps', [(np.float16, np.float32, 1e-3), (np.float32, np.float16, 1e-3),
                                               (np.float16, np.int32, 0), (np.int32, np.float32, 1e-5),
                                               (np.int32, np.float16, 1e-3)])
def test_cast(type1, type2, eps):
    t = Tester()
    a = np.random.normal(0, 100, [1024, 32]).astype(type1)
    x = t.load(a)
    x = t.copy(x)
    z = t.cast(x, type2.__name__)
    z = t.copy(z)
    t.store_expect(z, a.astype(type2), eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type1, type2, eps', [("bfloat16", "float32", 1e-2), ("float32", "bfloat16", 1e-2)])
def test_cast_bf16(type1, type2, eps):
    t = Tester()
    a = np.random.normal(0, 100, [1024, 32]).astype(np.float32)
    x = t.load(a, type1)
    x = t.copy(x)
    z = t.cast(x, type2)
    z = t.copy(z)
    t.store_expect(z, a, eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type1, type2, eps', [(np.float16, np.float32, 1e-3), (np.float32, np.float16, 1e-3)])
def test_cast_binary(type1, type2, eps):
    t = Tester()
    a = np.full([32, 32], 0.5, type1)
    b = np.full([32, 32], 1.5, type2)
    x = t.load(a)
    y = t.load(b)
    z = t.cast(x, type2.__name__)
    o = t.add(z, y)
    t.store_expect(o, 2.0, eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type1, type2, eps', [(np.float16, np.float32, 1e-3), (np.float32, np.float16, 1e-3)])
def test_cast_unary(type1, type2, eps):
    t = Tester()
    a = np.full([32, 32], 1.44, type1)
    x = t.load(a)
    z = t.cast(x, type2.__name__)
    o = t.sqrt(z)
    t.store_expect(o, 1.2, eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type1, type2, eps', [(np.float16, np.float32, 1e-3), (np.float32, np.float16, 1e-3)])
def test_cast_broadcast(type1, type2, eps):
    t = Tester()
    a = np.random.rand(32, 1).astype(type1)
    x = t.load(a)
    z = t.cast(x, type2.__name__)
    o = t.broadcast(z, (32, 128))
    t.store_expect(o, np.broadcast_to(a.astype(type2), (32, 128)), eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [(np.float32), (np.float16)])
def test_cast_bool_to_fp(type):
    t = Tester()
    a = np.random.choice([True, False], (1024, 32))
    x = t.load(a)
    y = t.cast(x, type.__name__)
    t.store_expect(y, a.astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [(np.float32), (np.float16)])
def test_cast_fp_to_bool(type):
    t = Tester()
    a = np.random.normal(0, 1, [1024, 32]).astype(type)
    x = t.load(a)
    y = t.cast(x, "bool")
    t.store_expect(y, a.astype(np.bool_), 0)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cast_multi():
    t = Tester()
    a = np.random.rand(32, 32).astype(np.float16)
    x = t.load(a)
    y = t.cast(x, "float32")
    z = t.sqrt(y)
    o = t.cast(z, "float16")
    t.store_expect(o, np.sqrt(a.astype(np.float32)).astype(np.float16), 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cast_block1_align():
    np.random.seed(1)
    t = Tester()
    a0 = np.random.normal(0, 1, [84]).astype(np.float32)
    x0 = t.load(a0)
    y0 = t.cast(x0, "float16")
    expect = a0.astype(np.float16)
    t.store_expect(y0, expect, eps=1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cast_bool_select():
    t = Tester()
    a = np.array([0.23, -0, 0.6, -1.2]).astype(np.float32)
    b = np.full([4], 1.0).astype(np.float16)
    c = np.full([4], 2.0).astype(np.float16)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.load(c)
    y = t.cast(x1, "bool")
    z = t.select(y, x2, x3)
    t.store_expect(z, np.where(a, b, c).astype(np.float16))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [(np.float32), (np.float16)])
def test_cast_fp_to_bool_fp(type):
    t = Tester()
    a = np.random.normal(0, 1, [1024, 32]).astype(type)
    a[0] = np.nan
    a[1] = -0.0
    a[2] = 0.0
    a[3] = np.inf
    x = t.load(a)
    y = t.cast(x, "bool")
    y = t.cast(y, type.__name__)
    t.store_expect(y, a.astype(np.bool_).astype(type), 0)
    assert (t.run_check())
