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
@pytest.mark.parametrize("shape", [(32, 32), (1024, 32), (1024, 2000)])
@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16, np.bool_])
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
@pytest.mark.gpu
def test_cmp(shape, type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=shape).astype(type)
    b = np.random.randint(1024, size=shape).astype(type)
    if type != np.int32 and type != np.bool_:
        a[0] = np.nan
        b[0] = np.nan
    x = t.load(a)
    y = t.load(b)
    y = t.copy(y)
    x = t.copy(x)
    z = op(t, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16, np.bool_])
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
@pytest.mark.gpu
def test_cmp_s_r(type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=(1024, 32)).astype(type)
    if type != np.int32 and type != np.bool_:
        a[0] = np.nan
    b = True if type == np.bool_ else 30
    x = t.load(a)
    x = t.copy(x)
    z = op(t, x, b)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16])
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
@pytest.mark.gpu
def test_cmp_s_l(type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=(1024, 32)).astype(type)
    b = 30
    x = t.load(a)
    x = t.copy(x)
    z = op(t, b, x)
    z = t.copy(z)
    t.store_expect(z, func(b, a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.not_equal, np.not_equal)])
@pytest.mark.gpu
def test_cmp_over_repeat(op, func):
    t = Tester()
    a = np.random.randint(1024, size=(100000, 1)).astype(np.float16)
    b = np.random.randint(1024, size=(100000, 7)).astype(np.float16)
    x = t.load(a)
    y = t.load(b)
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
@pytest.mark.gpu
def test_cmp_int(op, func):
    t = Tester()
    a = np.random.randint(-2147483648, 2147483647, (1024, 1024)).astype(np.int32)
    b = np.random.randint(-2147483648, 2147483647, (1024, 1024)).astype(np.int32)
    x = t.load(a)
    y = t.load(b)
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_cmp_ws_inplace():
    t = Tester()
    a = np.random.randint(1024, size=(32, 512)).astype(np.float16)
    b = np.random.randint(1024, size=(32, 512)).astype(np.float16)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.equal(x, y)
    t.store_expect(z, np.equal(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_cmp_s_ws_inplace():
    t = Tester()
    a = np.random.randint(1024, size=(32, 512)).astype(np.float16)
    x = t.load(a)
    x = t.copy(x)
    z = t.equal(x, 1.0)
    t.store_expect(z, np.equal(a, 1.0))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support bfloat16 compare op")
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
def test_cmp_bf16(op, func):
    t = Tester()
    a = Tester.bf16_random_normal(-1, 1, [1024, 1]).astype(np.float32)
    b = Tester.bf16_random_normal(-1, 1, [1024, 1]).astype(np.float32)
    x = t.load(a, "bfloat16")
    y = t.load(b, "bfloat16")
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_cmp_repeat_overflow():
    t = Tester()
    a = np.random.normal(-1, 1, [2048, 9]).astype(np.float16)
    x = t.load(a)
    x = t.cast(x, "float32")
    z = t.less_equal(x, 2.0)
    z = t.cast(z, "bool")
    t.store_expect(z, np.less_equal(a, 2))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
def test_cmp_int64(op, func):
    t = Tester()
    a = np.random.randint(0x10000000, 0x2000000000, size=[800, 1000], dtype=np.int64)
    b = np.random.randint(0x10000000, 0x2000000000, size=[800, 1000], dtype=np.int64)
    if op == Tester.equal or op == Tester.not_equal:
        a[10][100] = 1000
        a[10][100] = 1000
        a[100][2] = 1
        a[100][2] = 1
    x = t.load(a)
    y = t.load(b)
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.greater_equal, np.greater_equal)])
@pytest.mark.parametrize("lhs_shape,rhs_shape", [
    ((8, 1, 257, 1), (1, 5, 257, 129)),
    ((1, 9, 1, 257), (6, 9, 33, 1)),
    ((4, 1, 8, 1, 129), (1, 3, 1, 17, 129)),
])
def test_cmp_int64_broadcast(lhs_shape, rhs_shape, op, func):
    t = Tester()
    a = np.random.randint(0x10000000, 0x2000000000, size=lhs_shape, dtype=np.int64)
    b = np.random.randint(0x10000000, 0x2000000000, size=rhs_shape, dtype=np.int64)
    x = t.load(a)
    y = t.load(b)
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
def test_cmp_int64_s_r(op, func):
    t = Tester()
    a = np.random.randint(0x10000000, 0x2000000000, size=[800, 1000], dtype=np.int64)
    if op == Tester.equal or op == Tester.not_equal:
        a[10][100] = 1000
        a[10][100] = 1000
        a[100][3] = 1000
        a[100][3] = 1000
    x = t.load(a)
    y = op(t, x, 1000)
    t.store_expect(y, func(a, 1000))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.equal, np.equal), (Tester.less, np.less), (Tester.greater, np.greater),
                                      (Tester.greater_equal, np.greater_equal), (Tester.less_equal, np.less_equal),
                                      (Tester.not_equal, np.not_equal)])
def test_cmp_int64_s_l(op, func):
    t = Tester()
    a = np.random.randint(0x10000000, 0x2000000000, size=[800, 1000], dtype=np.int64)
    if op == Tester.equal or op == Tester.not_equal:
        a[10][100] = 1000
        a[10][100] = 1000
        a[100][3] = 1000
        a[100][3] = 1000
    x = t.load(a)
    y = op(t, 1000, x)
    t.store_expect(y, func(1000, a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_sequential_run_with_different_kernel_args():
    def run_copy(dtype):
        t = Tester()
        data = np.random.randint(1024, size=(1024, 2000)).astype(dtype)
        out = t.copy(t.load(data))
        t.store_expect(out, data)
        assert t.run_check()

    run_copy(np.int32)
    run_copy(np.float16)
