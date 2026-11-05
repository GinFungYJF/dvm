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
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [(Tester.sqrt, np.sqrt), (Tester.abs, np.abs), (Tester.log, np.log), (Tester.exp, np.exp),
                                      (Tester.reciprocal, np.reciprocal)])
@pytest.mark.gpu
def test_unary(type, op, func):
    t = Tester()
    a = np.full([32, 1024], 0.5, type)
    x = t.load(a)
    y = op(t, x)
    t.store_expect(y, func(a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.div, np.divide),
                                      (Tester.mul, np.multiply), (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binary(type, op, func):
    t = Tester()
    a = np.full([32, 1024], 0.5, type)
    b = np.full([32, 1024], 0.4, type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = op(t, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.bool_, np.float16, np.float32])
@pytest.mark.parametrize("shape", [(1024, 32), (1312, 131), (16, 11), (128, 7), (128, 777)])
@pytest.mark.parametrize('op, func', [(Tester.logical_or, np.logical_or), (Tester.logical_and, np.logical_and)])
@pytest.mark.gpu
def test_logical(type, shape, op, func):
    t = Tester()
    a = np.random.choice([True, False], shape).astype(type)
    b = np.random.choice([True, False], shape).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = op(t, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b).astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.logical_or, np.logical_or), (Tester.logical_and, np.logical_and)])
@pytest.mark.gpu
def test_logical_mixed_compare_dtype(op, func):
    t = Tester()
    shape = (128, 777)
    a = np.random.randint(-1024, 1024, shape).astype(np.int32)
    b = np.random.randint(-1024, 1024, shape).astype(np.int32)
    c = np.random.normal(0, 1, shape).astype(np.float32)
    d = np.random.normal(0, 1, shape).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.load(c)
    x3 = t.load(d)
    y0 = t.greater(x0, x1)
    y1 = t.less(x2, x3)
    z = op(t, y0, y1)
    t.store_expect(z, func(a > b, c < d))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.bool_, np.float16, np.float32, np.int32])
@pytest.mark.parametrize("shape", [(1024, 32), (1312, 131), (16, 11), (128, 7), (128, 777)])
@pytest.mark.gpu
def test_logical_not(type, shape):
    t = Tester()
    a = np.random.choice([True, False], shape).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.logical_not(x)
    z = t.copy(z)
    t.store_expect(z, np.logical_not(a).astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.mul, np.multiply), (Tester.div, np.divide),
                                      (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binary_s(type, op, func):
    t = Tester()
    a = np.random.normal(0, 1, [32, 1024]).astype(type)
    x = t.load(a)
    x = t.copy(x)
    y = op(t, x, 0.1)
    y = t.copy(y)
    t.store_expect(y, func(a, 0.1))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [(Tester.div, np.divide), (Tester.sub, np.subtract)])
@pytest.mark.gpu
def test_binary_s_l(type, op, func):
    t = Tester()
    a = np.random.normal(0.05, 1, [32, 1024]).astype(type)
    x = t.load(a)
    x = t.copy(x)
    y = op(t, 0.1, x)
    y = t.copy(y)
    t.store_expect(y, func(0.1, a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize("size", [1024, 24, 66666])
@pytest.mark.gpu
def test_isfinite(type, size):
    t = Tester()
    random_numbers = np.random.randn(size)
    special_values = np.array([np.inf, -np.inf, np.nan])
    a = np.concatenate((random_numbers, np.tile(special_values, size))).astype(type)
    np.random.shuffle(a)
    x = t.load(a)
    x = t.copy(x)
    y = t.isfinite(x)
    y = t.copy(y)
    t.store_expect(y, np.isfinite(a).astype(type), 0)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_repeat_overflow(type):
    repeat = 257
    a = np.full([32 * 64 * repeat], 0.5, type)
    t = Tester()
    x = t.load(a)
    y = t.add(x, 0.1)
    t.store_expect(y, 0.6)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_scalar(type):
    t = Tester()
    a = np.full([1], 0.4, type)
    b = np.full([1], 0.2, type)
    x = t.load(a)
    y = t.load(b)
    z = t.sub(x, y)
    t.store_expect(z, 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape", [(32, 1024), (17, 129)])
def test_cast_int64(shape):
    t = Tester()
    s32 = np.random.randint(low=-2 ** 30, high=2 ** 30, size=shape).astype(np.int32)
    s64 = np.random.randint(low=-2 ** 30, high=2 ** 30, size=shape, dtype=np.int64)
    f32 = np.random.randint(low=-8192, high=8192, size=shape).astype(np.float32)
    x32 = t.load(s32)
    x64 = t.load(s64)
    xf32 = t.load(f32)
    t.store_expect(t.cast(x32, "int64"), s32.astype(np.int64))
    t.store_expect(t.cast(x64, "int32"), s64.astype(np.int32))
    t.store_expect(t.cast(xf32, "int64"), f32.astype(np.int64))
    t.store_expect(t.cast(x64, "float32"), s64.astype(np.float32))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type, eps, shape', [(np.float16, 1e-03, [2, 3, 4, 5]), (np.float32, 1e-05, [1, 1, 4, 5])])
@pytest.mark.gpu
def test_scalar_tensor_div(type, eps, shape):
    t = Tester()
    x = t.full(2, shape, type.__name__)
    a = np.random.random(shape).astype(type)
    y = t.load(a)
    z = t.div(x, y)
    np_res = 2 / a
    t.store_expect(z, np_res)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.gpu
def test_mul1(data_type):
    np.random.seed(1)
    t = Tester()
    a0 = np.random.normal(0, 1, [16000, 39, 1]).astype(data_type)
    a1 = np.random.normal(0, 1, [16000, 39, 1]).astype(data_type)
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.mul(x0, x1)
    expect = a0 * a1
    t.store_expect(y0, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_pow_1(type):
    t = Tester()
    a = np.array([-2, -3, -4, 0, -2, 9]).astype(type)
    b = np.array([1, 2, 3, 0, 0.5, 0.5]).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.pow(x, y)
    t.store_expect(z, np.array([-2, 9, -64, 1, np.nan, 3]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize("shape", [(1024, 32), (13, 131), (16, 11), (3, 3)])
@pytest.mark.gpu
def test_pow_2(type, shape):
    t = Tester()
    a = 10 * np.random.rand(*shape).astype(type) - 5
    b = np.random.randint(6, size=shape).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.pow(x, y)
    z = t.copy(z)
    t.store_expect(z, np.power(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape", [(1024, 32), (13, 131)])
@pytest.mark.parametrize("exponent", [0.0, 1.2, 5.0, -7.0, -3.0])
@pytest.mark.gpu
def test_pow_3(shape, exponent):
    t = Tester()
    a = np.abs(np.random.normal(0, 1, shape).astype(np.float32) - 2)
    x = t.load(a)
    y = t.pow(x, exponent)
    z = t.add(y, 2)
    z = t.copy(z)
    t.store_expect(z, np.power(a, exponent) + 2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
@pytest.mark.gpu
def test_rsqrt(type, eps):
    t = Tester()
    a = np.random.rand(32, 32).astype(type)
    x = t.load(a)
    z = t.sqrt(x)
    y = t.reciprocal(z)
    t.store_expect(y, np.reciprocal(np.sqrt(a)).astype(type), eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.parametrize('op, func',
                         [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.mul, np.multiply), (Tester.maximum, np.maximum),
                          (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binary_int(type, op, func):
    t = Tester()
    a = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(32, 1024)).astype(type)
    b = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(32, 1024)).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = op(t, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("op, func", [(Tester.add, np.add), (Tester.sub, np.subtract)])
def test_int64_binary(op, func):
    t = Tester()
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=(3200, 1024), dtype=np.int64)
    b = np.random.randint(low=-0x2000000000, high=0x2000000000, size=(3200, 1024), dtype=np.int64)
    x = t.load(a)
    y = t.load(b)
    z = op(t, x, y)
    t.store_expect(z, func(a, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("op, func", [(Tester.add, np.add), (Tester.sub, np.subtract)])
def test_int64_binary_scalar(op, func):
    t = Tester()
    shape = (1024, 1025)
    scalar = -0x100000000
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=shape, dtype=np.int64)
    x = t.load(a)
    y0 = op(t, x, scalar)
    y1 = op(t, scalar, x)
    t.store_expect(y0, func(a, scalar))
    t.store_expect(y1, func(scalar, a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("op, func", [(Tester.add, np.add), (Tester.sub, np.subtract)])
def test_int64_binary_scalar_ref(op, func):
    t = Tester('vector:dyn')
    shape = (1024, 1025)
    x = t.load([-1], "int64")
    s = t.scalar(dvm.int64)
    y0 = op(t, x, s)
    y1 = op(t, s, x)
    out0 = t.store(y0)
    out1 = t.store(y1)
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=shape, dtype=np.int64)
    t.input(x, a)
    for scalar in [1, 0x100000000, -0x100000000]:
        s.update(scalar)
        t.run()
        assert t.check(out0, func(a, scalar))
        assert t.check(out1, func(scalar, a))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_int64_add_sub_scalar_codegen():
    t = Tester()
    shape = (1024, 1025)
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=shape, dtype=np.int64)
    x = t.load(a)
    y = t.sub(t.add(x, 1), 2)
    t.store_expect(y, np.subtract(np.add(a, 1), 2))
    assert (t.run_check())
    if dvm.Device.arch() != 'AscendC310':
        das = t.das()
        assert das.count("Pack.b32") == 1
        assert das.count("Extract.b32") == 2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_int64_mixed_add_sub_cast_le():
    t = Tester()
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=(1024), dtype=np.int64)
    b = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(256, 32, 1024), dtype=np.int64)
    c = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(1, 32, 1), dtype=np.int64)
    x = t.load(a)
    y = t.load(b)
    z = t.load(c)
    y64 = t.cast(t.cast(y, "int32"), "int64")
    z64 = t.cast(t.cast(z, "int32"), "int64")
    add = t.add(x, y64)
    sub = t.sub(add, z64)
    le = t.less_equal(sub, y)
    expect_add = np.add(a, b)
    expect_sub = np.subtract(expect_add, c)
    t.store_expect(add, expect_add)
    t.store_expect(sub, expect_sub)
    t.store_expect(le, np.less_equal(expect_sub, b))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.parametrize('op, func',
                         [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.mul, np.multiply), (Tester.maximum, np.maximum),
                          (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binarys_int(type, op, func):
    t = Tester()
    a = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(32, 1024)).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = op(t, x, 11111)
    z = t.copy(z)
    t.store_expect(z, func(a, 11111))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.gpu
def test_unary_int(type):
    t = Tester()
    a = np.random.randint(low=-2 ** 30, high=2 ** 30, size=(32, 1024)).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.abs(x)
    z = t.copy(z)
    t.store_expect(z, np.abs(a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('num, expect', [(3, 2), (10, 4), (-8, 3), (11, 5), (15, 6), (20, 5)])
@pytest.mark.gpu
def test_power_s(num, expect):
    t = Tester()
    arg = np.random.random([1024, ]).astype(np.float32)
    a = t.load(arg)
    b = t.pow(a, num)
    t.store_expect(b, np.power(arg, num))
    assert (t.run_check())
    assert (t.das().count("Mul.fp32") == expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_pow_s_left(type):
    t = Tester()
    a = np.random.normal(0, 1, (1024, 32)).astype(type)
    x = t.load(a)
    y = t.pow(0.91, x)
    t.store_expect(y, np.power(0.91, a))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [(Tester.round, np.round), (Tester.ceil, np.ceil), (Tester.floor, np.floor), (Tester.trunc, np.trunc)])
@pytest.mark.parametrize('shape', [[32, 32], [128 * 256 * 2 * 48]])
@pytest.mark.gpu
def test_trunc(type, op, func, shape):
    t = Tester()
    a = np.random.normal(-10, 10, shape).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = op(t, x)
    z = t.copy(z)
    t.store_expect(z, func(a).astype(type))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support bfloat16 op")
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.div, np.divide),
                                      (Tester.mul, np.multiply), (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
def test_binary_bf16(op, func):
    t = Tester()
    a = Tester.bf16_random_normal(-1, 1, [256, 1]).astype(np.float32)
    b = Tester.bf16_random_normal(-1, 1, [256, 1]).astype(np.float32)
    x = t.load(a, "bfloat16")
    y = t.load(b, "bfloat16")
    z = op(t, x, y)
    t.store_expect(z, func(a, b), 1e-2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support bfloat16 op")
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.div, np.divide),
                                      (Tester.mul, np.multiply), (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
def test_binarys_bf16(op, func):
    t = Tester()
    a = Tester.bf16_random_normal(-1, 1, [1024, 1]).astype(np.float32)
    x = t.load(a, "bfloat16")
    z = op(t, x, 1.5)
    t.store_expect(z, func(a, 1.5), 5e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.div, np.divide),
                                      (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binary_scalar_ref(op, func):
    t = Tester('vector:dyn')
    x = t.load([-1], "float32")
    s = t.scalar()
    a = op(t, x, s)
    out = t.store(a)
    d1 =  np.random.normal(-10, 10, [1024]).astype(np.float32)
    t.input(x, d1)
    for i in [-1, -2, -12, 5, 6, 4]:
        s.update(i)
        t.run()
        assert (t.check(out, func(d1, i)))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('op, func', [(Tester.add, np.add), (Tester.sub, np.subtract), (Tester.div, np.divide),
                                      (Tester.maximum, np.maximum), (Tester.minimum, np.minimum)])
@pytest.mark.gpu
def test_binary_float_scalar_ref(op, func):
    t = Tester('vector:dyn')
    x = t.load([-1], "float32")
    s = t.scalar()
    a = op(t, x, s)
    out = t.store(a)
    d1 =  np.random.normal(-10, 10, [1024]).astype(np.float32)
    t.input(x, d1)
    for i in [-1.3, -2.7, -1.111, 5.55]:
        s.update(i)
        t.run()
        assert (t.check(out, func(d1, i)))

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_bool_cast_eliminate():
    t = Tester()
    a = np.random.choice([True, False], [32, 64]).astype(np.bool_)
    b = np.random.choice([True, False], [32, 64]).astype(np.bool_)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.logical_or(t.logical_not(x0), x1)
    t.store_expect(x2, np.logical_or(np.logical_not(a), b))
    assert (t.run_check())
