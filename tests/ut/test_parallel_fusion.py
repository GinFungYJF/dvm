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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape1, shape2,  shape3', [
    ([256], [256], [1024]),
    ([8192 * 2], [8192 * 4], [4096])
])
@pytest.mark.gpu
def test_basic(shape1, shape2, shape3):
    t = Tester("parallel")
    # kernel 0
    t.parallel_add(Tester.K_VEC)
    a0 = np.full(shape1, 0.1, np.float32)
    a = t.load(a0)
    b = t.mul(a, 0.3)
    c = t.store_expect(b, 0.1 * 0.3)
    # kernel 1
    t.parallel_add(Tester.K_VEC)
    b0 = np.full(shape2, 0.1, np.float32)
    a = t.load(b0)
    b = t.add(a, 0.3)
    c = t.store_expect(b, 0.1 + 0.3)
    # kernel 3
    t.parallel_add(Tester.K_VEC)
    c0 = np.full(shape3, 0.1, np.float32)
    a = t.load(c0)
    b = t.add(a, 0.3)
    c = t.store_expect(b, 0.1 + 0.3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('determ', [False, True])
def test_reduce(determ):
    t = Tester("parallel")
    t.set_deterministic(determ)
    # kernel 0
    t.parallel_add(Tester.K_VEC)
    a0 = np.full((8192,), 0.1, np.float32)
    a = t.sum(t.load(a0), [0], False)
    t.store_expect(a, 8192 * 0.1)
    # kernel 1
    t.parallel_add(Tester.K_VEC)
    b0 = np.full((4, 4096), 0.1, np.float32)
    b = t.sum(t.load(b0), [0], False)
    t.store_expect(b, 4 * 0.1)
    # kernel 2
    t.parallel_add(Tester.K_VEC)
    c0 = np.full((1,), 0.1, np.float32)
    c = t.add(t.load(c0), 1.0)
    t.store_expect(c, 0.1 + 1.0)
    assert (t.run_check())
    t.set_deterministic(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_determ_all():
    t = Tester("parallel")
    t.set_deterministic(True)
    # kernel 0
    t.parallel_add(Tester.K_VEC)
    a0 = np.full((8192,), 0.1, np.float32)
    a = t.sum(t.load(a0), [0], False)
    t.store_expect(a, 8192 * 0.1)
    # kernel 1
    t.parallel_add(Tester.K_VEC)
    b0 = np.full((4, 4096), 0.1, np.float32)
    b = t.sum(t.load(b0), [0], False)
    t.store_expect(b, 4 * 0.1)
    assert (t.run_check())
    t.set_deterministic(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('ktype1, ktype2', [[Tester.K_CUBE, Tester.K_CUBE], [Tester.K_CUBE, Tester.K_MIX], [Tester.K_MIX, Tester.K_MIX]])
def test_cube_cube(ktype1, ktype2):
    t = Tester("parallel")
    t.parallel_add(ktype1)
    xa = Tester.fast_random_normal(0, 0.1, [1024, 768]).astype(np.float16)
    xb = Tester.fast_random_normal(0, 0.1, [768, 2000]).astype(np.float16)
    x0 = t.load(xa)
    x1 = t.load(xb)
    x2 = t.matmul(x0, x1, False, False)
    x2_e = np.matmul(xa.astype(np.float32), xb.astype(np.float32)).astype(np.float16)
    if ktype1 == Tester.K_MIX:
        x2 = t.mul(x2, 0.5)
        x2_e = x2_e * 0.5
    t.store_expect(x2, x2_e, 1e-3)
    t.parallel_add(ktype2)
    ya = Tester.fast_random_normal(0, 0.1, [500, 1024]).astype(np.float16)
    yb = Tester.fast_random_normal(0, 0.1, [1024, 1600]).astype(np.float16)
    y0 = t.load(ya)
    y1 = t.load(yb)
    y2 = t.matmul(y0, y1, False, False)
    y2_e = np.matmul(ya.astype(np.float32), yb.astype(np.float32)).astype(np.float16)
    if ktype2 == Tester.K_MIX:
        y2 = t.add(y2, 0.3)
        y2_e = y2_e + 0.3
    t.store_expect(y2, y2_e, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('ktype', [Tester.K_CUBE, Tester.K_MIX])
def test_cube_vector(ktype):
    t = Tester("parallel")
    t.parallel_add(ktype)
    xa = Tester.fast_random_normal(0, 0.1, [1024, 768]).astype(np.float16)
    xb = Tester.fast_random_normal(0, 0.1, [768, 2000]).astype(np.float16)
    x0 = t.load(xa)
    x1 = t.load(xb)
    x2 = t.matmul(x0, x1, False, False)
    x2_e = np.matmul(xa.astype(np.float32), xb.astype(np.float32)).astype(np.float16)
    if ktype == Tester.K_MIX:
        x2 = t.mul(x2, 0.5)
        x2_e = x2_e * 0.5
    t.store_expect(x2, x2_e, 1e-3)
    t.parallel_add(Tester.K_VEC)
    ya = np.random.normal(0, 0.1, [10, 1024]).astype(np.float16)
    y0 = t.load(ya)
    y1 = t.add(y0, 0.3)
    y1 = t.mul(y1, y0)
    y2 = t.cast(y1, "float32")
    t.store_expect(y2, ((ya + 0.3) * ya).astype(np.float32))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('m_limit, c_limit, v_limit', [[0, 0, 0], [8, 8, 0], [0, 0, 10]])
def test_mix_cube_vector(m_limit, c_limit, v_limit):
    t = Tester("parallel")
    t.parallel_add(Tester.K_MIX, 0, m_limit)
    xa = Tester.fast_random_normal(0, 0.1, [1024, 768]).astype(np.float16)
    xb = Tester.fast_random_normal(0, 0.1, [768, 2000]).astype(np.float16)
    x0 = t.matmul(t.load(xa), t.load(xb), False, False)
    x1 = t.mul(x0, 0.5)
    t.store_expect(x1, np.matmul(xa.astype(np.float32), xb.astype(np.float32)).astype(np.float16) * 0.5, 1e-3)
    t.parallel_add(Tester.K_CUBE, 0, c_limit)
    ya = Tester.fast_random_normal(0, 0.1, [1536, 768]).astype(np.float16)
    yb = Tester.fast_random_normal(0, 0.1, [768, 768]).astype(np.float16)
    y0 = t.matmul(t.load(ya), t.load(yb), False, False)
    t.store_expect(y0, np.matmul(ya.astype(np.float32), yb.astype(np.float32)).astype(np.float16), 1e-3)
    t.parallel_add(Tester.K_VEC, 0, v_limit)
    za = np.random.normal(0, 0.1, [10, 2000]).astype(np.float32)
    z0 = t.load(za)
    z1 = t.add(z0, 0.3)
    z2 = t.sum(z1, (0,), True)
    t.store_expect(z2, np.sum(za + 0.3, (0,), keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('ktype', [Tester.K_CUBE, Tester.K_MIX])
def test_cube_vector_dyn(ktype):
    t = Tester("parallel:dyn")
    t.parallel_add(ktype)
    xa = t.load([-1], "float16")
    xb = t.load([-1], "float16")
    x0 = t.matmul(xa, xb, False, False)
    if ktype == Tester.K_MIX:
        x0 = t.mul(x0, 0.5)
        x0 = t.add(x0, 0.02)
    x_out = t.store(x0)
    t.parallel_add(Tester.K_VEC)
    ya = t.load([-1], "float32")
    yb = t.load([-1], "float32")
    y0 = t.add(ya, yb)
    y1 = t.sum(y0, (1,), True)
    y_out = t.store(y1)
    iterations = [
        [[512, 2048], [2048, 1024], [1, 2000, 128], [4, 1, 128]],
        [[2, 1024, 768], [768, 1024], [1, 512], [4000, 512]],
        [[512, 1024], [1024, 768], [1024, 200], [1024, 200]]]
    for xa_shape, xb_shape, ya_shape, yb_shape in iterations:
        xa_data = np.random.normal(0, 0.1, xa_shape).astype(np.float16)
        xb_data = np.random.normal(0, 0.1, xb_shape).astype(np.float16)
        ya_data = np.random.normal(0, 0.1, ya_shape).astype(np.float32)
        yb_data = np.random.normal(0, 0.1, yb_shape).astype(np.float32)
        t.input(xa, xa_data)
        t.input(xb, xb_data)
        t.input(ya, ya_data)
        t.input(yb, yb_data)
        t.run()
        x_e = np.matmul(xa_data.astype(np.float32), xb_data.astype(np.float32)).astype(np.float16)
        if ktype == Tester.K_MIX:
            x_e = x_e * 0.5 + 0.02
        t.check(x_out, x_e, 1e-3)
        y_e = np.sum(ya_data + yb_data, (1,), keepdims=True)
        t.check(y_out, y_e, 1e-3)
