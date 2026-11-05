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
@pytest.mark.parametrize('mode', ["eager", "eager:unify_ws", "eager:lazy", "eager:lazy,unify_ws"])
def test_eager_split_end(mode):
    ''' reduce -> broad '''
    t = Tester(mode)
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([10, 1024], 0.1, np.float32)
    expect = np.sqrt(np.sum((g0 + g1) * 2.0, (0,), keepdims=True) - g2)
    for i in range(2):
        x = t.add(t.load(g0), t.load(g1))
        x = t.mul(x, 2.0)
        x = t.sum(x, [0], True)  # [1, 1024]
        x = t.sub(x, t.load(g2))
        x = t.sqrt(x)
        t.store_expect(x, expect)
        assert (t.run_check())
        t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_split_middle():
    ''' {broad -> reduce} broad '''
    t = Tester("eager")
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([32, 512], 0.1, np.float32)
    ex = g0 + 0.1
    ey = (ex + g1) * 0.1
    ez = np.sum(g2 + ex, (1,), keepdims=True)
    for i in range(3):
        x = t.load(g0)
        x = t.add(x, 0.1)
        y = t.add(x, t.load(g1))
        z = t.load(g2)
        z = t.add(z, x)
        z = t.sum(z, [1], True)
        t.store_expect(z, ez)
        y = t.mul(y, 0.1)
        t.store_expect(y, ey)
        assert (t.run_check())
        t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_split_join():
    ''' {Broadcast, Reduce} -> broadcast'''
    t = Tester("eager")
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([32, 512], 0.1, np.float32)
    ex = g0 + 0.1
    ey = ex + g1
    ez = np.sum(g2 + ex, (1,), keepdims=True)
    es = (ey + ez) * 0.1
    for i in range(3):
        x = t.load(g0)
        x = t.add(x, 0.1)
        y = t.load(g1)
        y = t.add(x, y)
        z = t.load(g2)
        z = t.add(z, x)
        z = t.sum(z, [1], True)  # [31, 1]
        s = t.add(z, y)
        s = t.mul(s, 0.1)
        s = t.store_expect(s, es)
        assert (t.run_check())
        t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape1, shape2', [
    {(1, 1280, 28, 36), (1, 1280, 1, 1)},  # broadcast
    {(1, 8, 28, 512), (1, 8, 1, 1)},  # broadcast store
    {(1, 1000, 28, 128), (1, 1000)},  # not affine
])
def test_eager_disconnect(shape1, shape2):
    x0 = np.random.normal(0, 1, shape1).astype(np.float16)
    x1 = np.random.normal(0, 1, shape1).astype(np.float16)
    x2 = np.random.normal(0, 1, shape2).astype(np.float16)
    x3 = np.random.normal(0, 1, shape2).astype(np.float16)
    t = Tester("eager")
    a = t.add(t.load(x0), t.load(x1))
    b = t.add(t.load(x2), t.load(x3))
    t.store_expect(a, x0 + x1)
    t.store_expect(b, x2 + x3)
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_dead_node():
    t = Tester("eager")
    g0 = np.full([32, 512], 0.1, np.float32)
    g1 = np.full([32, 512], 0.2, np.float32)
    g2 = np.full([32, 512], 0.3, np.float32)
    x0 = t.load(g0)
    x1 = t.load(g1)
    x2 = t.sub(x0, x1)  # dead
    x3 = t.load(g2)
    x4 = t.add(x3, 0.1)
    x5 = t.mul(x4, 2.0)
    t.store_expect(x5, (0.3 + 0.1) * 2.0)
    assert (t.run_check())
    assert (t.das().count("load.") == 1)
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_split_load():
    t = Tester("eager")
    g0 = np.random.normal(0, 1, (4, 32)).astype(np.float32)
    g1 = np.random.normal(0, 1, [10, 4, 32]).astype(np.float32)
    g2 = np.random.normal(0, 1, [12, 4, 32]).astype(np.float32)
    x0 = t.load(g0)
    x1 = t.load(g1)
    x2 = t.load(g2)
    x3 = t.add(x0, x1)
    x4 = t.add(x0, x2)
    t.store_expect(x3, g0 + g1)
    t.store_expect(x4, g0 + g2)
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_split_load_reduce():
    t = Tester("eager")
    g0 = np.random.normal(0, 1, (1, 4, 1024)).astype(np.float32)
    g1 = np.random.normal(0, 1, [1, 1, 1024]).astype(np.float32)
    g2 = np.random.normal(0, 1, [12, 4, 1024]).astype(np.float32)
    x0 = t.add(t.load(g0), t.load(g1))
    x1 = t.sum(t.load(g2), (0,), True)
    t.store_expect(x0, g0 + g1)
    t.store_expect(x1, np.sum(g2, axis=(0,), keepdims=True))
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_stop_fuse_reduce():
    t = Tester("eager")
    x0_a = np.random.normal(0, 0.1, [428, 16, 8]).astype(np.float32)
    x0 = t.load(x0_a)
    x1 = t.abs(x0)
    x2 = t.sum(x1, (2,), False)
    x3_a = np.random.normal(0, 0.1, (1,)).astype(np.float32)
    x3 = t.load(x3_a)
    x4 = t.mul(x2, x3)
    x5_a = np.random.normal(0, 0.1, [428, 16]).astype(np.float32)
    x5 = t.load(x5_a)
    x6 = t.add(x5, x4)
    y7_numpy = np.add(x5_a, np.multiply(np.sum(np.abs(x0_a), axis=(2,), keepdims=False), x3_a))
    y7 = t.store_expect(x6, y7_numpy)
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_reduce_multi_user():
    t = Tester("eager")
    x0_a = np.random.normal(0, 0.1, [428, 16, 8]).astype(np.float32)
    x0 = t.load(x0_a)
    x1 = t.abs(x0)
    x2 = t.sum(x1, (2,), False)
    x3 = t.mul(x2, x2)
    x4_a = np.random.normal(0, 0.1, [428, 1]).astype(np.float32)
    x4 = t.add(x3, t.load(x4_a))
    x2_e = np.sum(np.abs(x0_a), axis=(2,), keepdims=False)
    x4_e = (x2_e * x2_e) + x4_a
    t.store_expect(x4, x4_e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["eager", "eager:unify_ws"])
@pytest.mark.parametrize('shape_a, shape_b', [
    [[128, 1024], [1024, 256]],  # no pad
    [[128, 500], [500, 256]],  # pad a
    [[128, 1024], [1024, 500]],  # pad b
    [[4096, 34816], [34816, 2560]],  # split k
    [[1024, 1, 256], [256, 512]],  # batch fold
])
def test_eager_mm(mode, shape_a, shape_b):
    t = Tester(mode)
    a = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    b = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    x0 = t.load(a)
    x0 = t.add(x0, 0.01)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 1.0)
    expect = np.matmul((a + 0.01).astype(np.float32), b.astype(np.float32)) + 1.0
    t.store_expect(x3, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_mm_bias():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    c = np.random.normal(0, 0.01, [256]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.load(c)
    x3 = t.matmul(x0, x1, False, False, x2)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32)) + c
    t.store_expect(x3, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_mm_bias_fp16():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float32)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float32)
    c = np.random.normal(0, 0.01, [256]).astype(np.float32)
    x0 = t.load(a, "bfloat16")
    x1 = t.load(b, "bfloat16")
    x2 = t.load(c, "bfloat16")
    x3 = t.matmul(x0, x1, False, False, x2)
    expect = np.matmul(a, b) + c
    t.store_expect(x3, expect, 5e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_cube_vec_reuse():
    ''' 2 vec -> 1 cube -> 1 vec '''
    t = Tester("eager")
    # round 1
    a = np.random.normal(0, 0.1, [256, 2048]).astype(np.float32)
    x = t.sum(t.load(a), (0,), True)
    x = t.add(x, 0.2)
    t.store_expect(x, np.sum(a, (0,), keepdims=True) + 0.2)
    assert (t.run_check())
    t.reset()
    # round 2
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x = t.matmul(t.load(a), t.load(b), False, False)
    t.store_expect(x, np.matmul(a.astype(np.float32), b.astype(np.float32)), 2e-3)
    assert (t.run_check())
    t.reset()
    # round 3
    a = np.random.normal(0, 0.1, [256, 512]).astype(np.float32)
    x = t.add(t.load(a), 0.2)
    x = t.add(x, 0.2)
    t.store_expect(x, a + 0.4)
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_pv_reuse():
    t = Tester("eager")
    x0 = np.full([32, 1024], 0.1, np.float32)
    x1 = np.full([32, 1024], 0.2, np.float32)
    x2 = np.full([10, 1024], 0.4, np.float32)
    x3 = np.full([10, 1024], 0.1, np.float32)
    for dtype in range(3):
        y0 = t.mul(t.load(x0), t.load(x1))
        y1 = t.sub(t.load(x2), t.load(x3))
        t.store_expect(y0, 0.02)
        t.store_expect(y1, 0.3)
        assert (t.run_check())
        t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_mix_vec_reuse():
    t = Tester("eager")
    # round 1
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x = t.matmul(t.load(a), t.load(b), False, False)
    x = t.add(x, 0.1)
    t.store_expect(x, np.matmul(a.astype(np.float32), b.astype(np.float32)) + 0.1, 2e-3)
    assert (t.run_check())
    t.reset()
    # round 2
    a = np.random.normal(0, 0.1, [256, 512]).astype(np.float32)
    x = t.add(t.load(a), 0.2)
    x = t.add(x, 0.2)
    t.store_expect(x, a + 0.4)
    assert (t.run_check())
    t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_eager_extern_code():
    t = Tester("eager")
    ax = np.random.normal(0, 0.1, [512, 512]).astype(np.float32)
    bx = np.full([1, 512], 0.01, np.float32)
    a = t.load(ax)
    b = t.load(bx)
    for i in range(200):
        a = t.add(a, b)
    t.store_expect(a, ax + 0.01 * 200)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_workspace_inplace():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [128, 1024]).astype(np.float16)
    x0 = t.cast(t.load(a), "float32")
    x1 = t.sum(x0, (0,), False)
    x2 = t.cast(x1, "float16")
    t.store_expect(x2, np.sum(a.astype(np.float32), axis=(0,), keepdims=False).astype(np.float16), 2e-3)
    b = np.random.normal(0, 0.1, [1, 1024]).astype(np.float32)
    x3 = t.mul(t.load(b), x0)
    t.store_expect(x3, b * a.astype(np.float32))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_load_reuse():
    t = Tester("eager")
    ax = np.random.normal(0, 0.1, [288]).astype(np.float32)
    bx = np.random.normal(0, 0.1, [1]).astype(np.float32)
    x0 = t.load(ax)
    x1 = t.load(bx)
    x2 = t.mul(x0, x1)
    t.store_expect(x2, ax * bx)
    cx = np.random.normal(0, 0.1, [864, 1152]).astype(np.float32)
    x3 = t.load(cx)
    x4 = t.add(x3, x1)
    t.store_expect(x4, cx + bx)
    x5 = t.add(x2, 0.1)
    t.store_expect(x5, ax * bx + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_lazy_tuner():
    t = Tester("eager")
    Tester.set_lazy_tuning(True)
    shape = [4096, 4096]
    a = np.random.normal(0, 0.01, shape).astype(np.float16)
    b = np.random.normal(0, 0.01, shape).astype(np.float16)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32))
    for i in range(3):
        x0 = t.load(a)
        x1 = t.load(b)
        x2 = t.matmul(x0, x1, False, False)
        t.store_expect(x2, expect, 1e-3)
        assert (t.run_check())
        t.reset()
    Tester.set_lazy_tuning(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_pv_1():
    ''' parallel: {V, V} '''
    t = Tester("eager")
    g0 = np.full([32, 1024], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    x1 = t.add(t.load(g0), 1.0)
    x2 = t.add(t.load(g1), 2.0)
    t.store_expect(x1, 1.1)
    t.store_expect(x2, 2.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_pv_2():
    ''' parallel: V -> {V, V} '''
    t = Tester("eager")
    g0 = np.full([10, 1024], 0.1, np.float32)
    g1 = np.full([16, 1024], 0.2, np.float32)
    g2 = np.full([17, 1024], 0.3, np.float32)
    x0 = t.load(g0)
    x1 = t.sum(x0, (0,), True)
    x2 = t.add(t.load(g1), x1)
    x3 = t.add(t.load(g2), x1)
    t.store_expect(x2, 1.2)
    t.store_expect(x3, 1.3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_pv_3():
    ''' parallel: V -> {V, V} -> V '''
    t = Tester("eager")
    g0 = np.full([4, 5, 3, 1024], 0.1, np.float32)
    x0 = t.load(g0)
    x1 = t.sum(x0, (0,), True)
    x2 = t.sum(x1, (1,), True)
    x3 = t.sum(x1, (2,), True)
    x4 = t.add(x2, x3)
    t.store_expect(x4, 3.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_0():
    ''' matmul + area1 + area2 '''
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 0.1)
    x4 = t.add(x3, 0.1)
    expect_x2 = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    t.store_expect(x4, expect_x2 + 0.2)
    x5 = t.sub(x3, 0.05)
    t.store_expect(x5, expect_x2 + 0.05)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_1():
    ''' matmul -> {area1, area2} '''
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 0.1)
    x4 = t.add(x3, 0.1)
    expect_x2 = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    t.store_expect(x4, expect_x2 + 0.2)
    c = np.random.normal(0, 0.01, [2, 256, 256]).astype(np.float16)
    x5 = t.sub(x2, t.load(c))
    t.store_expect(x5, expect_x2 - c)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_2():
    ''' {area1, area2} -> matmul '''
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x0 = t.load(a)
    x0 = t.mul(x0, 1.2)
    x1 = t.load(b)
    x1 = t.mul(x1, 0.8)
    t.store_expect(x1, b * 0.8)
    x2 = t.matmul(x0, x1, False, False)
    expect_x2 = np.matmul(a.astype(np.float32) * 1.2, b.astype(np.float32) * 0.8).astype(np.float16)
    t.store_expect(x2, expect_x2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_same_input_1():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.matmul(x0, x0, False, False)
    x2 = t.add(x1, 0.1)
    x0_e = a.astype(np.float32)
    expect_x2 = np.matmul(x0_e, x0_e).astype(np.float16) + 0.1
    t.store_expect(x2, expect_x2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_same_input_2():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    b = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, x1)
    expect_x3 = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16) + b
    t.store_expect(x3, expect_x3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, shape_broadcast', [
    [[1024, 512], [512, 1024], [1, 1024]],
    [[512, 512], [512, 1024], [512, 1]],
    [[2, 4, 512, 512], [512, 256], [2, 1, 512, 1]],
    [[2, 4, 512, 512], [2, 1, 512, 256], [256]],
    [[1024, 1, 256], [256, 512], [1024, 1, 512]], # batch fold: all elemwise
    [[1024, 1, 256], [256, 512], [1, 1, 512]], # batch fold: batch broadcast
    [[1024, 1, 256], [256, 512], [1024, 1, 1]], # batch fold: n broadcast
    [[4, 128, 1, 256], [256, 512], [4, 1, 1, 512]], # batch fold: check fail
])
def test_eager_cv_broadcast(shape_a, shape_b, shape_broadcast):
    t = Tester("eager")
    a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    expect_x2 = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    c = np.random.normal(0, 0.01, shape_broadcast).astype(np.float16)
    x3 = t.load(c)
    x4 = t.add(x3, 0.1)
    x5 = t.add(x2, x4)
    t.store_expect(x5, expect_x2 + c + 0.1)
    t.store_expect(x4, c + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_pv_multi_parallel_area():
    input0 = np.random.random([2, 3, 4]).astype(np.float32)
    input1 = np.random.random([1]).astype(np.float32)
    input2 = np.random.random([2, 3, 4]).astype(np.float32)
    input3 = np.random.random([1]).astype(np.float32)
    t = Tester("eager")
    t0 = t.load(input0)
    t1 = t.cast(t0, "float16")
    t2 = t.load(input1)
    t3 = t.cast(t2, "float16")
    t4 = t.mul(t1, t3)
    t5 = t.store_expect(t4, input0.astype(np.float16) * input1.astype(np.float16))
    t6 = t.load(input2)
    t7 = t.cast(t6, "float16")
    e_t7 = input2.astype(np.float16)
    t8 = t.store_expect(t7, e_t7)
    t9 = t.load(input3)
    t10 = t.cast(t9, "float16")
    e_t10 = input3.astype(np.float16)
    t11 = t.store_expect(t10, e_t10)
    t12 = t.mul(t7, t10)
    t13 = t.cast(t12, "float16")
    t14 = t.store_expect(t13, e_t7 * e_t10)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_many_split():
    ''' from mindtest '''
    t = Tester("eager")
    g0 = np.full([5, 4], 0.1, np.float32)
    x = t.load(g0)
    expect = 0.1
    output = []
    for i in range(50):
        x = t.add(x, x)
        y = x
        expect *= 2
        output.append((y, expect))
    for o, e in output:
        t.store_expect(o, e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_depend():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [1, 32, 2, 4]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.125)
    x2 = t.sum(x1, (0, 2, 3), True)
    e_x2 = np.sum(a + 0.125, axis=(0, 2, 3), keepdims=True)
    t.store_expect(x2, e_x2)
    x3 = t.sub(x0, x2)
    x4 = t.mul(x3, -0.125)
    x5 = t.sum(x4, (0, 2, 3), False)
    e_x5 = np.sum((a - e_x2) * (-0.125), axis=(0, 2, 3), keepdims=False)
    t.store_expect(x5, e_x5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ["eager", "eager:unify_ws"])
def test_skip_output(mode):
    t = Tester(mode)
    a = np.random.normal(0, 0.1, [1, 32, 2, 4]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.125)
    x2 = t.sum(x1, (0, 2, 3), True)
    e_x2 = np.sum(a + 0.125, axis=(0, 2, 3), keepdims=True)
    t.store_expect(x2, e_x2)
    # skip ops
    x3 = t.sqrt(x1)
    x4 = t.mul(x3, -0.125)
    x5 = t.sum(x4, (0, 2, 3), False)
    x6 = t.add(x5, 0.02)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_store_inplace():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [64, 128]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.sum(x0, (1,), True)
    x2 = t.mul(x1, 0.5)
    x3 = t.add(x2, x2)
    x4 = t.store_expect(x3, np.sum(a, axis=(1,), keepdims=True) * 0.5 * 2)
    t.set_store_inplace(x4)
    assert (t.run_check())
    addrs = set()
    for line in t.das().split("\n"):
        if "store.u8" in line:
            addrs.add(line.strip().split(" ")[2])
    assert (len(addrs) == 2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_select():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [64, 128]).astype(np.float32)
    b = np.random.normal(0, 0.1, [64, 128]).astype(np.float32)
    c = np.random.normal(0, 0.1, [4, 64, 128]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.greater(x0, x1)
    x3 = t.load(c)
    x4 = t.add(x0, x3)
    x5 = t.mul(x1, x3)
    x6 = t.select(x2, x4, x5)
    x7 = t.add(x6, x0)
    x2_e = np.greater(a, b)
    t.store_expect(x7, np.select([x2_e, ~x2_e], [a + c, b * c]) + a)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_pv():
    t = Tester("eager")
    a = np.full([10, 0, 2], 0.1, np.float32)
    b = np.full([10, 1, 2], 0.2, np.float32)
    x = t.load(a)
    y = t.mul(x, 0.1)
    out1 = t.store(y)
    z = t.load(b)
    z = t.mul(z, 0.2)
    t.store_expect(z, 0.2 * 0.2)
    assert(t.run_check())
    assert(out1.shape() == (10, 0, 2))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_reduce():
    t = Tester("eager")
    a = np.full([10, 0, 2], 0.01, np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.1)
    x3 = t.sum(x2, (2,), True)
    x4 = t.add(x3, 0.2)
    x5 = t.sum(x4, (1,), True)
    out = t.store(x5)
    t.run()
    assert(out.shape() == (10, 1, 1))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_mm():
    t = Tester("eager")
    a = np.full([0, 1024], 0.001, np.float16)
    b = np.full([1024, 1024], 0.001, np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    out = t.store(x2)
    t.run()
    assert(out.shape() == (0, 1024))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_cv():
    t = Tester("eager")
    a = np.full([0, 1024], 0.001, np.float16)
    b = np.full([1024, 1024], 0.001, np.float16)
    x0 = t.load(a)
    x0 = t.add(x0, 0.01)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 1.0)
    out = t.store(x3)
    t.run()
    assert(out.shape() == (0, 1024))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_kernel_build():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [10, 1024]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.01)
    x2 = t.mul(x1, x1)
    x3 = t.add(x2, x2)
    x4 = t.add(x3, x1)
    x1_e = a + 0.01
    x2_e = x1_e * x1_e
    x3_e = x2_e + x2_e
    x4_e = x3_e + x1_e
    t.store_expect(x4, x4_e)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_broadcast_s():
    t = Tester("eager")
    x1 = t.full(0.1, [256], "float32")
    x2 = t.add(x1, 0.2)
    t.store_expect(x2, 0.3)
    x3 = t.full(0.4, [512], "float32")
    t.store_expect(x3, 0.4)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_broadcast_reduce_reuse():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [1, 2000]).astype(np.float32)
    b = np.random.normal(0, 0.1, [10, 2000]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.mul(x1, x2)
    x4 = t.sum(x3, (0,), True)
    x5 = t.add(x4, x2)
    x6 = t.sum(x5, (0,), True)
    e4 = np.sum(a * b, axis=(0,), keepdims=True)
    e6 = np.sum(e4 + b, axis=(0,), keepdims=True)
    t.store_expect(x6, e6, 1e-4)
    assert(t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_broadcast_store_reuse():
    t = Tester("eager")
    a0 = np.random.normal(-1, 1, [3000]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.01)
    x2 = t.sum(x1, (0,), True)
    y1 = t.mul(x1, 0.8)
    a1 = np.random.normal(-1, 1, [10, 3000]).astype(np.float32)
    y2 = t.add(y1, t.load(a1))
    e1 = (a0 + 0.01) * 0.8
    t.store_expect(x2, np.sum(a0 + 0.01, axis=(0,), keepdims=True))
    t.store_expect(y2, e1 + a1)
    t.store_expect(y1, e1)
    assert (t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_with_view_load():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    b = np.random.normal(0, 0.01, [1024, 512]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    c = np.random.normal(0, 1, [1024, 1000]).astype(np.float16)
    x3 = t.view_load([1024, 512], [1000, 1], c)
    x4 = t.add(x2, x3)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16) + c[:, :512]
    t.store_expect(x4, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["eager", "eager:lazy", "eager:lazy,unify_ws"])
def test_eager_cv_multi_user(mode):
    t = Tester(mode)
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    b = np.random.normal(0, 0.01, [1024, 512]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.cast(x2, "float32")
    x4 = t.cast(x2, "float32")
    x5 = t.add(x3, x4)
    e_cast = np.matmul(a.astype(np.float32), b.astype(np.float32))
    t.store_expect(x5, e_cast + e_cast, 1e-3)
    assert (t.run_check())


@pytest.mark.mix
def test_eager_cv_horizontal_user():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    b = np.random.normal(0, 0.01, [1024, 512]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 0.1)
    x4 = t.mul(x2, 0.5)
    e2 = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    t.store_expect(x3, e2 + 0.1, 1e-3)
    t.store_expect(x4, e2 * 0.5, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_eager_cv_multi_c():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    b = np.random.normal(0, 0.01, [1024, 512]).astype(np.float16)
    c = np.random.normal(0, 0.01, [1024, 768]).astype(np.float16)
    d = np.random.normal(0, 0.01, [768, 512]).astype(np.float16)
    x0 = t.matmul(t.load(a), t.load(b), False, False)
    x1 = t.matmul(t.load(c), t.load(d), False, False)
    x2 = t.add(x0, x1)
    t.store_expect(x2, np.matmul(a.astype(np.float32), b.astype(np.float32)) + np.matmul(c.astype(np.float32), d.astype(np.float32)), 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, shape_c', [
    [[1024, 512], [512, 256], [256, 1024]],  # no pad
    [[4096, 640], [640, 8], [8, 640]],  # pad
])
def test_eager_cube_chain(shape_a, shape_b, shape_c):
    t = Tester("eager")
    a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    c = np.random.normal(0, 0.01, shape_c).astype(np.float16)
    x0 = t.matmul(t.load(a), t.load(b), False, False)
    x1 = t.matmul(x0, t.load(c), False, False)
    e0 = np.matmul(a.astype(np.float32), b.astype(np.float32))
    e1 = np.matmul(e0, c.astype(np.float32)).astype(np.float16)
    t.store_expect(x1, e1, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_reshape():
    t = Tester("eager")
    a0 = np.random.normal(-1, 1, [600, 512]).astype(np.float16)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.01)
    x2 = t.reshape(x1, [100, 48, 64])
    x3 = t.mul(x2, 0.8)
    t.store_expect(x3, np.reshape(a0 + 0.01, [100, 48, 64]) * 0.8)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_reshape_with_store():
    t = Tester("eager")
    a0 = np.random.normal(-1, 1, [1, 60, 512]).astype(np.float16)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.01)
    x2 = t.reshape(x1, [60, 1, 512])
    x3 = t.mul(x2, x1)
    e2 = np.reshape(a0 + 0.01, [60, 1, 512])
    t.store_expect(x2, e2)
    t.store_expect(x3, e2 * (a0 + 0.01))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_eager_lazy_vector_ws():
    t = Tester("eager:unify_ws,lazy")
    in_shape, dims = [30, 4000], (0,)
    t.set_deterministic(True)
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x0 = t.load(a)
    x = t.sum(t.mul(x0, 1.5), dims, True)
    x = t.add(x, 0.1)
    t.store_expect(x, np.sum(a * 1.5, dims, keepdims=True) + 0.1)
    assert(t.run_check())
    t.set_deterministic(False)
