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
@pytest.mark.parametrize("shape1, shape2, shape3, reduce_dims",
                         [[(10, 3, 1), (10, 3, 4), (1, 10, 12), (2,)],
                          [(10, 3, 1), (10, 3, 4), (1, 10, 12), (1,)]])
@pytest.mark.gpu
def test_reduce_forward(shape1, shape2, shape3, reduce_dims):
    t = Tester()
    b = np.full(shape1, 0.1, np.float32)
    y = t.load(b)
    z = t.broadcast(y, shape2)
    z = t.reshape(z, shape3)
    z = t.sum(z, reduce_dims, True)
    c = np.copy(b)
    c = np.broadcast_to(c, shape2)
    print(c.shape)
    c = np.reshape(c, shape3)
    e = np.sum(c, reduce_dims, keepdims=True)
    t.store_expect(z, e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize("shape1", [(16, 16)])
@pytest.mark.parametrize("shape2", [(32, 8)])
def test_unary(shape1, shape2):
    t = Tester()
    x = np.full(shape1, 81.0, np.float32)
    a = t.load(x)
    b = t.sqrt(a)
    c = t.reshape(b, shape2)
    d = t.sqrt(c)
    t.store_expect(d, 3.0)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize("shape1", [(16, 16)])
@pytest.mark.parametrize("shape2", [(32, 8)])
def test_cast(shape1, shape2):
    t = Tester()
    x = np.full(shape1, 81.0, np.float32)
    a = t.load(x)
    b = t.sqrt(a)
    c = t.reshape(b, shape2)
    d = t.sqrt(c)
    d = t.cast(d, "float16")
    t.store_expect(d, 3.0)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(3, 3)])
@pytest.mark.parametrize("shape2", [(9,)])
@pytest.mark.parametrize('type, eps', [(np.float32, 1e-5)])
def test_select_forward(shape, shape2, type, eps):
    t = Tester()
    a = np.full(shape2, 1.0).astype(type)
    b = np.full(shape, 2.0).astype(type)
    c = np.random.choice([True, False], shape).astype(bool)
    x = t.load(c)
    y = t.load(a)
    y = t.reshape(y, shape)
    z = t.load(b)
    z = t.select(x, y, z)
    t.store_expect(z, np.select([c == True, c == False], [a.reshape(shape), b]), eps)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_backward():
    t = Tester()
    b = np.full([1, 8], 0.3, np.float32)
    y = t.load(b)
    z = t.broadcast(y, [10, 8])
    z = t.reshape(z, [10, 1, 8])
    z = t.broadcast(z, [10, 4, 8])
    t.store_expect(z, 0.3000)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_forward():
    t = Tester()
    b = np.full([10, 3, 4], 0.3, np.float32)
    y = t.load(b)
    z = t.reshape(y, [1, 10, 12])
    z = t.broadcast(z, [8, 10, 12])
    t.store_expect(z, 0.3000)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_s():
    t = Tester()
    z = t.full(0.3, [3, 10], "float32")
    z = t.reshape(z, [3, 1, 10])
    z = t.broadcast(z, [3, 10, 10])
    t.store_expect(z, 0.3000)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_multi_reshape_1():
    t = Tester()
    a = np.random.rand(54, 89).astype(np.float32)
    x = t.load(a)
    y = t.reshape(x, [89, 54])
    y = t.sqrt(y)
    z = t.reciprocal(x)
    z = t.reshape(z, [89, 54])
    z = t.add(y, z)
    expect = np.add(np.sqrt(a), np.reciprocal(a)).reshape([89, 54])
    t.store_expect(z, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_reshape_2():
    t = Tester()
    a = np.random.rand(54, 89).astype(np.float32)
    x = t.load(a)
    y = t.reshape(x, [89, 54])
    y = t.sqrt(y)
    z = t.reshape(y, [178, 27])
    t.store_expect(z, np.sqrt(a).reshape([178, 27]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_element_any_forward():
    t = Tester()
    a = np.full([10, 10], 0, np.float32)
    a[1, 1] = 1
    x = t.load(a)
    g = t.abs(x)
    g = t.reshape(g, [5, 10, 2])
    z = t.element_any(g)
    b = t.store(z)
    t.run_check()
    assert t.output(b)[0] == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_unalign_broadcast():
    t = Tester()
    a = np.full([1, 320], 0.3, np.float16)
    b = np.full((), 0.2, np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.cast(x1, "float32")
    x4 = t.reshape(x3, [320])
    x5 = t.mul(x4, x2)
    t.store_expect(x5, 0.06)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize("shape1, shape2, shape3", [
    [[10, 20, 8], [20, 8], [200, 8]],
    [[10, 20, 8], [8], [200, 8]],
    [[10, 20, 8], [8], [10, 160]],
])
def test_unalign_broadcast_2(shape1, shape2, shape3):
    t = Tester()
    a = np.full(shape1, 0.3, np.float32)
    b = np.full(shape2, 0.2, np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.add(x1, x2)
    x4 = t.reshape(x3, shape3)
    x5 = t.mul(x4, 0.1)
    t.store_expect(x5, 0.05)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reduce_lead_dim():
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [128, 1]).astype(np.float32)
    x = t.load(a)
    x = t.reshape(x, [1, 128])
    y = t.sum(x, [1], True)
    t.store_expect(y, np.sum(a, (0,), keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize('shape1, shape2', [([2, 3, 64], [2, 6, 32]),
                                            ([1000, 4], [100, 40]),  # lead dim
                                            ])
def test_reshape(shape1, shape2):
    t = Tester()
    a = np.full(shape1, 0.3, np.float32)
    x = t.load(a)
    y = t.mul(x, 0.6)
    z = t.reshape(y, shape2)
    r = t.sqrt(z)
    t.store_expect(r, 0.42426)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize('shape1, shape2', [([2, 3, 64], [2, 6, 32]), ([128], [128, 1])])
def test_reshape_direct_store(shape1, shape2):
    t = Tester()
    a = np.full(shape1, 0.3, np.float32)
    x = t.load(a)
    y = t.mul(x, 0.6)
    z = t.reshape(y, shape2)
    t.store_expect(z, 0.18)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
@pytest.mark.parametrize('shape1, shape2', [
    ([1000, 4], [100, 40]),  # 2 -> 2
    ([1000, 4], [20, 5, 40]),  # 2 -> 3
    ([3, 1000, 4], [3, 20, 5, 40]),  # 2-> 3, begin > 0
    ([3, 20, 5, 4, 2], [3, 10, 40, 2]),  # 3 -2, begin > 0, size <  end
    ([3, 2, 10, 64], [6, 10, 16, 4]),  # range 2
    ([3, 2, 1, 10], [6, 1, 10]),  # back 1: 1, 1
    ([3, 2, 1, 1, 10], [6, 1, 10]),  # back1: 2, 1
    ([3, 2, 1, 10], [6, 1, 1, 10])  # back1: 1, 2
])
def test_reshape_broker_shapes(shape1, shape2):
    t = Tester()
    a = np.random.normal(0.0, 1.0, shape1).astype(np.float32)
    x = t.load(a)
    y = t.mul(x, 0.6)
    z = t.reshape(y, shape2)
    r = t.add(z, 0.1)
    t.store_expect(r, (a * 0.6).reshape(shape2) + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape_a, shape_b, shape_c, red_dims', [
    ([10, 64], [10, 1], [2, 5, 64], (1,)),  # forward no split of elem
    ([10, 64], [10, 1], [10, 4, 16], (2,)),  # forward no split of broad
    ([10, 4, 64], [10, 4, 1], [10, 32, 8], (0,)),  # forward split
    ([2, 10, 64], [64], [20, 64], (1,)),  # forward dim broadcast
    ([10, 4, 64], [10, 4, 1], [10, 32, 8], (1,))  # splitcodegen: forward+backward split
])
@pytest.mark.gpu
def test_reshape_broker_prop(shape_a, shape_b, shape_c, red_dims):
    ''' add(a, b) -> reshape(c) - > reduce(dims) '''
    t = Tester()
    a = np.random.normal(0.0, 0.3, shape_a).astype(np.float32)
    b = np.random.normal(0.0, 0.2, shape_b).astype(np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.add(x1, x2)
    x4 = t.reshape(x3, shape_c)
    x5 = t.sum(x4, red_dims, True)
    t.store_expect(x5, np.sum((a + b).reshape(shape_c), axis=red_dims, keepdims=True), 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_matmul_post_fusion_forward():
    t = Tester("mix")
    shape_a = [1020, 3072]
    shape_b = [3072, 3072]
    ax = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    e = t.reshape(c, (2, 510, 48, 64))
    expect = np.reshape(np_c, (2, 510, 48, 64))
    t.store_expect(e, np.reshape(expect, (2, 510, 48, 64)), 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, shape_c', [
    ([1, 8, 2048, 512], [1, 8, 512, 128], [8, 2048, 128]),
    ([2048, 512], [512, 512], [2048, 1, 512]),
    ([2048, 512], [512, 512], [1024, 1024]),
    ([2048, 512], [512, 512], [1, 2048, 512, 1, 1]),
    ([2048, 512], [512, 512], [2048 * 512]),
])
def test_matmul_post_fusion_sload(shape_a, shape_b, shape_c):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    cx = Tester.fast_random_normal(0, 0.01, shape_c).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    cc = t.load(cx)
    c = t.matmul(a, b, False, False)
    e = t.reshape(c, shape_c)
    f = t.add(e, cc)
    t.store_expect(f, np.reshape(np_c, shape_c) + cx, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, shape_c, shape_d', [
    ([512, 1024], [1024, 768], [512, 1, 768], [768]),
    ([512, 1024], [1024, 768], [4, 128, 768], [768]),
])
def test_matmul_post_fusion_broadcast(shape_a, shape_b, shape_c, shape_d):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    dx = Tester.fast_random_normal(0, 0.01, shape_d).astype(np.float16)
    x0 = t.matmul(t.load(ax), t.load(bx), False, False)
    x1 = t.reshape(x0, shape_c)
    x2 = t.add(x1, t.load(dx))
    expect = np.reshape(np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16), shape_c) + dx
    t.store_expect(x2, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reshape_expand_dims():
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [128]).astype(np.float32)
    x = t.load(a)
    x = t.reshape(x, [1, 128])
    b = np.random.normal(-0.5, 0.5, [100, 128]).astype(np.float32)
    y = t.add(x, t.load(b))
    t.store_expect(y, np.reshape(a, [1, 128]) + b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reshape_share_domain():
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [128]).astype(np.float32)
    x0 = t.reshape(t.load(a), [1, 128])
    b = np.random.normal(-0.5, 0.5, [100 * 128]).astype(np.float32)
    x1 = t.reshape(t.load(b), [100, 128])
    x2 = t.mul(x0, x1)
    t.store_expect(x2, np.reshape(a, [1, 128]) * np.reshape(b, [100, 128]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reshape_fallback_sort():
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [4, 1, 1, 512]).astype(np.float32)
    b = np.random.normal(-0.5, 0.5, [4, 64, 512]).astype(np.float32)
    x0 = t.load(a)
    x1  = t.add(x0, 0.01)
    x2 = t.load(b)
    x3 = t.reshape(x2, [4, 64, 1, 512])
    x4 = t.reshape(x2, [4, 1, 64, 512])
    x5 = t.add(x3, x4)
    x6 = t.mul(x1, x5)
    t.store_expect(x6, (a + 0.01) * (np.reshape(b, [4, 64, 1, 512]) + np.reshape(b, [4, 1, 64, 512])))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reshape_before_after_connect():
    t = Tester()
    a = np.random.normal(0.0, 0.03, [10, 1, 1000]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.1)
    x2 = t.reshape(x1, [1, 10, 1000])
    x3 = t.mul(x2, x1)
    t.store_expect(x3, (a + 0.1).reshape([1, 10, 1000]) * (a + 0.1))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_auto_spec_reshape_cross_dim():
    t = Tester()
    a = np.random.normal(0.0, 0.03, [10, 1000]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.reshape(x0, [1, 10, 1000])
    x2 = t.reshape(x0, [10, 1, 1000])
    x3 = t.add(x1, x2)
    t.store_expect(x3, a.reshape([1, 10, 1000]) + a.reshape([10, 1, 1000]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reshape_expand_outmost_dim():
    t = Tester()
    a = np.random.normal(0.0, 0.01, [1, 10, 1, 200]).astype(np.float32)
    b = np.random.normal(0.0, 0.01, [10, 20, 200]).astype(np.float32)
    x0 = t.reshape(t.load(a), [10, 1, 200])
    x1 = t.add(x0, t.load(b))
    t.store_expect(x1, a.reshape([10, 1, 200]) + b)
    assert (t.run_check())
