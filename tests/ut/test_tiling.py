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
@pytest.mark.parametrize('dim', [1024, 1023])
def test_tiling_broadcast_part(dim):
    t = Tester()
    a = np.full([1, dim], 0.5, np.float32)
    b = np.full([512, dim], 0.2, np.float32)
    x = t.load(a)
    y = t.load(b)
    z = t.broadcast(x, [512, dim])
    r = t.add(z, y)
    t.store_expect(r, 0.7)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_part2():
    t = Tester()
    b = np.random.normal(0, 1, [16000, 39, 1]).astype(np.float32)
    y = t.load(b)
    z = t.broadcast(y, [16000, 39, 80])
    expect = np.broadcast_to(b, (16000, 39, 80))
    out = t.store_expect(z, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["split", "eager:unify_ws"])
def test_tiling_broadcast_prime_lead(mode):
    n = 65537
    a = np.broadcast_to(np.arange(n, dtype=np.float32) * 0.125, (3, n)).copy()
    b = np.array([[0.5], [1.0], [2.0]], dtype=np.float32)
    t = Tester(mode)
    result = t.sub(t.load(a), t.load(b))
    t.store_expect(result, a - b)
    assert t.run_check()
    assert "vmain.parallel()" in t.das()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_all():
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 64, 1024]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [2, 96, 64, 1024])
    expect = np.broadcast_to(a0, (2, 96, 64, 1024))
    t.store_expect(z0, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_round_2():
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 512]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [2, 3, 512])
    expect = np.broadcast_to(a0, (2, 3, 512))
    t.store_expect(z0, expect)
    t.tile(2, 2, 2)
    t.tile(1, 1, 3)
    t.tile(0, 0, 4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_round_3():
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, 512]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [2, 3, 4, 6, 512])
    expect = np.broadcast_to(a0, (2, 3, 4, 6, 512))
    t.store_expect(z0, expect)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_round_4():
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, 512]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [2, 3, 4, 5, 512])
    expect = np.broadcast_to(a0, (2, 3, 4, 5, 512))
    t.store_expect(z0, expect)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 5)
    t.tile(0, 1, 2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_broadcast_multi():
    t = Tester()
    a0 = np.full((1), 1.1, np.float32)
    b0 = np.full((1), 1.1, np.float32)
    la0 = t.load(a0)
    lb0 = t.load(b0)
    x = t.div(la0, lb0)
    y = t.broadcast(x, [5120])
    c0 = np.full([5120], 1.1, np.float32)
    lc0 = t.load(c0)
    z = t.mul(lc0, y)
    r = t.mul(z, -1.0)
    s = t.reshape(r, [5120, 1])
    l = t.broadcast(s, [5120, 21128])
    d0 = np.full([5120, 21128], 1, np.float32)
    ld0 = t.load(d0)
    m = t.mul(l, ld0)
    t.store_expect(m, -1.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiling_align():
    t = Tester()
    a = np.full([1, 1], 0.04, np.float32)
    x = t.load(a)
    y = t.sqrt(x)
    out = t.store_expect(y, 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_shape():
    t = Tester()
    x = t.full(0.2, [], "float32")
    x = t.mul(x, 0.3)
    t.store_expect(x, 0.06)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dim1, dim2', [([2, 8], [16]), ([16], [2, 8]), ([2, 16], [2, 4, 4])])
def test_multi_dom(dim1, dim2):
    bdim = [1024 * 10]
    t = Tester()
    a = np.full(dim1 + [1], 0.3, np.float32)
    x = t.load(a)
    x = t.broadcast(x, dim1 + bdim)
    y = t.mul(x, 0.6)
    z = t.reshape(y, dim2 + bdim)
    r = t.sqrt(z)
    t.store_expect(r, 0.42426)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_choise_dom_op_with_shape():
    t = Tester()
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    a = t.load(g0)
    b = t.load(g1)
    c = t.add(a, b)
    t.store_expect(c, 0.2)
    d = t.add(a, 0.2)
    t.store_expect(d, 0.3)
    assert (t.run_check())
