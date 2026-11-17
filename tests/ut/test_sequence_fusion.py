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
@pytest.mark.gpu
def test_seq_vec_vec():
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    ax = np.full([1024, 128], 0.05, np.float32)
    a = t.load(ax)
    b = t.add(a, 0.2)
    c = t.mul(a, b)
    t.seq_add(Tester.K_VEC)
    d = t.mul(c, 0.7)
    t.store_expect(d, (ax + 0.2) * ax * 0.7)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_seq_vec_mix():
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    ax = np.random.normal(0, 1, [1024, 512]).astype(np.float16)
    a = t.load(ax)
    b = t.add(a, 0.02)
    c = t.mul(a, b)
    t.seq_add(Tester.K_MIX)
    fx = np.random.normal(0, 1, [512, 1024]).astype(np.float16)
    f = t.load(fx)
    g = t.matmul(c, f, False, False)
    h = t.cast(g, "float32")
    expect = np.matmul(((ax + 0.02) * ax).astype(np.float32), fx.astype(np.float32))
    t.store_expect(h, expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_seq_mix_vec():
    t = Tester("seq")
    t.seq_add(Tester.K_MIX)
    ax = np.random.normal(0, 1, [768, 512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512, 128 * 7]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    t.seq_add(Tester.K_VEC)
    g = t.add(d, 0.02)
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32)) + 0.02
    t.store_expect(g, expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_inplace_reuse():
    ax = np.full([1024], 0.05, np.float32)
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    a0 = t.load(ax)
    a1 = t.add(a0, 0.1)
    t.seq_add(Tester.K_VEC)
    b1 = t.add(a1, 0.1)
    t.seq_add(Tester.K_VEC)
    c1 = t.add(b1, 0.1)
    c2 = t.store_expect(c1, 0.05 + 0.3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_inplace_stage():
    ax = np.full([1024], 0.05, np.float32)
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    a0 = t.load(ax)
    a1 = t.cast(a0, "float16")
    t.seq_add(Tester.K_VEC)
    b1 = t.cast(a1, "float32")
    t.seq_add(Tester.K_VEC)
    c1 = t.cast(b1, "float16")
    t.seq_add(Tester.K_VEC)
    d1 = t.add(c1, 0.1)
    t.seq_add(Tester.K_VEC)
    e1 = t.add(d1, 0.1)
    t.seq_add(Tester.K_VEC)
    f1 = t.cast(e1, "float32")
    f2 = t.store_expect(f1, 0.05 + 0.2, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_inplace_multi():
    ax = np.full([1024], 0.05, np.float32)
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    a0 = t.load(ax)
    a1 = t.add(a0, 0.1)
    a2 = t.mul(a0, 0.5)
    t.seq_add(Tester.K_VEC)
    b1 = t.div(a1, 0.8)
    b2 = t.add(b1, a2)
    t.store_expect(b2, np.divide(ax + 0.1, 0.8) + ax * 0.5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_seq_workspace_reuse():
    t = Tester("seq")
    t.seq_add(Tester.K_MIX)
    ax = np.random.normal(0, 1, [512, 512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512, 512]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    t.seq_add(Tester.K_VEC)
    f1 = t.cast(d, "float16")
    t.seq_add(Tester.K_VEC)
    g1 = t.cast(f1, "float32")
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    g2 = t.store_expect(g1, expect, 1e-2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_seq_extern_code():
    t = Tester("seq")
    t.seq_add(Tester.K_MIX)
    ax = np.random.normal(0, 1, [512, 512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512, 512]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    t.seq_add(Tester.K_VEC)
    f = d 
    for i in range(200):
        f = t.add(f, 0.01)
        f = t.sub(f, 0.01)
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    g2 = t.store_expect(f, expect, 1e-2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_vec_reduce():
    np.random.seed(1)
    a0 = np.random.normal(-1, 1, [1, 2048, 5120]).astype(np.float32)
    a1 = np.random.normal(-1, 1, [5120]).astype(np.float32)
    e1 = a0 * a0
    e2 = np.sum(e1, (2,), keepdims=True)
    e3 = e2 * 0.000195313
    e4 = e3 + 1e-6
    e5 = 1.0 / np.sqrt(e4)
    e6 = a0 * e5
    e7 = e6 * a1

    t = Tester("seq", use_pass_opt=True)
    t.seq_add(Tester.K_VEC)
    x0 = t.load(a0, "bfloat16")
    y0 = t.cast(x0, "float32")
    o0 = t.store_expect(y0, a0, 1e-2)
    y1 = t.mul(y0, y0)
    y2 = t.sum(y1, (2,), True)

    t.seq_add(Tester.K_VEC)
    y3 = t.mul(y2, 0.000195313)
    y4 = t.add(y3, 1e-6)
    y5 = t.reciprocal(t.sqrt(y4))
    o5 = t.store_expect(y5, e5, 1e-2)

    t.seq_add(Tester.K_VEC)
    l0 = o0
    l5 = o5
    y6 = t.mul(l0, l5)
    x1 = t.load(a1)
    y7 = t.mul(y6, x1)
    y8 = t.cast(y7, "bfloat16")
    t.store_expect(y8, e7, 1e-2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_broadcast_store_reuse():
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    a0 = np.random.normal(-1, 1, [3000]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.01)

    t.seq_add(Tester.K_VEC)
    y1 = t.mul(x1, 0.8)
    a1 = np.random.normal(-1, 1, [10, 3000]).astype(np.float32)
    y2 = t.add(y1, t.load(a1))
    e1 = (a0 + 0.01) * 0.8
    t.store_expect(y1, e1)
    t.store_expect(y2, e1 + a1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_seq_reloc_depend():
    """ stage1 dup tiling reloc -> stage 0 output reuse reloc"""
    t = Tester("seq")
    t.seq_add(Tester.K_VEC)
    a0 = np.random.normal(-1, 1, [3, 18911]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.01)

    t.seq_add(Tester.K_VEC)
    y1 = t.mul(x1, 0.8)
    a1 = np.random.normal(-1, 1, [3, 1]).astype(np.float32)
    y2 = t.add(y1, t.load(a1))
    e1 = (a0 + 0.01) * 0.8
    t.store_expect(y1, e1)
    t.store_expect(y2, e1 + a1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_seq_vec_mix_dyn():
    t = Tester("seq:dyn")
    t.seq_add(Tester.K_VEC)
    x0 = t.load([-1], "float16")
    x1 = t.add(x0, 0.02)
    x2 = t.mul(x1, x0)
    t.seq_add(Tester.K_MIX)
    y0 = t.load([-1], "float16")
    y1 = t.matmul(x2, y0, False, False)
    y2 = t.cast(y1, "float32")
    t.seq_add(Tester.K_VEC)
    z0 = t.sub(y2, 0.01)
    out = t.store(z0)
    iterations = [[[512, 1024], [1024, 512]], [[768, 1024], [1024, 1024]]]
    for x0_shape, y0_shape in iterations:
        x0_data = np.random.normal(0, 1, x0_shape).astype(np.float16)
        y0_data = np.random.normal(0, 1, y0_shape).astype(np.float16)
        t.input(x0, x0_data)
        t.input(y0, y0_data)
        t.run()
        expect = np.matmul(((x0_data + 0.02) * x0_data).astype(np.float32), y0_data.astype(np.float32)) - 0.01
        t.check(out, expect, 1e-3)
