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

import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_bytecode_reserve():
    """Reserve bytecode space for multiple deterministic reductions."""
    t = Tester("vector", use_pass_opt=True)
    t.set_deterministic(True)
    rng = np.random.default_rng(0)
    shapes = [(1, 128, 128), (128, 128), (1, 128, 128), (128,)]
    a, b, c, weight = [rng.normal(0, 0.5, shape).astype(np.float32) for shape in shapes]
    x0, x1, x2, x3 = [t.load(data) for data in (a, b, c, weight)]
    sum_a = t.sum(t.reshape(x0, [128, 128]), [0], True)
    added = t.add(x0, t.reshape(x1, [1, 128, 128]))
    sum_added = t.sum(added, [0, 1], True)
    sum_product = t.sum(t.mul(added, x2), [0, 1], True)
    weighted = t.mul(added, t.broadcast(x3, [1, 128, 128]))
    expected_added = a + b.reshape(1, 128, 128)
    t.store_expect(sum_a, a.reshape(128, 128).sum(axis=0, keepdims=True), eps=1e-4)
    t.store_expect(sum_added, expected_added.sum(axis=(0, 1), keepdims=True), eps=1e-4)
    t.store_expect(sum_product, (expected_added * c).sum(axis=(0, 1), keepdims=True), eps=1e-4)
    t.store_expect(weighted, expected_added * weight.reshape(1, 1, 128), eps=1e-4)
    result = t.run_check()
    t.set_deterministic(False)
    assert result


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_in_multi_out():
    np.random.seed(1)
    t = Tester()
    a0 = np.abs(np.random.normal(0, 1, [8, 32000]).astype(np.float32))
    a1 = np.abs(np.random.normal(0, 1, [8, 1]).astype(np.float32)) + 1e-6
    a2 = np.random.normal(0, 1, [8, 32000]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.div(x0, x1)
    expect0 = a0 / a1
    t.store_expect(y0, expect0)
    y1 = t.add(y0, 1e-24)
    y2 = t.log(y1)
    x2 = t.load(a2)
    y3 = t.mul(y2, x2)
    y4 = t.mul(y3, -1.0)
    expect1 = -(np.log(expect0 + 1e-24) * a2)
    t.store_expect(y4, expect1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_backward_sync_overlap():
    np.random.seed(1)
    t = Tester()
    a0 = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    a1 = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.load(a1)
    b = t.sub(x0, x1)
    a = t.sub(x0, x1)
    f = t.sub(x1, x0)
    c = t.store(a)
    g = t.store(f)
    h = t.store(b)
    t.run_check()
    assert np.allclose(t.output(h), a0 - a1, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_event_overflow():
    shape = [512]
    t = Tester()
    loads = []
    x = t.load(np.full(shape, 0.1, np.float32))
    for i in range(16):
        loads.append(t.load(np.full(shape, 0.2, np.float32)))
    result = 0.1
    for i in range(16):
        x = t.add(x, loads[i])
        result += 0.2
        t.store_expect(x, result)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_inplace_anti_dep():
    t = Tester()
    x = t.load([32, 1024], "float16")
    x2 = t.sqrt(x)
    x3 = t.abs(x)
    x4 = t.add(x2, x3)
    x5 = t.reciprocal(x3)
    x6 = t.add(x4, x5)
    t.store(x6)
    t.codegen()
    assert (t.das().count("bar") == 3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape():
    t = Tester()
    a = np.full([10, 0, 2], 0.1, np.float32)
    x = t.load(a)
    y = t.mul(x, 0.1)
    out = t.store(y)
    t.run()
    assert(out.shape() == (10, 0, 2))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_clean():
    t = Tester()
    a = np.full([10, 0, 2], 0.01, np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.1)
    x3 = t.sum(x2, (1,), True)
    out = t.store_expect(x3, 0.0)
    assert(t.run_check())
    assert(out.shape() == (10, 1, 2))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zero_shape_dyn_shape():
    t = Tester("vector:dyn")
    x1 = t.load([-1], "float32")
    x2 = t.add(x1, 0.1)
    x3 = t.sum(x2, (1,), True)
    out = t.store(x3)

    d1 = np.full([10, 0, 20], 0.01, np.float32)
    t.input(x1, d1)
    t.run()
    assert(t.check(out, 0.0))
    assert(out.shape() == (10, 1, 20))

    d1 = np.full([0, 10, 20], 0.01, np.float32)
    t.input(x1, d1)
    t.run()
    assert(out.shape() == (0, 1, 20))

    d1 = np.full([10, 40, 20], 0.01, np.float32)
    t.input(x1, d1)
    t.run()
    assert(t.check(out, (0.01 + 0.1) * 40))
    assert(out.shape() == (10, 1, 20))
