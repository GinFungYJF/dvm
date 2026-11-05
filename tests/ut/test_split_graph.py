# Copyright 2025 Huawei Technologies Co., Ltd
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
@pytest.mark.parametrize('mode', ["split", "split:unify_ws", "split:dyn", "split:dyn,unify_ws"])
def test_split_extern_code(mode):
    t = Tester(mode)
    ax = np.random.normal(0, 0.1, [512, 512]).astype(np.float32)
    bx = np.full([1, 512], 0.01, np.float32)
    a = t.load([512, 512], "float32")
    b = t.load([1, 512], "float32")
    x = a
    for _ in range(200):
        x = t.add(x, b)
    out = t.store(x)
    for offset in (0.0, 0.1):
        data = ax + offset
        t.input(a, data)
        t.input(b, bx)
        t.run()
        assert (t.check(out, data + 0.01 * 200))
        t.reset()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ["split", "split:unify_ws"])
def test_split_static_vec(mode):
    t = Tester(mode)
    in_shape, red_dims = [10, 4096], (0,)
    a = np.random.normal(0, 0.1, in_shape).astype(np.float16)
    x = t.load(a)
    x = t.add(x, 0.1)
    t.store_expect(x, a + 0.1)
    x = t.cast(x, "float32")
    x0 = t.sum(x, red_dims, True)
    x = t.cast(x0, "float16")
    out1 = t.mul(x, 1.5)
    b = np.random.normal(0, 0.1, in_shape).astype(np.float32)
    out2 = t.add(x0, t.load(b))
    res_x0 = np.sum((a + 0.1).astype(np.float32), red_dims, keepdims=True)
    t.store_expect(out2, res_x0 + b)
    t.store_expect(out1, res_x0.astype(np.float16) * 1.5)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["split", "split:unify_ws"])
def test_split_static_matmul_with_bias(mode):
    t = Tester(mode)
    x_data = np.random.normal(0, 0.1, [1000, 3333]).astype(np.float16)
    y_data = np.random.normal(0, 0.1, [3333, 1024]).astype(np.float16)
    bias_data = np.random.normal(0, 0.1, [1024]).astype(np.float16)
    x = t.load(x_data)
    y = t.load(y_data)
    z = t.load(bias_data)
    x = t.mul(x, 0.5)
    x = t.matmul(x, y, False, False, z)
    x = t.add(x, 0.1)
    res = np.matmul((x_data * 0.5).astype(np.float32), y_data.astype(np.float32)).astype(np.float16) + bias_data + 0.1
    t.store_expect(x, res)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ["split:dyn", "split:dyn,unify_ws"])
def test_split_dyn_vec(mode):
    t = Tester(mode)
    x = t.load([-1], "float16")
    y = t.load([-1], "float16")
    z = t.add(x, y)
    z = t.cast(z, "float32")
    ref = t.int_array()
    z = t.sum(z, ref, True)
    z = t.cast(z, "float16") 
    z = t.mul(z, 1.5)
    out = t.store(z)
    cases = [[[10, 4096], [10, 4096], (0,)],
             [[100, 1], [100, 4096], (1,)],
             [[16, 4, 128], [16, 1, 128], (2,)]]
    for shape1, shape2, dims in cases:
        x_data = np.random.normal(0, 0.1, shape1).astype(np.float16)
        y_data = np.random.normal(0, 0.1, shape2).astype(np.float16)
        t.input(x, x_data) 
        t.input(y, y_data) 
        ref.update(dims)
        t.run()
        res = np.sum((x_data + y_data).astype(np.float32), dims, keepdims=True).astype(np.float16) * 1.5
        assert(t.check(out, res))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["split:dyn", "split:dyn,unify_ws"])
def test_split_dyn_matmul(mode):
    t = Tester(mode)
    x = t.load([-1, -1], "float16")
    y = t.load([-1, -1], "float16")
    z = t.load([-1, -1], "float16")
    a = t.mul(x, 0.5)
    c = t.matmul(a, y, False, False)
    c = t.add(c, z)
    out = t.store(c)
    iterations = [[[256, 128], [128, 256], [256, 256]],
                  [[444, 3333], [3333, 1111], [1, 1111]],
                  [[256, 128], [128, 512], [256, 1]]]
    for x_shape, y_shape, z_shape in iterations:
        x_data = np.random.normal(0, 0.1, x_shape).astype(np.float16)
        y_data = np.random.normal(0, 0.1, y_shape).astype(np.float16)
        z_data = np.random.normal(0, 0.1, z_shape).astype(np.float16)
        t.input(x, x_data)
        t.input(y, y_data)
        t.input(z, z_data)
        t.run()
        res = np.matmul((x_data * 0.5).astype(np.float32), y_data.astype(np.float32)).astype(np.float16) + z_data
        assert (t.check(out, res, 2e-3))

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_dyn_atomic_reloc():
    t = Tester("split:dyn")
    x = t.load([-1, -1], "float32")
    y = t.sum(x, [0], True)
    out = t.store(y)
    cases = [[4, 4096], [4, 512]]
    for shape in cases:
        x_data = np.random.normal(0, 0.1, shape).astype(np.float32)
        t.input(x, x_data)
        t.run()
        assert(t.check(out, np.sum(x_data, (0,), keepdims=True)))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('x_shape, y_shape, mode', [
    [[1024, 2048], [2048, 2048], "split"],
    [[1024, 2048], [2048, 2048], "split:dyn"],
    [[1024, 40960], [40960, 4096], "split"],
])
def test_split_matmul_store(x_shape, y_shape, mode):
    t = Tester(mode)
    x_data = t.fast_random_normal(0, 0.1, x_shape).astype(np.float16)
    y_data = t.fast_random_normal(0, 0.1, y_shape).astype(np.float16)
    x = t.load(x_data)
    y = t.load(y_data)
    x = t.mul(x, 0.5)
    x = t.matmul(x, y, False, False)
    res = np.matmul((x_data * 0.5).astype(np.float32), y_data.astype(np.float32)).astype(np.float16)
    t.store_expect(x, res)
    x = t.add(x, 0.1)
    t.store_expect(x, res + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["split:unify_ws", "split:dyn,unify_ws"])
def test_split_ws_align(mode):
    t = Tester(mode)
    b_data = np.random.normal(0, 0.1, [1979]).astype(np.float32)
    b = t.load(b_data)
    b = t.cast(b, "float16")
    x_data = t.fast_random_normal(0, 0.1, [4, 512]).astype(np.float16)
    y_data = t.fast_random_normal(0, 0.1, [512, 1979]).astype(np.float16)
    x = t.load(x_data)
    y = t.load(y_data)
    z = t.matmul(x, y, False, False, b)
    z = t.add(z, 0.1)
    res = np.matmul(x_data.astype(np.float32), y_data.astype(np.float32)).astype(np.float16) + b_data + 0.1
    t.store_expect(z, res)
    assert (t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_ws_reloc():
    t = Tester("split:unify_ws")
    in_shape, dims = [30, 4000], (0,)
    t.set_deterministic(True)
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x0 = t.load(a)
    x = t.sum(t.mul(x0, 1.5), dims, True)
    x = t.add(x, 0.1)
    t.store_expect(x, np.sum(a * 1.5, dims, keepdims=True) + 0.1)
    assert(t.run_check())
    t.set_deterministic(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_tuner():
    Tester.set_online_tuning(True)
    t = Tester("split")
    shape = [4096, 4096]
    a = np.random.normal(0, 0.01, shape).astype(np.float16)
    b = np.random.normal(0, 0.01, shape).astype(np.float16)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32))
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    t.store_expect(x2, expect, 1e-3)
    assert (t.run_check())
    Tester.set_online_tuning(False)
