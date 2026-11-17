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

import copy
import pytest
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dim', [64, 256, 251])
@pytest.mark.gpu
def test_reduce_x(dim):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [10, dim]).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, [1], True)
    t.store_expect(y, np.sum(a, axis=(1,), keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('r_dim, s_dim', [(6, 64), (6, 60), (2, 383 * 16)])
@pytest.mark.gpu
def test_reduce_y(r_dim, s_dim):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [10, r_dim, s_dim]).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, [1], True)
    t.store_expect(y, np.sum(a, (1,), keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('i, j', [(32, 65), (31, 89), (31, 90), (31, 93), (30, 82), (30, 83), (30, 85)])
@pytest.mark.gpu
def test_reduce_i_j(i, j):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [3, 1280, i, j]).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, [0, 2, 3], False)
    expect = np.sum(a, axis=(0, 2, 3), keepdims=False)
    t.store_expect(y, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dims', [(1, 2, 3), (1, 3, 5), (1, 2, 5, 6), (5, 6), (0, 1, 2, 3, 4, 5, 6)])
@pytest.mark.gpu
def test_reduce_normalize(dims):
    t = Tester()
    shape = [6, 4, 2, 5, 1, 3, 8]
    a = np.random.normal(-0.5, 0.5, shape).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, dims, True)
    t.store_expect(y, np.sum(a, dims, keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dim', [1001, 1024])
@pytest.mark.gpu
def test_reduce_atomic(dim):
    t = Tester()
    a = np.full([64, dim], 0.01, np.float32)
    x = t.load(a)
    y = t.sum(x, [0], True)
    t.store_expect(y, 0.01 * 64)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dvm_op, np_op", [(pytest.param(Tester.sum, np.sum, marks=pytest.mark.gpu)), (Tester.min, np.min), ((Tester.max, np.max))])
@pytest.mark.parametrize('in_shape, dims', [[[511, 1024], (1,)], [[521, 1024], (1,)],
                                            [[35053], (0,)],
                                            [[120, 1], (0,)],  # lead 1 not include
                                            [[1, 4, 120, 136], (0, 2, 3)],  # reduce with 1
                                            [[120, 20, 1], (2,)],  # reduce one range with 1
                                            [[1, 1, 1, 1, 5, 1, 1, 300, 1, 100], (0, 2, 4, 6, 8)],  # opensora bugfix
                                            [[3, 4, 120, 136], (0, 2, 3)],  # from sdxl:  two atomic dim range
                                            [[3, 1280, 2, 2], (0, 2, 3)],  # tile_size_ bugfix
                                            [[128, 703], (0,)],  # reducey with tail=1: 40 blockdim: 703=18*39+1
                                            [[1, 1024, 14, 14], (0, 2, 3)],  # from video chat: output axis not divided
                                            [[11, 6000], (0,)],  # reducey red_size = 1
                                            [[324, 9, 80], (0, 2)]
                                            ])
def test_reduce(dvm_op, np_op, in_shape, dims):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x = t.load(a)
    y = dvm_op(t, x, dims, True)
    res = np_op(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dvm_op, np_op", [(Tester.min, np.min), ((Tester.max, np.max))])
@pytest.mark.parametrize('in_shape, dims', [[[511, 1024], (1,)], [[521, 1024], (1,)],
                                            [[35053], (0,)],
                                            [[120, 1], (0,)],  # lead 1 not include
                                            [[1, 4, 120, 136], (0, 2, 3)],  # reduce with 1
                                            [[120, 20, 1], (2,)],  # reduce one range with 1
                                            [[1, 1, 1, 1, 5, 1, 1, 300, 1, 100], (0, 2, 4, 6, 8)],  # opensora bugfix
                                            [[3, 4, 120, 136], (0, 2, 3)],  # from sdxl:  two atomic dim range
                                            [[3, 1280, 2, 2], (0, 2, 3)],  # tile_size_ bugfix
                                            [[128, 703], (0,)],  # reducey with tail=1: 40 blockdim: 703=18*39+1
                                            [[1, 1024, 14, 14], (0, 2, 3)],  # from video chat: output axis not divided
                                            [[11, 6000], (0,)],  # reducey red_size = 1
                                            [[324, 9, 80], (0, 2)]
                                            ])
def test_reduce_fp16(dvm_op, np_op, in_shape, dims):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float16)
    x = t.load(a)
    y = dvm_op(t, x, dims, True)
    res = np_op(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_x_tail():
    in_shape = [1949]
    dims = (0,)
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res)
    t.tile(0, 0, 122)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_y_tail():
    in_shape = [1949, 512]
    dims = (0,)
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    t.tile(1, 1, 122)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_fake_atomic():
    t = Tester()
    a = np.full([1, 640, 64, 64], 1.0, np.float32)
    x1 = t.load(a)
    x2 = t.sum(x1, [2, 3], True)
    x3 = t.sum(x2, [0], True)
    t.store_expect(x3, 4096.0)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('in_shape, dims, tile_depth', [
    [[400, 4096], (0,), 1],  # reduce y
    [[40000], (0,), 0],  # reduce x
    [[65, 40, 4096], (1,), 2],  # order 1
    [[8, 3, 7, 4, 1024], (1, 3), 3],  # order 2
    [[8, 3, 7, 4, 4096], (1, 3), 4],  # order 3
    [[8, 3, 6, 4, 3, 4, 1024], (1, 3, 5), 5],  # order 4
])
def test_atomic_determ(in_shape, dims, tile_depth):
    t = Tester()
    t.set_deterministic(True)
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    out = t.store(y)
    if tile_depth > 0:
        for i in range(tile_depth):
            idx = len(in_shape) - i - 1
            t.tile(idx, idx, in_shape[i])
    t.run()
    expect = copy.deepcopy(t.output(out))
    t.run()
    output = t.output(out)
    t.set_deterministic(False)
    assert (np.allclose(output, res, rtol=1e-4, atol=1e-4, equal_nan=True))
    assert (np.allclose(output, expect, rtol=1e-8, atol=1e-8, equal_nan=True))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_determ():
    t = Tester()
    in_shape, dims = [64, 8, 256, 8], (0, 2)
    t.set_deterministic(True)
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x0 = t.load(a)
    x = t.sum(t.mul(x0, 1.5), dims, True)
    out_x = t.store(x)
    y = t.sum(t.mul(x0, 0.8), dims, True)
    out_y = t.store(y)
    t.tile(3, 3, 64)
    t.tile(2, 2, 8)
    t.tile(1, 1, 16)
    t.run()
    e_x = copy.deepcopy(t.output(out_x))
    e_y = copy.deepcopy(t.output(out_y))
    t.run()
    res_x = t.output(out_x)
    res_y = t.output(out_y)
    t.set_deterministic(False)
    assert (np.allclose(res_x, np.sum(a * 1.5, dims, keepdims=True), rtol=1e-4, atol=1e-4, equal_nan=True))
    assert (np.allclose(res_y, np.sum(a * 0.8, dims, keepdims=True), rtol=1e-4, atol=1e-4, equal_nan=True))
    assert (np.allclose(res_x, e_x, rtol=1e-8, atol=1e-8, equal_nan=True))
    assert (np.allclose(res_y, e_y, rtol=1e-8, atol=1e-8, equal_nan=True))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_elemwise_reduce():
    t = Tester()
    a = np.full([32, 2048], 0.01, np.float32)
    x = t.load(a)
    x = t.add(x, x)
    y = t.sum(x, [1], True)
    t.store_expect(y, 0.02 * 2048)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_max_min_atomic_determ():
    t = Tester()
    t.set_deterministic(True)
    a = np.random.normal(-0.5, 0.5, [65, 40, 4096]).astype(np.float32)
    x = t.load(a)
    y = t.max(x, (1,), True)
    res = np.max(a, (1,), keepdims=True)
    out = t.store(y)
    t.run()
    expect = copy.deepcopy(t.output(out))
    t.run()
    output = t.output(out)
    t.set_deterministic(False)
    assert (np.allclose(output, res, rtol=1e-4, atol=1e-4, equal_nan=True))
    assert (np.allclose(output, expect, rtol=1e-8, atol=1e-8, equal_nan=True))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dims', [(0,), (0, 2)])
def test_mixed_reduce_determ(dims):
    t = Tester()
    t.set_deterministic(True)
    try:
        a = np.ones((128, 9, 257), dtype=np.float32)
        a[0] = 2
        a[1] = -2
        x = t.load(a)
        t.store_expect(t.sum(x, dims, False), np.sum(a, axis=dims), 0)
        t.store_expect(t.max(x, dims, False), np.max(a, axis=dims), 0)
        t.store_expect(t.min(x, dims, False), np.min(a, axis=dims), 0)
        assert t.run_check()
    finally:
        t.set_deterministic(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('in_shape, dims', [[[1024, 1000], (0, 1)],
                                            [[3, 4, 120, 136], (0, 2, 3)],
                                            [[3, 1280, 2, 2], (0, 2, 3)]])
@pytest.mark.gpu
def test_reduce_insert_accumulate(in_shape, dims):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x = t.load(a)
    x = t.copy(x)
    y = t.sum(x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_reduce_insert_accumulate_rank_3():
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [32, 12, 1, 1]).astype(np.float32)
    x = t.load(a)
    x = t.broadcast(x, [32, 12, 111, 111])
    x = t.add(x, 0.1)
    y = t.sum(x, [0, 2, 3], False)
    expect = np.sum(np.broadcast_to(a, [32, 12, 111, 111]) + 0.1, axis=(0, 2, 3), keepdims=False)
    t.store_expect(y, expect)
    t.store(x)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape, num, factor', [
    [[256 * 50 + 100], 51, 256],  # tail clear
    [[249 * 50 + 223], 51, 249],  # body and tail clear
    [[501 * 20], 20, 501]  # body clear
])
def test_reduce_x_clean_pad(shape, num, factor):
    t = Tester()
    a = np.random.normal(0.0, 1.0, shape).astype(np.float32)
    x = t.load(a)
    y = t.sum(x, (0,), True)
    expect = np.sum(a, axis=(0,), keepdims=True)
    t.store_expect(y, expect, 1e-4)
    t.tile(0, 0, num, factor)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_x_clean_pad_nolead():
    t = Tester()
    a0 = np.random.randn(8002, 2).astype(np.float32)
    a1 = np.random.randn(8002, 1).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.add(x0, x1)
    x3 = t.sum(x2, [0, 1], False)
    t.store_expect(x3, np.sum(a0 + a1, axis=(0, 1)), eps=1e-4)
    t.tile(1, 1, 8)
    assert (t.run_check())
