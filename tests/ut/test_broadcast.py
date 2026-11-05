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
@pytest.mark.parametrize('type', [np.float16, np.float32, np.int32])
@pytest.mark.gpu
def test_broadcast_x(type):
    t = Tester()
    a = np.random.normal(0, 1, [16, 1]).astype(type)
    x = t.load(a)
    y = t.broadcast(x, [16, 2])
    t.store_expect(y, np.broadcast_to(a, [16, 2]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32, np.int32])
@pytest.mark.gpu
def test_broadcast_x_2(type):
    t = Tester()
    a = np.random.normal(0, 1, [16, 1]).astype(type)
    b = np.random.normal(0, 1, [16, 64]).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.broadcast(x, [16, 64])
    r = t.add(z, y)
    t.store_expect(r, a + b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32, np.int32])
@pytest.mark.gpu
def test_broadcast_y(type):
    t = Tester()
    a = np.random.normal(0, 1, [1, 32]).astype(type)
    x = t.load(a)
    y = t.broadcast(x, [8, 32])
    t.store_expect(y, np.broadcast_to(a, [8, 32]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_broadcast_multi(type):
    t = Tester()
    a = np.random.normal(0, 1, [10, 33, 64]).astype(type)
    b = np.random.normal(0, 1, [10, 1, 1]).astype(type)
    y = t.load(b)
    z = t.broadcast(y, [10, 33, 1])
    z = t.add(z, 0.1)
    z = t.broadcast(z, [10, 33, 64])
    x = t.load(a)
    s = t.add(x, z)
    t.store_expect(s, a + b + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_large_shape():
    src_shape = [20, 30, 1, 50]
    dst_shape = [20, 30, 40, 50]
    t = Tester()
    a = np.random.random(src_shape).astype(np.float32)
    np_res = np.broadcast_to(a, dst_shape)
    x = t.load(a)
    y = t.broadcast(x, dst_shape)
    t.store_expect(y, np_res)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("src_shape,dst_shape", [
    ((8, 1, 257, 1), (8, 5, 257, 129)),
    ((1, 9, 1, 257), (6, 9, 33, 257)),
    ((4, 1, 8, 1, 129), (4, 3, 8, 17, 129)),
])
def test_broadcast_int64(src_shape, dst_shape):
    t = Tester()
    a = np.random.randint(0x10000000, 0x2000000000, size=src_shape, dtype=np.int64)
    x = t.load(a)
    y = t.broadcast(x, dst_shape)
    t.store_expect(y, np.broadcast_to(a, dst_shape))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcastx_repeat_overflow():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [1, 1]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [184965, 1])
    expect = np.broadcast_to(a0, (184965, 1))
    t.store_expect(z0, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcastx_unalign_load():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [3, 4, 1]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [3, 4, 5])
    expect = np.broadcast_to(a0, (3, 4, 5))
    t.store_expect(z0, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcastx_bf16():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [30, 40, 1]).astype(np.float32)
    la0 = t.load(a0, "bfloat16")
    z0 = t.broadcast(la0, [30, 40, 5])
    expect = np.broadcast_to(a0, (30, 40, 5))
    t.store_expect(z0, expect, 5e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_s():
    t = Tester()
    x = t.full(0.2, [2, 64], "float32")
    t.store_expect(x, 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('scalar', [0.0, 0.2, True])
@pytest.mark.gpu
def test_broadcast_s_bool(scalar):
    t = Tester()
    x = t.full(scalar, [2, 64], "bool")
    t.store_expect(x, np.full([2, 64], scalar, dtype=np.bool_))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(1024, 1025), (56, 10888)])
def test_broadcast_s_int64_scalar_ref(shape):
    t = Tester('vector:dyn')
    x = t.load([-1], "int64")
    s = t.scalar(dvm.int64)
    y = t.full(s, shape, "int64")
    z = t.add(x, y)
    out = t.store(z)
    a = np.random.randint(low=-0x2000000000, high=0x2000000000, size=shape, dtype=np.int64)
    t.input(x, a)
    for scalar in [1, 0x100000000, -0x100000000]:
        s.update(scalar)
        t.run()
        assert t.check(out, np.add(a, scalar))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('lead_dim', [1024, 511])
@pytest.mark.gpu
def test_broadcast_store_rank_1(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.add(la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 96, lead_dim])
    e2 = np.broadcast_to(e1, (2, 96, lead_dim))
    t.store_expect(z0, e2)
    t.tile(2, 2, 2)
    t.tile(1, 1, 48)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('lead_dim', [1024, 511])
@pytest.mark.gpu
def test_broadcast_store_rank_2(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4 * lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.add(la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4 * lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4 * lead_dim))
    t.store_expect(z0, e2)
    t.tile(2, 2, 2)
    t.tile(1, 1, 3)
    t.tile(0, 0, 4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('lead_dim', [1024, 511])
@pytest.mark.gpu
def test_broadcast_store_rank_3(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.add(la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4, 6, lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4, 6, lead_dim))
    t.store_expect(z0, e2)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('lead_dim', [1024, 511])
@pytest.mark.gpu
def test_broadcast_store_rank_4(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, 2 * lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.add(la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4, 5, 2 * lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4, 5, 2 * lead_dim))
    t.store_expect(z0, e2)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 5)
    t.tile(0, 1, 2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcast_bool():
    t = Tester()
    a0 = np.array([[True]]).astype(bool)
    x0 = t.load(a0)
    x1 = t.broadcast(x0, [2, 3])
    t.store_expect(x1, True)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_broadcastx_repeat_unalign():
    t = Tester()
    a = np.random.normal(0, 1, [555555, 1]).astype(np.float16)
    x = t.load(a)
    y = t.broadcast(x, [555555, 16])
    t.store_expect(y, np.broadcast_to(a, [555555, 16]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_broadcastx_removepad():
    t = Tester()
    a = np.random.normal(0, 0.1, [40, 20, 17, 1]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.broadcast(x0, [40, 20, 17, 32])
    x2 = t.sum(x1, (1,), True)
    t.store_expect(x2, np.sum(np.broadcast_to(a, [40, 20, 17, 32]), (1,), keepdims=True))
    t.tile(3, 3, 40)
    assert(t.run_check())
 