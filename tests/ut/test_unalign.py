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
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.gpu
def test_simd_unalign(type):
    t = Tester()
    a = np.full([16, 1], 0.5, type)
    b = np.full([16, 41], 0.2, type)
    x = t.load(a)
    y = t.load(b)
    r = t.add(x, y)
    t.store_expect(r, 0.7)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_tile_unalign(type):
    t = Tester()
    a = np.full([127, 1], 0.5, type)
    b = np.full([127, 256], 0.2, type)
    x = t.load(a)
    y = t.load(b)
    r = t.add(x, y)
    t.store_expect(r, 0.7)
    t.tile(1, 1, 8)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tile_unalign_loadstore2():
    t = Tester()
    a = np.full([127, 1], 0.5, np.float32)
    b = np.full([127, 249], 0.2, np.float32)
    x = t.load(a)
    y = t.load(b)
    r = t.add(x, y)
    t.store_expect(r, 0.7)
    t.tile(1, 1, 8)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.gpu
def test_tile_unalign_leaddim():
    t = Tester()
    a = np.full([4080], 0.2, np.float32)
    x = t.load(a)
    r = t.add(x, 0.3)
    t.store_expect(r, 0.5)
    t.tile(0, 0, 32)
    assert (t.run_check())
