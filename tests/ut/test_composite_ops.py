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


pytestmark = pytest.mark.gpu


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def Tanh(t, x):  # tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
    fx = t.mul(x, -1.0)
    e = t.exp(x)
    fe = t.exp(fx)
    add = t.add(e, fe)
    sub = t.sub(e, fe)
    div = t.div(sub, add)
    return div


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_gelu_tanh(type):
    t = Tester()
    a = np.random.normal(0, 1, [32, 1024]).astype(type)
    x = t.load(a)
    tmp = Tanh(t, x)
    t.store_expect(tmp, np.tanh(a))

    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_gelu_expander_8_op(type):
    t = Tester()
    np.random.seed(12)
    a = np.random.normal(0, 1, [32, 1024]).astype(type)
    x = t.load(a)
    tmp = t.mul(x, x)
    tmp = t.mul(tmp, x)
    tmp = t.mul(tmp, 0.044715)
    tmp = t.add(x, tmp)

    tmp = t.mul(tmp, -1.5957691)
    tmp = t.exp(tmp)
    tmp = t.add(tmp, 1.0)

    tmp = t.div(x, tmp)
    # ms_gelu = ops.GeLU()(Tensor(a)), consistent with Mindpore verification
    t.store_expect(tmp, gelu(a))

    assert (t.run_check())
