# Copyright 2026 Huawei Technologies Co., Ltd
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

import os
import subprocess
import sys

import numpy as np
import pytest

import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def isolated_workdir(monkeypatch, tmp_path):
    monkeypatch.delenv("DVM_VF_JIT_CACHE_DIR", raising=False)
    dvm.Kernel.set_vf_fusion(0)
    monkeypatch.chdir(tmp_path)


def run_codegen_in_child(request):
    child_flag = "DVM_VF_CODEGEN_CHILD"
    if os.environ.get(child_flag) == "1":
        return False
    env = os.environ.copy()
    env.setdefault("DVM_SOC_NAME", "Ascend950PR_9599")
    env[child_flag] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", request.node.nodeid],
        cwd=str(request.config.rootdir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def make_f32_multi_output_graph(t, shape=(1024, 1024)):
    x_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    y_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    z_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    x = t.load(x_np)
    y = t.load(y_np)
    z = t.load(z_np)
    abs_x = t.abs(x)
    scaled = t.add(abs_x, 0.25)
    add = t.add(scaled, y)
    cond = t.greater(add, 0.0)
    selected = t.select(cond, add, z)
    out = t.cast(selected, "float16")

    add_expect = np.abs(x_np) + 0.25 + y_np
    out_expect = np.where(add_expect > 0.0, add_expect, z_np).astype(np.float16)
    # `add` is deliberately both an internal value and an externally stored
    # value, exercising Custom primary output plus ExtOut metadata.
    t.store_expect(add, add_expect)
    t.store_expect(out, out_expect)


def make_gelu_graph(t, shape=(1024, 1024)):
    x_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    x = t.load(x_np)
    value = t.mul(x, x)
    value = t.mul(value, x)
    value = t.mul(value, 0.044715)
    value = t.add(x, value)
    value = t.mul(value, -1.5957691)
    value = t.exp(value)
    value = t.add(value, 1.0)
    out = t.div(x, value)

    x_cubed = x_np * x_np * x_np
    denominator = np.exp(
        np.float32(-1.5957691) * (x_np + np.float32(0.044715) * x_cubed)
    ) + 1.0
    t.store_expect(out, x_np / denominator)


def make_abs_chain(t, shape=(1024, 1024), length=24):
    input_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    value = t.load(input_np)
    for _ in range(length):
        value = t.abs(value)
    t.store_expect(value, np.abs(input_np))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_entry_symbol_lookup_codegen(request):
    if run_codegen_in_child(request):
        return
    dvm.Kernel.set_vf_fusion(1)
    input_np = np.ones((16, 16), dtype=np.float32)

    for _ in range(2):
        t = Tester()
        t.set_passes("VfFusion")
        value = t.load(input_np)
        value = t.abs(t.abs(value))
        t.store_expect(value, input_np)

        t.codegen()

        assert t.dump().count("Custom") == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_long_chain_codegen(request):
    if run_codegen_in_child(request):
        return
    dvm.Kernel.set_vf_fusion(1)

    t = Tester()
    t.set_passes("VfFusion")
    make_abs_chain(t)

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") > 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("length, regions", [(12, 1), (13, 2), (24, 2), (25, 3)])
def test_long_chain(length, regions):
    dvm.Kernel.set_vf_fusion(1)

    t = Tester()
    t.set_passes("VfFusion")
    make_abs_chain(t, length=length)

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == regions
    assert "Abs" not in dump
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dynamic_same_shape_select_codegen(request):
    if run_codegen_in_child(request):
        return
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("vector:dyn", use_pass_opt=True)
    x = t.load([-1], "float32")
    y = t.load([-1], "float32")
    condition = t.load([-1], "bool")
    value = t.add(t.abs(x), y)
    clipped = t.minimum(value, 1.5)
    scaled = t.mul(y, 0.5)
    t.store(t.cast(t.select(condition, clipped, scaled), "float16"))

    shape = (3, 17, 9)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    condition_np = np.random.randint(0, 2, shape).astype(np.bool_)
    t.input(x, x_np)
    t.input(y, y_np)
    t.input(condition, condition_np)

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 1
    assert "Broadcast" not in dump


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dynamic_same_shape_multi_output_codegen(request):
    if run_codegen_in_child(request):
        return
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("vector:dyn")
    t.set_passes("VfFusion")
    x = t.load([-1], "float32")
    y = t.load([-1], "float32")
    intermediate = t.abs(x)
    out = t.add(intermediate, y)
    t.store(intermediate)
    t.store(out)
    t.input(x, np.ones((3, 17, 9), dtype=np.float32))
    t.input(y, np.ones((3, 17, 9), dtype=np.float32))

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 1
    assert "Abs" not in dump and "Add" not in dump


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="VF Fusion only supports C310")
def test_dynamic_mix_preserves_vf_edges_codegen():
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("mix:dyn", use_pass_opt=True)
    x = t.load([-1, 512], "float16")
    w = t.load([768, 512], "float16")
    gate = t.load([-1, 768], "float16")
    shape = t.int_array()
    mm = t.matmul(x, w, False, True)
    value = t.cast(t.reshape(gate, shape), "float32")
    value = t.cast(t.abs(t.abs(value)), "float16")
    t.store(mm)
    t.store(t.mul(value, t.reshape(mm, shape)))

    for batch, seq in [(2, 128), (2, 128), (3, 96)]:
        t.input(x, np.zeros((batch * seq, 512), dtype=np.float16))
        t.input(w, np.zeros((768, 512), dtype=np.float16))
        t.input(gate, np.zeros((batch * seq, 768), dtype=np.float16))
        shape.update([batch, seq, 768])
        t.codegen()
        assert "Custom" in t.dump()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_multi_output():
    dvm.Kernel.set_vf_fusion(1)
    t = Tester()
    t.set_passes("VfFusion")
    make_f32_multi_output_graph(t)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_gelu():
    dvm.Kernel.set_vf_fusion(1)
    t = Tester()
    t.set_passes("VfFusion")
    make_gelu_graph(t)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_f16_tail():
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1025)
    input_np = np.random.normal(-1.0, 1.0, shape).astype(np.float16)
    y_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    z_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(input_np, "float16")
    y = t.load(y_np)
    z = t.load(z_np)
    x_f32 = t.cast(x, "float32")
    add = t.add(t.abs(x_f32), y)
    selected = t.select(t.greater(add, z), add, y)
    out = t.cast(selected, "float16")
    add_expect = np.abs(input_np.astype(np.float32)) + y_np
    out_expect = np.where(add_expect > z_np, add_expect, y_np).astype(np.float16)
    t.store_expect(out, out_expect)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_dynamic_same_shape_select():
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("vector:dyn")
    t.set_passes("VfFusion")
    x = t.load([-1], "float32")
    y = t.load([-1], "float32")
    condition = t.load([-1], "bool")
    value = t.add(t.abs(x), y)
    clipped = t.minimum(value, 1.5)
    scaled = t.mul(y, 0.5)
    out = t.store(t.cast(t.select(condition, clipped, scaled), "float16"))

    for shape in ((3, 17, 9), (5, 11, 7)):
        x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
        y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
        condition_np = np.random.randint(0, 2, shape).astype(np.bool_)
        value_np = np.abs(x_np) + y_np
        expect = np.where(condition_np, np.minimum(value_np, 1.5), y_np * 0.5).astype(np.float16)
        t.input(x, x_np)
        t.input(y, y_np)
        t.input(condition, condition_np)
        t.run()
        assert t.check(out, expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_dynamic_long_chain():
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("vector:dyn")
    t.set_passes("VfFusion")
    x = t.load([-1], "float32")
    value = x
    for _ in range(24):
        value = t.abs(value)
    out = t.store(value)

    for shape in ((3, 17, 9), (5, 11, 7)):
        input_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
        t.input(x, input_np)
        t.run()
        dump = t.dump()
        assert dump.count("Custom") == 2
        assert "Abs" not in dump
        assert t.check(out, np.abs(input_np))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_spec_vf_fusion_clone(request):
    if run_codegen_in_child(request):
        return
    dvm.Kernel.set_vf_fusion(2)
    shape = (2, 3, 4, 64)
    input_np = np.random.normal(0.0, 0.03, shape).astype(np.float32)

    t = Tester("vector:spec")
    t.set_passes("VfFusion")
    x = t.load(input_np)
    fused = t.add(t.abs(x), 0.25)
    reduced = t.sum(fused, (0,), True)
    t.spec_next()
    out = t.mul(reduced, 0.5)
    expect = np.sum(np.abs(input_np) + 0.25, axis=(0,), keepdims=True) * 0.5
    t.store_expect(out, expect)

    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="VF Fusion only supports C310")
def test_spec_cast():
    dvm.Kernel.set_vf_fusion(2)
    t = Tester("vector:spec", use_pass_opt=True)
    shape = (8, 128, 1024)
    rng = np.random.default_rng(17)

    arrays = [rng.normal(0.0, 0.1, shape).astype(np.float16) for _ in range(3)]
    scale_np = rng.uniform(0.5, 1.0, (*shape[:2], 1)).astype(np.float32)
    x, y, z = [t.load(array, "float16") for array in arrays]
    scale = t.load(scale_np)
    # The add starts a consumer partition before its producer cast partition
    # starts. Sorting partitions by their first node reverses this dependency.
    add = t.add(x, y)
    expanded = t.broadcast(scale, shape)
    product = t.cast(t.mul(t.cast(z, "float32"), expanded), "float16")
    value = t.cast(t.mul(add, product), "float32")
    reduced = t.sum(value, (0, 1), True)
    t.spec_next()
    out = t.cast(reduced, "float16")
    add_np = (arrays[0] + arrays[1]).astype(np.float16)
    product_np = (arrays[2].astype(np.float32) * scale_np).astype(np.float16)
    value_np = (add_np * product_np).astype(np.float16).astype(np.float32)
    expect = np.sum(value_np, axis=(0, 1), keepdims=True).astype(np.float16)
    t.store_expect(out, expect, eps=1e-3)

    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_reduce_boundary():
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)
    z_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np, "float16")
    y = t.load(y_np, "float16")
    z = t.load(z_np, "float16")
    add = t.add(x, y)
    sub = t.sub(add, z)
    cast = t.cast(sub, "float32")
    out = t.sum(cast, (1,), True)
    expect = np.sum(
        (x_np + y_np - z_np).astype(np.float32), axis=1, keepdims=True
    )
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    assert "Custom" in dump
    assert "Reduce" in dump
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="VF Fusion only supports C310")
def test_upcast_boundary():
    dvm.Kernel.set_vf_fusion(1)
    t = Tester("vector", use_pass_opt=True)
    array = np.linspace(-0.5, 0.5, 129, dtype=np.float32)
    x = t.load(array.astype(np.float16))
    value = t.mul(t.add(x, 0.25), 0.5)
    value = t.cast(value, "float32")
    value = t.mul(t.add(value, 0.25), 0.5)
    t.store_expect(value, ((array + 0.25) * 0.5 + 0.25) * 0.5, eps=0)
    assert t.run_check()
    assert t.dump().count("Custom") == 2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="VF Fusion only supports C310")
def test_downcast_boundary():
    dvm.Kernel.set_vf_fusion(1)
    t = Tester("vector", use_pass_opt=True)
    array = np.linspace(-0.5, 0.5, 129, dtype=np.float32)
    x = t.load(array)
    value = t.mul(t.add(x, 0.25), 0.5)
    value = t.cast(value, "float16")
    value = t.mul(t.add(value, 0.25), 0.5)
    t.store_expect(value, ((array + 0.25) * 0.5 + 0.25) * 0.5, eps=0)
    assert t.run_check()
    assert t.dump().count("Custom") == 2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("extra_ops, regions", [(0, 2), (24, 6), (64, 12)])
def test_cycle_guard(extra_ops, regions):
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    first = t.add(t.abs(x), 0.25)
    for _ in range(extra_ops):
        first = t.abs(first)
    # Contracting both VF regions would create Custom -> Reshape -> Custom.
    bridge = t.reshape(t.reshape(first, (512, 2048)), shape)
    out = t.abs(t.add(first, bridge))
    for _ in range(extra_ops):
        out = t.abs(out)
    t.store_expect(out, np.abs(2 * (np.abs(x_np) + 0.25)))

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == regions
    assert dump.count("Reshape") == 2
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("depth", [1, 26])
def test_diamond_fusion(depth):
    dvm.Kernel.set_vf_fusion(1)
    input_np = np.linspace(-1.0, 1.0, 32 * 65, dtype=np.float32).reshape(32, 65)
    t = Tester()
    t.set_passes("VfFusion")
    value = t.load(input_np)
    expect = input_np
    for index in range(depth):
        left = t.abs(value)
        right = t.add(value, 0.25)
        joined = t.add(left, right)
        value = t.mul(t.add(joined, joined), 0.25)
        joined_np = np.abs(expect) + (expect + np.float32(0.25))
        expect = (joined_np + joined_np) * np.float32(0.25)
        if index == 0:
            intermediate = left
    t.store_expect(intermediate, np.abs(input_np))
    t.store_expect(value, expect)

    t.codegen()
    dump = t.dump()
    assert dump.count("Custom") == 1 if depth == 1 else dump.count("Custom") > 1
    assert all(op not in dump for op in ("Abs", "Add", "Mul"))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_multi_fusion():
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    z_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    z = t.load(z_np)
    left = t.add(t.abs(x), y)
    right = t.sub(t.abs(z), y)
    t.store_expect(left, np.abs(x_np) + y_np)
    t.store_expect(right, np.abs(z_np) - y_np)

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 2
    assert all(op not in dump for op in ("Abs", "Add", "Sub"))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_extra_io():
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1024)
    inputs_np = [
        np.random.normal(0.0, 0.2, shape).astype(np.float32) for _ in range(4)
    ]

    t = Tester()
    t.set_passes("VfFusion")
    a, b, c, d = [t.load(value) for value in inputs_np]
    add = t.add(a, b)
    sub = t.sub(add, c)
    out = t.mul(sub, d)
    add_expect = inputs_np[0] + inputs_np[1]
    sub_expect = add_expect - inputs_np[2]
    t.store_expect(add, add_expect)
    t.store_expect(sub, sub_expect)
    t.store_expect(out, sub_expect * inputs_np[3])

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 1
    assert dump.count("ExtOut") == 2
    assert all(op not in dump for op in ("Add", "Sub", "Mul"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("shape", [(1024, 1024), (1024, 1025)])
def test_broadcast_chain(shape):
    dvm.Kernel.set_vf_fusion(1)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    bias_np = np.random.normal(0.5, 0.1, (1, shape[1])).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    bias = t.broadcast(t.load(bias_np), shape)
    value = t.abs(x)
    value = t.add(value, 0.25)
    value = t.mul(value, bias)
    value = t.add(value, y)
    value = t.maximum(value, -0.5)
    value = t.mul(value, 0.5)
    value = t.sub(value, bias)
    value = t.abs(value)
    value = t.add(value, 0.125)
    value = t.minimum(value, 2.0)
    value = t.mul(value, value)
    out = t.add(value, 0.1)

    bias_expect = np.broadcast_to(bias_np, shape)
    expect = np.abs(x_np)
    expect = expect + np.float32(0.25)
    expect = expect * bias_expect
    expect = expect + y_np
    expect = np.maximum(expect, np.float32(-0.5))
    expect = expect * np.float32(0.5)
    expect = expect - bias_expect
    expect = np.abs(expect)
    expect = expect + np.float32(0.125)
    expect = np.minimum(expect, np.float32(2.0))
    expect = expect * expect
    expect = expect + np.float32(0.1)
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    # One Broadcast plus twelve elementwise nodes.
    assert dump.count("Broadcast") == 1
    assert dump.count("Custom") == 1
    assert all(op not in dump for op in ("Abs", "Add", "Sub", "Mul", "Maximum", "Minimum"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("shape", [(1024, 1024), (1024, 2048)])
def test_broadcast_regions(shape):
    dvm.Kernel.set_vf_fusion(1)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    row_np = np.random.normal(0.0, 0.2, (1, shape[1])).astype(np.float32)
    column_np = np.random.normal(0.0, 0.2, (shape[0], 1)).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    row = t.mul(t.add(t.abs(t.load(row_np)), 0.1), 0.5)
    column = t.mul(t.add(t.abs(t.load(column_np)), 0.2), 0.25)
    row = t.broadcast(row, shape)
    column = t.broadcast(column, shape)
    value = t.add(x, row)
    value = t.sub(value, column)
    value = t.abs(value)
    value = t.mul(value, 0.75)
    value = t.add(value, row)
    value = t.maximum(value, column)
    value = t.minimum(value, 2.0)
    out = t.add(value, 0.05)

    row_expect = (np.abs(row_np) + np.float32(0.1)) * np.float32(0.5)
    column_expect = (np.abs(column_np) + np.float32(0.2)) * np.float32(0.25)
    row_expect = np.broadcast_to(row_expect, shape)
    column_expect = np.broadcast_to(column_expect, shape)
    expect = x_np + row_expect
    expect = expect - column_expect
    expect = np.abs(expect)
    expect = expect * np.float32(0.75)
    expect = expect + row_expect
    expect = np.maximum(expect, column_expect)
    expect = np.minimum(expect, np.float32(2.0))
    expect = expect + np.float32(0.05)
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    # Fourteen elementwise nodes form three VF regions around two Broadcasts.
    assert dump.count("Custom") == 3
    assert all(op not in dump for op in ("Abs", "Add", "Sub", "Mul", "Maximum", "Minimum"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_broadcast_select():
    dvm.Kernel.set_vf_fusion(1)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    bias_np = np.random.normal(0.0, 0.2, (1, shape[1])).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    bias = t.add(t.abs(t.load(bias_np)), 0.125)
    bias = t.broadcast(bias, shape)
    left = t.add(x, bias)
    right = t.sub(y, bias)
    product = t.mul(left, right)
    magnitude = t.abs(product)
    condition = t.greater(x, 0.0)
    clipped = t.minimum(magnitude, 1.5)
    scaled = t.mul(magnitude, 0.5)
    selected = t.select(condition, clipped, scaled)
    out = t.cast(selected, "float16")

    bias_expect = np.broadcast_to(np.abs(bias_np) + np.float32(0.125), shape)
    magnitude_expect = np.abs((x_np + bias_expect) * (y_np - bias_expect))
    selected_expect = np.where(
        x_np > np.float32(0.0),
        np.minimum(magnitude_expect, np.float32(1.5)),
        magnitude_expect * np.float32(0.5),
    )
    t.store_expect(magnitude, magnitude_expect, 1e-4)
    t.store_expect(out, selected_expect.astype(np.float16))

    t.codegen()
    dump = t.dump()

    assert dump.count("Broadcast") == 1
    assert dump.count("Custom") == 2
    assert dump.count("ExtOut") == 2
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="VF Fusion only supports C310")
@pytest.mark.parametrize("output_count", [7, 8, 9])
def test_custom_instruction_length_boundary(output_count):
    dvm.Kernel.set_vf_fusion(1)
    shape = (4, 4, 1)
    t = Tester()
    t.set_passes("VfFusion")
    arrays = [np.full(shape, i + 0.25, dtype=np.float32) for i in range(7)]
    loads = [t.load(array) for array in arrays]
    first = t.add(loads[0], loads[1])
    expected = arrays[0] + arrays[1]
    outputs = [(first, expected)]
    outputs.extend((t.add(first, load), expected + array)
                   for load, array in zip(loads[2:], arrays[2:]))
    for i in range(output_count - len(outputs)):
        outputs.append((t.mul(first, i + 2.0), expected * (i + 2.0)))
    for value, expect in outputs:
        t.store_expect(value, expect)

    assert t.run_check()
