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

import os
import subprocess
import sys

import pytest
import torch

from conftest import CUDA, fused, torch_dvm


@CUDA
@pytest.mark.parametrize("setting", [None, "0", "1"], ids=["default", "off", "on"])
def test_pointwise_backend(setting):
    env = os.environ.copy()
    if setting is None:
        env.pop("DVM_GPU_POINTWISE", None)
    else:
        env["DVM_GPU_POINTWISE"] = setting
    subprocess.run([sys.executable, __file__], env=env, check=True)


def check_case(case, enabled):
    width = {"full": 768, "packed_tail": 68, "odd_width": 769, "ragged_outputs": 259,
             "short_tail": 17, "offset": 256}.get(case, 64)
    opts = {"device": "cuda", "dtype": torch.float16 if case == "half" else torch.float32}
    shape = (32, 7, width)
    if case == "offset":
        x = torch.randn(32 * 7 * width + 1, **opts)[1:].view(shape)
    else:
        x = torch.randn(shape, **opts)
    row = torch.randn(32, 1 if case == "middle_broadcast" else 7,
                      width if case == "middle_broadcast" else 1, **opts)
    std = torch.rand(32, 7, 1, **opts) + .125
    col = torch.randn(width, **opts)
    beta = torch.randn_like(col)
    initial = torch.randn_like(x)
    scalar = torch.randn(1, 1, 1, **opts)
    if case == "values":
        x[0, 0, :5] = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan")], **opts)
        row[0, 0] = 0.
        std[0, 0] = -.375
        col[:5] = 1.

    def run(accumulator):
        centered = x - row
        if case == "scalar_store":
            scale = scalar * 1.25
            return scale, centered / scale
        if case == "long_chain":
            for _ in range(8):
                centered = centered * .75
            return (centered,)
        if case == "inplace":
            scale = std * .25
            result = accumulator.add_(scale * centered)
            return (result,)
        if case == "fanout":
            b = x * col
            c = x - std
            d = x * beta
            return (centered * b + c * d,)
        if case in ("column_store", "ragged_outputs"):
            shifted = col * 2.
            corrected = std + .375
            return shifted, corrected, (centered * shifted) / corrected
        if case == "other_formula":
            return (std / (centered * 2.) - beta,)
        weight = col * centered
        den = std + .375
        quotient = weight / den
        out = quotient + beta
        if case == "final_only":
            return (out,)
        if case == "five_outputs":
            return centered, weight, den, quotient, out
        return centered, weight, den, out

    reference_acc = initial.clone()
    expected = run(reference_acc)
    acc = initial.clone()
    alias = acc.detach()
    version = acc._version
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as trace:
        torch.sin(x)
        with fused():
            actual = run(acc)
        torch.cuda.current_stream().synchronize()
    kernels = [e for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    selected = [e for e in kernels if "dvm_vm_gpu_packed_code" in e.name]
    should_match = enabled and case in ("full", "packed_tail", "other_formula", "final_only", "five_outputs",
                                        "inplace", "scalar_store", "values")
    assert len(selected) == int(should_match), (case, [e.name for e in kernels])
    if should_match:
        assert len(kernels) == 2, (case, [e.name for e in kernels])
        assert selected[0].device_resource_id == kernels[0].device_resource_id
    elif case == "full":
        assert any("dvm::gpu::dvm_vm_gpu(" in e.name for e in kernels)
    for tensor, ref in zip(actual, expected):
        torch.testing.assert_close(tensor, ref, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(tensor[tensor == 0]), torch.signbit(ref[ref == 0]))
    if case == "inplace":
        assert actual[0].data_ptr() == alias.data_ptr() and alias._version == version + 1
        torch.testing.assert_close(alias, reference_acc, rtol=0, atol=0)


def check_partial_overlap():
    x_shape = (32, 7, 256)
    base = torch.randn(32 * 7 * 256 + 1, device="cuda")
    row = torch.randn(32, 7, 1, device="cuda")
    scale = torch.randn_like(row)

    def inputs():
        buf = base.clone()
        return buf[:-1].view(x_shape), buf[1:].view(x_shape)

    def update(x, acc):
        centered = x - row
        weighted = (scale * .25) * centered
        return acc.add_(weighted)

    expected = update(*inputs())
    x, acc = inputs()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as trace:
        with fused():
            actual = update(x, acc)
        torch.cuda.current_stream().synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    kernels = [e for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kernels and (len(kernels) > 1 or "dvm_vm_gpu_packed_code" not in kernels[0].name)


if __name__ == "__main__":
    torch.manual_seed(31)
    torch_dvm.disable()
    with torch.cuda.stream(torch.cuda.Stream()):
        for name in ("full", "packed_tail", "odd_width", "short_tail", "offset", "column_store", "ragged_outputs",
                     "other_formula", "final_only", "five_outputs", "fanout", "inplace", "scalar_store", "long_chain",
                     "values", "middle_broadcast", "half"):
            check_case(name, os.environ.get("DVM_GPU_POINTWISE", "1") == "1")
        check_partial_overlap()
