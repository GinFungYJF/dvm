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
"""Build dvm-torch against an installed dvm-rt and a system PyTorch."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.utils.cpp_extension import include_paths as torch_include_paths

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SOURCES = [
    "torch/csrc/bindings.cpp",
    "torch/csrc/flags.cpp",
    "torch/csrc/adapter/cuda_adapter.cpp",
    "torch/csrc/fusion/core/graph.cpp",
    "torch/csrc/fusion/core/side_table.cpp",
    "torch/csrc/fusion/core/tensor_meta.cpp",
    "torch/csrc/fusion/emit/emit.cpp",
    "torch/csrc/dispatch/overrides.cpp",
    "torch/csrc/dispatch/registry.cpp",
]


def dvm_rt_dirs():
    """Locate the dvm-rt headers and shared library.

    ``DVM_RT_ROOT`` lets build.sh point at a staged install so the wheel can be
    produced before dvm-rt is installed into the environment.
    """
    staged = os.environ.get("DVM_RT_ROOT")
    if staged:
        root = Path(staged).resolve()
        return root / "dvm" / "include", root / "dvm"
    try:
        import dvm_rt

        lib_dir = Path(dvm_rt.get_library_dir())
        return Path(dvm_rt.get_include()), lib_dir
    except ImportError as exc:  # pragma: no cover - build-time diagnostic
        raise RuntimeError(
            "dvm-torch needs dvm-rt at build time: pip install dvm-rt first "
            "or set DVM_RT_ROOT to a staged install"
        ) from exc


def cuda_include_dirs():
    """CUDA toolkit headers, resolved the same way the DVM Makefile does.

    Distributions ship two layouts: the full toolkit keeps headers in
    ``$CUDA_HOME/include``, while the minimal runtime tarball used in CI keeps
    them in ``$CUDA_HOME/targets/<arch>-linux/include``. ``include_paths('cuda')``
    only knows the first, so add the second when it exists.
    """
    dirs = []
    for path in torch_include_paths("cuda"):
        if path not in torch_include_paths("cpu") and path not in dirs:
            dirs.append(path)

    override = os.environ.get("CUDA_INCLUDE_DIR")
    if override:
        dirs.append(override)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        import platform

        targets = Path(cuda_home) / "targets" / "{}-linux".format(platform.machine()) / "include"
        if (targets / "cuda_runtime.h").exists():
            dirs.append(str(targets))

    missing = [d for d in dirs if (Path(d) / "cuda_runtime.h").exists()]
    if not missing:
        raise RuntimeError(
            "cuda_runtime.h not found; set CUDA_HOME to a CUDA 12+ toolkit "
            "or CUDA_INCLUDE_DIR to its include directory"
        )
    return dirs


def build_extension():
    include_dir, lib_dir = dvm_rt_dirs()
    version = (ROOT / "version.txt").read_text().strip()

    extra_link = [
        f"-L{lib_dir}",
        "-ldvm",
        # Resolve libdvm.so relative to the installed package, never via a
        # build-time path: inside the wheel the extension lives in dvm_torch/
        # while libdvm.so is installed as the sibling package dvm/.
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath,$ORIGIN/../dvm",
    ]
    extension = CppExtension(
        name="dvm_torch._C",
        sources=[str(ROOT / s) for s in SOURCES],
        # CppExtension does not add the CUDA toolkit headers, but the ATen CUDA
        # headers (CUDAMiscFunctions.h and friends) include cuda_runtime.h.
        include_dirs=[str(ROOT / "torch/csrc"), str(include_dir)] + cuda_include_dirs(),
        libraries=["c10_cuda", "torch_cuda"],
        extra_compile_args=["-std=c++20", "-O2", "-fvisibility=hidden"],
        extra_link_args=extra_link,
        language="c++",
    )
    return version, extension


VERSION, EXTENSION = build_extension()

setup(
    name="dvm-torch",
    version=VERSION,
    description="Eager operator fusion for native PyTorch on DVM",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=["torch>=2.13", f"dvm-rt=={VERSION}"],
    packages=["dvm_torch"],
    package_dir={"dvm_torch": str(ROOT / "torch/python/dvm_torch")},
    ext_modules=[EXTENSION],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=False)},
    zip_safe=False,
)
