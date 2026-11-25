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
"""Build the PyTorch fusion extension with the DVM CUDA runtime linked in."""

import os
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

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


class FusionBuild(BuildExtension):
    def build_extension(self, ext):
        # Run the archive target in its own directory to keep the checkout's
        # native build (which may target Ascend) independent of the wheel.
        with TemporaryDirectory() as tmp:
            native = Path(tmp)
            for name in ("src", "include", "scripts"):
                (native / name).symlink_to(ROOT / name, target_is_directory=True)
            jobs = os.environ.get("DVM_BUILD_JOBS", "8")
            subprocess.run(
                ["make", "-f", str(ROOT / "Makefile"), f"-j{jobs}", "DVM_ARCH=gpu",
                 f"CUDA_HOME={CUDA_HOME}", f"PYTHON={sys.executable}", "libdvm.a"],
                cwd=native, check=True,
            )
            archive = str(native / "libdvm.a")
            ext.extra_objects.append(archive)
            ext.depends.append(archive)
            super().build_extension(ext)


def fusion_extension():
    if CUDA_HOME is None:
        raise RuntimeError("Building torch_dvm requires CUDA-enabled PyTorch and a CUDA toolkit.")
    # Some toolkits keep their development files under targets/<arch>-linux.
    target = Path(CUDA_HOME) / "targets" / f"{platform.machine()}-linux"
    # Keep the embedded runtime private when another DVM build shares this process.
    extra_link = ["-Wl,--exclude-libs,libdvm.a"]
    if os.environ.get("asan"):
        extra_link.append("-fsanitize=address")
    return CUDAExtension(
        name="torch_dvm._C",
        sources=[str(ROOT / s) for s in SOURCES],
        include_dirs=[str(ROOT / "torch/csrc"), str(ROOT / "include"),
                      os.environ.get("CUDA_INCLUDE_DIR") or str(target / "include")],
        library_dirs=[os.environ.get("CUDA_LIB_DIR") or str(target / "lib")],
        extra_compile_args=["-std=c++20", "-O2", "-fvisibility=hidden"],
        extra_link_args=extra_link,
    )


setup(
    name="torch_dvm",
    version=(ROOT / "version.txt").read_text().strip(),
    description="Eager operator fusion for native PyTorch on DVM",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=["torch>=2.13"],
    packages=["torch_dvm"],
    package_dir={"torch_dvm": str(ROOT / "torch/python/torch_dvm")},
    ext_modules=[fusion_extension()],
    cmdclass={"build_ext": FusionBuild},
    zip_safe=False,
)
