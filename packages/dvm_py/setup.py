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
"""Build the DVM runtime wheel: libdvm.so plus the pybind11 binding.

The standalone Python binding depends only on NumPy.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

HERE = Path(__file__).resolve().parent
ROOT = HERE / "native" if (HERE / "native").exists() else HERE.parents[1]
SOURCES = ("Makefile", "version.txt", "LICENSE", "include", "src", "scripts", "prebuild", "python")

VERSION = (ROOT / "version.txt").read_text().strip()


def copy_sources(destination):
    """Copy the build inputs; the wheel is built from a clean tree."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        source = ROOT / name
        target = destination / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("*.so", "*.so.*", "__pycache__"),
            )
        else:
            shutil.copy2(source, target)


class RuntimeBuild(build_py):
    def run(self):
        super().run()
        # The default NPU build also publishes the checkout's native artifacts.
        if os.environ.get("DVM_BUILD_INPLACE") == "1":
            work = ROOT
        else:
            work = Path(self.get_finalized_command("build").build_temp).resolve() / "native"
            copy_sources(work)

        arch = os.environ.get("DVM_ARCH", "npu")
        if arch not in ("npu", "gpu"):
            raise RuntimeError("DVM_ARCH must be npu or gpu")
        if arch == "npu" and not os.environ.get("ASCEND_HOME_PATH"):
            raise RuntimeError("source CANN set_env.sh before building dvm_py")

        args = [
            "make",
            "-j{}".format(os.environ.get("DVM_BUILD_JOBS", "8")),
            "DVM_ARCH={}".format(arch),
            "PYTHON={}".format(sys.executable),
        ]
        # check-shared builds the runtime and binding before verifying the public symbols.
        subprocess.run(args + ["check-shared"], cwd=work, check=True)

        package = Path(self.build_lib) / "dvm"
        shutil.copy2(work / "LICENSE", package / "LICENSE")
        shutil.copy2(work / "_dvm_py.so", package)
        shutil.copy2(work / "libdvm.so.1", package)
        # Wheels do not preserve symlinks reliably; ship a real file.
        shutil.copy2(work / "libdvm.so.1", package / "libdvm.so")
        include_dir = package / "include"
        # A reused wheel build directory can contain obsolete headers.
        if include_dir.exists():
            shutil.rmtree(include_dir)
        include_dir.mkdir()
        shutil.copy2(work / "include" / "dvm.h", include_dir)
        (package / "_build_info.py").write_text(
            "version = {!r}\nbackend = {!r}\n".format(VERSION, arch)
        )


class RuntimeSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_sources(Path(base_dir) / "native")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="dvm_py",
    version=VERSION,
    description="Standalone Device Virtual Machine runtime (CUDA or Ascend)",
    license="Apache-2.0",
    python_requires=">=3.7",
    install_requires=["numpy"],
    packages=["dvm", "dvm_py"],
    package_dir={"dvm": str(ROOT / "python" / "dvm")},
    cmdclass={"build_py": RuntimeBuild, "sdist": RuntimeSdist},
    distclass=BinaryDistribution,
)
