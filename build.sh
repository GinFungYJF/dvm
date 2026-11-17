#!/bin/bash
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

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_DIR="${BASEPATH}/output"
DIST_DIR="${BASEPATH}/dist"
PYTHON=${PYTHON:-python3}

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-a] [-j[n]] [-b npu|gpu] [-l] [-r] [-t]"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -a Enable ASAN"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -b    Target backend: npu (default) or gpu"
  echo "    -l    Build only the C++ libraries into output/, no wheels"
  echo "    -r    Build only the dvm-rt wheel"
  echo "    -t    Build only the dvm-torch wheel"
  echo ""
  echo "By default the wheels are built into dist/. dvm-torch is CUDA-only, so"
  echo "backend=npu produces the dvm-rt wheel alone."
}

mk_new_dir()
{
  local create_dir="$1"

  if [[ -d "${create_dir}" ]]; then
    rm -rf "${create_dir}"
  fi

  mkdir -pv "${create_dir}"
}

write_checksum()
{
  PACKAGE_LIST=$(ls lib*.a) || exit
  for PACKAGE_NAME in $PACKAGE_LIST; do
    echo $PACKAGE_NAME
    sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
  done
}

MAKE_ARGS=""
THREAD_NUM=8
ARCH="${DVM_ARCH:-npu}"
BUILD_WHEELS=1
BUILD_LIBS_ONLY=0
BUILD_RT=0
BUILD_TORCH=0
while getopts 'hdaj:b:lrt' opt
do
  case "${opt}" in
    h)
      usage
      exit 0
      ;;
    d)
      MAKE_ARGS="${MAKE_ARGS} dbg=1"
      ;;
    a)
      MAKE_ARGS="${MAKE_ARGS} asan=1"
      ;;
    j)
      THREAD_NUM=${OPTARG}
      ;;
    b)
      ARCH=${OPTARG}
      ;;
    l)
      BUILD_LIBS_ONLY=1
      ;;
    r)
      BUILD_RT=1
      ;;
    t)
      BUILD_TORCH=1
      ;;
    *)
      echo "Unknown option ${opt}!"
      usage
      exit 1
  esac
done

if [[ "${ARCH}" != "npu" && "${ARCH}" != "gpu" ]]; then
  echo "[ERROR] backend must be npu or gpu, got '${ARCH}'"
  usage
  exit 1
fi

if [[ ${BUILD_LIBS_ONLY} -eq 1 ]]; then
  BUILD_WHEELS=0
fi
# A single-package request narrows the default set rather than widening it.
if [[ ${BUILD_RT} -eq 1 || ${BUILD_TORCH} -eq 1 ]]; then
  BUILD_WHEELS=1
fi
if [[ ${BUILD_WHEELS} -eq 1 && ${BUILD_RT} -eq 0 && ${BUILD_TORCH} -eq 0 ]]; then
  BUILD_RT=1
  BUILD_TORCH=1
fi
# dvm-torch links against dvm-rt, so it always needs the runtime wheel first.
if [[ ${BUILD_TORCH} -eq 1 ]]; then
  BUILD_RT=1
fi
# dvm-torch is CUDA-only: on Ascend the equivalent adapter lives in torch_npu
# (it needs private NPU symbols), so the npu backend ships dvm-rt alone.
if [[ "${ARCH}" == "npu" && ${BUILD_TORCH} -eq 1 ]]; then
  echo "[WARN] dvm-torch targets CUDA only; skipping it for backend=npu"
  BUILD_TORCH=0
fi
if [[ "${ARCH}" == "npu" && ${BUILD_RT} -eq 0 ]]; then
  BUILD_RT=1
fi

MAKE_ARGS="${MAKE_ARGS} -j${THREAD_NUM} DVM_ARCH=${ARCH}"

# env.sh parses "$@" for its own flags; clear the positional parameters so our
# options are not reported as unknown arguments.
set --
if [[ "${ARCH}" == "gpu" ]]; then
  source ${BASEPATH}/env_gpu.sh
else
  source ${BASEPATH}/env.sh
fi

build_libraries()
{
  TARGET_FILE=(libdvm.a)
  if [[ "${ARCH}" == "npu" ]]; then
    TARGET_FILE+=(g_vkernel_c220_bin g_vkernel_c310_bin)
  fi

  echo "---------------- build start ----------------"
  echo "MAKE_ARGS: ${MAKE_ARGS}"
  make clean
  make ${MAKE_ARGS}

  if [ $? -ne 0 ]; then
    echo "[ERROR] compile failed!"
    exit 1
  fi

  for target in "${TARGET_FILE[@]}"; do
    if [ ! -f "${target}" ]; then
      echo "[ERROR] missing build output '${target}'"
      exit 1
    fi
  done
  echo "---------------- build end ----------------"

  mk_new_dir "${OUTPUT_DIR}"

  # Write git info
  git_branch=`git branch | awk '{print $2}'`
  git_commit_id=`git rev-parse HEAD`
  git_info_file="${OUTPUT_DIR}/lib_info.txt"
  echo "[lib information]" > ${git_info_file}
  echo "git branch: ${git_branch}" >> ${git_info_file}
  echo "commit  id: ${git_commit_id}" >> ${git_info_file}
  echo "backend: ${ARCH}" >> ${git_info_file}

  # Copy target to output directory
  cp "${TARGET_FILE[@]}" ${OUTPUT_DIR}
  cp libdvm.so.1 ${OUTPUT_DIR} 2>/dev/null && ln -sf libdvm.so.1 ${OUTPUT_DIR}/libdvm.so
  (
    cd ${OUTPUT_DIR}
    write_checksum
  )
}

wheel_of()
{
  # Newest wheel matching the package name, so a rebuild over an old dist/
  # cannot silently pick up a stale artifact.
  ls -t "${DIST_DIR}"/"$1"-*.whl 2>/dev/null | head -n 1
}

build_wheels()
{
  local version
  version=$(cat "${BASEPATH}/version.txt")
  echo "---------------- wheel build start (backend=${ARCH}, version=${version}) ----------------"
  mkdir -p "${DIST_DIR}"

  # setup.py drives `make` itself, so the tree must be clean of stale objects
  # built for the other backend.
  make clean

  local pip_common=(--no-build-isolation --no-deps --quiet)

  if [[ ${BUILD_RT} -eq 1 ]]; then
    echo "[wheel] dvm-rt"
    "${PYTHON}" -m pip wheel "${BASEPATH}/packages/dvm-rt" \
      "${pip_common[@]}" --wheel-dir "${DIST_DIR}" || {
      echo "[ERROR] dvm-rt wheel build failed"; exit 1; }
  fi

  if [[ ${BUILD_TORCH} -eq 1 ]]; then
    local rt_wheel
    rt_wheel=$(wheel_of dvm_rt)
    if [[ -z "${rt_wheel}" ]]; then
      echo "[ERROR] dvm-rt wheel not found in ${DIST_DIR}; cannot build dvm-torch"
      exit 1
    fi

    # Stage the freshly built dvm-rt so dvm-torch links against exactly the
    # runtime it declares a dependency on, not whatever is installed.
    local stage
    stage=$(mktemp -d)
    trap 'rm -rf "${stage}"' EXIT
    "${PYTHON}" -m pip install --no-deps --quiet --target "${stage}" "${rt_wheel}" || {
      echo "[ERROR] failed to stage ${rt_wheel}"; exit 1; }

    echo "[wheel] dvm-torch"
    DVM_RT_ROOT="${stage}" "${PYTHON}" -m pip wheel "${BASEPATH}/packages/dvm-torch" \
      "${pip_common[@]}" --wheel-dir "${DIST_DIR}" || {
      echo "[ERROR] dvm-torch wheel build failed"; exit 1; }

    rm -rf "${stage}"
    trap - EXIT
  fi

  echo "---------------- wheel build end ----------------"
  ls -lh "${DIST_DIR}"/*.whl
}

if [[ ${BUILD_WHEELS} -eq 0 ]]; then
  build_libraries
else
  build_wheels
fi
