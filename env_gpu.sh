#!/bin/bash

CUR_SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# ----------------------------
# Arg parse (only supports --key=value)
# ----------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      echo "Usage:"
      echo "  source env_gpu.sh"
      return 0 2>/dev/null || exit 0
      ;;
    *)
      echo "WARNING: Unknown argument ignored: $1"
      shift
      ;;
  esac
done

# ----------------------------
# GPU (CUDA) toolchain
# ----------------------------
dvm_cuda_home=${CUDA_HOME:-/usr/local/cuda}
# NVCC can be a wrapper outside the toolkit, so prefer nvcc on PATH for discovery.
if [[ -z "${CUDA_HOME:-}" ]] && dvm_nvcc=$(command -v nvcc || command -v "${NVCC:-}"); then
  dvm_cuda_home=$(dirname "$(dirname "$(readlink -f "${dvm_nvcc}")")")
fi
if [[ ! -d "${dvm_cuda_home}" ]]; then
  echo "ERROR: CUDA toolkit not found. Please set CUDA_HOME or add nvcc to PATH." >&2
  return 1 2>/dev/null || exit 1
fi
PYTHON_BIN=${PYTHON:-$(command -v python3 || command -v python)}
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python is required to check the CUDA toolkit. Please set PYTHON or add python3 to PATH." >&2
  return 1 2>/dev/null || exit 1
fi
if ! PATH="${dvm_cuda_home}/bin:${PATH}" "${PYTHON_BIN}" "${CUR_SCRIPT_PATH}/scripts/check_cuda.py" \
    "${NVCC:-${dvm_cuda_home}/bin/nvcc}"; then
  return 1 2>/dev/null || exit 1
fi
export CUDA_HOME="${dvm_cuda_home}"
export DVM_ARCH="gpu"
export PATH="${CUDA_HOME}/bin:${PATH}"

# ----------------------------
# Set Pybind11 includes if not already set
# ----------------------------
if [[ -z "${PYBIND11_INCLUDES:-}" ]]; then
  export PYBIND11_INCLUDES=$("${PYTHON_BIN}" -m pybind11 --includes 2>/dev/null)
  if [ -z "${PYBIND11_INCLUDES}" ]; then
    echo "WARNING: Failed to get pybind11 includes. Please ensure pybind11 is installed."
  fi
fi

export PYTHONPATH="${CUR_SCRIPT_PATH}:${CUR_SCRIPT_PATH}/python:${PYTHONPATH:-}"

# ----------------------------
# Optional: Print environment summary
# ----------------------------
echo "Environment summary:"
echo "---------------------------------"
echo "DVM_ARCH: ${DVM_ARCH}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES:-Not set}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "---------------------------------"
