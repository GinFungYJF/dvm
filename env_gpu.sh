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

export DVM_ARCH="gpu"

# ----------------------------
# GPU (CUDA) toolchain
# ----------------------------
if [ -z "${CUDA_HOME}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
  fi
fi
export CUDA_HOME

if [ -z "${CUDA_HOME}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "ERROR: CUDA toolkit not found. Please set CUDA_HOME or add nvcc to PATH."
  return 1 2>/dev/null || exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# ----------------------------
# Set Pybind11 includes if not already set
# ----------------------------
if [[ -z "${PYBIND11_INCLUDES}" ]]; then
  PYTHON_BIN=$(command -v python3 || command -v python)
  if [ -n "${PYTHON_BIN}" ]; then
    export PYBIND11_INCLUDES=$(${PYTHON_BIN} -m pybind11 --includes 2>/dev/null)
  fi
  if [ -z "${PYBIND11_INCLUDES}" ]; then
    echo "WARNING: Failed to get pybind11 includes. Please ensure pybind11 is installed."
  fi
fi

export PYTHONPATH="${CUR_SCRIPT_PATH}:${CUR_SCRIPT_PATH}/python:${PYTHONPATH}"

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
