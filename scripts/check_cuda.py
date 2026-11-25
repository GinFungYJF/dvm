"""The current VM requires CUDA 12+ (including --jump-table-density)."""
import re
import subprocess
import sys

try:
    output = subprocess.check_output([sys.argv[1], "--version"], text=True)
except (OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f"Cannot run nvcc: {exc}. Set CUDA_HOME/NVCC to a CUDA 12+ toolkit.")
match = re.search(r"release (\d+)\.(\d+)", output)
if not match or int(match[1]) < 12:
    sys.exit("DVM GPU builds require CUDA toolkit >= 12.0; set CUDA_HOME/NVCC accordingly.")
