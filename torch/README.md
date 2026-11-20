# dvm-torch: Eager Operator Fusion for Native PyTorch

**English** | [简体中文](README.zh.md)

`dvm-torch` fuses adjacent eager operators into DVM kernels **without modifying PyTorch
source code or subclassing tensors**. Directly fusible chains use a single launch;
DVM splits execution as needed to satisfy reduction dependencies. Tensors remain real
`at::Tensor` instances with valid metadata and `data_ptr`; only their contents are
computed lazily.

## Installation

```bash
pip install dvm-torch
```

Importing `dvm-torch` loads PyTorch and the extension dependencies. Operator overrides
are enabled through `TORCH_DVM_FUSION` or `enable()`. Native PyTorch dispatch is used
by default.

```python
import torch, dvm_torch

dvm_torch.enable()                  # Install CUDA overrides.
y = (a * b + c).relu()              # Fuse three operators into one graph.
dvm_torch.flush()                   # Also triggered by an unfused op or sync point.
print(dvm_torch.stats()["summary"])  # emit=3 flush=1 sizes={3op*1}
```

You can also enable fusion by setting an environment variable before importing
`dvm_torch`:

```bash
TORCH_DVM_FUSION="True" python train.py
TORCH_DVM_FUSION="True level=O1 disable_ops=mul,div" python train.py
```

Captured graphs execute through DVM at flush time, including single-operator
graphs. Dead-store elimination skips writes to intermediate tensors that are no
longer observable, while preserving live aliases and autograd saved values.
Use `dvm_torch.enable(disable_ops="sum")` to keep selected operators native.

## Architecture

| Layer | Location | Responsibility |
|---|---|---|
| L1 | `include/`, `src/` | Pure C++: `dvm::Kernel`, code generation, and kernel launch. **No torch dependency.** |
| L2 | `torch/csrc/fusion/` | Device-independent fusion core: `SideTable`, `GraphKernel`, `GraphManager`, `OpEmitter`, `TrimPolicy` |
| L3 | `torch/csrc/dispatch/` | ATen CUDA dispatch key overrides, original kernel capture, coverage guards, and environment-variable gating |

`L1` has no knowledge of `L2`, and `L2` has no knowledge of `L3`. The `core-purity`
check ensures that `include/` and `src/` contain no references to `torch/`, `at::`,
`c10::`, or `ATen`.

## Correctness Mechanisms

| Mechanism | Failure mode addressed |
|---|---|
| Flush before native dispatch, including operators outside the fusion registry | Uncovered operators consume uncomputed Pending tensors and silently produce incorrect results |
| `need_flush_for_input` checks storage offset, numel, shape, and dtype | Partially overlapping views or changes in shape |
| `need_flush_for_writable` checks every operand, including read-only operands | `b` is still Pending in `a.add_(b)` |
| Mandatory flush before host-read operators | `.item()` / `.cpu()` reads zeros from uncomputed storage |
| Flush before Python Event / Stream synchronization APIs | Synchronization points precede Pending work that has not yet been submitted |
| Disable fusion during capture | CUDA Graph replay repeats deferred work |
| Dead-store elimination using weak anchors and storage reference counts | Only live values need to be materialized |
| Discard graphs with no stores | DVM cannot generate code for an empty graph |

## Operator Coverage

Fusible operators with handwritten unboxed wrappers:

`add` `sub` `mul` `div` `maximum` `minimum` `pow(Tensor,Scalar)` `neg` `relu`
`sqrt` `rsqrt` `abs` `exp` `log` `reciprocal` `sigmoid` `clamp` `round` `floor`
`ceil` `trunc` `sum` `mean`, plus `add_` `sub_` `mul_` `div_` `relu_`.

`mul.Scalar` is covered as well as `mul.Tensor`. `eq.Tensor` / `eq.Scalar`
accept matching floating-point or Bool inputs; Bool scalar comparisons fuse for
0 and 1. `masked_fill.Scalar` / `masked_fill_.Scalar` accept floating-point data
and a broadcastable Bool mask. These paths require nonempty contiguous CUDA
inputs; mixed dtypes, integer comparisons, Tensor fill values and unsupported
layouts fall back to native kernels.

`detach` preserves pending computation while calling the native metadata operation,
so storage aliases and version counters retain PyTorch semantics. Writes through
detached aliases and consumers that require materialized data still flush as
needed. Use `disable_ops="detach"` to retain a flush at detach calls.

Other ATen operators use a generic guard: **flush first, then forward to the original
kernel captured before registration**.

Known limitations:

- Fusion accepts contiguous, dense CUDA tensors with at most 10 dimensions,
  matching DVM's shape storage capacity.
- `int64` arithmetic is not fused because the GPU VM has no int64 arithmetic instructions.
- Python scalars are fused only with floating-point tensors. PyTorch promotes
  `int32 / 2` to float32, which the VM cannot express while preserving those semantics.
- Reductions with empty, integer, or FP64 inputs, unsupported output dtypes,
  noncontiguous inputs, or deterministic algorithms enabled use native kernels.
- `sum` joins fusion only when its input is produced by the current pending graph.
  A sum over an already materialized tensor uses the native kernel, including
  when it would otherwise start a chain with later operators. `mean` is unchanged.
- `sum.out`, `mean.out`, and other reductions such as `amax` / `amin` execute after
  the generic guard flushes the pending graph.
- `tanh` / `sigmoid` are synthesized from `exp`. `tanh` fusion is disabled by default
  and can be enabled with `compile_fast=True`.

## Performance

The initial V100 benchmark used a chain of 10 elementwise operators, keeping the
results alive so dead-store elimination would not change the workload.
These measurements predate reduction support.

| N | eager | fused | Speedup |
|---|---|---|---|
| 65536 | 0.285 ms | 0.276 ms | 1.03× |
| 1048576 | 0.286 ms | 0.231 ms | 1.24× |
| 4194304 | 0.880 ms | 0.249 ms | 3.54× |

Kernel count: 20 eager launches → 1 `dvm_vm_gpu` launch.

**Speedups increase with tensor size.** The DVM GPU backend is a VM interpreter
that decodes instructions within a single kernel. For small tensors, host-side
graph construction and code generation dominate the execution time.

## Debugging

```python
with dvm_torch.Explorer() as ex:
    run_model()
print(ex.report["summary"])     # emit / flush / bails / graph size histogram
print(ex.report["by_reason"])   # Group by FlushReason to find frequent flush boundaries.
```

`dvm_torch.flush()` materializes pending work explicitly;
`dvm_torch.disable()` restores the original dispatch table.

## Tests

```bash
python -m pytest tests/fusion -q
```

`tests/fusion/test_escape.py` is a key regression suite: it checks that Pending
tensors consumed by `linear`, `softmax`, `amax`, `conv2d`, and `.item()` still
produce results matching eager execution.
`tests/fusion/test_reductions.py` covers reduction dimensions, dtypes, dependencies
between reductions and broadcasting, live intermediates, and native fallback paths.
It also verifies that supported reductions actually enter the fusion graph.
`tests/fusion/test_dse.py` checks dead-store elimination and preservation of live
aliases and autograd saved values.
`tests/fusion/test_sync.py` checks that synchronization points independently trigger
materialization, compares two non-default streams against eager execution, and
tests CUDA Graph capture / replay boundaries as well as hook installation and restoration.
