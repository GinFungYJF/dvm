# torch_dvm: Eager Operator Fusion for Native PyTorch

**English** | [简体中文](README.zh.md)

`torch_dvm` fuses adjacent eager operators into DVM kernels **without modifying PyTorch
source code or subclassing tensors**. Directly fusible chains use a single launch;
DVM splits execution as needed to satisfy reduction dependencies. Tensors remain real
`at::Tensor` instances with valid metadata and `data_ptr`; only their contents are
computed lazily.

## Installation

```bash
pip install torch_dvm
```

The wheel includes the DVM CUDA backend. To build it from source, use an environment
with CUDA-enabled PyTorch and the CUDA toolkit, then run these commands from the repository root.

```bash
bash build.sh -t
python -m pip install dist/torch_dvm-*.whl
```

Importing `torch_dvm` loads PyTorch and the extension dependencies. Operator overrides
are enabled through `TORCH_DVM_FUSION` or `enable()`. Native PyTorch dispatch is used
by default.

```python
import torch
import torch_dvm

torch_dvm.enable()                  # Install CUDA overrides.
y = (a * b + c).relu()              # Fuse three operators into one graph.
torch_dvm.flush()                   # Also triggered by an unfused op or sync point.
print(torch_dvm.stats()["summary"])  # emit=3 flush=1 sizes={3op*1}
```

You can also enable fusion by setting an environment variable before importing
`torch_dvm`:

```bash
TORCH_DVM_FUSION="True" python train.py
TORCH_DVM_FUSION="True level=O1 disable_ops=mul,div" python train.py
```

## Architecture

| Layer | Location | Responsibility |
|---|---|---|
| L1 | `include/`, `src/` | Pure C++: `dvm::Kernel`, code generation, and kernel launch. **No torch dependency.** |
| L2 | `torch/csrc/fusion/` | Device-independent fusion core: `SideTable`, `GraphKernel`, `GraphManager`, `OpEmitter`, `TrimPolicy` |
| L3 | `torch/csrc/dispatch/` | ATen CUDA dispatch key overrides, original kernel capture, coverage guards, and environment-variable gating |

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
`sqrt` `rsqrt` `abs` `exp` `log` `reciprocal` `sigmoid` `tanh` `clamp` `round` `floor`
`ceil` `trunc` `sum` `mean` `native_dropout_backward` `tanh_backward`, plus `add_` `sub_` `mul_` `div_` `relu_`.

Floating-point `add`, `sub`, `add_`, and `sub_` support non-unit `alpha`.
The GPU VM uses an FP32 fused multiply-add for tensor and scalar operands, including broadcast inputs,
and rounds the result to the output dtype (FP32, FP16, or BF16). Tensor–tensor `add` and `add_`
join a graph when either input has a pending producer.

The GPU backend can encode a single scheduled region of 2–12 FP32 arithmetic instructions
as one `Pointwise` instruction. It supports add, subtract, multiply, divide, tanh, and fused
multiply-add. Selection depends on alignment, supported broadcasts, safe memory access,
and whether the live intermediate values fit in three saved slots plus the previous result.
This lets forward and backward GELU expressions use the same vectorized executor as other
pointwise chains. Other fused graphs use the regular GPU VM.

Other ATen operators use a generic guard: **flush first, then forward to the original
kernel captured before registration**.

Known limitations:

- Fusion accepts contiguous, dense CUDA tensors with at most 10 dimensions,
  matching DVM's shape storage capacity. Scalar division can also read an FP32
  tensor expanded along its last dimension from contiguous storage.
- `int64` arithmetic is not fused because the GPU VM has no int64 arithmetic instructions.
- Python scalars are fused only with floating-point tensors. PyTorch promotes
  `int32 / 2` to float32, which the VM cannot express while preserving those semantics.
- Reductions with empty, integer, or FP64 inputs, unsupported output dtypes,
  noncontiguous inputs, or deterministic algorithms enabled use native kernels.
- `sum` and `mean` join fusion only when their input is produced by the current
  pending graph. Reductions over already materialized tensors use native kernels,
  including when they would otherwise start a chain with later operators.
- `sum.out`, `mean.out`, and other reductions such as `amax` / `amin` execute after
  the generic guard flushes the pending graph.
- Functional `tanh` joins a pending graph when that graph produces its nonempty, contiguous FP32 input.
  The GPU VM uses CUDA `tanhf`. Standalone calls, other dtypes, and the `out=` and in-place variants
  use native PyTorch. The VM implements `sigmoid` using `exp`.
- `tanh_backward` joins fusion by default when the pending graph produces its upstream gradient.
  The gradient and saved tanh output must be nonempty, contiguous FP32 tensors of the same shape on the same GPU.
  The GPU VM evaluates `fma(-y, y, 1.0)` on the saved output and then multiplies by the upstream gradient.
  Standalone calls and the `out=` variant use native PyTorch.

## Performance

The V100 benchmark used a chain of 10 elementwise operators, keeping the
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

Use `torch_dvm.Explorer()` to inspect the graphs formed during a model step or an
operator chain. On entry, it resets the process-wide fusion counters and enables
fusion. On exit, it submits any pending graph and saves the statistics in `ex.report`.

```python
import torch
import torch_dvm

x = torch.ones(4, 8, device="cuda")
with torch_dvm.Explorer() as ex:
    y = (x * 2).sum(dim=1)

print(ex.report["graph_sizes"])  # {2: 1}
print(ex.report["by_reason"])    # {'Manual': 1}
```

Run the example again with `torch_dvm.Explorer(disable_ops="sum")` to see what
changes when `sum` runs natively.

- `graph_sizes` changes from `{2: 1}` to `{1: 1}`. By default, the multiplication
  and sum form one two-operator graph. With `sum` excluded, the graph contains only the multiplication.
- `by_reason` gains one `NonFusibleOp` entry for the flush requested before the
  native `sum`. In both runs, `Manual` counts the flush requested on exit from the `with` block.

The `emit` and `flush` counts in `ex.report["summary"]` give the total number of
operators and graphs submitted, respectively. If you enable fusion manually with
`torch_dvm.enable()`, you can call `torch_dvm.flush()` to submit the current graph
and `torch_dvm.disable()` to restore native dispatch when finished.

## Tests

```bash
python -m pytest tests/torch_eager_fusion -q
```
