# torch_dvm：原生 PyTorch 的无图（eager）算子融合

[English](README.md) | **简体中文**

`torch_dvm` 在**不修改 PyTorch 源码、不做张量子类化**的前提下，把相邻的 eager 算子
合并到 DVM kernel 中。可直接融合的链条使用一次 launch，归约依赖由 DVM 自动拆分执行。
张量始终是真实的 `at::Tensor`，metadata 与 `data_ptr`
都有效，只有内容是延迟计算的。

## 安装

```bash
pip install torch_dvm
```

wheel 已包含 DVM CUDA 后端。要从源码构建，请在装有 CUDA 版 PyTorch 和 CUDA 工具链的环境中，
从仓库根目录执行以下命令。

```bash
bash build.sh -t
python -m pip install dist/torch_dvm-*.whl
```

导入 `torch_dvm` 时会加载 PyTorch 和扩展依赖。算子覆盖由 `TORCH_DVM_FUSION`
或 `enable()` 启用，默认使用原生 PyTorch 分派。

```python
import torch
import torch_dvm

torch_dvm.enable()                  # 安装 CUDA 覆盖
y = (a * b + c).relu()              # 三个算子并入一张图
torch_dvm.flush()                   # 或由下一个不可融合算子/同步点触发
print(torch_dvm.stats()["summary"])  # emit=3 flush=1 sizes={3op*1}
```

也可以只用环境变量，在 import 前设置：

```bash
TORCH_DVM_FUSION="True" python train.py
TORCH_DVM_FUSION="True level=O1 disable_ops=mul,div" python train.py
```


## 架构

| 层 | 位置 | 职责 |
|---|---|---|
| L1 | `include/`、`src/` | 纯 C++：`dvm::Kernel`、codegen、launch。**零 torch 依赖** |
| L2 | `torch/csrc/fusion/` | 设备无关融合核：`SideTable`、`GraphKernel`、`GraphManager`、`OpEmitter`、`TrimPolicy` |
| L3 | `torch/csrc/dispatch/` | aten CUDA key 覆盖、原始 kernel 捕获、全量守卫、环境变量门控 |


## 正确性机制

| 机制 | 覆盖的失败模式 |
|---|---|
| 分派前 flush（命中/未命中注册表都 flush） | Pending 被未覆盖的算子消费 → 静默错数 |
| `need_flush_for_input`（匹配 storage offset、numel、shape、dtype） | view 部分重叠或形状变化 |
| `need_flush_for_writable`（查所有操作数，含只读操作数） | `a.add_(b)` 中 `b` 仍 Pending |
| host-read 算子强制 flush | `.item()` / `.cpu()` 读到 0 |
| Python Event / Stream 同步 API 前 flush | 同步点越过尚未提交的 Pending 计算 |
| capture 期间整体禁用融合 | CUDA Graph replay 重复执行 |
| 死存储消除（weak anchor + storage 引用计数） | 只落地真正活着的值 |
| 无 Store 的退化图直接丢弃 | DVM 无法对空图 codegen |

## 覆盖范围

融合集合（手写 unboxed 包装器）：

`add` `sub` `mul` `div` `maximum` `minimum` `pow(Tensor,Scalar)` `neg` `relu`
`sqrt` `rsqrt` `abs` `exp` `log` `reciprocal` `sigmoid` `tanh` `clamp` `round` `floor`
`ceil` `trunc` `sum` `mean` `native_dropout_backward` `tanh_backward`，以及 `add_` `sub_` `mul_` `div_` `relu_`。

浮点 `add`、`sub`、`add_` 和 `sub_` 支持非单位 `alpha`。GPU VM 以 FP32 执行融合乘加，
支持张量、标量及广播输入，最后将结果舍入到输出类型（FP32、FP16 或 BF16）。两个张量之间的
`add` 和 `add_` 需要至少一个输入由当前待执行图产生才能入图。

GPU 后端可以将单个调度区域内的 2 至 12 条 FP32 算术指令编码为一条 `Pointwise` 指令，
支持加、减、乘、除、tanh 和融合乘加。后端检查内存对齐、广播方式和读写安全性，并判断
同时存活的中间结果能否放入三个保存槽和一个前序结果槽。满足条件的 GELU 前向、反向和
其他逐元素链共用同一套向量化执行逻辑。其余融合图使用常规 GPU VM。

其余 aten 算子注册一条通用守卫：**先 flush 再转发到注册前捕获的原 kernel**。


已知边界：

- 融合输入为连续的密集 CUDA 张量，最多 10 维，与 DVM 的形状存储容量一致。
  上述只读标量除法广播是支持的非连续输入例外。
- `int64` 不参与算术融合（GPU VM 无 int64 算力指令）。
- Python 标量只对浮点张量融合：`int32 / 2` 在 torch 中提升为 float32，
  VM 无法在不改变语义的前提下表达。
- 空输入、整数或 FP64 归约、其他输出 dtype、非连续输入以及确定性算法模式走原生归约。
- `sum` 和 `mean` 仅在输入由当前待执行图产生时参与融合。对已物化张量的归约
  使用原生 kernel，包括原本可以由归约开始、再连接后续算子的链。
- `sum.out` / `mean.out` 和 `amax` / `amin` 等其余归约由通用守卫触发 flush 后执行。
- 当前待执行图产生非空、连续的 FP32 输入时，函数式 `tanh` 自动入图，由 GPU VM 执行 CUDA `tanhf`。
  独立调用、其他数据类型、`out=` 和原地版本走 PyTorch 原生路径。`sigmoid` 由 VM 使用 `exp` 合成。
- 当上游梯度由当前待执行图产生时，`tanh_backward` 默认参与融合。梯度和前向保存的 `tanh` 输出
  须为同一 GPU 上形状相同、非空、连续的 FP32 张量。GPU VM 对保存的输出 `y` 计算
  `fma(-y, y, 1.0)`，再乘以上游梯度。独立调用和 `out=` 重载走 PyTorch 原生路径。

## 性能

V100 实测采用 10 算子 elementwise 链，结果保持存活（避免死存储消除改变工作量）。
以下数据来自归约接入之前的实现。

| N | eager | fused | 比值 |
|---|---|---|---|
| 65536 | 0.285 ms | 0.276 ms | 1.03× |
| 1048576 | 0.286 ms | 0.231 ms | 1.24× |
| 4194304 | 0.880 ms | 0.249 ms | 3.54× |

kernel 数：20 个 eager launch → 1 个 `dvm_vm_gpu`。

**收益随张量增大而上升**，因为 DVM GPU 后端是 VM 解释器（单 kernel 内逐条解码），
小张量下 host 侧建图与 codegen 占主导。

## 调试

用 `torch_dvm.Explorer()` 查看一次模型迭代或一段算子链的成图情况。进入 `with` 块时，
它会清零当前进程的融合计数并启用融合。退出时，它会提交待执行的图，并将统计结果写入 `ex.report`。

```python
import torch
import torch_dvm

x = torch.ones(4, 8, device="cuda")
with torch_dvm.Explorer() as ex:
    y = (x * 2).sum(dim=1)

print(ex.report["graph_sizes"])  # {2: 1}
print(ex.report["by_reason"])    # {'Manual': 1}
```

改用 `torch_dvm.Explorer(disable_ops="sum")` 再运行一次，可以看到 `sum` 走原生实现时的变化。

- `graph_sizes` 从 `{2: 1}` 变为 `{1: 1}`。默认情况下，乘法和求和组成一张双算子图。禁用 `sum` 融合后，图中只剩乘法。
- `by_reason` 多出一次 `NonFusibleOp`，对应执行原生 `sum` 前的 flush 请求。两次运行中的 `Manual` 都来自退出 `with` 块时的 flush 请求。

`ex.report["summary"]` 中的 `emit` 和 `flush` 分别给出已提交的算子总数和图数。
如果用 `torch_dvm.enable()` 手动启用融合，可以调用 `torch_dvm.flush()` 提交当前图，调试结束后用 `torch_dvm.disable()` 恢复原生分派。

## 测试

```bash
python -m pytest tests/torch_eager_fusion -q
```
