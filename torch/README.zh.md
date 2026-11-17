# dvm-torch：原生 PyTorch 的无图（eager）算子融合

[English](README.md) | **简体中文**

`dvm-torch` 在**不修改 PyTorch 源码、不做张量子类化**的前提下，把相邻的 eager 算子
合并到 DVM kernel 中。可直接融合的链条使用一次 launch，归约依赖由 DVM 自动拆分执行。
张量始终是真实的 `at::Tensor`，metadata 与 `data_ptr`
都有效，只有内容是延迟计算的。

## 安装

```bash
pip install dvm-torch
```

`dvm-torch` 在 import 时加载 PyTorch 和扩展依赖。算子覆盖由 `TORCH_DVM_FUSION`
或 `enable()` 启用，默认使用原生 PyTorch 分派。

```python
import torch, dvm_torch

dvm_torch.enable()                  # 安装 CUDA 覆盖
y = (a * b + c).relu()              # 三个算子并入一张图
dvm_torch.flush()                   # 或由下一个不可融合算子/同步点触发
print(dvm_torch.stats()["summary"])  # emit=3 flush=1 sizes={3op*1}
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

`L1` 不感知 `L2`，`L2` 不感知 `L3`。`core-purity` 门禁保证 `include/`、`src/` 中
不出现 `torch/`、`at::`、`c10::`、`ATen`。

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
`sqrt` `rsqrt` `abs` `exp` `log` `reciprocal` `sigmoid` `clamp` `round` `floor`
`ceil` `trunc` `sum` `mean`，以及 `add_` `sub_` `mul_` `div_` `relu_`。

其余 aten 算子注册一条通用守卫：**先 flush 再转发到注册前捕获的原 kernel**。

已知边界：

- 融合输入为连续的密集 CUDA 张量，最多 10 维，与 DVM 的形状存储容量一致。
- `int64` 不参与算术融合（GPU VM 无 int64 算力指令）。
- Python 标量只对浮点张量融合：`int32 / 2` 在 torch 中提升为 float32，
  VM 无法在不改变语义的前提下表达。
- 空输入、整数或 FP64 归约、其他输出 dtype、非连续输入以及确定性算法模式走原生归约。
- `sum.out` / `mean.out` 和 `amax` / `amin` 等其余归约由通用守卫触发 flush 后执行。
- `tanh` / `sigmoid` 由 `exp` 合成。`tanh` 默认关闭，可由 `compile_fast=True` 启用。

## 性能

初版 V100 实测采用 10 算子 elementwise 链，结果保持存活（避免死存储消除改变工作量）。
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

捕获的图在 flush 时通过 DVM 执行，包括仅含一个算子的图。
死存储消除（DSE）跳过不再可观察的中间张量写入，并保留存活别名和
autograd 保存的值。可使用 `dvm_torch.enable(disable_ops="sum")`
让指定算子保持原生执行。

```python
with dvm_torch.Explorer() as ex:
    run_model()
print(ex.report["summary"])     # emit / flush / bails / 图大小直方图
print(ex.report["by_reason"])   # 按 FlushReason 分组，定位边界热点
```

`dvm_torch.flush()` 可手工物化；`dvm_torch.disable()` 完整还原分派表。

## 测试

```bash
python -m pytest tests/fusion -q
```

`tests/fusion/test_escape.py` 是最关键的一组：它验证 Pending 张量逃逸到
`linear`、`softmax`、`amax`、`conv2d`、`.item()` 时结果仍与 eager 一致。
`tests/fusion/test_reductions.py` 验证归约维度、dtype、归约与广播的依赖、
存活中间结果以及原生回退路径，并检查归约确实进入融合图。
`tests/fusion/test_dse.py` 验证无用 Store 消除，以及对存活别名和
autograd 保存值的保护。
`tests/fusion/test_sync.py` 验证同步点独立触发物化、两条非默认流与 eager 对拍、
CUDA Graph capture / replay 边界以及挂钩的安装和恢复。
