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

启用融合后，GPU 后端可将只需一个向量 kernel 的 eager 图编码为一条 `Pointwise.fp32`
VM 指令，自动选择轻量 CUDA 入口。它支持 2 到 8 个 FP32 加、减、乘、除操作，以及标量、
按行和按末维的广播。每个线程一次处理四个相邻元素，因此主输出末维须为 4 的倍数，
向量化读写的地址须按 16 字节对齐。指令支持多个输出、最多两个需要跨操作保存的中间值，
最后一个操作可原地更新按相同下标访问的稠密输入。其他图沿用常规 VM 指令生成。
要对照常规 VM，可在启动进程前设置 `DVM_GPU_POINTWISE=0`。

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
`ceil` `trunc` `sum` `mean` `native_dropout_backward`，以及 `add_` `sub_` `mul_` `div_` `relu_`。

其余 aten 算子注册一条通用守卫：**先 flush 再转发到注册前捕获的原 kernel**。

同时覆盖 `mul.Scalar` 和 `mul.Tensor`。`eq.Tensor` / `eq.Scalar` 支持同类型的
浮点或 Bool 输入，Bool 与标量比较仅融合 0 和 1。
`masked_fill.Scalar` / `masked_fill_.Scalar` 支持浮点数据和可广播的 Bool mask。
这些路径要求输入为非空、连续的 CUDA 张量；混合 dtype、整数比较、Tensor 填充值
以及不支持的布局回退到原生 kernel。

标量除法还支持仅沿末维扩展、压回该维后底层连续的 FP32 输入。`div.Scalar` 和
标量包装为 Tensor 的 `div.Tensor` 路径都会读取紧凑输入，在图内完成除法和广播。
除数转换成 FP32 后必须有限且非零；其他非连续布局走原生，扩展张量不作为写入目标。

`native_dropout_backward` 仅在梯度已有 pending 生产者时加入融合，要求非空、连续
FP32 梯度、同 shape 的连续 Bool mask，以及转换成 FP32 后有限的 scale。这样可以
融合广播除法 → `add_` → dropout 反向，独立 dropout 反向仍走原生。仍存活的累加
结果及别名照常写回，死亡中间结果的写回由 DSE 删除。

待执行图中的 Bool 与标量 0/1 的比较结果用于 `masked_fill` 时，emit 会使用原始
Bool 条件并调整 Select 分支。反向保存的 mask 和其他仍被使用的比较结果照常写回。

`masked_fill_` 的目标精确匹配图内新产生的结果，且该 storage 在图中没有内存 Load 时，
可以直接在原图中更新。此前的消费者继续引用旧值节点，后续消费者使用新值。
写入已经被 Load 读取的 storage，或存在元数据不匹配的重叠 view 时，仍先 flush。

Tensor-Tensor `add`/`add_` 和归约（`sum`/`mean`）仅在输入由当前待执行图产生时
加入融合，否则使用原生 kernel；标量 add 路径仍可作为图的起点。该检查在流、别名
和图大小边界处理之后执行。Tensor-Tensor `add_` 的目标 storage 不在图中时，可以
消费精确匹配的 pending 只读输入；写入图中 storage 或读取不匹配的重叠 view 仍会 flush。
因此，以原生 add 开头的 add → mean 也会整体走原生；消费 pending 输入的 add 和
归约仍可融合。

`add`/`sub` 及其原地版本的 Tensor、Scalar 入口均支持传入 `alpha`。Tensor 输入
在同一图内缩放右操作数；Scalar 输入将系数折叠到常量。Int32 Tensor 输入保持
整数运算，并保留原生系数范围检查（减法检查 `-alpha`）；FP16/BF16 的缩放加减
使用 FP32 中间值，最后转换回输出类型。

浮点张量的非 1 alpha 需要 `compile_fast=True`。CUDA 原生 kernel 使用 FMA，
当前 VM 则执行分开的乘法和加减法，在舍入、抵消及中间溢出时可能有不同结果；
默认模式对此保留原生执行。非法 alpha 类型、alpha 转换溢出以及不支持的 dtype
或布局也保留原生检查与回退，生产者筛选和别名规则继续适用。

`detach` 调用原生元数据操作并保留待执行计算，使 storage 别名和版本计数保持
PyTorch 语义。通过 detached 别名写入以及需要读取实际数据的消费者，仍会按需
触发 flush。可通过 `disable_ops="detach"` 恢复 detach 处的 flush。

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
