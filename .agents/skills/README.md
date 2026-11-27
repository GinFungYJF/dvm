# DVM 开发技能目录

DVM 开发技能分为设计与开发、问题调试、代码仓质量三类，共六项。

## 设计与开发

| 技能名称 | Skill 标识 | 适用场景 | 预期结果 |
| --- | --- | --- | --- |
| 融合算子开发 | [dvm-kernel-develop](dvm-kernel-develop/) | 已有计算表达式或算子规格 | 实现可调用的 DVM Kernel |
| 算子优化 | [dvm-kernel-optimization](dvm-kernel-optimization/) | 需要进一步降低现有算子延迟或提升吞吐 | 交付优化实现、正确性对照和相对原实现的实测收益 |

## 问题调试

| 技能名称 | Skill 标识 | 适用场景 | 预期结果 |
| --- | --- | --- | --- |
| 性能调试 | [dvm-perf-debug](dvm-perf-debug/) | 调用耗时异常、吞吐低于预期或新版本出现性能回退 | 定位并修复根因，在原场景验证性能是否恢复 |
| 精度调试 | [dvm-precision-debug](dvm-precision-debug/) | 输出超出容差、出现异常值或写回错误 | 定位偏差、修复根因，用失败输入和参考结果验证 |
| 编译与运行调试 | [dvm-kernel-debug](dvm-kernel-debug/) | 已进入 DVM，出现断言、编译或执行错误 | 确定失败阶段、触发条件和复现范围 |

## 代码仓质量

| 技能名称 | Skill 标识 | 适用场景 | 预期结果 |
| --- | --- | --- | --- |
| 单元测试生成 | [dvm-ut-generator](dvm-ut-generator/) | 需要覆盖新行为或防止已知缺陷复发 | 生成带数值或图结构断言的 pytest 用例 |
