# 参与 DVM 贡献

[English](CONTRIBUTING.md) | **简体中文**

DVM 欢迎问题反馈、文档完善、测试、新算子、编译器与运行时优化，以及框架集成等贡献。
没有上游仓库写权限也可以通过 Fork 参与。请在 Issue 和 Pull Request 中记录方案讨论、评审决定与验证结果，便于社区协作。

## 选择贡献方向

- **报告问题：** 先搜索[已有 Issue](https://github.com/mindspore-ai/dvm/issues)，再使用[缺陷模板](https://github.com/mindspore-ai/dvm/issues/new/choose)。提供 DVM 提交版本、环境、最小复现、预期与实际行为及相关日志。
- **改进文档或测试：** 提交范围明确的 PR。本指南的中英文版本需保持一致；接口变化时同步更新示例。
- **提议新特性或集成：** 较大实现开始前先提交 Feature Issue 或 RFC，说明使用场景、接口设计、支持的设备与数据类型、兼容性影响和验收标准，并邀请相关[维护者](CODEOWNERS)评审方案。
- **优化编译或执行性能：** 说明工作负载与瓶颈，在相同硬件、Shape、数据类型和配置下，对比正确性、编译开销、执行耗时及适用的内存指标。

不确定修改归属时，可在 Issue 中描述需求，并参考 [CODEOWNERS](CODEOWNERS)。请保持尊重、回应技术意见，并将决定记录在 Issue 或 PR 中。

## 准备开发环境

先在 GitHub 上 Fork [DVM](https://github.com/mindspore-ai/dvm)，替换以下命令中的 `YOUR_GITHUB_USERNAME`，从目标上游分支创建开发分支。
普通开发面向 `master`；版本修复请先与维护者确认目标分支。

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/dvm.git
cd dvm
git remote add upstream https://github.com/mindspore-ai/dvm.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

以下命令均在仓库根目录、已激活的 Python 环境中执行。先安装依赖，再加载 DVM 环境脚本，以便脚本找到 pybind11：

```bash
python -m pip install -r requirements.txt
```

### Ascend NPU

按照 README 的[环境配置](README.md#环境配置)准备 Linux、Ascend NPU 硬件、CANN 8.5 或更高版本，以及支持 C++17 的编译器。
已验证的编译器版本和支持的 Ascend 硬件见 README。将以下 CANN 路径替换为实际安装路径：

```bash
source /path/to/ascend-toolkit/set_env.sh
source env.sh
make -j8
python -c "import dvm; print(dvm.__file__)"
DEVICE_ID=0 python examples/01_add.py
```

[Makefile](Makefile) 会生成 `libdvm.a` 和 `_dvm_py.so`，并将 Python 扩展复制到 `python/dvm/`。
[env.sh](env.sh) 将当前代码目录加入 `PYTHONPATH`；导入路径应指向本工作区的 `python/dvm/__init__.py`。
修改原生代码后需重新编译。运行示例或测试时通过 `DEVICE_ID` 选择可用设备。
如需使用 Clang 编译 host 代码，可执行 `CXX=clang++ make -j8`。

仿真调试见[开发技巧](docs/development.md)，Python API 见[用户手册](docs/tutorial.md)，
链接 `libdvm.a` 见 [C++ 示例](examples/cc/README.md)，框架接入见 [TorchNPU 指南](docs/pytorch.md)。

### CUDA 后端开发

仓库提供 [env_gpu.sh](env_gpu.sh) 和 `DVM_ARCH=gpu` 构建路径。修改该后端时，请准备兼容的 NVIDIA GPU、驱动、CUDA Toolkit、host 编译器及 Python 环境。
如果脚本无法自动找到 Toolkit，请设置 `CUDA_HOME`。将以下 `sm_70` 替换为设备与 Toolkit 支持的目标架构：

```bash
source env_gpu.sh
make clean
make -j8 DVM_ARCH=gpu CUDA_ARCH=sm_70
python -c "import dvm; print(dvm.__file__)"
DEVICE_ID=0 python -m pytest -v tests/ut/test_elemwise.py -m gpu
```

两个后端共用构建产物名称，切换后端时需清理并重新构建；切回 Ascend 时，使用新 Shell，或先清除 `DVM_ARCH` 再加载 `env.sh`。
CUDA 验证请选择带 `gpu` 标记的测试，并明确记录已验证配置与不支持的用例。

## 遵循代码与文档约定

- 遵循周围代码风格，C++ 格式参考仓库 [.clang-format](.clang-format)。保留许可头，新源文件添加适当的许可头。
- 保持修改聚焦。Bug 修复补充回归用例，新算子、数据类型、Shape、编译优化和运行时行为补充相应测试。
- 修改公开 API 时，同步维护 [C++ 接口](include/dvm.h)、[Python 绑定](src/pybind_api.cc)、[Python 类型声明](python/dvm/_dvm_py.pyi)、文档和示例。
- 说明设备限制、数值容差、兼容性变化和新增依赖。贡献指南保持中英文同步，其他文档沿用其现有语言。
- 检查 Markdown 格式、相对链接、锚点和示例命令；提交前执行 `git diff --check`。

## 验证修改

构建 DVM 后，根据受影响的行为选择测试。`tests/ut/` 中不少用例会执行设备 Kernel，不能仅凭目录名称认为它们可在纯 CPU 环境运行。
请检查具体测试的设备要求及 [pytest.ini](tests/ut/pytest.ini) 中的标记。

| 修改类型 | 所需验证 |
| --- | --- |
| 纯文档 | Markdown 格式、链接、锚点和修改的命令或示例；说明设备测试不适用的原因。 |
| 算子、Tiling、Pass 或代码生成 | `tests/ut/` 中的相关测试，Bug 修复需有回归用例；覆盖相关 Shape、数据类型和设备架构。 |
| 运行时或集成行为 | 在目标硬件运行 `tests/st/` 中的相关用例及受影响的 Python 或 C++ 示例。 |
| 集合通信 | 单独选择 `tests/ut/comm/` 中的测试，准备所需设备与通信环境。 |
| 性能或内存优化 | 正确性检查及可复现的修改前后对比；按场景选用 `tests/perf/` 或 `tests/fuzz/` 中的相关用例。 |
| 构建或依赖修改 | 在声明的环境中重新构建、导入本地扩展并运行相关示例。 |

例如先运行受影响的 Ascend 算子测试，再按修改范围扩大覆盖：

```bash
DEVICE_ID=0 python -m pytest -v tests/ut/test_elemwise.py
DEVICE_ID=0 python -m pytest -v tests/ut/
DEVICE_ID=0 python -m pytest -v tests/st/
```

单元测试配置排除了对 `comm` 和 `hccl` 目录的递归发现，通信测试需要单独运行。
使用内置进程启动方式时，通过 `ASCEND_RT_VISIBLE_DEVICES` 选择可用 Ascend 设备。例如在设备 0 和 1 可用的机器上：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest -v tests/ut/comm/test_allreduce.py
```

[通信 fixture](tests/ut/comm/conftest.py) 在非 MPI 环境使用 `CommScope`，在 MPI 作业中使用 `mpi4py`。
内置启动器已负责创建工作进程，请按具体测试选择启动方式和设备数量。
[run_all_test.sh](tests/ut/run_all_test.sh) 固定设置 `DEVICE_ID=3`，并使用相对于 `tests/ut/` 的路径，使用前需检查这些前提。
修改 GitHub 门禁脚本时，运行其已有测试：

```bash
node --test .github/scripts/ci-label-expiry.test.mjs .github/scripts/reviewer-approval-gate.test.mjs
```

在 PR 中记录被测提交、命令、结果、设备型号与数量、Python 和编译器版本及相关运行时版本。
Ascend 需记录 CANN 和驱动版本，NVIDIA GPU 需记录 CUDA 和驱动版本，集成改动需记录框架版本。
性能结果应包含 Shape、数据类型、预热与测量方式。明确列出缺少的硬件及未运行或跳过的测试，并与维护者协调必要验证。

## 提交并更新 Pull Request

1. 每个 PR 聚焦一个可独立评审的修改。使用清晰的提交主题，例如 `docs: clarify DVM build steps` 或 `fix: correct workspace size calculation`。
2. 推送到个人 Fork，向正确的目标分支发起 PR。使用[默认模板](.github/PULL_REQUEST_TEMPLATE.md)、[英文模板](.github/PULL_REQUEST_TEMPLATE.en.md)或[中文模板](.github/PULL_REQUEST_TEMPLATE.zh-CN.md)。将 `/kind <label>` 替换为对应类型，说明问题、修改后行为及实际测试结果。关联真实 Issue；没有适用 Issue 时明确说明。自检项如实填写，不适用项说明原因。
3. 设计、实现或必要验证尚未完成时使用 Draft PR；准备好后向相关[维护者](CODEOWNERS)请求评审。
4. 回应评审意见，同步更新测试和文档。推送修改后，为最新提交取得有效的 CI 和评审结果，问题处理完成后再解决评审讨论。

在 GitCode 提交时使用对应的[模板](.gitcode)。若同一修改在两个平台都有请求，请相互关联，并与维护者协调更新。

## CI、评审与合入

仓库中的 GitHub 工作流实现了以下检查。哪些检查与批准属于必需项，以目标分支的实时[仓库规则](https://github.com/mindspore-ai/dvm/rules)为准。

- **`label-check`：** [标签工作流](.github/workflows/label-check.yml)要求有效的 `ci-pipeline-passed` 或 `ci-pipeline-passed-but-overtime` 标签。标签在添加后 24 小时过期；新增推送会移除旧的 CI 通过标签，需要新的验证结果。超时标签保留耗时异常信息。
- **`all-reviewers-approved`：** [评审门禁](.github/scripts/reviewer-approval-gate.mjs)在一名评审人时要求一次批准，两名及以上评审人时要求两名不同评审人批准。批准必须针对当前提交，作者本人不计入。未解决的修改请求、待处理的团队评审请求及 Draft 状态都会阻断。适用分支与自动合入设置见[评审策略](.github/reviewer-approval-policy.json)。
- **分支规则：** 单项检查通过不能替代目标分支的其他要求。申请合入前，请确认 PR 的最新检查与评审状态。

验证失败时，先查看失败阶段与日志。代码回归由作者与相关维护者处理，执行资源、环境和结果回报问题请联系 CI 维护者。
通过仓库支持的流程申请重测，并确认结果对应最新提交。不要手工添加通过标签或关闭检查来促成合入。

## 兼容性、版本与问题跟进

通过[版本说明](RELEASE.md)了解版本变化，发布目标及将修复移植到版本分支的安排需和维护者确认。
公开 API、编译器、运行时或构建方式变化应说明兼容边界与迁移步骤。
框架集成应提供可运行示例，并记录 DVM 与下游框架的精确版本及验证结果；参考 [TorchNPU 接入指南](docs/pytorch.md)。

可复现缺陷和跨 PR 问题通过 Issue 跟踪，提供环境信息、最小复现和脱敏日志。
在原 Issue 中记录根因、受影响与修复版本，以及回归验证结果。

## 许可证

贡献遵循仓库的 [Apache 2.0 许可证](LICENSE)。引入第三方代码时保留必要署名与许可信息，
并按需更新[第三方软件声明](Third_Party_Open_Source_Software_Notice)。
