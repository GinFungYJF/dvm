# Contributing to DVM

**English** | [简体中文](CONTRIBUTING.zh.md)

DVM welcomes bug reports, documentation, tests, new operators, compiler and runtime improvements, and framework
integrations. You can contribute through a fork without write access to the upstream repository. Use issues and pull
requests to keep design discussions, review decisions, and validation results available to the community.

## Choose a contribution

- **Report a problem:** Search [existing issues](https://github.com/mindspore-ai/dvm/issues), then use the
  [bug-report template](https://github.com/mindspore-ai/dvm/issues/new/choose). Include the DVM commit, environment,
  minimal reproduction, expected and actual behavior, and relevant logs.
- **Improve documentation or tests:** Submit a focused PR. Keep this guide and its Chinese translation consistent,
  and update examples when the interfaces they use change.
- **Propose a feature or integration:** Open a Feature Issue or RFC before substantial implementation. Describe the
  use case, proposed interfaces, supported devices and data types, compatibility impact, and acceptance criteria.
  Ask the relevant [maintainers](CODEOWNERS) to review the design.
- **Optimize compilation or execution:** Describe the workload and bottleneck. Compare correctness, compilation
  overhead, execution time, and memory use as applicable under the same hardware, shapes, data types, and configuration.

If you are unsure where a change belongs, describe it in an issue and refer to [CODEOWNERS](CODEOWNERS).
Keep discussion respectful, respond to technical feedback, and record decisions in the issue or PR.

## Prepare your development environment

Fork [DVM on GitHub](https://github.com/mindspore-ai/dvm), replace `YOUR_GITHUB_USERNAME` below, and create a branch
from the intended upstream target. Use `master` for normal development; agree on the target branch with maintainers
for release fixes.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/dvm.git
cd dvm
git remote add upstream https://github.com/mindspore-ai/dvm.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

Run the following commands from the repository root in an activated Python environment. Install dependencies before
sourcing the DVM environment script so that it can locate pybind11:

```bash
python -m pip install -r requirements.txt
```

### Ascend NPU

Follow the [environment requirements](README.md#环境配置): Linux, Ascend NPU hardware, CANN 8.5 or later, and a
C++17 compiler. The README lists the tested compiler versions and supported Ascend hardware. Replace the CANN path
below with your installation path:

```bash
source /path/to/ascend-toolkit/set_env.sh
source env.sh
make -j8
python -c "import dvm; print(dvm.__file__)"
DEVICE_ID=0 python examples/01_add.py
```

The [Makefile](Makefile) builds `libdvm.a` and `_dvm_py.so`, and copies the Python extension into `python/dvm/`.
[env.sh](env.sh) adds this checkout to `PYTHONPATH`; the printed import path should point to `python/dvm/__init__.py`
in this checkout. Rebuild after native-code changes. Select an available device with `DEVICE_ID` when running examples
or tests. To use Clang for host compilation, run `CXX=clang++ make -j8`.

See the [development notes](docs/development.md) for simulator debugging, the [user guide](docs/tutorial.md) for the
Python API, the [C++ example](examples/cc/README.md) for linking `libdvm.a`, and the
[TorchNPU guide](docs/pytorch.md) for framework integration.

### CUDA backend development

The repository also provides [env_gpu.sh](env_gpu.sh) and the `DVM_ARCH=gpu` build path. For changes to that backend,
prepare a compatible NVIDIA GPU, driver, CUDA toolkit, host compiler, and Python environment. Set `CUDA_HOME` if the
toolkit is not found automatically. Replace `sm_70` below with the architecture supported by your GPU and toolkit:

```bash
source env_gpu.sh
make clean
make -j8 DVM_ARCH=gpu CUDA_ARCH=sm_70
python -c "import dvm; print(dvm.__file__)"
DEVICE_ID=0 python -m pytest -v tests/ut/test_elemwise.py -m gpu
```

The two backends share build output names. Clean and rebuild when switching backends; when returning to Ascend,
use a fresh shell or unset `DVM_ARCH` before sourcing `env.sh`. Select GPU-marked tests for CUDA validation and report
the tested configuration and unsupported cases explicitly.

## Follow the code and documentation conventions

- Follow the surrounding code and the repository's [.clang-format](.clang-format) for C++ formatting. Preserve license
  headers and add appropriate headers to new source files.
- Keep changes focused. Add regression coverage for bug fixes and tests for new operators, data types, shapes,
  compiler passes, and runtime behavior.
- Keep the [C++ interfaces](include/dvm.h), [Python bindings](src/pybind_api.cc),
  [Python type declarations](python/dvm/_dvm_py.pyi), documentation, and examples consistent when changing public APIs.
- Document device restrictions, numerical tolerances, compatibility changes, and new dependencies. Keep the English
  and Chinese contribution guides in sync; retain the established language of other documents.
- Check Markdown formatting, relative links, anchors, and example commands. Run `git diff --check` before submission.

## Validate your change

Run tests appropriate to the affected behavior after building DVM. Many tests in `tests/ut/` execute device kernels;
the directory name does not imply that they can run on a CPU-only machine. Check each test's device requirements and
the markers in [pytest.ini](tests/ut/pytest.ini).

| Change | Expected validation |
| --- | --- |
| Documentation only | Markdown formatting, links, anchors, and changed commands or examples; explain why device tests are not applicable. |
| Operators, tiling, passes, or code generation | Affected tests in `tests/ut/`, including a regression case for a bug fix; cover relevant shapes, data types, and device architectures. |
| Runtime or integration behavior | Relevant tests in `tests/st/` and affected Python or C++ examples on the target hardware. |
| Collective communication | Explicitly selected tests in `tests/ut/comm/` with the required devices and communication environment. |
| Performance or memory optimization | Correctness tests and reproducible before/after measurements; use relevant cases in `tests/perf/` or `tests/fuzz/` as appropriate. |
| Build or dependency changes | Rebuild, import the local extension, and run relevant examples in the declared environment. |

For example, run the affected Ascend operator tests, then broaden coverage when the change requires it:

```bash
DEVICE_ID=0 python -m pytest -v tests/ut/test_elemwise.py
DEVICE_ID=0 python -m pytest -v tests/ut/
DEVICE_ID=0 python -m pytest -v tests/st/
```

The unit-test configuration excludes recursive discovery of `comm` and `hccl`. Run communication cases separately.
For the built-in process launcher, select available Ascend devices through `ASCEND_RT_VISIBLE_DEVICES`; for example,
on a machine where devices 0 and 1 are available:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest -v tests/ut/comm/test_allreduce.py
```

The [communication fixture](tests/ut/comm/conftest.py) uses `CommScope` outside MPI and `mpi4py` inside an MPI job.
The built-in launcher already creates worker processes. Use the launcher and device count appropriate to the test.
[run_all_test.sh](tests/ut/run_all_test.sh) sets `DEVICE_ID=3` and uses paths relative to `tests/ut/`; check those
assumptions before using it. For changes to the GitHub gate scripts, run their existing tests:

```bash
node --test .github/scripts/ci-label-expiry.test.mjs .github/scripts/reviewer-approval-gate.test.mjs
```

Record the tested commit, commands, results, device model/count, Python and compiler versions, and relevant runtime
versions in the PR. Include CANN and driver versions for Ascend, CUDA and driver versions for NVIDIA GPU, and framework
versions for integration changes. For performance results, include shapes, data types, warm-up and measurement methods.
List unavailable hardware and unrun or skipped tests explicitly; coordinate required validation with maintainers.

## Submit and update a pull request

1. Keep each PR focused on one independently reviewable change. Use a clear commit subject, such as
   `docs: clarify DVM build steps` or `fix: correct workspace size calculation`.
2. Push to your fork and open a PR against the intended branch. Use the
   [default template](.github/PULL_REQUEST_TEMPLATE.md), [English template](.github/PULL_REQUEST_TEMPLATE.en.md), or
   [Chinese template](.github/PULL_REQUEST_TEMPLATE.zh-CN.md). Replace `/kind <label>` with the appropriate kind,
   describe the problem and resulting behavior, and provide actual test results. Link real issues and explain when
   none applies. Complete checklist items truthfully and explain items that are not applicable.
3. Keep the PR in Draft while its design, implementation, or required validation is incomplete. Request review from
   the relevant [maintainers](CODEOWNERS) when ready.
4. Address review feedback and update tests and documentation together. After pushing changes, obtain fresh CI and
   review results for the latest commit, and resolve discussions once their concerns have been addressed.

For GitCode submissions, use the corresponding [templates](.gitcode). If a change is tracked on both platforms,
link the corresponding requests and coordinate updates with maintainers.

## CI, review, and merge

The checked-in GitHub workflows implement the following checks. The target branch's
[repository rules](https://github.com/mindspore-ai/dvm/rules) determine which checks and approvals are required.

- **`label-check`:** The [label workflow](.github/workflows/label-check.yml) requires a valid `ci-pipeline-passed`
  or `ci-pipeline-passed-but-overtime` label. Labels expire 24 hours after being added. A new push removes previous
  CI pass labels and requires a fresh result. The overtime label retains the indication of an abnormal duration.
- **`all-reviewers-approved`:** The [review gate](.github/scripts/reviewer-approval-gate.mjs) requires one approval
  when there is one reviewer, or two distinct approvals when there are two or more reviewers. Approvals must apply to
  the current commit, and the author does not count. Outstanding change requests, pending team-review requests, and
  Draft status block the gate. Target branches and automatic merge settings are in the
  [review policy](.github/reviewer-approval-policy.json).
- **Branch rules:** Passing an individual check does not replace the target branch's other requirements. Confirm the
  PR's current checks and review status before requesting a merge.

When validation fails, inspect the failing stage and logs. Address code regressions with the relevant maintainer;
ask CI maintainers to investigate runner, environment, or reporting failures. Request a retest through the repository's
supported process and verify that the result applies to the latest commit. Do not add pass labels manually or disable
checks to make a PR mergeable.

## Compatibility, releases, and follow-up

Use the [release notes](RELEASE.md) to understand version changes, and agree on release targets or backports with
maintainers. For public API, compiler, runtime, or build changes, document compatibility boundaries and any migration
steps. Framework integrations should include a runnable example and results for the exact DVM and downstream versions;
see the [TorchNPU integration guide](docs/pytorch.md).

Track reproducible defects and cross-PR problems in issues. Include environment details, minimal reproductions, and
sanitized logs. Record the root cause, affected and fixed versions, and regression results in the original issue.

## License

Contributions are made under the repository's [Apache 2.0 license](LICENSE). Preserve required attribution and license
information when adding third-party code, and update the
[third-party notices](Third_Party_Open_Source_Software_Notice) when applicable.
