# Command quick reference

Commands assume the tinygrad repository root and an editable installation. Set
`DEV` explicitly in investigations so logs and results identify the path used.

## Record the environment

```bash
git rev-parse HEAD
python3 --version
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
DEV=CUDA .venv/bin/python -c 'from tinygrad import Device; d=Device.DEFAULT; print(d, type(Device[d]).__name__, type(Device[d].renderer).__name__, Device[d].renderer.target)'
```

Repeat the last command with `DEV=NVK+NV` if investigating tinygrad's lower-level
NVIDIA backend. An initialization failure is useful evidence; do not hide it by
falling back to automatic device selection.

## Device and renderer routes

| Setting | Use |
| --- | --- |
| `DEV=PYTHON` | Execute lowered UOps in Python; slow but transparent semantic oracle. |
| `DEV=NULL` | Exercise scheduling/codegen without meaningful numerical execution. |
| `DEV=CPU` | Portable compiled execution through the host CPU path. |
| `DEV=CUDA` | NVIDIA execution through the CUDA Driver API and default renderer selection. |
| `DEV=CUDA:PTX` | CUDA runtime with the direct PTX renderer. |
| `DEV=NVK+NV` | NVIDIA execution through tinygrad's HCQ path, requiring the ordinary NVIDIA kernel-driver interface. |
| `DEV=PYTHON::sm_89` | Python execution using an NVIDIA Ada-oriented target for codegen/tensor-core correctness. |

The general syntax is `interface+device:renderer:architecture`. Empty components
are allowed, as in the double colon above. Target syntax is snapshot-sensitive;
confirm it in `Target.parse`, official runtime docs, and CI examples before
publishing a command.

Bare `DEV=NV` allows interface fallback in this snapshot. Do not use it when a
course command is meant to guarantee the kernel-driver route. `DEV=PCI+NV` is a
specialized direct-PCI path for dedicated driver work, not a routine fallback.

## Debug output

| Level | Adds in the recorded snapshot |
| ---: | --- |
| `DEBUG=1` | Device opening and high-level scheduling information. |
| `DEBUG=2` | Synchronized per-kernel timing, operation/memory estimates, and throughput. |
| `DEBUG=3` | Kernel AST/optimization information and applied opts. |
| `DEBUG=4` | Rendered source or printable assembly during program construction. |
| `DEBUG=5` | Earlier UOp/kernel representations in relevant code paths. |
| `DEBUG=6` | More detailed/lower representation output. |
| `DEBUG=7` | Compiler disassembly where the selected compiler implements it. |

Higher levels include lower ones and can produce enormous output. Begin at 2 or
3, minimize the workload, and capture output to a file when moving higher. The
precise presentation varies by renderer and has changed historically; inspect
the `DEBUG` guards in the code path under study.

Examples:

```bash
DEV=CPU DEBUG=3 .venv/bin/python your_reproducer.py
DEV=CUDA DEBUG=4 .venv/bin/python your_reproducer.py
DEV=CUDA DEBUG=2 .venv/bin/python your_benchmark.py
```

`DEBUG=2` changes timing behavior by waiting for execution. That is useful for
diagnosis but is not a substitute for a deliberately synchronized benchmark.

## Rewrite and profile visualization

```bash
VIZ=1 DEV=CPU DEBUG=0 .venv/bin/python your_reproducer.py
.venv/bin/python -m tinygrad.viz.cli
.venv/bin/python -m tinygrad.viz.cli -s TINY | rg Schedule
.venv/bin/python -m tinygrad.viz.cli -s TINY 'Schedule 1 Kernel n1' --ls
DEBUG=6 .venv/bin/python -m tinygrad.viz.cli -s TINY 'Schedule 1 Kernel n1'
DEBUG=7 .venv/bin/python -m tinygrad.viz.cli -s TINY 'Schedule 1 Kernel n1' 'pass name'
```

Names contain per-process counters; copy the actual name from the first listing.
For machine processing:

```bash
.venv/bin/python -m tinygrad.viz.cli --json > /tmp/tinygrad-events.jsonl
```

See the snapshot-pinned
[`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md)
for filters, intervals, markers, and backend-specific profiling.

## Focused validation modes

| Setting | Question it helps answer |
| --- | --- |
| `SPEC=2` | Does each checked graph satisfy the expected UOp specification? |
| `CHECK_OOB=1` | Can symbolic bounds prove indexed accesses remain in range? Requires the supported Z3 dependency. |
| `VALIDATE_WITH_CPU=1` | Does a compiled device kernel numerically agree with an inserted CPU shadow execution? |
| `DEBUG_RANGEIFY=1` | Why did range inheritance/materialization make this scheduling decision? |
| `NOOPT=1` | Does the problem disappear without kernel optimization? |
| `BEAM=0` / `BEAM=2` | Does heuristic versus measured kernel scheduling change the outcome? |
| `SCACHE=0` / `CCACHE=0` | Is a schedule/compiler cache obscuring the experiment? |
| `CAPTURE_PROCESS_REPLAY=1` | Capture compiler inputs for later comparison against another revision. |

Change one variable at a time and retain the correctness oracle. A flag that
makes a failure disappear narrows the stage; it does not prove the disabled
feature owns the root cause.

Many integer options can also be scoped inside Python:

```python
from tinygrad import Context

with Context(NOOPT=1, BEAM=0):
  result = workload()
```

Environment values are read when their `ContextVar` is created, usually during
import. Set shell variables before starting Python.

## Test routing

| Area | Typical location |
| --- | --- |
| Backend-neutral UOps, symbolic algebra, schedules | `test/null/` |
| Same operation semantics across runtimes | `test/backend/` |
| Tests run on one backend in CI | `test/unit/` |
| Kernel optimization and tensor cores | `test/opt/` |
| Allocators, queues, runtimes, hardware behavior | `test/device/` and `test/mockgpu/` |
| Fuzzers, process replay, models, large integrations | `test/external/` |

Start with the exact regression and its nearest existing file:

```bash
DEV=CPU .venv/bin/python -m pytest test/unit/test_example.py::TestExample::test_case -x -q
DEV=NULL SPEC=2 .venv/bin/python -m pytest test/null/test_schedule.py -x -q
DEV=CUDA .venv/bin/python -m pytest test/backend/test_ops.py -x -q
```

If `pytest-xdist` is installed, upstream asks agents to use twelve workers:

```bash
DEV=CPU .venv/bin/python -m pytest test/unit/ -x -q -n12
```

Broaden only after the focused test fails before the fix and passes after it.
Consult `.github/workflows/test.yml` for the current CI matrix rather than
assuming the full suite runs on one backend.

## Static checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy tinygrad/
pre-commit run --all-files
```

Use versions from the checkout's `pyproject.toml` extras. Formatting unrelated
code or fixing unrelated warnings makes a contribution harder to review.

## Find ownership and history

```bash
rg -n 'def create_linear_with_vars|class PatternMatcher' tinygrad test
rg -n 'Ops\.AFTER|DEBUG_RANGEIFY' tinygrad test
git log --oneline --follow -- tinygrad/schedule/rangeify.py
git blame -L 555,580 tinygrad/schedule/rangeify.py
git log -S 'PatternMatcher' --oneline --all -- tinygrad
```

Use `rg` to find definitions, callers, and tests before browsing directories.
Use history to recover intent after you understand the current behavior, not as
a replacement for reproducing it.

## Process replay outline

For a refactor or speed change, follow the exact current upstream instructions.
At this snapshot the local shape is:

1. run relevant tests on the branch with `CAPTURE_PROCESS_REPLAY=1`;
2. switch a separate checkout to the comparison revision; and
3. run `test/external/process_replay/process_replay.py` against the captured
   process inputs.

The authoritative details are in
[`test/external/process_replay/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md).
Do not switch revisions in a dirty working tree.
