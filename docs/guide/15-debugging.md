# 15. Debugging across the pipeline

## Purpose

A tinygrad failure often becomes visible far from its cause. A frontend mistake
can survive until generated code runs; a correct lowered kernel can be
misrendered; a correct ordinary execution can fail only after TinyJit capture.
The useful debugging question is therefore not “which large subsystem looks
suspicious?” but:

> What is the first artifact whose invariant is false?

This chapter gives you an evidence ladder for answering that question, a
minimization method that preserves the failure, and a controlled
predict-observe-change-regress exercise. It applies to wrong values, exceptions,
illegal graphs, compilation failures, crashes, and JIT-only failures.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

Continue only if you can trace an ordinary realization through:

```text
Tensor UOp graph
  → scheduled LINEAR calls
  → lowered program UOps
  → rendered SOURCE and BINARY
  → runtime load/launch
```

You should also be able to distinguish a backend, renderer, compiler, and
runtime, and understand TinyJit's ignore/capture/replay calls. Review
[the first trace](03-first-trace.md), [lowering](10-lowering.md),
[rendering](11-rendering.md), [runtimes](12-runtime.md), and
[TinyJit](13-jit.md) until you can name the artifact on both sides of each
arrow.

For Python control-flow or state bugs, the official
[`pdb` documentation](https://docs.python.org/3/library/pdb.html) is the
bounded prerequisite: learn breakpoints, `where`, `up`/`down`, `print`, `next`,
`step`, `continue`, and post-mortem inspection. A Python debugger shows how an
artifact was produced; it does not establish that the artifact is semantically
correct.

## The first-bad-artifact method

For an ordinary non-JIT run, inspect adjacent boundaries from left to right.
Stop at the first violated invariant rather than carrying the final symptom
backward by intuition.

| Boundary | Artifact and invariant | Useful evidence | If it is good |
| --- | --- | --- | --- |
| Public behavior | Minimal Tensor expression has a stated expected value, dtype, shape, device, and tolerance. | Concrete inputs and a NumPy/PyTorch or hand-computed oracle. | Inspect the lazy frontend graph. |
| Frontend | `Tensor.uop` represents the requested operation, movement, dtype promotion, and dependencies. | UOp counts, numbered graph, shape/dtype/device, metadata. | Schedule it without executing. |
| Scheduling | `LINEAR` has the intended kernel/copy boundaries, buffers, dependency order, and variables. Each kernel `SINK` still represents the requested computation. | `linear_with_vars()`, VIZ schedule trace, `DEBUG_RANGEIFY=1`. | Lower the suspect kernel for a fixed renderer. |
| Lowering | The program `SINK` contains legal target operations, correct indices/gates, effects, and launch dimensions. Its `LINEAR` ordering preserves them. | Consecutive VIZ pass outputs, `SPEC=2`, `CHECK_OOB=1`, `PROGRAM` children. | Compare `SOURCE` with lowered operations. |
| Rendering/compilation | `SOURCE` expresses the same operations and ABI; `BINARY` was compiled for the requested target. | `DEBUG=4`, direct inspection of `PROGRAM`, compiler diagnostics, optional disassembly. | Inspect arguments, dimensions, copies, and synchronization. |
| Runtime | The loaded program receives the correct buffers, scalars, dimensions, and ordering and produces the expected bytes. | `DEBUG=2`, cross-runtime comparison, `VALIDATE_WITH_CPU=1`, driver diagnostics. | Separate ordinary execution from JIT. |
| TinyJit | Capture preserves the ordinary calls; replay updates every input, variable, pointer, and symbolic launch dimension correctly. | `JIT=0/2/1` comparisons, captured `LINEAR`, call-by-call versus graph replay. | Minimize the capture/replay contract itself. |

TinyJit is not simply one final compiler pass. It intercepts realization during
capture, parameterizes and compiles a combined `LINEAR` plan, and may replace
batches with device-graph calls. The last row is a branch around parts of the
ordinary path. Always prove the ordinary function first.

The artifacts need not print identically across backends. Preserve semantic
invariants—values, accesses, dependencies, dtype, and launch coverage—while
allowing legal target-specific scheduling and syntax.

## Build an evidence ladder before changing code

Use this order for a new failure:

1. **Reproduce.** Record commit, Python version, `DEV` target, renderer, flags,
   driver/device where relevant, exact command, seed, and whether the failure is
   deterministic.
2. **Define the oracle.** State expected values and tolerance or the precise
   structural invariant. “It looks wrong” is not an oracle.
3. **Minimize while preserving the symptom.** Reduce model, operation chain,
   shape, dtype, layout, device count, optimization, and JIT independently.
4. **Freeze the experiment.** Use a new process, explicit `DEV`, fixed inputs,
   and an isolated `CACHEDB`. Change one variable at a time.
5. **Compare adjacent artifacts.** Find the last good and first bad boundary.
6. **Find ownership.** Locate the pass, renderer, runtime, or JIT update code
   that creates the first bad artifact, plus its nearest tests and history.
7. **Make the smallest justified change.** Do not repair a renderer symptom
   when the scheduler already emitted the wrong address.
8. **Regress and escalate.** Prove red-before/green-after with a focused test,
   then widen according to the affected contract. Chapter 16 supplies the test
   matrix.

Save observations, not entire noisy logs. For each step write one sentence:
“artifact X still satisfies invariant Y” or “pass Z is the first to violate Y.”
That record is useful in a review and prevents circular investigation.

## Minimization without erasing the bug

Every reduction must preserve both the external symptom and, once known, the
same first bad artifact.

| Dimension | Useful reductions | Common trap |
| --- | --- | --- |
| Workload | Model → layer → operation chain → one expression | Removing a materialization or alias changes kernel boundaries. |
| Shape | Large → small, then vary one axis and edge values such as 0, 1, vector width ±1 | A power-of-two shape may bypass the failing mask or tail. |
| Dtype | Original → a simpler concrete dtype | Promotion, overflow, NaN, signed zero, and tensor-core eligibility may be the bug. |
| Layout | Remove permute, pad, shrink, expand, non-contiguity one at a time | Calling `contiguous()` can hide an indexing defect by creating a new buffer. |
| Optimizer | `NOOPT=1`, `BEAM=0`, `TC=0` as separate experiments | “Disappears with optimization off” localizes a path; it does not prove the optimizer is wrong. |
| Backend | Python → CPU → requested accelerator; or hold runtime fixed and vary renderer | Different backends can also choose different legal schedules. |
| JIT | `JIT=0` → `JIT=2` → normal graphing mode | First/capture/replay calls may execute different Python branches. |
| Cache | New process, `SCACHE=0`, `CCACHE=0`, isolated `CACHEDB` | Clearing everything at once hides which cache mattered. |

Do not begin by shrinking random values alone. First reduce graph structure and
shape while retaining a fixed counterexample. Once the smallest example is
stable, vary values to discover the actual domain of the defect.

## Choose the smallest observation tool

The full command inventory is in
[Command quick reference](../reference/commands.md#debug-output). The important
debugging progression is:

| Tool | Use it to answer |
| --- | --- |
| `DEBUG=1` | Which device opened, and what high-level schedule/JIT information was emitted? |
| `DEBUG=2` | Which calls ran, with what argument count and synchronized timing? |
| `DEBUG=3` | Which kernel AST/optimization choices were selected? |
| `DEBUG=4` | What source or printable assembly was produced? |
| `DEBUG=5`/`6`/`7` | Which earlier UOps, pass graphs, individual rewrites, or disassembly explain it? |
| `VIZ=1` plus `tinygrad.viz.cli` | At which named rewrite did a graph first change incorrectly? |
| `DEBUG_RANGEIFY=1` | Which nodes inherited/newly created ranges and which axes forced materialization? |
| `VALIDATE_WITH_CPU=1` | Does each compiled device kernel agree numerically with a CPU-rendered shadow of the same scheduled kernel? |

Higher `DEBUG` levels include more data and can obscure the first useful
difference. `DEBUG=2` waits for execution to report timing, so it changes
synchronization and is diagnostic evidence, not a neutral benchmark.

For rewrite tracing:

```bash
VIZ=1 DEV=CPU DEBUG=0 CACHEDB=/tmp/tinygrad-debug-viz.db .venv/bin/python reproducer.py
.venv/bin/python -m tinygrad.viz.cli -s TINY | rg 'Schedule|Kernel'
.venv/bin/python -m tinygrad.viz.cli -s TINY 'copy the actual event name' --ls
DEBUG=6 .venv/bin/python -m tinygrad.viz.cli -s TINY 'copy the actual event name'
```

Use `DEBUG=7` only on the one named pass after `DEBUG=6` identifies the
transition. Event names include process-local counters; copy them from the
listing rather than guessing.

`DEBUG_RANGEIFY=1` is narrower than VIZ. In this snapshot it prints, per tensor
node, consumer count, operation, shape, ending-range count, and input/output
ranges. A leading `***` marks a node selected for realization. It explains a
scheduling choice; it does not validate the chosen computation.

## Cross-backend comparisons are controlled experiments

Use the same host inputs, concrete dtype, seed, tolerance, and public expression:

```bash
DEV=PYTHON DEBUG=0 .venv/bin/python reproducer.py
DEV=CPU DEBUG=0 .venv/bin/python reproducer.py
DEV=PYTHON::sm_89 DEBUG=0 .venv/bin/python reproducer.py
DEV=CUDA DEBUG=0 .venv/bin/python reproducer.py       # NVIDIA hardware evidence
DEV=CUDA:PTX DEBUG=0 .venv/bin/python reproducer.py   # hold runtime, vary renderer
DEV=NVK+NV DEBUG=0 .venv/bin/python reproducer.py     # explicit safe NV interface
```

Interpret the split:

- Python and CPU both wrong usually points before their renderer/runtime split:
  frontend semantics, shared scheduling/lowering, or a wrong oracle.
- Python right and CPU wrong moves attention toward target lowering, the C/LLVM
  renderer route, compilation, or CPU runtime.
- `CUDA` wrong and `CUDA:PTX` right holds the CUDA runtime approximately
  constant and isolates renderer/compiler routes.
- Ordinary CUDA right and NV wrong points toward runtime/queue behavior only
  after you prove the selected renderer and target are equal.
- Every ordinary route right but replay wrong points to capture,
  parameterization, memory reuse, graph updates, or ordering.

These are hypotheses, not verdicts. Two routes can share the faulty frontend or
lowering, and switching a target may also change optimization. Compare the
adjacent artifacts to confirm the inferred boundary.

### What `VALIDATE_WITH_CPU` proves

In an ordinary non-JIT run, `VALIDATE_WITH_CPU=1` expands each scheduled kernel
call with CPU shadow buffers, renders the same kernel `SINK` for CPU, executes
both, and uses NumPy `assert_allclose` on output buffers with `rtol=atol=1e-3`.
It is valuable when a device renderer or runtime is suspect:

```bash
VALIDATE_WITH_CPU=1 DEV=CUDA DEBUG=1 .venv/bin/python reproducer.py
```

It is not an independent frontend/scheduler oracle: a wrong scheduled kernel
can be wrong on both routes and agree. It also adds copies, CPU compilation,
synchronization, and memory, and the JIT compilation path does not enable this
validation in the recorded snapshot. Disable JIT and minimize before using it.
Keep a separate public-result assertion.

## Localize JIT failures in three comparisons

Use fresh input buffers on every call and retain an ordinary semantic oracle.

1. Run with `JIT=0`. If this fails, leave JIT and debug the ordinary pipeline.
2. Run with `JIT=2`. TinyJit still captures/replays, but device-graph batching is
   disabled at this snapshot. A failure here implicates capture,
   parameterization, memory planning, or replay.
3. Run with the normal `JIT` setting. If only this fails, inspect graph batching,
   supported-call classification, pointer/scalar updates, and launch dimensions.

Record outputs separately for the ignore, capture, and first replay calls. A
test that checks only the third returned value cannot show when divergence
began. Also vary one input buffer and one symbolic value between replays; a
constant repeated input can hide a stale-pointer or stale-scalar bug.

## Use source history after current behavior is understood

Find definitions, callers, and tests first:

```bash
rg -n 'def full_rewrite_to_sink|def run_linear|class _TinyJit' tinygrad test
rg -n 'DEBUG_RANGEIFY|VALIDATE_WITH_CPU' tinygrad test
git log --oneline --follow -- tinygrad/schedule/rangeify.py
git blame -L 555,580 tinygrad/schedule/rangeify.py
git log -S 'DEBUG_RANGEIFY' --oneline --all -- tinygrad test
```

`git blame` answers who last changed a line, not why it is correct. Read the
whole introducing diff and its tests. Use the official
[`git log` documentation](https://git-scm.com/docs/git-log) for `-L`,
`--follow`, and pickaxe searches.

When the regression interval is unknown, use a clean dedicated checkout and
the official [`git bisect` workflow](https://git-scm.com/docs/git-bisect):

```bash
git bisect start BAD_COMMIT KNOWN_GOOD_COMMIT
git bisect run env DEV=CPU DEBUG=0 .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -q
git bisect reset
```

The command must classify every tested revision: exit 0 for good, 1–127 except
125 for bad, and 125 when that revision cannot be tested. Keep dependencies and
`CACHEDB` compatibility in mind across old revisions. A bisect identifies the
introducing commit; the artifact ladder still identifies the broken contract.

## Source tour

All links are pinned to
`874d33128b4e4785beea736d97df6716e0321717`.

| Read this | What to extract |
| --- | --- |
| [`Tensor.linear_with_vars` and `Tensor.realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L169-L196) | The frontend-to-schedule boundary and ordinary execution entry. |
| [`create_linear_with_vars`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L107-L199) | Schedule tracing, kernel calls, copies, variable bindings, and buffer resolution. |
| [`run_rangeify` debug output](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L179-L323) | Meaning of `DEBUG_RANGEIFY=1` rows and realization markers. |
| [`full_rewrite_to_sink` and `do_to_program`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284-L478) | Named lowering passes and construction of `SINK`/`LINEAR`/`SOURCE`/`BINARY`. |
| [`run_linear`, validation, compilation, and execution](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L180-L286) | Runtime dispatch and exactly where CPU shadow validation is inserted. |
| [TinyJit lowering and state machine](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L67-L289) | Parameterization, memory planning, graph splitting, ignore/capture/replay. |
| [Debug and validation `ContextVar` definitions](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L232-L280) | Snapshot defaults and cache/debug knobs. |
| [Rewrite tracking internals](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1490-L1565) and [VIZ usage](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md) | How named groups and individual rewrites become inspectable events. |
| [CPU validation tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/device/test_validate_with_cpu.py) | Supported examples and the expected test style. |

## Lab — Predict, observe, change, regress

This lab injects a process-local, float-only bug into the CPU renderer. It does
not edit your checkout. Save this as `debug_renderer_fault.py` in your study
notebook:

```python
import os, unittest
from collections import Counter
from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import Ops

if int(os.getenv("INJECT_RENDERER_FAULT", "0")):
  old_add = ClangRenderer.code_for_op[Ops.ADD]
  ClangRenderer.code_for_op = {
    **ClangRenderer.code_for_op,
    Ops.ADD: lambda a, b, dtype:
      f"({a}-{b})" if dtype in dtypes.floats else old_add(a, b, dtype),
  }

class TestAddRegression(unittest.TestCase):
  def test_nonzero_rhs(self):
    x = Tensor([1.0, 2.0, 3.0]).realize()
    out = x + 4
    print("frontend:", Counter(u.op.name for u in out.uop.toposort()))

    linear, var_vals = out.linear_with_vars()
    call = next(c for c in linear.src if c.src[0].op is Ops.SINK)
    print("schedule:", Counter(u.op.name for u in call.src[0].toposort()))

    program = to_program(call.src[0], Device[call.device].renderer)
    print("lowered:", Counter(u.op.name for u in program.src[0].toposort()))
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    print("source has +4/-4:", "+4.0f" in source, "-4.0f" in source)

    run_linear(linear, var_vals)
    self.assertEqual(out.tolist(), [5.0, 6.0, 7.0])

if __name__ == "__main__":
  unittest.main()
```

### Predict

Before running, fill in the expected result for each cell:

| Run | Frontend/scheduled/lowered operation | Rendered CPU expression | Test |
| --- | --- | --- | --- |
| CPU, injected | ? | ? | pass/fail? |
| Python, injected | ? | Python-rendered representation | pass/fail? |
| CPU, no injection | ? | ? | pass/fail? |

The fault is restricted to floating-point `ADD` so integer address arithmetic
remains valid. A deliberately broad renderer mutation can corrupt indices and
crash the process; fault injection should be as bounded as a real hypothesis.

### Observe

Run three new processes:

```bash
DEV=CPU DEBUG=0 SCACHE=0 CCACHE=0 \
  CACHEDB=/tmp/tinygrad-debug-fault.db INJECT_RENDERER_FAULT=1 \
  .venv/bin/python debug_renderer_fault.py

DEV=PYTHON DEBUG=0 SCACHE=0 CCACHE=0 \
  CACHEDB=/tmp/tinygrad-debug-python.db INJECT_RENDERER_FAULT=1 \
  .venv/bin/python debug_renderer_fault.py

DEV=CPU DEBUG=0 SCACHE=0 CCACHE=0 \
  CACHEDB=/tmp/tinygrad-debug-fixed.db INJECT_RENDERER_FAULT=0 \
  .venv/bin/python debug_renderer_fault.py
```

The injected CPU run should fail with `[-3.0, -2.0, -1.0]`. Its frontend,
scheduled, and lowered artifacts still contain `ADD`, while its source contains
`-4.0f`. The Python run and fixed CPU run should return `[5.0, 6.0, 7.0]`.
The first bad artifact is therefore `SOURCE`, and ownership is the CPU renderer
mapping—not Tensor addition, scheduling, or runtime argument packing.

### Change and regress

Unsetting `INJECT_RENDERER_FAULT` is the controlled fix. The numerical assertion
is the regression; the source substring is diagnostic evidence, not a golden
test. Exact temporary names and whitespace are free to change.

Now find the existing upstream coverage:

```bash
rg -n 'def test_add|def test_plus|test_repeat_add' test
DEV=CPU .venv/bin/python -m pytest test/backend/test_ops.py::TestOps::test_add -x -q
DEV=CPU .venv/bin/python -m pytest \
  test/backend/test_renderer_failures.py::TestCStyleFailures::test_repeat_add -x -q
```

The general add semantics already have coverage, so submitting a duplicate test
for this artificial defect would add little. In a real regression, first prove
that the nearest test fails on the faulty revision. Add a minimal case only
when the existing test does not encode the missing domain or invariant, then
follow the escalation method in [Testing](16-testing.md).

## Troubleshooting decision table

| Observation | Next move |
| --- | --- |
| Failure vanishes in a fresh process | Isolate schedule/compiler/disk cache one at a time; record import-time environment values. |
| Failure changes under `DEBUG=2` | Suspect synchronization, lifetime, race, or timing; use explicit waits without treating debug timing as a benchmark. |
| Minimal values pass but original values fail | Preserve dtype and special values; test overflow, NaN/Inf, signed zero, and tolerance. |
| Contiguous input passes, view fails | Compare movement/range artifacts and final indices/gates; do not “fix” by forcing a copy. |
| `NOOPT=1` passes | Compare pre/post optimization graphs and applied opts; keep the semantic oracle. |
| Python and CPU both fail identically | Inspect frontend/schedule/lowering or the oracle before target-specific code. |
| Python passes, compiled backends fail | Compare lowered target legality, renderer output, compiler, then runtime. |
| `VALIDATE_WITH_CPU=1` reports a mismatch | Minimize one kernel; compare its lowered `SINK`, CPU/device source, args, and launch dimensions. |
| CPU validation agrees but public result is wrong | The shared scheduled semantics or public oracle may be wrong; return to earlier artifacts. |
| CUDA C fails, `CUDA:PTX` passes | Hold target/runtime fixed and inspect renderer/compiler differences. |
| Ordinary execution passes, `JIT=2` fails | Inspect capture, parameter slots, memory planning, changing inputs, and symbolic values. |
| `JIT=2` passes, normal JIT fails | Inspect device-graph grouping and pointer/scalar/dimension update metadata. |
| Crash or GPU fault occurs | Minimize without repeated broad submissions; enable Python faulthandler/driver diagnostics and use mock or dedicated hardware where possible. |
| History points at a large refactor | Bisect with the minimized test, then compare the first bad commit's adjacent artifacts rather than reverting by guess. |

## Checkpoint

You are ready to proceed when you can:

- turn a report into a deterministic reproducer with a stated oracle;
- minimize it without changing the first bad artifact;
- distinguish frontend, schedule, lowering, renderer, runtime, and JIT evidence;
- explain what `DEBUG`, VIZ, `DEBUG_RANGEIFY`, and
  `VALIDATE_WITH_CPU` do and do not prove;
- design a controlled cross-backend or JIT comparison;
- use source history only after locating current ownership; and
- produce a focused red-before/green-after regression rather than a log-only
  claim.

## Quick reference

```text
reproduce → define oracle → minimize → freeze environment
          → last good artifact / first bad artifact
          → source owner + history → focused regression → widen

ordinary: Tensor UOp → schedule → lowered SINK/LINEAR
                     → SOURCE/BINARY → runtime result

JIT split: JIT=0 → JIT=2 (capture/replay, no graph batching)
                 → normal JIT (device graph when supported)

semantic routes: PYTHON → CPU → requested accelerator
renderer split:  CUDA ↔ CUDA:PTX
```

Keep [the command reference](../reference/commands.md) open while debugging and
use [Learning resources](../reference/learning-resources.md) only for the
specific compiler, GPU, or testing concept exposed by the checkpoint.
