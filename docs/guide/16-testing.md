# 16. Testing a contribution

## Purpose

A fix is not complete when the reproducer turns green. A contribution needs an
argument that:

- the old code violated a named contract;
- the focused test observes that violation and fails for the intended reason;
- the new code satisfies the contract over the relevant domain; and
- broader compiler, backend, runtime, and CI behavior remains intact.

This chapter teaches how to route a test into tinygrad's suite, choose semantic
and structural assertions, use differential/property/fuzz testing, turn on IR
and bounds validation, and escalate in proportion to the change.

**Verified against tinygrad:** `874d331` (2026-08-05).

## Prerequisite gate

You should already be able to produce the first bad artifact using
[the debugging method](15-debugging.md), and to state whether the broken
contract belongs to the frontend, a rewrite, scheduling, lowering, rendering,
runtime, or JIT. Test location follows contract ownership, not merely the file
you edited.

Be comfortable selecting one pytest node and reading its failure report. The
official pytest pages on
[invocation](https://docs.pytest.org/en/stable/how-to/usage.html) and
[assertions](https://docs.pytest.org/en/stable/how-to/assert.html) are enough.
For floating arrays, understand `rtol`/`atol` in NumPy's
[`assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html).
Return when you can explain why exact equality is wrong for many floating
reductions and why an unnecessarily loose tolerance is also wrong.

Property-based testing is not a prerequisite for every patch. When the failure
describes a family of shapes, dtypes, or expressions, follow the bounded
Hypothesis route in
[Learning resources](../reference/learning-resources.md#testing-transformations):
constrained strategies, invariants, shrinking, deterministic replay, then back
to the focused tinygrad test.

## Testing is a risk argument

No single suite establishes every kind of correctness. Build evidence outward
from the first bad artifact:

```text
focused regression
  → nearest subsystem tests
  → semantic/structural/spec/property checks
  → relevant backend or mock matrix
  → real hardware or model integration when the claim requires it
  → static checks and the actual CI jobs
```

The focused test must be red before the fix and green after it. Broader green
suites without that red state can miss a no-op fix, an accidentally disabled
test, or a reproducer that never reached the affected path.

Test cost should match risk. A local symbolic identity may need dozens of
negative examples but no GPU. A two-line CUDA argument-packing change may need
one tiny numerical case plus real CUDA hardware even if thousands of NULL tests
pass.

## Route by the contract under test

The pinned [`test/README`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/README)
defines three core groups: `backend` tests run across backends, `null` tests do
not require a backend, and `unit` tests run on one backend in CI. The other
directories below have roles visible in their contents and workflow jobs.

| Route | Put or run a test here when the contract is | Typical evidence and caution |
| --- | --- | --- |
| `test/null/` | Backend-independent UOp construction, symbolic algebra, rewrite traversal, scheduling structure, memory planning, spec, or helpers | Fast structural evidence, normally exercised in CI with `SPEC=2 DEV=NULL`. NULL is not a numerical oracle. |
| `test/backend/` | Expected to hold across runtime/renderer targets: Tensor op semantics, dtype behavior, generated programs, scheduling, JIT, transfers | This is the portability lane. Use skip predicates only for a documented unsupported capability, not to hide a backend failure. |
| `test/unit/` | A subsystem or integration contract that CI intentionally exercises on one selected backend rather than the full backend matrix | In the snapshot Linux uses CPU and macOS uses Metal. “Unit” does not mean backend-free. |
| `test/opt/` | Kernel optimization actions, tensor cores, vectorization/upcast/unroll choices, and optimizer legality | Test both chosen structure and semantic output where possible; use emulated target routes for portability, hardware for performance claims. |
| `test/device/` | Allocator, queue, graph, compiler/runtime, synchronization, or device-specific behavior | State required device/interface. A CPU or mock pass cannot establish actual GPU timing or driver behavior. |
| `test/mockgpu/` | Support code for emulated/mocked drivers and devices used by AMD/NVIDIA/HCQ tests | This directory is primarily infrastructure, not a standalone generic suite. Run the tests that consume the appropriate `MOCK...` route. |
| `test/external/` | Fuzzers, process replay, large integrations, model checks, benchmarks, and specialized/manual scenarios | Many files deliberately avoid ordinary auto-collection. Run the named script/job explicitly and document dependencies and cost. |

`test/models/`, `test/amd/`, `test/testextra/`, and specialized `extra/` tests
add further routes. Use them when the change owns that domain; do not run a
large model merely because it eventually calls the modified function.

### Find the nearest precedent

Search for the symbol, operation, invariant, and historical symptom:

```bash
rg -n 'full_rewrite_to_sink|Ops\.ADD|test_add' test tinygrad
rg -n 'VALIDATE_WITH_CPU|CHECK_OOB|JIT=2' test .github/workflows
git log --oneline --all -- test/backend/test_ops.py
```

Read the test helper it uses before copying its shape. For example,
`test/helpers.py` exposes helpers for schedule counts, lowered UOps, JIT cache
length, dtypes, and device requirements. Reimplementing those poorly makes a
test harder to maintain.

## Write the focused regression first

A useful regression contains five explicit decisions:

1. **Trigger.** The smallest graph, shape, dtype, layout, flags, and backend that
   reaches the faulty path.
2. **Oracle.** A hand-computed value or independent implementation, or a
   structural invariant defined by the pass contract.
3. **Boundary.** The stage at which the assertion belongs.
4. **Domain.** The nearby positive, negative, edge, dtype, and shape cases the
   fix claims to cover.
5. **Failure reason.** Evidence that the pre-fix failure is the target defect,
   not missing hardware, an import error, a timeout, or an unrelated assertion.

Name the test after the behavior, not an issue number alone. A comment should
explain a non-obvious precondition or why an apparently redundant edge case
matters. It should not narrate the implementation.

Run the node alone while iterating:

```bash
DEV=CPU .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -x -q
```

`-x` stops on the first failure and `-q` keeps the signal readable. Do not start
with `-n12` on one test: parallel output and process state complicate debugging.
Use the repository's requested workers when the selection becomes broad:

```bash
DEV=CPU .venv/bin/python -m pytest test/unit/ -x -q -n12
```

Environment-backed `ContextVar` values are generally created during import.
Set global test modes before Python starts; use `Context(...)` inside a test
only for the deliberately scoped variant.

## Structural assertions: precise versus brittle

Compiler tests often need structure as well as values. The question is whether
the structure is part of the contract.

| Durable when relevant | Usually brittle |
| --- | --- |
| A forbidden op is absent after a named lowering boundary. | Full `repr` of a large graph. |
| A replacement is the same interned UOp when canonical identity is promised. | UOp object IDs or cache population counts. |
| Dtype, shape, address space, buffer role, gate, or dependency is preserved. | Temporary UOp ordering unrelated to effects. |
| Exactly one kernel is formed when fusion/kernel count is the behavior being changed. | Kernel count in a semantic test that does not own fusion. |
| Program input/output slots or launch dimensions satisfy an ABI contract. | Generated function and local variable names. |
| A renderer emits required syntax/ordering for a target ABI or parser. | Whole-source golden strings, whitespace, or formatting. |
| A rewrite reaches a fixed point or removes a forbidden form. | Exact number of internal rewrite attempts. |

Pair a structural assertion with a semantic one when either could pass alone.
For example, “`REDUCE` is gone after lowering” does not prove accumulation is
correct, while a few numerical samples do not prove the intended canonical form
was selected.

Source-string assertions are justified for syntax, ABI qualifiers, instruction
mnemonics, or a renderer regression that has no better intermediate contract.
Match the smallest meaningful fragment or parse/normalize the source. Do not
freeze unrelated names and whitespace.

For floating output, derive tolerance from dtype and operation. Reductions and
reassociation can accumulate error; float16 and transcendental decompositions
need different bounds from float32 elementwise addition. Include an input that
would fail if the tolerance were accidentally much larger. Handle NaN, Inf, and
signed zero explicitly when the contract distinguishes them.

## Four complementary ways to cover a domain

### Example regression

Keep the smallest concrete counterexample even if a fuzzer found it. It gives
reviewers a stable explanation and protects the exact historical failure.

### Differential test

Compare tinygrad with a hand computation, NumPy, PyTorch, another renderer, or
another backend. Prefer an oracle that does not share the suspected logic.
`DEV=PYTHON` and `DEV=CPU` share frontend/scheduling/lowering, so their agreement
does not validate those stages independently. Cross-backend comparisons are
stronger for target rendering/runtime defects than for shared semantic defects.

Match dtype and semantics before comparing. NumPy's default integer widths,
promotion, division, and reduction accumulation may differ; PyTorch may make
different layout or precision choices. An “independent” oracle with a different
contract creates false bugs.

### Property-based test

Use Hypothesis when one invariant spans a constrained family:

- rewrite output is semantically equivalent and idempotent;
- movement operations preserve element mapping;
- symbolic simplification agrees with concrete evaluation;
- generated indices remain in bounds;
- a dtype operation agrees with an independent scalar model; or
- JIT replay agrees with ordinary execution as inputs and bound variables vary.

Generate only legal inputs for the contract. Encode relationships in a
composite strategy instead of discarding most examples with `assume`. Bound
shapes and example count so the test remains a unit test. Save the minimized
counterexample as a concrete regression if it explains a new boundary.

### Fuzz or integration test

Use a fuzzer for interaction space too large for a reviewable matrix, and a
model/integration test when graph scale or subsystem composition is essential.
Record seed/settings and retain minimized failures. A long random loop without
shrinking, deterministic replay, or a clear invariant is neither a good unit
test nor a good fuzzer.

The snapshot includes Hypothesis tests in `null` and `backend` plus explicit
fuzz scripts such as `test/external/fuzz_shape_ops.py`. Copy their resource
discipline, not merely their decorators.

## Validate legality with `SPEC` and `CHECK_OOB`

`SPEC` checks representation legality, not numerical equivalence.

- With any nonzero `SPEC`, key schedule/codegen boundaries call `type_verify`
  against tensor or program specs.
- At `SPEC=2`, this snapshot also checks each newly created UOp against the full
  spec, catches inferred-dtype mismatches, and performs Python-render
  round-tripping at boundary verification.
- Higher values add snapshot-specific checks and are not the ordinary CI
  contract; use the mode selected by current CI unless investigating those
  checks themselves.

Run the focused test in the stricter CI mode:

```bash
SPEC=2 DEV=NULL .venv/bin/python -m pytest test/null/test_uops.py -x -q
SPEC=2 DEV=CPU .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -x -q
```

`CHECK_OOB=1` asks the program spec to prove final indexed loads/stores remain
in bounds, using cheap min/max first and Z3 when necessary. The
`testing_minimal` extra installs the supported `z3-solver` dependency.

```bash
SPEC=2 CHECK_OOB=1 DEV=NULL .venv/bin/python -m pytest \
  test/null/test_validate_oob.py -x -q
```

It is a symbolic proof for modeled index/gate forms, not a runtime memory
sanitizer. In the recorded implementation image indices and graphs containing
certain bitcast/vector stack forms bypass this proof. Read the skip conditions
before claiming universal bounds safety. Keep numerical or runtime evidence for
the actual bug.

## Process replay protects generated-program behavior

Process replay answers a different question from unit tests: “Given compiler
inputs captured on the change branch, what generated programs change when
replayed on the comparison revision?”

At the snapshot, `do_to_program` records its arguments, relevant context values,
location, and returned program when `CAPTURE_PROCESS_REPLAY=1`. The replay tool
loads those rows from `CACHEDB`, regenerates programs on the comparison
revision, and prints source diffs. Upstream documents `[pr]` as the
refactor/speedup convention; the workflow and replay script together decide
whether inputs are captured and differences become CI errors, so inspect both
on current `master`.

Use it for broad rewrite, scheduling, codegen, renderer, and optimization
changes:

1. capture focused and relevant broad tests on the change branch;
2. replay from a separate clean checkout of the comparison revision;
3. classify each program difference as intended, incidental but legal, or a
   regression; and
4. retain ordinary correctness/performance tests.

Follow the exact
[process replay README](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md)
and the outline in
[Command quick reference](../reference/commands.md#process-replay-outline).
Do not switch revisions in a dirty working tree. Captures are tied to the
snapshot's serialization/cache schema, and replay may stop early when too many
kernels differ. A clean replay does not prove runtime correctness; it proves
generated output did not unexpectedly change for captured compiler inputs.

## Escalation matrix

Start with the focused node in every row. Add the evidence listed for the
actual surface changed.

| Change surface | Next tests | Final evidence when applicable |
| --- | --- | --- |
| Tensor API or autograd rule | Nearby `null`/`unit` tests; forward values, gradient or finite-difference oracle; dtype/shape edges | Relevant `backend` tests and a small model only if composition is part of the claim |
| Symbolic or graph rewrite | Positive/negative/order/termination tests in `null`; `SPEC=2`; differential/property cases | Downstream backend test and process replay if the rule reaches generated kernels |
| Rangeify, fusion, scheduling, memory planning | `null/test_schedule*` plus focused `backend` semantic case; `DEBUG_RANGEIFY` evidence; `SPEC=2 CHECK_OOB=1` | Process replay, multi-device/alias cases, relevant backend matrix |
| Kernel optimization or tensor core | `test/opt/` structure and semantics; `NOOPT`/target comparison; emulated target | Real target correctness, then synchronized benchmark/profiler evidence for a speed claim |
| Renderer or compiler wrapper | Renderer-isolation/negative case; backend operation on Python and one compiled route; `SPEC=2` | Every affected renderer target, process replay, real toolchain/device when compilation support is claimed |
| Runtime allocator, copy, queue, or launch | `test/device/` or consuming mock-GPU test; focused backend op; synchronization/lifetime cases | Actual device/driver/interface, recovery-safe fault cases, multi-process/device if affected |
| TinyJit or device graph | `test/unit/test_jit*.py` and `test/backend/test_jit.py`; ignore/capture/replay outputs; `JIT=0/2/1` | Graph-capable runtime, changing pointers/scalars/symbolic dims, multiple replays |
| NVIDIA-specific path | Python `sm_89` or mock route for compiler semantics; PTX/CUDA route split | RTX 4090 or required architecture for hardware/runtime/performance claims |
| Performance-only change | All relevant correctness/structure tests; process replay to enumerate kernel changes | Warmed, synchronized distribution with output oracle and profiler/bottleneck evidence |
| Documentation or typing | Executable snippet or strict docs build; closest import/type check | Current link/API verification and CI's actual docs/lint job |

Passing a mock route establishes behavior of the model encoded by that mock.
It cannot establish real command submission, timing, memory coherence, or a
vendor compiler's behavior unless those components are actually exercised.

## Read CI as executable documentation

Before declaring coverage, inspect the current branch's workflow rather than
remembering a matrix:

```bash
rg -n '^  [a-zA-Z0-9_]+:|DEV=|SPEC=|CHECK_OOB|test/(null|unit|backend|opt|device)' \
  .github/workflows/test.yml
git diff --name-only
```

At the recorded snapshot:

- workflow-global `CHECK_OOB=1` applies unless a job overrides it;
- NULL tests run with `SPEC=2 DEV=NULL`;
- Linux unit tests run `test/unit/` on CPU, while macOS unit tests use Metal;
- a separate `SPEC=2` job covers selected `unit`, `backend`, and `opt` tests;
- backend jobs cover Python, CPU renderer variants, OpenCL/WebGPU, mocked
  NVIDIA/AMD routes, and Metal in different selections;
- fuzz, model, docs, lint, and process-replay work live in separate jobs.

Read job environment, setup dependencies, `--ignore`/`-k` filters, mock
interface, architecture, and per-job overrides. A test file being in
`test/backend/` does not guarantee every backend job collects every case.
Conversely, a local full directory run may not reproduce CI's `SPEC`,
`CHECK_OOB`, renderer, or mock target.

## Static checks

During iteration, check changed files; before handoff, run the current
repository-prescribed scope:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy tinygrad/
.venv/bin/pre-commit run --all-files
```

Use the versions in `pyproject.toml`'s linting extra. Static checks complement
execution:

- Ruff and pre-commit catch syntax, imports, whitespace, and configured policy.
- Mypy catches type-contract errors on checked paths.
- None validates Tensor semantics, legal lowering, runtime behavior, or speed.

Do not format or repair unrelated files to make a patch look globally clean.
Report a pre-existing unrelated failure separately with the exact command.

## Source and test tour

All tinygrad links are pinned to
`874d33128b4e4785beea736d97df6716e0321717`.

| Read this | What to extract |
| --- | --- |
| [Test grouping contract](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/README) | The project's stated meaning of `null`, `backend`, and `unit`. |
| [`test/null/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/null), [`test/backend/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/backend), and [`test/unit/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/unit) | Find the nearest test style and which layer its assertions own. |
| [`test/opt/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/opt), [`test/device/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/device), and [`test/mockgpu/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/test/mockgpu) | Optimization, hardware-runtime, and mock driver/device boundaries. |
| [Shared test helpers](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/helpers.py) | Schedule, UOp, JIT, dtype, device, and timing helpers already available. |
| [Pattern/rewrite tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py) and [backend op tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/backend/test_ops.py) | Structural, property-based, and differential test styles. |
| [UOp construction checks](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L193-L221) and [`type_verify`/bounds checks](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L8-L44) | Exactly what `SPEC=2` and `CHECK_OOB=1` validate. |
| [Bounds-validation tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_validate_oob.py) | Positive, masked, symbolic, and expected-failure OOB cases. |
| [Shape-operation fuzzer](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/fuzz_shape_ops.py) | Constrained Hypothesis strategies and differential invariants. |
| [Process replay implementation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py) | Captured row format, context replay, source comparison, assertion, and early-stop behavior. |
| [Pinned CI workflow](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1-L259) and [backend/device jobs](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L425-L727) | Actual flags, dependencies, targets, selections, mocks, and process-replay calls. |
| [Testing and lint dependencies](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L57-L100) | Pinned tools and the distinction between minimal, unit, full, and linting extras. |

## Lab 1 — Select tests for the renderer fault

Use Chapter 15's controlled CPU-renderer fault as if it were a real change.
Before running anything, write a route card:

```text
contract:
first bad artifact:
focused semantic test:
focused structural/renderer test:
independent comparison:
broader directory/job:
hardware required:
process replay required:
static checks:
```

Find candidates, then collect and run only the selected nodes:

```bash
rg -n 'def test_add|def test_plus|test_repeat_add' \
  test/test_tiny.py test/backend

.venv/bin/python -m pytest --collect-only -q \
  test/backend/test_ops.py::TestOps::test_add \
  test/backend/test_renderer_failures.py::TestCStyleFailures::test_repeat_add

DEV=CPU .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add \
  test/backend/test_renderer_failures.py::TestCStyleFailures::test_repeat_add \
  -x -q

SPEC=2 DEV=PYTHON .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -x -q
```

Explain the evidence: `TestOps.test_add` protects values; `test_repeat_add`
protects a C-style rendering property; Python protects shared semantics but
does not exercise `ClangRenderer`. A real change to the shared
`CStyleLanguage.code_for_op` mapping would require more C-style targets and
process replay than a `ClangRenderer`-only change. Route from the code actually
changed, not from the artificial symptom alone.

## Lab 2 — Turn one example into a bounded property

Suppose a permute-plus-reduction failure appears only for some rectangular
shapes. First keep the concrete failing shape. Then save this candidate outside
the tinygrad checkout as `test_regression_candidate.py`:

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from tinygrad import Tensor

@settings(max_examples=20, deadline=None, derandomize=True)
@given(rows=st.integers(1, 6), cols=st.integers(1, 8))
def test_permute_add_reduce(rows, cols):
  data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
  got = (Tensor(data).permute(1, 0) + 1).sum(axis=1).numpy()
  expected = (data.T + 1).sum(axis=1)
  np.testing.assert_allclose(got, expected, rtol=0, atol=0)
```

Predict which stages and tests this property covers and which it does not. Then
run:

```bash
SPEC=2 CHECK_OOB=1 DEV=CPU .venv/bin/python -m pytest \
  /absolute/path/test_regression_candidate.py -x -q
SPEC=2 CHECK_OOB=1 DEV=PYTHON .venv/bin/python -m pytest \
  /absolute/path/test_regression_candidate.py -x -q
```

The data are small integers exactly represented in float32, so exact comparison
is intentional here. For arbitrary random floats or a larger reduction, justify
a nonzero tolerance.

If this finds a failure:

1. retain Hypothesis's minimized `rows`/`cols` as a concrete focused case;
2. identify the first bad artifact;
3. route the concrete test near the owning scheduler/movement/backend tests;
4. keep the bounded property only if it catches a meaningful family at
   acceptable CI cost; and
5. rerun without `CHECK_OOB` only to distinguish proof failure from execution,
   never to hide an illegal access.

## Lab 3 — Build the escalation plan before a broad run

Choose one hypothetical change:

- a rule in `tinygrad/uop/symbolic.py`;
- input validation in `tinygrad/engine/jit.py`; or
- scalar argument packing in `tinygrad/runtime/ops_cuda.py`.

Use `rg` and the pinned workflow to fill this table without running every test:

| Question | Your evidence |
| --- | --- |
| Which focused node must be red before the fix? | |
| Which directory owns the contract? | |
| Which independent semantic oracle is available? | |
| Which `SPEC`/OOB/property mode adds a different failure detector? | |
| Which CI jobs actually collect the affected test? | |
| Which mock/emulated route helps, and what can it not prove? | |
| Is process replay relevant? | |
| What real hardware or model evidence is necessary? | |
| Which static checks cover the changed files? | |

Then run only the first two escalation levels. Review the output and update the
plan before launching a backend directory, fuzzer, model, or hardware suite.
The checkpoint is a defensible selection, not the largest command your machine
can finish.

## Checkpoint

You are ready to turn a fix into a contribution when you can:

- place a test according to `null`/`backend`/`unit`/`opt`/`device`/mock/external
  ownership and explain that choice;
- prove the focused test is red before and green after the fix;
- choose semantic, structural, differential, property, and fuzz evidence
  without confusing their claims;
- write structural assertions that encode a contract without freezing incidental
  IR/source details;
- explain what `SPEC=2`, `CHECK_OOB=1`, and process replay do not prove;
- escalate to the relevant backend, mock, real-hardware, and model routes;
- read current CI flags and selections from the workflow; and
- run and report the appropriate static checks.

## Quick reference

```text
1. locate first bad artifact
2. find nearest existing test/helper
3. write smallest red regression
4. fix → focused green
5. add negative/edge/domain cases
6. run SPEC/OOB/differential/property checks that add a new oracle
7. widen to owning directory and affected backend/runtime
8. process replay for generated-program changes
9. real hardware/model only when the claim requires it
10. static checks + current CI matrix

null     backend-free internals and structure
backend  same contract across targets
unit     one selected backend in CI
opt      kernel optimization and tensor-core behavior
device   allocator/queue/runtime/hardware contracts
mockgpu  support for mocked/emulated device routes
external fuzz/replay/model/benchmark/specialized runs
```

Keep [Command quick reference](../reference/commands.md) beside the test run and
return to [Learning resources](../reference/learning-resources.md) only when a
specific testing or hardware prerequisite blocks the next checkpoint.
