# 16. Testing a contribution

## The promise of this chapter

Chapter 15 ended with a localized defect: a known input, a known correct
answer, a last good artifact, and a first bad artifact.  That is enough to know
where to work.  It is not yet enough to make the work safe for everyone else.

A test turns that one debugging episode into a claim that can be checked again
after the code changes.  A useful contribution needs an argument that:

- the old code violated a named contract;
- the focused test really observes that contract and goes red for the intended
  reason;
- the new code makes the same test green;
- nearby cases show the boundary of the fix rather than only its easiest
  example; and
- broader checks give evidence about the other compiler, renderer, runtime,
  and hardware surfaces that might have changed.

The word **argument** is important.  “I ran 5,000 tests” is not automatically a
stronger argument than “this five-line test fails on the old rewrite, passes on
the new rewrite, checks the transformation's output against an independent
model, and then passes under `SPEC=2`.”  Test count measures work performed, not
what that work was capable of detecting.

This chapter starts from ordinary Python functions and assertions.  It does
not assume prior compiler-testing knowledge.  It then builds the distinctions
needed in tinygrad:

- behavior versus implementation structure;
- example, differential, metamorphic, property, fuzz, and integration tests;
- an implementation under test versus its oracle;
- a focused regression versus a broad suite;
- a backend-independent proof obligation versus a physical-device claim;
- representation legality versus numerical correctness; and
- local evidence versus the actual CI matrix.

By the end, you should be able to write a small regression that genuinely goes
red before a fix, place it in the appropriate part of tinygrad's suite, choose
nearby cases and independent checks, and explain both what the resulting green
tests establish and what they leave untested.

**Source snapshot:** `874d331` (2026-08-05).  Test names and CI jobs are
particularly changeable.  Use the pinned links to learn the recorded system,
then inspect current `master` before proposing a patch.

## Route through the chapter

Read the chapter in this order on the first pass:

1. turn an informal bug report into a precise contract;
2. separate the implementation, input, observation, oracle, and assertion;
3. watch a deliberately weak test stay green for a known bug;
4. make a focused counterexample go red for the right reason;
5. run the same contract against tinygrad and make it green;
6. decide which extra cases add a genuinely different detector;
7. place the test according to the layer that owns the contract;
8. add structural and validation checks only where their facts matter;
9. widen through the relevant backend, runtime, replay, and hardware routes;
10. read CI as executable documentation; and
11. record a reviewable testing argument rather than a list of commands.

The executable `labs/phase5/testing_walk.py` lab anchors the first five steps.
It has one mode in which a known-bad implementation is expected to
fail and another in which the exact same five-test contract must pass on
tinygrad's portable Python backend.

## The five pieces of a test

Consider this expression for a two-dimensional tensor `x`:

```python
y = (x.permute(1, 0) + 1).sum(axis=1)
```

For a reader used to ML code, the operations may look routine.  Spell out the
contract anyway:

```text
input x has shape (rows, cols)
permute(1, 0) changes the logical shape to (cols, rows)
adding 1 changes every logical element
sum(axis=1) removes the rows dimension
output therefore has shape (cols,)
output[col] = sum(x[row, col] + 1 for every row)
```

For this concrete input:

```text
x = [[0, 1, 2],
     [3, 4, 5]]
```

the output must be:

```text
column 0: (0 + 1) + (3 + 1) = 5
column 1: (1 + 1) + (4 + 1) = 7
column 2: (2 + 1) + (5 + 1) = 9
result: [5, 7, 9]
```

A test of that statement has five separable pieces:

| Piece | In this example | Why it must be explicit |
| --- | --- | --- |
| **Contract** | Permute changes the coordinate map; reduction then sums each original column and returns `cols` values. | Without a contract, a failure says only that two values differed. |
| **Input/trigger** | A non-symmetric `2 × 3` matrix. | An input can accidentally avoid the suspected path or make a defect invisible. |
| **Implementation under test** | tinygrad's Tensor graph through realization on the chosen backend. | The test must actually reach the code and route named in the claim. |
| **Observation** | Output shape and numerical values; perhaps an intermediate artifact during localization. | A test sees only what it observes.  Unobserved properties are not tested. |
| **Oracle** | `[5, 7, 9]`, derived by literal column loops. | Comparing with another answer is useful only if that answer has a trustworthy origin. |

The final assertion is the comparison that turns these choices into a pass or
failure:

```python
assert got == [5.0, 7.0, 9.0]
```

An assertion is not the whole test.  Most poor tests contain a perfectly valid
`assert`; the weakness lies in the trigger, oracle, observation, or unstated
contract around it.

### What red and green mean

In test language:

- **red** means the test failed;
- **green** means the test passed; and
- a **regression test** is retained so the behavior does not silently return.

For a bug fix, the focused test should be red on the unmodified buggy revision
and green after the fix.  That two-state experiment proves two important facts:

1. the test had power to detect the original defect; and
2. something about the patch changed the observed behavior.

It still does not prove that the patch is the best fix or works over the whole
claimed domain.  That is why the test needs readable inputs, nearby negative
cases, and broader evidence.  But skipping the red state is especially weak:
the test may have been green all along, may never reach the changed code, or
may assert a fact unrelated to the bug.

When a bug cannot safely be recreated on the current checkout, keep the two
revisions or worktrees separate and run the same test command in each.  Do not
edit a dirty checkout back and forth merely to manufacture a screenshot.

## A green example can be powerless

Suppose an incorrect implementation forgets the transpose and instead sums the
original rows.  Test it only on this symmetric square matrix:

```text
[[0, 1],
 [1, 0]]
```

The correct column sums and the incorrect row sums are both `[3, 3]` after
adding one.  The test is green, but it cannot distinguish the behavior from the
known defect.  The mutant therefore **survives this particular input**: the
wrong implementation and right implementation happen to be observationally
identical here.  In mutation-testing terminology, reserve *equivalent mutant*
for a mutant that no possible test can distinguish because its behavior is
semantically equivalent over the whole relevant domain.

Change to the earlier non-square matrix and the distinction becomes obvious:

```text
correct column-oriented result: [5, 7, 9]
wrong row-oriented result:       [6, 15]
```

The shape differs as well as the values.  Rectangular shapes are not random
extra coverage here; they are selected because they break a symmetry that
conceals the hypothesized defect.  This is the kind of reasoning reviewers can
evaluate.  “Added a `2 × 3` case because a square symmetric input cannot detect
an omitted permutation” is stronger than “added another shape.”

The idea of a **mutation test** is to introduce or model a known fault and ask
whether the test suite kills it.  The lab does this safely with a local
`row_sum_mutant`; it does not patch tinygrad.  You do not need to mutate every
contribution.  It is a teaching device for the central question: *what precise
wrong implementation would this assertion catch?*

## Build an oracle that does not repeat the bug

An oracle supplies the expected result.  There is no universally best oracle;
choose the simplest source of truth that is sufficiently independent of the
implementation under test.

### Hand-derived values

Literal values are excellent for tiny examples.  A reviewer can verify
`[5, 7, 9]` without running another framework.  The limitation is scale: hand
answers become error-prone for large shapes, symbolic expressions, exotic
dtypes, and long reductions.

### A simple independent model

The lab implements the contract with ordinary Python loops:

```python
def independent_oracle(data, rows, cols):
  return [
    sum(data[row * cols + col] + 1.0 for row in range(rows))
    for col in range(cols)
  ]
```

It does not call `Tensor.permute`, tinygrad shape movement, or tinygrad
reduction.  If the suspected bug is in those shared mechanisms, this separation
matters.  A second implementation that calls the same faulty helper is not an
independent oracle even if it lives in another test file.

### Another library or backend

NumPy or PyTorch can be useful differential oracles for Tensor semantics.
Another tinygrad backend can expose target-specific rendering or runtime bugs.
But first ask which code is shared:

```text
tinygrad PYTHON result == tinygrad CUDA result
```

is evidence against a CUDA-only defect.  It is weak evidence against a bug in
the shared Tensor frontend, scheduling, or lowering that produced both
programs.  Two agreeing routes are independent only over the parts where their
implementations diverge.

External frameworks also have their own contracts.  Match shape, dtype,
promotion, overflow, reduction accumulation, NaN behavior, and layout before
declaring one the oracle.  NumPy choosing `int64` where tinygrad retains a
narrower integer is a contract mismatch, not automatically a tinygrad bug.

### A relation instead of a complete answer

A **metamorphic test** applies a transformation whose effect is known even when
the complete output is inconvenient to calculate.  For the running example,
adding a scalar `delta` to every input element must add `rows * delta` to each
reduced column:

```text
f(x + delta)[col] == f(x)[col] + rows * delta
```

This checks a useful relationship without producing every expected sum
independently.  It is complementary evidence, not an infallible oracle.  Two
wrong computations can preserve the same relation.  The lab therefore keeps
the literal counterexample and loop oracle too.

## Numerical comparison is part of the contract

The lab's data are small integers stored in `float32`, and its sums are exactly
representable, so exact equality is intentional.  That choice would be wrong
for many other floating computations.

Floating-point arithmetic rounds after operations.  Reassociating a reduction,
using a fused instruction, changing accumulation precision, or choosing a
different transcendental approximation can change the last bits without
changing the accepted mathematical behavior.  NumPy's `assert_allclose`
checks approximately:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

Here `atol` is an absolute allowance near zero and `rtol` scales with the
expected magnitude.  The exact implementation also has policies for NaNs and
asymmetry; read the
[`assert_allclose` documentation](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html)
before relying on remembered formulas.

Choose tolerances from the dtype and operation, not by increasing them until a
test passes.  A useful numerical test includes:

- values near zero, where `atol` dominates;
- nonzero values at the relevant scale, where `rtol` matters;
- a case that would fail if the tolerance became suspiciously loose;
- explicit NaN/Inf/signed-zero policy when those distinctions matter; and
- the same accumulation dtype and overflow semantics as the contract.

For a suspected numerical bug, report the maximum error, expected scale,
dtype, reduction length, and selected tolerance.  “Close enough” is not a
reviewable specification.

## Prerequisite ladder

You can begin the lab if you understand Python functions, loops, lists, and
`assert`.  Before writing a real repository test, add only the background the
next step requires:

1. **Python `unittest`.** Learn `TestCase`, `assertEqual`, discovery, and
   `subTest`.  The lab uses only the standard library so the testing mechanism
   is visible.
2. **pytest invocation.** tinygrad uses pytest to collect many unittest-style
   tests.  Read pytest's official pages on
   [invocation](https://docs.pytest.org/en/stable/how-to/usage.html) and
   [assertions](https://docs.pytest.org/en/stable/how-to/assert.html), then be
   able to select one exact node.
3. **Numerical comparison.** Read NumPy's `assert_allclose` page when the
   contract is not exactly representable.
4. **Property testing.** Learn Hypothesis strategies, invariants, shrinking,
   and deterministic reproduction only when the bug describes a family that a
   small table cannot express well.  The bounded route is linked from
   [Learning resources](../reference/learning-resources.md#testing-transformations).
5. **Compiler/runtime-specific validation.** Return to Chapters 5–14 for the
   artifact whose legality or behavior the test will assert.  Do not copy an IR
   assertion whose meaning you cannot explain.

The goal is not to finish a generic software-testing curriculum before making
progress.  It is to notice the exact missing concept, study it, and return with
enough understanding to state what the next assertion proves.

## Lab 0 — red, then green, with one unchanged contract

Run the lab from the **documentation repository root**, not from `~`.  Replace
the checkout and interpreter paths if yours differ:

```bash
cd ~/Documents/projects/tinygrad_docs

PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python \
  labs/phase5/testing_walk.py --mode red

PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python \
  labs/phase5/testing_walk.py --mode green
```

The red demonstration should still exit successfully.  That is deliberate:
the script requires the expected four owning test methods, exactly 22
`AssertionError` failure records from their parameterized/subtest cases, and
zero unittest error records.  An import error, unexpected exception, different
failure set or count, or mutant that passes makes the script itself fail.

Both modes require the literal caller setting `DEV=PYTHON` before importing
tinygrad; an omitted route or `DEV=PYTHON:PYTHON` is rejected rather than
canonicalized.  The script also pins the optimizer, cache, validation,
visualization, dtype, broadcast, local-memory, and rewrite-tracking settings
printed in its `controlled env` line.  In particular,
`DISALLOW_BROADCAST=0` keeps the deliberate scalar broadcast legal, while
`TRACK_MATCH_STATS=0` prevents an inherited setting from writing a rewrite
capture despite `VIZ=0`.

Interpret the first mode line by line:

```text
candidate: known-bad row-sum mutant
tests run: 5
assertion failures/errors: 22 0
weak symmetric example passed: True
failed contract tests: ['test_10_focused_rectangular_counterexample', 'test_20_output_shape_contract', 'test_30_bounded_differential_grid', 'test_40_add_constant_metamorphic_relation']
red reason: transpose was omitted before the reduction
```

This establishes that the five-test contract is not merely always green.  The
weak symmetric example survives the mutation; four deliberately selected
checks kill it for explainable reasons.

The green mode passes the same contract to `tinygrad_candidate`.  It also
prints the frontend shape and operation names as **localization observations**.
Those names confirm that this recorded expression contains `PERMUTE` and
`REDUCE` before realization.  They are not part of the reusable semantic
contract: a future legal canonicalization could change the frontend artifact
while preserving the required values and shape.

The final two lines delimit the evidence:

```text
claim: semantic contract passed on the portable Python route
non-claim: no compiled renderer, driver, GPU, timing, or full CI matrix was tested
```

Write that distinction in your lab notes.  A portable green test is a real
result; inflating it into a hardware or integration claim makes it less useful.

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

## Expand one counterexample into a claimed domain

The focused counterexample explains the bug.  It is not automatically the
whole domain of the fix.  Before adding cases, write the dimensions over which
the changed contract might vary:

| Dimension | Questions for a Tensor/compiler change |
| --- | --- |
| Shape | Scalar, empty, singleton, square, rectangular, broadcast, symbolic, or very large? |
| Dtype | Boolean, signed/unsigned integer, float widths, bfloat, image, vector, pointer? |
| Values | Zero, one, negative, extrema, overflow boundary, NaN, Inf, repeated, random? |
| Layout/view | Contiguous, permuted, expanded stride-zero, shrunk, padded, offset, non-contiguous? |
| Graph context | Alone, fused with another op, consumed twice, aliased, reduced, differentiated? |
| Compiler mode | `NOOPT`, ordinary optimization, tensor cores, `SPEC`, symbolic variables? |
| Execution mode | Lazy, realized, JIT ignore/capture/replay, graph replay, process replay? |
| Target | NULL analysis, Python interpreter, compiled CPU, renderer target, physical device? |

Do not mechanically form the Cartesian product.  A matrix of 8 shapes × 12
dtypes × 6 layouts × 5 backends can become expensive while still missing the
one relationship that matters.  Use the defect mechanism to select cases.

For an omitted permutation, non-square and non-symmetric data are high-value
because they distinguish coordinate maps.  For an integer rewrite such as
`x % c`, negative values and divisibility boundaries may matter.  For a masked
load, the first invalid index, an all-false gate, and an offset view matter more
than another random contiguous tensor.  For an allocator lifetime defect,
reuse after asynchronous submission matters; ten synchronous arithmetic
shapes do not add that detector.

Classify each selected case:

- a **positive case** reaches the rule and should transform or execute;
- a **negative case** looks similar but must not take the rule;
- an **edge case** sits at a domain boundary such as zero, one, maximum width,
  empty range, or last legal offset;
- an **interaction case** combines the changed behavior with aliasing,
  fusion, symbolic binding, JIT, or another stage; and
- a **portability case** takes the same contract through an implementation
  route that differs at the suspected layer.

Negative cases are particularly important for compiler rewrites.  A rewrite
can be correct wherever it fires and still be wrong because it fires too
widely.  If a rule requires “constant divisor is positive,” test a zero,
negative, and non-constant divisor that remain unchanged.  If it requires a
specific dtype, test the nearest dtype whose semantics differ.  The assertions
should state the semantic or structural reason the rule is inapplicable, not
merely that the graph happens to retain an old `repr`.

### Keep the small case even after finding a family

Suppose a Hypothesis run or deterministic grid finds 73 failing shapes.  Keep
the smallest intelligible counterexample.  It provides:

- a stable red/green reproducer;
- a value a reviewer can calculate;
- a fast case for debugging under verbose modes; and
- a durable record if a property strategy changes later.

Then retain the bounded family test only if it protects a meaningful general
invariant at acceptable cost.  The example and family have different jobs;
neither makes the other redundant.

## Understand what the test runner did

When pytest reports a failure, several earlier stages have already succeeded:

```text
start Python process
  → import pytest/plugins and test modules
  → collect tests and construct node IDs
  → evaluate skips/fixtures/setup
  → call the selected test body
  → execute the code under test
  → evaluate assertions
  → teardown and summarize
```

A command such as:

```bash
python -m pytest test/backend/test_ops.py::TestOps::test_add -x -q
```

selects a **node ID**: file, class, and test method.  Collection must find that
exact node before its body can run.  `-x` stops after the first failure; `-q`
reduces reporting noise.  Neither changes what the test means.

Classify the observed outcome before attributing it to the patch:

| Outcome | What it establishes | What to do next |
| --- | --- | --- |
| Collection error or “not found” | The intended test never ran. | Check checkout, spelling, imports, plugins, and current node ID. |
| Import/setup/fixture error | The body usually never reached its assertion. | Fix or report environment/setup separately from the product behavior. |
| Focused assertion mismatch | The observed contract differed, if the trigger reached the intended path. | Inspect actual/expected values and adjacent artifacts. |
| Unexpected Python exception | A path failed, but not necessarily for the hypothesized reason. | Localize the exception; do not count any exception as the desired red state. |
| Native crash, device loss, or hang | Process/runtime safety failed. | Preserve logs and minimize safely; use timeouts and bounded recovery procedures. |
| Skip | No claim was tested on this route. | Verify the capability is genuinely unsupported and the reason is visible. |
| Expected failure (`xfail`) | A known failure was observed under declared policy. | Ensure an unexpected pass is handled and do not use it to conceal a new regression. |
| Intermittent pass/failure | At least one uncontrolled variable remains. | Record seed/order/device/process state and reduce before weakening the assertion. |

This is why “the old test failed” is incomplete.  A missing CUDA library and an
incorrect CUDA launch can both make a command nonzero, but only one is evidence
for a launch regression.  Capture the relevant assertion or first bad artifact,
not only the exit code.

### Skips describe capability, not convenience

A skip is appropriate when the test's contract cannot exist on a route—for
example, a target deliberately lacks a dtype or a physical device is absent.
Make the predicate narrow and the reason explicit.  Do not catch `Exception`
and call the test skipped: that can turn import bugs, compiler regressions,
driver errors, and assertion failures into apparent success.

Likewise, an `xfail` records known broken behavior; it is not a substitute for
fixing or scoping a new contribution.  Check whether strict unexpected-pass
behavior is configured.  A test that begins passing should prompt removal or
reassessment of the expected-failure marker.

## Process state, imports, and deterministic reproduction

Many tinygrad configuration values are environment-backed `ContextVar`s created
when modules import.  The device registry, compiled-program caches, method
caches, UOp interning, JIT state, allocator caches, and driver contexts can also
survive for the life of a process.  A test that changes `os.environ["DEV"]`
after importing tinygrad has not necessarily selected a fresh backend.

For global modes, prefer a fresh process:

```bash
DEV=PYTHON JIT=0 SPEC=2 CACHEDB=/tmp/focused-python.db \
  .venv/bin/python -m pytest path/to/test.py::TestClass::test_case -x -q

DEV=CPU JIT=0 SPEC=2 CACHEDB=/tmp/focused-cpu.db \
  .venv/bin/python -m pytest path/to/test.py::TestClass::test_case -x -q
```

Use distinct cache paths when the experiment is about compilation or process
replay.  A cache hit is not inherently wrong, but it can prevent the code you
thought you were testing from running.  State whether caches are deliberately
warm or cold.

Inside one test, `Context(FLAG=value)` is useful for a genuinely scoped
configuration whose code reads the context at call time.  It cannot undo
import-time class construction, reopen a device, or erase arbitrary module
state.  When unsure, inspect where the variable is read and isolate the modes
in subprocesses.

Control random seeds, shape generators, worker count, and test order while
minimizing.  If the bug depends on concurrency, do not remove concurrency and
declare it solved; instead bound the schedule, repeat count, timeout, and
resource use so the failure remains safe and reportable.  A deterministic
single-process reproducer is ideal, but the retained test must preserve the
actual condition that triggered the defect.

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
  against tensor or program specs.  The schedule-side tensor check runs while
  constructing a schedule; an in-process `SCACHE` hit reuses its cached
  `LINEAR` and does not repeat that check.
- At `SPEC=2`, this snapshot also checks each newly created UOp against the full
  spec and performs Python-render round-tripping inside boundary
  `type_verify`.  Its constructor-side inferred-dtype check deliberately skips
  constants, nodes with an invalid source, cases with no inferred dtype, and
  weak-equivalent `INDEX` access dtypes.  State those exceptions instead of
  calling it a universal dtype proof.
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
location, and returned program when `CAPTURE_PROCESS_REPLAY=1`. Persistence is
not controlled by that flag alone: pinned
[`diskcache_put`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L415-L434)
is a no-op below `CACHELEVEL=1`, so keep a stable `CACHEDB` and
`CACHELEVEL>=1` for a useful capture.  The replay tool loads those rows,
regenerates programs on the comparison revision, and prints source diffs.

This exact snapshot also contains a case-sensitive tag mismatch that must not
be hidden behind “inspect the files.”  The
[README](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md#L1-L17)
and [workflow](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1-L7)
advertise lowercase `[pr]`; in a pull-request title that enables capture and
the [conditional replay action](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/actions/process-replay/action.yml#L1-L16).
GitHub Actions'
[`contains()`](https://docs.github.com/en/actions/reference/workflows-and-actions/expressions#contains)
is case-insensitive, so an uppercase `[PR]` title also satisfies that lowercase
workflow expression.  The [replay implementation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L1-L8),
however, uses Python's case-sensitive uppercase `[PR]` test when deciding
`ASSERT_DIFF`.  Therefore lowercase `[pr]` alone captures and runs replay but
does **not** turn a generated-source difference into an error; uppercase `[PR]`
in the exported title both enables capture and satisfies the assertion test.
An uppercase tag only in the exported commit message can satisfy the latter
after a lowercase title enabled the action.  Treat this as a pinned
observation, not a timeless convention, and re-read all three live files before
relying on either spelling.

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
- fuzz, model, docs, and lint work have dedicated jobs; process replay is not a
  separate job.  Its conditional composite-action step appears inside many
  unit, backend, model, and platform jobs so it can consume rows captured by
  the tests that ran earlier in that same job.

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
| [UOp construction checks](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L193-L221), [`type_verify`/bounds checks](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L8-L44), [tensor/program/full specs](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L131-L254), and [Python-render round trip](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L256-L278) | Which checks and explicit exceptions compose the pinned `SPEC=2` and `CHECK_OOB=1` behavior. |
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
