# 10. Lowering a kernel

## Purpose

Scheduling has decided *which* tensor work belongs in one kernel. Lowering must
turn that decision into operations a selected target can actually execute:
explicit iterations, addresses, loads and stores, supported dtypes and math,
GPU dimensions, barriers, and control flow.

This chapter gives you a pass-by-pass map and a debugging method. It does not
ask you to memorize every rewrite in `codegen/`.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

Before continuing, you should be able to:

- translate an elementwise operation and reduction into loops and buffer
  accesses;
- distinguish a semantics-preserving canonicalization from a lowering into a
  more explicit representation; and
- explain GPU global/workgroup/lane dimensions and when shared-memory access
  needs a barrier.

If the first two are unclear, use the JAX, MLIR, and TensorIR route in
[Learning resources](../reference/learning-resources.md#compiler-bridge-for-an-ml-reader).
If the GPU terms are unclear, take the bounded CUDA detour in
[GPU execution on the RTX 4090 path](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path).
Return when you can annotate a simple loop nest with reads, writes, and parallel
dimensions.

## Two schedules with different jobs

The word *schedule* is overloaded:

- The **execution schedule** is a `LINEAR` list of `CALL`s. It orders kernels,
  copies, views, and other work.
- A **kernel schedule** chooses an equivalent internal organization for one
  kernel: axes, grouping, locality, vectorization/upcasting, unrolling, and
  tensor-core use.

`compile_linear` walks the first kind and compiles each kernel body. The second
kind is applied inside `to_program`. Confusing them leads to changes in the
wrong subsystem: fusion is not fixed by an `UPCAST`, and a coalescing problem is
not fixed by reordering independent `CALL`s.

## What enters lowering

A kernel enters codegen as a `SINK`-rooted UOp graph with `KernelInfo`. By this
point, scheduling has selected a kernel boundary and exposed ranges and buffer
parameters. It has not yet guaranteed that every operation, dtype, vector
shape, or control construct is directly renderable on the target.

The target's `Renderer` is an input to lowering. Its capabilities answer
questions such as:

- Which ALU operations and dtypes exist directly?
- Does the target have local memory and thread dimensions?
- Which vector forms and tensor-core layouts are legal?
- How large may global, local, and shared-memory dimensions be?

This is why lowering is target-dependent even before source text is emitted.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Compile each kernel in an execution plan | [`compile_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L268) |
| Cached entry from kernel `SINK` to compiled `PROGRAM` | [`to_program`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L481) |
| Main kernel rewrite pipeline | [`full_rewrite_to_sink`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284) |
| Range and symbolic simplification | [`codegen/simplify.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/simplify.py) and [`uop/symbolic.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py) |
| Shared, target-conditioned operation/dtype decomposition machinery | [`codegen/decomp/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/decomp) |
| GPU dimensions | [`codegen/gpudims.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/gpudims.py) |
| Coalescing, gates, control flow, and register allocation | [`codegen/late/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late) |
| Tensor-IR and program-IR legality | [`uop/spec.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py) |

Read `full_rewrite_to_sink` once from top to bottom before opening individual
matcher definitions. Its named `graph_rewrite` calls are the current pipeline
map and also the labels shown by VIZ.

## The pass sequence by responsibility

The exact calls are snapshot-sensitive. The responsibilities are more durable.

### 1. Normalize and optimize ranges

The pipeline first resolves remaining multi-device markers and movement
semantics, collapses suitable loads, splits/normalizes ranges, and runs symbolic
simplification. It then applies the chosen kernel optimizations—hand-coded or
BEAM-selected.

At the end of this group, the loop organization should be intentional. A wrong
axis choice or missed optimization is already visible here; a renderer change
is too late to fix it cleanly.

### 2. Expand high-level iteration forms

Vector/upcast and unroll choices become explicit shapes and operations.
Reductions become accumulators and loop-carried state. Local buffers are added,
then suitable ranges become GPU dimensions such as global and local IDs.

This group explains several apparent discontinuities in a trace:

- `REDUCE` disappears because accumulation is explicit;
- an on-chip buffer and a later barrier may appear;
- ranges selected for hardware parallelism become `SPECIAL` nodes; and
- one vector-shaped UOp may expand into lane-level work.

### 3. Make memory operations concrete

Broadcasting is expanded, implicit buffer values become `LOAD`s, vector forms
unsupported by the target are devectorized, indices are simplified, and memory
access is coalesced where possible. Index dtypes are lowered only after the
bounds and validity information needed to choose them has been simplified.

If the final address is wrong, compare artifacts before and after movement/range
lowering, load insertion, and index simplification. Do not start by reading the
printed CUDA expression in isolation.

### 4. Decompose for the target

Unsupported operations and dtypes are replaced with supported primitives.
Transcendentals, division/modulo forms, weak dtypes, and renderer-specific
patterns are handled here. The relevant rule set depends on `renderer.code_for_op`
and target capability.

A decomposition must preserve more than a textbook algebraic identity. Test
dtype rounding, overflow, signed zero, infinities, NaNs, validity gates, and
target support as applicable.

### 5. Finalize effects and control flow

The final rules remove invalid forms, insert implicit barriers for local-memory
hazards, turn graph dependencies into explicit control flow, and number scalar
parameters. With `SPEC` enabled, the output is checked against the program spec.

After this point, the graph is ready for linearization or instruction selection.
Changing semantics during text formatting is a design smell: the necessary
decision normally belongs in a preceding lowering pass.

## Invariants to carry through a trace

For each pass, record these rather than trying to compare the entire `repr`:

- output buffer parameters and dtypes;
- logical output shape and valid output indices;
- which values are read and written;
- reduction identity and covered range;
- ordering of effects and required synchronization;
- symbolic inputs plus their bounds; and
- target legality at the pass output.

Kernel scheduling may alter floating-point association, so “equivalent” can
mean the project's permitted numerical tolerance rather than bit identity.
Make that tolerance explicit in a test.

## Lab 1 — Inspect the compiled container

**Portable.** From the recorded tinygrad study checkout, point
`TINYGRAD_DOCS` at this guide's repository and run the supplied probe with the
study virtual environment:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=CPU DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
```

Before running, predict:

1. how many execution calls the realized-input expression needs;
2. which four artifact roles will be children of `PROGRAM`;
3. whether the reduction is still represented by `Ops.REDUCE` in the linear
   program; and
4. the result.

The result must be `54.0`. Do not assert the generated function name or full
source: those details are intentionally free to improve.

Now repeat with `DEV=PYTHON` and, if available, `DEV=CUDA`. Record which source
first line and launch dimensions change. Backend selection is part of the
experiment, so never leave `DEV` implicit.

## Lab 2 — Find the first transforming pass

Use a realized input and a small reduction such as:

```python
out = (x.square() + 2 * x).sum(axis=1)
```

Capture its rewrite trace:

```bash
VIZ=1 DEV=CPU DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db .venv/bin/python your_probe.py
.venv/bin/python -m tinygrad.viz.cli -s TINY | rg 'Schedule|Kernel'
```

Select the kernel name reported by the CLI, list its passes with `--ls`, and
inspect successive representations as documented in upstream's
[`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md).

Build a table with one row for the first appearance or disappearance of each:
`REDUCE`, accumulator/register storage, `LOAD`, `STORE`, `SPECIAL`, and explicit
control flow. Use pass names, not guessed file names.

### Change and regress

Choose one decomposition or late rewrite exercised by the probe. In a study
branch:

1. add a focused test that records its semantic precondition and output
   property;
2. temporarily disable the rule and confirm the test or program-spec check
   exposes the difference;
3. restore it; and
4. run the nearest existing test file with `SPEC=2` where supported.

The goal is learning pass ownership and test design, not submitting the
temporary edit.

## Debugging method: bisect representations

When generated code is wrong, compare one artifact on each side of the suspected
pass group:

1. prove the scheduled kernel computes the intended expression;
2. find the earliest saved pass output whose invariant is violated;
3. reduce the graph while preserving that first divergence;
4. locate the named `graph_rewrite` and matching tests;
5. add the minimized failure before changing the matcher; and
6. validate later stages and another backend after the fix.

Reading all of `codegen/` before locating the first bad pass is usually slower
and produces weaker evidence.

## Checkpoint

Continue when you can:

- distinguish execution scheduling from kernel scheduling;
- explain why target capabilities affect lowering before rendering;
- place range optimization, reduction expansion, load insertion, decomposition,
  barrier insertion, and control flow in the current pass sequence;
- use a trace to find the earliest wrong representation; and
- name the invariant and test oracle for a proposed rewrite change.

## Quick reference

| Symptom | Inspect first |
| --- | --- |
| Wrong kernel count/boundary | Scheduler and rangeification, before codegen |
| Wrong loop/layout choice | Applied kernel opts and pre/post optimization graph |
| Wrong address or mask | Movement/ranges → load insertion → index simplification |
| Unsupported dtype/op reaches renderer | Decomposition and renderer capability set |
| Shared-memory race | Local-buffer dependencies and implicit barrier pass |
| Bad branch/loop form | Gate movement and control-flow insertion |
| Native ISA register corruption | Instruction selection and register allocation |
