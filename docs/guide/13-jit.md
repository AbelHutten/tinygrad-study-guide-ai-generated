# 13. TinyJit and graph replay

## Purpose

Compilation caches can avoid compiling the same kernel twice, but an ML step
still pays Python graph construction, scheduling, buffer planning, and per-kernel
submission overhead. `TinyJit` captures a parameterized execution plan so later
calls can skip most of that work; supported runtimes may batch calls into a
device graph as well.

This chapter separates four mechanisms: kernel/program caching, TinyJit capture,
device-graph batching, and the experimental HCQ2 linking path.

**Verified against tinygrad:** `874d331` (2026-08-05).

## Prerequisite gate

You should be able to follow a `LINEAR` plan into allocated buffers and runtime
program calls, and explain why asynchronous launch overhead is different from
kernel duration. You should also understand that a symbolic shape is an
expression plus bindings, not arbitrary dynamic Python control flow.

If queues, signals, or pointer updates are unfamiliar, review
[Devices and runtimes](12-runtime.md) before continuing.

## Four layers of reuse

| Mechanism | Reuses | Still happens on the next call |
| --- | --- | --- |
| Schedule cache | A lowered schedule form for the same graph key | Input handling, execution-plan resolution, compilation lookup, launch |
| Program/compiler cache | Rendered/compiled kernel artifacts | Python frontend, scheduling, memory planning, per-call launches |
| `TinyJit` | Parameterized compiled `LINEAR` plan and temporary-memory layout | Input validation/rebinding and execution |
| Device graph | A supported batch of launches/copies encoded for efficient replay | Pointer/scalar/dimension updates plus graph submission |

Do not report “JIT speed” without naming which cost changed. A one-kernel
workload may benefit from cached compilation but not graph batching; a many-kernel
step can be dominated by submission overhead after kernels are already optimal.

## The three-call state machine

For a newly decorated function with JIT enabled:

```text
call 1, cnt == 0: ignore
  execute the Python function normally and realize returned parameters

call 2, cnt == 1: capture
  run Python while realization records LINEAR plans instead of executing them
  combine plans → parameterize inputs → plan memory → compile → optionally graph
  execute the captured plan once

call 3+, cnt >= 2: replay
  validate input structure/view/dtype/device → bind new buffers and variables
  execute the captured plan without calling the Python function
```

The first call lets initialization and one-time lazy state happen outside the
captured plan. The second establishes the stable computation. If call 1 and
call 2 take different Python branches, the captured branch is the one replayed;
arbitrary future Python control flow is not re-evaluated.

`JIT=0` disables this behavior. At this snapshot, `JIT=2` keeps TinyJit capture
but disables device-graph batching, which is a useful localization experiment.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Decorator and three-call state machine | [`_TinyJit`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L222) |
| Input extraction and validation | [`_prepare_jit_inputs`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L202) |
| Parameterize, re-plan, compile, and graph | [`jit_lower`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L68) |
| Split graphable call batches | [`graph_split_rewrite`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L32) |
| Generic graph runner and update bookkeeping | [`GraphRunner`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L91) |
| Captured replay and written-input protection | [`CapturedJit`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L165) |
| CUDA and HCQ graph implementations | [`runtime/graph/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/graph) |

## What capture changes

During capture, `create_linear_with_vars` hands each schedule to the active
TinyJit object and returns an empty plan, so the ordinary realization does not
execute it. After the decorated function finishes, TinyJit:

1. flattens all captured `LINEAR`s;
2. optionally prunes calls that do not contribute to input-dependent results;
3. replaces each input buffer with a numbered `PARAM`;
4. plans temporary memory while protecting externally held buffers;
5. compiles kernel bodies, optionally with `JITBEAM`;
6. groups supported calls into graph custom functions when graphing is enabled;
7. stores the return structure and input contract; and
8. executes the new captured plan.

Replay substitutes current input buffers for numbered parameters. A graph
runner additionally tracks which command fields contain input addresses,
symbolic scalar values, and launch dimensions so it can patch only what changes.

## The input contract

TinyJit records more than a shape tuple. The expected input description includes
argument names/positions, view structure and symbolic variables, dtype, and
device. Inputs must be backed by real buffers; duplicate aliases among JIT
inputs are rejected. Shallow tensors inside lists, tuples, or dictionaries are
recognized, but capture is not a general recursive Python serializer.

Important consequences:

- a new `Tensor` object with a compatible view/dtype/device can bind to the same
  input slot;
- a symbolic variable may change when the captured representation and bounds
  support it;
- an incompatible view or dtype must fail instead of silently reusing a wrong
  plan;
- reading tensor data during capture is rejected because it would bake a runtime
  value into Python control flow; and
- captured returns may contain tensors or containers of tensors, not arbitrary
  non-tensor values.

Ordinary Python `int`, `float`, and `bool` arguments—and values read from a
closure—are not parameterized JIT inputs. The function is not called on replay,
so their capture-time effect and any Python branch they selected are silently
frozen even if a later call passes a different scalar. Represent a genuinely
dynamic value as a supported Tensor or symbolic UOp input, or reset/use a
separate capture for a different Python specialization.

Treat each rejection as protection of a replay invariant, not an inconvenience
to bypass with an unsafe flag.

## Outputs, mutation, and memory

Captured return tensors refer to the captured plan's storage. Replays update
that storage; retaining outputs across calls requires understanding whether the
same buffer is overwritten. Inputs that the captured plan writes are copied on
replay where necessary so user-visible input buffers are not unexpectedly
clobbered.

Temporary buffers are memory-planned across the combined capture. `free_intermediates`
can release graph runners and eligible allocations, but only after respecting
base/view and live-tensor ownership.

Mutation bugs often appear only under JIT because eager-looking Python order is
replaced by one persistent dependency plan. Trace `STORE`, `AFTER`, input/output
roles, and graph dependencies rather than adding synchronization blindly.

## Lab 1 — Observe all three calls

**Portable.** From the tinygrad study checkout, run:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=CPU DEBUG=1 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/jit_three_calls.py"
```

Before running, predict when `captured` becomes non-`None`, whether the Python
function is invoked on call 3, and how new input values reach the old program.
The outputs must change on every call even though the third call replays.

Add a Python counter inside `add_one` and assert that it increments twice, not
three times. Then repeat with `JIT=0`: all three calls should execute Python and
the probe should report that capture is disabled:

```bash
DEV=CPU JIT=0 DEBUG=1 CACHEDB=/tmp/tinygrad-guide-cache-jit0.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/jit_three_calls.py"
```

## Lab 2 — Separate TinyJit from graphing

Use a function with at least two realized kernels—for example, force a
`contiguous()` boundary between two computations. Run it enough times to reach
replay under:

```bash
DEV=CUDA JIT=1 DEBUG=2 .venv/bin/python your_jit_probe.py
DEV=CUDA JIT=2 DEBUG=2 .venv/bin/python your_jit_probe.py
```

Predict the captured kernel/copy count and which calls the backend can graph.
Inspect `add_one.captured._linear` only in a study probe, or use `VIZ=1` and the
“View captured linear” / “View graphed linear” stages.

Use those `DEBUG=2` runs for structure and synchronized per-call device timing,
not host-submission latency: this debug level makes `run_linear` wait. For a
host-overhead comparison, repeat at `DEBUG=0`, time only already-captured replay
calls without reading their outputs inside the interval, then call
`Device[Device.DEFAULT].synchronize()` *after* the interval. Measure device
execution separately with events or the synchronized debug/profile path.

Record separately:

- number of compiler-level calls before graph splitting;
- number and contents of graph custom-function calls afterward;
- steady-state host submission time;
- device execution time; and
- numerical equality.

One faster wall-clock result does not show whether the improvement came from
graph submission, cache warmth, synchronization, or kernel variation. Always
state whether the timed interval included device completion.

## Lab 3 — Trigger protective failures

In small tests, deliberately try:

1. passing the same realized tensor twice as two JIT inputs;
2. changing dtype between capture and replay;
3. passing an unrealized/virtual input;
4. reading `.item()` or `.tolist()` inside the captured function; and
5. changing a Python integer or boolean that selects a branch after capture;
   observe that the capture-time branch is frozen; and
6. returning a Python scalar alongside a tensor.

For each, predict the invariant being protected, assert the specific `JitError`
or supported behavior, and find the nearest existing test under `test/unit/` or
`test/backend/`. Do not weaken validation merely to accept the call.

## Failure localization

| Symptom | Compare |
| --- | --- |
| First call already wrong | Ordinary frontend/schedule/runtime; JIT has not captured yet. |
| First correct, capture call wrong | Recorded linears, input parameterization, combined memory plan, compilation. |
| Capture correct, replay wrong | Input binding, symbolic values, output reuse, mutation, graph update. |
| `JIT=2` works, `JIT=1` fails | Device graph support, batching, dependencies, pointer/dimension patching. |
| One shape works, another rejects | Expected view/symbolic contract and bounds; determine whether rejection is intended. |
| Replay recompiles or reschedules | Check that the function actually reached replay and identify which cache/capture was reset. |
| Only retained old outputs change | Captured output-buffer reuse; clone/realize at the ownership boundary if semantics require a snapshot. |
| Nested decorated function fails | Nested TinyJit capture is explicitly unsupported in this snapshot. |

## HCQ2 boundary

When `HCQ2` is enabled, `compile_linear` and `link_linear` add another
environment-gated command-queue compilation/linking path. It is active frontier
code in this snapshot, not a stable prerequisite for understanding TinyJit.
First prove whether a failure exists with the default path; then study
`runtime/support/hcq2.py` and its tests as a separate lowering/link layer.

## Checkpoint

Continue when you can:

- narrate ignore, capture, and replay without calling all of them “compilation”;
- state what TinyJit parameterizes and validates;
- distinguish captured `LINEAR` reuse from device-graph batching;
- explain how buffer addresses, scalar variables, and launch dimensions change
  during replay; and
- use `JIT=0`, `JIT=2`, VIZ, and the first/capture/replay boundary to localize a
  failure.

## Quick reference

```text
normal program cache: same kernel → reuse compiled artifact

TinyJit call 1: Python + normal execution
TinyJit call 2: Python + record → parameterize → memory plan → compile → graph → execute
TinyJit call 3+: validate/bind new inputs → replay captured plan

JIT=0: no capture
JIT=1: capture + graph when supported
JIT=2: capture, graph batching disabled
```
