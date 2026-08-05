# 3. First trace: follow one expression end to end

## Purpose

This chapter turns the orientation pipeline into evidence from one process. You
will inspect the Tensor UOp DAG, stop after scheduling, execute the resulting
`LINEAR` plan, print generated CPU code, and deliberately change a fusion
boundary.

The goal is not to understand every rewrite. It is to learn how to locate the
earliest artifact that already contains a bug or performance decision. That
skill is the spine of later compiler and GPU work.

## Prerequisite gate

Before continuing, complete the [environment card](02-setup.md#lab-checkpoint-write-an-environment-card)
and answer:

- Does `DEV=PYTHON` produce `[0.0, 0.0, 1.0, 3.0]` for the Phase 1 expression?
- Can you distinguish a Tensor root, a graph node, a scheduled call, generated
  source, and a launch?
- Can you say which checkout and backend produced an observation?

You need only basic graph traversal and the host/device/kernel model from
Chapter 1. The snapshot-only introspection methods used below are study tools,
not APIs to build application code around.

## Mental model: trace artifacts, not Python call stacks

For this expression:

```python
y = (x * 2 + 1).relu()
```

the useful questions change as it moves downward:

| Stage | Question | Artifact in this chapter |
| --- | --- | --- |
| Tensor construction | What semantics did the frontend record? | UOps reachable from `y.uop` |
| Callification | Which values need storage and which buffers become parameters? | A normalized `CALL` body and buffer mapping |
| Scheduling | Where are kernel/copy/view boundaries, and what order preserves dependencies? | An `Ops.LINEAR` root containing calls |
| Kernel lowering | How are shapes and views expressed as iteration and memory access? | A kernel `SINK`, then lower-level UOps |
| Rendering/compilation | What program will this target load? | `PROGRAM` with `SOURCE`, `BINARY`, and `ProgramInfo` |
| Execution | Which buffers and launch dimensions reach the runtime? | `exec_kernel` and one `DEBUG=2` launch line |

At snapshot `874d331`, realization follows this route:

```text
Tensor.realize
  └─ Tensor.linear_with_vars
      ├─ transform_to_call
      └─ create_linear_with_vars
          ├─ get_kernel_graph
          ├─ create_schedule
          └─ memory_plan_rewrite
  └─ run_linear
      ├─ compile_linear → to_program → SOURCE + BINARY
      └─ pm_exec → exec_kernel / exec_copy / exec_view / ...
```

This is not a promise that every future tinygrad version has the same call
tree. It is a map for the pinned source. The durable debugging method is:

1. capture the artifact before a transformation;
2. capture the artifact after it;
3. state the invariant that should have survived; and
4. find the first boundary where it did not.

### Why this example fuses

`x * 2`, `+ 1`, and `relu()` are all elementwise over the same four output
positions. There is no need to materialize their intermediates. In the Tensor
graph, ReLU is already expressed as `CMPLT` plus `WHERE`; after lowering, one
kernel can load an input vector, perform all arithmetic, and store the output.

That is a possibility, not a universal rule. Explicit realization, cross-device
copies, dependencies involving mutation, reductions, and target constraints
can introduce boundaries. Always inspect the plan rather than inferring launch
count from Python syntax.

## Source tour

Read these ranges in order. Stop once you can connect each row to the route
above.

| Boundary | Snapshot behavior | Source |
| --- | --- | --- |
| Tensor → work request | `linear_with_vars` callifies one or more roots; `realize` runs the plan only for roots that still need storage. | [`tinygrad/tensor.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L169-L196) |
| Tensor graph → normalized call | `transform_to_call` marks materializations, creates stores/buffers, replaces concrete buffers with parameters, and returns a buffer map. | [`tinygrad/callify.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L201-L221) |
| Call body → kernel graph | `get_kernel_graph` applies early/movement rewrites, converts movement to ranges, bufferizes, and splits stores into kernel calls. | [`tinygrad/schedule/rangeify.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L520-L577) |
| Kernel graph → ordered plan | `create_schedule` builds read/write dependency edges and topologically emits a `LINEAR` list. | [`tinygrad/schedule/__init__.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L29-L80) |
| Full schedule construction | `create_linear_with_vars` resolves scheduled parameters to buffers, creates copy calls, resolves variables, and applies memory planning. | [`tinygrad/schedule/__init__.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L170-L199) |
| Kernel `SINK` → `PROGRAM` | `do_to_program` performs target rewrites, derives launch metadata, linearizes, renders source, and compiles bytes. | [`tinygrad/codegen/__init__.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L433-L485) |
| Plan → dispatch | `compile_linear` converts kernel bodies to programs; `run_linear` dispatches each call by its body op. | [`tinygrad/engine/realize.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L247-L281) |
| Program → launch | `exec_kernel` allocates referenced buffers, obtains a cached runtime, resolves launch dimensions, and invokes it. | [`tinygrad/engine/realize.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L176-L186) |

## Lab A: stop between scheduling and execution

Use the value-producing Python backend so hardware cannot obscure the graph
transition. Before running, predict:

- the root op before scheduling;
- the number and body type of calls in the `LINEAR` plan;
- whether scheduling alone marks the output realized; and
- the final value.

```bash
CACHEDB=/tmp/tinygrad-guide-trace-python.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python - <<'PY'
from collections import Counter
from tinygrad import Device, Tensor
from tinygrad.engine.realize import run_linear

x = Tensor([-2.0, -1.0, 0.0, 1.0])
y = (x * 2 + 1).relu()

print("device:", Device.DEFAULT)
print("tensor root:", y.uop.op.name, "realized:", y.uop.is_realized)
print("tensor ops:", {op.name:n for op,n in Counter(u.op for u in y.uop.toposort()).items()})

linear = y.schedule_linear()
print("plan:", linear.op.name, "calls:", len(linear.src))
for n, call in enumerate(linear.src):
  print(f"  {n}: {call.op.name}({call.src[0].op.name}), "
        f"device={call.device}, buffers={len(call.src)-1}")

print("after scheduling, realized:", y.uop.is_realized)
run_linear(linear)
print("after execution, realized:", y.uop.is_realized)
print("value:", y.tolist())
PY
```

The pinned output has these stable fields:

```text
device: PYTHON
tensor root: WHERE realized: False
tensor ops: {'CONST': 4, 'BUFFER': 1, 'MUL': 1, 'ADD': 1, 'CMPLT': 1, 'WHERE': 1}
plan: LINEAR calls: 1
  0: CALL(SINK), device=PYTHON, buffers=2
after scheduling, realized: False
after execution, realized: True
value: [0.0, 0.0, 1.0, 3.0]
```

Interpret each line:

- `WHERE` is a semantic frontend root, not a Python control-flow branch.
- `CALL(SINK)` is a scheduled but not yet target-compiled kernel call.
- Two buffers are the output and input; scalar constants are embedded in the
  kernel rather than passed as buffers.
- `LINEAR` records one call, but no runtime has executed it yet.
- `run_linear` compiles and dispatches that call. The Python backend completes
  synchronously here; on an accelerator, “realized” does not itself mean that
  the host has synchronized with queued device work.

`schedule_linear` and `run_linear` expose internals intentionally here. In
ordinary tinygrad code call `realize()`, `numpy()`, `tolist()`, or `item()` as
appropriate and let the Tensor API own this transition.

## Lab B: see one fused compiled program

The Python renderer is portable but its generated representation is not a
friendly first code-reading target. Switch to CPU. Realize the input before
enabling debug output so the host-to-CPU copy and CPU runtime setup do not hide
the computation of interest.

Predict that the scoped debug region will show:

1. one scheduled kernel;
2. one generated C function containing multiply, add, comparison/selection,
   and one output store; and
3. one `*** CPU` execution line after the counters are reset.

```bash
CACHEDB=/tmp/tinygrad-guide-trace-cpu.db DEV=CPU DEBUG=0 \
  .venv/bin/python - <<'PY'
from tinygrad import Context, Device, GlobalCounters, Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()
GlobalCounters.reset()
y = (x * 2 + 1).relu()

print("device:", Device.DEFAULT)
with Context(DEBUG=4):
  y.realize()
print("result:", y.tolist())
PY
```

At `874d331`, the function is named `E_4` and uses a four-wide vector. Names,
vectorization, and timings can change with target and optimizer revisions. The
invariant to check is that the generated function contains the whole
expression and the debug summary reports one compute launch.

`DEBUG=4` is source, not proof of execution. The `*** CPU ... E_4 ...` line is
the separate runtime observation. Keeping those two artifacts distinct becomes
important when compilation succeeds but a launch fails, or when correct source
is launched with wrong dimensions or arguments.

## Lab C: move the boundary yourself

An explicit realization is a materialization barrier. Predict the launch count
for each case, then run:

```bash
CACHEDB=/tmp/tinygrad-guide-fusion.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python - <<'PY'
from tinygrad import GlobalCounters, Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()

GlobalCounters.reset()
fused = (x * 2 + 1).relu().realize()
print("fused launches:", GlobalCounters.kernel_count, fused.tolist())

GlobalCounters.reset()
barrier = (x * 2 + 1).realize()
split = barrier.relu().realize()
print("barrier launches:", GlobalCounters.kernel_count, split.tolist())
PY
```

Checkpoint output:

```text
fused launches: 1 [0.0, 0.0, 1.0, 3.0]
barrier launches: 2 [0.0, 0.0, 1.0, 3.0]
```

The semantics did not change; the execution plan did. This is the smallest
example of a recurring contributor question: “Is this boundary required for
correctness, imposed by a representation, or merely chosen by the compiler?”

Despite its name, `GlobalCounters.kernel_count` is incremented for tracked
execution-plan calls, including copies or views. In this controlled region the
only counted calls are compute kernels; do not use the counter alone to
classify calls in a larger trace.

## Accelerator branch: replay the trace on the 4090

Run Lab B once with `DEV=CUDA` and once with `DEV=NV`. Keep input warm-up and
counter reset identical:

```bash
for dev in CUDA NV; do
  echo "--- $dev ---"
  CACHEDB="/tmp/tinygrad-guide-trace-$dev.db" DEV="$dev" DEBUG=0 \
    .venv/bin/python - <<'PY'
from tinygrad import Context, Device, GlobalCounters, Tensor
x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()
GlobalCounters.reset()
y = (x * 2 + 1).relu()
print("device:", Device.DEFAULT)
with Context(DEBUG=4): y.realize()
print("launches:", GlobalCounters.kernel_count, "result:", y.tolist())
PY
done
```

For each backend, save:

- the generated source;
- the single compute-launch line;
- the numerical result;
- the selected renderer if you overrode it; and
- any failure boundary.

The default CUDA and NV paths commonly render similar CUDA-style source at
this snapshot, but they load and submit it through different runtimes. A
shared source-level error points upward; agreement in source followed by a
runtime disagreement points downward. Later chapters make that localization
precise.

## Troubleshooting

| Observation | Explanation or next check |
| --- | --- |
| The plan includes a copy before the compute call | Constructing a Python list on a non-Python device can require staging. Realize `x` before tracing `y`, then reset `GlobalCounters`. |
| The plan has zero calls | The output may already be realized or virtual. Start a fresh process and construct fresh Tensors. |
| Calling `.tolist()` seems to add work | Value observation must make data host-readable. Separate compute realization from observation as Lab B does. |
| CPU debug output contains support programs | `DEBUG=4` was active during CPU device initialization. Keep `DEBUG=0` in the environment and use `Context(DEBUG=4)` only around `y.realize()`. |
| Generated code is absent on a repeated in-process run | A compiled program may already be in an in-process cache. Use a fresh process; use a fresh `CACHEDB` path when compilation itself is under study. |
| `NULL` cannot return the expected list | `NULL` is for compiler-path tests and fake launches, not numerical copyout. Use `PYTHON` for the semantic control. |
| Kernel name or exact source differs | Confirm the snapshot. On current master, compare invariants and re-find symbols rather than matching old text. |
| `schedule_linear()` was called and then `.realize()` gives surprising work | The scheduling helper updates Tensor roots to planned buffers. For this lab execute the returned plan exactly once with `run_linear`; in application code use only `.realize()`. |

## Phase 1 checkpoint

Pass this checkpoint using artifacts from your own run, not the expected text
above:

1. Draw the path from `y.uop` to the runtime launch and label the artifact at
   every boundary.
2. Explain why the expression is lazy, what forces it, and why one launch is
   sufficient in the fused case.
3. Point to the `LINEAR` call count, generated function body, runtime line, and
   final value that support the explanation.
4. Insert the materialization barrier and explain why it changes launch count
   without changing values.
5. Repeat on your primary accelerator, or record the exact layer preventing
   that run while preserving a passing portable trace.

If you cannot explain why `CALL(SINK)` becomes `CALL(PROGRAM)` before dispatch,
reread the source-tour rows for codegen and execution. If you can explain the
route but individual UOps remain opaque, continue: Phase 2 is devoted to the
graph language and rewrite machinery.

## Quick reference

### Observation controls

| Control | Use |
| --- | --- |
| `DEV=PYTHON` | Numerical semantic control with minimal hardware assumptions. |
| `DEV=CPU` | Readable generated C plus a real compile/execute path. |
| `DEBUG=1` | Device openings and multi-kernel schedule summaries. |
| `DEBUG=2` | One execution/timing line per tracked call; forces timing synchronization. |
| `DEBUG=3` | Single-kernel scheduling and applied optimization detail. |
| `DEBUG=4` | Generated target source. |
| `CACHEDB=/tmp/name.db` | Isolate the persistent compiler/search cache for an experiment. |
| `GlobalCounters.reset()` | Make subsequent launch counts local to the region under study. |

### Pinned pipeline entry points

```text
Tensor.uop.toposort()       inspect the frontend graph
Tensor.schedule_linear()    derive a LINEAR plan without running it
run_linear(linear)          compile and dispatch that exact plan
Tensor.realize()            normal combined schedule-and-run entry point
Context(DEBUG=4)            scope generated-source output
```

These names describe snapshot `874d331`; use
[`rg` and the translation workflow](../reference/source-snapshot.md#translate-the-guide-to-current-master)
before applying them to current `master`.

[← Development setup](02-setup.md) · Continue to Phase 2 after passing the checkpoint.
