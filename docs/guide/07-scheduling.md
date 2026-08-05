# 7. Scheduling and realization

## Purpose

A lazy tensor expression is not yet an execution plan. Before tinygrad can
compile or run it, the expression must acquire storage, be divided into
kernels, ordered around reads and writes, and assigned reusable temporary
memory.

This chapter teaches you to answer two contribution questions:

- Why did this expression become these kernel boundaries?
- Why must these calls execute in this order?

Those questions come before tuning the code inside any one kernel.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

Before continuing, you should be able to:

- read a UOp DAG by following `src` edges rather than treating it as a tree;
- distinguish a pure value from a memory side effect; and
- turn a small tensor expression into conceptual loops, reads, and writes.

If the first two are unclear, use the JAX and MLIR route in
[Learning resources](../reference/learning-resources.md#compiler-bridge-for-an-ml-reader).
If loops and buffers are unfamiliar, work through the bounded
[TensorIR creation tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
until you can annotate a reduction with its input reads and output write.

## Mental model: planning happens in four decisions

For an ordinary realization, the path in this snapshot is:

```text
lazy tensor UOp DAG
  Tensor.linear_with_vars
        │
        ├─ transform_to_call       storage and effects become explicit
        │
        ├─ get_kernel_graph
        │    └─ run_rangeify       iteration, fusion, materialization
        │
        ├─ create_schedule         RAW/WAR-safe topological order
        │
        └─ memory_plan_rewrite     temporary lifetimes and arena slices
                ↓
        LINEAR(CALL, CALL, ...), symbolic values
```

`Tensor.realize` obtains this pair and passes it to `run_linear`. Compilation
comes later. A useful debugging rule is therefore:

> Kernel count is mainly a scheduling question; instructions inside a kernel
> are mainly a codegen question.

The boundary is not absolute, but it puts the first investigation in the right
part of the source tree.

## Callification: from values to storage states

`Tensor.linear_with_vars` first groups the requested tensor roots under a
`SINK` and calls `transform_to_call`. Callification gives lazy values an
explicit relationship to storage and normalizes concrete inputs into call
parameters.

In broad strokes, `transform_to_call`:

1. tags roots that need storage;
2. rewrites required `CONTIGUOUS` values into a buffer, a `STORE`, and an
   `AFTER` state;
3. collects the effects that form the call body;
4. replaces concrete input `BUFFER`, `SLICE`, and bound-symbol forms with
   normalized `PARAM`s; and
5. returns the call plus a map from old tensor expressions to their planned
   storage.

That parameterization makes structurally equivalent work reusable even when it
uses different concrete buffers.

### Storage/effect UOps to recognize

| UOp | Meaning at this boundary |
| --- | --- |
| `BUFFER` | Storage identity with device, dtype, shape/size, and eventually an allocation. Merely seeing one does not prove that bytes have been computed. |
| `PARAM` | A normalized external input to a call. A nonnegative slot denotes a buffer argument; an ALU-space parameter can denote a symbolic scalar. |
| `STORE` | An explicit write into a destination storage state. It is an effect, not a pure returned value. |
| `AFTER` | The value/state in `src[0]`, constrained to occur after the effects or dependencies in later sources. It carries buffer history through mutation. |
| `SLICE` | A contiguous typed region of a parent buffer at an element/byte-compatible offset. Callification can use it for a zero-copy contiguous view; memory planning also uses it for arena suballocation. |

`AFTER` is especially important. A buffer name alone does not say *which
version* of mutable storage a reader observes. Following the `AFTER` chain gives
the state and the effects that produced it.

## Rangeification decides iteration and kernel boundaries

`get_kernel_graph` performs multi-device and early rewrites, converts copies to
stores, calls `run_rangeify`, simplifies removable buffers, converts remaining
staging points to stores, and finally splits closed stores into kernel `CALL`s.

Rangeification walks backward from consumers. It gives values explicit
`RANGE`s and rewrites movement semantics into index expressions. A producer can
inherit a consumer's ranges, allowing its arithmetic to remain inside the same
kernel. If the ranges cannot be shared safely, an axis is partially or fully
materialized and later becomes a kernel boundary.

Important causes of a boundary in this snapshot include:

- explicit `CONTIGUOUS` and `STORE`, which the initial realization map always
  keeps;
- incompatible range needs across multiple consumers;
- reductions interacting with ended or broadcast ranges;
- local/partial staging needed by an indexing pattern;
- a backend's maximum kernel-buffer limit; and
- copies or precompiled/custom calls that are already opaque units.

Afterward, `pm_remove_bufferize` removes staging points when substituting the
consumer's indices is allowed and cheap enough. Its checks include explicit
non-removable buffers, the number of accessed buffers, and reductions. Fusion
is therefore not “fuse every connected elementwise node,” nor is it a complete
global cost optimizer. It is the result of range compatibility, required
effects, and current local heuristics.

### Fusion versus materialization

Fusion keeps a producer expression in a consumer kernel. It usually avoids an
intermediate write and later read, but can duplicate computation, increase the
number of live values, or produce worse indexing. Materialization computes the
producer into storage. It costs traffic and a launch, but creates reuse and a
clean boundary.

Treat `contiguous()` as a semantic request to materialize contiguous storage,
not as a general “make this faster” operation. When investigating a surprising
kernel count, locate the first surviving `STAGE`/store rather than assuming the
original Tensor method directly maps one-to-one to a kernel.

## `create_schedule`: order effects, not source lines

The kernel graph is still a dependency graph. `create_schedule` builds edges
between calls and topologically sorts them into `LINEAR`.

Two hazards matter:

- **RAW (read after write):** a reader must run after the call(s) that produced
  the buffer state it reads. Explicit `AFTER` dependencies are included.
- **WAR (write after read):** a reader of state *S* must finish before a later
  write supersedes *S*. Without this edge, mutation could overwrite bytes while
  an earlier logical value still needs them.

For a kernel, `_states` unwraps views/casts to `AFTER`, `BUFFER`, or `PARAM`
states. The scheduler records writes by underlying buffer, records reads of
particular states, adds RAW and WAR edges, then uses an in-degree queue. A cycle
is an error rather than an order chosen arbitrarily.

The output is:

```text
LINEAR
├── CALL(kernel or copy, output/input buffers...)
├── CALL(...)
└── CALL(...)
```

This `LINEAR` is an execution schedule across kernels. It is different from the
kernel `Scheduler` in the optimization chapter, which changes axes *inside one
kernel*.

## Buffer resolution and memory planning

`create_linear_with_vars` resolves normalized buffer `PARAM`s back to the
actual input/output buffers, recognizes cross-device copy kernels, collects
values for symbolic parameters used by the plan, and then runs memory planning.

The memory planner computes the first and last call in which each eligible
internal buffer appears. Non-overlapping lifetimes can occupy the same arena.
At this snapshot it:

- excludes caller-held buffers and devices that cannot support planned views;
- keeps copy and compute temporaries in separate per-device lanes so reuse does
  not introduce false copy/compute dependencies;
- rounds blocks to 256 bytes and uses a TLSF suballocator; and
- replaces each temporary `BUFFER` with a typed `SLICE` of an `int8` arena.

This is logical planning. The device allocator obtains actual storage later.
A memory regression can therefore originate from missed fusion, a longer
lifetime, an ineligible buffer, or worse arena packing—not only from an
allocator.

## Inspection has a mutation hazard

`Tensor.linear_with_vars` and `Tensor.schedule_linear` are not read-only graph
printers. `linear_with_vars` applies the callification `becomes_map` to all
in-scope Tensor objects, replacing their lazy expressions with planned output
buffers. `schedule_linear` calls it and discards only the symbolic-value map.

Consequences in this snapshot:

- execute the returned plan with `run_linear`, or inspect it in a disposable
  process;
- do not call `schedule_linear()`, throw away the result, and later assume
  `tensor.realize()` will reconstruct the lost work; and
- create a fresh expression/process for before-versus-after schedule
  comparisons.

An unexecuted planned `BUFFER` has storage identity but not the intended
contents. This distinction explains otherwise baffling zero or uninitialized
results in ad-hoc probes.

Use `linear_with_vars` when symbolic values may be present. `schedule_linear`
asserts that its returned variable map is empty.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Public planning and realization boundary | [`Tensor.linear_with_vars`, `schedule_linear`, and `realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L175) |
| Storage/effect normalization | [`transform_to_call`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L202) |
| Initial realization rules and explicit ranges | [`run_rangeify`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L180) |
| Rangeification-to-kernel pipeline | [`get_kernel_graph`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L555) |
| Removable-buffer/fusion cost checks | [`remove_bufferize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L220) |
| RAW/WAR graph and topological linearization | [`create_schedule`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L29) |
| Parameter resolution, copies, symbols, and planner entry | [`create_linear_with_vars`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L171) |
| Lifetime-based arena planning | [`memory_plan_rewrite`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/memory.py#L20) |

Read the functions in table order once. On a real issue, enter at the artifact
that is first wrong: callified storage graph, rangeified kernel graph, ordered
`LINEAR`, or planned buffers.

## Lab 1 — Plan, inspect, and execute one expression

**Portable.** Run from the recorded tinygrad checkout. Predict the root before
planning, the number of calls, and the result before executing:

```bash
DEV=PYTHON DEBUG=0 CACHELEVEL=0 .venv/bin/python - <<'PY'
from tinygrad import Tensor
from tinygrad.engine.realize import run_linear
from tinygrad.uop import Ops

x = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
y = ((x + 1) * 2).sum()
old = y.uop

linear, var_vals = Tensor.linear_with_vars(y)
print("before root:", old.op)
print("mapped root:", y.uop.op)
print("variables:", var_vals)
print("calls:", len(linear.src))
for i, call in enumerate(linear.src):
  print(i, call.op, "body=", call.src[0].op,
        "arguments=", [s.op for s in call.src[1:]])

assert linear.op is Ops.LINEAR and len(linear.src) == 1
assert y.uop is not old                 # planning changed the live Tensor
run_linear(linear, var_vals)            # execute the plan we created
print("result:", y.item())
assert y.item() == 28.0
PY
```

Save the command, output, prediction, surprise, and the source locations that
explain the mapping. Then repeat with `DEBUG=3` and distinguish scheduling
messages from later compilation messages.

## Lab 2 — Force a materialization boundary

**Portable.** Each case runs in a fresh process because schedule inspection
mutates live tensors:

```bash
for boundary in fused materialized; do
  BOUNDARY="$boundary" DEV=NULL DEBUG=0 CACHELEVEL=0 .venv/bin/python - <<'PY'
import os
from tinygrad import Tensor

x = Tensor.empty(16)
mid = x + 1
if os.environ["BOUNDARY"] == "materialized":
  mid = mid.contiguous()
y = (mid * 2).sum()

linear = y.schedule_linear()
print(os.environ["BOUNDARY"], "calls:", len(linear.src),
      "bodies:", [call.src[0].op for call in linear.src])
PY
done
```

At the snapshot, predict and verify one call for `fused` and two for
`materialized`. `DEV=NULL` is sufficient because the question is planning, not
execution.

### Change and regress

Change one feature at a time: add a second consumer of `mid`, move the reduction,
or add/remove `contiguous()`. For each variant:

1. draw the expected reads, writes, and reusable value;
2. predict the boundary before running;
3. find the first `STAGE`/`STORE` that explains the observed plan; and
4. identify whether a semantic rule or a cost heuristic owns it.

On a study branch, temporarily alter only that rule and add a focused test near
the existing rangeify/schedule tests. The regression should assert a durable
property—correctness, required ordering, or an intentional boundary—not a full
unstable graph dump. Restore the source change when the experiment is done.

## Checkpoint

Continue when you can:

- explain what callification adds beyond a lazy value DAG;
- distinguish `BUFFER`, `PARAM`, `STORE`, `AFTER`, and `SLICE` in a storage
  graph;
- trace how range compatibility, explicit storage, and heuristics affect fusion;
- derive RAW and WAR edges for a small mutation example;
- explain why memory planning uses lifetimes and arena `SLICE`s; and
- inspect a schedule without accidentally abandoning the only executable plan.

## Quick reference

| Observation | Inspect first |
| --- | --- |
| Requested output has no planned storage | `transform_to_call` tags, buffer map, and Tensor remapping |
| Unexpected extra kernel | Realization map, incompatible ranges, surviving `STAGE`, buffer-limit rule |
| Unexpected fusion | `remove_bufferize` preconditions and consumer index substitution |
| Mutation reads old/new value incorrectly | `AFTER` states and RAW/WAR edges in `create_schedule` |
| Plan is ordered but memory is high | Temporary lifetimes, held/ineligible buffers, arena peaks |
| Probe gives zeros after schedule inspection | A planned tensor was remapped but its returned `LINEAR` was never run |
| Dynamic-shape schedule loses a value | Use `linear_with_vars`; inspect `used_vars` and returned `var_vals` |
