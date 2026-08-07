# 7. Scheduling and realization

## The promise of this chapter

The frontend graph from Chapter 5 says that one value depends on other values.
It does not yet say how many programs will run, where intermediate results will
live, or which call must happen before another. This chapter derives those
decisions from one familiar Tensor expression:

```python
x = Tensor([1.0, 2.0, 3.0]).realize()
mid = x * x + 2 * x
loss = mid.sum()
```

We will begin with Python loops, not compiler passes. From those loops we will
derive fusion, materialization, iteration ranges, storage effects, dependency
hazards, a linear call plan, and temporary-memory lifetimes. Only then will we
read the tinygrad functions that implement each decision.

No compiler-scheduling, memory-allocation, or GPU-programming background is
assumed. By the end, you will be able to:

- distinguish a lazy value, storage identity, allocated bytes, and valid
  contents;
- explain why a Python name does not imply an intermediate array;
- translate a one-dimensional Tensor expression into conceptual loops;
- predict when elementwise work can fuse into a reduction and when an
  intermediate must be stored;
- distinguish an execution call, an uncompiled compute kernel, and a physical
  GPU launch;
- read `BUFFER`, `PARAM`, `STORE`, `AFTER`, `SLICE`, `RANGE`, `INDEX`,
  `STAGE`, and `END` in their scheduling context;
- derive read-after-write and write-after-read edges for versioned storage;
- explain lifetime-based arena reuse without confusing it with Python object
  lifetime;
- inspect and execute a `LINEAR` plan without abandoning the only plan that
  contains the requested work; and
- choose the first scheduling artifact to inspect for an unexpected boundary,
  order, or memory use.

The source and exact outputs are pinned to tinygrad commit `874d331` from
2026-08-05. Retest every implementation-specific boundary on current `master`
before proposing a contribution.

## Compute the carried expression before scheduling it

The arithmetic is small enough to check without tinygrad:

| Name | Formula | Value |
| --- | --- | --- |
| `x` | input | `[1, 2, 3]` |
| `square` | `x*x` | `[1, 4, 9]` |
| `scaled` | `2*x` | `[2, 4, 6]` |
| `mid` | `square+scaled` | `[3, 8, 15]` |
| `loss` | `sum(mid)` | `26` |

A deliberately array-oriented Python implementation allocates several lists:

```python
square = [0.0] * 3
scaled = [0.0] * 3
mid = [0.0] * 3

for i in range(3): square[i] = x[i] * x[i]
for i in range(3): scaled[i] = 2 * x[i]
for i in range(3): mid[i] = square[i] + scaled[i]

loss = 0.0
for i in range(3): loss += mid[i]
```

That is one legal implementation of the formula. It is not forced by the
Tensor source. The Python assignment `mid = ...` binds the name `mid` to a
`Tensor` wrapper whose UOp root is `ADD`; it does not allocate a three-element
device array for that result.

Because every elementwise operation uses the same position `i`, the arithmetic
can instead remain inside the reduction:

```python
loss = 0.0
for i in range(3):
  xi = x[i]
  loss += xi * xi + 2 * xi
```

This is the chapter's **fused** semantic loop. There is no stored `square`,
`scaled`, or `mid`. If `mid` must exist in storage, two loops are needed:

```python
mid = [0.0] * 3
for i in range(3):
  xi = x[i]
  mid[i] = xi * xi + 2 * xi

loss = 0.0
for i in range(3):
  loss += mid[i]
```

This is the **materialized** form. Materialization means computing a value into
storage so later work can read it. It introduces an intermediate write, a later
read, and a boundary between the two loops. On an accelerator, that boundary
usually also means another program invocation and launch.

These loops express semantics, not tinygrad's eventual instruction order. A
GPU reduction may use many threads, partial accumulators, local memory, and
barriers. Chapters 9 and 10 cover those choices inside one kernel. For now the
loops answer the scheduling question: which tensor operations may belong to
the same unit of work?

## Four states that the word “realized” can obscure

Before reading a schedule, keep four facts separate.

### A value description

`ADD`, `MUL`, and `REDUCE` UOps describe values. Their sources say what those
values depend on. They do not by themselves name storage or prove that tensor
elements were computed.

### A storage identity

A `BUFFER` UOp identifies typed storage on a device. A scalar result is commonly
backed by a flat one-element buffer and presented through a zero-dimensional
`RESHAPE`. Both forms can have a buffer identity.

Storage identity answers “which storage?” It does not answer whether physical
bytes have been allocated or filled.

### An allocation

The runtime `Buffer` object can reserve physical or virtual device memory.
`uop.is_realized` becomes true for ordinary global buffers when the backing
buffer is allocated. Planning creates buffer identities but does not normally
allocate them. Execution obtains the allocation when a call needs it.

### Valid contents

Allocated bytes are not automatically a meaningful tensor value.
`Tensor.empty(3)` deliberately has unspecified contents. In this snapshot it
already has buffer identity, while its runtime buffer remains unallocated until
some execution call needs it. Calling `.realize()` directly on that empty
Tensor does not initialize or even necessarily allocate it, because there is no
computation to perform.

The distinctions are visible in two edge cases:

```python
constant = Tensor(2.0)
empty = Tensor.empty(3)
```

- The device-less scalar constant is a virtual value. It schedules zero calls,
  can be observed as `2.0`, and never needs its own buffer.
- `empty` has buffer identity but `empty.uop.is_realized` is false. Its contents
  must not be read as if they had been initialized.

The term **realization boundary** in the scheduler means “this value must be
available from storage.” The internal `realize_map` records such planning
decisions. It is not a map of allocations that already exist, and it is not the
same thing as the public `Tensor.realize()` call.

For an ordinary non-virtual derived Tensor, `.realize()` performs the full
path: plan required work, compile it, execute it, and leave the wrapper pointing
at backed output storage. The special constant, empty, and zero-copy-view cases
are why “realize every Tensor into a new allocation” is an incorrect mental
model.

## The planning ladder

For an ordinary single-device request at this snapshot, the high-level path is:

```text
lazy Tensor/UOp value DAG
        │
        │ Tensor.linear_with_vars
        ▼
transform_to_call
  parameterized whole-request CALL
  explicit output storage and effects
        │
        ▼
get_kernel_graph
  run_rangeify: iteration and candidate staging
  remove_bufferize: fuse removable staging
  surviving staging becomes stores and per-kernel CALLs
        │
        ▼
create_schedule
  dependency graph + topological order
        │
        ▼
parameter resolution, copy recognition, symbolic values
        │
        ▼
memory_plan_rewrite
  eligible temporary lifetimes become arena SLICEs
        │
        ▼
LINEAR(CALL(SINK), CALL(SINK), ...), var_vals
```

The returned `SINK` bodies are not compiled programs yet. Chapter 3 showed the
next transition from `CALL(SINK)` to `CALL(PROGRAM)`. Planning therefore decides
mainly *which work belongs together and in what call order*. Code generation
later decides how the work inside each compute kernel is represented and
executed.

The boundary is not perfectly device-independent. Copies, multi-device graphs,
and selected device buffer limits affect scheduling. It is nevertheless much
earlier than target instruction selection.

## Plan and execute the carried expression

The checked-in lab uses `PYTHON` so the same artifact works without a native
compiler toolchain or GPU; tinygrad still runs its Python-target compilation
pipeline. Run it from the guide repository root with the default memory
planner enabled:

```bash
NO_MEMORY_PLANNER=0 DEV=PYTHON DEBUG=0 \
  ../tinygrad-study/.venv/bin/python labs/phase3/schedule_walk.py
```

If the checkouts are not siblings, replace the interpreter path using the
environment card from Chapter 2. The exact output is:

```text
scalar constant calls: 0
empty identity/allocated: True False
before: REDUCE False False
after planning: RESHAPE True False
calls/variables: 1 {}
call forms: ['CALL(SINK)']
math nodes: ['MUL', 'MUL', 'ADD', 'REDUCE']
after execution: True 26.0
fused math by call: [['MUL', 'MUL', 'ADD', 'REDUCE']]
materialized math by call: [['MUL', 'MUL', 'ADD'], ['REDUCE']]
contiguous BUFFER same UOp/calls: True 0
mutation order: ['read old (+10)', 'overwrite (*2)', 'read new (+100)']
mutation values: [11.0] [2.0] [102.0]
arena offsets: {'a': 0, 'b': 256, 'c': 0}
arena bytes: 512
same arena: True
```

Focus first on the lines surrounding planning:

```text
before: REDUCE False False
after planning: RESHAPE True False
```

Before planning, `loss.uop` is a value-producing `REDUCE`; it has neither a
buffer identity nor an allocation. `loss.linear_with_vars()` returns a plan and
changes the live wrapper to a scalar `RESHAPE` of planned storage. The new root
has buffer identity, but execution has not happened, so `is_realized` remains
false.

The old `REDUCE` object has not been mutated. The lab retains it in
`old_loss`; Chapter 5's logical-immutability model still applies. The Tensor
wrapper now points somewhere else because planning applies a replacement map
to every live wrapper whose graph contains a mapped expression.

The one call's dependency-first math inventory is:

```text
MUL, MUL, ADD, REDUCE
```

That list is a compact UOp inventory, not a rendered instruction stream. It
proves that all four math nodes occur in one scheduled `SINK`. `run_linear`
then compiles and executes the returned plan. Only afterward is the output
backing storage allocated and filled with `26.0`.

## Planning is a mutation-bearing operation

Normal application code should call `.realize()`, `.item()`, or another
observation method and let tinygrad manage the plan. Contributors sometimes
need to inspect `linear_with_vars()` or `schedule_linear()` directly. Those
methods are not read-only graph printers.

`linear_with_vars()` callifies the request, then applies its mapping to all
relevant live Tensor wrappers. The formulas needed to fill the new buffers are
now held by the returned `LINEAR`, not by the remapped output wrapper. Planning
the same wrapper again therefore does not recover the work:

```bash
DEV=NULL DEBUG=0 ../tinygrad-study/.venv/bin/python - <<'PY'
from tinygrad import Tensor

x = Tensor.empty(3)
y = (x + 1).sum()

first = y.schedule_linear()
second = y.schedule_linear()
print(len(first.src), len(second.src), y.uop.is_realized)
PY
```

The pinned output is:

```text
1 0 False
```

The second plan is empty because `y` already names planned storage, not because
the first call executed. Calling `y.realize()` after discarding `first` also
cannot infer the abandoned computation from the remapped wrapper. Fresh output
bytes can appear as zero on some backends, but their value is unspecified; zero
is not the mathematical answer and must not be asserted.

!!! danger "Keep or execute the plan"

    If you call `linear_with_vars`, execute the returned `LINEAR` with its
    `var_vals`, or do the inspection in a disposable process. Never call
    `schedule_linear()`, throw its result away, and treat a later observation
    as a correctness check.

Use a fresh graph or process for before-versus-after boundary comparisons.
Prior execution also removes work from a later schedule: a plan contains
*pending* calls, not a history of calls already performed.

`schedule_linear()` is only a convenience for graphs with no used symbolic
values. It internally calls `linear_with_vars()` and then asserts that the
returned map is empty. The assertion occurs after wrapper remapping. If symbols
might be present, use `linear_with_vars()` from the start rather than catching
an assertion after losing access to the plan.

## Callification: give pure values storage and effects

The frontend carried graph contains `MUL`, `ADD`, and `REDUCE`. Those operations
are **pure** in this context: their values depend on their sources, and merely
describing them does not change external bytes. Execution must eventually
write the requested result somewhere. A write is an **effect** because later
readers can observe whether it happened and which write happened last.

`transform_to_call` makes that relationship explicit. For the fused carried
loss, its essential result is:

```text
CALL
├── SINK
│   └── AFTER
│       ├── RESHAPE(PARAM slot 0)              scalar output state
│       └── STORE
│           ├── RESHAPE(PARAM slot 0)          destination
│           └── REDUCE(... PARAM slot 1 ...)   value to write
├── actual output BUFFER, shape (1,)
└── actual x BUFFER, shape (3,)
```

The exact graph also contains shape-descriptor constants and a scalar reshape
descriptor. The drawing omits those already-explained metadata nodes so that
the storage relationship remains visible.

Read the new UOps as follows.

### `BUFFER`: storage identity, not proof of contents

The actual call arguments are concrete `BUFFER` UOps. Each identifies device,
dtype, and flat capacity. The output buffer can exist as a UOp before runtime
allocation, and allocation can exist before meaningful contents. Follow the
producing effect as well as the buffer identity.

### `PARAM`: a positional call placeholder

Inside the reusable body, concrete buffer identities become `PARAM` slots.
Slot zero corresponds to `CALL.src[1]`, slot one to `CALL.src[2]`, and so on.
Here slot zero is the output destination and slot one is the input `x`.

`PARAM` therefore does **not** mean “input tensor” and does not mean “trainable
model parameter.” It is a positional placeholder. Whether a buffer is read or
written is determined by how that placeholder is used. Later, an ALU-address-
space `PARAM` can represent a symbolic scalar rather than a buffer argument.

Replacing concrete buffers with slots allows structurally equivalent work to
reuse the in-process schedule cache with different actual allocations. The
outer `CALL` reattaches the concrete buffers after the reusable body.

### `STORE`: a visible write effect

`STORE(destination, value)` says that executing the body changes destination
storage. It is not a pure returned array. After rangeification, an `INDEX`
selects the exact destination element for each iteration.

### `AFTER`: one state of mutable storage

`AFTER(state, effect, ...)` represents the value or storage state in its first
source constrained by the effects or dependencies in later sources. It does
not allocate a second buffer or copy the first source.

A useful notation is:

```text
S0 = buffer before a write
W  = STORE(S0, new_value)
S1 = AFTER(S0, W)
```

`S0` and `S1` refer to versions of the same underlying storage. A reader of
`S1` requires `W` first. A reader that logically captured `S0` may need to run
before `W` overwrites the bytes. This version distinction drives the ordering
section later in the chapter.

### `SLICE`: a contiguous sub-buffer view

A scheduling `SLICE` is a typed, contiguous view into a parent buffer. It is
not the general semantics of Python slicing; movement and arbitrary indexing
are covered in Chapter 8.

Its offset is measured in elements of the parent buffer. When the memory
planner uses an `int8` arena, one parent element is one byte, so the printed
offset is a byte offset. Callification can also create a `SLICE` when movement
operations collapse to a zero-copy contiguous region of an existing buffer.

### The buffer map reconnects Tensor wrappers

Callification also returns a mapping from pre-callification value roots to
their planned storage views. For the carried loss it maps:

```text
REDUCE shape ()  ->  RESHAPE(BUFFER shape (1,)) shape ()
```

`Tensor.linear_with_vars` applies that map before returning the schedule. This
is the source of both the useful post-execution buffer identity and the
inspection hazard.

## Rangeification: turn shapes into iteration

The fused Python loop introduced one integer `i` ranging from zero through two.
Scheduling needs an IR representation of that iteration. A `RANGE` UOp
provides an axis with an extent and an axis type. It is a loop-like index at
this stage, not yet a CUDA thread, CPU loop instruction, or vector lane.

For the carried loss, the reduction creates a reduction range conceptually
equivalent to:

```text
r0 = 0, 1, 2
```

An `INDEX(buffer, r0)` means “the element of this buffer selected by the
current `r0`.” The two multiplications and addition all consume `x[r0]`. The
reduction combines their three per-position values. Because those producers
need exactly the same range as their consumer, their arithmetic can remain in
the reduction kernel.

`run_rangeify` walks backward from consumers toward producers. For each value,
the key decisions are:

1. If the value is already marked as requiring storage, give its output fresh
   ranges and end the relevant old ranges there.
2. If it has one ranged consumer, inherit that consumer's compatible ranges.
3. If it has several ranged consumers, merge compatible axes; introduce a new
   range and partial or full realization where the requirements cannot be
   reconciled.
4. For a `REDUCE`, add ranges for the reduced input axes.
5. For movement operations, transform the consumer indices back into source
   indices. Chapter 8 derives those transformations.

### `STAGE`: a candidate stored intermediate

When a value or axis must be available across a boundary, rangeification can
wrap it in `STAGE` with the ranges that close there and `BufferizeOpts` that say
where it would live and whether it is removable. `STAGE` is an intermediate
planning marker, not the final allocated buffer.

The later `remove_bufferize` rewrite asks whether a consumer can substitute
its indices into the staged producer. If so, it removes the `STAGE`, leaving
the producer fused into the consumer. If not, the stage survives.

For a surviving global stage, `bufferize_to_store` creates a `BUFFER`, indexes
it, stores the computed element, closes the relevant ranges with `END`, and
wraps the resulting storage state in `AFTER`. A closed `STORE` or `END` can
then be split into a per-kernel `CALL(SINK)`.

`END` does not mean that the whole program has executed. It closes the loop or
range context attached to an effect so kernel splitting can recognize a
self-contained store.

### The internal `realize_map` is not runtime state

The initial planning map marks surviving `CONTIGUOUS` and `STORE` nodes, plus
selected multi-device or self-access cases. Multi-consumer range conflicts can
add axes later. A star in `DEBUG_RANGEIFY` output means “the planner is placing
a boundary here,” not “this value is already backed by initialized memory.”

This distinction prevents a common misreading: rangeification plans where
storage *will be needed*; `run_linear` later makes the planned writes happen.

## Fusion and materialization in the exact plan

The lab constructs a fresh graph for each alternative. Its compact output is:

```text
fused math by call: [['MUL', 'MUL', 'ADD', 'REDUCE']]
materialized math by call: [['MUL', 'MUL', 'ADD'], ['REDUCE']]
```

The fused schedule contains one compute call:

```text
CALL 0
  read x
  compute x*x + 2*x at reduction range r0
  reduce those values
  store scalar loss
```

The materialized expression adds one line:

```python
mid = (x*x + 2*x).contiguous()
```

It contains two compute calls:

```text
CALL 0
  compute x*x + 2*x over a three-element ordinary range
  store mid[0:3]

CALL 1
  read mid over reduction range r0
  reduce it
  store scalar loss
```

Calling `.contiguous()` constructs a lazy boundary marker here; it does not
execute call zero immediately. Planning the final loss sees both pending calls,
and executing that plan runs them in dependency order.

### `contiguous()` has three possible outcomes

Do not memorize “one `contiguous()` equals one kernel.” At this snapshot:

| Starting value | Planning outcome |
| --- | --- |
| Computed `ADD` such as the carried `mid` | Adds `CONTIGUOUS`; the surviving request produces stored `mid` and a boundary. |
| Existing `BUFFER` or recognized buffer identity | Tensor-level `contiguous()` returns a wrapper around the same UOp; asking only for that value schedules zero calls. |
| Movement chain that is a contiguous subregion | Callification may replace it with a zero-copy `SLICE`, if that device supports such views. |
| Non-contiguous view such as an expanded `(4,1)` buffer viewed as `(4,4)` | Must write a new contiguous representation, producing a call. |

The checked-in lab proves the second row:

```text
contiguous BUFFER same UOp/calls: True 0
```

OpenCL and WebGPU paths reject the sub-buffer-view optimization in the pinned
implementation, so a view result can depend on the selected device. The simple
fused/materialized comparison is portable, but a real backend issue should be
reproduced with the actual backend rather than only `DEV=NULL`.

### Why not fuse everything?

Fusion usually removes an intermediate global-memory write, a later read, and
one execution-call boundary. It can also:

- duplicate a producer used through incompatible paths;
- increase the number of live values inside a kernel;
- make index expressions more complicated;
- combine too many input buffers for a selected backend or configuration; or
- interact poorly with reductions and local/partial staging.

The pinned removal heuristic is local rather than a complete global cost
optimizer. It keeps a user-requested surviving `CONTIGUOUS`, counts reachable
buffers, treats reductions conservatively, and can retain stages whose index
substitution is not allowed or attractive under current `PCONTIG` behavior.
`limit_bufs` can add boundaries when `MAX_KERNEL_BUFFERS` is explicitly set or
when the snapshot's default table supplies a limit for `METAL`, `WEBGPU`, or
`CPU`. That table does not supply a CUDA/NV default.

Copies, precompiled calls, and custom calls can already be opaque execution
units. Consequently, “fuse every connected elementwise node” is neither the
implementation nor a sound performance rule. Measure the resulting kernels in
Chapter 17 before claiming a boundary should move.

## Read the returned `LINEAR` without overclaiming

`LINEAR` is a shapeless container whose `src` tuple is ordered. For ordinary
compute at this stage, each source has the form:

```text
CALL
├── SINK          parameterized uncompiled kernel body
├── BUFFER        actual positional argument
├── BUFFER
└── ...
```

The first source describes what kind of call this is. The later sources are
actual call arguments corresponding to `PARAM` slots inside that body. Inspect
the body's `STORE` targets and reads to determine output and input roles; a
`PARAM` node alone does not state its role.

After all scheduling steps, a public `LINEAR` may also contain:

| Call body | Meaning |
| --- | --- |
| `SINK` | An uncompiled compute kernel. Compilation will turn it into `PROGRAM`. |
| `COPY` | A recognized cross-device or storage copy, possibly handled by a copy engine. |
| `SLICE` | Runtime creation of a supported zero-copy buffer view. |
| `CUSTOM_FUNCTION` | An opaque custom, graph, validation, or backend-specific unit. |
| `PROGRAM` | The body has already been lowered, rendered, and compiled. |

Therefore `len(linear.src)` counts execution calls. In the chapter's small
single-device examples all calls are `CALL(SINK)`, so “one call” and “one
uncompiled compute kernel” coincide. In general, do not report that number as
GPU kernel launches until you classify the bodies. One multi-device call can
also execute once per device, and a copy or view call is not the same as a
compute launch.

`run_linear` iterates through `LINEAR.src` in order. Accelerator runtimes may
submit work asynchronously; the host does not necessarily synchronize after
every call. `LINEAR` is the ordered host execution/submission plan, while
queueing, synchronization, and possible overlap belong to Chapter 12.

## From a dependency graph to one legal order

The materialized carried expression already has a simple dependency:

```text
produce and store mid  ──►  read mid and produce loss
```

The loss call cannot run first because its read would observe storage before
the producing write. This is a **read after write**, abbreviated **RAW**. It is
a true data dependency: the writer must precede the reader.

A schedule is a partial order before it becomes `LINEAR`. Two independent
calls with no path between them can run in either order without changing
values. A **topological sort** chooses one sequence in which every predecessor
appears before its dependent:

1. count each call's unsatisfied predecessors;
2. place calls with count zero in a ready queue;
3. emit one ready call and remove its outgoing dependency edges;
4. enqueue children whose count becomes zero; and
5. fail if calls remain but none is ready, because that means the dependency
   graph contains a cycle.

Do not make tests depend on the relative order of independent calls unless
that order is itself an intended contract. Kernel bodies, required edges, and
numeric results are more durable evidence.

## Mutable storage needs versions and anti-dependencies

Mutation introduces a second hazard. Consider this eager Python analogue:

```python
x = [1.0]
before = [x[0] + 10]  # reads old x
x[0] = x[0] * 2       # overwrites the same storage
after = [x[0] + 100]  # reads new x
```

The intended values are:

```text
before = [11]
x      = [2]
after  = [102]
```

The old reader must finish before the overwrite. That is a **write after
read**, abbreviated **WAR**. It is an anti-dependency: the write does not need
the reader's output, but it must not destroy bytes while the earlier logical
state still needs them.

The new reader must run after the overwrite, which is RAW. The required order
is therefore:

```text
read old x (+10) ──WAR──► overwrite x (*2) ──RAW──► read new x (+100)
```

The lab constructs the Tensor version with explicit contiguous outputs so all
three calls remain visible:

```python
x = Tensor([1.0]).contiguous().realize()
before = (x + 10).contiguous()
x.assign(x * 2)
after = (x + 100).contiguous()

linear, var_vals = before.linear_with_vars(x, after)
```

It verifies:

```text
mutation order: ['read old (+10)', 'overwrite (*2)', 'read new (+100)']
mutation values: [11.0] [2.0] [102.0]
```

`Tensor.assign` represents the write with `STORE` and the resulting state with
`AFTER`. In `create_schedule`, `_states` unwraps views and casts until it finds
an `AFTER`, `BUFFER`, or `PARAM` state. The scheduler records writes by
underlying buffer, records the exact states read by calls, adds RAW edges from
the calls that produced an `AFTER`, and adds WAR edges from readers of an old
state to a write that immediately supersedes that state. It then topologically
sorts the resulting graph.

### A pinned limitation: edge construction is not a proof of complete safety

The source implements RAW and WAR edge construction for the states that reach
`create_schedule`. That fact must not be inflated into a guarantee that every
aliasing, fusion, and lazy-mutation pattern is correct at this snapshot.

`test/unit/test_assign.py` states the intended principle: valid lazy programs
should agree with eager execution. It also records known violations. This
minimal case builds a read between two writes:

```python
buf = Tensor.zeros(4).contiguous().realize()
buf.assign(Tensor.ones(4) * 3)
mid_sum = buf.sum()                    # logically reads four threes
buf.assign(Tensor.ones(4) * 5)
final_sum = buf.sum()
```

The intended values are `mid_sum=12` and `final_sum=20`. At commit `874d331`,
realizing `final_sum` first and then `mid_sum` can produce `20` for both. The
upstream test accepts that result under an explicit `TODO`. Another test notes
an old expression being fused past an assignment and observing new bytes.

!!! warning "Do not document a known wrong value as semantics"

    The eager-equivalent values are the contract the tests describe. The
    pinned wrong result is a known implementation gap and a possible
    contribution target. When diagnosing such a case, find the earliest stage
    where the old state or required boundary disappears; do not assume the
    final topological queue is automatically the guilty pass.

A good assignment regression realizes requested outputs in more than one
order. If only one ordering passes, a missing state or dependency may be
hidden by the observation order.

## Memory planning begins with a timeline

Fusion tries to avoid intermediate storage. Some graphs still need temporary
buffers. Allocating a separate device block for every temporary wastes memory
when their lifetimes do not overlap.

Here **lifetime** means scheduled-call lifetime, not Python reference lifetime
or garbage-collector reachability. For an eligible internal buffer, the
planner records the first and last `LINEAR` call whose arguments contain it.
Consider three 64-element `float32` temporaries:

```text
call index   0   1   2   3
A            █   █
B                █   █
C                        █
```

- `A` and `B` overlap at call 1, so they cannot occupy overlapping bytes.
- `C` begins after both earlier lifetimes end, so it can reuse either region.

At this snapshot each temporary's 256 data bytes already occupy one 256-byte
planner block. One valid arena layout is:

```text
byte offset 0       A, later reused by C
byte offset 256     B
arena size          512 bytes
```

The lab constructs a synthetic `LINEAR` containing only those buffer
appearances and invokes `memory_plan_rewrite`. It verifies:

```text
arena offsets: {'a': 0, 'b': 256, 'c': 0}
arena bytes: 512
same arena: True
```

The planner creates one `int8` arena `BUFFER` and replaces each temporary with
a typed `SLICE` of that arena. Since the parent dtype is one byte wide, the
`SLICE` offsets above are byte offsets. The child views preserve their own
`float32` dtype and element count.

This is logical suballocation. The arena buffer itself is allocated later when
execution needs it.

### Which buffers are eligible?

The public scheduling path excludes concrete outer-call buffers that must keep
their identity, including inputs and requested output storage. The planner also
excludes devices whose runtime path does not support the needed views at this
snapshot: `DISK`, `TINYFS`, `CL`, and `WEBGPU` prefixes.

Eligible internal buffers are grouped by device and by whether they participate
in a copy. Copy and compute temporaries use separate arena lanes so reuse does
not introduce a false copy-compute-copy dependency. The implementation also
extends copy-buffer close events conservatively. Within each lane it rounds
blocks to 256 bytes and uses a TLSF suballocator to choose offsets.

These are snapshot implementation details, not hardware laws. The durable
reasoning is:

1. determine which storage identities may be replaced;
2. derive first and last scheduled uses;
3. prohibit overlap for simultaneously live values; and
4. preserve any ordering constraints that reuse itself could introduce.

If disabling the planner makes a correctness failure disappear, inspect
overlapping `SLICE` regions and lifetime endpoints. If memory is merely high,
the cause may be an extra materialization, an unnecessarily long lifetime, an
ineligible held buffer, conservative copy handling, or poor arena packing—not
only a device allocator.

## Symbolic values travel beside the plan

A fixed-shape graph returns an empty `var_vals` map. With a bound symbolic
extent, a kernel can still contain a symbolic expression that needs a concrete
integer at execution. For example:

```python
from tinygrad import Tensor, Variable
from tinygrad.engine.realize import run_linear

x = Tensor([1., 2., 3., 4., 5., 6., 7., 8.]).realize()
n = Variable("n", 1, 8).bind(3)
y = x[:n].sum()

linear, var_vals = y.linear_with_vars()
print("calls:", len(linear.src))
print("variables:", var_vals)
run_linear(linear, var_vals)
print("result:", y.item())
```

The pinned output is:

```text
calls: 1
variables: {'n': 3}
result: 6.0
```

The plan describes reusable symbolic work; the adjacent dictionary supplies
the binding for this execution. Chapter 8 explains symbolic shapes and index
expressions. The scheduling rule needed now is simple: retain and pass
`var_vals`, and use `linear_with_vars()` whenever the graph might use them.

## Inspect range decisions without mistaking print order for execution order

For a focused fusion question, run a disposable graph with:

```bash
DEV=NULL DEBUG=0 DEBUG_RANGEIFY=1 \
  ../tinygrad-study/.venv/bin/python your_reproducer.py
```

`DEV=NULL` is suitable only when the question is planning and no values will be
read. It does not establish numerical correctness or accelerator behavior.

In the fused carried graph, the range trace shows one starred scalar output
`STORE`; the reduction creates `r0`, and the `ADD` and `MUL` producers use that
same range. With the computed `mid.contiguous()`, it shows a second starred
three-element `STORE` with a different range.

`run_rangeify` prints while walking backward from consumers to producers. The
materialized trace therefore prints the loss-side store before the producer-
side store. That is not execution order. Inspect the returned `LINEAR` to see
the producer call before the consuming reduction call.

Use range output to answer bounded questions:

- Which node first received a fresh range?
- Is the starred point an explicit boundary or a multi-consumer decision?
- Which producer inherited a consumer range?
- Which axis became incompatible?

Do not try to understand every line of a large model trace at once. Reduce the
graph until the first unexpected boundary remains.

## A scheduling failure map

Find the earliest representation already showing the problem:

| Observation | Inspect first | What a divergence means |
| --- | --- | --- |
| Formula, dtype, or shape already wrong | Frontend UOp DAG | The scheduling layer received the wrong value graph. |
| Requested value lacks planned storage or maps to the wrong buffer | Callified body and buffer map | Output tagging, storage normalization, or wrapper remapping is wrong. |
| Compute-call count or body membership is surprising | Range trace, `STAGE`s, closed stores, final `CALL(SINK)` bodies | Fusion/materialization, range compatibility, or a boundary heuristic differs. |
| Calls are right but mutation order is wrong | `AFTER` states and RAW/WAR edges | A state was lost, misclassified, or not ordered. |
| Order is right but temporaries alias incorrectly | Pre/post memory-plan buffers, lifetimes, `SLICE`s | Arena eligibility, endpoints, lane choice, or packing is wrong. |
| Plan is correct but generated code is wrong | Chapters 9–11 artifacts | The first divergence is inside kernel optimization, lowering, or rendering. |
| Program is correct but allocation, launch, or synchronization fails | Chapter 12 runtime trace | The problem is beyond planning. |
| A second inspection reports zero work | Did the first inspection remap live Tensors? | The plan was consumed or discarded; rebuild in a fresh process. |
| Dynamic graph loses a binding | `used_vars` and returned `var_vals` | `schedule_linear()` or an incomplete execution call dropped symbolic state. |

For an unexpected extra call, compare two fresh processes that differ by one
feature. Draw the conceptual loops, inventory each final `SINK`, and locate the
first range or store that separates them. Decide whether it is required
semantics, a device constraint, or a heuristic before editing code.

For an ordering bug, write the eager expected values, label storage versions,
and request all relevant outputs together in several orders. For a memory bug,
draw an interval table before reading allocator code.

Tests should assert the durable property under change. Exact call count is
appropriate when a boundary is the behavior being protected. A full graph dump
is usually too brittle. Pair a scheduling assertion with numeric correctness
when values can be executed, and add mutation-order or aliasing assertions when
those are the risk.

## Guided source tour: one question per stop

Do not open each file at line one. Each stop begins with an observable question
and links only the source needed to answer it.

### Stop 1: can storage exist without allocation or work?

Read [the buffer and constant behavior tests at lines
14–60](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_schedule.py#L14-L60).

Question: which assertions distinguish a `BUFFER` UOp, an allocated runtime
buffer, a derived value without buffer identity, and a constant with zero
scheduled calls?

Translation: lines 16–26 show allocation occurring only when a consuming plan
runs. Lines 46–55 contrast an unrealized `ADD` with its post-execution buffer.
Lines 57–60 show that a virtual constant does not need storage.

Ignore multi-device buffers and non-contiguous views; this stop establishes
only the four-state vocabulary.

### Stop 2: when is `contiguous()` free?

Read [the contiguous schedule tests at lines
82–105](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_schedule.py#L82-L105),
then [the Tensor/UOp-level method at lines
55–62](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L55-L62).

Question: which roots return unchanged, which view becomes a zero-copy region,
and which expanded view requires one call?

Translation: known buffer identity prevents a new `CONTIGUOUS` UOp. A later
callify rewrite can still recognize a contiguous movement region. A genuinely
non-contiguous expansion needs a write.

Ignore backward-contiguous behavior; it belongs to autograd and is not the
forward storage decision here.

### Stop 3: where do planning, remapping, and execution separate?

Read [`linear_with_vars`, `schedule_linear`, and `realize` at lines
175–196](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L175-L196),
then [the live-wrapper mapping helper at lines
19–33](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L19-L33).

Question: which line creates the callified graph, which line changes Tensor
wrappers, which line creates `LINEAR`, and which line invokes `run_linear`?

Translation: planning and wrapper replacement happen before execution. The
scope visitor updates every live Tensor whose reachable graph contains a mapped
root, not every Tensor in the process indiscriminately.

Ignore the weak-dtype guard until a reproducer actually reaches it.

### Stop 4: how does callification make storage reusable?

Read [final-state mapping and argument replacement at lines
166–199](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L166-L199),
then [the orchestration at lines
201–221](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L201-L221).

Question: where is an original value mapped to its stripped storage view, and
where do concrete buffers, slices, or bindings become positional `PARAM`s?

Translation: `ctx.buffer_map` reconnects old roots to planned storage. The
parameterized body plus actual replacements forms one whole-request `CALL`.

Ignore disk and precompiled-call rules. They are neighboring cases, not needed
to explain the carried single-device expression.

### Stop 5: when does `CONTIGUOUS` become a store or a view?

Read [the store conversion at lines
40–48](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L40-L48)
and [the contiguous-view helper at lines
55–87](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L55-L87).

Question: what evidence permits a `SLICE`, and what explicit effect is built
when no zero-copy view is legal?

Translation: a movement chain with a constant contiguous offset can become a
typed sub-buffer view. Otherwise a surviving request receives an output buffer,
`STORE`, and `AFTER` state.

Ignore disk-specific exceptions and multi-device `UNSHARD` handling.

### Stop 6: how are ranges inherited or split?

Read [initial realization rules at lines
10–35](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L10-L35),
[one-consumer range handling at lines
179–234](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L179-L234),
and [multi-consumer handling at lines
235–271](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L235-L271).

Question: why do the carried multiplications inherit `r0`, and which branch
introduces a fresh range for incompatible consumers?

Translation: a single consumer passes its range backward. Multiple consumers
compare indices and validity; incompatible axes enter `realize_map`. A
surviving explicit `CONTIGUOUS` is already in that map.

Ignore movement-index transformations until Chapter 8 and partial local
staging until Chapter 9.

### Stop 7: where does a candidate stage become a kernel?

Read [the removal decision at lines
220–285](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L220-L285),
[global stage-to-store conversion at lines
362–416](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L362-L416),
and [store splitting at lines
520–533](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L520-L533).

Question: which early return keeps a user boundary, what buffer/reduction facts
make removal conservative, and what condition says a store is closed enough to
become a `CALL`?

Translation: returning `None` from `remove_bufferize` retains the stage. A
surviving global stage becomes indexed storage and an ended store; a store with
no open non-device ranges becomes a parameterized kernel body.

Ignore `PCONTIG>2` partial-local logic on the first reading. Return only when a
real reproducer takes that branch.

### Stop 8: how are old and new storage states ordered?

Start with [the intended assignment-order contract and known limitation at
lines 689–748](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_assign.py#L689-L748).
Also read [the old-expression fusion limitation at lines
877–887](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_assign.py#L877-L887).
Then read [state unwrapping at lines
9–27](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L9-L27)
and [dependency construction and topological sorting at lines
29–80](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L29-L80).

Question: which state produces each RAW edge, which old reader produces a WAR
edge, and why does the queue reject a cycle instead of guessing an order?

Translation: the scheduler operates on represented buffer states. The tests
state the semantic goal and also prove that known graph/state cases remain
incomplete, so source inspection must follow the earliest lost state.

Ignore `MSELECT`, `MSTACK`, and multi-device joins for the carried example.

### Stop 9: why may two temporaries share an arena?

Read [the test's lifetime invariant at lines
36–69](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_memory_planner.py#L36-L69),
then [the planner at lines
20–64](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/memory.py#L20-L64).

Question: how are first and last appearances computed, why may dead intervals
overlap in address space, and how does each original `BUFFER` become a typed
`SLICE`?

Translation: calls define inclusive use intervals. The arena's peak live
layout determines its size; non-overlap is checked per live call, not by unique
buffer count.

Ignore TLSF internals. The allocator algorithm is relevant only after a failing
interval/layout example exists.

### Stop 10: where are symbols and held buffers resolved?

Read [`create_linear_with_vars` at lines
170–199](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L170-L199).

Question: where are cached `PARAM`s replaced with actual buffers, where are
copy calls recognized, how are only used bindings collected, and which concrete
buffers are passed as held to memory planning?

Translation: the final public result combines a structurally reusable schedule
with actual storage and execution-time integer bindings. Memory planning runs
after those facts are known.

Ignore JIT capture until Chapter 13.

## Controlled extensions with worked answers

Run each experiment in a fresh process or fresh graph. Predict the math bodies,
required storage, and call order before inspecting.

1. Ask only for `Tensor(2.0)`. Why is a zero-call schedule correct?
2. Compare `Tensor.empty(3).contiguous()` with
   `(Tensor.empty(3)+1).contiguous()`. Which one introduces work?
3. Start from the materialized variant `mid=(x*x+2*x).contiguous()`, then remove
   `contiguous()`. Draw the fused conceptual loop and predict the math UOps in
   its one `SINK`.
4. Use one `mid` twice in `((mid*2)+(mid*3)).sum()`. At the pinned snapshot,
   predict whether the shared producer needs storage.
5. Request two roots together:
   `a=(mid*2).sum()` and `b=(mid*3).sum()`. Predict the pinned call count, then
   state why that count is not a timeless compiler law.
6. Reverse the requested-root argument order in the mutation lab. Which
   dependency edges must remain even if independent queue order changes?
7. In the arena example, move `C` into call 1. Can it still reuse `A`'s bytes?
8. Call `schedule_linear()` twice on the same lazy result. Why is the second
   call count evidence about mutation rather than execution?
9. Use the symbolic `x[:n].sum()` example but discard `var_vals`. Which part of
   the execution contract is missing?
10. Reproduce the known write-read-write limitation with both observation
    orders. Which values are intended, and which pinned result is explicitly a
    bug rather than a new semantic rule?

??? success "Answers"

    1. A device-less scalar `CONST` is virtual. Its value can be resolved
       without materializing device storage, so no execution call is needed.
    2. The empty Tensor already has buffer identity, so `contiguous()` returns
       the same UOp and asking for it schedules zero calls. The computed `ADD`
       has no buffer identity; its surviving `CONTIGUOUS` creates a write and
       one compute call when that value is requested.
    3. The loop accumulates `x[i]*x[i]+2*x[i]` directly. The one body contains
       dependency-first `MUL, MUL, ADD, REDUCE`, plus its indices, params,
       store, constants, and sink.
    4. The two uses are compatible with the same position range in this graph.
       The pinned plan remains one call. This does not prove the producer is
       computed only once in every later lowered representation; inspect the
       kernel if duplicated arithmetic is the performance question.
    5. The pinned plan contains two calls, one closed output store for each
       requested scalar root. Future fusion rules could legally combine
       multiple outputs, so preserve the numeric outputs and intentional
       boundary property rather than treating the count as universal theory.
    6. The old-reader call must precede the overwrite through WAR, and the
       overwrite must precede the new reader through RAW. Reordering roots
       cannot remove those edges. Only calls without a dependency may exchange
       relative order.
    7. No. `A` and `C` would both be live at call 1, so overlapping arena
       regions could corrupt one before its last use. `C` needs a distinct
       region until one lifetime ends.
    8. The first planning call remaps the live result to buffer-backed storage.
       The second sees no lazy formula to schedule even though the storage is
       unallocated. Only executing the first returned plan computes the value.
    9. The reusable plan still contains a symbolic `n`; execution lacks the
       concrete `{'n': 3}` binding needed to infer bounds and launch values.
       Pass the map returned with that exact plan.
    10. Both observation orders should implement eager semantics:
        `mid_sum=12`, `final_sum=20`. At the pinned snapshot, realizing the
        final result first can make both read as `20`. Upstream marks that as a
        `TODO`; it is evidence for a scheduling/state bug, not intended
        mutation semantics.

## Checkpoint: save scheduling evidence

Keep these artifacts. Items 2, 4, 5, and 6 come directly from the lab; for
item 3, redraw and annotate the chapter's verified callification diagram—the
lab intentionally prints a compact inventory rather than the entire graph:

```text
1. hand-computed values and fused/materialized conceptual loops
2. the checked-in lab output
3. one annotated callified graph showing output PARAM, STORE, and AFTER
4. one fused-versus-materialized LINEAR comparison from fresh graphs
5. the RAW/WAR edge drawing and realized values
6. the A/B/C lifetime table and arena offsets
```

You are ready for the indexing chapter when you can answer all of these without
guessing:

1. Why does assigning a Tensor expression to `mid` not allocate an array?
2. How can a UOp have buffer identity but no allocation or valid contents?
3. Why does a scalar constant legitimately schedule zero calls?
4. Why does planning change `loss.uop` from `REDUCE` to `RESHAPE` without
   computing `26`?
5. Which `PARAM` is the output in the callified carried graph, and how did you
   determine its role?
6. How do `STORE` and `AFTER` differ from a pure `ADD`?
7. Why do the two multiplications inherit the reduction's range?
8. What does a `STAGE` mean before and after `remove_bufferize`?
9. When can `contiguous()` be a no-op, a zero-copy `SLICE`, or a real boundary?
10. Why is one `LINEAR` source not always one physical GPU kernel launch?
11. Which edge is RAW and which is WAR in the mutation lab?
12. Why do the source's RAW/WAR rules not prove every pinned mutation case is
    correct?
13. Why may `A` and `C` share an arena offset while `A` and `B` may not?
14. Why must plan inspection either execute the returned plan or occur in a
    disposable process?
15. When is `linear_with_vars()` required instead of `schedule_linear()`?

If an answer is vague, return to the corresponding lab line and source stop.
The goal is not to memorize every rangeify rule. It is to derive a boundary,
state edge, or lifetime from a small graph and know where current heuristics
implement it.

## Quick reference

| Need | Rule |
| --- | --- |
| Decide whether work is pending | Inspect the value root and buffer identity; do not infer from a Python variable name. |
| Distinguish storage states | Identity names storage; allocation reserves bytes; effects establish intended contents. |
| Request normal execution | Use `Tensor.realize()` or an observation method. |
| Inspect a plan safely | Keep and run `(linear, var_vals)`, or use a disposable process. |
| Compare two boundary choices | Build fresh graphs; prior planning and execution change pending work. |
| Interpret `PARAM` | Positional call placeholder; determine read/write role from uses. |
| Interpret `AFTER` | Same underlying storage state constrained after named effects/dependencies. |
| Interpret `SLICE` | Typed contiguous child view; offset is in parent elements. |
| Explain fusion | Producer and consumer can share compatible ranges and removable staging. |
| Explain materialization | A surviving boundary becomes storage, a store, and later a read. |
| Count compute kernels | Classify each `CALL` body; count `SINK`/`PROGRAM` and any opaque custom body whose contract is compute, not every call blindly. |
| Derive call order | Add RAW and WAR edges between exact states, then topologically sort. |
| Check memory reuse | Draw inclusive first/last call intervals and prohibit overlap while both are live. |
| Investigate dynamic work | Preserve `var_vals`; avoid `schedule_linear()` if bindings may be used. |
| Unexpected boundary | Locate the first fresh range, retained `STAGE`, or closed store. |
| Unexpected mutation value | Compare eager semantics, `AFTER` chains, and multiple realization orders. |
| Suspect planner aliasing | Compare with `NO_MEMORY_PLANNER=1` in a fresh process and inspect arena `SLICE`s. |

## Optional reinforcement—not missing prerequisites

- Use the bounded [TensorIR creation route](../reference/learning-resources.md#for-loop-and-tensor-scheduling-intuition)
  only if translating array algebra into loops still feels uncertain. Stop once
  you can annotate the carried reduction's input read, accumulator, and output
  write. TVM's scheduling APIs are not tinygrad's implementation.
- Use the Triton vector-add and fused-softmax route in
  [GPU execution resources](../reference/learning-resources.md#write-two-kernels-elsewhere-first)
  if you want a GPU-oriented picture of why eliminating intermediate traffic
  can matter. Return once you can state the saved reads, writes, and launch.
- Read `test/null/test_schedule.py` around a boundary that resembles your
  reproducer before reading the full rangeify pass. Tests provide small graphs
  and intended call counts.
- Run the carried schedule with `DEV=CPU` after the portable `PYTHON` result.
  Agreement exercises native rendering and execution without claiming GPU
  correctness.

## What is deliberately left for later

- Chapter 8 derives broadcasting, movement operations, symbolic shapes,
  validity, and the exact index transformations only summarized here.
- Chapter 9 maps ranges inside one kernel to global, local, reduction, upcast,
  unroll, and tensor-core choices. Its `Scheduler` is not the cross-kernel
  `LINEAR` ordering in this chapter.
- Chapters 10 and 11 lower a `SINK`, linearize its operations, render target
  source, and compile it into `PROGRAM`.
- Chapter 12 explains physical allocation, argument packing, asynchronous
  submission, synchronization, copies, and runtime execution.
- Chapter 13 explains why JIT capture may consume a plan without executing it
  immediately.
- Chapter 15 uses VIZ and rewrite histories to inspect large scheduling
  transformations without relying only on text dumps.
- Chapters 16–18 turn these artifacts into regression tests, performance
  evidence, and a contribution brief against current `master`.

You do not need those details to proceed. You need to see a lazy formula as
possible loops, identify which values truly require storage, preserve the
state edges that make mutation meaningful, and treat temporary bytes as
lifetimes rather than permanent arrays.

[← Rewrites](06-rewrites.md) · [Next: Shapes and indexing →](08-shapes-and-indexing.md)
