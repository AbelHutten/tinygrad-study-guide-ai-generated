# 10. Lowering a kernel

## The promise of this chapter

Chapter 7 followed a Tensor expression until scheduling had chosen one kernel.
Chapter 8 derived the ranges and index expressions inside that kernel. Chapter
9 showed that the same work can be organized in different ways. None of those
decisions is yet a complete program that a selected target can execute.

This chapter crosses that gap. We will start with an ordinary two-row Python
loop, derive every address it reads, and then recognize the same ideas in a
scheduled UOp graph. We will watch lowering replace an abstract reduction with
an explicit accumulator, turn one logical output range into a target-supplied
grid/workgroup coordinate, insert explicit loads, and place the remaining loop
in an ordered program. Only after those changes make sense will we read the
rewrite pipeline that performs them.

No prior compiler construction or GPU programming is assumed. Missing terms
are taught where they become useful. By the end, you will be able to:

- translate a row reduction into nested loops, a flat element-index formula,
  and reads and writes;
- distinguish an optimization from canonicalization, lowering, and target
  decomposition;
- distinguish an execution schedule, a kernel schedule, a scheduled kernel,
  a lowered graph, and a linear program;
- explain why scheduled IR may contain `REDUCE` and implicit buffer values
  while program IR contains accumulator storage and explicit `LOAD`s;
- explain how a logical range can become a target-supplied `SPECIAL` index and
  distinguish grid/workgroup coordinates from local work-item coordinates;
- read static UOp counts without confusing them with dynamic instruction or
  memory-operation counts;
- state exactly what a Python-executed CUDA-oriented target does and does not
  establish;
- locate the pinned pass responsible for reduction state, GPU dimensions,
  memory operations, decomposition, barriers, control flow, or legality; and
- select the first useful artifact when a lowered program is wrong.

The source links and exact observations target tinygrad commit `874d331` from
2026-08-05. Reproduce an observation on current `master` before using it as
contribution evidence.

## Begin with the value, not the compiler

We will carry one expression through the whole chapter:

```python
from tinygrad import Tensor, dtypes

x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]], dtype=dtypes.float32).realize()
out = (x*x + 2*x).sum(axis=1)
```

`x` has shape `(2, 3)`: two rows and three columns. The expression before
`sum` is elementwise. At each coordinate it computes

```text
x[row, column] * x[row, column] + 2 * x[row, column]
```

The reduction uses `axis=1`, the column axis. It combines the three values in
each row and removes that axis. The result therefore has shape `(2,)`, one
number per row.

Compute it by hand before discussing implementation:

| Coordinate | `x` | `x*x` | `2*x` | Sum of the two terms |
| --- | ---: | ---: | ---: | ---: |
| `(0, 0)` | 1 | 1 | 2 | 3 |
| `(0, 1)` | 2 | 4 | 4 | 8 |
| `(0, 2)` | 3 | 9 | 6 | 15 |
| `(1, 0)` | 4 | 16 | 8 | 24 |
| `(1, 1)` | 5 | 25 | 10 | 35 |
| `(1, 2)` | 6 | 36 | 12 | 48 |

The first output is `3 + 8 + 15 = 26`. The second is
`24 + 35 + 48 = 107`:

```text
out = [26, 107]
```

This arithmetic is the semantic oracle for the chapter. Every legal
representation may organize the calculation differently, but it must still
compute these two row sums within the numerical behavior permitted for the
chosen dtype and reduction order.

### Reduction vocabulary

A **reduction** combines several input values into fewer output values using an
operation such as addition, multiplication, maximum, or minimum. Its
**identity** is the value from which accumulation can start without changing
the answer. The identity for addition is zero because `0 + value = value`.

The three values in a row are the reduction's **covered range**. Addition is
mathematically associative over exact real numbers, but floating-point
addition is rounded after operations and is not perfectly associative. A
compiler may use a tree, partial sums, or another permitted association. The
small integers in this example are exactly representable, so that issue does
not obscure the first lesson. It matters when testing a real rewrite.

## Derive the loops one step at a time

An intentionally array-oriented implementation first writes the elementwise
intermediate and then reduces it:

```python
temporary = [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]

for row in range(2):
  for column in range(3):
    xi = x[row][column]
    temporary[row][column] = xi * xi + 2 * xi

out = [0.0, 0.0]
for row in range(2):
  accumulator = 0.0
  for column in range(3):
    accumulator += temporary[row][column]
  out[row] = accumulator
```

There are two units of work separated by stored `temporary` data. The Tensor
expression does not require that intermediate to have its own buffer. Because
the elementwise value is consumed only by the reduction at the same
`(row, column)` coordinate, scheduling can fuse it into the reduction:

```python
out = [0.0, 0.0]
for row in range(2):
  accumulator = 0.0
  for column in range(3):
    xi = x[row][column]
    accumulator += xi * xi + 2 * xi
  out[row] = accumulator
```

This is a semantic loop nest: a clear description of the work, not a promise
that tinygrad will emit this exact source text or run it serially.

### What each loop variable means

`row` and `column` are integer coordinates. Each `range` has an **extent**, the
number of values it visits:

- `row` has extent two and takes values zero and one;
- `column` has extent three and takes values zero, one, and two.

The outer loop chooses an output. Its iterations are independent: calculating
`out[0]` does not require `out[1]`. The inner loop contributes several values
to one output. Its iterations appear dependent in this serial algorithm because
each update reads the previous value of `accumulator`.

That accumulator is **loop-carried state**. “Carried” means that a value
produced by one loop iteration becomes an input to the next. More explicitly,
the first row evolves as follows:

| Moment | Accumulator value |
| --- | ---: |
| Before the loop | 0 |
| After column 0 | `0 + 3 = 3` |
| After column 1 | `3 + 8 = 11` |
| After column 2 | `11 + 15 = 26` |

An abstract `REDUCE` says “combine this range with addition.” An executable
loop needs operational details: initialize state, read it, update it, ensure
the update occurs inside the right loop, leave the loop, and use the final
state. One central job of lowering is to make those details explicit.

### Flatten the two-dimensional input

The input for this example is contiguous in row-major order. “Contiguous” here
means consecutive logical elements occupy consecutive element positions in
storage. “Row-major” means all columns of row zero come before all columns of
row one:

```text
logical coordinates: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
flat elements:           1     2     3     4     5     6
element index:           0     1     2     3     4     5
```

Moving down one row skips three elements, so the row stride is three. Moving
one column advances one element, so the column stride is one. The flat element
index is therefore

```text
row * 3 + column
```

Check every coordinate rather than accepting the formula on faith:

| `row` | `column` | `row*3 + column` | Stored value |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 1 |
| 0 | 1 | 1 | 2 |
| 0 | 2 | 2 | 3 |
| 1 | 0 | 3 | 4 |
| 1 | 1 | 4 | 5 |
| 1 | 2 | 5 | 6 |

These are **element indices**, not byte addresses. A `float32` element occupies
four bytes, so its byte offsets would be `0, 4, 8, 12, 16, 20`. Kernel indexing
algebra normally reasons in typed elements; the later renderer and target ABI
handle the corresponding address representation.

With a flat input, the complete loop becomes:

```python
for row in range(2):
  accumulator = 0.0
  for column in range(3):
    element_index = row * 3 + column
    xi = x_flat[element_index]
    accumulator += xi * xi + 2 * xi
  out_flat[row] = accumulator
```

Chapter 8 showed how reshapes, permutes, expansion, padding, masks, and symbolic
sizes produce less obvious formulas. This chapter deliberately keeps the
address simple so lowering—not movement algebra—is the changing variable.

## What “lowering” means

An **intermediate representation**, or IR, is a program form intended for
analysis and transformation. A tensor expression, a range-based UOp graph, and
an ordered list of low-level UOps can all describe the same calculation while
making different facts convenient to express.

A tensor-oriented representation makes these facts easy to see:

- an addition is a reduction over axis one;
- the result has shape `(2,)`;
- the elementwise producer can fuse with the reduction; and
- reshapes or broadcasting retain mathematical shape meaning.

A target-program representation needs different facts:

- which target-supplied grid/workgroup and local work-item coordinates, or
  which remaining loops, compute each output;
- which loop remains inside each work item or target invocation;
- which buffer element is loaded;
- where the partial sum lives;
- when it is initialized, updated, and read;
- which operations and dtypes the target accepts; and
- which effects require explicit ordering or synchronization.

**Lowering** replaces a convenient or abstract form with a more explicit form
suited to the next stage or target. It does not necessarily mean “convert
directly to machine code.” There can be several lowering steps.

### Four transformation words that should not be interchangeable

The same pipeline contains several kinds of rewrite:

| Word | Question | Small example |
| --- | --- | --- |
| **Canonicalization** | Can this idea be put in one simpler, standard form at the same broad abstraction level? | Simplify `row*3 + 0` to `row*3`. |
| **Optimization** | Can equivalent work be organized to improve a cost such as time, traffic, or code size? | Unroll the three reduction iterations. |
| **Lowering** | Can an abstract operation be expressed using forms accepted by a later representation? | Replace `REDUCE` with accumulator initialization and updates. |
| **Decomposition** | Can an unsupported operation or dtype be expressed using supported primitives? | Replace an unavailable high-level operation with several available ALU (arithmetic/logic) operations. |

These categories overlap at their edges. A canonical form may enable an
optimization; decomposition is a target-conditioned kind of lowering; an
optimization option later expands into lower-level operations. The useful
habit is not policing terminology. It is asking what contract held before the
rewrite, what contract must hold afterward, and whether the change is required
for legality or chosen for cost.

At the pinned snapshot, the function named `full_rewrite_to_sink` contains both
optimization-related rewrites and mandatory lowering. The source itself marks
a point after GPU-dimension creation as “optimizations are done, now we lower
to actual code.” The broad chapter title covers the complete scheduled-kernel
to program-IR transition; it does not imply that every call in the function is
the same kind of transformation.

### Legality is a contract, not a vague claim

An IR is **legal** for a stage when every node satisfies that stage's rules.
Tensor IR permits forms useful before lowering, including movement operations
and `REDUCE`. Program IR has a different contract: weak dtypes and ordinary
movement operations are rejected, `SPECIAL` indices have a required integer
dtype, and local/register buffers must have valid address spaces.

Legality does not prove that the program computes the intended answer. A
well-typed program can use the wrong address. Nor does a correct mathematical
formula prove legality for a target. Correctness needs both semantic evidence
and structural/target evidence.

With `SPEC` enabled, tinygrad checks the scheduled input against the tensor
spec and the lowered output against the program spec. A failing check identifies
an illegal representation; it does not automatically identify which earlier
rewrite caused it.

## The artifact ladder around lowering

Keep the complete path visible:

```text
lazy Tensor/UOp value graph
        │
        │ execution scheduling and rangeification
        ▼
execution LINEAR
  CALL ──► scheduled SINK with KernelInfo
                │
                │ kernel opts + full_rewrite_to_sink(renderer)
                ▼
         lowered program-spec SINK
                │
                │ optional instruction selection, then linearization
                ▼
         ordered LINEAR UOps
                │
                │ render or assemble, then compile where applicable
                ▼
         SOURCE and BINARY in PROGRAM
                │
                │ runtime
                ▼
         allocation, launch/execution, synchronization
```

The same UOp class is reused throughout. Do not infer the abstraction level
from the Python class alone. Inspect the root, node vocabulary, metadata, and
the function that produced the artifact.

### Two meanings of schedule

The word *schedule* is overloaded in tinygrad discussions:

1. The **execution schedule** is the outer `LINEAR` list of `CALL`s. It decides
   which kernels, copies, views, or custom work exist and in what dependency
   order they execute.
2. A **kernel schedule** chooses the internal organization of one kernel:
   global and local axes, grouping, upcast/vector work, unrolling, and
   tensor-core forms.

`compile_linear` walks the first kind and transforms eligible `CALL(SINK)`
bodies into compiled `CALL(PROGRAM)` bodies. `apply_opts`, inside codegen,
handles the second kind.

This distinction localizes changes. Unexpected fusion or an extra kernel is an
execution-scheduling problem. A poor global/local split is a kernel-scheduling
problem. A correct schedule whose `REDUCE` was expanded incorrectly is a
lowering problem. A correct lowered program printed with an invalid token is a
rendering problem.

## What enters lowering for the carried example

After planning, the carried expression has one execution `CALL`. Its body is a
`SINK`-rooted graph with `KernelInfo`. A simplified dependency sketch is:

```text
output PARAM, shape 2
input PARAM, shape 6

row RANGE, extent 2, AxisType.WEAK
column RANGE, extent 3, AxisType.REDUCE

output INDEX  = row
input INDEX   = row*3 + column
element value = x*x + 2*x
row value     = REDUCE(ADD, element value, column RANGE)
STORE(output INDEX, row value)
END(row RANGE)
SINK(STORE)
```

This is not the raw `repr`, and the graph has dependency edges rather than the
textual sequence suggested by the listing. It highlights the facts needed for
comparison.

Several details are already explicit:

- scheduling selected one kernel boundary;
- output and input buffers are kernel parameters;
- the kernel has an output range and a reduction range;
- the input element index is `row*3 + column`;
- the result is stored in output storage; and
- `KernelInfo` carries the kernel's scheduling information.

Several details are still abstract or incomplete for program IR:

- `REDUCE` describes a combination without explicit accumulator operations;
- accessing an indexed buffer value does not yet require an explicit `LOAD`;
- the weak output range has not yet become a target program dimension;
- some index and constant dtypes are still weak;
- target-unsupported operations may remain; and
- graph dependencies have not all become an ordered control-flow list.

The target's `Renderer` is passed into lowering because the desired output is
not universally identical. A renderer advertises supported ALU operations and
dtypes, local/thread support, shared-memory limits, vector behavior, tensor
cores, and dimension limits. Lowering can therefore make target-dependent
choices before any source text exists.

## Controlled lab: compare scheduled and lowered forms

The supplied probe records a compact set of stable facts rather than printing
an enormous raw UOp graph. Run it from the pinned tinygrad study checkout. Point
`TINYGRAD_DOCS` at this guide repository:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=PYTHON::sm_89 DEBUG=0 NOOPT=1 SPEC=2 \
  CACHEDB=/tmp/tinygrad-guide-lowering.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/lowering_walk.py"
```

Use one physical shell line per displayed continuation: the backslash must be
the final character on its line, with no space after it.

The command deliberately controls five variables:

- `DEV=PYTHON::sm_89` selects the Python runtime with a CUDA-oriented `sm_89`
  target. It needs no NVIDIA GPU and does not execute CUDA.
- `DEBUG=0` prevents ambient debug settings from changing or breaking output.
- `NOOPT=1` suppresses default heuristic kernel opts so the three-element
  reduction remains a visible loop.
- `SPEC=2` enables strong UOp construction/spec checks in this snapshot.
- `CACHEDB` isolates this exercise's disk cache from ordinary development.

The lab also sets `dtype=dtypes.float32` in code. An exported
`DEFAULT_FLOAT=double` therefore cannot silently invalidate the four-byte
address derivation or change the asserted accumulator dtype.

`NOOPT=1` does **not** turn codegen off. `apply_opts` first converts eligible
loop axes into global axes, then skips the hand-coded heuristic. Explicit opts
or BEAM have different precedence. The rest of mandatory lowering still runs.

The lab compiles the execution plan once, then takes its lowered graph from
the resulting `PROGRAM` child. Its marker/address assertions and final numeric
execution therefore describe the same compiled artifact rather than two
independently lowered copies.

The exact pinned output is:

```text
target: CUDA:PYTHON:sm_89
calls/variables: 1 {}
applied opts: ()
scheduled markers: [('RANGE', 2), ('REDUCE', 1), ('LOAD', 0), ('SPECIAL', 0)]
lowered markers: [('RANGE', 1), ('REDUCE', 0), ('LOAD', 3), ('SPECIAL', 1)]
accumulator: REG float 1
global axis: gidx0 2
remaining loop: REDUCE 3
input addresses: [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((1, 0), 3), ((1, 1), 4), ((1, 2), 5)]
barriers: 0
launch global/local: (2, 1, 1) (1, 1, 1)
linear control: [('SPECIAL', 'gidx0'), ('RANGE', 'REDUCE'), ('END', None)]
result: [26.0, 107.0]
```

If this differs, first confirm the commit, the five environment settings, and
that the command uses the pinned checkout's Python environment. Do not edit a
current `master` checkout until it resembles an old observation; record the
difference and locate the current symbols instead.

## Read the lab output as a sequence of claims

### `target: CUDA:PYTHON:sm_89`

This target string has three relevant pieces. `CUDA` describes the
architecture-oriented target family, `PYTHON` identifies the renderer/runtime
route, and `sm_89` is the selected architecture. An RTX 4090 is an Ada GPU with
compute capability 8.9, so this target is useful for examining its target
shape without requiring the card.

The actual objects on this route are `PythonDevice`, `PythonRenderer`,
`PythonCompiler`, and `PythonProgram`. `PythonRenderer` changes its target to a
CUDA-oriented target when the architecture begins with `sm`, but it still
serializes UOps for Python execution. It does not use the CUDA driver, launch
physical GPU threads, exercise concurrent work items, emit native CUDA source,
or measure an RTX 4090.

### `calls/variables: 1 {}`

Scheduling selected one execution call. The realized input is already stored,
so only the fused elementwise row reduction needs computation. `{}` says there
are no symbolic variable bindings for this fixed `(2, 3)` example.

This line is about the execution schedule. The remaining lines examine the
single kernel inside that call. One execution call will later correspond to one
program invocation here, but “call,” “kernel body,” and “physical GPU launch”
are not synonyms across every backend or stage.

### `applied opts: ()`

The empty tuple confirms that this controlled kernel carries no explicit or
heuristic optimization recipe. `NOOPT=1` suppressed the hand-coded heuristic.
Mandatory codegen still converted the eligible output range to a global role,
which is why `gidx0` appears later despite the empty tuple. If you repeat the
probe with optimization enabled, this line exposes the complete chosen recipe
before you interpret changed range or launch structure.

### Scheduled markers

```text
[('RANGE', 2), ('REDUCE', 1), ('LOAD', 0), ('SPECIAL', 0)]
```

The two ranges are the output-row range and reduction-column range. `REDUCE`
still expresses the abstract add reduction. There are no explicit `LOAD`
nodes: at this level an indexed buffer value can flow into elementwise
arithmetic without spelling the load operation separately. No range has yet
become a hardware/program coordinate, so there is no `SPECIAL`.

The scheduled graph already contains a `STORE`. This asymmetry is useful:
scheduling needs an explicit output effect to establish the kernel boundary,
while codegen can still treat reads as values reached through indexed buffers.
Do not assume that every operation becomes explicit in one all-or-nothing step.

### Lowered markers

```text
[('RANGE', 1), ('REDUCE', 0), ('LOAD', 3), ('SPECIAL', 1)]
```

One range remains and one became `SPECIAL`. The `REDUCE` node is gone, but the
reduction semantics are not. They now live in a reduction loop plus explicit
accumulator state. Explicit `LOAD`s have been added where computations consume
values from global or compiler-managed register storage.

These are **static graph counts**. A node appearing once in the program graph
can execute many times. Conversely, a compiler can reuse one loaded value in
several arithmetic consumers because a graph node can have several outgoing
dependency edges.

The three `LOAD` nodes are not three global-buffer reads:

1. one loads the input element selected by `row*3 + column`;
2. one reads the accumulator during its loop update; and
3. one reads the completed accumulator for the output store.

The input-load node executes dynamically for two rows times three columns: six
input elements. The accumulator update load also participates in repeated loop
iterations, while the final accumulator load occurs once per output work item
in this local-size-one schedule. Static UOp count, dynamic operation count, and
actual hardware memory transactions are three different measurements.

### `accumulator: REG float 1`

Reduction lowering created a compiler buffer with `AddrSpace.REG`, float
element dtype, and one element. Conceptually it holds the partial sum for one
output row:

```text
initialize accumulator to 0
for each column:
  accumulator = accumulator + expression(x[row, column])
use final accumulator
```

This is not a new two-element global Tensor buffer. It is one scalar piece of
compiler-managed storage per active work item in this lowered model.
`REG` expresses the intended program address space. For native ISA paths,
physical register assignment and possible spilling are later concerns; do not
infer a measured hardware register count from this one buffer UOp.

Keep the three address-space intentions separate:

| Address space | Intended visibility in this program model | Typical role |
| --- | --- | --- |
| `GLOBAL` | Device/runtime-visible kernel buffers | Tensor inputs and outputs |
| `LOCAL` | Work items in one workgroup | Cooperative tiles or partial sums |
| `REG` | One work item | Accumulators and private temporaries |

These are compiler contracts, not measurements of a final physical placement.
A downstream compiler can assign registers or spill values according to its
own target rules.

The reduction rewrite also introduces effect dependencies around initialization
and update. A graph cannot treat stores as freely reorderable pure arithmetic.
The ordered program must initialize before the first read, update inside the
reduction loop, and read the final value only after the loop ends.

### `global axis: gidx0 2`

`gidx0` is a target-supplied **grid/workgroup coordinate** with extent two. On
the pinned CUDA renderer, `gidx0` maps to `blockIdx.x`; a local index such as
`lidx0` maps to `threadIdx.x`. Correspondingly, tinygrad's program
`global_size` records grid/workgroup counts and `local_size` records work items
(CUDA threads) within each group.

An executing GPU work item is therefore identified by both its group coordinate
and its local coordinate. This controlled schedule has `local_size=(1,1,1)`,
so each group contains exactly one work item. In this special case `gidx0=0`
identifies the sole work item computing `out[0]`, and `gidx0=1` identifies the
sole work item computing `out[1]`. If local extent were greater than one,
calling `gidx0` a complete work-item ID would be wrong.

You can understand the controlled case without GPU hardware by imagining two
groups, each containing one invocation with a built-in integer:

```python
kernel_workgroup(group_id=0, local_id=0)  # computes row 0
kernel_workgroup(group_id=1, local_id=0)  # computes row 1
```

On the Python route, the interpreter emulates these coordinates' semantics.
`SPECIAL` means “this coordinate is supplied by the execution model,” not
“this graph proves a physical GPU block or thread existed.”

Why parallelize the row range? Each row writes a distinct output and does not
depend on the other row. Why not make the reduction column a global axis too?
Independent columns contribute to the same accumulator, so doing so requires a
parallel reduction organization—partial sums, grouping, local storage, or
another combination strategy. Under this controlled schedule, columns remain
a serial reduction loop within each work item.

### `remaining loop: REDUCE 3`

The remaining `RANGE` still has `AxisType.REDUCE` and extent three. `REDUCE`
the operation has disappeared; `REDUCE` the axis classification remains. The
two uses answer different questions:

- `Ops.REDUCE`: what abstract value-combining operation is this node?
- `AxisType.REDUCE`: what role does this iteration range play?

For each group coordinate `gidx0`, its sole work item visits three column
coordinates. The accumulator turns those visits into one output value.

### Input address samples

```text
[((0, 0), 0), ((0, 1), 1), ((0, 2), 2),
 ((1, 0), 3), ((1, 1), 4), ((1, 2), 5)]
```

The lab locates the lowered input `INDEX`, substitutes concrete values for
`gidx0` and the reduction range, simplifies the expression, and records all
six coordinates. They agree with `row*3 + column`:

```text
(0, 0) -> 0*3 + 0 -> 0
(0, 1) -> 0*3 + 1 -> 1
(0, 2) -> 0*3 + 2 -> 2
(1, 0) -> 1*3 + 0 -> 3
(1, 1) -> 1*3 + 1 -> 4
(1, 2) -> 1*3 + 2 -> 5
```

The lowered expression may be represented by a compact multiply-add UOp such
as `MULACC`. Treat that as IR, not an automatic claim about one fused hardware
instruction or its rounding. The selected renderer decides how a supported
operation is emitted or executed. Address substitution is the stronger lesson:
despite representation changes, each logical coordinate reaches the intended
element.

### `barriers: 0`

The executed lowered graph contains no `BARRIER`. That is expected: each
workgroup has one work item, the accumulator is private `REG` state, and no
work item communicates through workgroup-local storage. This structural check
supports the claim that this particular program needs no workgroup barrier; it
does not prove that an unrelated cooperative kernel is race-free.

### Launch dimensions

```text
global = (2, 1, 1)
local  = (1, 1, 1)
```

Launch metadata is derived from the lowered `SPECIAL` nodes. There are two
workgroups along grid dimension zero, each with one work item along every local
dimension.

A **workgroup** is a group of work items that can cooperate through a target's
local/shared storage and workgroup-scoped synchronization. Its members have
local IDs. This example uses no such cooperation, so local extent one is the
neutral shape.

Do not generalize the exact tuple from the Tensor shape alone. The default
heuristic or explicit kernel opts can factor ranges into different global,
local, unrolled, or per-work-item roles while preserving the same result.

### Linear control

```text
[('SPECIAL', 'gidx0'), ('RANGE', 'REDUCE'), ('END', None)]
```

The lowered `SINK` is still a dependency graph. Linearization produces an
ordered list in which dependencies appear in a legal execution order and
control structures have explicit placement. This normalized inventory shows
the global coordinate, opening reduction loop, and loop end. Arithmetic,
loads, and stores also appear in the full ordered list; the lab prints only
control markers to keep the assertion readable.

An ordered list is not necessarily native machine instruction order. A source
renderer still converts it to a target language; an ISA renderer may perform
instruction selection and register allocation. Chapter 11 follows those
branches.

### Final result

`run_linear(..., jit=True)` executes the plan that was already compiled. The
result `[26.0, 107.0]` agrees with the hand calculation. This numeric check is
necessary, but it is not sufficient evidence for every lowering property. A
wrong address could remain hidden in a symmetric input; a race could fail only
under real concurrency; an unsupported dtype could be absent from this
example. Tests should be chosen for the claim being made.

## A conceptual before-and-after program

Putting the observations together gives this approximate transition. The left
side describes intent; the right side describes operational state:

```text
SCHEDULED                              LOWERED

row = RANGE(2, WEAK)                   row = SPECIAL(2, "gidx0")
column = RANGE(3, REDUCE)              acc = REG_BUFFER(float, 1)
                                      STORE(acc, 0.0)       # initialize
idx = row*3 + column                   column = RANGE(3, REDUCE)
xvalue = INDEX(input, idx)             idx = row*3 + column
term = xvalue*xvalue + 2*xvalue        xvalue = LOAD(INDEX(input, idx))
row_sum = REDUCE(ADD, term, column)     old = LOAD(acc)
STORE(INDEX(output, row), row_sum)      STORE(acc, old + term)
END(row)                               END(column)
                                      row_sum = LOAD(acc)
                                      STORE(INDEX(output, row), row_sum)
```

The real graph contains effect dependencies and target-normalized forms that
this pseudocode omits. The comparison nonetheless explains the marker changes:

- the output range is supplied by the execution model;
- the reduction range remains a loop;
- the abstract reduction becomes explicit state;
- implicit buffer consumption becomes `LOAD`; and
- the final graph contains enough control/effect information to linearize.

## The pinned pass sequence by responsibility

Do not memorize every matcher. Learn the sequence well enough to put a bad
artifact on the correct side of a pass group. The precise order below is the
recorded snapshot, not a permanent API.

### 1. Verify and preprocess the scheduled graph

`full_rewrite_to_sink` optionally records the base AST for VIZ and, when
`SPEC` is enabled, verifies it against the tensor-IR spec. It resolves remaining
in-kernel multi-device markers and lowers early movement semantics.

At this boundary, ask:

- Does the scheduled kernel represent the intended output and inputs?
- Are the two ranges and `row*3 + column` address correct?
- Is the reduction identity and covered range correct?
- Is the kernel boundary itself correct?

If not, codegen is receiving the wrong program. Return to scheduling or
indexing rather than compensating in a renderer.

### 2. Normalize ranges and apply kernel opts

When optimization is enabled, the pipeline collapses certain indexed-load
forms, splits/normalizes ranges, runs symbolic rewrites, simplifies ranges, and
calls `apply_opts`.

`apply_opts` constructs the kernel `Scheduler`, converts eligible loop ranges
to global ranges, then uses explicit opts, BEAM, or the hand-coded heuristic
according to its precedence. It returns an optimized AST carrying the applied
option recipe.

This is why `NOOPT=1` in the lab still produces `gidx0`: conversion of an
eligible output loop to a global role occurs before the heuristic skip. It also
explains why a wrong unroll, grouping, or local split should be debugged before
reduction lowering.

Without `NOOPT=1`, the pinned heuristic chose an unrolled reduction and a
local output factor of two for this tiny target-shaped example. It removed the
remaining `RANGE` and produced global/local sizes `(1, 1, 1)` and `(2, 1, 1)`.
That is an optional snapshot observation, not a rule that two rows must map to
two local work items in one group.

### 3. Expand scheduled organization

Post-option symbolic rewrites prepare the chosen organization. The expander
makes upcast, unroll, and related vector-shaped work more explicit. A compact
option recipe is therefore not the final operation graph.

For a simple `NOOPT=1` reduction there is little exotic organization to
expand. On an optimized kernel, this group can make one UOp into several lane
values, remove an unrolled range, or expose grouped-reduction structure. If a
chosen `UNROLL` is correct before expansion but values are missing afterward,
the expander boundary is the useful comparison.

### 4. Replace reductions with state

For a range reduction like the carried example, the `remove reduces` rewrite
creates register accumulator placeholders, initializes them with the reduction
identity, updates them inside the covered ranges, and makes the completed value
available after those ranges end. Other cases, including horizontal reductions,
can take a different lowering branch without this exact scalar `REG` shape.

This is the central transition in the carried example. Its invariant is not
“there is still a `REDUCE` node.” It is:

- initialization uses the correct identity and dtype;
- every intended iteration contributes exactly once;
- the combine operation and input value are correct;
- updates occur in the intended ranges;
- final reads occur after the range; and
- separate output work items do not accidentally share state.

Grouped reductions can require additional local storage and cross-work-item
combination. The same invariants apply, but their operational form is more
complex.

### 5. Add local buffers and program dimensions

The next rewrites add local buffers required by suitable scheduled forms and
convert ranges assigned to target execution dimensions into `SPECIAL` indices.

`gpudims.py` collects ranges by axis role and substitutes them with special
coordinates such as global or local IDs. The renderer supplies maximum
dimensions and whether the relevant execution model exists. The carried
output range becomes `gidx0`; its reduction range remains an ordinary loop.

A local buffer is storage shared at a target-defined cooperation scope, often
a GPU workgroup. It is not the same as the per-work-item register accumulator.
This example contains no local buffer, so no workgroup communication or
barrier is needed.

### 6. Make memory operations explicit

After the source's “optimizations are done” marker, the pipeline expands
broadcast forms and inserts `LOAD` wherever ALU or stores consume values from
global, local, or register address spaces. It then devectorizes unsupported
forms, simplifies indices, performs memory coalescing, handles image-specific
forms, runs more symbolic rewrites, and lowers index dtypes.

Several distinct correctness questions live here:

- Does an `INDEX` name the intended buffer and element?
- Is its validity gate correct for padding or bounds?
- Does inserted loading preserve shared graph values rather than duplicate or
  drop them incorrectly?
- Does devectorization preserve lane order and dtype?
- Does coalescing combine only compatible adjacent accesses?
- Is the selected index dtype wide enough for proven bounds?

The final printed address is the end of this chain. If it is wrong, compare the
first representation before and after the pass that changed it. Starting with
rendered source loses the provenance of the expression.

### 7. Decompose according to renderer support

The renderer's `code_for_op` keys help determine which operations are directly
supported. Early simplifying patterns, dtype decomposition, late patterns,
transcendental patterns, and renderer-specific extra matchers replace or
normalize forms until the target contract can be met.

An algebraically plausible replacement is not automatically correct for
finite dtypes. Depending on the operation, tests may need to cover rounding,
overflow, integer signedness, division/modulo semantics, signed zero,
infinities, NaNs, validity gates, and vector lanes. Test only properties that
the original operation and project actually promise; do not invent stronger
cross-target guarantees.

The Python renderer has its own `code_for_op` execution functions. A successful
`PYTHON::sm_89` run therefore proves those Python-executed lowered UOps for the
example, not that an actual CUDA renderer supports the identical operation set
or emits the same instruction.

### 8. Finalize effects, barriers, and control flow

Final rewrite rules commit weak values to concrete forms, apply renderer
extras, split ends, and remove remaining invalid forms where rules exist. The
pipeline then inserts implicit barriers for identified local-memory hazards,
adds explicit control-flow structure, and numbers previously unnumbered scalar
parameters. With `SPEC` enabled, the result is checked against program IR.

A **barrier** is a synchronization point. For a workgroup-scoped barrier, each
participating work item must reach it before any proceeds past it. Consider:

```text
work item 0 stores local[0]
work item 1 later loads local[0]
```

Program order within item zero alone does not order item one's load. A barrier
between the cooperative write phase and read phase can establish the required
visibility/order at the supported scope. Barriers are not a generic cure: the
correct scope, participation, control flow, and memory dependency all matter.

The carried kernel has no local buffer and no cross-work-item read, so a barrier
would add no needed semantics. Its accumulator effects still require ordinary
dependency/control ordering within each work item.

### 9. Derive metadata, linearize, render, and compile

After `full_rewrite_to_sink`, `ProgramInfo.from_sink` derives global/local
dimensions, parameter slots, input/output roles, and symbolic variables. ISA
renderers can then perform instruction selection. `do_linearize` topologically
orders the graph with control dependencies.

Only `ISARenderer` paths run the pinned linear-scan register-allocation branch.
Other renderers produce source from ordered UOps and pass it to a compiler;
Python serializes the UOps for its interpreter. The resulting `PROGRAM`
progressively gains `SINK`, `LINEAR`, `SOURCE`, and `BINARY` children.

Those source, binary, ABI, and physical-register details are Chapter 11's
subject. Chapter 10 stops when the reader can justify the lowered program and
its order.

## Evidence boundaries: say only what the experiment shows

“It ran” is incomplete unless the route and claim are named. Use this table
when recording chapter artifacts:

| Evidence | What it supports | What it does **not** support |
| --- | --- | --- |
| Hand arithmetic and loop/address table | Intended mathematical result and contiguous row-major element mapping | Any claim about tinygrad's current implementation |
| Scheduled/lowered graph under `PYTHON::sm_89` | Pinned CUDA-oriented target shape, marker transition, address substitutions, and PythonRenderer program legality | Physical GPU execution, driver behavior, concurrency, native CUDA rendering, or performance |
| Result from `PythonProgram` | Numeric semantics of these lowered UOps in the Python interpreter | Race detection, native floating-point instruction choice, launch timing, or hardware resource use |
| `DEV=PYTHON` | Hardware-neutral Python backend control | CUDA-oriented dimension or architecture behavior |
| `DEV=CPU` | Native host rendering, Clang compilation, and host execution when the toolchain is present | Portability to systems without that toolchain or any GPU claim |
| `DEV=CUDA` or `NVK+NV` on a reported device | That selected real-device path compiled and executed the tested input | Exhaustive correctness, absence of races, other shapes/dtypes, stable performance, or another driver/device |
| Generated source or ISA inspection | What the selected renderer/compiler artifact contains | Dynamic execution behavior or performance by itself |
| Timed/profiler run | Behavior for the controlled workload and measurement conditions | A general performance improvement without representative comparisons |

The existing provenance records a manual RTX 4090 smoke test of the bundled
runner. It does not validate a new lab absent from that run. Record a fresh run
before claiming hardware validation of this probe.

### Optional native reruns

After the portable controlled observation succeeds, you may rerun the probe by
changing only `DEV`, while retaining `DEBUG=0`, `NOOPT=1`, and `SPEC=2`:

```bash
DEV=CPU DEBUG=0 NOOPT=1 SPEC=2 \
  CACHEDB=/tmp/tinygrad-guide-lowering-cpu.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/lowering_walk.py"
```

or, on the intended configured hardware:

```bash
DEV=CUDA DEBUG=0 NOOPT=1 SPEC=2 \
  CACHEDB=/tmp/tinygrad-guide-lowering-cuda.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/lowering_walk.py"
```

Detailed structural assertions activate only for the controlled target. Other
renderers report their marker counts, launch metadata, and numeric result
without requiring the same lower-level shape. Compare those artifacts; do not
assume a difference makes the renderer wrong. Never leave `DEV` implicit in a
recorded backend experiment.

## Optional VIZ extension: find each transforming pass

The direct lab supplies a small before/after comparison. VIZ is useful once you
understand what changed and want to find the first named pass responsible.
Capture the same controlled run without launching the interactive server:

```bash
VIZ=-1 NO_COLOR=1 DEV=PYTHON::sm_89 DEBUG=0 NOOPT=1 SPEC=2 \
  CACHEDB=/tmp/tinygrad-guide-lowering-viz.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/lowering_walk.py"

cp "/tmp/rewrites.pkl.$USER" /tmp/tinygrad-guide-lowering-rewrites.pkl
```

Why `VIZ=-1`? Any nonzero value records rewrites, while a positive value on an
interactive terminal replaces the lab process with the VIZ server at exit.
Negative one records and returns to the shell, which makes the following CLI
steps reproducible. The capture's default path is
`/tmp/rewrites.pkl.$USER`; it is independent of `CACHEDB`, which isolates only
the compiler disk cache. Copying it gives this exercise a stable input before
another VIZ run overwrites the default.

First list captured events as documented by the pinned
[VIZ README](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md#L47-L76):

```bash
REWRITES=/tmp/tinygrad-guide-lowering-rewrites.pkl
NO_COLOR=1 DEBUG=0 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$REWRITES" -s TINY
```

Copy the exact `do_to_program for ...` event name from that output; generated
suffixes can vary. The example below shows the pinned name, but replace it with
what your first command printed. A positional event name is required—bare
`-s TINY --ls` does not select one event.

```bash
EVENT='do_to_program for r_2_3'

NO_COLOR=1 DEBUG=0 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$REWRITES" -s TINY "$EVENT" --ls

NO_COLOR=1 DEBUG=6 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$REWRITES" -s TINY "$EVENT"
```

The `--ls` command lists pass names. The `DEBUG=6` command prints each saved
input graph. The state *after* pass N is normally visible as the input to pass
N+1, so compare adjacent entries when filling the table below. The selected
pinned trace includes pass labels corresponding to:

```text
View Base AST
early movement ops
load collapse
split ranges
initial symbolic
simplify ranges
postopt symbolic
expander
remove reduces
add local buffers
add gpudims
expand broadcast / add loads
devectorize2
memory coalescing and image/index cleanup
decompositions
final rewrite
add implicit barriers
add control flow
number params
View Output AST
linearize/render
```

VIZ can also record nested tracked rewrites, so do not demand a one-to-one row
for every displayed name. Build a table from actual saved states:

| Observation | First pass after which it is true | Evidence to record |
| --- | --- | --- |
| `REDUCE` is absent |  | before/after counts and accumulator nodes |
| Register accumulator exists |  | address space, dtype, size |
| Output range is `SPECIAL` |  | old range role and new special name/extent |
| Explicit input `LOAD` exists |  | indexed buffer and simplified address |
| Program control exists |  | `RANGE`/`END` or other control nodes |

Fill the blank pass names from the trace. The answer for the pinned run should
agree with the pipeline source, but the exercise is to observe the transition,
not copy the headings above.

## Debug a lowering failure by bisecting representations

Suppose a kernel produces the wrong second row. Avoid reading all of
`codegen/` or staring only at generated source. Preserve one small reproducer
and compare artifacts in order:

1. **Semantic oracle:** Does the hand calculation distinguish every relevant
   coordinate and edge case?
2. **Execution plan:** Is the right work fused into the right number/order of
   calls?
3. **Scheduled kernel:** Are output shape, ranges, reduction identity, covered
   columns, buffers, gates, and `row*3 + column` correct?
4. **Applied kernel opts:** Did an unexpected split, unroll, local axis, or
   grouped reduction first change the organization incorrectly?
5. **Post-reduction graph:** Does initialization use zero? Does each update use
   the correct value and range? Is final state read after the loop?
6. **Post-dimension graph:** Does the special ID replace the intended range and
   have the right extent?
7. **Post-memory graph:** Does the input `LOAD` use the correct buffer, index,
   gate, and dtype?
8. **Post-decomposition graph:** Is the first changed operation semantically
   valid for difficult dtype values?
9. **Ordered `LINEAR`:** Are effects, loop boundaries, barriers, and parameters
   ordered correctly?
10. **Source/binary/runtime:** If all earlier artifacts are correct, continue
    through Chapter 11 and the relevant backend/runtime chapter.

The **first divergence** is more informative than the final symptom. Minimize
the graph while preserving that divergence, identify the named rewrite, find
nearby tests, and add a focused regression before changing the rule.

Choose an oracle appropriate to the claim:

- exact values for integer/index transformations;
- dtype-aware numerical comparisons for floating-point operations;
- spec checks for illegal IR;
- differential execution on another suitable backend;
- generated-program comparison when code shape is the claim;
- process replay for wider compiler-impact checks; and
- real hardware for concurrency, driver, or performance behavior.

## Guided source tour: one question per stop

The explanation and lab are the primary lesson. These source stops confirm it.
At each stop, answer the stated question and ignore unrelated machinery; do not
read the whole directory.

### Stop 1 — Which call body does `compile_linear` change?

Read [`realize.py` lines 247–273](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L247-L273).

Look for the pattern that recognizes a `CALL` whose body is a `SINK` or an
existing `PROGRAM`, passes it through `to_program`, and returns an execution
plan with compiled bodies. The answer is: outer execution order remains a
`LINEAR` plan while eligible kernel bodies progress to `PROGRAM`.

Ignore validation, BEAM plumbing, HCQ lowering, and local-size optimization
details on this pass. They matter later but are not needed to understand the
boundary.

### Stop 2 — Why does `NOOPT=1` not prevent `gidx0`?

Read [`postrange.py` lines 339–356](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L339-L356).

Follow the order inside `apply_opts`: construct `Scheduler`, call
`convert_loop_to_global`, then choose explicit opts, BEAM, or the hand-coded
heuristic. `NOOPT` guards the last path; it is not an early return from all
lowering.

Ignore the internals of BEAM search and every `OptOps` handler. Chapter 9 is
their map.

### Stop 3 — What is the top-level rewrite order?

Read [`codegen/__init__.py` lines 284–325](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284-L325).

Match named `graph_rewrite` calls to the lab: preprocessing, range work,
`apply_opts`, expansion, reduction removal, local buffers, and GPU dimensions.
The answer is an ordered pass map, not the implementation of every matcher.

Ignore pattern definitions on the first read. Write down which two adjacent
saved states would bracket a suspected failure.

### Stop 4 — Where does `REDUCE` become accumulator state?

Read [`codegen/__init__.py` lines 210–241](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L210-L241).

In `reduce_ranges_to_acc`, identify the register placeholder, identity store,
update, `end`, and final value. Then identify `maybe_load` and the matcher that
inserts loads for address-space values.

Ignore grouped-reduction cleanup and horizontal-reduction variants until you
can narrate the simple accumulator path.

### Stop 5 — Where does a range become `SPECIAL`?

Read [`gpudims.py` lines 27–88](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/gpudims.py#L27-L88).

Look for range collection by axis type, renderer dimension checks, creation of
special indices, and substitution. Connect the output range of extent two to
`gidx0`. Then read the pinned CUDA renderer's
[`code_for_workitem`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L408-L415)
and confirm that `gidx0` maps to `blockIdx.x` while `lidx0` maps to
`threadIdx.x`.

Ignore multi-dimensional grouping edge cases on the first pass. The key model
is “selected range coordinate becomes execution-model coordinate.”

### Stop 6 — Where are memory and index forms finalized?

Read [`codegen/__init__.py` lines 327–349](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L327-L349).

Find broadcast expansion/load insertion, devectorization/index simplification,
memory coalescing, image cleanup, and index-dtype lowering. The answer is that
the final address is produced by several ordered transformations, not one
“address lowering” function.

Ignore image-specific details unless your reproducer uses an image dtype.

### Stop 7 — How does a renderer affect decomposition?

Read the capability fields in [`renderer/__init__.py` lines 59–84](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L59-L84), then the decomposition selection in
[`codegen/__init__.py` lines 351–371](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L351-L371).

Connect `code_for_op` and renderer extras to the rewrite pattern set. The
answer is: lowering targets a capability contract before text rendering, so
the same scheduled graph can lower differently for different renderers.

Ignore the implementation of every transcendental approximation until a task
actually concerns one.

### Stop 8 — When are barriers and control dependencies added?

Read [`codegen/__init__.py` lines 258–282](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L258-L282) and [`linearizer.py` lines 53–85](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/linearizer.py#L53-L85).

Identify local-memory read-after-write/write-after-read hazards and the extra
dependency edges used during ordering. The answer is not “every store gets a
barrier.” Barriers are conditional on address space, dependency, loop, and
scope. Confirm that the carried graph has none.

Ignore target-specific barrier syntax; rendering comes next chapter.

### Stop 9 — What differs between tensor and program legality?

Read [`spec.py` lines 135–200](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L135-L200) and [`spec.py` lines 202–225](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L202-L225).

Find examples of tensor-only movement/`REDUCE` forms and program restrictions
on weak dtypes, movement, invalid values, and `SPECIAL`. The answer is that
legality changes with stage; using the same UOp class does not imply one
unchanging dialect.

Ignore the remainder of the full shared spec unless a failing node directs you
there.

### Stop 10 — How does the graph become an ordered list?

Read [`linearizer.py` lines 8–51](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/linearizer.py#L8-L51) and [`codegen/__init__.py` lines 409–421](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L409-L421).

Look for priority-aware topological ordering and the conditional
`ISARenderer` register-allocation branch. The answer is: all renderers receive
an ordered `LINEAR`, but native ISA register allocation is not a universal
pre-render pass.

Ignore register-allocation heuristics until Chapter 11 or an ISA-specific task.

### Stop 11 — Where do launch and buffer roles come from?

Read [`uop/ops.py` lines 1214–1260](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1214-L1260).

In `ProgramInfo.from_sink`, find global/local sizes from `SPECIAL`, global
buffer slots from `PARAM`, and input/output roles from `LOAD`/`STORE` indices.
This connects lowered graph facts to the lab's launch tuple.

Ignore ELF signatures and runtime launch APIs. This stop answers metadata
derivation only.

## Worked exercises

Attempt each question before opening its answer. Use paper first; only the
shape and optimizer extensions in Exercises 5 and 8 need a fresh tinygrad
process.

### Exercise 1 — Complete the address table

For contiguous shape `(2, 3)`, compute `row*3 + column` for every coordinate.

??? note "Answer"

    ```text
    (0,0)->0  (0,1)->1  (0,2)->2
    (1,0)->3  (1,1)->4  (1,2)->5
    ```

    The row stride is three elements. For `float32`, byte offsets are those
    values multiplied by four.

### Exercise 2 — Explain the range transition

Why do two scheduled `RANGE`s become one lowered `RANGE` and one `SPECIAL`?

??? note "Answer"

    The row iterations produce independent output elements, so the controlled
    kernel schedule maps that range to grid/workgroup coordinates. `gidx0`
    supplies the row coordinate; because local extent is one, each group has
    one work item. The column iterations combine into that work item's
    accumulator, so they remain a serial `AxisType.REDUCE` range. Other legal
    schedules could parallelize the reduction using additional machinery.

### Exercise 3 — Where did the reduction go?

The lowered graph contains zero `Ops.REDUCE` nodes. Why does it still calculate
a reduction?

??? note "Answer"

    `Ops.REDUCE` was an abstract value operation. Lowering replaced it with a
    zero-initialized register accumulator, a loop over three columns, an add
    update in that loop, and a final accumulator read after the loop. The
    semantics moved into several more operational nodes.

### Exercise 4 — Interpret `LOAD=3`

Does the lowered marker count prove that the program performs three global
memory reads?

??? note "Answer"

    No. It counts static UOp nodes, not dynamic executions or hardware memory
    transactions. One `LOAD` reads the global input element and two read the
    register accumulator at different program points. The one static global
    input-load node executes for six `(row, column)` combinations.

### Exercise 5 — Change the shape to `(3, 2)`

Reshape values `1` through `6` into three rows of two columns and evaluate
`(x*x + 2*x).sum(axis=1)`. Derive the address formula and predicted axis
extents under the same controlled strategy.

??? note "Answer"

    The address is `row*2 + column`. The elementwise values remain
    `3, 8, 15, 24, 35, 48`, grouped as pairs. The result is
    `[11, 39, 83]`. The global row extent is three and the reduction extent is
    two. Run a fresh process or construct a fresh graph because planning
    mutates in-scope Tensor wrappers to their planned buffer forms.

### Exercise 6 — Reduce to a scalar

What happens conceptually if the same six elementwise values are all summed,
with no row output axis?

??? note "Answer"

    The result is `3+8+15+24+35+48 = 133`. There is no two-element output row
    axis to map to `gidx0`; one workgroup containing one work item can loop over
    a reduction range covering six elements in the simplest controlled form.
    A default optimizer may organize that reduction differently.

### Exercise 7 — Remove the reduction

Suppose the output is only `x*x + 2*x`. Predict the principal marker changes.

??? note "Answer"

    There is no abstract `REDUCE`, reduction range, identity, or accumulator.
    The output has six independent positions. A simple schedule can map those
    positions to a global coordinate and load each corresponding input before
    storing its elementwise result. Exact dimensional factoring is
    target/schedule dependent.

### Exercise 8 — Turn default optimization back on

Rerun a fresh equivalent probe without `NOOPT=1`. Why is it legitimate for the
range counts and launch tuple to change while the result stays fixed?

??? note "Answer"

    Kernel optimization changes organization, not intended tensor semantics.
    At the pin, the heuristic unrolled this reduction and chose a local output
    factor of two, leaving no ordinary `RANGE` and using local extent two.
    That is one heuristic result for this snapshot/target, not a required form.
    Compare the lab's printed `applied opts` and pass states before interpreting
    the later graph. At the pin this run prints `UNROLL` followed by `LOCAL`
    with factor two.

### Exercise 9 — Classify a failure

Match each symptom to the first likely artifact:

1. two kernels appear instead of one;
2. `row=1, column=0` simplifies to element index two;
3. accumulator starts from one for an add reduction;
4. lowered graph is correct but rendered source has malformed syntax;
5. source and launch are correct, but a real local-memory kernel fails only
   under concurrent GPU execution.

??? note "Answer"

    1. execution scheduling/kernel boundary;
    2. scheduled or lowered indexing—compare the first address divergence;
    3. reduction lowering/identity initialization;
    4. Chapter 11 renderer;
    5. runtime/hardware synchronization evidence, after confirming barrier and
       dependency lowering.

### Exercise 10 — State the emulator claim precisely

Complete the sentence: “The `PYTHON::sm_89` run proves ___, but not ___.”

??? note "Answer"

    One defensible answer is: “It proves that the pinned pipeline produced the
    recorded CUDA-oriented Python target shape and that `PythonProgram`
    evaluated these lowered UOps to `[26, 107]`; it does not prove CUDA driver
    execution, physical parallelism, native CUDA rendering, race freedom,
    timing, or RTX 4090 resource use.”

### Exercise 11 — Decide whether a barrier is required

Case A: each work item updates only its private register accumulator.
Case B: several local work items store partial sums in workgroup-local storage,
then each may read a neighbor's partial sum. Which needs a workgroup barrier
between the phases?

??? note "Answer"

    Case A has no cross-work-item shared-memory communication and needs no
    workgroup barrier for its accumulator. Case B needs synchronization if
    later reads must observe all earlier local stores. Every participating
    work item must follow valid control flow to the barrier, and the barrier
    scope must cover that local memory. The precise target memory model still
    governs the native program.

### Exercise 12 — Choose a transformation category

Classify these by their primary purpose:

- simplify `row*3 + 0`;
- unroll the three-column loop;
- replace `REDUCE` with accumulator state;
- replace an unsupported target operation with supported ALU primitives.

??? note "Answer"

    Respectively: canonicalization, optimization, lowering, and decomposition.
    Real pipelines combine these and one rewrite may enable another, so the
    labels describe the primary contract change rather than disjoint code
    directories.

## Checkpoint

Keep the following artifacts in your study notes:

1. the six-row value and address tables;
2. the explicit nested loop with `row*3 + column` and accumulator state;
3. the exact scheduled/lowered marker output;
4. an annotation identifying `gidx0`, the remaining reduction range, input
   index, register accumulator, explicit loads, output store, and loop `END`;
5. one evidence statement that includes both supported claims and explicit
   nonclaims; and
6. one sentence answering each guided source-stop question.

You are ready to continue when you can answer these without treating a UOp
name as magic:

- What did execution scheduling decide, and what did kernel scheduling decide?
- Why is the scheduled input address `row*3 + column`?
- Why does scheduled IR have no explicit `LOAD`, while lowered IR does?
- Why can `Ops.REDUCE` disappear while `AxisType.REDUCE` remains?
- What initialization, update, and ordering replace the reduction?
- Why is a one-element `REG` buffer not a global Tensor allocation or measured
  physical register count?
- Why does the output range become `gidx0`, what does extent two mean, and why
  is it a workgroup coordinate rather than a complete work-item ID?
- Why are three static loads not three global reads?
- What did `NOOPT=1` suppress, and what lowering still occurred?
- Which renderer capabilities can change decomposition?
- When is a local-memory barrier needed, and why is there none here?
- What does the program spec establish, and what semantic property does it not?
- Which artifact would you inspect first for a wrong address, wrong identity,
  unsupported operation, invalid control order, or runtime-only failure?

If an answer is unclear, return to the corresponding lab-output subsection,
then use only its question-led source stop.

## Optional reinforcement and bounded detours

External reading is optional; this chapter contains the required first model.
Use the [Learning resources](../reference/learning-resources.md) page as a
router for a specific remaining gap:

- Use [Next: transformations and lowering](../reference/learning-resources.md#next-transformations-and-lowering)
  if you cannot yet explain multiple abstraction levels, canonicalization, or
  legality. Stop when those four ideas make sense; do not learn MLIR APIs.
- Use [For-loop and tensor-scheduling intuition](../reference/learning-resources.md#for-loop-and-tensor-scheduling-intuition)
  if translating a tensor formula into loops, buffers, and indices remains
  difficult. Stop after doing that translation for one elementwise operation
  and one reduction.
- Use [GPU execution on the RTX 4090 path](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
  only if global/local IDs, workgroups, address spaces, or barriers remain
  unclear. Learn the execution and memory model, not the full CUDA API.
- Use the pinned VIZ README only after the deterministic comparison works. VIZ
  is an inspection tool, not a substitute for understanding the loop.

## Deliberately deferred topics

Finishing this chapter does not require mastering every lowering-adjacent
specialty:

- heuristic and BEAM cost decisions remain in Chapter 9 and Chapter 17;
- complicated movement, masking, symbolic bounds, and index algebra remain in
  Chapter 8;
- generated source, compiler behavior, binary containers, instruction
  selection, and ISA register allocation are Chapter 11;
- allocation, submission, launch, asynchrony, and synchronization are Chapter
  12;
- TinyJit capture and replay are Chapter 13;
- CUDA, PTX, NVK, and `NV` backend/driver details are Chapter 14;
- large rewrite traces and cross-stage failure localization continue in
  Chapter 15;
- contribution-grade regression strategy is Chapter 16;
- measurement and performance claims are Chapter 17; and
- translating a finding into a scoped upstream change is Chapter 18.

Grouped reductions, tensor cores, images, multi-device fragments, symbolic
launch sizes, and target transcendental algorithms are all valid contribution
areas. The common skill from this chapter is knowing which additional domain
reference to open and which before/after artifact must remain invariant.

## Quick reference

| Term or symptom | Meaning or first inspection |
| --- | --- |
| Scheduled `SINK` | One selected kernel with ranges, indices, effects, and `KernelInfo`, still legal tensor-oriented IR |
| Lowered `SINK` | Renderer-conditioned graph intended to satisfy program IR |
| `Ops.REDUCE` disappears | Reduction became explicit accumulator state and control |
| `AxisType.REDUCE` remains | The loop still has a reduction role |
| `SPECIAL(gidx0)` | Target-supplied grid/workgroup coordinate; maps to `blockIdx.x` on the pinned CUDA renderer |
| `AddrSpace.REG` buffer | Compiler-managed per-work-item temporary intent, not a global Tensor allocation |
| `LOAD` count | Static IR nodes; classify address spaces and dynamic iteration before making traffic claims |
| Wrong kernel count | Execution scheduling before codegen |
| Wrong loop/layout choice | Applied kernel opts and pre/post-option graph |
| Wrong reduction identity/state | Reduction removal/accumulator rewrite |
| Wrong address or gate | Movement/ranges, load insertion, index simplification, and first diverging pass |
| Unsupported dtype/op reaches renderer | Decomposition and renderer capability contract |
| Shared-local race | Local-buffer dependencies, barrier lowering, then real-device evidence |
| Bad loop/branch order | Control-flow insertion and linearizer dependencies |
| Native source/ISA problem | Chapter 11 after proving lowered IR correct |
| Runtime-only failure | Device/runtime/backend path after proving compiled artifact correct |

Previous: [Kernel optimization](09-kernel-optimization.md)

Next: [Rendering and compilation](11-rendering.md)
