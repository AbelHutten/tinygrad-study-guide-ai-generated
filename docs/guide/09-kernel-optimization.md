# 9. Kernel optimization

## Purpose

Scheduling has chosen which tensor operations belong in one kernel. Kernel
optimization chooses how that kernel's equivalent work is divided into loops,
work items, workgroups, vector-shaped values, reduction participants, and
specialized instructions. The result must compute the same tensor result,
within the numerical contract of the operation, but it can have a very
different launch and very different cost.

This chapter starts from the GPU concepts that the optimization code assumes.
It does not assume that you already know CUDA, compiler scheduling vocabulary,
or tinygrad's option names. The running example is one 16×16 matrix
multiplication carried from Python loops through tinygrad's range model and
three concrete schedules. By the end, you should be able to explain a proposed
kernel change, construct a small reproducer, inspect its structure portably,
and measure it honestly on the RTX 4090.

**Source snapshot:** `874d331` (2026-08-05).

## Route through the chapter

Read the chapter front to back the first time. The sequence is intentional:

1. separate whole-model scheduling from one-kernel optimization;
2. derive the loop and address structure of a small matrix multiplication;
3. learn the device-independent GPU execution and memory model;
4. map that model to CUDA terminology;
5. translate performance ideas into tinygrad range transformations;
6. inspect a deterministic, hardware-free structural lab that executes in
   Python;
7. study grouped reduction and padding as bounded extensions; and
8. design a real measurement and a contribution-shaped experiment.

If a later term becomes fuzzy, return to the execution-model table, the
16×16 range-state table, or the quick reference. Optional links are collected
near the end; no outside reading is required merely to understand the main
example.

This chapter deliberately defers the exact lowering pass sequence, renderer
syntax, runtime queues, and native NVIDIA instruction encoding. Chapters 10–14
cover those layers. Here the goal is to reason correctly at the kernel-schedule
boundary and to know what evidence that boundary can and cannot provide.

## First decide: model problem or kernel problem

The word *schedule* is used for two related but different decisions:

- An **execution schedule** decides which kernels, copies, views, and other
  calls execute, and in what order. Fusion and realization change this plan.
- A **kernel schedule** reorganizes the work inside one already chosen kernel.
  `UPCAST`, `LOCAL`, `UNROLL`, grouped reductions, and tensor-core matching live
  here.

Suppose a model performs an elementwise operation, a matrix multiplication,
and a copy. Eliminating the copy or fusing the elementwise operation changes
the execution plan. Giving each matrix-multiplication work item more output
elements changes only that kernel's schedule. An optimization at one layer
cannot automatically repair a problem at the other.

Start performance work with an end-to-end trace or profile. Select one kernel
only after showing that it matters to the target workload. Amdahl's law gives a
useful sanity check. If a kernel takes 2% of total time and its own time falls
by 20%, normalized total time becomes

```text
0.98 + 0.02 * 0.80 = 0.996
```

That is a 0.4% reduction in total time, before secondary effects. Be careful
with the wording: “20% less time” is not the same as “1.2× faster.” A literal
1.2× speedup would leave `0.98 + 0.02 / 1.2 = 0.99667`, about a 0.333% total
time reduction.

Use different evidence at the two layers:

| Question | Inspect | Measure |
| --- | --- | --- |
| Is the execution plan wrong? | Kernel boundaries, copies, materializations, buffer sizes, JIT/graph path | End-to-end synchronized time and memory |
| Is one kernel organized poorly? | Ranges, applied options, launch dimensions, addresses, generated program | Isolated kernel time and hardware counters |

This chapter assumes that the execution boundary is appropriate and studies
one kernel. If no kernel dominates, return to Chapter 7 or the end-to-end
workflow in Chapter 17 before tuning an arbitrary schedule.

## Running example: matrix multiplication before optimization

Let `A` and `B` be 16×16 row-major float32 matrices, and let `C = A @ B`.
Ignore vector instructions and GPU terms initially. A direct scalar program is:

```python
for i in range(16):
  for j in range(16):
    acc = 0.0
    for k in range(16):
      acc += A[i, k] * B[k, j]
    C[i, j] = acc
```

There are three logical ranges:

- `i` selects an output row;
- `j` selects an output column; and
- `k` is a reduction range because many products combine into one output.

For row-major storage, the last axis is contiguous. With element indices rather
than byte addresses, the accesses are:

```text
A[i, k] -> i*16 + k
B[k, j] -> k*16 + j
C[i, j] -> i*16 + j
```

Holding `i` and `j` fixed while advancing `k` reads adjacent elements of `A`
but jumps by 16 elements through `B`. Holding `i` and `k` fixed while advancing
`j` reuses the same `A[i,k]` and reads adjacent elements of `B`. This second
observation is the reason that computing several adjacent `j` outputs together
can reduce repeated loads.

The program performs `16*16*16 = 4096` multiply-accumulate iterations.
tinygrad's static operation model counts a multiply-accumulate as two
arithmetic operations, so this form has `8192` modeled operations. That number
is a convention of the lowered UOp cost model, not a universal definition of
the algorithm's “useful FLOPs.” We will return to that distinction when tensor
cores and padding change the lowered representation.

The loop order above is only one legal organization. The mathematical result
constrains which products are combined, but it does not require that every
output use one serial worker, that `i` and `j` remain unsplit, or that hardware
execute workgroups in a particular order. A kernel optimizer searches this
space of equivalent organizations.

## A GPU execution model from first principles

### Work items, workgroups, subgroups, and the grid

A GPU kernel describes a large set of invocations of one program. A single
invocation is a **work item**. Work items are collected into **workgroups**, and
all workgroups for one launch form a **grid**.

Workgroups are not guaranteed to execute simultaneously or in index order. A
GPU can keep only some of them resident and schedule later groups as resources
become free. Consequently, ordinary kernels cannot use a workgroup-local
barrier to coordinate two different workgroups. Cross-workgroup communication
usually requires a later kernel launch or a specialized global mechanism.

Within a workgroup, hardware executes fixed-width collections of work items in
lockstep or near-lockstep. The device-independent name is a **subgroup**; an
individual position in it is a **lane**. Branches can cause lanes to take
different paths, in which case the hardware must mask or serialize paths. That
is why a branch that looks cheap in scalar Python can be expensive when nearby
lanes disagree.

This hierarchy assigns different kinds of coordinates:

```text
grid
└── workgroup coordinate
    └── hardware subgroup
        └── work item (its position in the subgroup is its lane)
```

The grid and workgroup coordinates are explicit launch coordinates. Subgroups
are the hardware's grouping of work items inside a workgroup; “lane” names a
work item's position in that subgroup, not another child below the work item.

A launch can have up to three explicit dimensions at each of the grid and
workgroup levels. A compiler maps an arbitrary number of logical tensor ranges
into these dimensions or keeps them as loops inside each invocation.

### CUDA names for the same hierarchy

On the RTX 4090's CUDA path, the common names are:

| Device-independent term | CUDA term | Typical CUDA coordinate |
| --- | --- | --- |
| Work item | Thread | `threadIdx` within a block |
| Workgroup | Thread block | `blockIdx` within a grid |
| Subgroup | Warp | lane within a 32-thread warp |
| Grid | Grid | block count in up to three dimensions |

Do not generalize “warp” to every target; subgroup widths and rules differ.
Likewise, do not assume that a tinygrad field named `THREAD` means a CUDA
thread. It does not. tinygrad's CUDA work-item coordinates are represented by
`LOCAL` ranges. The `THREAD` option is a separate target facility discussed
later.

### Four storage scopes, plus caches

The execution hierarchy matters because storage has visibility and lifetime:

| Storage idea | Visible to | Typical use | Main limit/cost |
| --- | --- | --- | --- |
| Device/global memory | All workgroups and later kernels | Tensor buffers | Large but high latency and finite bandwidth |
| Workgroup-shared memory | Work items in one workgroup | Cooperative tiles and partial reductions | Small, explicitly managed; cross-item communication needs synchronization |
| Per-work-item values | One work item | Accumulators, indices, temporaries | Usually registers; too many reduce residency or spill |
| Constant/immediate state | Kernel or instruction | Sizes and fixed coefficients | Target-specific limits and encoding |

Hardware caches sit between instructions and device memory. They can make a
repeated global load cheaper without changing the program's address space, but
their contents are not a portable correctness mechanism. A compiler should not
remove a required barrier merely because a cache is likely to contain a value.

Three uses of the word *local* are especially easy to confuse:

- tinygrad `OptOps.LOCAL` and `AxisType.LOCAL` create a **workgroup-local launch
  coordinate**. On CUDA, this is a `threadIdx` dimension. Applying `LOCAL` alone
  does not allocate shared memory.
- tinygrad `AddrSpace.LOCAL` identifies a buffer shared within a workgroup. On
  CUDA this is commonly rendered as `__shared__` storage.
- CUDA documentation's **local memory** means per-thread storage that is backed
  by device memory, commonly because values spilled from registers. It is not
  shared memory and is not tinygrad's `AddrSpace.LOCAL`.

Whenever you write “local,” state which of these you mean.

### Synchronization and reduction

Imagine eight work items each computing part of a sum. If they write partial
sums into workgroup-shared storage, no participant may read another
participant's entry until all required writes are visible. A workgroup barrier
provides that rendezvous. It does not synchronize a different workgroup.

A correct cooperative reduction therefore needs more than parallel launch
coordinates: it needs an allocation with the correct scope, non-conflicting
addresses, and barriers in the right places. tinygrad's `GROUP` and `GROUPTOP`
options express a grouped reduction; later lowering makes local buffers and
barriers explicit. `LOCAL`, by itself, expresses only who performs ordinary
output work.

### Latency, throughput, and occupancy

**Latency** is the time from issuing one operation to receiving its result.
**Throughput** is the amount of work a device completes per unit time once many
operations overlap. GPUs tolerate long memory latency by keeping other warps
ready to execute. A kernel with many independent workgroups can hide a stalled
warp behind another; a kernel with too little parallel work cannot.

**Occupancy** is the fraction of the hardware's possible resident warps that
are actually resident. Registers per work item, shared bytes per workgroup,
workgroup size, and architectural limits all constrain it. Higher occupancy can
improve latency hiding, but maximum occupancy is not the goal. A tile using more
registers may reduce occupancy yet win by reusing data and issuing far fewer
loads. Treat occupancy as a constraint to investigate, not a score to maximize.

The same caution applies to parallelism. Moving four outputs into each work
item reduces the number of work items by four. That may improve reuse and still
lose if too few warps remain. Performance is a balance among per-item work,
memory behavior, instruction efficiency, and enough independent work.

## Memory behavior: coalescing, reuse, and tiling

### Coalescing is about transactions, not visual adjacency

When lanes in a warp issue a global-memory instruction, hardware combines their
addresses into memory transactions. Access is well coalesced when active lanes
touch few appropriately aligned memory segments. “Neighboring lanes read
neighboring elements” is a useful first approximation, but it is incomplete:
alignment, element size, active-lane masks, and segment boundaries also matter.

For the row-major output `C[i,j]`, mapping adjacent lanes to adjacent `j` values
usually gives contiguous stores. For `B[k,j]`, the same mapping gives contiguous
loads at fixed `k`. Mapping adjacent lanes to `i` instead would stride through
`B` by a full row. This does not prove one schedule is faster—caches, reuse, and
other instructions still matter—but it gives a precise address hypothesis to
test with generated code and transaction counters.

tinygrad has no `OptOps.COALESCE`. Options change range and address structure;
a later `memory_coalescing` pass combines compatible adjacent operations after
expansion and devectorization. A failed vector combine can originate either in
the schedule's axis-to-address mapping or in the late pass's dtype, alignment,
or validity preconditions.

### Reuse and per-work-item tiles

Return to fixed `i` and `k`. One `A[i,k]` contributes to all 16 columns. If a
work item computes four adjacent `j` outputs, it can load `A[i,k]` once and use
it in four accumulators. It still needs four different `B[k,j]` values. This is
a small per-work-item output tile.

Larger tiles can increase reuse of both inputs, but each output normally needs
an accumulator. More accumulators extend live ranges and consume registers.
At some point register pressure lowers occupancy or causes spills into CUDA
local memory. The compiler's static load count cannot tell you where that point
is; compiler resource reports or profiler counters can.

### Cooperative tiles

A workgroup can also load a tile into workgroup-shared memory so several work
items reuse it. That adds explicit stores, loads, and barriers. It helps only if
the avoided global traffic or improved access pattern outweighs those costs.
It also consumes shared capacity and can reduce resident workgroups.

Do not describe every grouped reduction as “caching global loads in shared
memory.” In this snapshot, `GROUP`/`GROUPTOP` specifically parallelize reduction
partials and use a local buffer to combine them. The modeled local-buffer
traffic can increase even when unique global buffer coverage is unchanged.
Generic input tiling is a broader transformation than applying `GROUP`.

## Arithmetic intensity and bounds

Arithmetic intensity relates work to data movement:

```text
arithmetic intensity = arithmetic operations / bytes moved

attainable performance <= min(peak compute throughput,
                              memory bandwidth * arithmetic intensity)
```

A low-intensity kernel is more likely to be limited by memory bandwidth; a
high-intensity kernel is more likely to reach a compute or instruction limit.
This roofline model is a bound, not a benchmark. Which bytes count depends on
the memory level being modeled, and real kernels also pay launch, dependency,
cache, and instruction costs.

tinygrad's `Estimates.from_uops` reports three useful structural quantities:

- `ops` counts operations in the lowered UOp form. A scalar multiply-accumulate
  counts as two. A `WMMA` uses a tensor-core-specific formula. Padding or a
  representation change can change `ops` even when the unpadded mathematical
  result has the same number of useful products.
- `lds` counts modeled bytes for non-register loads and stores, including
  repeated accesses and workgroup-local/shared-buffer traffic. It is not a
  hardware L2 or DRAM transaction count.
- `mem` caps coverage per buffer and access kind in the estimator's model. It is
  neither actual DRAM traffic nor a strict count of unique allocated bytes.

Use estimates to explain structural changes and catch surprising ones. Do not
turn them into hardware claims. A lower `lds` estimate with unchanged elapsed
time might mean that the removed accesses hit cache, a different bottleneck now
dominates, occupancy fell, instructions increased, or timing noise masks the
effect.

## From logical loops to tinygrad ranges

After rangeification, tinygrad represents iteration using explicit `RANGE`
UOps inside a kernel `SINK`. The codegen optimization `Scheduler` holds that AST
and a selected `Renderer`. It classifies ranges with `AxisType` roles,
including:

- `GLOBAL`: workgroup/grid-oriented output work;
- `LOCAL`: work-item coordinates inside a workgroup;
- `REDUCE`: ordinary serial reduction work;
- `GROUP_REDUCE`: reduction work divided among work items;
- `UPCAST`: multiple values performed together by one work item;
- `UNROLL`: reduction iterations expanded into per-item work;
- `WARP`: tensor-core/subgroup organization;
- `THREAD`: target-managed host-thread work on supporting CPU-style renderers;
- `WEAK`: a loop whose eventual hardware role is not yet fixed.

These roles are compiler concepts. They describe how a range should be lowered;
they are not Tensor dimensions and not all of them become visible source loops.

### One factorization operation

Most options factor a range. Suppose an original coordinate has size 12 and we
split by 3. The remaining coordinate has size 4 and the new coordinate has size
3. A bottom split reconstructs the original as:

```text
original = remaining * 3 + new
```

Enumerating `(remaining,new)` in row-major nested-loop order gives original
coordinates `0,1,2,3,...,11`. A top split reconstructs it as:

```text
original = new * 4 + remaining
```

The coordinate set is still `0...11`, but ownership differs. Hold `new` fixed,
as if it identifies one reduction participant. In the bottom form, `new=1`
owns `1,4,7,10`: one residue class with stride 3. In the top form, `new=1`
owns `4,5,6,7`: one contiguous block of four. `GROUPTOP` and `THREAD` request
the top form; the core `LOCAL`, `UPCAST`, `UNROLL`, and `GROUP` moves use the
bottom form in `shift_to`. The rewritten index expression, not the option's
English name, is the source of truth.

The amount must divide the range unless an earlier `PADTO` made it divisible.
An argument of zero for a factor-moving option means the full eligible range.

### Axis numbers are current and option-specific

An `Opt.axis` is not necessarily a Tensor axis. Earlier options can split,
remove, or reorder ranges. Moreover, the namespace depends on the option:

- most options index the scheduler's current `rngs` list;
- `UNROLL` indexes the current list of `REDUCE` and `GROUP_REDUCE` ranges;
- `GROUP` and `GROUPTOP` index the current `REDUCE` list; and
- `TC` indexes the Cartesian product of ranges exclusive to one multiply
  input, ranges exclusive to the other input, and reduction ranges. These are
  candidate N/M/K range triples, not Tensor axes or `rngs` indices.

For this reason, “apply `UNROLL` to axis 0” can mean the fifth range in the
current full shape if that is the first remaining reduction. Print
`shape_str()`, `full_shape`, and `axis_types` after every exploratory option.
Copy the `Scheduler` before trying alternatives because `apply_opt` mutates it.

## Four core options, derived on the 16×16 example

Start with three ranges:

```text
g0=16 GLOBAL    g1=16 GLOBAL    R0=16 REDUCE
```

The names are debug labels, not stable identifiers. Here `g0` corresponds to
`i`, `g1` to `j`, and `R0` to `k`.

### `UPCAST`: several values per work item

Apply `Opt(OptOps.UPCAST, axis=1, arg=4)`. The current `g1` range is factored
into a remaining global range of 4 and an upcast range of 4:

```text
g0=16 GLOBAL    g1=4 GLOBAL    u0=4 UPCAST    R0=16 REDUCE
```

One logical work item now owns four adjacent `j` outputs. In this example that
allows one `A[i,k]` value to feed four accumulators. `UPCAST` expresses
per-work-item multi-value work; it does not promise a machine vector
instruction. Later lowering and rendering decide whether values remain scalar,
become vector-shaped, or feed another specialized form.

The opportunity is reuse, vector-friendly structure, and fewer loop/control
operations. The costs are more live values, more generated code, and four times
fewer output work items. In this snapshot a non-DSP `UPCAST` amount is limited
to at most 16 by `apply_opt`.

### `LOCAL`: move output work into the workgroup

Next apply `Opt(OptOps.LOCAL, axis=0, arg=4)` to the current first global range:

```text
g0=4 GLOBAL    g1=4 GLOBAL    l0=4 LOCAL
u0=4 UPCAST    R0=16 REDUCE
```

The `i` coordinate is now reconstructed from a grid/workgroup coordinate and a
four-way work-item coordinate. On CUDA this produces a block with a four-thread
dimension. It does **not** create an `AddrSpace.LOCAL` shared buffer and does
not, by itself, reduce the modeled input loads. Its opportunity is cooperative
parallel launch geometry and a different mapping of indices to lanes. Its
risks include an invalid or inefficient workgroup shape and reduced residency.

### `UNROLL`: expand part of a reduction

Finally apply `Opt(OptOps.UNROLL, axis=0, arg=4)`. Axis 0 here means the first
entry of `unrollable_dims`, not full-shape axis 0:

```text
g0=4 GLOBAL    g1=4 GLOBAL    l0=4 LOCAL    u0=4 UPCAST
R0=4 REDUCE    r0=4 UNROLL
```

Each work item still covers all 16 `k` values, reconstructed from a four-iteration
reduction loop and a four-way unrolled range. Unrolling can remove loop-control
work and expose independent instructions. It can also enlarge code, lengthen
live ranges, consume registers, and pressure instruction cache. The snapshot
rejects unroll amounts above 32.

### The exact state transition

The deterministic lab prints this table in list form:

| State | Current axes `(label, size, type)` |
| --- | --- |
| Start | `(g0,16,GLOBAL) (g1,16,GLOBAL) (R0,16,REDUCE)` |
| After `UPCAST` | `(g0,16,GLOBAL) (g1,4,GLOBAL) (u0,4,UPCAST) (R0,16,REDUCE)` |
| After `LOCAL` | `(g0,4,GLOBAL) (g1,4,GLOBAL) (l0,4,LOCAL) (u0,4,UPCAST) (R0,16,REDUCE)` |
| After `UNROLL` | `(g0,4,GLOBAL) (g1,4,GLOBAL) (l0,4,LOCAL) (u0,4,UPCAST) (R0,4,REDUCE) (r0,4,UNROLL)` |

The resulting launch has grid/global size `(4,4,1)` and local/workgroup size
`(4,1,1)`. Compare that with the unoptimized `(16,16,1)` grid and
`(1,1,1)` local size. The output coordinate set has not changed; only ownership
and within-item work have.

## Derive the static estimates instead of merely reading them

The unoptimized 16×16 kernel has `ops=8192`. In the estimator's repeated-load
model, each of 256 outputs performs 16 loads from `A` and 16 from `B`, then one
store to `C`:

```text
A loads: 16*16*16 * 4 bytes = 16384
B loads: 16*16*16 * 4 bytes = 16384
C stores:   16*16 * 4 bytes =  1024
lds total:                         33792 bytes
```

The modeled `mem` is `3 * 16 * 16 * 4 = 3072` bytes across the two inputs and
one output. The model's arithmetic intensity using `lds` is
`8192/33792 = 8/33`, approximately `0.242` operations per byte.

After the four-way `j` upcast, each `A[i,k]` is reused across four outputs:

```text
A loads: 16 rows * 4 j-tiles * 16 k * 4 bytes =  4096
B loads: 16*16*16 * 4 bytes                       = 16384
C stores: 16*16 * 4 bytes                         =  1024
lds total:                                          21504 bytes
```

The model intensity is now `8192/21504 = 8/21`, approximately `0.381`.
Applying `LOCAL` and `UNROLL` after that leaves `lds=21504` in this example.
That is important: the lower load estimate came from the upcast's explicit
reuse, not merely from seeing a local launch dimension or an unrolled range.

The default tensor-core schedule later shown by the lab reports `lds=3072`,
equal to the model's buffer coverage, and model intensity `8192/3072 = 8/3`,
approximately `2.667`. This is a structural statement about one lowered
representation. It is not proof of one fetch from DRAM, perfect cache behavior,
or a particular RTX 4090 speed.

## Grouped reductions: `GROUP` and `GROUPTOP`

`GROUP` and `GROUPTOP` factor a `REDUCE` range into a `GROUP_REDUCE` range. Work
items compute partial results, later lowering introduces workgroup-local
storage when required, and a barrier makes the partials safe to combine.
`GROUP` uses the bottom factorization; `GROUPTOP` uses the top factorization, so
they assign original reduction coordinates differently.

For a sum over 64 float32 values grouped by 8, both options in the pinned
snapshot produce:

```text
launch:    global=(1,1,1), local=(8,1,1)
estimates: ops=136, lds=576, mem=260
structure: one AddrSpace.LOCAL buffer of length 8 and one barrier
result:    0+1+...+63 = 2016
```

The ungrouped form has launch `(1,1,1)/(1,1,1)` and estimates
`ops=64, lds=260, mem=260`. Grouping increases the static `lds` here because it
adds local-buffer traffic; it does not reduce `mem` in this model. Its possible
win is parallel reduction and latency hiding, not a universal promise to remove
global input loads.

The factorization determines which `k` values each participant owns. Work it
out from `original = remaining*8 + group` for `GROUP` and
`original = group*8 + remaining` for `GROUPTOP`. Contiguous versus interleaved
ownership can interact with addresses and later reduction structure.

`apply_opt` checks a simplified shared-storage bound for grouped reductions and
rejects choices beyond `renderer.shared_max`. It also rejects unsupported nested
reduction forms. These checks establish selected invariants, not performance.
They do not model registers, spills, achieved occupancy, cache traffic, bank
conflicts, or actual elapsed time.

## Remaining options, one precise meaning at a time

### `PADTO`

`PADTO` rounds a constant range up to a multiple and introduces validity so
padded reads do not affect the result and padded stores do not escape the
original domain. It enables tile sizes that would otherwise fail divisibility.
The extra iterations still consume instructions, and masked memory operations
still have a cost. The snapshot rejects padding that expands a range beyond its
bounded policy, and it cannot pad every axis type.

### `SWAP`

`SWAP` exchanges two current `GLOBAL` range identifiers. This can change which
logical coordinate maps to a launch dimension and therefore which coordinate
varies across nearby lanes. It is useful when it improves launch geometry or
address coalescing; it can just as easily make them worse. Inspect reconstructed
indices after the swap rather than assuming that an axis reorder is cosmetic.

### `NOLOCALS`

`NOLOCALS` records that ordinary local/work-item indexing should not be used.
It is legal only before local, warp, or grouped-reduction ranges exist. Some
targets or kernels benefit from a simpler launch/index path, while others lose
useful cooperative structure. It is a scheduling choice, not a command to
remove `AddrSpace.LOCAL` from arbitrary already-lowered code.

### `THREAD`

`THREAD` is **not** the CUDA thread option. It factors a suitable weak/globalizable
range only for renderers with `has_threads=True` and uses that target's separate
thread facility. At this snapshot, that capability is exposed by the
Clang/LLVM/x86 family rather than CUDA GPU renderers. Learn it when working on
one of those targets; using it as a generic GPU mental model will make CUDA
launches harder to understand.

### `TC`

`TC` recognizes a compatible multiply-accumulate reduction and reorganizes
selected M, N, and K factors for a renderer-advertised tensor-core instruction.
The option must be the first recorded option in this snapshot. Its argument is
a tuple `(tc_select, tc_opt, use_tensor_cores)`:

- `tc_select=-1` tries advertised descriptions in order; a nonnegative value
  selects one descriptor;
- `tc_opt` controls matcher permissiveness, from strict (`0`) through broader
  matching (`1`) to padding-capable (`2`); and
- `use_tensor_cores=1` emits `WMMA`, while mode `2` applies tensor-core-shaped
  scheduling without using the `WMMA` UOp.

Renderer tensor-core descriptor dimensions are stored in source order
`(N,M,K)`, even though the familiar mathematical description is
`D[M,N] += A[M,K] * B[K,N]`. Check the field names before interpreting a tuple
as human M/N/K order.

On the CUDA/NV path at this snapshot, float32-input tensor-core matching is
gated by `ALLOW_TF32`; half-input paths have different dtype rules. A compatible
GPU does not make an arbitrary reduction eligible: the operation shape, casts,
dtypes, range roles, divisibility or padding policy, and advertised descriptor
must all match.

Tensor-core forms change grouping and often floating-point association. TF32
also changes input precision relative to ordinary float32 multiplication.
Validate with a tolerance justified by dtype and workload, including signed
and difficult magnitudes. Do not weaken an entire project's tolerance merely
to make one optimized case pass.

With `TC_OPT=2`, the matcher may invoke internal `PADTO` operations using
`append_opt=False`. Therefore a printed `applied_opts` tuple can contain `TC`
without showing every padding transformation that enabled it. Inspect the
resulting ranges and validity, not just the option log.

## What `apply_opt` proves—and what it does not

`Scheduler.apply_opt` resolves the option's current axis, checks relevant
renderer capabilities, verifies divisibility or padding constraints, enforces
axis-role and ordering rules, checks `THREAD`'s selected bound and a simplified
grouped-shared-storage bound where relevant, rewrites the AST, and normally
appends the option to `applied_opts`. Invalid requests raise `KernelOptError`.
Final GPU dimension limiting happens later; this function does not validate a
finished workgroup's registers, spills, or achieved occupancy.

A rejected option means only that this exact AST, target, current axis, amount,
and option sequence violate a contract. It does not show that the general
optimization idea is useless. Conversely, acceptance does not show that the
schedule is fast or even exhaustively resource-safe.

In particular, `apply_opt` does not calculate final register allocation,
compiler spills, achieved occupancy, cache transactions, warp stalls, or
elapsed time. Its grouped-reduction shared-memory check is deliberately simpler
than the final generated program and hardware behavior. Correct workflow is:

1. establish legality and structural intent;
2. compile and inspect the lowered program;
3. run a semantic oracle; and
4. measure the actual target under a controlled protocol.

## How tinygrad selects options

The codegen `apply_opts` path uses a priority order:

1. preserve an already tagged/optimized AST;
2. replay explicit `KernelInfo.opts_to_apply` options in order;
3. if `BEAM >= 1`, run BEAM search;
4. otherwise, if default optimization is enabled and the kernel is eligible,
   run `hand_coded_optimizations`; and
5. emit the optimized AST with its `KernelInfo.applied_opts` recipe.

`NOOPT=1` suppresses the default hand-coded heuristic. It does not erase an
explicit `opts_to_apply` recipe, prevent a requested `BEAM` search, or undo an
already tagged schedule. Say “heuristic disabled,” not “all optimization
disabled,” unless you have verified the other routes are absent.

### Hand-coded heuristic

The default heuristic is deterministic policy encoded in Python. At this
snapshot it attempts tensor-core matching first, includes a matrix-vector
special case, may group reductions, upcasts masked or reuse-friendly axes,
unrolls suitable reductions, and selects local or target-thread factors within
limits.

Heuristics are cheap enough for ordinary compilation and make decisions
predictably from structure. They also encode assumptions about common shapes
and devices. A heuristic change needs a workload matrix, including nearby
divisibility and symbolic cases, rather than one motivating kernel.

### BEAM search

BEAM starts from a `Scheduler`, enumerates legal next actions, compiles
candidates, times them on the selected target, and keeps the best `amt`
candidates for another round. It deduplicates identical generated binaries and
caches selected option sequences.

Measured search explores interactions that a fixed heuristic may miss, but its
measurement is contextual. With `BEAM_ESTIMATE=1`, search may run a reduced
global size and scale the duration. That reduces search cost but can distort
cache behavior, occupancy, launch overhead, and full-shape work distribution.
Always benchmark the winning full-size kernel independently.

BEAM is not a correctness oracle. Candidates still need semantic validation,
and timing noise, clocks, temperature, other processes, driver state, and cache
state can change the winner. Record whether BEAM's option cache was reused.
Run measured search only where the renderer's target device is a runnable real
target. In this snapshot, `PYTHON::sm_89` carries a CUDA target inside a Python
renderer, but BEAM's timing path asks for the target device (`CUDA`); it is not
a portable measured-search route.

## Portable lab: three schedules, one exact result

The repository lab makes the preceding discussion executable. It compiles the
same 16×16 float32 matmul as:

- an explicit empty option tuple (`baseline`);
- the exact `UPCAST`, `LOCAL`, `UNROLL` sequence (`manual`); and
- the snapshot's default heuristic (`heuristic`).

It uses inputs
`A[i,k] = 1 + i%4` and `B[k,j] = 1 + j%4`, so the independent exact result is
`C[i,j] = 16*(1+i%4)*(1+j%4)`. Every value is exactly representable in float32.

Run from the **guide repository root**, not from your home directory and not
from the tinygrad checkout. Substitute your actual checkout paths if they are
not siblings:

```bash
cd "/path/to/tinygrad_docs"
PYTHONPATH=../tinygrad-study \
  DEV=PYTHON::sm_89 DEBUG=0 CACHELEVEL=0 \
  "../tinygrad-study/.venv/bin/python" \
  labs/phase3/kernel_optimization.py --mode core
```

The guide runner invokes all three modes automatically on their one intended
hardware-free route. The core-mode output discussed first is:

```bash
cd "/path/to/tinygrad_docs"
python3 scripts/run_labs.py \
  --tinygrad "../tinygrad-study" \
  --python "../tinygrad-study/.venv/bin/python"
```

The important output is deterministic at the pinned snapshot:

```text
target: CUDA:PYTHON:sm_89
range states:
start: [('g0', 16, 'GLOBAL'), ('g1', 16, 'GLOBAL'), ('R0', 16, 'REDUCE')]
after UPCAST: [('g0', 16, 'GLOBAL'), ('g1', 4, 'GLOBAL'), ('u0', 4, 'UPCAST'), ('R0', 16, 'REDUCE')]
after LOCAL: [('g0', 4, 'GLOBAL'), ('g1', 4, 'GLOBAL'), ('l0', 4, 'LOCAL'), ('u0', 4, 'UPCAST'), ('R0', 16, 'REDUCE')]
after UNROLL: [('g0', 4, 'GLOBAL'), ('g1', 4, 'GLOBAL'), ('l0', 4, 'LOCAL'), ('u0', 4, 'UPCAST'), ('R0', 4, 'REDUCE'), ('r0', 4, 'UNROLL')]

mode: baseline
opts: ()
launch: (16, 16, 1) (1, 1, 1)
estimates: Estimates(ops=8192, lds=33792, mem=3072)
WMMA: 0
samples: 16.0 64.0 64.0 256.0
checksum: 25600.0

mode: manual
opts: (Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4))
launch: (4, 4, 1) (4, 1, 1)
estimates: Estimates(ops=8192, lds=21504, mem=3072)
WMMA: 0
samples: 16.0 64.0 64.0 256.0
checksum: 25600.0

mode: heuristic
opts: (Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=2))
launch: (1, 1, 1) (32, 1, 1)
estimates: Estimates(ops=8192, lds=3072, mem=3072)
WMMA: 2
samples: 16.0 64.0 64.0 256.0
checksum: 25600.0
```

The lab asserts these option recipes exactly and fails if target, range state,
launch, estimates, `WMMA` count, samples, or checksum drift.

### What `PYTHON::sm_89` means

This device string selects `PythonRenderer` and `PythonProgram`, while
configuring the renderer with CUDA target capabilities whose architecture is
`sm_89`. It is a hardware-free, Python-executed CUDA-targeted structural route:
excellent for checking Ada-targeted matching, range lowering, UOp structure,
and deterministic small-result smoke tests without launching a GPU. It does
not select `CUDARenderer` or compile CUDA C.

It is not a CUDA simulator or performance model:

- `PythonRenderer` stores an opaque pickle/base64 representation rather than
  CUDA source or native machine code;
- its barrier is effectively a no-op because the executor advances a logical
  group in synchronized fashion;
- its generic `WMMA` helper computes with ordinary Python multiply-and-sum
  behavior; and
- it does not reproduce warp concurrency, races, CUDA memory ordering, caches,
  register allocation, occupancy, TF32 rounding, native accumulation details,
  or elapsed hardware time.

Therefore the lab supports the claim “this pinned CUDA-targeted Python route
selected this structure and produced these smoke-test values.” It cannot
support “the kernel is race-free on CUDA,” “this is numerically identical to
native TF32,” or “this schedule is faster on an RTX 4090.” The runner
deliberately never replays this lab on user-added physical backends.

## Bounded extension: the 17×17 padding boundary

The same lab has two bounded 17×17 modes. Run them as separate commands and
therefore separate Python processes:

```bash
cd "/path/to/tinygrad_docs"
PYTHONPATH=../tinygrad-study DEV=PYTHON::sm_89 DEBUG=0 CACHELEVEL=0 \
  "../tinygrad-study/.venv/bin/python" \
  labs/phase3/kernel_optimization.py --mode padding-strict

PYTHONPATH=../tinygrad-study DEV=PYTHON::sm_89 DEBUG=0 CACHELEVEL=0 \
  "../tinygrad-study/.venv/bin/python" \
  labs/phase3/kernel_optimization.py --mode padding-enabled
```

Fresh processes matter here. The pinned in-process `to_program_cache` key does
not include `TC_OPT` or `TC_SELECT`; changing either after compiling an
otherwise identical AST in one process can reuse the first program. Each mode
sets its own optimization context before compiling and asserts every output
element, not only the printed samples and checksum.

With strict `TC_OPT=0`, the Ada tensor-core shape no longer divides the ranges.
The heuristic instead records a full reduction unroll:

```text
target: CUDA:PYTHON:sm_89
mode: padding-strict
opts: (Opt(op=OptOps.UNROLL, axis=0, arg=0),)
launch: (17, 17, 1) (1, 1, 1)
estimates: Estimates(ops=9537, lds=21964, mem=3468)
WMMA: 0
samples: 17.0 68.0 68.0 272.0
checksum: 28577.0
```

With `TC_OPT=2`, matching may pad internal axes and the result becomes:

```text
target: CUDA:PYTHON:sm_89
mode: padding-enabled
opts: (Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=3))
launch: (1, 1, 1) (32, 1, 1)
estimates: Estimates(ops=37568, lds=6528, mem=3468)
WMMA: 6
samples: 17.0 68.0 68.0 272.0
checksum: 28577.0
```

The full elementwise assertion establishes this small oracle; the samples and
checksum are only compact evidence that the asserted run completed. The jump
from `ops=9537` to `ops=37568` demonstrates why `Estimates.ops` must not be
described as a stable count of semantic useful work: padded tensor-core tiles
perform extra modeled operations. `mem` remains tied to the original buffers
in this model, while `lds` reflects the different lowered form.

This is a boundary test, not evidence that padding is profitable. On real
hardware compare 16, 17, nearby multiples, larger sizes, and realistic dtypes.
The cost of padded tiles matters far more for small or awkward dimensions.

## Measure one kernel on the RTX 4090

Portable inspection should precede hardware measurement, not replace it. The
following protocol answers a narrow question: how long does one already
compiled, warm-cache matmul call take on the CUDA `sm_89` route under this
recorded environment?

Save the script below as `bench_one_kernel.py` in the **guide repository root**.
It realizes inputs first, requires one computational program, prints hashes of
the generated source and compiled artifact, seeds tinygrad's runtime cache,
synchronizes, prints every timing sample, and validates against an independent
NumPy oracle.

```python
import hashlib, math, os, statistics
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.engine.realize import compile_linear, run_linear, time_call
from tinygrad.uop import Ops

n = int(os.getenv("N", "2048"))
warmup = int(os.getenv("WARMUP", "10"))
sample_count = int(os.getenv("SAMPLES", "50"))
assert n == 2048, \
  "the exact half-output oracle is pinned to N=2048; revise dtype, overflow guard, and tolerance before varying N"
assert sample_count > 0

# a[i,k]=1+i%4 and b[k,j]=1+j%4. Realize setup before scheduling out.
ones = Tensor.ones(n, n, dtype=dtypes.half).realize()
row = (Tensor.arange(n) % 4).reshape(n, 1).expand(n, n).cast(dtypes.half)
a = (ones + row).contiguous().realize()
b = (ones + row.permute(1, 0)).contiguous().realize()
out = a @ b

linear, var_vals = out.linear_with_vars()
compiled = compile_linear(linear)
calls = [c for c in compiled.src if c.src[0].op is Ops.PROGRAM]
assert len(calls) == 1, [c.src[0].op for c in compiled.src]
call, program = calls[0], calls[0].src[0]
kernel = program.src[0]

target = Device[Device.DEFAULT].renderer.target
assert Device.DEFAULT == "CUDA" and target.arch == "sm_89", \
  (Device.DEFAULT, target)
print("target:", target)
print("opts:", kernel.arg.applied_opts)
print("launch:", program.arg.global_size, program.arg.local_size)
print("estimates:", kernel.arg.estimates)
assert program.src[2].op is Ops.SOURCE and program.src[3].op is Ops.BINARY
source = program.src[2].arg.encode("utf-8")
compiled_artifact = program.src[3].arg
print("source sha256:", hashlib.sha256(source).hexdigest())
print("compiled artifact sha256:", hashlib.sha256(compiled_artifact).hexdigest())

# Load this compiled program once. Then complete setup before timing samples.
run_linear(compiled, var_vals, jit=True)
Device[Device.DEFAULT].synchronize()
for _ in range(warmup):
  time_call(call, var_vals)

samples = [time_call(call, var_vals) for _ in range(sample_count)]
ordered = sorted(samples)
p90 = ordered[math.ceil(0.90 * sample_count) - 1]  # nearest-rank p90
print("samples_us:", [round(x * 1e6, 3) for x in samples])
print("us min/median/p90:",
      *(round(x * 1e6, 3) for x in
        (ordered[0], statistics.median(ordered), p90)))

# Independent CPU construction of the closed-form result.
axis = 1 + np.arange(n, dtype=np.float32) % 4
expected = n * np.multiply.outer(axis, axis)
observed = out.numpy().astype(np.float32)
np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
print("checksum:", float(observed.sum(dtype=np.float64)))
```

Run from the guide root with an explicit backend and record the environment:

```bash
cd "/path/to/tinygrad_docs"
nvidia-smi --id=0 \
  --query-gpu=name,compute_cap,driver_version,pstate,temperature.gpu,clocks.sm \
  --format=csv,noheader

# Controlled heuristic with tensor-core matching enabled.
env PYTHONPATH=../tinygrad-study CUDA_VISIBLE_DEVICES=0 \
  DEV=CUDA DEBUG=0 N=2048 WARMUP=10 SAMPLES=50 \
  BEAM=0 NOOPT=0 TC=1 TC_OPT=0 TC_SELECT=-1 NOLOCALS=0 IMAGE=0 \
  CACHELEVEL=0 \
  "../tinygrad-study/.venv/bin/python" bench_one_kernel.py

# Same controls, but tensor-core matching disabled. This can change the whole
# heuristic recipe, not only one instruction.
env PYTHONPATH=../tinygrad-study CUDA_VISIBLE_DEVICES=0 \
  DEV=CUDA DEBUG=0 N=2048 WARMUP=10 SAMPLES=50 \
  BEAM=0 NOOPT=0 TC=0 TC_OPT=0 TC_SELECT=-1 NOLOCALS=0 IMAGE=0 \
  CACHELEVEL=0 \
  "../tinygrad-study/.venv/bin/python" bench_one_kernel.py
```

Replace physical GPU index `0` and its matching CUDA visibility selection if
the RTX 4090 has another index; record the mapping on a multi-GPU host. Use a
fresh process for each configuration. The `TC=1` versus `TC=0` comparison asks
what the complete controlled heuristic schedules do with matching enabled or
disabled. It does **not** isolate only the `WMMA` instruction: inspect and
report both full `opts` recipes. `NOOPT=1` is a useful intentionally weak
reference, but remember that it only suppresses the heuristic and may be
orders of magnitude slower; it is not the main competitor for a proposed
schedule. If testing BEAM, also report `BEAM`, `BEAM_ESTIMATE`, cache policy,
and the winning complete recipe.

Do not copy timings from this guide into an expectation. GPU clocks,
temperature, driver, power policy, host load, tinygrad commit, and generated
artifact all affect them. Save:

- tinygrad commit and dirty diff;
- exact `DEV`, target, dtype, shape, and optimization environment;
- GPU, driver, clocks/power policy, temperature, and competing load;
- applied options, launch dimensions, static estimates, source hash, and
  compiled-artifact hash;
- warm-up count, raw ordered or original samples, p90 definition, and summary;
  and
- the oracle, dtype conversion, and tolerance.

The analytic input is a strong exact smoke test for this shape, not a complete
floating-point suite. Before contributing a tensor-core or accumulation change,
also compare nonuniform signed values and difficult magnitudes against an
independent CPU/NumPy reference using a justified tolerance.

At this pinned default `DEV=CUDA` route, `Ops.BINARY` contains NVRTC-produced
PTX bytes passed toward the CUDA driver. Its hash is useful for detecting a
changed tinygrad compiler artifact, but it is not a hash of driver-JIT native
SASS/cubin and does not prove that final machine code is identical.

### Warm cache, perturbed cache, and compilation are different experiments

`time_call` returns synchronized device duration on CUDA through events. The
protocol above measures a loaded program after warm-up. Compile/search time is
excluded; measure `compile_linear` separately if startup latency is the target.

The `clear_l2=True` argument is not proof of a cold CUDA L2 in this snapshot.
`CUDADevice` has no `invalidate_caches` method, so `time_call` falls back to
touching a 1024×1024 default-float32 tensor—about 4 MiB. That is much smaller
than the RTX 4090's L2 and cannot guarantee eviction of the tested data. If you
use it, describe it as best-effort cache perturbation, print the exact method,
and do not label the result “cold L2.” A rigorous cold-cache experiment needs
a working set and access protocol designed for the relevant cache.

After an isolated win, rerun the original model with the same JIT/capture and
synchronization policy. Report an end-to-end improvement only if that workload
also improves.

## Profile only after you have a question

Timing tells you whether a configuration changed; it does not explain why. Use
Nsight Compute or compiler resource output after timings are repeatable and you
can state a concrete hypothesis. Examples include:

- “Mapping adjacent lanes to `j` should reduce memory sectors per request.”
- “Four output accumulators should reduce input load instructions but increase
  registers per thread.”
- “This grouped reduction should increase shared traffic and barriers while
  shortening the serial reduction.”
- “The tensor-core route should execute `WMMA`/tensor-pipe work rather than
  ordinary scalar multiply-add instructions.”

Then collect only counters that discriminate the hypothesis: achieved memory
bandwidth or sectors, tensor-pipe utilization, warp stall reasons, register and
spill counts, shared bytes, and resident warps. A large profiler report without
a predicted cause is difficult to interpret and easy to overfit.

## Question-led source stops

Do not read these files as isolated declarations. Bring the stated question,
read the narrow range, answer it in your own words, and return to the example.
All links are pinned to the chapter snapshot.

### Stop 1: What is an option record?

Question: which fields are recorded, what does `KernelOptError` mean, and which
option names exist? Read [`OptOps`, `Opt`, and `KernelOptError`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/__init__.py#L6-L20).
The answer is intentionally small: an `Opt` is an immutable operation/axis/arg
recipe. The declarations do not explain axis namespaces, legality, or hardware
effects; the following stops provide that context.

### Stop 2: How does one split preserve coordinates?

Question: where do `remaining*amount+new` and `new*remaining_size+remaining`
come from? Read [`shift_to`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L95-L102),
then enumerate the size-12, amount-3 case. Next read
[`real_axis`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L109-L124)
and explain why `UNROLL axis=0` did not select `g0`.

### Stop 3: What does legality actually cover?

Question: which checks apply to `UPCAST`, `LOCAL`, `UNROLL`, and grouped
reductions, and which resource claims are absent? Read the relevant branches of
[`apply_opt`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L126-L217).
Write two lists: checked invariants and measurements still required. Include
the internal `PADTO`/`append_opt` caveat in the second half of the range.

### Stop 4: Where do grouped storage and barriers appear?

Question: why did applying `GROUP` eventually produce an `AddrSpace.LOCAL`
buffer and a barrier? Follow grouped-reduction lowering at
[`codegen/__init__.py` lines 170–184](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L170-L184)
and barrier insertion at
[`lines 258–282`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L258-L282).
Notice that these effects occur after the option recipe.

### Stop 5: How do ranges become launch dimensions?

Question: which range types become GPU global and local dimensions, and where
are size limits enforced? Read
[`get_grouped_dims` and GPU dimension lowering](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/gpudims.py#L27-L88).
Relate the result to the manual lab's `(4,4,1)/(4,1,1)` launch.

### Stop 6: Who chose this recipe?

Question: was the schedule tagged, explicit, searched, or heuristic? Read
[`apply_opts`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L339-L356),
then the tensor-core and early policy in
[`hand_coded_optimizations`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/heuristic.py#L27-L41)
and its later choices at
[`lines 111–192`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/heuristic.py#L111-L192).
Predict what `NOOPT=1`, explicit options, and `BEAM=2` each do.

### Stop 7: What does tensor-core matching require?

Question: how are descriptor dimensions and dtypes recorded, and how does the
matcher find M/N/K ranges? Read
[`TensorCore` fields](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/tc.py#L5-L15),
the [CUDA descriptions](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/tc.py#L73-L97),
and the [matcher](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L219-L278).
Keep descriptor `(N,M,K)` storage order separate from mathematical notation.
For deeper work, continue to the
[`WMMA` construction](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L280-L315).

### Stop 8: What do the static numbers count?

Question: why are `lds` and `mem` different, and why can padded `ops` change?
Read [`Estimates.from_uops`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L10-L57).
Re-derive the baseline and upcast byte counts before trusting the printout.

### Stop 9: What can the Python target prove?

Question: what arithmetic does the generic Python `WMMA` helper perform, how
does the executor treat barriers/group execution, and what representation does
the Python renderer produce? Read the Python executor's
[`WMMA` arithmetic helper](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L28-L40),
[`RANGE`/barrier execution](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L47-L79),
and [renderer/runtime representation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L202-L225).
These lines establish ordinary Python multiply-and-sum behavior, not the
target-specific CUDA lane mapping. Write one supported claim and three
unsupported hardware claims.

### Stop 10: How are options searched and validated?

Question: what candidates exist, what gets timed, and what regression coverage
already exists? Read
[`get_kernel_actions`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/search.py#L89-L111),
[`beam_search`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/search.py#L114-L186),
and the pinned
[`tensor-core tests`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/opt/test_tensor_cores.py#L34-L148).
The snapshot's CI also exercises the emulated Ada tensor-core route in
[`test.yml` lines 118–124](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L118-L124).

### Stop 11: When can adjacent memory operations combine?

Question: which loads/stores are grouped, which dtypes and renderer
capabilities permit wider accesses, and where do validity and alignment enter?
Read [`memory_coalescing`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/coalesce.py#L100-L165).
Relate its offset groups and divisibility check to the earlier address
hypothesis; do not infer actual DRAM transactions from this source pass.

### Stop 12: What does the benchmark load, cache, and time?

Question: why do the padding modes use fresh processes, when is a runtime
object cached, what does `time_call` recompile or perturb, and how does CUDA
produce a synchronized duration? Read the
[`to_program` cache key](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L480-L485),
[`runtime` cache](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L111-L118),
[`time_call`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L277-L291),
and [CUDA event timing](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L26-L35).
Separate compilation, runtime construction, cache perturbation, launch, and
completion in your answer.

## Exercises

Try each question on paper before opening the answer.

### 1. Factor a coordinate

For a size-12 range split by 3, list the `(remaining,new)` pairs that reconstruct
original coordinates 0, 1, 2, 3, 4, and 11 under the bottom formula.

??? answer
    With `original = remaining*3 + new`, the pairs are `(0,0)`, `(0,1)`,
    `(0,2)`, `(1,0)`, `(1,1)`, and `(3,2)`. `remaining` is integer division by
    3 and `new` is remainder modulo 3.

### 2. Predict coalescing

At fixed `i` and `k`, adjacent lanes vary `j`. Which of `A[i,k]`, `B[k,j]`, and
`C[i,j]` varies across lanes, and which accesses are contiguous in row-major
storage?

??? answer
    `A[i,k]` is identical and can be broadcast/reused. `B[k,j]` and `C[i,j]`
    vary by one element across adjacent `j`, so they are contiguous. Actual
    transaction count still depends on alignment, active lanes, and access
    width.

### 3. Reproduce `lds`

Why does the baseline have `lds=33792`, and why does four-way `j` upcast reduce
it to `21504`?

??? answer
    Baseline input loads are two times `16^3` float32 values, or 32768 bytes,
    plus 1024 output-store bytes. Upcast reuses each `A[i,k]` over four `j`
    outputs, reducing A traffic in the model from 16384 to 4096 bytes. B remains
    16384 and C remains 1024, for 21504.

### 4. Interpret `LOCAL`

The manual schedule adds `LOCAL` but its `lds` does not change. Is that a bug?

??? answer
    No. `LOCAL` creates a work-item/workgroup launch coordinate; it does not
    itself allocate `AddrSpace.LOCAL` shared storage. This example's explicit
    reuse came from `UPCAST`. A grouped reduction or other later lowering may
    introduce shared buffers, but that is a separate effect.

### 5. Resolve the axis

After `UPCAST` and `LOCAL`, why does `UNROLL axis=0` select the size-16 reduction
rather than the first full-shape axis?

??? answer
    `UNROLL` resolves its axis through `unrollable_dims`, the filtered list of
    current `REDUCE` and `GROUP_REDUCE` ranges. The size-16 `R0` is entry zero of
    that list even though it appears later in `rngs`.

### 6. Compare grouped ownership

For reduction size 64 and amount 8, write the original-coordinate formula for
`GROUP` and `GROUPTOP`. What should you inspect before declaring either faster?

??? answer
    `GROUP` uses `remaining*8 + group`; `GROUPTOP` uses
    `group*8 + remaining`. Inspect which addresses each participant accesses,
    local-buffer/barrier lowering, generated instructions, workgroup resource
    use, and measured time. Equal launch sizes do not imply equal access order.

### 7. Explain the 17×17 operation jump

Why can the tensor-core form report `ops=37568` while the strict form reports
`9537`, even though both return the same 17×17 result?

??? answer
    `TC_OPT=2` permits padding to tensor-core tile multiples. The lowered form
    executes modeled work for padded lanes/tiles and `WMMA` uses its own count
    formula. `ops` describes that lowered representation, not only the 17³
    semantic products.

### 8. Apply selection priority

Predict the route for each case: `NOOPT=1` alone; `NOOPT=1` with explicit
`opts_to_apply`; and `NOOPT=1 BEAM=2` without explicit options.

??? answer
    `NOOPT=1` alone suppresses the hand-coded heuristic. Explicit options still
    replay because they have higher priority. A requested BEAM search still runs
    because the BEAM branch precedes the heuristic/`NOOPT` decision.

### 9. Match claims to backends

Which route supports each claim: ordinary `PYTHON`, `PYTHON::sm_89`, or physical
`CUDA`? Claims: generic UOp smoke correctness; Ada tensor-core matcher selected
`WMMA`; native TF32 numerical behavior; RTX elapsed time; absence of a
workgroup race.

??? answer
    Ordinary `PYTHON` supports generic UOp smoke correctness.
    `PYTHON::sm_89` additionally supports the pinned Ada-targeted matcher and
    UOp-structure claim. Native TF32 behavior and RTX elapsed time require
    physical CUDA. Race absence needs a stress-capable real target plus code
    inspection or an appropriate race tool; Python's synchronized logical
    execution cannot establish it.

### 10. Check Amdahl's law

A kernel is 2% of model time. Its time falls by 20%. What is the new normalized
model time and percent reduction?

??? answer
    `0.98 + 0.02*0.8 = 0.996`, a 0.4% total time reduction, assuming all other
    work is unchanged.

### 11. Investigate unchanged time

A change lowers `lds` substantially but median CUDA time is unchanged. Give
four plausible hypotheses and one discriminating measurement for each.

??? answer
    Removed loads may already hit cache (collect cache/DRAM transactions); a
    compute bottleneck may dominate (collect instruction or pipeline
    utilization); extra registers may lower residency (collect registers and
    active warps); or samples may be noisy (inspect raw samples and rerun fresh
    processes under stable clocks). Generated-source and compiled-artifact
    hashes also reveal whether the intended compiler output changed, while not
    fingerprinting driver-JIT native code.

### 12. Build a nearby-shape matrix

For a heuristic motivated by size 16, choose a minimal shape set that probes
divisibility and padding rather than testing only the winning case.

??? answer
    At minimum include 15, 16, 17, a larger exact multiple such as 32, and a
    larger nonmultiple such as 31 or 33 for each affected dimension. Vary one
    dimension at a time, then representative combined M/N/K cases. Add relevant
    symbolic bounds and production shapes; run correctness and timing on all,
    not merely `TC` presence.

## Contribution-shaped workflow

For a kernel-performance contribution:

1. Identify a kernel that matters in an end-to-end workload.
2. Save a minimal reproducer with exact shape, dtype, target, variables, inputs,
   and correctness oracle.
3. Print the pre-optimization ranges and current option-selection route.
4. State an index/resource hypothesis before editing the heuristic or lowering.
5. Test legality and semantic structure on portable routes, respecting what
   emulation cannot prove.
6. Compile the real target and compare options, launch, generated-source and
   compiled-artifact hashes, estimates, and compiler resource information.
7. Benchmark fresh processes with raw synchronized samples and a recorded
   environment.
8. Test nearby divisible, non-divisible, small, large, and symbolic cases.
9. Run existing targeted tests plus a representative kernel/workload set to
   catch heuristic regressions.
10. Re-run the originating end-to-end workload and report both kernel and model
    effects.

A faster number is a lead. A contribution needs a causal explanation, a stable
oracle, a stated workload scope, and evidence that the change belongs in kernel
optimization rather than execution scheduling, lowering, rendering, or the
runtime.

## Checkpoint

Continue when you can do all of the following without relying on an unexplained
source declaration:

- derive the `i`, `j`, and `k` loops and row-major addresses of the carried
  matmul;
- distinguish work item, workgroup, subgroup, lane, grid, and their CUDA names;
- distinguish tinygrad `LOCAL`, `AddrSpace.LOCAL`, and CUDA local memory;
- explain coalescing as aligned memory transactions and reuse as an index fact;
- reconstruct a factored coordinate and resolve an option's current axis;
- explain `UPCAST`, `LOCAL`, `UNROLL`, grouped reduction, padding, swapping,
  tensor cores, and their main resource risks;
- derive the baseline and upcast estimates and state what `ops`, `lds`, and
  `mem` do not measure;
- predict explicit/BEAM/heuristic selection, including the narrow meaning of
  `NOOPT`;
- state exactly what `PYTHON::sm_89` can and cannot prove; and
- produce a synchronized, correctness-checked CUDA comparison with raw samples
  and enough metadata for another contributor to reproduce it.

## Quick reference

| Observation | Inspect or measure next |
| --- | --- |
| Model slow, no dominant kernel | Kernel count, fusion/materialization, copies, JIT/graph path |
| One kernel dominates | Ranges, applied opts, launch, generated program, hardware counters |
| `LOCAL` appears but no shared buffer | Expected: launch-local axis is distinct from `AddrSpace.LOCAL` |
| Grouped reduction is slower | Partial ownership, barriers, local traffic, shared bytes, resident workgroups |
| No tensor core | Target descriptor, multiply/reduce form, dtype/casts, M/N/K divisibility, `ALLOW_TF32`, `TC_OPT` |
| `TC` appears but no printed `PADTO` | With `TC_OPT=2`, inspect internal padded ranges/validity; padding may not be appended |
| High `lds` relative to `mem` | Repeated accesses/reuse opportunity; verify actual transactions with counters |
| Adjacent lanes generate many transactions | Address formula, alignment, active mask, width, late coalescing preconditions |
| More upcast/unroll is slower | Registers, spills, code size, remaining parallel work |
| BEAM winner regresses at full size | Reduced-size estimate, noise, cache state, shape representativeness |
| `NOOPT=1` still shows options | Check explicit recipe, BEAM, or already tagged schedule |
| `PYTHON::sm_89` looks fast/slow | Discard timing claim; use only targeted codegen/semantic inspection |
| `clear_l2=True` on CUDA | Treat as best-effort 4 MiB perturbation, not a guaranteed L2 flush |
| Kernel improves but model does not | Amdahl's law, new bottleneck, launch/JIT effects, end-to-end policy |

## Optional background

Use these only when a concept remains unclear or when hardware evidence becomes
necessary:

- The bounded [GPU execution route](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
  links the CUDA programming model, Best Practices Guide, and Ada tuning guide.
- NVIDIA's [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
  is the primary reference for grids, blocks, threads, warps, and memory scopes.
- The [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  develops coalescing, occupancy, measurement, and numerical guidance.
- The [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)
  supplies architecture-specific limits after the generic model is solid.
- Triton's [vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
  and [matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
  provide a second compiler-scheduling vocabulary. Translate concepts, not API
  names, back into tinygrad.
- The [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  is useful once you have a stable timing and a specific counter question.

## Deliberate deferrals

You now have enough background to reason about option selection, but several
implementation layers deserve their own chapters:

- Chapter 10 follows range optimization through lowering, buffer insertion,
  barriers, and control flow.
- Chapter 11 explains renderer output and generated-source inspection.
- Chapter 12 covers runtime program loading, synchronization, and device timing.
- Chapter 14 develops the CUDA/NV paths, native NVIDIA artifacts, and
  architecture-specific debugging.
- Chapter 17 expands isolated measurements into rigorous performance work.

Also defer exhaustive tensor-core lane layouts, shared-memory bank-conflict
analysis, SASS-level scheduling, and complete BEAM internals until a concrete
contribution requires them. The question-led stops above tell you where to
resume without making those advanced topics prerequisites for a first kernel
change.

[← Shapes and indexing](08-shapes-and-indexing.md) · [Next: Lowering a kernel →](10-lowering.md)
