# 17. Performance engineering

## The promise of this chapter

“Faster” sounds like one property.  In a compiler and GPU stack it can refer to
very different intervals:

- less Python time constructing or scheduling a graph;
- less time rendering and compiling programs;
- fewer or better realization boundaries across a model;
- less host overhead submitting already compiled work;
- less device time in one kernel;
- more examples completed per second through overlapping work; or
- lower first-token, per-token, training-step, or end-to-end latency.

An improvement to one interval does not automatically improve the others.  A
kernel can run 20% faster while a model is unchanged because that kernel was
only 1% of the step.  Fusing two kernels can remove a launch but increase
register pressure and make the new kernel slower.  A compiler cache hit can
look like a compilation speedup.  An unsynchronized Python timer can report
that a GPU operation took microseconds when it measured only queue submission.
And a wrong result can be arbitrarily fast.

A performance contribution is therefore not an attractive generated kernel
followed by a smaller number.  It is a falsifiable claim about a named workload,
layer, metric, and environment, supported by a correctness oracle and a
comparison another contributor can reproduce.

This chapter begins with the vocabulary of measurement and develops the four
performance layers used in tinygrad.  It gives you a route from an end-to-end
symptom to the artifact that owns the cost, then back to the end-to-end workload
after a change.  The kernel mechanics build on
[Chapter 9](09-kernel-optimization.md); this chapter is about deciding whether
a kernel is actually the right thing to optimize and proving that a local
change matters.

By the end, you should be able to:

- distinguish latency, throughput, host time, device time, compile time, and
  critical-path time;
- write a benchmark claim that says exactly what begins and ends the timer;
- bracket every performance experiment with a correctness oracle;
- choose cold/warm cache, compilation, JIT, allocation, and device-frequency
  states deliberately;
- collect raw samples and summarize variation without selecting a favorite;
- use static operation/traffic estimates without calling them hardware
  measurements;
- use `DEBUG=2`, `time_call`, VIZ, generated source, and hardware profiling for
  the questions they actually answer;
- form roofline, occupancy, launch-overhead, and Amdahl-law hypotheses without
  turning any of them into conclusions prematurely; and
- report a cross-revision result with enough environment and artifact evidence
  for review.

**Source snapshot:** `874d331` (2026-08-05).  Performance behavior depends on
the exact tinygrad revision, renderer/compiler route, driver, device state, and
workload.  Treat every number in your own experiment as local evidence, not as
a portable property of the project.

## Route through the chapter

Read front to back once:

1. define the workload and distinguish latency from throughput;
2. draw the host/device timeline and place the timer endpoints;
3. separate useful work, estimated work, elapsed time, and achieved rate;
4. split compile, model/scheduler, kernel, and submission costs;
5. write a falsifiable baseline/candidate claim before editing code;
6. make correctness, warm-up, cache state, and synchronization part of the
   protocol;
7. preserve a distribution of samples and estimate the noise floor;
8. use tinygrad's diagnostics to localize rather than to manufacture the final
   number;
9. form a resource/bottleneck hypothesis from the first costly artifact;
10. estimate its maximum possible end-to-end effect;
11. run the bundled mechanics lab and a bounded physical-GPU follow-up; and
12. return to the full workload to accept or reject the original claim.

The quick reference is for later recall.  It omits the reasoning that prevents
most misleading benchmarks.

## Performance vocabulary from first principles

### A workload is part of the result

A **workload** is the exact computation and input regime being measured.  “An
LLM” is not one workload.  These can exercise different bottlenecks:

```text
prefill:       batch 1, sequence 2048, float16
decode:        batch 1, one new token, KV cache length 2048
training:      batch 8, sequence 512, forward + backward + optimizer
microkernel:   M=128, N=4096, K=4096 matrix multiplication
```

Shapes, dtype, strides/views, batch size, model state, JIT phase, and input
distribution can change the chosen schedule and generated programs.  Record
them.  A win at one shape may be valuable, but it is not evidence for every
shape hidden behind the same Python function.

A **baseline** is the unmodified comparison state.  A **candidate** is the
state containing the proposed change.  Identify both by commit, dirty diff,
configuration, and environment.  “Before” and “after” are ambiguous if caches,
driver state, or unrelated edits changed too.

### Latency and throughput answer different questions

**Latency** is elapsed time for one named unit from a defined start to a defined
completion:

```text
latency = completion_time - start_time
```

Examples include time to first token, synchronized time for one decoder step,
or cold time from process start through the first correct output.

**Throughput** is completed useful units per interval:

```text
throughput = completed_items / elapsed_time
```

Examples include tokens/s, images/s, or training samples/s.  Queueing a batch
and synchronizing once can improve throughput by overlapping work even if the
latency of an individual item does not improve.  Conversely, synchronizing
after every item can produce a valid latency protocol while destroying the
pipeline whose throughput a user cares about.

Always name the unit.  “1,000 ops/s” is meaningless if *op* might mean a Tensor
operation, scheduled call, generated instruction, floating-point operation, or
whole model step.

### The host and device have separate timelines

On a CPU-only synchronous route, a function may return after its computation is
finished.  An accelerator runtime often submits work and returns before the
device completes it:

```text
host:    prepare args ─ submit K1 ─ submit K2 ─ other Python ─ synchronize
                          │           │                         │
device:                   └─ execute K1 ─┴─ execute K2 ─────────┘ complete

host submission interval: call entry → enqueue/return
device execution interval: device starts → device finishes
completed wall interval:   drained start → synchronization returns
```

`time.perf_counter()` observes the host clock.  If the end timestamp is taken
immediately after submission, it need not include device execution.  To measure
completed latency, drain prior work before the start and synchronize after the
work before the stop.  To measure submission overhead, intentionally omit the
final wait—but label that interval submission time and establish device
completion separately.

Device events can measure time on a device timeline with less host overhead.
They still need correct placement, queue/stream semantics, and synchronization
before the host reads the result.  They do not include Python graph building,
compiler work, or time before/after the recorded events.

### Work, time, and achieved rate are not interchangeable

Suppose a kernel performs an estimated `W` floating-point operations in `t`
seconds.  Its derived rate is:

```text
achieved FLOP/s = W / t
```

If it transfers an estimated `B` bytes:

```text
achieved byte/s = B / t
```

The formulas are simple; the meanings of `W`, `B`, and `t` are not.  tinygrad's
static estimates count operations and modeled loads/stores in its program
representation.  A hardware profiler counts or estimates different events
such as issued instructions, cache sectors, or DRAM transactions.  An
application may call only some operations useful work.  A wall timer may
include launch overhead; a device event may not.

Keep the nouns attached:

```text
static estimated operations / synchronized wall time
static capped buffer-role bytes / runtime-reported call time
profiled DRAM bytes / device event time
tokens / end-to-end wall time
```

Do not shorten all four to “performance.”

### A bottleneck is a limiting mechanism, not the largest-looking number

A **bottleneck** is the mechanism currently limiting the metric.  Common
possibilities include:

- Python/frontend or compiler work;
- serial launch and synchronization overhead;
- device arithmetic throughput;
- memory bandwidth or cache behavior;
- dependency/critical-path latency;
- insufficient parallel work;
- register or local-memory pressure limiting resident workgroups;
- host↔device transfer; or
- an unnecessary realization/copy chosen earlier by the scheduler.

The slowest isolated kernel is not automatically the model bottleneck.  It may
overlap another stream, run only once during initialization, or occupy a small
fraction of the steady step.  Conversely, hundreds of tiny launches can be
collectively expensive even when no kernel is individually prominent.

A good investigation moves in both directions:

```text
end-to-end symptom
  ↓ attribute time/calls/dependencies
owning schedule, kernel, copy, compile stage, or host gap
  ↓ change and validate locally
same end-to-end workload under the original protocol
```

Stopping at the local artifact proves a local effect, not the user-visible
claim.

## Amdahl's law: bound the possible win before coding

If the affected component accounts for fraction `f` of total time and the
change makes that component `s` times as fast, the idealized total speedup is:

```text
total speedup = 1 / ((1 - f) + f/s)
```

If one kernel is 5% of a step and becomes twice as fast:

```text
1 / (0.95 + 0.05/2) ≈ 1.026
```

The maximum modeled end-to-end improvement is about 2.6%, before considering
new overhead, changed overlap, or noise.  Even making that kernel infinitely
fast cannot improve the step by more than about 5% in this simplified model.

Amdahl's law is not a proof because real changes can alter fusion, overlap,
memory pressure, and downstream scheduling.  It is a sanity bound.  If a local
artifact cannot account for the claimed whole-model delta, either the
attribution is incomplete or another variable changed.

## Prerequisite gate

Before continuing, you should be able to:

- trace one tensor expression through scheduling, lowering, rendering, and a
  runtime launch;
- distinguish a realization boundary from a fused operation;
- explain asynchronous GPU execution and why a host timer needs an explicit
  completion point;
- read a generated kernel's launch dimensions and memory accesses; and
- write a correctness test with an independent oracle.

If kernel schedules, upcasting, local memory, or tensor cores are unfamiliar,
return to [Kernel optimization](09-kernel-optimization.md). If GPU latency,
bandwidth, occupancy, or profiling counters are unfamiliar, take the bounded
[GPU-performance route](../reference/learning-resources.md#gpu-performance),
then return when you can distinguish occupancy from achieved performance.

## Mental model: four clocks, one outcome

The upstream speed note separates performance into four layers. Preserve that
separation in every investigation:

| Layer | Question | Typical artifacts | Representative measurement |
| --- | --- | --- | --- |
| **Compile/Python** | How long does tinygrad take to construct, schedule, rewrite, render, compile, and possibly search before useful execution? | Tensor/UOp construction, schedule and rewrite traces, renderer/compiler calls, BEAM candidates, Python profiles | Fresh-process and warm-cache host wall time, with stages split where possible |
| **Model/scheduler** | Did tinygrad choose the right realization boundaries, fusion, recomputation, copies, and kernel sequence for the whole workload? | `LINEAR` calls, kernel count, dependency graph, intermediate buffers, estimated traffic | Synchronized steady-state step latency plus schedule/kernel/memory evidence |
| **Kernel/codegen** | Given one scheduled operation, is its equivalent lowered program efficient? | Kernel AST, applied `OptOps`, generated source, launch sizes, resource use | One-call device time, achieved work rates, hardware counters where needed |
| **Execution/submission** | How much time is spent launching, queueing, graphing, waiting, and moving between host and device after work is compiled? | `TinyJit` capture/replay, graph or HCQ calls, queue timeline, host/device profile gap | Synchronized replay latency or throughput, compared with device critical-path time |

These layers interact. A scheduler change can create fewer but slower kernels;
a kernel optimization can make compilation much slower; a graph can reduce
submission cost without changing any kernel; a cache hit can masquerade as a
compiler speedup. Report the layer you changed and the user-visible metric it
affects.

For a latency workload, a useful accounting model is:

```text
cold latency  = graph/model construction + scheduling + rewrite/search
              + source/binary compilation + initialization + first execution

steady latency = remaining host work + submission/waits + device critical path
```

This is not always an additive stopwatch equation. CPU work, copies, multiple
streams, and kernels can overlap. Summing every device event can exceed wall
time; the critical path is what bounds latency. Conversely, timing only the
Python call can undercount asynchronous device work.

### Turn an intuition into a claim

“Matmul is slow” and “this source looks better” are optimization hunches. A
claim must be capable of being false. For example:

```text
On commit C, DEV=CUDA, device D and driver R, the recorded decoder step at
shape S/dtype T has synchronized steady-state median latency X ms. Kernel K is
Y% of the device critical path. Replacing schedule A with semantically
equivalent schedule B should reduce K's median hot-cache time without changing
kernel count, numerical output, or compile time by more than Z%.
```

Before changing code, write down:

1. the exact workload and input distribution;
2. the observable metric and layer;
3. the proposed bottleneck mechanism;
4. evidence that would disprove it; and
5. the correctness and performance acceptance criteria.

The [contribution brief](../reference/contribution-brief.md) has fields for
this evidence. Filling it in early prevents an isolated microbenchmark from
silently becoming a model-speed claim.

## Measurement is part of the experiment

### Correctness brackets every benchmark

Check correctness before timing the baseline and after timing the candidate.
Use fixed inputs or seeds and an independent oracle such as NumPy, PyTorch, a
simpler backend, a specification, or a known invariant. Record tolerances,
dtype, accumulation behavior, and any nondeterminism.

A wrong kernel can be extremely fast. So can a scheduler that omitted work, a
JIT replay that reused stale buffers, or a benchmark whose result was never
forced. Check both output values and relevant side effects. For training, that
may include loss, parameter updates, and gradients rather than only the forward
tensor.

### Warm-up defines the state you are measuring

“Warm up three times” is not a complete protocol. Identify which transitions
can occur in your workload:

- device/context initialization and allocation;
- tensor graph construction and scheduler-cache population;
- UOp rewriting, rendering, binary compilation, and compiler-cache population;
- BEAM or local-size search;
- `TinyJit` capture and graph construction; and
- device frequency, page mapping, and data-cache state.

For **cold compile latency**, use a fresh process and a dedicated temporary
`CACHEDB`, and explicitly record `CCACHE`, `SCACHE`, `BEAM`, renderer, and all
other relevant flags. Do not erase a user's general cache to manufacture a
cold result. For **warm compile latency**, prime the named caches in the same
way on both revisions. For **steady execution**, construct and realize inputs,
finish compilation/search/JIT capture, synchronize, and only then start taking
samples.

GPU cache state is a separate choice. Hot-cache latency can represent repeated
model execution; cold- or disturbed-cache latency can expose traffic
sensitivity. Keep the choice constant. tinygrad's internal `time_call(...,
clear_l2=True)` path is backend-dependent and may use cache invalidation or a
fallback displacement workload. Inspect that implementation before calling
the result “cold L2”; it is not a universal hardware guarantee.

### Synchronize the boundary you claim

For synchronized latency of one operation:

```python
Device[device].synchronize()
start = time.perf_counter()
result = workload()
Device[device].synchronize()
elapsed = time.perf_counter() - start
```

The first synchronization drains earlier work; the second makes the timed
endpoint include completion. Keep input preparation, host-to-device copies,
and output copies inside or outside this interval according to the claim, and
say which.

For throughput, synchronizing each item can destroy the queueing and overlap
being measured. Submit a declared batch or time window, synchronize at its end,
and report items/second plus latency if it matters. Do not compare a serialized
latency protocol with a pipelined throughput protocol.

### Report a distribution, not a favorite sample

Keep raw samples. For routine comparisons, report at least sample count,
median, p10, and p90; min can be useful as an estimate of best attainable time,
but it does not describe typical latency. State the percentile convention for
small samples. Repeat enough times to expose compilation leaks, clock changes,
thermal drift, and sporadic synchronization.

Run baseline and candidate under the same power, clock, process, driver, and
background-load conditions. Interleave A/B when practical, or repeat both
orders, so a warming GPU does not systematically favor the second revision.
Estimate the noise floor before treating a small delta as a speedup.

### Turn samples into a decision without hiding variation

Suppose baseline medians from repeated blocks are around `100 µs` and candidate
medians are around `99.7 µs`, while either revision naturally ranges from
`97–104 µs`.  Reporting the single best baseline and candidate can manufacture
almost any result.  The candidate may be better, but this experiment has not
separated a `0.3%` effect from its noise.

Preserve at least:

```text
raw samples
number of warm-ups and measured samples
sample/block ordering
summary convention
absolute delta = candidate - baseline
relative delta = candidate/baseline - 1
device temperature/clock/power policy when relevant
outliers with causes, not silently deleted rows
```

For a small fixed-cost microbenchmark, measure batches as well as individual
calls.  If timer resolution or launch bookkeeping is a large fraction of one
sample, timing 100 prepared repetitions inside one synchronized interval and
dividing by 100 can estimate average throughput cost.  That changes the metric:
it may include queueing/overlap and hides per-call tail latency.  Keep both when
both questions matter.

Use paired or interleaved comparisons when environment drift is material:

```text
A1, B1, B2, A2, A3, B3 ...
```

Randomize or balance order, but retain it.  Comparing all baseline samples in a
cold morning process with all candidate samples after the GPU warms biases the
revision effect.  For compiler cold-start claims, pairing may instead mean
fresh processes with alternating revision order and separate temporary cache
paths.

There is no universal sample count or significance test.  Choose a protocol
that can resolve the acceptance threshold you wrote before measuring.  Useful
steps are:

1. run baseline versus itself in the intended harness to estimate natural
   variation;
2. choose a minimum effect that matters to users and maintainers;
3. collect enough independent blocks to distinguish that effect under normal
   variation;
4. inspect distributions and time order, not only one aggregate; and
5. repeat on another session or machine when the claim is important and the
   effect is close to the noise floor.

Statistical confidence cannot rescue a changing workload or incorrect timer
boundary.  Ten thousand precise samples of queue submission do not establish
device completion latency.

### Performance tests and benchmarks have different jobs

A benchmark records a measurement under a protocol.  A performance regression
test turns part of that protocol into an automated acceptance gate.  The gate
must tolerate expected CI variation without permitting the regression it is
meant to catch.

Prefer deterministic structural or complexity assertions when they express the
contract more directly:

- one copy rather than two;
- no accidental realization boundary;
- bounded generated operation count;
- a required tensor-core or vector path for a controlled target;
- compile search visits no more than a justified bounded set; or
- JIT replay does not execute the Python body.

Pair those with measured evidence during review.  A wall-time threshold in
shared CI is appropriate only when the effect is large, the environment and
workload are controlled, timeouts are safe, and the failure message preserves
enough evidence to distinguish load from a real regression.  Never loosen an
arbitrary threshold until the current machine happens to pass.

## tinygrad's performance instruments

No single tool answers every layer. Use instrumentation to localize, then use a
clean benchmark to establish the final number.

### `DEBUG=2`: per-call diagnostic timing

At `DEBUG=2`, `track_stats` prints each executed call with elapsed time,
cumulative time, allocated memory, static operation estimates, and two static
memory estimates. It also updates `GlobalCounters`:

```text
kernel_count, global_ops, global_mem, time_sum_s
```

This is excellent for discovering kernel count, large calls, unexpected copies,
and a first model-to-kernel ranking. It is not a transparent end-to-end timer.
In this snapshot, `run_linear` requests waits whenever `DEBUG >= 2`; when a
runtime does not return elapsed time, `track_stats` synchronizes and uses host
wall time. Printing and serialization can perturb tiny workloads and remove
asynchronous overlap.

Despite its name, `GlobalCounters.kernel_count` increments for every call that
reaches `track_stats`'s update block, including modeled copies, views, graphs,
and other tracked call bodies—not only physical GPU kernel launches.  Calls
with `ExecContext.update_stats=False`, including `time_call`, return before that
block.  The separate `exec_validate` CPU comparison does not itself pass
through `track_stats` and is not an additional counted call, although enabling
validation can still add other work to the experiment.  Classify the call body
before reporting a kernel count.

The counters are process-global. `GlobalCounters.reset()` clears operations,
estimated memory, elapsed time, and call count, but deliberately does not reset
`mem_used`; treat allocation state separately when comparing runs.

Use `DEBUG=2` for attribution. Re-run the final comparison with `DEBUG=0` and
explicit synchronization around the boundary you actually claim.

### `Estimates`: static work, not hardware counters

The compiled program carries an `Estimates` value with three fields:

- `ops`: statically counted modeled ALU work (with special weights for
  `MULACC` and `WMMA`), displayed as FLOPs even though the group can include
  non-floating ALU operations;
- `lds`: bytes represented by repeated modeled loads and stores whose address
  space is not `REG`; register-addressed accesses are excluded; and
- `mem`: only loads/stores whose address-source chain reaches a `PARAM`, capped
  separately for each such buffer's `LOAD` and `STORE` role.  Repeated accesses
  in one role cannot exceed that parameter buffer's full extent in the model,
  while one parameter that is both read and written can contribute in both
  roles.  Local/shared/register-rooted storage is not counted in this field.

`DEBUG=2` divides these values by measured elapsed time to display derived
FLOP/s and `mem|lds` bandwidth. They are valuable consistent models for
comparing equivalent generated programs. They are not measurements of DRAM
transactions, cache hits, instruction issue, or useful application work.
Predication, cache behavior, replay, spills, and hardware transactions can make
actual traffic differ. If a claim depends on those facts, inspect generated
source and use the target vendor's hardware profiler.

### `time_call`: isolate one scheduled call

`tinygrad.engine.realize.time_call` passes a one-call `LINEAR` through
`compile_linear(..., beam=0)`, links with its execution-context cache disabled,
executes with `wait=True`, and returns elapsed time. Existing in-process program
or runtime entries and the compiler's disk cache may still be hits; the helper
does not insert a missing runtime into `runtime_cache`. It is useful after a
full-workload profile has identified a call. It is an internal,
snapshot-specific helper—not a stable public benchmark API.

Hold the call's buffers, bound variables, cache policy, and launch state fixed.
Run multiple samples, validate its output, and return to the whole workload.
Because the helper forces `beam=0`, it does not by itself measure the cost or
selected result of a normal BEAM search. A fast isolated call also says nothing
about launch count or overlap.

### `VIZ=1`, profiling, and Python-stage timing

`VIZ=1` supplies the import-time default `PROFILE=1`; an explicitly inherited
`PROFILE=0` overrides that default.  When both artifacts are required, set
`VIZ=1 PROFILE=1 TRACK_MATCH_STATS=2` before Python imports tinygrad.  Changing
VIZ later with a local `Context` does not install the import-time tracked
matchers and exit hooks. The snapshot CLI can inspect the latest capture:

```bash
.venv/bin/python -m tinygrad.viz.cli
DEBUG=3 .venv/bin/python -m tinygrad.viz.cli --json > /tmp/tinygrad-events.jsonl
.venv/bin/python -m tinygrad.viz.cli -t
.venv/bin/python -m tinygrad.viz.cli | rg MARKER
```

Add `profile_marker` around a repeated step, use an interval between markers,
and inspect both `TINY` CPU ranges and device events. Higher CLI `DEBUG` levels
include generated source and progressively more rewrite detail; consult the
snapshot VIZ README before choosing one. Capturing traces and large graphs adds
overhead, so profiles establish *where* to measure, not the final latency.

For compile/Python investigations, `Timing` gives coarse host regions and
`Profiling` wraps `cProfile`. The external schedule benchmark demonstrates how
to split model Tensor construction, scheduling, rewriting, linearization, and
verification instead of reporting one opaque first-run time.

## Roofline reasoning without false precision

For a chosen memory level, arithmetic intensity is:

```text
I = operations / bytes moved
```

Given peak compute rate `P` and bandwidth `B`, the roofline bounds are:

```text
attainable rate <= min(P, B * I)
time lower bound >= max(operations / P, bytes / B)
```

The units and level must match. DRAM bandwidth needs DRAM bytes; L2 bandwidth
needs L2 traffic. Peak compute depends on dtype, instruction path, clocks, and
whether tensor cores are actually used. Use measured or authoritative numbers
for the recorded device rather than a product-page headline.

`Estimates.mem` can approximate capped program-visible buffer-role bytes and
`Estimates.lds` can approximate repeated program-visible accesses. Neither
establishes traffic at a hardware memory level. Treat “compute-bound” or “memory-bound” as a
hypothesis: check achieved rates, source, and—when the decision depends on
it—hardware counters or controlled perturbations.

### Derive a simple intensity before reading a profiler

Consider elementwise addition of two `float32` arrays into a third:

```text
out[i] = a[i] + b[i]
```

Ignoring cache reuse and write-allocation details, each element performs about
one floating add and moves at least:

```text
4 bytes from a + 4 bytes from b + 4 bytes to out = 12 bytes
```

Its algorithmic arithmetic intensity is roughly:

```text
1 operation / 12 bytes ≈ 0.083 operations per byte
```

Even an accelerator with enormous arithmetic throughput can be limited by
memory traffic for a large streaming add.  Changing an ADD instruction is
unlikely to matter; fusion that avoids an intermediate write/read might.

Now consider a dense square matrix multiplication of size `N`.  It performs on
the order of `2N³` floating operations.  If each input and output matrix could
be moved only once from a chosen memory level, algorithmic bytes grow like
`N²`, so intensity grows with `N`.  A naive kernel may reload input elements
many times, making actual program-visible and hardware traffic much larger.
Tiling, local memory, registers, and tensor-core fragments try to create reuse
closer to the arithmetic units.

This comparison tells you where to look; it does not predict the exact kernel
time.  For each roofline argument write:

```text
operation definition:
byte definition and memory level:
dtype/instruction path:
measured or assumed compute roof:
measured or assumed bandwidth roof:
achieved rate and timer boundary:
evidence that the kernel actually follows this path:
```

If `B * I` is far below `P`, bandwidth is a plausible limit at the named level.
If `P` is lower, arithmetic throughput is a plausible limit.  Real kernels can
sit below both roofs because of dependencies, instruction mix, launch size,
bank conflicts, uncoalesced access, spills, synchronization, or insufficient
parallelism.  “Below the roofline” is the beginning of attribution, not a bug by
itself.

### Memory levels must stay named

“Memory bandwidth” can refer to several paths:

- registers/local values used by one thread;
- workgroup/shared memory visible to cooperating threads;
- L1 and texture/read-only cache paths;
- shared L2 cache;
- device DRAM;
- host-pinned transfer memory; or
- PCIe/another host↔device link.

Bytes can be served by a cache and never reach DRAM.  Loads visible in UOps can
be combined, predicated away, replayed, or spill into additional traffic after
register allocation.  A hardware counter also needs interpretation: requested
bytes, sectors, transactions, and DRAM bytes are not necessarily the same
quantity.

Use controlled perturbations to test a hypothesis.  Vary working-set size
across an expected cache boundary, vary reuse while preserving operation count,
or vary arithmetic work while preserving bytes.  Keep the semantic output and
schedule facts visible.  A response consistent with the hypothesis is useful;
confirm consequential claims with the most direct counters available.

### Occupancy is a constraint, not the objective

Active warps/workgroups are limited by threads, registers per thread, shared
memory per workgroup, and architecture limits. More `UPCAST`, `UNROLL`, local
storage, or tensor-core tiling can increase reuse and arithmetic intensity but
also raise register/shared-memory demand. The result may be fewer resident
warps, spills to local memory, or both.

High occupancy does not guarantee fast execution; some kernels have enough
latency hiding at lower occupancy and benefit from greater reuse. Low occupancy
does not prove it is the bottleneck. Inspect resource usage and spills, vary one
schedule dimension at a time, and measure the resulting kernel *and model*.

## Attribute a model regression to the owning layer

Use this funnel rather than jumping from model latency to codegen:

1. **Freeze semantics.** Save inputs, outputs/oracle, shape, dtype, mode, flags,
   commit, device, renderer, driver, and target.
2. **Reproduce the full metric.** Separate cold first-run and steady execution;
   establish a raw-sample distribution and noise floor.
3. **Compare host wall with the device timeline.** A large gap after warm-up
   suggests scheduling, submission, synchronization, allocation, copies, or
   Python work. A dominated device critical path suggests kernels or transfers.
4. **Inspect the execution plan.** Record calls, kernel count, copies,
   materializations, recomputation, intermediate storage, and static estimates.
   If those differ, investigate the scheduler before tuning a kernel.
5. **Rank stable contributors.** Identify the kernel or gap that can account
   for the observed regression. Preserve its AST, generated source, launch
   dimensions, estimates, and share of the critical path.
6. **Isolate only now.** Use repeated `time_call` samples or a purpose-built
   microbenchmark. Form a roofline/resource hypothesis and inspect the relevant
   generated code or hardware counters.
7. **Change the first owning layer.** Avoid compensating downstream for a bad
   fusion decision or changing scheduling to hide a renderer bug.
8. **Return upward.** Revalidate the isolated call, execution plan, full model,
   compile time, correctness, and at least one relevant alternate backend.

Apply Amdahl's law before investing in a hotspot:

```text
maximum total speedup = 1 / ((1 - f) + f / s)
```

If a kernel is 10% of latency and becomes twice as fast, the total speedup is
only about `1.053x`. If the model improves more than attribution permits, the
change affected another layer—or the benchmark states differ.

### Common attribution traps

| Observation | Do not conclude | Next evidence |
| --- | --- | --- |
| Unsynchronized Python call is faster | Device execution improved | Synchronize the same endpoint; compare host/device timeline |
| `DEBUG=2` kernel sum fell | End-to-end latency fell | Clean synchronized wall-time distribution |
| Kernel count fell | Scheduler improved | Compare traffic, individual kernels, critical path, and full latency |
| `Estimates.mem` fell | DRAM traffic fell | Hardware traffic evidence or a controlled bandwidth response |
| Isolated kernel doubled | Model doubled | Kernel's critical-path fraction and full-model result |
| Occupancy increased | Kernel improved | Device time, stalls, resource use, spills |
| First run fell | Compiler became faster | Fresh-process stage breakdown with identical cache/search policy |
| Minimum sample fell | Typical latency fell | Raw samples and robust distribution across repeated runs |

## A reviewable comparison protocol

Capture this table for baseline and candidate. Change one variable at a time
unless the contribution is explicitly a coupled design.

| Category | Record and hold constant |
| --- | --- |
| Revision | Commit, dirty diff, branch, build/toolchain state |
| Machine | Device, CPU, memory, power/clock state, driver and OS |
| tinygrad route | `DEV`, backend, renderer, target, `BEAM`, `JIT`, `TC`, optimization and validation flags |
| Workload | Script/test, input source, shape, dtype, model mode, batch/sequence length, seed |
| Correctness | Oracle, tolerance, expected side effects, result before and after timing |
| Compile state | Fresh/warm process, `CACHEDB`, `CCACHE`, `SCACHE`, search and capture warm-up |
| Data-cache state | Hot, naturally evolving, or deliberately disturbed; exact mechanism |
| Timing | Host or device clock, synchronization endpoints, included setup/copies/output |
| Sampling | Warm-ups, sample count, raw values, median/p10/p90/min, run order, noise floor |
| Pipeline evidence | Schedule/call count, static estimates, affected AST/source, launch and resource changes |
| Outcome | Compile, steady model, submission, and affected-kernel deltas—without relabeling one as another |

A speedup still has a cost. State new rules, branches, lines, cache states,
backend-specific behavior, compile-time impact, and maintenance burden. A
marginal gain inside noise does not justify a harder optimizer. A larger gain
may still be wrong if it narrows semantics, regresses other backends, or makes
future reasoning substantially harder.

## Question-led source stops

Do not open a large module and attempt to understand every declaration.  Each
stop below has one question, a bounded range, and a translation.  All links
target the pinned snapshot.

### Stop 1: what are the four project performance categories?

Read [`docs/developer/speed.md` lines 5–26](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/developer/speed.md#L5-L26).

Question: which layer owns compilation Python, driver execution, model grouping,
and kernel code generation?

Translation: these categories motivate this chapter's four-clock model.  The
note's statement that execution speed is “almost never” the bottleneck is a
project perspective, not evidence about your workload.  Measure before
excluding submission or graph behavior.

Then read [lines 45–57](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/developer/speed.md#L45-L57).
Identify the connection it draws among OptOps, arithmetic intensity, register
pressure, occupancy, and spilling.  Treat the RTX 4090 rates in that prose as
rough context from the note, not a calibrated roof for your current clocks and
instruction path.

### Stop 2: what exactly do `ops`, `lds`, and `mem` count?

Read [`Estimates` lines 10–21](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L10-L21),
then its counting loop at
[lines 32–57](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L32-L57).

Question: which field accumulates non-register-addressed modeled loads/stores,
and which follows only `PARAM`-rooted addresses before capping repeated accesses
by parameter-buffer size?

Translation: `lds` is repeated program-visible load/store bytes except
register-addressed access; `mem` follows only address chains that reach a
`PARAM` and caps bytes per `(parameter buffer, LOAD/STORE role)` at that
buffer's extent.  Local/shared/register-rooted traffic is absent from `mem`.
`ops` counts modeled ALU/WMMA work over ranges and is not dtype-filtered to
floating arithmetic.  None reads a performance counter.  Notice the
implementation choices—optionally excluded index arithmetic and special
treatment of `MULACC`/`WMMA`—before interpreting a derived “FLOP/s” rate.

### Stop 3: why does `DEBUG=2` alter the run?

Read [`track_stats` lines 51–84](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L51-L84),
then [`run_linear` lines 277–281](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L277-L281).

Question: when does `track_stats` synchronize, and what value is placed in the
execution context when `DEBUG >= 2`?

Translation: the diagnostic requests waiting.  If the runtime supplies no
duration, it synchronizes and uses host elapsed time.  Its printed operation and
bandwidth rates divide static estimates by that duration.  Use this for
attribution, not as an observationally invisible final benchmark.

### Stop 4: what interval does `time_call` construct?

Read [`time_call` lines 283–291](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L283-L291)
alongside the bounded
[`get_runtime` cache lookup/insertion](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L110-L120).

Question: what happens to BEAM, `wait`, cache insertion, linking, and multiple
linked calls?

Translation: it builds a one-call `LINEAR`, compiles with `beam=0`, links with
its execution cache flag false, waits, and returns the maximum reported
duration across the linked calls.  A runtime object already present in the
global cache can still be found; the false cache flag prevents inserting a
missing one.  The bundled lab primes that path explicitly and controls one
call.

Also inspect the two branches at lines 284–288 before using `clear_l2=True`.
One calls a backend-specific invalidation method; the fallback executes a large
Tensor.  Neither is a portable named-cache-state guarantee.

### Stop 5: where does an unsynchronized timer stop too early?

Read the complete
[`external_benchmark_kernel_launch.py` lines 7–38](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/external_benchmark_kernel_launch.py#L7-L38).

Question: compare the timestamp positions in `nosync` and `precise`.  Why can
`GlobalCounters.time_sum_s` remain zero before the `DEBUG=2` block?

Translation: the script is a compact demonstration that host return,
synchronized completion, and tracked kernel duration are different intervals.
It also checks the arithmetic output.  Run it as a mechanics lesson, not as a
portable launch-overhead number.

### Stop 6: how can cold Python work be decomposed?

Read
[`external_benchmark_schedule.py` lines 19–44](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/external_benchmark_schedule.py#L19-L44).
Then inspect the tiny
[`Timing` context manager](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L307-L313).

Question: which nested regions separately time model Tensor construction,
scheduling, rewriting, linearization, and verification?

Translation: one opaque “compile” timer hides the first owning stage.  Adapt
the nested-region idea to a minimized workload, while remembering that the
`Timing` helper is a host timer and does not synchronize device work by itself.

### Stop 7: how are VIZ captures consumed after the workload exits?

Read the pinned
[`tinygrad/viz/README.md` lines 1–24](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md#L1-L24).

Question: which command reads the saved capture, and how does `DEBUG` change the
view shown by the CLI?

Translation: capture with `VIZ` set before imports, then use CLI options and
display-time `DEBUG` to inspect the saved rewrite/runtime data.  Instrumented
runs localize the clean benchmark; they do not replace it.

## Lab — Build a model-to-kernel performance dossier

This lab has three parts. The first two teach the instruments; the third
applies the attribution funnel to a representative workload.

### Part 1: expose asynchronous timing

**Accelerator.** Run the snapshot's launch benchmark:

```bash
DEV=CUDA DEBUG=0 JIT=1 BEAM=0 NOOPT=0 TC=0 TC_OPT=0 TC_SELECT=-1 \
  ALLOW_TF32=0 NOLOCALS=0 MAX_KERNEL_BUFFERS=0 SCACHE=0 CCACHE=0 \
  CACHELEVEL=0 PROFILE=0 VIZ=0 TRACK_MATCH_STATS=0 \
  .venv/bin/python test/external/external_benchmark_kernel_launch.py
```

Before running it, predict which interval omits device completion. Explain the
difference among `nosync`, `precise`, `GlobalCounters.time_sum_s`, and full
wall time. The exact numbers are not portable; the durable result is that the
location of synchronization changes what the timer means.  These settings pin
the JIT and main compiler/search controls needed by this teaching run; this is
still an upstream diagnostic script, not a sealed benchmark harness.  Record
any additional tinygrad variables exported by your shell before treating two
runs as comparable.

### Part 2: inspect estimates and time one call

**Portable mechanics; accelerator follow-up recommended.**  The bundled
`labs/phase5/performance_walk.py` turns the protocol into checked code.  From
the guide repository root, run:

```bash
PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python \
  labs/phase5/performance_walk.py --samples 15
```

The script requires a nonempty explicit `DEV` before importing tinygrad.  It
rejects bare `NV`, whose pinned interface selection can fall through from the
driver interface to direct PCI, and rejects explicit `PCI` routes.  Use a
driver-backed spelling such as `CUDA` or `NVK+NV` for NVIDIA hardware.  Other
explicit backends remain available.  `CACHELEVEL=0` and `CCACHE=0` make
`CACHEDB` deliberately inert in this lab, so the commands omit a database path
instead of implying persistent-cache evidence.

The printed `controlled env` line is generated from the same mapping the lab
applies before importing tinygrad.  Besides the familiar JIT/cache/debug flags,
it fixes reduction splitting, matvec heuristics, occupancy floor, late
coalescing (`DMC`), SSA expansion/alignment, default and accumulation dtypes,
local/buffer limits, rewrite implementation/tracking, and validation.  This
also disables the optional NVIDIA ioctl/PMA hooks, preventing inherited
instrumentation from changing a physical CUDA or NV timing route.  Together,
these controls prevent an inherited experimental knob from silently changing
the program or timing protocol.  The lab deliberately does not claim to pin the OS, Python build,
compiler/driver/library versions, power state, or toolchain paths; those remain
fields in a real benchmark's environment card.

The lab uses two fixed `8 × 8` matrices whose entries are multiples of `1/8`.
Their products and short sums are exactly representable in `float32`, so an
ordinary nested-loop matmul/ReLU is an exact independent oracle.  The small
shape is chosen to keep the interpreter route quick, not to represent a useful
GPU workload.

It then performs these steps explicitly:

1. realizes the two input buffers and synchronizes;
2. measures lazy Tensor expression construction while asserting that the
   output is still unrealized;
3. measures scheduling and requires exactly one controlled `SINK` call;
4. compiles that call with `BEAM=0` and records its static estimates;
5. executes once through the ordinary runtime-cache path;
6. checks every output against the independent oracle;
7. warms three completed runs outside the sample region;
8. collects raw wall samples with synchronization before and after every run;
9. checks the oracle immediately after that completed-wall family;
10. collects separate internal `time_call` samples;
11. checks the oracle again after `time_call`; and
12. reports min, p10, median, p90, max, and every raw sample without imposing a
    speed threshold.

No timing assertion is intentional.  CI load, CPU frequency, drivers, and
hardware vary.  The executable contracts are that the intended call ran,
timing values were well formed, timer boundaries included completion, and
correctness survived.  A future performance contribution would compare two
revisions under the same protocol and apply an acceptance/noise criterion
chosen for that experiment.

Read the two distributions as different intervals:

- **completed-wall samples** drain the device, start the host timer, submit the
  prepared call, wait for device completion, and stop the host timer.  They
  include host dispatch and synchronization around the completed call.
- **`time_call` samples** use tinygrad's internal one-call diagnostic with
  `wait=True`.  The lab first populates the normal runtime-object cache, so the
  diagnostic can find that loaded runtime even though its context will not
  insert a missing one.  On this one-call plan, its maximum-over-linked-calls
  result has one term.

Those values need not match.  On a physical device, runtime-reported device
duration can be much smaller than synchronized host wall time for a tiny
kernel.  On the synchronous Python interpreter they can be much closer.  The
difference is an attribution clue, not automatically launch overhead: inspect
the backend's timing implementation and the exact calls between timer endpoints
before naming the gap.

Run the same mechanics on the two recorded RTX 4090 routes in fresh processes:

```bash
PYTHONPATH=../tinygrad-study DEV=CUDA \
  ../tinygrad-study/.venv/bin/python \
  labs/phase5/performance_walk.py --samples 15

PYTHONPATH=../tinygrad-study DEV=NVK+NV \
  ../tinygrad-study/.venv/bin/python \
  labs/phase5/performance_walk.py --samples 15
```

Record the requested and canonical device, backend/interface class, concrete
renderer/compiler/runtime classes, renderer target, SOURCE/BINARY sizes and
hash prefixes, launch dimensions, static estimates, and both raw
distributions.  Those fields distinguish, for example, CUDA C/NVRTC from a PTX
compiler even when a target string alone would look similar.  Do not compare
these tiny-kernel numbers as a verdict on the overall backends.  The routes
have different runtime/submission implementations, and the experiment is
dominated by a shape selected for teaching mechanics.

As a separate source exercise, inspect `time_call(..., clear_l2=True)` on the
pinned checkout.  If the device exposes `invalidate_caches`, that method is
used.  Otherwise the fallback realizes a displacement Tensor.  Neither branch
is a universal guarantee that every relevant hardware cache began in a named
state.  Do not enable it in a comparison until you can state what the selected
backend actually did.

### Part 3: attribute one representative step

Choose a real inference or training step relevant to a current contribution.
Keep the scope to one stable input shape initially.

1. Write the falsifiable claim and correctness oracle in the
   [contribution brief](../reference/contribution-brief.md).
2. In a fresh process, time first run and split Tensor construction,
   `schedule_linear`, rewrite/compile, and execution as far as the workload
   allows. Record all cache/search flags.
3. Finish compilation and `TinyJit` capture, then collect at least 20 clean,
   synchronized steady-state wall samples. Preserve the raw data and report
   median, p10, and p90.
4. In a separate diagnostic run, use `DEBUG=2`. Record call/kernel count,
   device-time total, copies, and the top calls. Do not reuse this run as the
   clean result.
5. Run the workload with `VIZ=1 PROFILE=1 TRACK_MATCH_STATS=2`, placing
   `profile_marker("steady start")` and `profile_marker("steady end")` around
   several replayed steps. Inspect the interval and aggregate views:

   ```bash
   .venv/bin/python -m tinygrad.viz.cli | rg MARKER
   .venv/bin/python -m tinygrad.viz.cli --interval "steady start" "steady end"
   .venv/bin/python -m tinygrad.viz.cli -t
   ```

6. Pick the call or host gap that can explain the largest portion of the
   end-to-end result. Save its AST, generated source, launch dimensions,
   estimates, and critical-path fraction. If it is a kernel, repeat Part 2 on
   that real call and form a roofline/resource hypothesis.
7. Make no source change yet. Predict the maximum end-to-end win with Amdahl's
   law and name the source layer that owns the first costly decision.

Your dossier is complete only if a second contributor can distinguish compile,
model/scheduler, kernel, and submission costs from the evidence you saved.

## Exercises

Write an answer before opening the answer section.

### 1. Name the interval

A CUDA function is called between two `perf_counter()` reads.  The second read
occurs before any synchronization.  The returned number falls from `20 µs` to
`8 µs`, but a device event around the kernel remains `50 µs`.

What improved, and what did not?

### 2. Bound a local optimization

One kernel is 12% of synchronized model latency.  You expect to make it three
times as fast.  Under the simple non-overlapping Amdahl model, what total
speedup is possible?  What is the upper bound if the kernel became free?

### 3. Form an intensity hypothesis

Two equivalent elementwise schedules have the same modeled operations.  The
candidate eliminates one full-size intermediate write and later read.  Which
roofline quantity changes, and what evidence is still needed before claiming
DRAM traffic fell?

### 4. Diagnose a first-run-only win

Baseline and candidate steady-state distributions are indistinguishable, but
the candidate's first result arrives 30% sooner.  List at least four stages or
states to separate before calling it a compiler speedup.

### 5. Interpret `DEBUG=2`

With `DEBUG=0`, a TinyJit replay submits several calls asynchronously.  With
`DEBUG=2`, the printed sum is stable and the failure disappears.  Can that
output serve as the final performance result?  What does disappearance of the
failure suggest?

### 6. Read two distributions

The baseline synchronized samples have median `100 µs`, p10 `97 µs`, and p90
`105 µs`.  The candidate has median `99 µs`, p10 `96 µs`, and p90 `105 µs`.
What should you report, and what experiment should precede a speedup claim?

### 7. Separate a local win from a model win

An isolated kernel becomes `2×` faster, but model latency rises by 3%.  Name
five pipeline facts to compare before reverting or defending the change.

### 8. Decide what a performance CI test should assert

A rewrite removes a redundant device copy deterministically, producing a large
local speedup.  Shared CI wall time is noisy.  What should the focused automated
test assert, and what performance evidence belongs in the contribution?

## Exercise answers

### 1. Name the interval

Only the observed host interval—most likely preparation/submission—has evidence
of improvement.  The device event says the kernel execution duration did not
improve.  A synchronized end-to-end distribution is still required to learn
whether the user-visible operation improved; the shorter submission could
overlap device work or could merely move a wait elsewhere.

### 2. Bound a local optimization

Here `f = 0.12` and `s = 3`:

```text
1 / ((1 - 0.12) + 0.12/3)
= 1 / (0.88 + 0.04)
≈ 1.087×
```

The simplified maximum is about an 8.7% speedup.  If the kernel became free,
the bound would be `1 / 0.88 ≈ 1.136×`, about 13.6%.  Changed overlap or fusion
can invalidate the simple fractions, so remeasure the full timeline.

### 3. Form an intensity hypothesis

The candidate reduces algorithmic/program-visible bytes while operations stay
constant, so modeled arithmetic intensity increases.  `Estimates.lds` and
possibly `mem` can describe the compiler model.  To claim DRAM traffic fell,
inspect cache/working-set conditions and obtain an appropriate hardware counter
or a controlled response that isolates DRAM bandwidth.  The eliminated traffic
might previously have been served from cache.

### 4. Diagnose a first-run-only win

Separate at least Tensor/model construction, scheduling/rewrite time, BEAM or
local-size search, source/binary compilation, persistent compiler-cache state,
device/context initialization, allocation/page mapping, and first execution.
Use fresh processes and dedicated cache files for both revisions.  Name the
region that moved before calling it a compiler, scheduler, or initialization
speedup.

### 5. Interpret `DEBUG=2`

No.  At the pinned snapshot, `DEBUG>=2` requests waits and may synchronize when
a runtime supplies no duration; printing also perturbs tiny calls.  Use it to
localize.  A disappearing failure suggests a queue dependency, resource
lifetime, reuse, or race condition masked by the added waits.  Compare explicit
wait/no-wait boundaries and inspect dependencies rather than leaving debug mode
enabled as the fix.

### 6. Read two distributions

Report all summaries and raw samples, including the heavily overlapping p10–p90
ranges.  The median delta is `-1 µs` or about `-1%`, but the current evidence
does not distinguish it from natural variation.  First run baseline-versus-
baseline under the same harness, then collect balanced/interleaved blocks sized
to resolve a predeclared meaningful effect.  Do not select the candidate's best
sample.

### 7. Separate a local win from a model win

Compare schedule/call count, fusion and realization boundaries, intermediate
and copy traffic, launch dimensions/resource use, critical-path placement,
host submission/waits, compile/search cost, JIT graph grouping, and correctness.
The isolated kernel may have changed surrounding work or may not be the same
call/state measured in the model.  Preserve both results until the attribution
explains the contradiction.

### 8. Decide what a performance CI test should assert

Assert the semantic result and the durable structural contract: for example,
the relevant `LINEAR` contains one fewer copy or no call with the forbidden
source/destination relationship.  Add nearby negative cases so the rewrite does
not remove necessary copies.  Put synchronized raw baseline/candidate
distributions, environment, and end-to-end impact in the contribution evidence.
A noisy shared-runner microsecond threshold is a weaker detector for this
deterministic regression.

## Checkpoint

Continue when you can:

- state a falsifiable speed claim with workload, layer, metric, environment,
  and rejection criterion;
- produce a correctness-bracketed benchmark with explicit warm-up, cache,
  synchronization, and distribution policy;
- explain what `DEBUG=2`, `Estimates`, `time_call`, and VIZ do—and what each
  cannot establish;
- trace an end-to-end latency result to a schedule decision, kernel, or
  submission gap that can account for it;
- use roofline and occupancy as measured hypotheses rather than labels; and
- report full-model, compile, and complexity tradeoffs alongside a local win.

## Quick reference

```text
claim = workload + layer + metric + environment + falsifier

correctness before and after timing
cold compile: fresh process + explicit cache/search policy
steady run: finish allocation/compile/JIT warm-up, then synchronize
latency: drain -> start -> submit -> synchronize -> stop
report: raw samples + n + median + p10/p90 (+ min when useful)

DEBUG=2: per-call diagnostic; waits and can perturb execution
Estimates: static modeled ALU work/repeated accesses/capped buffer-role bytes; not counters
time_call: internal isolated CALL timer, beam=0, forced wait
VIZ/PROFILE: attribution capture; not final benchmark

roofline: rate <= min(peak compute, bandwidth * arithmetic intensity)
occupancy: resource/latency-hiding constraint, not a performance goal

model -> timeline -> execution plan -> owning call/gap -> isolate
      -> change first wrong/costly layer -> full-model validation
```

[← Testing a contribution](16-testing.md) · [Next: From idea or bounty to a reviewable contribution →](18-contributing.md)
