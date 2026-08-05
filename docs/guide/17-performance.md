# 17. Performance engineering

## Purpose

A performance contribution is not an interesting generated kernel followed by
a smaller number. It is a falsifiable claim about a named workload, layer,
metric, and environment, supported by a correctness oracle and a comparison
that another contributor can reproduce.

This chapter unifies the four meanings of speed used by tinygrad: compile
speed, model/scheduler speed, kernel speed, and execution/submission speed. It
gives you a route from an end-to-end regression to the artifact that owns it,
then back to the end-to-end workload after a change. The kernel mechanics build
on [Chapter 9](09-kernel-optimization.md); this chapter is about deciding
whether a kernel is actually the right thing to optimize and proving that the
result matters.

**Verified against tinygrad:** `874d331` (2026-08-05).

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

The counters are process-global. `GlobalCounters.reset()` clears operations,
estimated memory, elapsed time, and call count, but deliberately does not reset
`mem_used`; treat allocation state separately when comparing runs.

Use `DEBUG=2` for attribution. Re-run the final comparison with `DEBUG=0` and
explicit synchronization around the boundary you actually claim.

### `Estimates`: static work, not hardware counters

The compiled program carries an `Estimates` value with three fields:

- `ops`: statically counted floating-point operations;
- `lds`: bytes represented by loads and stores, including repeated accesses;
  and
- `mem`: unique buffer bytes, capping repeated accesses to a buffer.

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

`VIZ=1` records rewrite and runtime profile events and implies `PROFILE`. The
snapshot CLI can inspect the latest capture:

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

`Estimates.mem` can approximate unique algorithmic bytes and `Estimates.lds`
can approximate repeated program-visible accesses. Neither establishes traffic
at a hardware memory level. Treat “compute-bound” or “memory-bound” as a
hypothesis: check achieved rates, source, and—when the decision depends on
it—hardware counters or controlled perturbations.

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

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| tinygrad's four categories of speed | [`docs/developer/speed.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/developer/speed.md#L1) |
| Static operation and memory estimates | [`Estimates`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L10) |
| Per-call timing, counters, and `DEBUG=2` output | [`estimate_uop` and `track_stats`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L41) |
| Compile/run boundary and isolated timing | [`run_linear` and `time_call`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L268) |
| Cache/profile context, global counters, host timers, and markers | [`helpers.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L265) |
| Rewrite and runtime profile CLI | [`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md#L1) |
| Synchronized versus unsynchronized launch timing | [`external_benchmark_kernel_launch.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/external_benchmark_kernel_launch.py#L1) |
| Python compile-stage decomposition | [`external_benchmark_schedule.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/external_benchmark_schedule.py#L9) |
| A performance-test example with explicit warm-up and static work | [`speed_v_theoretical.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/speed_v_theoretical.py#L35) |

The speed note is a compressed project perspective, not a substitute for
measurement. In particular, never assume execution/submission is irrelevant to
your workload merely because it is often small upstream.

## Lab — Build a model-to-kernel performance dossier

This lab has three parts. The first two teach the instruments; the third
applies the attribution funnel to a representative workload.

### Part 1: expose asynchronous timing

**Accelerator.** Run the snapshot's launch benchmark:

```bash
DEV=CUDA DEBUG=0 CACHEDB=/tmp/tinygrad-guide-perf.db \
  .venv/bin/python test/external/external_benchmark_kernel_launch.py
```

Before running it, predict which interval omits device completion. Explain the
difference among `nosync`, `precise`, `GlobalCounters.time_sum_s`, and full
wall time. The exact numbers are not portable; the durable result is that the
location of synchronization changes what the timer means.

### Part 2: inspect estimates and time one call

**Portable mechanics; accelerator timing recommended.** This uses internal
snapshot APIs deliberately. It schedules a small matmul/relu, compiles one
kernel to expose its estimates, executes the original scheduled call repeatedly,
and validates the output:

```bash
DEV=CUDA DEBUG=0 CACHEDB=/tmp/tinygrad-guide-perf.db .venv/bin/python - <<'PY'
import statistics
from tinygrad import Tensor
from tinygrad.engine.realize import compile_linear, estimate_uop, run_linear, time_call
from tinygrad.uop.ops import Ops, UOp

n = 64
values = [[float(i*n+j)/n for j in range(n)] for i in range(n)]
a, b = Tensor(values).realize(), Tensor.eye(n).realize()
out = (a @ b).relu()
linear = out.schedule_linear()
call = next(c for c in linear.src if c.src[0].op is Ops.SINK)
compiled = compile_linear(UOp(Ops.LINEAR, src=(call,)), beam=0)
compiled_call = compiled.src[0]
print("static estimates:", estimate_uop(compiled_call))

# Execute through the normal cached path once so each timing sample reuses the
# same loaded runtime Program instead of constructing a module outside the timer.
run_linear(compiled, jit=True)
for _ in range(3): time_call(compiled_call)
hot = [time_call(compiled_call) for _ in range(20)]
disturbed = [time_call(compiled_call, clear_l2=True) for _ in range(20)]
print("hot median/min/max (us):",
      *(x*1e6 for x in (statistics.median(hot), min(hot), max(hot))))
print("clear_l2-path median/min/max (us):",
      *(x*1e6 for x in (statistics.median(disturbed), min(disturbed), max(disturbed))))
assert out.tolist() == values
PY
```

This tiny kernel teaches mechanics; it is not evidence for a useful speedup.
Inspect `time_call` on your checkout and describe exactly what its
`clear_l2=True` path did on your backend. Explain why `ops`, `lds`, and `mem`
cannot tell you the observed DRAM bytes.

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
5. Run the workload with `VIZ=1`, placing `profile_marker("steady start")` and
   `profile_marker("steady end")` around several replayed steps. Inspect the
   interval and aggregate views:

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
Estimates: static ops/loads-stores/unique bytes; not hardware counters
time_call: internal isolated CALL timer, beam=0, forced wait
VIZ/PROFILE: attribution capture; not final benchmark

roofline: rate <= min(peak compute, bandwidth * arithmetic intensity)
occupancy: resource/latency-hiding constraint, not a performance goal

model -> timeline -> execution plan -> owning call/gap -> isolate
      -> change first wrong/costly layer -> full-model validation
```
