# 9. Kernel optimization

## Purpose

Kernel optimization reorganizes equivalent work inside one scheduled kernel.
It can change launch axes, per-thread work, reduction structure, memory reuse,
vector forms, and tensor-core use without changing the tensor result beyond the
project's allowed numerical tolerance.

This chapter teaches you to locate the right optimization layer, read and apply
tinygrad's kernel options, reason about their hardware costs, and produce
performance evidence that is credible on an RTX 4090.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

Before continuing, you should be able to explain:

- CUDA grids, thread blocks, warps, lanes, and global/shared/register memory;
- coalesced global-memory access;
- latency versus throughput and compute-bound versus bandwidth-bound work; and
- why registers and shared memory can limit resident warps.

If not, take the bounded route through the
[CUDA Programming Guide and Ada Tuning Guide](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path),
then read the measurement and occupancy portions of the
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
Return when you can annotate one tiled matrix multiplication with global loads,
shared-memory reuse, synchronization, and per-thread accumulators. You do not
need to write CUDA before reading this chapter.

## First decide: model problem or kernel problem

A fast kernel cannot compensate for unnecessary kernels around it. Keep these
optimization layers separate:

| Layer | Typical controls | Evidence to collect |
| --- | --- | --- |
| Model/execution plan | Fusion, materialization, copies, kernel count, memory planning, JIT/graph replay | `LINEAR` calls, tensor sizes, allocated memory, model trace, end-to-end time |
| One kernel | Axis types, local/grouping choices, upcast, unroll, tensor cores, address pattern | Applied opts, launch dimensions, generated code, static estimates, kernel timings, profiler counters |

Start with an end-to-end profile. Select a kernel only after showing it occupies
a meaningful fraction of the target workload. If a change makes that kernel
20% faster but it accounts for 2% of model time, the maximum model improvement
is about 0.4% before secondary effects.

Conversely, if the plan has the right boundaries but one large kernel dominates,
changing fusion is a much larger and riskier intervention than changing its
kernel schedule.

## Mental model: factor ranges into hardware roles

After rangeification, a kernel is a `SINK` containing explicit ranges and
indices. Codegen's `Scheduler` holds that kernel plus the selected `Renderer`.
The renderer supplies legality and limits: local/thread support, maximum launch
sizes, shared-memory capacity, vector support, and tensor-core descriptions.

The scheduler classifies ranges with `AxisType`s such as `GLOBAL`, `LOCAL`,
`REDUCE`, `GROUP_REDUCE`, `UPCAST`, `UNROLL`, and `WARP`. An optimization usually
factors one range:

```text
old range of size N
        ↓ split by amount A
remaining range of size N/A + new range of size A with a new AxisType
```

For example, `LOCAL` moves a factor into a workgroup dimension, while `UPCAST`
moves a factor into work performed by each work item. `Scheduler.apply_opt`
checks divisibility, target capabilities, resource limits, and
operation-specific invariants before mutating its kernel AST and recording the
`Opt`.

The `axis` in an `Opt` never means “the original Tensor axis,” and its current
namespace is option-dependent. Most options index the scheduler's current
`rngs`; `UNROLL` indexes the current unrollable dimensions; `GROUP` and
`GROUPTOP` index the current reduction-axis list; and `TC` indexes candidate
M/N/K triples constructed by tensor-core matching. Previous opts can split or
remove members of all these namespaces. Read `real_axis` and the selected
option's handler, and print `shape_str()`, `full_shape`, and `axis_types` after
each step instead of carrying an old number in your head. Copy the scheduler
before exploring alternatives because `apply_opt` mutates it.

## The `OptOps` vocabulary

| Option | Current effect | Main opportunity | Main risk |
| --- | --- | --- | --- |
| `TC` | Matches a matrix multiply-accumulate reduction and maps selected M/N/K factors to a target `WMMA` form. Must be the first recorded opt. | Tensor-core throughput | Shape/dtype/layout mismatch, padding work, altered floating-point association |
| `UPCAST` | Moves a factor of a global/local/weak range into `UPCAST` work, at most 16 outside DSP. | Vectorization, per-thread reuse, fewer instructions | Registers, code size, too few work items |
| `UNROLL` | Moves a factor of a reduce/group-reduce range into `UNROLL`, at most 32. | Less loop overhead, more instruction-level parallelism | Registers, code size, instruction-cache pressure |
| `LOCAL` | Splits a global/weak range into a workgroup-local range. | Cooperative parallelism and address layout | Invalid launch shape, reduced occupancy |
| `THREAD` | Splits a globalizable weak range for targets exposing a thread dimension. | Host/target parallelism | Oversubscription or target-specific constraints |
| `GROUP` / `GROUPTOP` | Moves a reduction factor into `GROUP_REDUCE`; the two choose which side of the factorization the new range occupies. | Cooperative reduction with local storage | Shared memory, barriers, occupancy, unsupported nested reductions |
| `NOLOCALS` | Forbids local indexing when no local/warp/group ranges already exist. | Simpler launch/index path on some kernels | Loses locality/cooperation |
| `PADTO` | Rounds a constant range upward and adds validity so padded iterations do not change results. | Enables divisible tiles or tensor-core shapes | Extra work and masked memory operations |
| `SWAP` | Exchanges identifiers/order for two global ranges. | Better mapping between logical axes and launch/address order | Worse coalescing or launch geometry |

An argument of zero for factor-moving options means the full eligible range.
`apply_opt` rejects illegal combinations with `KernelOptError`; a failed option
is not evidence that the optimization idea is useless, only that this exact AST,
axis, amount, and target do not satisfy its contract.

## How `apply_opts` chooses a schedule

`apply_opts` is the entry point used by codegen. Its priority is important:

1. a tagged/already optimized AST is returned;
2. explicit `KernelInfo.opts_to_apply` options are replayed in order;
3. if `beam >= 1`, BEAM search chooses options;
4. otherwise, if default optimization is enabled and the kernel is eligible,
   `hand_coded_optimizations` chooses options; and
5. the scheduler emits an optimized AST with `KernelInfo.applied_opts`.

This happens after early range simplification and before expansion, reduction
lowering, local-buffer insertion, and GPU-dimension lowering. The options are a
compact recipe; later passes make their effects explicit.

### Heuristics

The hand-coded path is deterministic source code. In this snapshot it tries
tensor cores first, has a matrix-vector special case, may group reductions,
upcasts masked or reuse-friendly axes, unrolls small reductions, and selects
local/thread factors within target limits.

Heuristics are fast and predictable, but encode assumptions about common
shapes and hardware. A heuristic change must be evaluated over a workload set,
not only the motivating kernel.

### BEAM search

BEAM starts from a `Scheduler`, generates legal next schedules with
`get_kernel_actions`, compiles candidates, times them on the selected target,
and retains the best `amt` candidates for another round. It deduplicates
identical generated binaries and caches the chosen option sequence.

The default `BEAM_ESTIMATE=1` permits testing a reduced global size and scaling
the measured duration. That makes search cheaper, but the scaled number is a
proxy: cache behavior, occupancy, and launch overhead may differ at full size.
Always benchmark the winning full-size kernel independently.

BEAM is measured search, not a correctness oracle. Every selected schedule
still needs semantic tests, and search noise can select a different winner when
clocks, cache state, driver, or workload shape changes.

## Relate each option to hardware cost

### Coalescing and upcast

On a GPU, neighboring lanes should usually access neighboring addresses. Range
order and factorization determine whether that is possible. `UPCAST` can expose
adjacent scalar accesses as vector-shaped work and improve reuse, but it can
also assign too much work to one lane.

Coalescing itself is a later codegen pass: `memory_coalescing` groups compatible
adjacent loads/stores after expansion and devectorization. There is no
`OptOps.COALESCE`. If coalescing fails, inspect both the scheduler's chosen
axis/address relationship and the late pass's grouping preconditions.

### Local memory and grouped reductions

`GROUP`/`GROUPTOP` create a cooperative reduction range. Later codegen can add
local buffers and barriers so a workgroup combines partial results. This can
replace repeated global traffic with on-chip reuse.

The resource is finite. More shared memory per block can reduce resident blocks,
and synchronization adds latency. `apply_opt` estimates grouped-reduction
shared storage and rejects choices above `renderer.shared_max`, but being legal
does not make a choice fast.

### Unroll and registers

Unrolling removes loop-control work and exposes independent operations, but it
duplicates the body and lengthens live ranges. Upcast has a similar register
tradeoff because each work item carries more values or accumulators. Register
pressure can reduce occupancy or cause spills; inspect compiler resource data
or profiler counters rather than inferring registers only from UOp count.

### Tensor cores

`TC` looks for an additive reduction whose value is a multiplication (possibly
through accepted casts), then matches target tensor-core input/output dtypes,
M/N/K ranges, and layout. It may factor warp/local/upcast/unroll ranges and
replace the reduction with `WMMA`. With the appropriate optimization level,
non-divisible axes may first receive `PADTO`.

On CUDA/NV in this snapshot, float-input tensor-core matching is gated by
`ALLOW_TF32`; half-precision paths do not require enabling TF32. A `sm_89`
target advertises Ada tensor-core descriptions, but a 4090 alone does not make
an arbitrary reduction eligible.

Tensor cores change accumulation order and sometimes precision. Validate with
a tolerance justified by dtype and workload, including adversarial magnitudes;
do not weaken a global tolerance merely to make one optimization pass.

## Roofline and occupancy are constraints, not scores

For a first bound, compute arithmetic intensity:

```text
arithmetic intensity = useful operations / bytes moved
attainable performance ≤ min(peak compute,
                             memory bandwidth * arithmetic intensity)
```

If the kernel is bandwidth-bound, a change that only reduces ALU instructions
may be invisible. Fusion, reuse, coalescing, or fewer bytes are better targets.
If it is compute-bound, tensor cores, instruction mix, and parallel utilization
matter more.

`Estimates.from_uops` records:

- `ops`: statically counted arithmetic work;
- `lds`: statically counted load/store bytes, including repeated accesses; and
- `mem`: unique/capped buffer bytes in its model.

These are compiler estimates, not L2/DRAM transactions or hardware counters.
Use them to form hypotheses and detect large structural changes. Use
[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
when a measured kernel justifies collecting achieved bandwidth, tensor-pipe
use, warp stalls, registers, shared memory, and occupancy.

Occupancy means resident warps relative to a hardware limit. Low occupancy can
fail to hide latency, but maximum occupancy is not the objective. A schedule
using more registers or shared memory can be faster if it greatly increases
reuse or instruction efficiency. Optimize elapsed time while explaining the
resource tradeoff.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Option enum and immutable `Opt` record | [`OptOps` and `Opt`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/__init__.py#L6) |
| Kernel range model | [`Scheduler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L15) |
| Per-option legality and transformation | [`Scheduler.apply_opt`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L126) |
| Explicit/BEAM/heuristic selection | [`apply_opts`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L339) |
| Default schedule policy | [`hand_coded_optimizations`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/heuristic.py#L8) |
| Candidate generation and measured search | [`beam_search`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/search.py#L114) |
| Late adjacent-memory grouping | [`memory_coalescing`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/coalesce.py#L100) |
| Static operation/memory estimates | [`Estimates`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L10) |
| Optimization's position in lowering | [`full_rewrite_to_sink`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284) |
| Ada-target Python UOp executor | [`PythonRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L205) |

## Lab 1 — Inspect optimization without claiming performance

**Portable.** `DEV=PYTHON::sm_89` uses the Python UOp executor with a CUDA Ada
renderer target. It can test lowering and tensor-core semantics without a GPU;
its runtime says nothing about 4090 performance. It also cannot prove
warp/workgroup concurrency, barrier sufficiency, race freedom, memory ordering,
or hardware resource behavior; those claims require a real accelerator and an
appropriate stress/test oracle.

Run the same 16×16 float matmul with and without the default heuristic:

```bash
for mode in default noopt; do
  if [ "$mode" = noopt ]; then noopt=1; else noopt=0; fi
  MODE="$mode" NOOPT="$noopt" ALLOW_TF32=1 \
    DEV=PYTHON::sm_89 DEBUG=0 CACHELEVEL=0 .venv/bin/python - <<'PY'
import os
from tinygrad import Device, Tensor
from tinygrad.engine.realize import compile_linear
from tinygrad.uop import Ops

a, b = Tensor.empty(16, 16), Tensor.empty(16, 16)
compiled = compile_linear((a @ b).schedule_linear())

print("mode:", os.environ["MODE"])
print("target:", Device[Device.DEFAULT].renderer.target)
for call in compiled.src:
  if call.src[0].op is not Ops.PROGRAM:
    continue
  program = call.src[0]
  kernel = program.src[0]
  print("opts:", kernel.arg.applied_opts)
  print("launch:", program.arg.global_size, program.arg.local_size)
  print("estimates:", kernel.arg.estimates)
  print("WMMA UOps:", sum(u.op is Ops.WMMA for u in program.src[1].src))
PY
done
```

Before running, predict which case contains `TC`, how launch dimensions change,
and whether useful operation count changes. At the snapshot, the default case
uses `TC` plus `UPCAST` and contains `WMMA`; `NOOPT=1` has no applied opts or
`WMMA`. The useful operation count is unchanged, while statically counted load
bytes differ substantially.

### Change and regress

Change all three matrix dimensions from 16 to 17. With the default `TC_OPT=0`,
the tensor core no longer matches. Then repeat with `TC_OPT=2` and predict the
padding tradeoff before observing `TC` again.

Run the relevant correctness regression on the emulated Ada target:

```bash
ALLOW_TF32=1 DEV=PYTHON::sm_89 DEBUG=0 \
  .venv/bin/python -m pytest -q test/opt/test_tensor_cores.py
```

This target is also used by the snapshot's CI for tensor-core tests. Save the
chosen opts and generated representation with the regression. A test that only
checks for `TC` is insufficient; preserve numerical correctness and expected
eligibility/ineligibility cases.

## Lab 2 — Measure one kernel on the RTX 4090

**NVIDIA.** Save the following as `bench_one_kernel.py` in your study notebook.
It realizes inputs before scheduling, compiles once, loads and caches the
runtime program, warms that exact compiled call, measures synchronized device
durations with `time_call`, reports a distribution, and validates the full
result after timing:

```python
import os, statistics
from tinygrad import Device, Tensor, dtypes
from tinygrad.engine.realize import compile_linear, run_linear, time_call
from tinygrad.uop import Ops

n = int(os.getenv("N", "2048"))
# a[i,k] = 1 + i%4 and b[k,j] = 1 + j%4. The nonuniform result has
# the independent closed form n*(1+i%4)*(1+j%4).
base = Tensor.ones(n, n, dtype=dtypes.half).realize()
row = (Tensor.arange(n)%4).reshape(n, 1).expand(n, n).cast(dtypes.half)
a = (base + row).contiguous().realize()
b = (base + row.permute(1, 0)).contiguous().realize()
out = a @ b

linear, var_vals = Tensor.linear_with_vars(out)
compiled = compile_linear(linear)
calls = [call for call in compiled.src if call.src[0].op is Ops.PROGRAM]
assert len(calls) == 1, [call.src[0].op for call in compiled.src]
call, program = calls[0], calls[0].src[0]

target = Device[Device.DEFAULT].renderer.target
assert Device.DEFAULT == "CUDA" and target.arch == "sm_89", (Device.DEFAULT, target)
print("target:", target)
print("opts:", program.src[0].arg.applied_opts)
print("launch:", program.arg.global_size, program.arg.local_size)
print("estimates:", program.src[0].arg.estimates)

# Seed tinygrad's runtime cache with one loaded Program. time_call deliberately
# disables cache insertion for isolated candidates, but reuses an existing hit.
run_linear(compiled, var_vals, jit=True)
for _ in range(int(os.getenv("WARMUP", "10"))):
  time_call(call, var_vals)
samples = [time_call(call, var_vals)
           for _ in range(int(os.getenv("SAMPLES", "50")))]
ordered = sorted(samples)
p90 = ordered[int(0.9*(len(ordered)-1))]
print("us min/median/p90:",
      *(round(x*1e6, 2) for x in
        (ordered[0], statistics.median(ordered), p90)))

expected_axis = (Tensor.arange(n)%4).reshape(n, 1) + 1
expected = expected_axis * expected_axis.permute(1, 0) * n
max_error = (out.float() - expected.float()).abs().max().item()
print("max error:", max_error)
assert max_error == 0.0
```

Run it with an explicit backend and record the environment:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,pstate,temperature.gpu,clocks.sm \
  --format=csv,noheader
DEV=CUDA N=2048 WARMUP=10 SAMPLES=50 DEBUG=0 \
  .venv/bin/python bench_one_kernel.py
```

The assertion deliberately prevents an accidental `NV`, CPU, or non-`sm_89`
run from being labeled a 4090 CUDA result. If your renderer's target spelling
changes on current `master`, print it first and update the assertion explicitly.

Now make one controlled change—`BEAM=2`, an explicit option sequence in a study
branch, or a heuristic edit—and run it in a fresh process. Record:

- tinygrad commit and dirty diff;
- exact `DEV`, renderer target, dtype, shape, and relevant environment variables;
- driver, GPU, power/clock policy, temperature, and competing load;
- applied opts, launch dimensions, generated source/hash, and static estimates;
- warm-up policy, cache policy, every sample, and summary statistic; and
- correctness tolerance and reference.

The analytic input above is a strong smoke test but not a complete numerical
suite. Before changing accumulation or tensor-core behavior, also compare
nonuniform signed inputs and difficult magnitudes against an independent
NumPy/CPU reference with a justified tolerance.

This script answers a warm-cache single-kernel question. If cold-cache behavior
matters, use `clear_l2=True` deliberately and report it as a different
experiment. If compile/search time matters, time `compile_linear` separately;
do not mix it into kernel duration. When comparing BEAM itself, state whether
its option cache was reused or disabled.

Finally rerun the original model/workload. Report a model improvement only if
the end-to-end benchmark, with the same JIT/capture and synchronization policy,
also improves. Use Nsight Compute only after the timing is repeatable and you
have a concrete question such as “did sectors per request improve?” or “did
register pressure reduce active warps?”

## Contribution-shaped workflow

For a kernel-performance change:

1. minimize and save the exact scheduled kernel plus buffers/variables;
2. prove correctness on a portable backend and the affected accelerator;
3. classify it with roofline reasoning and profiler evidence;
4. predict which `Opt` changes which address/resource;
5. compare applied opts, generated code, and repeated timings;
6. test nearby shapes, especially divisibility and symbolic boundaries;
7. run process replay or a representative kernel set to catch heuristic
   regressions; and
8. confirm the target model improves.

Optimize a cause, not a number. A faster result without an explanation is a
lead; a contribution needs a stable test, a workload scope, and evidence that
the change belongs at this layer.

## Checkpoint

Continue when you can:

- distinguish model scheduling from a schedule inside one kernel;
- explain how `Scheduler`, `Opt`, and `apply_opts` interact;
- describe every `OptOps` member and its main resource tradeoff;
- contrast fixed heuristics with measured BEAM search;
- connect coalescing, local memory, upcast, unroll, and tensor cores to indices
  and hardware limits;
- use roofline and occupancy as constraints rather than scores; and
- produce a synchronized, repeatable, correctness-checked `DEV=CUDA` `sm_89`
  comparison whose claims do not rely on Python emulation.

## Quick reference

| Observation | Inspect or measure next |
| --- | --- |
| Model slow, no dominant kernel | Kernel count, fusion/materialization, copies, JIT/graph path |
| One kernel dominates | Applied opts, indices, launch dimensions, generated code, counters |
| No tensor core | Target tensor-core list, reduce/multiply pattern, dtype, M/N/K divisibility, `ALLOW_TF32`, `TC_OPT` |
| High `lds` relative to `mem` | Repeated accesses and reuse opportunity; verify actual traffic with counters |
| Adjacent lanes issue scattered transactions | Global-axis/address mapping, upcast/vector grouping, late coalescing preconditions |
| Local/grouped version slower | Barrier cost, shared bytes/block, resident blocks, problem size |
| More upcast/unroll is slower | Registers, spills, code size, remaining parallel work |
| BEAM winner regresses at full size | Reduced-size estimate, timing noise, cache state, shape representativeness |
| `DEV=PYTHON::sm_89` looks fast/slow | Discard timing claim; use it only for codegen/correctness inspection |
| Kernel improves but model does not | Amdahl's law, new bottleneck, launch/JIT effects, model benchmark policy |
