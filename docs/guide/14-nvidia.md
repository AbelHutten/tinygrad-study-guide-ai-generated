# 14. NVIDIA path on Ubuntu

## Purpose

An RTX 4090 gives you several distinct ways to study tinygrad: execute through
the CUDA Driver API, execute through tinygrad's lower-level NV/HCQ path, render
CUDA C or PTX, or emulate an Ada-targeted lowered program in Python. This chapter
makes those choices explicit and gives you a safe escalation path.

The main learning and performance lane is `DEV=CUDA`. The `NV` backend is an
advanced runtime/driver subject, not a prerequisite for understanding tinygrad's
compiler.

**Source snapshot:** `874d331` (2026-08-05).

## Hardware identity

The desktop GeForce RTX 4090 is an NVIDIA Ada GPU with compute capability 8.9,
represented as `sm_89`. That target controls legal instructions, tensor-core
descriptions, resource limits, and toolchain output; it does not choose the
runtime by itself.

Record the actual environment before an investigation:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
git rev-parse HEAD
DEV=CUDA .venv/bin/python -c 'from tinygrad import Device; d=Device.DEFAULT; x=Device[d]; print(d, type(x).__name__, type(x.renderer).__name__, x.renderer.target)'
```

If the installed `nvidia-smi` lacks the `compute_cap` query, record its normal
output and tinygrad's renderer target instead. Never infer the selected backend
from the physical GPU name.

## Prerequisite gate

Before changing NVIDIA-specific code, be able to annotate one generated kernel
with:

- grid, block/workgroup, thread, and 32-lane warp roles;
- global loads/stores and whether adjacent lanes coalesce;
- register and shared-memory values;
- barriers and their scope; and
- launch dimensions plus pointer/scalar arguments.

Use the bounded CUDA and Triton route in
[Learning resources](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
if needed. Return after annotating elementwise and reduction kernels. Direct NV
work additionally requires command queues, virtual memory, MMIO/driver
interfaces, and NVIDIA command formats; take that detour only for a task that
actually crosses the CUDA Driver API boundary.

## Four useful routes

| Route | Executes where | What it isolates |
| --- | --- | --- |
| `DEV=PYTHON::sm_89` | Python lowered-UOp interpreter with CUDA/Ada target metadata | Targeted codegen and tensor-core semantics without GPU timing or driver behavior |
| `DEV=CUDA` | CUDA Driver API backend with default renderer selection | Conventional NVIDIA allocation, module loading, launch, CUDA graphs, and real performance |
| `DEV=CUDA:PTX` | Same CUDA runtime with direct PTX rendering | PTX renderer versus the default CUDA-C/toolchain route |
| `DEV=NVK+NV` | tinygrad HCQ path through the ordinary NVIDIA kernel-driver interface | Direct queue submission, signals, allocators, and userspace-driver behavior without interface fallback |

`DEV=NVK+NV:PTX` similarly combines the NV runtime with PTX rendering when supported.
Renderer selection and runtime selection are orthogonal; compare one dimension
at a time.

The Python Ada route is an arithmetic and lowering oracle, not a concurrency
oracle. Its interpreter cannot establish warp/workgroup scheduling, barrier
sufficiency, race freedom, device memory ordering, occupancy, or resource-limit
behavior. Test those properties on hardware with a targeted oracle.

Automatic device discovery tries `NV` before `CUDA` in this snapshot. Explicit
`DEV=CUDA` is therefore essential when the conventional path is intended. Bare
`DEV=NV` also permits fallback from the `NVK` interface to `PCI` when available;
course commands use `NVK+NV` so a failed kernel-driver initialization stops
instead of changing the machine-access path.

## CUDA path

[`CUDADevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L97)
uses CUDA Driver API bindings to:

- create/select a context and query compute capability;
- allocate device or pinned host memory;
- copy asynchronously and synchronize pending transfers;
- load compiled program bytes as a CUDA module;
- get a named function and pack pointer/scalar arguments;
- call `cuLaunchKernel`; and
- optionally batch work with `CUDAGraph`.

Its renderer candidates are CUDA C, PTX, and NVCC-oriented routes. The selected
renderer supplies the target; its compiler produces bytes accepted by
`cuModuleLoadData`. This is the best first path for codegen and performance work
because the driver owns device initialization, memory management, and command
submission.

## NV path

[`NVDevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L585)
extends tinygrad's `HCQCompiled` abstraction. It exposes substantially more of
the device machinery:

- an NVIDIA kernel-driver or direct PCI interface;
- GPU virtual-address and memory mapping;
- channel groups and GPFIFO rings;
- compute and DMA command queues;
- QMD/program state and argument buffers;
- timeline signals, waits, timestamps, and submission; and
- optional low-level profiling/fault reporting.

The key entry points are
[`NVComputeQueue`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L128),
[`NVCopyQueue`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L194),
[`NVProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L248), and
[`NVAllocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L341).

Read the generic
[`HWQueue`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L75)
and
[`HCQCompiled`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L384)
contracts before NVIDIA register/packet details.

### Interface safety boundary

Use `DEV=NVK+NV` to require the backend's ordinary kernel-driver interface for
routine study. `DEV=PCI+NV` and experimental transport interfaces are
specialized driver-development paths: they can conflict with the normal driver,
require elevated device access, and turn a Python error into a GPU or system
stability problem. Do not select, unbind, or reconfigure a production/display
GPU merely to complete this course. Use mock/emulated tests or dedicated
hardware until the contribution explicitly requires the direct interface and
you understand its recovery procedure.

## Source map

| Topic | Snapshot source |
| --- | --- |
| CUDA allocation, module, arguments, and launch | [`runtime/ops_cuda.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py) |
| CUDA graph node construction and updates | [`runtime/graph/cuda.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/graph/cuda.py) |
| NV queues, program, allocation, device initialization | [`runtime/ops_nv.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py) |
| Generic hardware command queues | [`runtime/support/hcq.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py) |
| HCQ graph replay | [`runtime/graph/hcq.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/graph/hcq.py) |
| CUDA C renderer | [`CUDARenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L394) |
| PTX renderer | [`PTXRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L137) |
| NIR → Mesa NAK native NVIDIA path | [`NAKRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/nir.py#L251) |
| NVIDIA target/tensor-core descriptions | [`codegen/opt/tc.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/tc.py) |

## Lab 1 — Build a route matrix

From the tinygrad study checkout, point `TINYGRAD_DOCS` at this guide and run
the probe through the available safe routes:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=PYTHON::sm_89 DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
DEV=CUDA DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
DEV=CUDA:PTX DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
DEV=NVK+NV DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
```

The last command is optional and should use the normal supported interface.
Before each run, predict:

- backend, renderer, and target printed by the probe;
- source representation;
- runtime `Program` class;
- which layer compiles/assembles the source; and
- whether the result is semantic evidence, device-runtime evidence, or
  performance evidence.

All successful routes must return `54.0`. If one fails, minimize the failing
transition: device initialization, lowering, compiler, module/program load,
argument/launch, or execution.

## Lab 2 — Annotate one real kernel

**NVIDIA.** Use a realized input and generate one elementwise kernel, one
reduction, and one matrix multiplication. Save `DEBUG=4` output for
`DEV=CUDA:PTX`.

For each kernel, mark:

1. parameter declarations and pointer address spaces;
2. global and local/thread indices;
3. address expression for two adjacent lanes;
4. loads, stores, and reused values;
5. any shared-memory allocation and barrier;
6. tensor-core instruction or ordinary ALU path; and
7. global/local launch dimensions from `ProgramInfo`.

Use `DEV=PYTHON::sm_89` as a semantic comparison, then run on CUDA hardware.
Only the CUDA run establishes behavior and timing on the 4090.

## Lab 3 — Compare runtimes without changing renderers

If `DEV=NVK+NV:PTX` is available, compare it with `DEV=CUDA:PTX` on the same
correct, already-compiled workload. Hold shape, dtype, PTX route, warm-up, and
synchronization constant.

Measure separately:

- first-call initialization/compilation;
- steady-state non-JIT submission;
- TinyJit replay with graphing enabled/disabled; and
- device kernel time.

Inspect generated source to prove the renderer stayed constant. A difference in
end-to-end latency can then be attributed more credibly to runtime submission,
graphing, memory, or synchronization—not to a different kernel schedule.

## Failure localization and recovery discipline

| Failure | Next safe comparison |
| --- | --- |
| Automatic default surprises you | Set `DEV=CUDA` or `DEV=NVK+NV` explicitly and print the renderer target and interface. |
| CUDA initialization fails | Check driver visibility/context creation with a minimal device query before compiling a model. |
| CUDA C fails, direct PTX works | Compare renderer/compiler output and toolchain diagnostics. |
| Python `sm_89` works, CUDA result wrong | Check actual generated program, argument packing, launch, and device runtime. |
| CUDA works, NV initialization fails | Keep compiler work on CUDA; isolate NV interface/device initialization separately. |
| NV launches then faults | Stop broad workloads; capture the first minimal fault, queue/program metadata, and driver logs. Do not repeatedly submit a known-faulting program on a display GPU. |
| Only graph replay fails | Compare `JIT=2` and ordinary launches before changing kernel code. |
| Only tensor-core path fails | Compare `TC=0`, dtype/layout requirements, `DEV=PYTHON::sm_89`, and tensor-core tests. |

Low-level device work needs a recovery plan appropriate to the interface and
machine. The guide deliberately does not prescribe unbinding, PCI reset, or
kernel-module operations: those actions are system-specific and outside the
safe common path.

## Checkpoint

Continue when you can:

- identify Ada `sm_89` as a target without confusing it with a backend;
- choose among Python-targeted, CUDA, CUDA/PTX, and NV routes for a specific
  question;
- explain where the CUDA path delegates to the driver and where NV exposes
  queues/memory directly;
- annotate one generated PTX kernel's indices, accesses, and synchronization;
  and
- design a runtime comparison that holds the renderer and kernel constant.

## Quick reference

```text
RTX 4090: Ada, sm_89, warp size 32

arithmetic/codegen oracle: DEV=PYTHON::sm_89 (not concurrency evidence)
main hardware path:       DEV=CUDA
direct PTX comparison:    DEV=CUDA:PTX
advanced queue path:      DEV=NVK+NV (kernel-driver interface required)

always record:
tinygrad commit + driver + device + backend + renderer + target + flags
```
