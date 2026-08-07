# 14. NVIDIA execution from first principles

## The promise of this chapter

An RTX 4090 is not one tinygrad backend. It is a physical GPU which several
tinygrad software paths can target. Those paths make different compilers,
runtimes, driver layers, and therefore different claims available to you.

By the end of this chapter, you should be able to start from a tinygrad
`PROGRAM` and answer all of these questions without guessing:

- Which text in `DEV=NVK+NV:PTX` selects the interface, backend, renderer, and
  architecture?
- Why are `sm_89`, PTX ISA 7.8, PTX text, a cubin, and SASS five different
  things?
- How do a grid, block, thread, warp, and streaming multiprocessor relate?
- Where can a value live, who can observe it, and which synchronization
  operation makes an ordering claim?
- What does the CUDA backend ask the NVIDIA driver to do between device
  initialization and `cuLaunchKernel`?
- What extra machinery does tinygrad own on the `NV`/HCQ path?
- What has actually been proved by source inspection, Python emulation, one GPU
  result, an event time, or a profiler trace?

The chapter begins with the ordinary CUDA path. You do **not** need to
understand NVIDIA command packets before contributing compiler work. The lower
level `NV` path is taught after the execution and artifact models it depends
on.

**Source snapshot:** `874d331` (2026-08-05). Every tinygrad link in this chapter
points at that snapshot. Recheck names and behavior against live `master`
before preparing a contribution.

## Route through the chapter

Read this chapter front to back once:

1. separate the physical GPU from the software route;
2. learn the minimum GPU execution and memory model;
3. follow source code through PTX, cubin, and native execution;
4. walk the CUDA Driver API lifecycle;
5. descend one layer into `NV` and hardware command queues;
6. run a hardware-free lab, then optional physical-device probes; and
7. use the evidence ladder to design a contribution-sized investigation.

On later visits, use the route table, artifact table, failure matrix, and quick
reference as lookups.

## Prerequisite and background gate

### What this chapter assumes

You should already be able to follow a Python call, inspect a tinygrad UOp
graph, and recognize the `SINK → LINEAR → SOURCE → BINARY` children of a
compiled `PROGRAM`. Chapters 1–13 build those skills.

This chapter does **not** assume that you already know CUDA, GPU programming,
assembly, driver APIs, or computer architecture. When one of those subjects is
needed, the concept is introduced before the tinygrad source which uses it.

### A three-question readiness check

Try to answer these without looking ahead:

1. If 1,024 output elements are independent, why might a compiler create many
   GPU threads rather than one long GPU loop?
2. If a kernel launch returns before the GPU finishes, why is a host stopwatch
   around the launch not necessarily a kernel timer?
3. If tinygrad prints PTX, does that prove the GPU executed those exact textual
   instructions?

It is fine if all three answers are “I do not know.” The execution model answers
the first, concurrency and events answer the second, and the artifact ladder
answers the third.

### Background ladders: stop when the exit condition is met

The companion [learning-resources page](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
routes to primary references. Do not read every NVIDIA manual before
continuing.

Use the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
if the hierarchy or memory sections below remain unclear. Stop when you can
draw one grid containing blocks, one block containing threads, and label global
and shared memory plus a block barrier.

Use the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
only when you begin reading generated PTX. Stop when you can identify the
`.version`, `.target`, entry function, parameter declarations, registers,
loads, arithmetic, stores, and `ret` in one tiny kernel.

Use the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
when changing `ops_cuda.py`. Stop when you can explain device, context, module,
function, device pointer, stream, event, and synchronization in your own words.

Direct `NV` work has no equally small prerequisite. First understand all three
models above. Then learn only the Linux device, virtual-memory, queue, and
NVIDIA command-format details touched by the task. If your change does not
cross the CUDA Driver API boundary, this detour is optional.

## One card, several software paths

Start by separating four objects which are often collapsed into “the GPU”:

| Object | Example in this chapter | What it answers |
| --- | --- | --- |
| Physical device | GeForce RTX 4090 | Which hardware ultimately runs native instructions? |
| Architecture target | `sm_89` | Which NVIDIA feature and instruction contract is code generated for? |
| Compiler path | CUDA C through NVRTC, or direct PTX | How does tinygrad turn lowered UOps into loadable bytes? |
| Runtime/interface path | CUDA Driver API, or `NVK` plus tinygrad HCQ | Who allocates memory, submits work, and waits for completion? |

The RTX 4090 is an Ada-generation GPU with compute capability 8.9, written
`sm_89` when selecting an NVIDIA machine target. That fact does not select a
tinygrad backend. Both `CUDA` and `NV` can target it; Python can also emulate a
lowered program whose renderer carries that target.

A useful first sentence is therefore:

> “I ran the `CUDA` backend with the `PTX` renderer targeting `sm_89`.”

That is more informative than “I ran on NVIDIA.”

## Decode `DEV` exactly

The pinned [`Target`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L193-L218)
parser gives the spelling a precise grammar. In the forms used here:

```text
[INTERFACE+]DEVICE[:RENDERER[:ARCH]]
```

The part left of `+` is the interface. On the right, colons separate device,
renderer, and architecture. Empty fields matter: the two colons in
`PYTHON::sm_89` leave the renderer unspecified while supplying an architecture.

| Spelling | Interface | Device/backend | Requested renderer | Requested architecture |
| --- | --- | --- | --- | --- |
| `PYTHON::sm_89` | — | `PYTHON` | default Python renderer | `sm_89` |
| `CUDA` | — | `CUDA` | first available candidate | learned from hardware |
| `CUDA:PTX` | — | `CUDA` | `PTX` | learned from hardware |
| `NV` | first available candidate | `NV` | first available candidate | learned from hardware |
| `NVK+NV` | `NVK` only | `NV` | first available candidate | learned from hardware |
| `NVK+NV:PTX` | `NVK` only | `NV` | `PTX` | learned from hardware |

Three consequences are easy to miss.

First, `CUDA:PTX` does not mean “a PTX runtime.” `CUDA` still selects
`CUDADevice` and `CUDAProgram`; only the renderer is forced to `PTXRenderer`.

Second, the `NVK` in `NVK+NV` is not a renderer. It selects `NVKIface`, the
ordinary NVIDIA kernel-driver interface used by this backend. The runtime is
still `NVDevice`/`NVProgram`.

Third, no renderer name means selection, not a timeless guarantee. The
[`Compiled._select_renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L333-L363)
code tries candidate renderer classes and keeps the first which initializes.
The candidate lists are supplied by
[`CUDADevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L120-L122)
and
[`NVDevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L633-L637):
for `CUDA` they are CUDA C, PTX, and NVCC-oriented renderers; for `NV`, NAK is
an additional candidate in this snapshot. Print the selected class when the
distinction matters.

### Why the guide never relies on automatic discovery

The snapshot's [`ALL_DEVICES`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14-L55)
tries `NV` before `CUDA`. Discovery suppresses initialization exceptions while
looking for a usable device. A machine with an NVIDIA card can therefore take
a route you did not intend if `DEV` is absent.

Use an explicit `DEV` in every reproducible investigation. Record both the
requested spelling and the selected backend, renderer, compiler, runtime, and
target.

## The safe route matrix

| Route | Where arithmetic executes | Best first question | What it cannot prove alone |
| --- | --- | --- | --- |
| `DEV=PYTHON::sm_89` | Python lowered-UOp interpreter | Did Ada-targeted lowering and arithmetic produce the expected structure/result? | Driver, GPU concurrency, barriers, occupancy, native code, timing |
| `DEV=CUDA` | RTX 4090 through CUDA Driver API | Does the normal NVIDIA path compile, load, launch, and produce the right result? | Behavior of the lower-level NV queue path |
| `DEV=CUDA:PTX` | Same CUDA runtime, direct PTX renderer | Is a problem in CUDA-C/NVRTC rendering or in later CUDA runtime stages? | NV runtime behavior; source-to-native identity |
| `DEV=NVK+NV` | RTX 4090 through tinygrad HCQ and `NVKIface` | Does tinygrad's lower-level allocation/queue/program path work? | CUDA runtime behavior |
| `DEV=NVK+NV:PTX` | Same NV runtime, forced PTX renderer | Can runtime behavior be compared while holding the renderer family fixed? | Perfectly identical final native code |

The order is deliberate. Begin with the deterministic Python route, then the
conventional CUDA path. Use direct PTX to isolate code generation. Use
`NVK+NV` only when the task concerns the lower-level runtime or when a CUDA/NV
comparison is itself the experiment.

### Interface safety boundary

[`NVDevice.ifaces`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L583-L592)
contains `NVKIface`, `PCIIface`, and a mock interface. Generic
[`_select_iface`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L494-L502)
tries ordinary candidates when no interface is named. It excludes mock as an
automatic fallback, but bare `DEV=NV` can try the direct PCI interface after
`NVKIface` fails.

That is why routine commands in this guide use `DEV=NVK+NV`. If opening
`/dev/nvidiactl` fails, the experiment stops instead of changing to a more
direct machine-access path.

`DEV=PCI+NV` is driver-development territory. It can interact with PCI BARs,
device memory management, firmware communication, and a GPU used by the
display. Do not select, unbind, reset, or reconfigure a production GPU to
complete this course. Use mock infrastructure or dedicated hardware until a
specific contribution requires direct PCI work and you have a recovery plan.

## GPU execution: turn one array expression into many workers

Suppose a kernel computes:

```python
out[i] = x[i] * 2.0 + 1.0
```

for 1,024 elements. A CPU implementation could loop over `i`. A GPU launch
describes many logical threads, each of which can compute one or several
indices. Hardware schedules those threads across many execution resources.

The programming hierarchy is:

```text
host Python process
  submits a kernel grid
    grid contains thread blocks (CUDA) / CTAs (NVIDIA command terminology)
      block contains threads
        consecutive groups of 32 threads form warps
          each thread is one lane in its warp

physical RTX 4090
  contains streaming multiprocessors (SMs)
    each resident block lives on one SM
      an SM schedules ready warps from its resident blocks
```

This diagram combines a logical hierarchy and a physical hierarchy. A grid is
not an SM, and a block is not a warp. The runtime gives the GPU grid and block
dimensions; hardware decides which SM executes each block and when each ready
warp issues.

### Grid

A **grid** is one kernel launch's collection of blocks. Grid dimensions may be
one-, two-, or three-dimensional. Independent blocks may run in any order, at
the same time, or at different times. An ordinary kernel must not assume that
block 0 finishes before block 1 begins.

In the pinned CUDA runtime, tinygrad passes `ProgramInfo.global_size` directly
to the grid-dimension arguments of `cuLaunchKernel`. Here, “global” means grid
dimensions, not total CUDA threads. The total logical threads along an axis are
grid blocks multiplied by threads per block on that axis.

### Block or CTA

A **thread block**, also called a CTA in lower-level NVIDIA structures, is a
group of threads which is scheduled as a unit on one SM. Its threads may share
on-chip shared memory and may synchronize with a block-wide barrier.

`ProgramInfo.local_size` becomes the CUDA block dimensions. A larger block is
not automatically faster. It changes how work is grouped, how many warps the
block contains, and how registers and shared memory limit simultaneous
residency.

### Thread

A **thread** executes the kernel's instruction sequence for its own logical
indices and per-thread state. Threads in the same block have different local
thread indices; blocks have different block indices. Generated address
expressions combine those indices to select array elements.

When reading a kernel, write out one concrete address for thread 0, thread 1,
and the first thread of the next block. This catches indexing mistakes faster
than staring at symbolic names.

### Warp and lane

NVIDIA groups 32 threads from a block into a **warp**. A thread's position in
that group is its **lane**. The warp is a scheduling and instruction-execution
unit, but it does not erase per-thread state: each lane has its own registers
and may follow a different control-flow path.

When lanes take different branches, the warp may have to execute paths with
different active-lane masks. This is **divergence**. “Warps execute together”
is useful intuition, but it is not permission to omit required synchronization
or memory-ordering operations.

### Streaming multiprocessor

An **SM** is a physical processor complex which holds and schedules multiple
resident warps, often from multiple blocks. A block stays on one SM for its
lifetime. Resource limits determine how many blocks and warps can reside at
once: threads, registers, shared memory, and architecture limits all matter.

**Occupancy** describes resident warps relative to the supported maximum. It is
not a synonym for utilization or speed. More occupancy can help hide latency,
but a kernel with lower occupancy may still be faster because it does less
work, reuses data better, or keeps more useful values in registers.

## Memory: location, ownership, lifetime

GPU “memory” is not one uniform box. For each value, ask four questions:

1. Where is it stored?
2. Which threads can name or observe it?
3. How long does it live?
4. What ordering or synchronization makes another observer's read valid?

| Storage | Typical owner/scope | Typical lifetime | Important cost or trap |
| --- | --- | --- | --- |
| Register | one thread | thread/kernel | Fast, but excessive use reduces residency; spills can go to local memory |
| Local memory | logically one thread, physically device memory | thread/kernel | The CUDA word “local” does not mean on-chip shared memory |
| Shared memory | one block | block/kernel | Explicit reuse and communication; limited capacity; requires correct barriers |
| Global memory | device allocation, visible to kernels with a pointer | allocation lifetime | High latency; access pattern and caching matter |
| Constant/parameter space | kernel arguments or read-only data | launch/module dependent | Specialized access and ABI rules |
| Pinned host memory | CPU allocation locked for device transfer | host allocation lifetime | Enables asynchronous DMA; must remain alive until the transfer completes |

### Registers

Registers hold per-thread scalars: indices, addresses, intermediate arithmetic,
and loaded values. Generated PTX declares virtual registers; later compiler and
driver stages assign physical registers or spill values.

A PTX register count is not definitive physical register allocation. To make a
native resource claim, inspect the compiled artifact or profiler evidence for
the actual target.

### Local memory is per-thread but not necessarily nearby

In CUDA terminology, **local memory** is a private address space for a thread.
Large per-thread arrays, address-taken values, or register spills may live
there. Despite the name, it is generally backed by device memory and serviced
through caches. Do not confuse it with block-shared memory.

### Shared memory

Shared memory is an explicitly managed, low-latency address space shared by
threads in one block. It is useful when several threads reuse the same data or
exchange partial results.

Correctness usually has two parts:

1. each producer writes the intended shared address; and
2. a block-scoped barrier and its memory semantics separate production from
   consumption when required.

A barrier cannot synchronize different blocks. If an algorithm needs a
grid-wide phase boundary, the common solution is a second kernel launch, whose
ordering is established by the runtime.

### Global memory and coalescing

Global memory holds tensor buffers. Adjacent lanes often achieve efficient
transactions when they access adjacent, properly aligned elements. This is
called **coalescing**. The exact transaction and cache behavior is
architecture-specific, but the first diagnostic remains simple: write the
addresses touched by lanes 0–31.

Coalescing is not the only performance property. Reuse, cache behavior,
instruction count, dependency chains, and occupancy also matter. Establish the
bottleneck with measurement before redesigning a kernel.

### Host memory and staging lifetime

The CPU cannot assume that an asynchronous copy has consumed its source when
the API call returns. A source buffer used by DMA must remain valid until the
copy completes. This lifetime fact explains a seemingly fussy detail in
`CUDAAllocator`: tinygrad retains pinned staging allocations in
`pending_copyin`, then frees them after context synchronization.

## Concurrency and ordering: four different operations

People often use “sync” for several different guarantees. Keep these separate:

| Operation | Participants | Guarantee |
| --- | --- | --- |
| Block barrier | threads in one block | participating threads reach a point; associated block memory ordering applies |
| Queue/stream order | commands in an ordered submission stream | later command begins according to that queue's ordering rules |
| Event or timeline signal | producer and waiter | records progress and lets another stream/queue/host wait for a value or timestamp |
| Device/context synchronization | host versus submitted device work | host waits until the relevant prior work completes |

None is a universal replacement for the others.

### Kernel launch is normally asynchronous

The host prepares arguments and submits a launch. Successful submission does
not mean the kernel has completed or even that every execution fault has been
reported. A later synchronize or result copy can surface an earlier launch
error.

This is why a host timer around only the launch often measures submission
overhead. For device kernel time, place device events or hardware timestamps
around the work and wait for the ending event/signal. For end-to-end latency,
include the synchronization boundary deliberately and say so.

### Default stream is still a stream choice

The pinned CUDA backend passes `None` for the stream in its launches and
asynchronous copies. In the Driver API that selects the default stream. The
absence of a Python stream object does not make execution synchronous.

### Events carry ordering; timestamps add measurement

An event can mark a point in a stream. Another stream can wait for it, or the
host can synchronize it. Timestamp-capable events also support elapsed device
time. These are related uses, but “I recorded an event” does not by itself say
whether it was used for dependency ordering, timing, or both.

The NV/HCQ path expresses the analogous idea with timeline signals: queue work
waits for a prior value, later work signals the next value, and host
synchronization waits until the device has reached the submitted timeline.

## Architecture target is not PTX version

The RTX 4090's compute capability is 8.9. tinygrad represents this target as:

```text
sm_89
```

A direct PTX file emitted for this target begins, in the pinned snapshot, with:

```ptx
.version 7.8
.target sm_89
.address_size 64
```

Those first two lines answer different questions.

| Name | Example | Meaning |
| --- | --- | --- |
| Compute capability / SM target | `sm_89` | NVIDIA architecture feature and native-code compatibility target |
| PTX ISA version | `7.8` | Version of the virtual PTX language accepted by a downstream assembler/JIT |
| Address size | `64` | Address model declared by the PTX module |

In this snapshot,
[`PTXCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cuda.py#L73-L86)
does not run a native assembler for the CUDA direct-PTX route. It replaces the
renderer placeholders: targets from `sm_89` through before `sm_120` receive
PTX `.version 7.8`, and `TARGET` becomes `sm_89`.

That mapping is a tinygrad toolchain decision. It does not mean “Ada is PTX
7.8,” nor does it reveal the installed driver version. A compatible CUDA driver
later accepts PTX and JIT-compiles it for the actual GPU.

For a different NVIDIA device, query the backend's selected target rather than
copying `sm_89`. This course's physical lab asserts `sm_89` because it is scoped
to the stated RTX 4090 machine; the conceptual model is architecture-agnostic.

## The NVIDIA artifact ladder

“The kernel” changes representation several times:

```text
lowered tinygrad UOps
  ├─ CUDARenderer ─> CUDA C source ─> NVRTC/NVCC ─┐
  └─ PTXRenderer  ─> PTX source ───────────────────┤
                                                   v
                          PTX bytes or cubin/ELF bytes
                                                   v
                          runtime module/program loading
                                                   v
                          native NVIDIA instructions (SASS)
                                                   v
                          scheduled warps on an SM
```

Each arrow can introduce a bug or transformation. Keep the vocabulary exact.

### CUDA C

CUDA C is a high-level kernel language. tinygrad's
[`CUDARenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L394-L472)
turns linearized UOps into a CUDA-flavored C function with indices, pointer
operations, local values, and built-ins.

Generated CUDA C is evidence about tinygrad's renderer. It is not the code the
SM directly executes.

### NVRTC and NVCC

NVRTC is NVIDIA's runtime compilation library for CUDA C. In the ordinary
`CUDA`/`CUDARenderer` combination, tinygrad asks NVRTC for PTX. In the
`NV`/`CUDARenderer` combination, it asks for a cubin because the lower-level NV
runtime needs a native-loadable artifact.

NVCC is NVIDIA's offline compiler driver. tinygrad also has an NVCC-oriented
renderer/compiler candidate. Because candidate initialization can fall back,
record `type(backend.renderer).__name__` and
`type(backend.compiler).__name__` rather than assuming which compiler ran.

### PTX

PTX is a documented virtual instruction set and module format. It expresses
register operations, address spaces, control flow, synchronization, and target
directives. It is lower level than CUDA C but still not the final instruction
encoding executed by an SM.

The pinned
[`PTXRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L137-L170)
chooses its compiler based on the runtime target:

- for `CUDA`, `PTXCompiler` produces target/version-substituted PTX bytes;
- for `NV`, `NVPTXCompiler` feeds the PTX to nvJitLink and returns a cubin.

Thus “using the PTX renderer” does not guarantee that the runtime receives PTX
text. On `NVK+NV:PTX`, PTX is an intermediate before a cubin.

### Cubin

A **cubin** is an NVIDIA binary code object/container for a target. It can hold
native code plus symbols, relocations, and resource metadata. In this source,
`NVProgram` parses an ELF-form cubin, applies relevant relocations, reads
register/shared/local resource information, and places the program into GPU
memory.

Do not use “cubin” and “SASS” as synonyms. The cubin is a container; SASS is the
native NVIDIA instruction set encoded in its code sections.

### SASS

SASS is the native machine instruction level executed by NVIDIA SMs. A driver
may JIT PTX into native code during module loading. That means PTX source and
actual SASS are connected but not identical artifacts.

If a claim depends on instruction selection, register allocation, or native
resource use, inspect the target's loaded/compiled binary with appropriate
binary utilities or profiler evidence. The [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
manual documents `cuobjdump` and `nvdisasm`. A `DEBUG=4` source dump alone is
not native-code evidence.

## Exact artifact paths in this snapshot

| Route and selected renderer | Source child | Compiler output passed onward | Who reaches native code? |
| --- | --- | --- | --- |
| `CUDA` + `CUDARenderer` | CUDA C | PTX bytes from NVRTC | CUDA driver during module load |
| `CUDA:PTX` | PTX with placeholders in `SOURCE` | PTX 7.8 / `sm_89` bytes | CUDA driver during module load |
| `CUDA` + `NVCCRenderer` | CUDA C | PTX bytes in the CUDA configuration | CUDA driver during module load |
| `NVK+NV` + `CUDARenderer` | CUDA C | cubin from NVRTC | compiler produces native container before `NVProgram` |
| `NVK+NV:PTX` | direct PTX | cubin from nvJitLink | nvJitLink before `NVProgram` |
| `NVK+NV` + `NAKRenderer` | [serialized NIR](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/nir.py#L180-L243) | [NAK metadata plus native image](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_mesa.py#L53-L73) | [`NVProgram` separates metadata from code](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L248-L256) |

This table describes the pinned implementation, not an eternal interface.
Establish the actual row by printing renderer/compiler classes and inspecting
the `BINARY` child.  Magic bytes and header substrings are format hints, not
standalone validity proofs; successful compiler and loader boundaries provide
the stronger evidence.

## `DEV=PYTHON::sm_89`: what the oracle does and does not do

[`PythonRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L205-L239)
sees an architecture beginning with `sm`, changes its renderer target's device
to `CUDA`, and loads CUDA tensor-core descriptions for that architecture. The
backend remains `PythonDevice`; its rendered transport is a pickled UOp list,
unpickled and executed by
[`PythonProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L42-L80).

That makes the route useful for:

- deterministic target parsing;
- architecture-aware lowering and tensor-core description selection;
- inspecting the `PROGRAM` boundary without a GPU driver; and
- checking arithmetic semantics for supported lowered operations.

It is not a GPU simulator. In particular, its barrier operation is effectively
a no-op under the interpreter's assumption that the emulated warp is in sync.
It does not model real warp scheduling, independent blocks, memory-system
latency, occupancy, asynchronous submission, device faults, or native timing.

The disciplined statement is:

> “The Ada-targeted Python interpreter produced this lowered structure and
> result.”

It is not:

> “This kernel is synchronized correctly on an RTX 4090.”

## The CUDA Driver API path, one object at a time

The CUDA backend delegates device management and submission to NVIDIA's Driver
API. Follow one process from startup to result.

### 1. Initialize the driver and select a device

[`CUDADevice.__init__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L97-L122)
calls `cuInit` and obtains device 0 unless a device index was requested.  The
compute-capability query comes only after context creation in the next step.

### 2. Create a context

A CUDA **context** is the driver-managed execution state associated with a
device for this process: allocations, modules, functions, and submissions are
used with a current context. The backend creates one with `cuCtxCreate_v2`;
later operations such as synchronization explicitly restore this context when
needed.

“Device” and “context” are not synonyms. The device is physical hardware; a
context is a software execution environment on it. A context error can occur
even when `nvidia-smi` sees the card.

### 3. Query the architecture, then choose renderer and compiler

Only after `cuCtxCreate_v2`, the backend calls `cuDeviceComputeCapability`.
On the RTX 4090, major 8 and minor 9 become `arch="sm_89"`. This is stronger
evidence than inferring a target from a marketing name because it is the
runtime's actual device query.  `CUDADevice` then registers:

```text
renderers: CUDARenderer, PTXRenderer, NVCCRenderer
runtime:   CUDAProgram
graph:     CUDAGraph
arch:      sm_89 on this machine
```

An explicit `CUDA:PTX` filters this list to PTX. Bare `CUDA` lets generic
selection try candidates. The compiler attaches bytes to the `BINARY` child of
the `PROGRAM` before runtime construction.

### 4. Allocate device and pinned host memory

[`CUDAAllocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L67-L95)
uses `cuMemAlloc` for ordinary device buffers. For a host staging buffer, it
uses `cuMemHostAlloc`.

Pinned host memory is page-locked so the driver can use it for asynchronous
transfer. It is a finite system resource, not a free optimization to apply to
every Python object.

### 5. Copy input bytes without destroying their lifetime

For copy-in, tinygrad:

1. allocates a pinned staging buffer;
2. copies the Python memoryview into it on the CPU;
3. retains the allocation in `pending_copyin`; and
4. submits `cuMemcpyHtoDAsync` on the default stream.

Step 3 is a correctness condition. Releasing or reusing the staging memory
before the asynchronous transfer finishes could change bytes still being read
by the device.

For copy-out, the pinned implementation first synchronizes CUDA devices, then
performs `cuMemcpyDtoH`. A host read such as `Tensor.tolist()` is therefore a
possible synchronization boundary, not a passive peek into already-finished
memory.

### 6. Load a module

[`CUDAProgram.__init__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L37-L50)
sets the context current and passes the compiled bytes to `cuModuleLoadData`.

A CUDA **module** is a loaded code/data object in the context. If those bytes
are PTX, module loading may JIT them to native code. If loading fails, the
problem is later than tinygrad source rendering but earlier than function
launch.

### 7. Look up a function

The backend asks the module for a named `CUfunction` using the `PROGRAM` name.
A module can contain more than one symbol; a function handle identifies the
kernel entry which will be launched.

Dynamic shared-memory configuration is attached to this function when the
program requests it.

### 8. Pack pointers and scalar values

[`encode_args`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L18-L24)
builds an aligned C structure containing device pointers followed by scalar
values according to the program signature. `CUDAProgram` caches the argument
container and updates pointer/scalar fields on later calls.

This is an ABI boundary. Correct arithmetic in generated source cannot rescue
a pointer in the wrong slot, a scalar with the wrong width, or bad alignment.

### 9. Launch the function

[`CUDAProgram.__call__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L55-L65)
calls `cuLaunchKernel` with:

- three grid dimensions;
- three block dimensions;
- dynamic shared-memory bytes;
- the default stream (`None`); and
- the packed argument buffer.

Returning from `cuLaunchKernel` proves that submission returned successfully.
It does not, without a wait, prove completion or a correct result.

### 10. Wait or time with events

When `wait=True`, [`cu_time_execution`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L26-L35)
creates two events, records one before and one after the launch on the default
stream, synchronizes the ending event, and asks for elapsed event time.

That measures the device interval bracketed by those events. It is not the
same as first-call end-to-end latency, which can include import, context
creation, rendering, compilation, module loading, allocation, and transfers.

### 11. Synchronize and release staging memory

`CUDADevice.synchronize()` sets the context current, calls
`cuCtxSynchronize`, frees every retained pending copy-in buffer, and clears the
list. The ordering is essential: wait first, release staging memory second.

For a device-to-device transfer, the source context records an event after the
copy and the destination default stream waits for that event. This is a concrete
example of an event carrying cross-stream/context dependency rather than merely
measuring time.

## The lower-level `NV`/HCQ path

The `NV` backend does not call `cuLaunchKernel`. It exposes more of the work a
vendor driver normally performs: resource-manager objects, virtual addresses,
channels, GPU FIFO submission, launch descriptors, DMA queues, and timeline
signals.

This path is valuable because tinygrad owns more of the runtime. It is also
harder to change safely because a malformed command or mapping can fault the
device rather than return a friendly compiler exception.

### Interface: how tinygrad reaches the device

[`NVKIface`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L370-L468)
opens NVIDIA device nodes such as `/dev/nvidiactl`, `/dev/nvidia-uvm`, and the
per-GPU node. It uses NVIDIA resource-manager controls and mappings supplied by
the installed kernel driver.

The interface is below the backend but above the physical card:

```text
NVProgram / NVAllocator / NV queues
                 |
              NVKIface
                 |
       NVIDIA kernel driver nodes
                 |
              RTX 4090
```

`DEV=NVK+NV` fixes the middle choice. A missing `/dev/nvidiactl` is then an
interface-initialization failure, not permission to try PCI.

### Device initialization: build an execution world

[`NVDevice.__init__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L590-L641)
selects the interface and creates or configures, among other things:

- resource-manager device, subdevice, and virtual-memory objects;
- a GPU virtual address space;
- a channel group and context share;
- a compute GPFIFO and a DMA GPFIFO;
- CPU-visible command storage;
- queried GPU topology and SM version; and
- generic HCQ compute/copy queues and signals.

A **virtual address** is the address used by GPU commands and kernel pointers;
it must be mapped to backing memory with suitable permissions. A **channel** is
a submission context. A **GPFIFO** is a GPU-consumed ring whose entries point
at command buffers.

You do not need to memorize register or class numbers. First understand the
ownership chain: allocate/map memory, place commands in reachable storage,
publish a FIFO entry, ring the device-facing doorbell, and later observe a
completion signal.

### Program: turn a binary into launchable state

[`NVProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L248-L339)
accepts the compiled program object. For an ELF cubin it reads sections and
metadata, applies supported CUDA relocations, allocates GPU memory, copies the
program image, and records resources such as registers, shared memory, and
local memory.

It then prepares a QMD template and validates launch/resource limits. A
**QMD** is a queue metadata/launch descriptor: it tells the GPU which program
to run, where arguments live, how large the grid and block are, and which
resources the launch uses.

Compare this with CUDA module/function loading. Both ultimately produce
launchable native state, but the CUDA driver owns the details in `CUDAProgram`;
tinygrad owns much more of them in `NVProgram`.

### Kernel arguments

Generic
[`CLikeArgsState` and `HCQProgram.fill_kernargs`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L317-L382)
allocate an argument-buffer slice and fill pointer addresses and scalar values
using the program signature.  The device creates the underlying allocation
with
[`BufferSpec(cpu_access=True)`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L418-L420),
which is why the host can write it. `NVComputeQueue.exec` places that
argument-buffer address into the QMD constant-buffer field.

The ABI question is the same as on CUDA even though the mechanics differ:
does the launched native program read the same pointer/scalar layout that
tinygrad wrote?

### Compute queue: wait, describe, execute, signal

An HCQ program call constructs this logical sequence:

```text
wait(previous timeline value)
memory barrier
execute(program, arguments, grid, block)
signal(next timeline value)
submit(queue)
```

The generic sequence appears in
[`HCQProgram.__call__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L356-L382).

[`NVComputeQueue.exec`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L128-L192)
specializes it. It copies the QMD template into argument storage, binds grid
and block dimensions, binds the constant-buffer address, emits launch commands,
and arranges a release signal after execution.

The signal is not decorative bookkeeping. It is the observable fact that lets
later work and the host know how far the GPU progressed.

### GPFIFO submission

[`_submit_to_gpfifo`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L114-L126)
makes the command queue available, writes a ring entry containing its address
and length, advances the put pointer, performs a CPU memory barrier, and writes
a device-facing token through MMIO.

The CPU memory barrier here is not a CUDA thread-block barrier. Its job is to
ensure the command and ring writes are visible before the doorbell tells the
GPU to consume them. Same word, different participants and scope.

### Copy queue and staging

[`NVCopyQueue`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L194-L212)
emits DMA copy commands. Generic
[`HCQAllocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L543-L627)
maintains host staging buffers and timeline information so a staging slot is
not reused while an earlier copy still needs it.

This is the same lifetime principle seen in CUDA's `pending_copyin`, expressed
through explicit queues and signals.

### Host synchronization

[`HCQCompiled.synchronize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L427-L450)
waits until the timeline signal reaches the last submitted value. It records a
persistent error state after a failed wait and invokes backend hang handling
when available.

On a low-level path, repeated submission after a device fault can obscure the
first failure or worsen machine stability. Stop after the first minimal fault,
capture its program/queue metadata, and use a deliberate recovery procedure on
dedicated hardware if the task requires one.

## Hold one variable constant when comparing routes

A comparison is interpretable only if you know what changed.

### Renderer comparison

Compare `DEV=CUDA` with `DEV=CUDA:PTX` to study renderer/compiler differences
while retaining `CUDADevice` and `CUDAProgram`. Still print the actual renderer
selected by bare `CUDA`; a fallback may change the premise.

Keep constant:

- tensor shapes, dtypes, and values;
- optimization flags;
- tinygrad commit;
- device and driver;
- warm-up and synchronization; and
- correctness oracle.

### Runtime comparison

Compare `DEV=CUDA:PTX` with `DEV=NVK+NV:PTX` to request the PTX renderer on
both runtimes. This holds tinygrad's source renderer family constant, but not
the entire compilation chain: CUDA receives PTX for driver JIT, whereas NV's
`NVPTXCompiler` produces a cubin through nvJitLink.

Therefore an observed difference can involve runtime submission **or** the
different PTX-to-native step. Inspect artifacts and native code before making a
narrow attribution.

### Timing comparison

Separate at least these intervals:

1. first call: initialization, compilation, allocation, load, copy, launch;
2. warm end-to-end call: host submission plus intentional synchronization;
3. device kernel interval: events/timestamps around the kernel; and
4. model-level throughput: repeated representative work with correctness
   checks and a stated synchronization policy.

Never call interval 1 “kernel time.” Never compare an asynchronous host launch
on one path with a synchronized call on another.

## An evidence ladder for NVIDIA claims

Climb only as high as the claim requires, but do not claim a higher rung than
you observed.

| Rung | Evidence | A claim it can support | A claim it cannot support alone |
| --- | --- | --- | --- |
| 0 | Read pinned source | “This implementation calls `cuModuleLoadData` here.” | That the call succeeds on your system |
| 1 | Parse/print target and classes | “This process selected `CUDA:PTX` and `sm_89`.” | That compilation or execution works |
| 2 | Inspect `SOURCE` | “PTXRenderer emitted this address/barrier sequence.” | That these are final native instructions |
| 3 | Inspect/classify `BINARY` | “The compiler produced PTX text/cubin bytes.” | That module load or relocation succeeds |
| 4 | Initialize device/runtime | “The context or NVK interface initialized.” | That a kernel loads or launches |
| 5 | Load one program | “This module/function or NVProgram was constructed.” | That arguments and launch are correct |
| 6 | Synchronized result versus oracle | “This bounded kernel returned the expected values on this route.” | Race freedom for other shapes/kernels |
| 7 | Targeted concurrency/resource tests | “This barrier/limit behavior holds for these tested cases on this GPU.” | General performance or portability |
| 8 | Device events and profiler counters | “This warmed kernel interval and measured bottleneck changed.” | End-to-end benefit without representative workload |
| 9 | Repeated end-to-end benchmark plus tests | “This change improves the stated workload under the recorded setup.” | Unmeasured devices, drivers, or models |

Two useful rules follow.

First, a failure also localizes a rung. If PTX renders but module loading fails,
do not rewrite tensor scheduling until you inspect the PTX/driver compatibility
boundary.

Second, two weak oracles do not automatically make one strong oracle. Python
emulation plus source inspection still does not establish GPU memory ordering.

## Lab — prove paths without overstating them

The lab is `labs/phase4/nvidia_paths.py`. It has one
deterministic mode and four bounded physical modes. It deliberately has no
physical bare-`NV` mode because interface fallback is outside the routine
safety boundary.

### Locate both checkouts first

The script lives in this documentation repository, not in your home directory
and not in tinygrad's `scripts/` directory. Use absolute paths so the command
works from any shell location:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
export TINYGRAD_STUDY=/absolute/path/to/tinygrad-study
cd "$TINYGRAD_STUDY"
```

If your repositories are siblings under `~/Documents/projects`, for example:

```bash
export TINYGRAD_DOCS="$HOME/Documents/projects/tinygrad_docs"
export TINYGRAD_STUDY="$HOME/Documents/projects/tinygrad-study"
cd "$TINYGRAD_STUDY"
```

The commands below use the tinygrad checkout's virtual-environment Python so
imports refer to that checkout.

### Mode 1: deterministic static and semantic evidence

```bash
DEV=PYTHON::sm_89 .venv/bin/python \
  "$TINYGRAD_DOCS/labs/phase4/nvidia_paths.py" --mode static
```

Before running it, predict:

- how every route string splits into interface/device/renderer/arch;
- why the backend class is `PythonDevice` but the renderer target begins
  `CUDA`;
- the three direct-PTX header lines before and after compilation; and
- the four numeric outputs of `x * 2 + 1`.

The mode asserts all of those facts. It takes one original kernel `SINK`,
renders it as PTX for inspection, separately compiles it for the Python
interpreter, and checks the arithmetic result.

Its final two lines name both claims and non-claims. Read them. A successful
static run establishes no NVIDIA driver or GPU fact.

### Mode 2: conventional CUDA path

```bash
DEV=CUDA .venv/bin/python \
  "$TINYGRAD_DOCS/labs/phase4/nvidia_paths.py" --mode cuda
```

This initializes `CUDADevice`, records the actual selected renderer/compiler,
compiles exactly one four-element kernel, loads it, launches with `wait=True`,
synchronizes, and compares the result.

Bare `CUDA` intentionally demonstrates default renderer selection. Do not
assume it picked `CUDARenderer`; read the printed class.

### Mode 3: CUDA runtime with direct PTX

```bash
DEV=CUDA:PTX .venv/bin/python \
  "$TINYGRAD_DOCS/labs/phase4/nvidia_paths.py" --mode cuda-ptx
```

This requires `PTXRenderer` and `PTXCompiler`, asserts exact textual PTX header
bytes beginning with `.version 7.8` and `.target sm_89`, then makes the same
synchronized CUDA-runtime result check.

### Mode 4: lower-level NV runtime through the kernel driver

```bash
DEV=NVK+NV .venv/bin/python \
  "$TINYGRAD_DOCS/labs/phase4/nvidia_paths.py" --mode nvk-nv
```

This requires `NVDevice`, `NVKIface`, and `NVProgram`. It never permits PCI
fallback. Run it only when the installed NVIDIA driver exposes the required
device nodes and the card is not in a faulted state.

### Mode 5: NV runtime with forced PTX renderer

```bash
DEV=NVK+NV:PTX .venv/bin/python \
  "$TINYGRAD_DOCS/labs/phase4/nvidia_paths.py" --mode nvk-nv-ptx
```

This requires `PTXRenderer` plus `NVPTXCompiler`, asserts ELF magic rather than
PTX-text bytes, and requires `NVProgram` to load and execute that artifact
successfully.  The printed `binary artifact` field is only a quick format hint:
ELF magic or PTX header substrings do not, by themselves, validate a cubin or a
PTX program.  Compiler provenance and successful load provide the stronger
route-specific evidence.

On the guide author's RTX 4090 host on 2026-08-07, the `cuda`, `cuda-ptx`, and
`nvk-nv` modes each compiled, loaded, launched, synchronized, and returned
`[3.0, 5.0, 7.0, 9.0]`.  The `nvk-nv-ptx` mode stopped during renderer
preflight because the installed `/lib/x86_64-linux-gnu/libnvJitLink.so.12`
lacked the `nvJitLinkVersion` symbol required by this tinygrad snapshot.  It
therefore established no compile or execution result.  These are bounded local
observations, not promises about another toolkit, driver, or machine.

### Understand `passed`, `unavailable`, and failed

The physical modes handle only narrowly recognized **preflight** failures such
as CUDA “no device,” a missing `/dev/nvidiactl`, a missing required NVIDIA
compiler library, or the exact nvJitLink ABI-symbol mismatch described above:

```text
status: unavailable
note: unavailable is not passed; no lab arithmetic kernel was compiled or launched
```

They return normally so the same teaching lab can be checked on a hardware-free
host, but the text explicitly denies compilation, launch, result, and timing
claims.  On the NV route, constructing `NVDevice` itself submits and
synchronizes compute- and DMA-queue setup commands before renderer construction;
therefore `unavailable` does **not** mean that no device command was submitted.
It means the lab did not create, compile, or launch its arithmetic kernel.
Automation which requires physical coverage must inspect the status, not merely
the process exit code.  Add `--require-available` to a physical-mode command to
turn a recognized unavailable state into a nonzero failure.  The bundled guide
runner uses that flag for every backend the caller explicitly selects with
`--device`; its final success line therefore cannot silently include an
unavailable requested physical route.

After preflight succeeds, compile, module/program load, launch, synchronization,
and result exceptions are **not** swallowed. An assertion failure or traceback
then means the lab failed and needs localization.

The script also rejects a missing or mismatched `DEV`, overrides ambient knobs
before importing tinygrad, disables optimization/JIT/tensor cores, requires one
`PROGRAM`, and asserts `sm_89`. Those constraints make the observation bounded
and reproducible; they are not benchmark settings.

### Extend the lab without weakening it

For a contribution, add one variable at a time:

1. retain the four-element correctness case;
2. add the smallest shape/dtype which triggers the behavior;
3. state which new evidence rung it reaches;
4. keep expected unavailability separate from a test failure; and
5. never catch an arbitrary exception around compilation or execution.

If you use a non-4090 NVIDIA GPU, print its queried architecture and adapt a
copy of the physical assertion. Do not relabel a different target as `sm_89`.

## Failure localization by boundary

| First failing observation | Likely boundary | Next safe comparison |
| --- | --- | --- |
| `DEV` fields are surprising | target parsing/configuration | Run static parse table; inspect `Target.parse` before opening hardware |
| Automatic backend surprises you | device discovery | Set explicit `DEV`; print `Device.DEFAULT` and class |
| CUDA error before renderer prints | driver/device/context initialization | Minimal `DEV=CUDA` device query; check driver visibility, not kernel math |
| `/dev/nvidiactl` missing | NVK interface availability | Stop; do not switch to bare `NV` or PCI as a “fix” |
| CUDA C fails but `CUDA:PTX` renders | CUDARenderer/NVRTC route | Compare source, compiler diagnostic, target, and direct PTX case |
| PTX renders but module load fails | PTX version/target/driver JIT | Inspect PTX header and module-load status; test smallest program |
| Module loads but launch rejects | arguments, dimensions, resources | Print signature, grid/block, shared memory, and function metadata |
| Launch returns but sync fails | asynchronous execution/device fault | Attribute error to earlier submitted work; minimize kernel and wait immediately |
| Python result passes, CUDA result fails | real codegen/runtime/concurrency | Inspect exact GPU source/binary, arguments, indices, and synchronization |
| CUDA passes, NVK initialization fails | NV interface/device setup | Continue compiler work on CUDA; isolate NVK setup separately |
| Same PTX source, different runtime result | compilation-to-native or runtime | Inspect binaries/native code, arguments, launch dimensions, queue ordering |
| Only first timing is slow | initialization/compile/load/warm-up | Separate first call from warmed event interval |
| Only graph replay fails | JIT/graph layer | Return to Chapter 13; compare ordinary synchronized launches |

Do not respond to a device fault by repeatedly launching a broad model. Capture
the first minimal failure and relevant driver logs. Direct interface recovery,
kernel-module operations, PCI resets, and display-GPU reconfiguration are
machine-specific and intentionally outside this guide's common path.

## Question-led source stops

Each stop has a question, a bounded range, and an exit condition. Read the
surrounding explanation here first; the source then answers a concrete
question instead of presenting declarations in isolation.

### Stop 1: Which characters select which route component?

Read [`helpers.py`, `Target` and `Target.parse`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L193-L230).

Before reading, write the five fields you expect for `NVK+NV:PTX`. Then answer:

1. Why is the string split at `+` before `:`?
2. Which two fields are uppercased?
3. Why does `PYTHON::sm_89` retain a blank renderer?
4. Does parsing instantiate a device?

Stop when you can construct the route table from the parser alone.

### Stop 2: When is a backend opened, and when is a renderer chosen?

Read [`device.py`, device selection](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14-L55),
then [`Compiled` renderer selection](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L333-L363).

Answer:

1. Why can `Device.DEFAULT` be known before the backend object initializes?
2. Why can automatic discovery hide one failed backend and choose another?
3. Where does the hardware-derived architecture fill an unspecified target?
4. Why is the selected renderer a runtime observation rather than an inference
   from `DEV=CUDA`?

Stop when you can distinguish requesting a target, opening a backend, and
constructing a renderer.

### Stop 3: What does the Python Ada route emulate?

Read
[`PythonProgram`, including its transport and barrier case](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L42-L80),
then
[`PythonRenderer` and `PythonDevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L205-L239).

Answer:

1. Which object remains Python, and which target field becomes CUDA?
2. How are lowered UOps transported to `PythonProgram`?
3. What does the barrier behavior prevent this backend from proving?

Stop when your semantic claim contains the word “interpreter,” not “GPU.”

### Stop 4: Where do PTX 7.8 and `sm_89` enter?

Read [`PTXRenderer`'s prefix and compiler choice](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L137-L170),
then [`PTXCompiler` and `NVPTXCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cuda.py#L73-L92).

Answer:

1. Which object emits `VERSION` and `TARGET` placeholders?
2. Which object replaces them for CUDA?
3. Why does the NV PTX path return cubin bytes instead?
4. Which line proves PTX version and architecture are separate directives?

Stop when you can explain the static lab's source header and binary header.

### Stop 5: Why does CUDA C compile differently for CUDA and NV?

Read the initialization of
[`CUDARenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L394-L472)
and [`NVRTCCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cuda.py#L36-L72),
then the
[`CUDAProgram` module-load boundary](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L37-L50).

Answer:

1. What boolean decides whether NVRTC produces PTX or cubin?
2. Why is it true for `CUDA` and false for ordinary `NV`?
3. Which runtime can ask a driver to JIT PTX during module load?

Stop when you can fill both CUDARenderer rows of the artifact table without
looking.

### Stop 6: What exists before the first CUDA kernel compiles?

Read [`CUDADevice.__init__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L97-L122).

Answer:

1. What is the order of driver initialization, device selection, context
   creation, and compute-capability query?
2. Where is `sm_89` constructed?
3. Which renderer and runtime classes are registered only after that query?

Stop when you can localize a CUDA Error 100 to preflight rather than codegen.

### Stop 7: Why retain a copy-in allocation?

Read [`CUDAAllocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L67-L95)
and [`CUDADevice.synchronize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L126-L134).

Answer:

1. Which memory is device memory and which is pinned host memory?
2. Why does `pending_copyin` own the staging allocation after the API returns?
3. Why is copy-out preceded by system synchronization?
4. Why is the staging list cleared only after context synchronization?

Stop when you can state the lifetime invariant as one sentence.

### Stop 8: What separates module load, lookup, launch, and wait?

Read [`encode_args`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L18-L24),
[`CUDAProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L37-L65),
and [`cu_time_execution`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L26-L35).

Answer:

1. Which call can JIT/load bytes but does not select a function?
2. Which call resolves the named entry?
3. Where are pointer/scalar arguments packed?
4. Which exact operation makes `wait=True` wait?

Stop when you can place a module-load failure and an execution fault on
different evidence rungs.

### Stop 9: What is the generic HCQ contract?

Read [`HWQueue`'s common operation docstrings](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L75-L180),
then [`HCQProgram.__call__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L333-L382).

Do not begin with NVIDIA packet numbers. First answer:

1. What abstract operations must a compute queue implement?
2. Why does a call wait on `timeline_value - 1`?
3. In what order are barrier, execute, signal, and submit constructed?
4. What additional work happens only when `wait=True` or profiling is active?

Stop when you can draw the five-operation queue sequence.

### Stop 10: Why is `NVK+NV` safer than bare `NV` for a lab?

Read [`HCQCompiled._select_iface`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/hcq.py#L494-L502),
[`NVDevice.ifaces`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L583-L592),
and the opening portion of
[`NVKIface`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L370-L410).

Answer:

1. What does an empty interface filter permit?
2. What does the explicit `NVK` filter permit?
3. Why is mock never an ordinary fallback?
4. Which device-node failure should produce `unavailable` in the lab?

Stop without instantiating bare `NV`.

### Stop 11: What does the NV backend initialize before submission?

Read [`NVDevice.__init__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L590-L641).

Group the lines into:

1. resource-manager and virtual-memory objects;
2. channel and GPFIFO objects;
3. CPU-visible command storage;
4. architecture/topology query; and
5. generic HCQ registration.

Stop when you can say what each group enables without decoding every constant.

### Stop 12: How does one NV launch reach the FIFO?

Read [`NVComputeQueue`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L128-L192),
then [`_submit_to_gpfifo`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L114-L126).

Answer:

1. Where do grid and block dimensions enter the QMD?
2. Where does the argument-buffer address enter?
3. How is completion attached to the active QMD?
4. What must be visible before the MMIO doorbell write?

Stop when you can trace dimensions and arguments from `HCQProgram.__call__` to
the FIFO entry.

## Common misconceptions, repaired

### “RTX 4090 means `DEV=CUDA`”

The card identifies hardware. `DEV` chooses software. Set `DEV=CUDA` explicitly
when that is the intended runtime.

### “`sm_89` is the PTX version”

`sm_89` is the architecture target. `.version 7.8` is the PTX language version
chosen by this toolchain for the target range.

### “PTX is NVIDIA assembly”

PTX is a virtual ISA. SASS is native NVIDIA machine code. The driver or a
linker/assembler translates between them.

### “A cubin is a string of SASS instructions”

A cubin is a binary container with code and metadata. Native instructions are
inside relevant sections.

### “`CUDA:PTX` bypasses CUDA”

It bypasses the CUDA-C renderer path. It still uses `CUDADevice`, CUDA memory,
CUDA module loading, and `cuLaunchKernel`.

### “`NVK+NV` is another architecture”

`NVK` selects an interface; `NV` selects the backend; hardware query supplies
the architecture.

### “Bare `NV` is just shorter spelling”

It changes interface-selection semantics and may try direct PCI after NVK.
That is not acceptable ambiguity for a routine lab.

### “Python `sm_89` passing proves my barrier”

The interpreter is a semantic/lowering oracle, not a concurrency oracle. Test
barriers and memory ordering on real hardware with a race-sensitive oracle.

### “The launch returned, so the result is correct”

Launch is asynchronous. Wait or copy the result, then compare to an oracle.

### “A host stopwatch measures the kernel”

It may measure only submission or may accidentally include initialization and
compilation. Use device events for a kernel interval and define end-to-end
boundaries separately.

### “High occupancy means fast”

Occupancy is resource residency, not achieved performance. Measure the actual
bottleneck and workload.

## Exercises

### 1. Parse before opening

For each string, write `Target(device, renderer, arch, interface)` without
running tinygrad:

```text
CUDA:PTX
PYTHON::sm_89
NVK+NV
NVK+NV:PTX
```

Which one explicitly names an architecture? Which explicitly names an
interface?

### 2. Name the artifact

Classify each description as CUDA C, PTX, cubin, or SASS:

1. text beginning `.version 7.8`;
2. an ELF container returned by nvJitLink;
3. native instructions shown by `nvdisasm`;
4. a kernel containing CUDA built-in index expressions.

For each, name the next boundary on `CUDA:PTX` or `NVK+NV:PTX`.

### 3. Calculate logical threads

A launch has `global_size=(80, 1, 1)` and `local_size=(128, 1, 1)` in the
CUDA backend. How many blocks, threads per block, total threads, and warps per
full block are described? Can block 79 assume block 0 has completed?

### 4. Find the missing synchronization

Threads in a block write partial sums to shared memory. Thread 0 immediately
reads every shared slot and writes a result. The Python route returns the right
number. What evidence is missing, and what is the smallest real-hardware test
which could expose the defect?

### 5. Localize an asynchronous error

`cuLaunchKernel` returns successfully, but `Tensor.tolist()` raises an illegal
address error. Which earlier operation is suspect, why can the error appear at
copy-out, and where would you place the next wait?

### 6. Compare renderers soundly

Design a comparison between bare `CUDA` and `CUDA:PTX`. List the properties you
must print or hold constant before attributing a result to the renderer.

### 7. Compare runtimes soundly

You render identical-looking PTX source for `CUDA:PTX` and `NVK+NV:PTX`, but
the binaries differ. Explain why that is expected and which evidence is needed
before blaming queue submission.

### 8. Repair an unsafe command

A debugging note says: “If `DEV=NV` fails, try `DEV=PCI+NV` with elevated
permissions.” Rewrite the note to preserve the learning goal without crossing
the routine safety boundary.

## Exercise answers

### 1. Parse before opening

```text
CUDA:PTX     -> Target(device="CUDA", renderer="PTX")
PYTHON::sm_89-> Target(device="PYTHON", arch="sm_89")
NVK+NV       -> Target(device="NV", interface="NVK")
NVK+NV:PTX   -> Target(device="NV", renderer="PTX", interface="NVK")
```

Only `PYTHON::sm_89` explicitly names an architecture. The two strings with
`NVK+` explicitly name an interface.

### 2. Name the artifact

The answers are PTX, cubin, SASS, and CUDA C. On `CUDA:PTX`, PTX bytes go to
CUDA module loading and driver JIT. On `NVK+NV:PTX`, direct PTX goes through
nvJitLink to a cubin, which `NVProgram` parses and loads.

### 3. Calculate logical threads

There are 80 blocks, 128 threads per block, 10,240 total logical threads, and
four 32-thread warps per full block. Block 79 cannot assume any completion order
relative to block 0.

### 4. Find the missing synchronization

The Python interpreter does not prove real block concurrency. A block barrier
with appropriate memory semantics is likely required between writes and the
read phase. Test a block with several warps, distinct recognizable values per
lane, repeated launches, and a CPU oracle on the actual GPU. A passing case is
bounded evidence, so include shapes which exercise all shared slots.

### 5. Localize an asynchronous error

The kernel can issue an invalid address while launch submission itself succeeds.
Copy-out synchronizes prior CUDA work, so it can report the latent fault. Place
an immediate synchronized/waiting launch around the minimized kernel to move
the error boundary closer to its cause, then inspect indices and arguments.

### 6. Compare renderers soundly

Record the selected backend, renderer, compiler, target, `SOURCE`, `BINARY`
kind, tinygrad commit, driver, flags, dimensions, inputs, oracle, warm-up, and
synchronization. Keep all but the requested renderer path fixed. If bare CUDA
falls back to PTX, the planned comparison did not occur.

### 7. Compare runtimes soundly

CUDA receives PTX and the driver JITs it during module load. NV's PTX compiler
uses nvJitLink to return a cubin before `NVProgram`. Equal source therefore does
not imply equal binary or SASS. Compare cubin/native code, arguments,
dimensions, result, and synchronization before isolating queue submission.

### 8. Repair an unsafe command

A safe note is: “Set `DEV=NVK+NV` to require the installed NVIDIA kernel-driver
interface. If it is unavailable, record the device-node/permission failure and
continue compiler work on `DEV=CUDA` or hardware-free tests. Use direct PCI
only on dedicated hardware for a task which explicitly requires it, with an
understood recovery plan.”

## Contribution-shaped workflows

### Change PTX rendering

1. Minimize one UOp pattern which reaches the PTX renderer.
2. Run the static lab to preserve target/version and `PROGRAM` invariants.
3. Assert the relevant source fragment in a focused renderer test.
4. Compile/load on `CUDA:PTX` and compare a synchronized result with an oracle.
5. Add edge shapes/dtypes and the closest existing renderer tests.
6. Inspect native output only if the claim concerns final instruction selection
   or resources.

Do not make a native-performance claim from prettier PTX alone.

### Fix CUDA allocation or copy behavior

1. Write the ownership and completion timeline for every allocation involved.
2. Reproduce with the smallest number of bytes and one explicit sync.
3. Separate host-to-device, device-to-host, and peer-copy cases.
4. Test early release/reuse boundaries deliberately.
5. Verify both correct bytes and absence of hidden global synchronization if
   concurrency is part of the intended change.

Lifetime correctness comes before throughput.

### Fix CUDA program load or launch

1. Preserve the `SOURCE` and `BINARY` artifacts from the failing program.
2. Distinguish module load, function lookup, argument packing, launch, and wait.
3. Print signature plus grid/block/shared-memory values.
4. Force an immediate wait on the minimal launch.
5. Compare `CUDA` and `CUDA:PTX` only when that isolates the suspect stage.

### Change the NV queue path

1. Reproduce through explicit `NVK+NV`; never rely on interface fallback.
2. First express the bug in HCQ terms: allocation, map, wait, barrier, exec,
   signal, submit, or synchronize.
3. Reduce it to one queue and one program/copy where possible.
4. Assert QMD fields, command sequence, or signal progression in hardware-free
   tests if the infrastructure permits.
5. Use dedicated physical hardware for malformed-command/fault work.
6. Capture the first fault and stop submission before recovery.

Only then descend into packet fields or generated headers.

### Claim a performance improvement

1. Preserve correctness tests and identify a representative workload.
2. Separate first-call, warm end-to-end, and event-timed kernel intervals.
3. Record target, renderer, compiler, runtime, driver, clocks/power conditions,
   shapes, dtypes, flags, warm-up, and synchronization.
4. Use profiler counters to identify the resource which changed.
5. Repeat enough times to report variance, not only the best sample.
6. Check for regressions on nearby shapes and the other relevant backend.

The strongest claim is usually narrow: “on this commit, driver, RTX 4090,
shape, dtype, and timing boundary.” Narrow is credible.

## Checkpoint

Continue when you can do all of the following:

- parse `PYTHON::sm_89`, `CUDA`, `CUDA:PTX`, `NV`, and `NVK+NV` without opening
  a device;
- explain why bare `NV` is excluded from routine physical labs;
- draw grid → block → thread → warp and relate blocks/warps to an SM;
- distinguish register, local, shared, global, and pinned host memory;
- distinguish a block barrier, stream order, event/signal, and host sync;
- trace CUDA C → NVRTC → PTX → driver JIT → SASS and direct PTX → cubin paths;
- keep `sm_89` separate from PTX ISA 7.8;
- narrate CUDA device/context/module/function/arguments/launch/wait;
- narrate NV interface/virtual memory/QMD/GPFIFO/signal/synchronize at a high
  level; and
- choose an evidence rung which actually supports a proposed claim.

You need not memorize QMD fields or CUDA numeric constants. You need to know
which abstraction owns the behavior and where to read next.

## Quick reference

```text
hardware                 RTX 4090 (Ada)
architecture target      sm_89 / compute capability 8.9
PTX language in snapshot .version 7.8 for sm_89 direct PTX
native ISA               SASS
binary container         cubin (ELF-form on these paths)

target grammar           [INTERFACE+]DEVICE[:RENDERER[:ARCH]]

semantic/lowering oracle DEV=PYTHON::sm_89
ordinary NVIDIA runtime  DEV=CUDA
same runtime, direct PTX DEV=CUDA:PTX
lower-level safe route   DEV=NVK+NV
NV with PTX renderer     DEV=NVK+NV:PTX
not a routine fallback   DEV=NV or DEV=PCI+NV

CUDA lifecycle
driver init -> device -> context -> allocate/copy -> module -> function
-> pack args -> launch on default stream -> event/sync -> observe result

NV/HCQ lifecycle
NVK interface -> VA/resources -> program/QMD -> wait/barrier/exec/signal
-> command buffer -> GPFIFO doorbell -> timeline wait -> observe result

always record
tinygrad commit + DEV + backend + interface + renderer + compiler + target
+ source/binary kind + driver/device + flags + shape/dtype + sync/timing boundary
```

[← TinyJit and graph replay](13-jit.md) · [Next: Debugging across the pipeline →](15-debugging.md)
