# 12. Devices and runtimes

## Purpose

The compiler has produced an ordered plan of compiled programs, copies, and
views. The runtime must attach that plan to real memory and a real execution
mechanism without violating lifetimes or asynchronous dependencies.

This chapter teaches the common runtime contracts. A backend contribution then
becomes “implement or debug this contract for one device,” rather than “read a
large `ops_*.py` file and hope the architecture emerges.”

**Verified against tinygrad:** `874d331` (2026-08-05).

## Prerequisite gate

You should be able to distinguish:

- a compiled program from the source used to produce it;
- host memory from device memory;
- synchronous completion from asynchronous submission; and
- a virtual address from the bytes it addresses.

You do not need driver-development experience. If module loading, arguments,
streams, or events are unfamiliar, skim the CUDA Driver API route in
[Learning resources](../reference/learning-resources.md#nvidia-code-generation-and-runtime-work)
and return when you can narrate allocate → copy → load → launch → synchronize →
copy back.

## Three things called “buffer”

Keep these layers separate:

```text
Ops.BUFFER UOp
  compiler/scheduler representation of storage identity, dtype, shape, device
          │ maps to
          ▼
tinygrad.device.Buffer
  Python object managing size, dtype, views, allocation state, and copies
          │ owns or views
          ▼
backend allocation handle
  pointer, device address, memory object, mmap, Python memoryview, or mock value
```

An `Ops.BUFFER` can exist before memory is allocated. A `Buffer` can lazily
allocate on first use. A backend handle has no tensor shape semantics unless the
higher layers supply them. Confusing the three produces lifetime bugs and tests
that accidentally exercise only representation, not allocation.

## The common contracts

| Object | Responsibility | It should not decide |
| --- | --- | --- |
| `Device` singleton | Parse/canonicalize a device name, import its backend class, cache opened devices, and choose a default | Tensor semantics or kernel fusion |
| `Compiled` | Bundle one logical device's allocator, renderer choices, runtime `Program` class, optional graph runner, and architecture | Individual tensor ownership |
| `Allocator` | Allocate/free opaque memory and implement copy/offset operations; `LRUAllocator` may cache freed allocations | Kernel math or launch dimensions |
| `Buffer` | Track device, element count, dtype, views/base/offset, refcounts, and lazy allocation | How an op is lowered |
| `Compiler` | Convert rendered source into bytes, optionally through a persistent cache | Load or launch the program |
| `Program` | Load one `TinyELF` program description and invoke it with buffers, scalar values, and launch dimensions | Fusion or memory planning |
| Graph runner | Batch/update/replay supported calls with lower submission overhead | Change the semantics of those calls |

The interfaces are intentionally small. Device-specific complexity belongs
behind them unless a capability genuinely changes compiler legality.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Device discovery and dynamic backend import | [`_Device`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14) |
| Buffer and lazy allocation | [`Buffer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L101) |
| Allocator and LRU wrapper | [`Allocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L221) |
| Compiler, program, and compiled-device contracts | [`device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L303) |
| Call compilation, dispatch, and statistics | [`engine/realize.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py) |
| Executable definition of lowered UOps | [`runtime/ops_python.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py) |
| Minimal no-op compiled backend | [`runtime/ops_null.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_null.py) |
| Conventional CUDA Driver API backend | [`runtime/ops_cuda.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py) |

Read `ops_null.py` to see all required pieces in one short file. Read
`ops_python.py` to understand lowered-UOp semantics. Then read the backend you
intend to change. Neither small backend is a performance or synchronization
model for real hardware.

## From `CALL` to action

`run_linear` first compiles and links a non-JIT plan, then dispatches every
`CALL` with `pm_exec`. The body in `call.src[0]` selects the action:

| Body | Runtime action |
| --- | --- |
| `SLICE` | Create a buffer view at an offset; no device kernel. |
| `COPY` | Transfer bytes, using a direct device transfer when supported or a fallback path. |
| `PROGRAM` | Resolve arguments, ensure allocations, obtain/load the runtime program, calculate launch dimensions, and invoke it. |
| `CUSTOM_FUNCTION("graph")` | Invoke a cached graph runner for a batch of calls. |
| `CUSTOM_FUNCTION("validate")` | Compare device work with CPU shadow buffers. |
| Other special custom bodies | Run the associated encode/decode or HCQ path. |

For a compiled kernel, `exec_kernel`:

1. resolves `PARAM`s against actual input UOps;
2. unwraps per-device buffers for multi-device calls;
3. allocates the program's referenced buffers;
4. obtains a cached runtime `Program` from the compiled `PROGRAM`;
5. evaluates symbolic global/local dimensions and scalar arguments; and
6. calls the program under statistics/profiling bookkeeping.

This is the compile/execute boundary. If the `PROGRAM` and argument map are
correct here, a later numerical failure belongs in runtime, ABI, synchronization,
or hardware behavior—not tensor scheduling.

## Allocation and views

`Buffer.ensure_allocated()` delays allocation until bytes are needed. Views
refer to a base allocation plus an offset and constrain deallocation/reuse.
`LRUAllocator` can retain freed opaque allocations for later requests of the
same size/options.

Ask four lifetime questions for any allocator change:

1. Who owns the base allocation?
2. Which views or in-flight commands still reference it?
3. Has the device finished reading/writing it?
4. Can the cached allocation be reused with these options and alignment?

Python garbage collection is not a device synchronization primitive. A wrapper
object becoming unreachable does not prove asynchronous work is complete.

## Asynchrony and timing

A program call may enqueue work and return before completion. `wait=True` asks
the program path for a duration/completion where supported; `Device.synchronize`
waits for pending work at device scope. Copies may also be asynchronous and may
retain temporary host buffers until synchronization.

`DEBUG>=2` makes `run_linear` wait so per-kernel statistics can be printed. That
changes execution behavior. Use it to inspect, but design benchmarks explicitly
around warm-up, events/synchronization, and repeated samples.

Ordering has two levels:

- the compiler's `LINEAR` order and buffer dependencies describe what must
  precede what; and
- the backend translates those requirements into stream/queue order, signals,
  waits, or synchronization operations.

A legal compiler schedule can still execute incorrectly if the backend omits a
necessary device dependency.

## Lab 1 — Identify the selected contracts

**Portable, then NVIDIA.** Run:

```bash
DEV=CPU .venv/bin/python -c 'from tinygrad import Device; d=Device.DEFAULT; x=Device[d]; print(d, type(x).__name__, type(x.allocator).__name__, type(x.renderer).__name__, x.runtime_t, x.graph)'
DEV=PYTHON .venv/bin/python -c 'from tinygrad import Device; d=Device.DEFAULT; x=Device[d]; print(d, type(x).__name__, type(x.allocator).__name__, type(x.renderer).__name__, x.runtime_t, x.graph)'
DEV=CUDA .venv/bin/python -c 'from tinygrad import Device; d=Device.DEFAULT; x=Device[d]; print(d, type(x).__name__, type(x.allocator).__name__, type(x.renderer).__name__, x.runtime_t, x.graph)'
```

Predict which object changes for each field. For every route, find the class
definition and answer:

- What is its opaque allocation handle?
- How does it compile or preserve program bytes?
- How are pointer and scalar arguments encoded?
- Which operation actually submits work?
- What does `synchronize` wait for?

Do not run the CUDA command on a machine where opening that backend is unsafe or
unavailable; its failure path is not required for the portable checkpoint.

## Lab 2 — Trace allocation through one kernel

Use `labs/phase3/inspect_program.py` with `DEV=CPU`, then a selected accelerator.
In a disposable study checkout, add temporary logging at:

- `Buffer.allocate` and `ensure_allocated`;
- the backend allocator's `_alloc`, `_copyin`, and `_copyout`;
- the backend `Program.__init__`; and
- `Program.__call__` plus `synchronize`.

Before running, predict the order and count for a realized input followed by one
fused output kernel. Explain extra calls rather than editing them out. Repeat
the same process twice and identify which allocation/program/compiler caches
change the second trace.

Remove the logging, then turn one observed contract into a focused test. Good
examples assert lazy allocation, view/base lifetime, argument update, or cache
reuse; avoid a test that asserts incidental debug text.

## Failure localization

| Symptom | First evidence to gather |
| --- | --- |
| Wrong result on every backend | Compare pre-runtime graph/program; likely not device-specific. |
| Correct on Python/CPU, wrong on one device | Compiled source/binary, arguments, launch dims, synchronization, backend tests. |
| Allocation only fails after repeated runs | Base/view lifetimes, LRU reuse, pending work, options/alignment. |
| Copy result stale | Source/destination device, async copy completion, stream/queue dependency. |
| Kernel launch rejects arguments | `ProgramInfo` signature, `TinyELF` signature, pointer/scalar packing, ABI widths. |
| Timing is zero or unstable | `wait`, event scope, synchronization, warm-up, cache state. |
| View corrupts adjacent data | Byte offset, dtype item size, base size, suballocation bounds. |
| Works normally, fails under JIT | Input parameterization, graph runner, buffer replacement/lifetime—not basic `Program` first. |

## Checkpoint

Continue when you can:

- distinguish `Ops.BUFFER`, `Buffer`, and a backend allocation handle;
- state the contracts of `Compiled`, `Allocator`, `Compiler`, and `Program`;
- follow each `CALL` body to its executor;
- explain why enqueue completion and Python object lifetime differ; and
- localize a backend failure after proving the compiled program and arguments
  entering the runtime.

## Quick reference

```text
Device name
  → Compiled backend
      ├── renderer → compiler → BINARY
      ├── allocator → opaque memory
      ├── Program(TinyELF) → loaded function
      └── optional graph runner

CALL(PROGRAM, buffer UOps...)
  → resolve actual buffers
  → ensure allocation
  → evaluate launch/scalar values
  → Program(...)
  → optional wait/synchronize
```
