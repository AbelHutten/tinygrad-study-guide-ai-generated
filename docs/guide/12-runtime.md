# 12. Devices, memory, and runtime execution

## The promise of this chapter

Chapter 11 ended with an `Ops.PROGRAM`.  Its `BINARY` child contains bytes, its
`ProgramInfo` says which arguments are inputs and outputs, and its launch fields
describe the work.  None of that means the calculation has run.  Something
still has to choose a device, obtain memory, turn the program bytes into a
callable runtime object, pass the right memory handles and scalar values, submit
the work, and decide when that work is complete.

Those last steps are the **runtime boundary**.  They are where compiler facts
meet operating-system and device facts:

```text
compiled plan says                    runtime must provide
------------------                    --------------------
buffer slot 0 is output               an allocated handle for slot 0
buffer slot 1 is input                an allocated handle for slot 1
program name r_2_3                    a loaded/interpretable callable
global size (2,1,1)                   a launch or equivalent iteration
local size  (1,1,1)                   a workgroup shape or equivalent
ordered CALLs                         legal queue/submission ordering
```

This chapter builds that boundary from the beginning for a reader who knows
Python and ML but has not written a GPU runtime.  It uses `PYTHON` as the
portable baseline because its allocation handles are ordinary `memoryview`
objects and its runtime is readable Python.  That makes object and buffer
lifecycles observable without a driver.  It does **not** make `PYTHON` a model
of real device asynchrony: the Python interpreter completes the program before
returning even when `wait=False`.

By the end, you should be able to:

- distinguish a device name, a target, a `Compiled` backend, and a renderer
  target;
- distinguish an `Ops.BUFFER`, a `Buffer`, and a backend allocation handle;
- predict `Buffer.is_allocated()` and `Buffer.is_initialized()` for a base and
  a view;
- explain lazy allocation, offset handles, explicit deallocation, and allocator
  caching without confusing logical and physical memory;
- distinguish compilation, LINEAR linking, runtime loading, launch, and
  completion;
- explain what `TinyELF` carries without assuming its bytes are ELF;
- trace a `CALL(PROGRAM, ...)` through argument resolution, allocation, runtime
  caching, launch dimensions, and `Program.__call__`;
- state exactly what `run_linear(..., jit=False)` and `jit=True` mean;
- reason about queues, submission, `wait`, and device synchronization without
  borrowing evidence from the synchronous Python route; and
- choose the first useful artifact when a backend result, allocation, copy,
  launch, or timing fails.

The exact observations and source links target tinygrad commit `874d331` from
2026-08-05.  Runtime code changes quickly and interacts with drivers.  Recheck
current source and reproduce on the affected hardware before turning a pinned
observation into a contribution claim.

## Route through the chapter

Read the chapter front to back once.  The order is deliberate:

1. recover the carried calculation and locate the runtime boundary;
2. separate device-name canonicalization from target selection;
3. unpack the backend bundle;
4. follow a base buffer and an offset view through allocation states;
5. separate logical `Buffer` accounting from allocator and driver retention;
6. follow program bytes through runtime construction and caching;
7. read `run_linear` and each common `CALL` executor;
8. build an honest queue and synchronization model;
9. run the deterministic `PYTHON` lab and interpret each observation;
10. use bounded source stops only after the model is in place; and
11. practice failure localization and contribution-shaped exercises.

No C, CUDA, virtual-memory, or ABI background is required to begin.  When one
of those subjects becomes necessary, the background ladders say what to learn
and when to stop.  Chapter 13 adds TinyJit capture and replay.  Chapter 14 then
develops physical NVIDIA routes on Ubuntu.

## Recover the calculation and its oracle

The lab carries the same expression as Chapters 10 and 11:

```python
x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]], dtype=dtypes.float32).realize()
out = (x*x + 2*x).sum(axis=1)
```

For each element it computes `x*x + 2*x`, then sums each row:

```text
row 0: 3 + 8 + 15  = 26
row 1: 24 + 35 + 48 = 107
```

The semantic oracle is therefore:

```text
[26.0, 107.0]
```

The controlled `PYTHON` program has two global buffer parameters:

```text
slot 0: float output, 2 elements, written
slot 1: float input,  6 elements, read
```

Its `ProgramInfo` reports:

```text
globals=(0, 1)   outs=(0,)   ins=(1,)
```

These are **roles**, not addresses.  The compiler knows that slot zero is
written and slot one is read.  The runtime later finds the corresponding
`Buffer` objects, allocates them if necessary, obtains their backend handles,
and passes those handles in global-slot order.

### The boundary in one picture

```text
Tensor expression
  ↓ scheduling, lowering, rendering, compilation (earlier chapters)
LINEAR(
  CALL(
    PROGRAM(name, BINARY bytes, ProgramInfo),
    BUFFER output,
    BUFFER input))
  ↓ run_linear / exec_kernel
resolve BUFFER UOps to tinygrad.device.Buffer objects
  ↓ ensure_allocated
allocator returns backend handles
  ↓ get_runtime(PROGRAM.to_elf())
loaded or interpreted Program object
  ↓ Program(handle0, handle1, global_size=..., local_size=..., vals=...)
host submission or synchronous interpretation
  ↓ wait / synchronize / copyout as required
observable result [26.0, 107.0]
```

Every arrow is a failure boundary.  “The tensor is wrong” is too broad to be a
useful diagnosis.  A contribution starts becoming tractable when you can say,
for example, “the compiled signature and handles are correct, but the CUDA
argument packer gives one scalar the wrong width.”

## Vocabulary bridge: one word, several layers

### Three things commonly called a buffer

Keep these separate:

```text
Ops.BUFFER UOp
  an IR node naming storage, dtype, size/shape, device, and parameter role
        │ .buffer resolves to
        ▼
tinygrad.device.Buffer
  a Python object tracking element count, dtype, device, base/view relation,
  byte offset, options, allocation state, and per-device handles
        │ allocator creates or derives
        ▼
backend allocation handle
  memoryview, host pointer, device address, API memory object, mmap, and so on
```

An `Ops.BUFFER` can exist before physical memory exists.  A `Buffer` can exist
before its own handle exists.  A device pointer has no tensor shape semantics
by itself.  The higher layers supply the element count, dtype, role, and legal
access range.

For the controlled Python route:

```text
Ops.BUFFER → Buffer(device="PYTHON", size=...) → memoryview(bytearray(...))
```

For a CUDA route, the last object is a CUDA device pointer instead.  The common
`Buffer` API stays recognizable while allocation, offset, copy, and free
operations change behind the allocator.

### Four moments commonly called compilation

Use more precise verbs:

| Moment | Meaning in this chapter |
| --- | --- |
| **Render/compile** | Turn target-legal UOps into `SOURCE` and `BINARY` bytes inside an `Ops.PROGRAM`. |
| **LINEAR link** | Transform or combine an already compiled execution plan; at this snapshot it is an identity unless experimental `HCQ2` is enabled. |
| **Runtime construction/load** | Construct a backend `Program` from `TinyELF`; this may unpickle UOps, load a module, resolve a function, or perform another backend-specific action. |
| **Launch/interpret** | Call that runtime object with handles, launch dimensions, scalar values, and `wait`. |

Chapter 11 also showed that one compiler, such as the CPU compiler, may invoke
a native object linker internally.  That is different from
`link_linear(...)`.  Similar names do not make the stages identical.

### Submission is not completion

On an accelerator, a Python call can submit work to a queue and return while
the device is still running it.  Three times must remain separate:

```text
t0  host calls Program(...)
t1  command has been accepted/enqueued; host may return
t2  device has finished all reads and writes for that command
```

For a synchronous implementation, `t1` and `t2` are effectively joined from
the caller's perspective.  For an asynchronous implementation they are not.
Freeing or reusing a resource after `t1` can be wrong if work still refers to
it until `t2`.

## Device strings are not targets

This distinction prevents a surprising number of false explanations.

### Device canonicalization answers “which runtime instance?”

`Device.canonicalize` normalizes the **device string** used to open/cache a
backend instance.  At the pinned snapshot it uppercases the portion before the
first colon and removes a trailing `:0`:

```text
"python"   → "PYTHON"
"python:0" → "PYTHON"
"python:2" → "PYTHON:2"
```

It does not parse renderer and architecture fields.  `Device[canonical_name]`
uses the base before `:` to import `tinygrad.runtime.ops_<base>`, finds the
matching `<Base>Device` class, constructs it, and caches it by the complete
canonical device string.

`Device.DEFAULT` is the device field selected by `DEV`, or an automatically
probed device when none is specified.  With:

```text
DEV=PYTHON::sm_89
```

the default device name is still `PYTHON`.  The architecture belongs to target
selection, not to the key used to open `PythonDevice`.

### `DEV.target` answers “how should this device be rendered?”

`Target` stores five fields:

```text
device, renderer, arch, interface, indices
```

`DEV.target(base_device, ...)` finds the selected target for that base, fills
missing defaults such as a detected architecture, and forces the returned
target's device field to the requested base.  Later,
`Compiled._select_renderer` uses that target to select and initialize a
renderer.  The renderer can refine its own target.  For example, the ordinary
lab prints:

```text
Device.DEFAULT:                 PYTHON
DEV target for selected device: PYTHON
selected renderer target:       PYTHON:PYTHON
```

The extra `PYTHON` is the renderer choice installed by `PythonRenderer`.  Under
an emulation target such as `PYTHON::sm_89`, that renderer also models CUDA
target capabilities, so its final target need not print the same device field
as `Device.DEFAULT`.

Do not use `program.arg.target.device` as a substitute for “which runtime
instance owns this allocation.”  Do not use `Device.DEFAULT` as a substitute
for “which renderer and architecture produced this program.”  Inspect the
object that answers the question you are actually asking.

### Colons appear in two related contexts

`CUDA:1` can be a canonical device-instance string meaning CUDA device index
one.  `DEV=CUDA:PTX:sm_89` is parsed as target selection: CUDA device family,
PTX renderer, and `sm_89` architecture.  Code passes these through different
APIs.  When debugging, print both the canonical device string and the parsed or
selected `Target` instead of guessing from punctuation.

## The `Compiled` backend bundle

Despite its name, `Compiled` is the common bundle for a device runtime.  A
backend instance supplies or selects:

| Field/property | Question it answers |
| --- | --- |
| `device` | Which canonical runtime instance is this? |
| `allocator` | How are bytes allocated, freed, copied, mapped, and offset? |
| `renderers` / `renderer` | Which target code generator is available/selected? |
| `compiler` | How does rendered source become `lib: bytes`? |
| `runtime_t` / `runtime(obj)` | Which `Program` class consumes a `TinyELF`, and how is it constructed? |
| `graph` | Can this backend replay a batch through a graph runner? |
| `arch` | Which architecture did the device detect, if any? |
| `synchronize()` | How does the host wait for device-scope pending work? |

For `PYTHON`, the lab finds:

```text
PythonDevice
  allocator  PythonAllocator
  renderer   PythonRenderer
  compiler   PythonCompiler
  runtime    PythonProgram
```

These objects cooperate, but they are not interchangeable.  `PythonCompiler`
turns the renderer's base64 text back into pickle bytes.  `PythonProgram`
unpickles and interprets those bytes.  `PythonAllocator` owns memoryviews.  A
bug in one contract should not be patched in another simply because all four
live in the same small source file.

### Required and optional allocator operations

The common `Allocator` wrappers expose `alloc`, `free`, and `map`, while a
backend implements primitives such as:

```text
_alloc(size, options)       create an opaque handle
_free(handle, options)      release it physically when required
_copyin(dest, host_view)    host bytes → backend allocation
_copyout(host_view, src)    backend allocation → host bytes
_offset(handle, nbytes, o)  derive a handle for a subrange
```

Mapping, direct transfer, zero-copy exposure, and encode/decode hooks are
capability-dependent.  The absence of a fast path does not necessarily mean a
copy is impossible; `exec_copy` has fallbacks.  Conversely, the presence of a
method does not prove it is asynchronous or zero-copy.  Read the implementation
and its tests.

## Buffer lifecycle from first principles

### Size is in elements; offsets and allocation are in bytes

`Buffer(device, size, dtype)` stores an element count.  Its byte size is:

```text
nbytes = size * dtype.itemsize
```

For four `float32` elements:

```text
size = 4 elements
itemsize = 4 bytes
nbytes = 16 bytes
```

`Buffer.view(size, dtype, offset)` receives a new element count and dtype, but
the offset is a **byte offset**.  The lab creates:

```python
base = Buffer("PYTHON", 4, dtypes.float32)       # 16 bytes
view = base.view(2, dtypes.float32, 4)           # 8 bytes starting at byte 4
```

That view covers base elements one and two.  The pinned `view` method checks
that the starting offset is below the base byte size.  Do not infer from that
single check that every possible view extent is universally validated there;
the caller and the construction path still owe valid ranges, types, and
alignment.

### Lazy allocation

Constructing a `Buffer` normally does not call the allocator.  Allocation is
deferred until `ensure_allocated()` or another operation requires a handle.
This matters because scheduling and memory planning can create storage objects
before execution knows which allocations must be materialized.

For a base buffer, allocation does roughly this:

```text
allocator.alloc(nbytes, options)
  → opaque handle
store handle under this device
increase logical active-byte counters
```

For a view, allocation does something different:

```text
ensure base is allocated
increase base.allocated_views
allocator._offset(base_handle, view_nbytes, byte_offset)
  → view handle
store view handle under this device
```

The view does not allocate another `view_nbytes` of backing storage in the
ordinary path.  It creates another way to address a subrange of the base.

### `is_allocated` and `is_initialized` answer different questions

The names are close, so use their exact questions:

- `is_allocated()` asks whether the **underlying base storage** is allocated.
- `is_initialized()` additionally asks whether **this Buffer object** has a
  handle for its device.

For a base, the answers usually move together.  For a view they do not:

| Moment | Base `(allocated, initialized)` | View `(allocated, initialized)` | Why |
| --- | --- | --- | --- |
| After construction | `(False, False)` | `(False, False)` | No base handle exists. |
| After `base.ensure_allocated()` | `(True, True)` | `(True, False)` | Base storage exists, but this view has no offset handle yet. |
| After `view.ensure_allocated()` | `(True, True)` | `(True, True)` | `_offset` created the view's handle. |
| After `view.deallocate()` while base lives | `(True, True)` | `(True, False)` | Base storage remains, but the view handle was cleared. |
| After base deallocation | `(False, False)` | `(False, False)` | Underlying storage no longer counts as allocated. |

The impossible combination is `(False, True)`, because `is_initialized()` is
defined as allocated storage **and** a handle in this object's handle map.

This distinction is not cosmetic.  Calling `ensure_allocated()` on the view
must initialize the offset handle even when its base already exists.  A guard
that checks only `is_allocated()` would skip required work.

### What `allocated_views` counts

Creating a Python view object does not immediately increment
`base.allocated_views`.  Initializing its offset handle does.  Deallocating that
view handle decrements the counter.  It is therefore closer to a count of
initialized view handles than a count of every Python view object ever
constructed.

Also keep three notions of “reference” separate:

| Notion | What it tracks |
| --- | --- |
| Python references | Whether an object can be garbage-collected. |
| `uop_refcount` | tinygrad's storage/liveness accounting on the base. |
| `allocated_views` | Initialized offset handles derived from the base in this implementation. |

They solve different problems.

### A view does not create a universal deallocation lock

A view stores a Python reference to its base.  That prevents ordinary Python
garbage collection from finalizing the base object while the reachable view
still refers to it.  That is useful, but it is not a universal runtime rule
that rejects every explicit base deallocation.

At the pinned snapshot, the base branch of `Buffer.deallocate()` calls the
allocator's `free` path without first rejecting a nonzero `allocated_views`
count.  Backend handles and allocators differ.  Therefore:

- do not explicitly deallocate a base while an initialized view is in use;
- do not claim `allocated_views` universally makes that operation impossible;
- when testing a lifecycle, deallocate views first, synchronize if device work
  could still refer to storage, then deallocate the base; and
- when changing this area, test the exact backend and automatic as well as
  explicit lifetime paths.

The lab uses that conservative order.  It observes state; it does not perform
an intentionally unsafe use-after-free experiment.

### `__del__` is cleanup, not synchronization

`Buffer.__del__` deallocates an initialized handle when the Python object is
finalized.  Python does not know whether an accelerator queue still uses that
handle.  Correct queue/resource tracking must keep resources alive or establish
completion before physical reuse.  “The Python wrapper went out of scope” is
not evidence that the GPU finished.

Likewise, garbage-collection timing is an implementation detail, not a stable
performance or correctness boundary.  Prefer explicit runtime ownership and
synchronization reasoning in tests.

## Allocation policy and accounting

### `BufferSpec` describes requested allocation properties

The common options include flags for uncached allocation, CPU access, host
allocation, bypassing the allocator LRU, and an external pointer.  Not every
backend gives every flag the same physical implementation.  Treat
`BufferSpec` as a request/contract interpreted by that allocator, not as a
portable hardware description.

External pointers deserve particular care: tinygrad did not necessarily create
or own that memory.  The allocation and logical accounting branches explicitly
special-case them.  Ownership, lifetime, accessibility, and synchronization
must be stated when adding an interop path.

### Base allocation and view initialization account differently

For ordinary non-DISK, non-external base storage, `Buffer.allocate` increments:

```text
GlobalCounters.mem_used
GlobalCounters.mem_used_per_device[device]
```

by the base `nbytes`.  Initializing a view does not add the view size again.
The lab therefore observes:

```text
initial                           +0 logical bytes
allocate 4 × float32 base        +16 logical bytes
initialize 2 × float32 view      +16 logical bytes (unchanged)
deallocate view                  +16 logical bytes (unchanged)
deallocate base                   +0 logical bytes
```

These are tinygrad's logical active-`Buffer` counters.  They are not a query of
the process resident set, CUDA driver, VRAM page tables, or device allocator.
`GlobalCounters.reset()` intentionally does not reset `mem_used`.

### LRU retention breaks the “counter equals physical memory” assumption

`LRUAllocator.free` can place an opaque allocation handle into a cache keyed by
size and options instead of calling the physical `_free`.  A later compatible
allocation can reuse it.  If a new allocation fails, the allocator empties the
cache and retries.

The order across layers is important:

```text
Buffer.deallocate(base)
  ↓ decrement logical mem_used
allocator.free(handle, size, options)
  ↓ when eligible and LRU enabled
retain opaque handle in allocator cache
```

So `mem_used` can decrease while physical memory remains retained for reuse.
It tracks logically active base buffers, not physical LRU residency.  The
portable `PythonAllocator` is a direct `Allocator`, not an `LRUAllocator`, so
the Python lab demonstrates logical accounting but cannot demonstrate physical
LRU retention.  CUDA's allocator is an LRU allocator at this snapshot, but
physical-memory claims still require driver/tool evidence.

### A safe lifetime checklist

Before freeing, reusing, or exposing an allocation, answer:

1. Which object owns the base allocation?
2. Which initialized views address it, and what byte ranges do they cover?
3. Which queued commands can still read or write those ranges?
4. Which event, signal, queue order, or synchronization proves completion?
5. Does the allocator physically free the handle or retain it for reuse?
6. Do the new request's size and `BufferSpec` match the reuse key?
7. Is an external party responsible for the pointer's lifetime?

If question four has no evidence-backed answer, do not infer safety from Python
scope or `mem_used`.

## From program bytes to a runtime object

### `TinyELF` is a transport object, not a file-format promise

At this snapshot `TinyELF` is a small dataclass:

```text
TinyELF(
  lib: bytes,
  name: str,
  target: Target,
  signature: tuple[(name, slot, dtype, shape), ...])
```

The name is historical or suggestive; the type is used as a generic transport
between `Ops.PROGRAM` and a backend `Program`.  Nothing in the dataclass
requires `lib` to start with an ELF header or contain native machine code.

For routes already studied:

| Route | What `TinyELF.lib` carries |
| --- | --- |
| `PYTHON` | Pickle bytes for ordered UOps. |
| `CPU:CLANG` | The linked host image described in Chapter 11. |
| direct PTX/CUDA route | PTX or other backend-consumable bytes for module loading. |

Always identify the producer and consumer before naming the format.  A useful
statement is “`PythonProgram` unpickles this `lib`.”  “tinygrad generated an ELF”
is unsupported merely because the wrapper class is called `TinyELF`.

### Signature and program roles cooperate

The signature preserves parameter names when available, slots, dtypes, and
shapes.  It lets a runtime know widths and ordering needed for argument
encoding.  `ProgramInfo.globals`, `outs`, and `ins` say which buffer slots are
passed and how the program accesses them.  Scalar symbolic parameters are
evaluated separately through `ProgramInfo.vals(var_vals)`.

On CUDA, for example, the runtime argument encoder places device pointers and
then integer scalar fields with dtype-aware alignment.  A wrong numerical
result can therefore come from a correct kernel paired with incorrect runtime
packing.  Check both sides of the ABI:

```text
compiler side                         runtime side
-------------                         ------------
parameter order                       argument order
pointer versus scalar role            pointer/scalar encoding
dtype width and signedness             ctypes/API field width
shape/slot metadata                   selected handle
name                                  loaded function symbol
```

An **ABI** is the low-level agreement about how separately implemented pieces
communicate: calling convention, argument layout, widths, alignment, symbol
names, and related details.  You do not need a full ABI course to trace this
table.

### Runtime construction is lazy and cached

Compiling an `Ops.PROGRAM` creates bytes; it does not necessarily load them.
`get_runtime(device, program)` uses a process-local cache keyed by:

```text
(program.key, device)
```

On a miss it calls:

```text
Device[device].runtime(program.to_elf())
```

For Python, construction unpickles ordered UOps and precomputes loop lookup
tables.  For CUDA, construction selects the current context, loads a module
from `lib`, resolves the named function, and retains its handle.  Those are
different meanings of “load,” hidden behind the same `Program` contract.

The lab proves the distinction by checking that its program's runtime-cache
key is absent after `compile_linear`, then present as a `PythonProgram` after
dispatch.

### Keep the caches separate

Several caches can make a second run look different:

| Cache | Stores | Typical symptom when misunderstood |
| --- | --- | --- |
| `Device[...]` cache | Opened backend instance per canonical device string | Initialization appears only once. |
| renderer selection cache | Initialized renderer for a selected target | Target/tool probing is not repeated. |
| compiler cache | `SOURCE` → `BINARY` bytes, possibly persistent | Compiler changes seem ignored. |
| program-construction cache from codegen | Reused `Ops.PROGRAM` artifacts | Rendering is not repeated. |
| `runtime_cache` | Loaded/interpretable `Program` per key and device | Module load/unpickle appears only once. |
| allocator LRU | Freed opaque allocation handles | Physical allocation call is skipped on reuse. |
| graph/JIT caches | Parameterized batches or captured plans | Individual dispatch behavior changes. |

Do not “clear the cache” indiscriminately and then claim a diagnosis.  Name the
artifact that is stale, identify its key and lifetime, and isolate one cache at
a time in a fresh process or controlled configuration.

## `run_linear`: compile state, not TinyJit capture state

This API has an especially misleading parameter if read without its body.

At the pinned snapshot, default `jit=False` means:

```python
linear = link_linear(compile_linear(linear, ...))
```

before execution.  The usual input can therefore contain:

```text
CALL(SINK, ...)
```

and `compile_linear` replaces the kernel body with `PROGRAM`.

By contrast, `jit=True` means **the caller is supplying an already
compiled/linked execution plan**.  `run_linear` skips those two transformations
and visits the calls directly.  The expected input contains dispatchable bodies
such as:

```text
CALL(PROGRAM, ...)
```

This flag does not itself:

- create a `TinyJit` decorator;
- perform the ignore/capture/replay lifecycle;
- capture Python execution;
- graph a device queue; or
- compile a raw `SINK` plan.

Chapter 13 explains how TinyJit eventually supplies prepared plans.  In this
chapter the lab deliberately does the preparation itself:

```python
compiled = compile_linear(raw, beam=0)
run_linear(compiled, var_vals, jit=True)
```

It also passes a separate raw `SINK` plan to the default `jit=False` path and
observes that it compiles and executes.  Never translate `jit=True` as “turn JIT
on” without the `run_linear` context.

### What the execution-context flag still affects

`run_linear` copies the Boolean into `ExecContext.jit`.  Statistics use it to
color the debug label.  At this snapshot that statistics read is the only
direct use of `ctx.jit`; capture machinery does not inspect the field.  The
function argument also controls whether `run_linear` performs compile/link
before constructing the context.  Neither behavior is the TinyJit capture
operation itself.

## Dispatching a LINEAR plan

After optional compile/link work, `run_linear` visits the plan's outer calls in
order.  `pm_exec` chooses an executor from the body at `call.src[0]`:

| Body | Executor | Runtime effect |
| --- | --- | --- |
| `SLICE` | `exec_view` | Create/register a `Buffer.view`; no compute kernel. |
| `COPY` | `exec_copy` | Ensure source/destination allocation and select transfer, disk, zero-copy, or copy fallback. |
| `PROGRAM` | `exec_kernel` | Resolve buffers, allocate used globals, obtain a runtime, calculate arguments, and invoke it. |
| custom `encdec` | `exec_encdec` | Call a backend encode/decode hook. |
| custom `graph` | `exec_graph` | Invoke a cached graph runner. |
| custom `hcq` | `exec_hcq` | Execute the experimental hardware-command-queue path. |
| custom `validate` | `exec_validate` | Compare against a CPU validation route. |

The compiler's outer `LINEAR` order is a host-side execution plan.  It can
contain metadata actions, copies, and kernels.  Not every call launches a
compute kernel, and “kernel count” should not be inferred from total call count.

### `exec_kernel` line by line

For the ordinary single-device carried program, the important sequence is:

1. **Resolve parameters.**  Captured/replayed plans may contain `PARAM` buffer
   placeholders.  `_resolve` replaces them with the actual input UOps.  An
   ordinary non-parameter buffer remains itself.
2. **Unwrap multi-device storage.**  The executor selects the per-device
   `Buffer` objects and binds a device-index variable where required.
3. **Select program globals.**  `ProgramInfo.globals=(0,1)` chooses the output
   and input buffers from the resolved call arguments.
4. **Ensure allocation.**  The input is already realized in the lab.  The lazy
   output becomes allocated here.
5. **Obtain the runtime.**  `get_runtime` constructs or reuses the loaded
   `Program` for `(program.key, device)`.
6. **Evaluate dimensions and scalars.**  Symbolic launch dimensions and scalar
   parameters become concrete from `var_vals`.
7. **Obtain per-device handles.**  `Buffer.get_buf(device)` initializes or maps
   the required handle.
8. **Call the program.**  The runtime receives handles in global-slot order,
   plus `global_size`, `local_size`, scalar `vals`, `wait`, and an optional
   timeout.
9. **Track evidence.**  Statistics/profiling record roles, estimates, timing,
   and device information according to the current context.

For the lab, the output buffer changes from unallocated to allocated at step
four.  The runtime cache changes at step five.  The actual values appear at
step eight.  Observing each boundary is more informative than inserting print
statements throughout the source.

### Views and copies are execution-plan actions too

`exec_view` calculates a byte offset from a slice's element offset and source
dtype item size, creates a `Buffer.view`, and records it for the destination
buffer UOp.  It does not ask `Program.__call__` to run a kernel.

`exec_copy` ensures both buffers are allocated, then selects among direct
same-family transfer, disk-specific paths, a destination zero-copy buffer, or
ordinary copyin/copyout.  A stale copied result may therefore belong to:

- wrong source/destination UOps;
- wrong byte count;
- a backend transfer implementation;
- host staging lifetime;
- missing queue/event dependency; or
- completion before copyout.

Do not start by changing math lowering when the failing outer call is `COPY`.

## Queues and synchronization

### A minimal queue model

A **queue** or **stream** is an ordered destination for submitted device work.
The host builds commands; the device consumes them.  Within one ordered queue,
later commands commonly wait for earlier commands according to that API's
rules.  Across queues or devices, explicit events/signals/dependencies may be
required.

Use this timeline for two dependent kernels:

```text
host:    submit A ─ submit B ───────── synchronize ─ read result
queue:        [ A writes temp ][ B reads temp, writes out ]
device:       execute A ─────── execute B ───────── complete
```

The host can reach `submit B` before A completes while the queue still preserves
the device dependency.  Adding a full device synchronization between A and B
would preserve correctness but can destroy overlap and hide the missing
fine-grained dependency you actually need.

### `wait` is a request interpreted by the backend

The common `Program.__call__` accepts `wait=False` and returns `float | None`.
That type deliberately permits different implementations:

- an asynchronous backend may enqueue and return `None` when `wait=False`;
- `wait=True` may use events or another completion mechanism and return elapsed
  time;
- a synchronous implementation may finish work and return a duration even
  when `wait=False`; and
- a backend can have additional synchronization in copies or graph paths.

Therefore neither the Boolean nor the return type alone proves actual overlap.
Read the selected `Program`, measure a host timeline when relevant, and use
backend events/profiling for device completion.

### The Python runtime is deliberately synchronous

`PythonProgram.__call__` runs ordinary Python loops:

```text
for each global coordinate:
  build interpreter state
  walk ordered UOps
  perform loads, arithmetic, stores, and control flow
return perf_counter() - start
```

It does not enqueue work on another processor.  Its `wait` parameter is not
used to choose a different path, and it always returns an elapsed-time float
after interpretation finishes.  `PythonDevice` inherits the base no-op
`synchronize` method because there is no pending device queue to drain.

This makes `PYTHON` excellent for:

- observing program and buffer contracts;
- checking lowered-UOp semantics;
- deterministic correctness oracles; and
- testing state transitions that do not require driver behavior.

It cannot establish:

- that a CUDA launch returns before the kernel completes;
- that two device queues overlap;
- that a signal or event has the right scope;
- that pinned staging memory remains alive long enough;
- that a GPU allocation can be reused safely after submission; or
- that a physical device timer is accurate.

The lab explicitly prints that `wait=False` returned an elapsed-time float.  It
labels that as evidence of synchronous Python execution, not evidence of an
asynchronous implementation.

### CUDA illustrates the missing machinery

The pinned conventional CUDA backend makes the contrast concrete:

- `CUDAProgram.__init__` loads module bytes and resolves a function;
- `CUDAProgram.__call__` packs device pointers/scalars and calls
  `cuLaunchKernel`;
- when `wait=True`, CUDA events surround the launch and the ending event is
  synchronized to obtain elapsed time;
- CUDA copyin uses pinned host staging plus an asynchronous host-to-device copy;
- the device retains those staging allocations in `pending_copyin`; and
- `CUDADevice.synchronize()` waits for the context, then releases the retained
  staging allocations through the allocator.

That final sequence shows why lifetime and synchronization are coupled.  The
temporary host allocation cannot be discarded merely because `_copyin`
returned.  The queue may still read it.

For peer device-to-device transfer, the backend records an event on the source
context and makes the destination default stream wait on it.  That is a
fine-grained dependency, not the same operation as globally synchronizing both
devices after every copy.

Chapter 14 covers NVIDIA-specific routes in more depth.  Here the important
lesson is where to look: argument packing, API submission, event/stream order,
staging lifetime, and synchronization are runtime contracts, not renderer
spelling.

### Debug timing can change execution

`run_linear` sets the execution context's wait field to:

```text
wait argument OR DEBUG >= 2
```

If debug statistics need a duration and the runtime returns `None`, the stats
path calls `Device.synchronize()` and measures host elapsed time.  Thus
`DEBUG>=2` can turn asynchronous-looking submission into a waited path.

Use debug output to inspect calls, but do not cite its timings as an unchanged
steady-state execution experiment.  For performance work, state warm-up,
compilation/cache state, number of repetitions, wait/event scope, and the exact
device.  Chapter 17 builds that methodology.

## Portable lab: observe, do not source-edit

The lab reads state and cached objects directly.  It does not monkeypatch
allocators, replace runtime methods, or ask you to insert temporary logging in
tinygrad source.  It creates and cleans up one small explicit base/view pair,
then executes the carried calculation through two `run_linear` modes.

### Evidence contract

On the controlled `PYTHON` route the lab establishes:

- the documented device canonicalization examples;
- the distinction between the selected `DEV` target and renderer target;
- exact Python backend component classes;
- base/view allocation and initialization states;
- byte-offset view behavior for four `float32` values;
- logical active-byte changes without double-counting the view;
- raw `SINK` versus compiled `PROGRAM` call bodies;
- lazy output allocation;
- `TinyELF.lib` as pickle bytes for ordered UOps on this route;
- absence then presence of the process runtime-cache entry;
- synchronous result visibility after Python `wait=False`;
- the oracle `[26.0, 107.0]`; and
- a raw plan accepted by default `run_linear(..., jit=False)`.

It does not establish physical allocation retention, GPU queue overlap, CUDA
event correctness, or RTX execution.

### Run it from the pinned tinygrad checkout

Follow Chapter 2's directory model.  From the pinned tinygrad study checkout,
set `TINYGRAD_DOCS` to this guide repository and run:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=PYTHON \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/runtime_walk.py"
```

The script fixes `NOOPT=1`, `SPEC=2`, `VIZ=0`, and the other plan-affecting
variables before importing tinygrad.  It also supplies `dtypes.float32`
explicitly, so a caller's `DEFAULT_FLOAT` cannot change the signature.  `DEV`
remains the caller's explicit choice.  Run without Python's `-O` flag: the lab
fails immediately under optimized Python because removing its assertions would
invalidate the evidence contract.

### Expected portable output

The Python route should report:

```text
controlled env: BEAM=0 CACHELEVEL=0 DEBUG=0 HCQ2=0 IMAGE=0 NOOPT=1 SPEC=2 TC=0 THREADS=1 VALIDATE_WITH_CPU=0 VIZ=0
selection
  Device.DEFAULT: PYTHON
  canonical samples: {'python': 'PYTHON', 'python:0': 'PYTHON', 'python:2': 'PYTHON:2'}
  DEV target for selected device: PYTHON
  selected renderer target: PYTHON:PYTHON
  backend: PythonDevice
  allocator/renderer/compiler/runtime: PythonAllocator PythonRenderer PythonCompiler PythonProgram
  assertion set: common Compiled contracts
buffer lifecycle: (is_allocated, is_initialized)
  initial base/view, logical-byte delta: ((False, False), (False, False), 0)
  after base allocation: ((True, True), (True, False), 16)
  after view initialization, view count: ((True, True), (True, True), 1, 16)
  PYTHON view reads base elements 1:3: [20.0, 30.0]
  after view deallocation: ((True, True), (True, False), 0, 16)
  after base deallocation: ((False, False), 0)
program lifecycle
  raw/compiled CALL bodies: SINK PROGRAM
  roles globals/outs/ins: (0, 1) (0,) (1,)
  output/input allocated before dispatch: False True
  TinyELF fields: lib/name/target/signature = bytes r_2_3 PYTHON:PYTHON 2
  loaded runtime cached before dispatch: False
  PYTHON transport payload decodes to ordered UOps: True
  PYTHON result present before explicit synchronize: [26.0, 107.0]
  PYTHON wait=False returned an elapsed-time float: True
  loaded runtime cached after dispatch: True PythonProgram
  result after device synchronize: [26.0, 107.0]
  TinyJit capture object created: False
  raw SINK accepted by run_linear default jit=False: [4.0, 5.0]
```

### Read the selection section

The first target line is what `DEV.target("PYTHON")` supplies before renderer
initialization.  The second is the selected renderer's target.  The canonical
samples come from the device-name normalizer and do not require opening three
Python devices.

Predict this variation without running it:

```text
DEV=PYTHON::sm_89
```

`Device.DEFAULT` remains `PYTHON`, but target capability and the generated
program organization can change.  Revisit Chapters 9–11 before treating that
mode as a pure runtime comparison.

### Read the buffer state table

The important observation is not merely “allocation was lazy.”  It is the
intermediate view state:

```text
base  (True, True)
view  (True, False)
```

The view sees that its base is allocated, so `is_allocated()` is true.  Its own
offset `memoryview` has not been installed, so `is_initialized()` is false.
After initialization, reading the view gives `[20.0, 30.0]`, proving that byte
offset four over `float32` addresses elements one and two.

The logical byte counter stays at 16 because only the base owns storage.  The
lab frees the view first and then the base.  It does not claim that the final
zero says anything about a GPU driver's retained memory.

### Read the program section

Before dispatch:

```text
raw call body       SINK
compiled call body  PROGRAM
output allocated    False
input allocated     True
runtime cached      False
```

The input was explicitly realized.  The output is lazy.  Compilation produced
the program transport but did not construct `PythonProgram`.

After `run_linear(compiled, ..., jit=True)`:

```text
output allocated    True
runtime cached      True, PythonProgram
result              [26.0, 107.0]
```

The direct Python invocation with `wait=False` returns a float only after the
interpreter has written the result.  That is the synchronous Python contract.
The printed `TinyJit capture object created: False` prevents the prepared-plan
flag from being silently reinterpreted as capture.

Finally, the `[4.0, 5.0]` probe passes an uncompiled `SINK` plan through the
default `jit=False` route.  The fact that it executes is consistent with the
source: default `run_linear` compiles and links internally.

### Optional CPU structural comparison

If the pinned checkout has a working Clang route, run:

```bash
DEV=CPU:CLANG \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/runtime_walk.py"
```

The script conditionally asserts the expected `CPUDevice`, `CPUAllocator`,
`ClangRenderer`, and `CPUProgram` classes.  It retains the common state and
oracle assertions but skips Python-payload and direct-memoryview claims.  Its
target, compiler artifact, allocation handle, and synchronization mechanics are
different even when the result is the same.

### Optional physical CUDA comparison

On an explicitly selected, working CUDA setup:

```bash
DEV=CUDA \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/runtime_walk.py"
```

The script conditionally checks the `CUDADevice`, `CUDAAllocator`, and
`CUDAProgram` structure and then synchronizes before reading the common result.
Treat driver initialization, allocation, compilation, module loading, and
execution failures as real optional-route failures to localize; do not relabel
them as portable-lab failures.

One successful run proves this small program on that recorded route.  It does
not prove queue overlap or lifetime safety.  To investigate those, construct a
separate backend-specific test with events/profiling and a workload long enough
to measure, then state exactly which submission and completion times you
observed.

### Optional `NVK+NV` comparison on the RTX 4090 route

The guide's second NVIDIA command can run the same common runtime contract in
a separate process:

```bash
DEV=NVK+NV \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase4/runtime_walk.py"
```

On the recorded Ubuntu/RTX 4090 host at this snapshot, the device string opens
the canonical runtime instance `NV`, while the requested and selected target
prints `NVK+NV::sm_89`; the small result is again `[26.0,107.0]` after explicit
synchronization. The lab intentionally applies only its common structural and
semantic assertions to this route. Chapter 14 explains why the `NVK`
interface, `NV` runtime instance, CUDA-family renderer, and resulting target
are related names rather than interchangeable synonyms. Treat a failure on
another driver or GPU as a backend-specific result, not as evidence that the
portable Python route failed.

### Safe extensions

Extend the lab without editing tinygrad source:

1. Add another uninitialized view at byte offset eight and predict every state.
2. Print `base.nbytes`, `view.nbytes`, and the logical counter delta; explain
   why the numbers are 16, 8, and 16.
3. Change the default-path input to three elements and predict its program key
   and runtime-cache behavior in a fresh process.
4. Run `PYTHON` and `CPU:CLANG` in separate processes and compare only the
   fields whose contracts are common.
5. On a physical backend, record allocator class inheritance and whether it
   uses `LRUAllocator`; do not infer retained bytes from `mem_used`.

Avoid patching methods just to count calls.  A wrapper can alter object identity,
cache keys, destructor timing, or synchronization.  Prefer explicit state,
focused tests, profiler events, or a backend's documented tracing facility.

## Failure localization

Start with the earliest adjacent boundary whose input is correct and output is
wrong.

| Symptom | First evidence | Likely boundary |
| --- | --- | --- |
| Wrong result on `PYTHON` and every hardware backend | Compare lowered `LINEAR`, program roles, and oracle | Before device-specific runtime |
| Correct `PYTHON`, wrong one backend | Program bytes/target, signature, handles, dimensions, scalar packing, completion | Backend runtime or target-specific compilation |
| Runtime-cache entry exists but module-load error occurred earlier | Record key, constructor exception, `lib` format, target, name | `Program.__init__` / artifact compatibility |
| Launch rejects arguments | Compare signature slots/dtypes with runtime packing and selected handles | ABI/argument encoder |
| Output buffer remains unallocated | Inspect `ProgramInfo.globals`, resolved call buffers, and executor body | Call/role resolution or wrong executor |
| View reports allocated but `_buf` access fails | Check `is_initialized`, base state, device mapping, and offset initialization | View-handle initialization |
| View corrupts adjacent values | Byte offset, view byte extent, dtype item sizes, base range | View construction / offset contract |
| Logical memory returns to zero but VRAM remains high | Allocator inheritance/cache, driver metrics, external ownership | LRU/driver retention, not `mem_used` bug by itself |
| Allocation fails after repeated runs | Active bases/views, in-flight work, LRU key/options, physical cache flush/retry | Lifetime or allocator policy |
| Host-to-device copy intermittently corrupts | Staging buffer lifetime and stream completion | Async copy ownership/synchronization |
| Result becomes correct after global synchronize | Identify the missing producer→consumer dependency | Queue/event ordering; synchronize is evidence, not final fix |
| `wait=False` returns a float on `PYTHON` | Read Python runtime body | Expected synchronous behavior |
| `DEBUG=2` is correct but `DEBUG=0` fails | Compare waits/synchronization and timing path | Hidden async dependency/race |
| `jit=True` plan does nothing or fails on `SINK` | Inspect call bodies before execution | Caller supplied uncompiled plan |
| Works normally, fails in TinyJit replay | Capture parameters, buffer replacement, graph dependencies, lifetime | Chapter 13; not `run_linear` flag alone |

### Use synchronization as a diagnostic, not a reflex

If adding a device-wide synchronize makes a failure disappear, you have learned
that completion/order is implicated.  You have not proved where the missing
dependency belongs.  Next reduce the scope:

1. identify the producer command and consumer command;
2. identify their queues/devices;
3. identify the exact buffer byte range connecting them;
4. find the event/signal/order contract that should connect them; and
5. write a regression test that fails without that dependency.

A permanent global barrier may serialize unrelated work and conceal future
lifetime bugs.

## Question-led source stops

Do not open these links as isolated declarations.  Each stop gives you a
prediction and a bounded question.  Answer it in plain language, relate it to
the lab, and close the source.  All links target the recorded snapshot.

### Stop 1: Which string opens the backend?

Prediction: `python:0` and `PYTHON` share the same canonical cache key, while
`python:2` does not.  Read the complete bounded
[`_Device` selection and cache path](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14-L55).

Question: which substring chooses the imported module, which full string keys
the opened instance, and where does `Device.DEFAULT` come from?

### Stop 2: Which fields choose code generation?

Prediction: `DEV=PYTHON::sm_89` opens `PythonDevice` but preserves an `sm_89`
target choice.  Read
[`Target`, its parser, and `_DEV.target`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L192-L230),
then
[`Compiled._select_renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L333-L363),
and finally the bounded
[`PythonRenderer` target refinement](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L205-L223).

Question: which layer forces a base device, which fills detected architecture,
and which can refine the final renderer target?

### Stop 3: Why can a view be allocated but not initialized?

Prediction: after allocating only the base, the view returns `(True, False)`.
Read
[`Buffer` construction through allocation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L101-L161).

Question: which expression follows the base for `is_allocated`, and which
additional dictionary membership is required for `is_initialized`?  Find the
line that creates the offset handle.

### Stop 4: What does explicit deallocation actually guarantee?

Prediction: view deallocation decrements `allocated_views`, but base
deallocation has no universal guard against a nonzero count in this range.
Read
[`Buffer.deallocate`, finalization, host exposure, and view construction](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L162-L216).

Question: which branch calls allocator `free`, which branch only clears a view
handle/count, and where can host exposure synchronize?

### Stop 5: When does “free” retain memory?

Prediction: logical accounting can decrease before the opaque allocation is
physically released.  Read the base
[`Allocator` contract and `LRUAllocator`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L221-L269),
then the small
[`GlobalCounters` definition](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L295-L301).

Question: what is the LRU reuse key, when is the cache flushed, and which
counters does `reset()` deliberately leave untouched?

### Stop 6: What can a `Program` assume about `lib`?

Prediction: the common transport promises bytes, a name, target, and signature,
not ELF syntax.  Read
[`Compiler`, `TinyELF`, `Program`, and the start of `Compiled`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L303-L353).

Question: which method produces bytes, which class consumes the wrapper, and
where is any file-format validation required to live?

### Stop 7: When is a runtime constructed?

Prediction: `compile_linear` alone does not populate `runtime_cache`.  Read the
complete
[`get_runtime` cache](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L111-L125).

Question: what forms the key, what exact constructor call occurs on a miss, and
what does `cache=False` change?

### Stop 8: What happens between `CALL` and `Program.__call__`?

Prediction: only the buffers listed in `ProgramInfo.globals` are ensured and
passed to the runtime.  Read
[`resolve_params`, view/copy execution, and `exec_kernel`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L141-L186).

Question: write the carried output/input objects after parameter resolution,
then point to allocation, runtime lookup, launch evaluation, handle lookup, and
the final call.

### Stop 9: What does the `jit` flag mean here?

Prediction: raw `SINK` is compiled only when `jit` is false.  Read
[`pm_compile`, dispatch, compile/link, and `run_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L247-L281).

Question: which branch invokes `compile_linear` and `link_linear`, and why must
the lab's `jit=True` input already contain `PROGRAM`?  Identify every line in
the range that mentions neither TinyJit capture nor Python decorators.

### Stop 10: Why is the Python route synchronous?

Prediction: `wait=False` still returns a float after the output store has run.
Read
[`PythonProgram` setup, its main loop, and stores](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L42-L104),
the narrow
[`RANGE` and `LOAD` interpreter cases](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L117-L136),
and
[`ordinary ALU execution and the return`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L194-L200).

Question: where is the timer started, where are global coordinates and UOps
iterated, where are stores performed, and where is elapsed time returned?  Find
any branch that changes behavior based on `wait`; there is none in this range.

### Stop 11: What does the Python backend allocate and transport?

Prediction: base allocations and view offsets are memoryviews; program bytes
are pickled ordered UOps.  Read the small
[`PythonProgram` constructor](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L42-L46),
then the complete
[`Python compiler, renderer, allocator, and device bundle`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L202-L239).

Question: follow `LINEAR UOps → base64 text → pickle bytes → PythonProgram`, and
follow `bytearray → base memoryview → sliced view memoryview`.

### Stop 12: What additional contracts appear on CUDA?

Prediction: launch submission, `wait=True` event timing, asynchronous staging,
and device synchronization are separate mechanisms.  First read
[`CUDA argument packing and program launch`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L18-L65),
then
[`CUDA allocation, copies, transfer dependencies, and synchronization`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L67-L133).

Question: which operations only submit, which one waits for an event, which
temporary allocations survive copyin, and what releases them after context
completion?

## Background ladders

Use these only when the named concept blocks you.  The goal is not to finish a
textbook before returning.

### Level 0: enough Python to run the portable route

Know these standard ideas:

- an object can hold a reference to another object;
- `bytes` are immutable byte content, while `bytearray` is mutable;
- `memoryview` exposes a byte buffer without necessarily copying it;
- dictionary membership can represent whether a handle has been installed;
- a callable object's `__call__` can behave like a function; and
- `try/finally` or deliberate teardown order protects cleanup.

If `memoryview` casting or slicing is unfamiliar, use the
[Python `memoryview` documentation](https://docs.python.org/3/library/stdtypes.html#memory-views).
Stop when you can explain why a slice beginning at byte four of four float32s
starts at the second element and can share the same underlying `bytearray`.

### Level 1: allocation and virtual addresses

Learn only these concepts:

- allocation reserves a byte range and returns a handle/address;
- a virtual address names bytes through a mapping, not necessarily a physical
  RAM location;
- an offset pointer/view names a subrange of a base allocation;
- ownership says who is allowed/required to release a resource; and
- caching an allocation is different from keeping a logical tensor alive.

Stop when you can draw one base allocation, two non-overlapping byte-range
views, and the owner responsible for physical release.  You do not need to
implement page tables.

### Level 2: queues, events, and synchronization

Use the execution-and-memory-model route in
[Learning resources](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path).
Focus on host versus device execution, asynchronous launches, streams, events,
and memory visibility/order.  Then use the
[CUDA Driver API route](../reference/learning-resources.md#nvidia-code-generation-and-runtime-work)
when reading `ops_cuda.py`.

Stop when you can explain:

- why a launch call can return before a kernel finishes;
- why same-stream order can preserve a dependency without a host wait;
- why two streams/devices may need an event or signal;
- why an async copy's host staging buffer must remain alive; and
- the difference between waiting for one event and synchronizing a device.

### Level 3: ABI and argument packing

Learn pointer width, integer width/signedness, alignment, structure layout,
function symbols, and calling conventions.  Start from the concrete carried
signature rather than a complete platform ABI manual.

Stop when you can take:

```text
two buffer pointers + one int32 scalar
```

and explain why sender and receiver must agree on order, offsets, widths,
alignment, and symbol name.  Then trace `TinyELF.iter_sig` and the selected
backend encoder.

### Level 4: backend-specific command machinery

Only enter this level for a contribution that needs it.  Conventional CUDA
uses driver API calls.  Lower-level routes can manage virtual memory, queues,
signals, packets, or command buffers more directly.  Use Chapter 14 and the
vendor documentation appropriate to the exact backend.

Stop expanding scope once you can state the broken contract and build a focused
reproducer.  Reading every backend before changing one is not required.

## Common misconceptions, corrected

| Misconception | Correction |
| --- | --- |
| `Device.DEFAULT` is the compilation target. | It is the selected runtime device string.  Inspect `DEV.target` and `renderer.target` for renderer/architecture choices. |
| `Device.canonicalize` parses renderer and architecture. | It normalizes a device-instance string; target parsing is separate. |
| A `Buffer` object means memory was allocated. | Allocation is normally lazy. |
| `view.is_allocated()` means the view has an offset handle. | It can be true because the base is allocated while `view.is_initialized()` is false. |
| A view allocates and accounts another full backing buffer. | The ordinary path derives an offset handle and accounts the base bytes once. |
| `allocated_views > 0` universally prevents explicit base deallocation. | The pinned base deallocation path does not enforce that universal guard.  Use safe ordering and inspect the backend. |
| `GlobalCounters.mem_used` is physical VRAM use. | It is logical active base-`Buffer` accounting; an LRU or driver can retain physical memory after it drops. |
| `GlobalCounters.reset()` resets all counters. | It leaves `mem_used` and per-device memory accounting untouched. |
| `TinyELF.lib` must be an ELF file. | `TinyELF` is a generic runtime transport; Python carries pickle bytes, and other routes carry their own formats. |
| Compiling a `PROGRAM` loads it into the runtime. | Runtime construction is lazy and separately cached. |
| `run_linear(jit=True)` activates TinyJit capture. | It says the supplied LINEAR plan is already compiled/linked. |
| `run_linear(jit=False)` disables compilation. | It is the default path that compiles and links before dispatch. |
| `wait=False` proves asynchronous execution. | Python is synchronous even with `wait=False`; each backend defines the behavior. |
| A Python `synchronize()` call proves a wait occurred. | `PythonDevice` inherits a no-op synchronize because execution already completed. |
| If adding a global synchronize fixes a bug, that is the right patch. | It diagnoses a missing completion/dependency; the correct fix may be a narrower event, queue edge, or lifetime rule. |
| Python garbage collection knows when GPU work is done. | Object reachability and device completion are separate systems. |
| Correct Python execution proves a CUDA queue implementation. | It proves common semantics and Python contracts only. |

## Contribution-shaped runtime work

### Allocator change

Record:

- exact device and allocator class;
- requested byte sizes and `BufferSpec` values;
- base/view ranges and ownership;
- logical counter changes;
- physical allocator calls or cache reuse evidence;
- pending device work at free/reuse; and
- a focused repeated-allocation test.

Test both the first allocation and reuse path.  A test that only asserts
`mem_used` cannot validate physical LRU behavior.

### View or mapping change

Record element count, dtype item size, byte offset, byte extent, base device,
and handle type.  Test beginning, middle, and end boundaries; nested
construction if supported; deallocation order; and mapping to another device
where relevant.  Distinguish an object that sees allocated base storage from an
initialized per-device handle.

### Program loader or ABI change

Preserve a minimal `TinyELF` artifact, target, name, signature, launch
dimensions, scalar values, and handle order.  Separate constructor/load failure
from call/launch failure.  Include widths and alignment in the regression, not
only numerical output.

### Queue or synchronization change

Name producer, consumer, queues/devices, byte ranges, and the missing or added
dependency.  Demonstrate the failure without a global wait, then prove the
narrow dependency.  Add stress/repetition because races may not fail on every
run.  Measure performance impact after correctness.

### New backend

Start with the smallest common contract:

```text
Device bundle
  allocator: alloc/free/copy/offset
  renderer + compiler: target program bytes
  Program: construct/load and call
  synchronize: completion scope
```

Use `PYTHON` to understand semantics, a short backend such as `NULL` to see
interface shape, and a physically similar backend for real lifetime/queue
requirements.  `NULL` does not validate results, and `PYTHON` does not validate
asynchrony.  A new backend needs real-device tests in addition to shared
semantic tests.

## Exercises

Try each before opening its answer.

### 1. Canonicalize without parsing a target

What do `python`, `python:0`, and `python:2` canonicalize to?  Does that answer
which architecture `PythonRenderer` models?

??? answer
    They become `PYTHON`, `PYTHON`, and `PYTHON:2`.  This normalizes the device
    runtime key.  It does not answer renderer architecture; inspect the selected
    `Target` and final `renderer.target` for that.

### 2. Predict four view states

A base and view are constructed.  Then only the base is allocated.  Then the
view is initialized.  Then the view is deallocated while the base remains.
Give the view's `(is_allocated, is_initialized)` at each moment.

??? answer
    The states are `(False,False)`, `(True,False)`, `(True,True)`, and
    `(True,False)`.  Allocation follows the base; initialization additionally
    requires this view's offset handle.

### 3. Count logical bytes

A four-element float32 base has a two-element float32 initialized view.  What
logical `mem_used` delta does the pinned ordinary path report, and why?

??? answer
    It reports 16 bytes.  The base owns `4*4=16` bytes.  View initialization
    derives an offset handle and does not account another eight backing bytes.

### 4. Interpret zero logical bytes with a nonempty LRU

After base deallocation, `mem_used` returns to its starting value but the
allocator cache contains a handle for that size/options key.  Is either
observation inconsistent?

??? answer
    No.  `mem_used` tracks logically active base buffers.  `LRUAllocator.free`
    can retain the opaque physical allocation for reuse after the logical
    counter drops.  Driver/device metrics are needed for a physical-memory
    claim.

### 5. Deallocate a base with a live view

Does `allocated_views == 1` prove an explicit `base.deallocate()` call will be
rejected at this snapshot?  What order should a test use instead?

??? answer
    No.  The base branch does not universally guard on `allocated_views`.
    Finish any device work, deallocate initialized views, then deallocate the
    base.  A lifecycle change needs backend-specific tests rather than an
    assumption that the counter is a lock.

### 6. Classify `TinyELF`

The Python lab's `transport.lib` unpickles to a list of ordered UOps.  What file
format claim can you make?

??? answer
    You can say the Python transport contains pickle bytes consumed by
    `PythonProgram`.  You cannot say it is an ELF object merely because its
    wrapper is named `TinyELF`.

### 7. Separate compile and load

`compile_linear` has produced a `PROGRAM`, but the `(program.key, device)` entry
is absent from `runtime_cache`.  Is the state valid?  What event should create
the entry?

??? answer
    Yes.  Compilation creates the artifact without constructing its runtime.
    `exec_kernel` calls `get_runtime`; on a miss it builds a `Program` from
    `program.to_elf()` and caches it when caching is enabled.

### 8. Choose the correct `run_linear` call

You hold raw `LINEAR(CALL(SINK,...))`.  Which is correct: default
`run_linear(raw)` or `run_linear(raw, jit=True)`?

??? answer
    Use the default `jit=False` call.  It compiles and links the raw plan.
    `jit=True` says the caller already supplied a compiled/linked plan and skips
    replacement of `SINK` with `PROGRAM`.

### 9. Explain why this is not TinyJit capture

The lab calls `compile_linear(raw)` and then `run_linear(compiled, jit=True)`.
Which TinyJit phases occurred?

??? answer
    None.  The lab constructed and executed a prepared plan directly.  It did
    not decorate a function, perform ignore/capture/replay calls, or build a
    captured object.  Chapter 13 introduces those phases.

### 10. Interpret Python `wait=False`

`PythonProgram(..., wait=False)` returns a float, and the output bytes are
already correct before an explicit synchronize call.  Does this show overlap?

??? answer
    No.  It shows the opposite: ordinary Python loops interpreted all UOps and
    returned elapsed time after completion.  The `wait` flag does not select an
    asynchronous path in `PythonProgram`.

### 11. Diagnose a CUDA-only wrong scalar

The same compiled semantics pass on Python.  CUDA launches, but results are
wrong only when a symbolic int64 scalar exceeds 32-bit range.  Where should you
look first?

??? answer
    Compare the program signature and evaluated scalar order with CUDA argument
    encoding: dtype width, signedness, alignment, and structure offsets.  The
    symptom points to the ABI/runtime packer before tensor algebra or buffer
    allocation.

### 12. Use a synchronize experiment correctly

Kernel B reads a buffer written by kernel A on another queue.  Adding a device
synchronize between them fixes the result.  What has the experiment proved,
and what remains?

??? answer
    It implicates missing completion/order.  It has not shown that a global
    barrier is the right fix.  Identify A's completion event, B's queue, the
    shared byte range, and the intended cross-queue dependency; test that
    narrower edge.

### 13. Localize an uninitialized view

`view.is_allocated()` is true, but accessing `view._buf` raises because the
device key is absent.  Is the predicate wrong?

??? answer
    Not necessarily.  The predicate correctly reports allocated base storage.
    The view is not initialized.  Trace the path that should call
    `view.ensure_allocated()` or `get_buf(device)` and create the offset handle.

### 14. Compare two caches

After freeing a tensor, the runtime program and an allocation both appear to
be reused.  Name the two independent caches and their keys.

??? answer
    `runtime_cache` reuses the loaded/interpretable `Program` by
    `(program.key, device)`.  `LRUAllocator` can reuse an opaque allocation by
    `(size, options)`.  Reusing one says nothing about whether the other hit.

### 15. Design an async-copy regression

What must a focused test record if a host-to-device copy sometimes reads
destroyed staging memory?

??? answer
    Record the staging allocation's owner and lifetime, copy queue/stream,
    submission and completion evidence, destination range, and cleanup point.
    Stress multiple copies, ensure cleanup happens only after the promised
    event/synchronization, and avoid relying on Python garbage-collection
    timing.

### 16. State the optional CUDA lab claim

The CUDA invocation returns `[26.0,107.0]` after `backend.synchronize()`.  What
is the strongest justified claim?

??? answer
    The recorded tinygrad snapshot, selected CUDA target/toolchain, driver, and
    physical device correctly executed this small program in that run.  It does
    not by itself prove asynchronous overlap, all dtypes/shapes, allocator reuse
    safety, or correctness of another NVIDIA backend.

## Checkpoint

Continue to TinyJit when you can do all of the following without source open:

- draw the path from `Ops.BUFFER` to `Buffer` to backend handle;
- explain why a view can be allocated but not initialized;
- calculate base/view byte sizes and logical memory accounting;
- state the safe view-before-base teardown order and its limits;
- explain why `mem_used` can fall while an LRU retains physical allocation;
- distinguish device canonicalization, `DEV.target`, and `renderer.target`;
- describe `TinyELF` as a generic transport and identify the Python payload;
- distinguish compilation, LINEAR linking, runtime construction, and launch;
- narrate all nine `exec_kernel` steps for the carried program;
- say why default `jit=False` compiles while `jit=True` expects a prepared plan;
- explain why that flag is not TinyJit capture;
- distinguish queue submission, event completion, and device synchronization;
- state why the Python route is synchronous even with `wait=False`; and
- choose runtime/ABI/lifetime evidence after Python semantics already pass.

If allocation states are unclear, rerun only the base/view portion and redraw
the table.  If queue reasoning is unclear, complete Background Level 2 before
Chapter 14.  If `jit=True` still sounds like capture, reread the `run_linear`
section before Chapter 13; that chapter uses the prepared-plan meaning rather
than replacing it.

## Quick reference

```text
device selection
  Device.canonicalize(name) → runtime-instance key
  DEV.target(base)          → renderer/arch/interface choice
  Device[key]               → cached Compiled backend

buffer state
  is_allocated   = underlying base storage exists
  is_initialized = base exists AND this object has a device handle

base allocation
  allocator.alloc(nbytes, options)
  logical mem_used += nbytes

view initialization
  ensure base
  allocator._offset(base_handle, view_nbytes, byte_offset)
  logical mem_used unchanged

program lifecycle
  CALL(SINK, ...) --compile_linear--> CALL(PROGRAM, ...)
  PROGRAM.to_elf() → TinyELF(lib bytes, name, target, signature)
  get_runtime      → cached Program object
  exec_kernel      → handles + dimensions + scalars + wait

run_linear
  jit=False (default): compile_linear → link_linear → dispatch
  jit=True:           already compiled/linked plan → dispatch
  neither operation is TinyJit capture by itself

completion
  submit/enqueue ≠ device completion
  wait semantics are backend-specific
  Device.synchronize is device-scope and can be expensive
  PythonProgram is synchronous even when wait=False

accounting
  mem_used = logical active base-Buffer bytes
  allocator LRU/driver physical retention is separate
```
