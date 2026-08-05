# Architecture map

tinygrad is easiest to navigate as a sequence of graph states, not as a stack of
directories. The same `UOp` data structure represents tensor expressions,
symbolic indices, kernel ASTs, linear execution plans, and program artifacts.
The meaning of a node therefore depends on both its `Ops` member and the stage
at which you are inspecting it.

## The end-to-end path

```text
Python Tensor operations
        │  Tensor._apply_uop and mixin methods
        ▼
lazy tensor UOp DAG
        │  transform_to_call
        ▼
storage/effect graph (BUFFER, STORE, AFTER, CALL)
        │  get_kernel_graph + run_rangeify
        ▼
fused kernel dependency graph
        │  create_schedule + memory planning
        ▼
LINEAR(CALL, CALL, ...)
        │  compile_linear → to_program
        ▼
PROGRAM(kernel SINK, linear UOps, SOURCE/assembly, BINARY)
        │  run_linear → pm_exec
        ▼
allocator + loaded program + device launch
```

The arrows are the important part. When debugging, ask which arrow first turns
a correct representation into an incorrect one. When optimizing, ask which
arrow controls the cost you measured.

## Stage-by-stage source map

All links in this table are pinned to the guide's
[source snapshot](source-snapshot.md).

| Stage | Main job | Start reading at |
| --- | --- | --- |
| Tensor frontend | Turn familiar tensor operations into lazy UOps | [`Tensor._apply_uop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L105) and [`tinygrad/mixin/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin) |
| Autograd | Transform a forward UOp graph into gradient expressions | [`Tensor.backward`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L429) and [`mixin/gradient.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/gradient.py) |
| IR infrastructure | Define operations, interned nodes, traversal, patterns, and rewrites | [`Ops`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/__init__.py#L13), [`UOp`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L245), and [`PatternMatcher`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1454) |
| Callification | Choose storage boundaries and make buffers/effects explicit | [`transform_to_call`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/callify.py#L202) |
| Rangeification and fusion | Express shapes as ranges/indexing and split work into kernels | [`run_rangeify`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L180) and [`get_kernel_graph`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L555) |
| Dependency scheduling | Order calls while respecting reads and writes | [`create_schedule`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L29) |
| Buffer resolution and memory planning | Bind parameters to buffers and reuse storage by lifetime | [`create_linear_with_vars`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L171) and [`schedule/memory.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/memory.py) |
| Kernel optimization | Choose equivalent loop, locality, vector, and tensor-core forms | [`Scheduler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L15), [`OptOps`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/__init__.py#L6), and [`apply_opts`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/opt/postrange.py#L339) |
| Kernel lowering | Add loads, GPU dimensions, decompositions, barriers, and control flow | [`full_rewrite_to_sink`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284) and [`codegen/late/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late) |
| Rendering and assembly | Convert lowered UOps to source text or native instructions | [`Renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L59) and [`do_to_program`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L453) |
| Device execution | Allocate buffers, load programs, launch calls, and synchronize | [`device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py) and [`run_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L277) |
| JIT and graph replay | Capture a stable execution plan and replay it with new inputs | [`jit_lower`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L68) and [`_TinyJit`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L222) |

## The UOp roles to recognize first

Do not try to memorize every member of `Ops`. Begin with the roles that mark
representation boundaries:

| UOp | Meaning in this snapshot |
| --- | --- |
| `SINK` | A root that groups outputs or side effects for a transformation. |
| `BUFFER` | Storage with a device, size/shape, dtype, and eventually an allocation. |
| `PARAM` | A normalized external input: buffer or symbolic scalar, distinguished by its metadata. |
| `STORE` / `LOAD` | Explicit memory effects inside a kernel or storage graph. |
| `AFTER` | Pass a value through while recording that specified effects must happen first. |
| `RANGE` | An iteration dimension after shape/index work becomes explicit. |
| `INDEX` | A buffer plus address expression, optionally with validity information. |
| `CALL` | Invoke an opaque unit such as a kernel, copy, view, graph, or special function. |
| `LINEAR` | An ordered list of calls or lowered operations. |
| `PROGRAM` | A compiled-program container with kernel metadata and staged artifacts. |
| `SOURCE` / `BINARY` | Rendered text and compiled bytes attached during program construction. |

Ordinary arithmetic and movement UOps fill in the computation between these
markers. Their exact legality changes by stage and is checked by specs in
[`tinygrad/uop/spec.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py).

## Cross-cutting subsystems

Some work does not fit on one horizontal layer:

- **Symbolic algebra and validity** are used by shapes, indexing, scheduling,
  and code generation. Start in `uop/symbolic.py` and `uop/divandmod.py`.
- **Multi-device execution** changes tensor graphs, scheduling, memory, and
  runtime launch. Start in `schedule/multi.py` and `schedule/allreduce.py`.
- **Effects and mutation** are carried through storage, scheduling, and JIT
  capture. Follow `AFTER`, `STORE`, and the buffer state rather than searching
  for a single mutation module.
- **Target capability** flows backward from the selected `Renderer`: supported
  operations, types, tensor cores, local memory, and launch limits determine
  which lowerings and optimizations are legal.
- **Tracing and validation** wrap many transformations. `VIZ`, `DEBUG`, `SPEC`,
  and process replay reveal different views of the same pipeline.

Imports therefore do not form a perfectly layered architecture. For example,
scheduling reuses symbolic/codegen simplifiers, while codegen reuses range and
multi-device rules. Follow data transformations and public responsibilities,
not import direction alone.

## Locate a problem by its first bad artifact

| Observation | First places to compare |
| --- | --- |
| Tensor result or gradient is wrong before realization | Tensor mixins, dtype rules, gradient rules, raw lazy UOp graph |
| Too many or too few kernels | Callification, rangeification, kernel splitting, dependency schedule |
| Correct kernel boundary but wrong index | Movement lowering, symbolic simplification, range/index graph |
| Kernel AST is correct but generated program is wrong | Decomposition, GPU dimensions, loads/stores, control flow, renderer |
| Program is correct but result or synchronization is wrong | Argument mapping, allocator, runtime program, queues, graph runner |
| Correct result but model is slow | Fusion/materialization and memory plan before tuning an individual kernel |
| One kernel is slow | Applied opts, memory access, vectorization, occupancy, tensor-core use, renderer output |
| First call works but JIT replay fails | Input parameterization, capture exclusions, symbolic values, graph runner |

The table gives a starting hypothesis, not a verdict. Reduce the case and
inspect the artifact on both sides of the suspected transformation.

## NVIDIA branches

An Ubuntu/NVIDIA checkout has two importantly different paths:

```text
CUDA backend: CUDA driver API → CUDA graph support
NV backend:   tinygrad HCQ/userspace queue path → direct NVIDIA queue handling

Both can use CUDA C or PTX-oriented rendering depending on target selection.
```

At this snapshot, automatic discovery tries `NV` before `CUDA` when both can
initialize. Always print `Device.DEFAULT` and set `DEV` explicitly in an
experiment; “ran on my 4090” is not enough to identify the backend.

The relevant entry points are
[`NVDevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L585)
and
[`CUDADevice`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L97).

## Quick reference

When opening an unfamiliar issue, write down five answers:

1. What is the smallest input that reproduces it?
2. Which graph/program state is the earliest one that is wrong or costly?
3. Which transformation produced that state?
4. Which backend-neutral test can pin the intended behavior?
5. Which hardware-specific evidence, if any, remains necessary?

If you cannot answer question 2, gather artifacts rather than reading more
files at random.
