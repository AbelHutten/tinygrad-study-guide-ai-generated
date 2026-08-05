# 1. Orientation: from tensor expression to device launch

## Purpose

tinygrad is small enough that a contributor can follow a model operation all
the way to the machine, but only after choosing the right units of thought. A
`Tensor` method, a UOp, a scheduled kernel, generated source, and a device
launch are related; they are not interchangeable.

This chapter gives those terms precise working meanings. It is enough context
to read an unfamiliar path without assuming that every Python operation runs
immediately or becomes one GPU kernel.

Exact source links below refer to the guide's
[recorded snapshot](../reference/source-snapshot.md). Treat the model as
durable and names such as `schedule_linear` as snapshot-sensitive.

## Prerequisite gate

Continue when you can explain these two ideas in your own words:

- A directed acyclic graph (DAG) records dependencies; a topological order
  places every producer before its consumers.
- In a heterogeneous program, host code launches a *kernel* on a device. One
  launch can make many device threads execute the same kernel over different
  indices.

If either is unfamiliar, read only the relevant short resource:

- Python's [`graphlib` introduction](https://docs.python.org/3/library/graphlib.html)
  for DAGs and topological order.
- NVIDIA's language-neutral
  [CUDA programming-model introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
  through “GPU Hardware Model” for host, device, kernel, launch, threads, and
  streaming multiprocessors. These concepts remain useful on non-CUDA
  backends.
- LLVM's [code generation](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl03.html)
  and [optimizer/JIT introduction](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl04.html)
  if “IR”, “pass”, or “JIT compilation” is new. Skim the introductions and
  constant-folding example; the C++ API is not a prerequisite.

Do not wait until you know compiler theory or CUDA C++. The guide will name
the narrower prerequisite at the point where it becomes useful.

## Mental model

Start with this pipeline:

```text
Python Tensor expression
        │ builds
        ▼
tensor UOp DAG
        │ callification, rangeification, kernel splitting, ordering
        ▼
LINEAR execution plan: kernel/copy/view calls in dependency order
        │ target-specific rewrites, linearization, rendering, compilation
        ▼
PROGRAM objects: source + binary + launch metadata
        │ allocator and runtime
        ▼
buffers, copies, and device launches
```

The arrows are transformations, not necessarily files or permanent IR
classes. tinygrad deliberately reuses `UOp` nodes across several levels. Ask
“what invariant does this graph have *at this point*?” instead of assuming
that “a UOp” names one fixed abstraction level.

### Translate familiar ML terms carefully

| Familiar idea | Useful tinygrad interpretation |
| --- | --- |
| `Tensor` | A Python handle whose central state is a root `UOp`; it is not by itself proof that storage is allocated or computation ran. |
| Computational graph | The dependency graph reachable from one or more Tensor roots. It includes math, shape/view, storage, and effect nodes—not only autograd history. |
| Lazy evaluation | Tensor algebra can build graph nodes now and perform required device work when a value is realized or observed. |
| Intermediate representation (IR) | A graph or sequence on which the compiler can state and preserve invariants while rewriting it. tinygrad represents several IR stages with UOps. |
| Rewrite/pass | A transformation that recognizes graph patterns and replaces them while preserving the property that matters at that stage. |
| Schedule | The ordered work needed to realize outputs: decide materialization and kernel boundaries, respect dependencies, and plan memory. It does not mean an ML learning-rate schedule. |
| Kernel | One compiled device program invocation. It may implement several fused Tensor operations; copies and views can also appear in the execution plan. |
| Lowering | Moving from a more semantic representation toward explicit ranges, indices, loads/stores, control flow, and target instructions. |
| Renderer/compiler/runtime | The renderer emits target source or IR; the compiler turns it into loadable bytes; the runtime binds buffers and launch dimensions and submits it. |
| `TinyJit` | Capture and replay of already-derived work, covered later. It is not the same thing as ordinary Tensor laziness. |

Three consequences prevent a great deal of confusion:

1. **One Tensor operation is not one launch.** Elementwise operations often
   fuse; a copy, reduction boundary, explicit realization, or device boundary
   can split work.
2. **A view need not move data.** Shape and movement operations can remain
   descriptions of indexing until lowering, or become a buffer view.
3. **Autograd and execution are different questions.** Reverse-mode
   differentiation derives gradient computations; scheduling and realization
   decide how computations actually run.

## Source tour

Read the linked ranges, not the entire files. The line numbers are preserved
by commit permalinks.

| Landmark | What to notice at snapshot `874d331` | Source |
| --- | --- | --- |
| `Ops` | The same operation vocabulary includes math, memory, control flow, scheduling-only ops, and program artifacts. | [`tinygrad/uop/__init__.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/__init__.py#L12-L103) |
| `UOp` | A node stores `op`, `dtype`, `src`, `arg`, and `tag`; `src` edges point to dependencies. | [`tinygrad/uop/ops.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L243-L287) |
| `Tensor._apply_uop` | Tensor methods derive a new UOp and wrap it in a new Tensor without executing a runtime call. | [`tinygrad/tensor.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L105-L123) |
| `Tensor.realize` | Realization selects non-virtual roots that do not already have buffer identity, creates a linear plan, and runs it. | [`tinygrad/tensor.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L175-L196) |
| `_Device` | `DEV` selects a backend explicitly; otherwise tinygrad probes available device implementations. | [`tinygrad/device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14-L55) |
| `Compiled` | A device bundles an allocator, one or more renderers, a runtime program type, and optional graph support. | [`tinygrad/device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L328-L363) |

## Lab: prove that the expression is lazy

If you have not prepared tinygrad yet, make the predictions now and run the
lab immediately after [Chapter 2](02-setup.md). Run from the tinygrad study
checkout.

Before executing, predict:

1. the numerical result of `relu(x * 2 + 1)`;
2. whether the output root is realized before and after `.realize()`;
3. whether `relu` appears as an `Ops.RELU`; and
4. how many kernel lines `DEBUG=2` prints on the Python backend.

```bash
CACHEDB=/tmp/tinygrad-guide-orientation.db DEV=PYTHON DEBUG=2 \
  .venv/bin/python - <<'PY'
from collections import Counter
from tinygrad import Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0])
y = (x * 2 + 1).relu()

print("before:", y.uop.is_realized)
print("ops:", {op.name: n for op, n in Counter(u.op for u in y.uop.toposort()).items()})
y.realize()
print("after:", y.uop.is_realized)
print("value:", y.tolist())
PY
```

At the recorded snapshot, ignore timings and process IDs and look for these
structural facts:

```text
before: False
ops: {'CONST': 4, 'BUFFER': 1, 'MUL': 1, 'ADD': 1, 'CMPLT': 1, 'WHERE': 1}
*** PYTHON ... E_4 ...
after: True
value: [0.0, 0.0, 1.0, 3.0]
```

There is no `RELU` UOp. The frontend expresses ReLU using comparison and
selection, and the compiler fuses multiply, add, comparison, and selection
into one launch. The `BUFFER` is the input storage; constants and ALU nodes do
not each imply a separate allocation.

Record the command, prediction, observed op counts, launch line, and the source
landmark that explains each observation. That record is the first artifact in
your study notebook.

### Troubleshooting

- `No such file or directory: .venv/bin/python`: complete Chapter 2, then
  return.
- `ValueError` while reading an environment variable such as `DEBUG`: a shell
  variable with the same name contains a non-integer value. Override it as the
  command above does or run `unset DEBUG`.
- More than one launch: confirm `DEV=PYTHON`, use a fresh process, and compare
  the expression exactly. Other devices may need a host-to-device copy.
- Different op names or counts: first confirm
  `git rev-parse HEAD` is `874d33128b4e4785beea736d97df6716e0321717`.

## Checkpoint

You are ready for setup when you can answer without guessing:

- Why can `y.shape` be available while `y.uop.is_realized` is false?
- Why did the four elementwise UOps (`MUL`, `ADD`, `CMPLT`, and `WHERE`)
  become one launch?
- Why is a `WHERE` in the graph even though the Python expression called
  `relu()`?
- Which word would you use for each artifact: Tensor root, UOp DAG, execution
  plan, generated source, compiled program, launch?

If “graph” still means only “autograd graph” to you, reread the translation
table before continuing.

## Quick reference

| Term | Refresher question |
| --- | --- |
| Tensor root | Which UOp currently represents this value? |
| UOp DAG | What are this node's sources, dtype, shape, and consumers? |
| Realized | Does the value have concrete backing storage with its required work submitted? This is not the same as host synchronization on an asynchronous device. |
| Schedule / `LINEAR` | Which calls must run, and in what dependency order? |
| Kernel | What one device program does this launch execute? |
| Renderer | What target source or IR was emitted? |
| Compiler | What loadable bytes were produced or retrieved from cache? |
| Runtime | How were buffers, scalar arguments, launch dimensions, submission, and synchronization handled? |

[Next: build a reproducible development setup →](02-setup.md)
