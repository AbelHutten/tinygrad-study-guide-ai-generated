# 1. Orientation: from a Tensor expression to running work

## The promise of this chapter

This guide assumes that you can read Python and that tensors, shapes, dtypes,
and operations such as ReLU are familiar from machine learning. It does **not**
assume that you have studied compilers, graphs, GPU programming, or tinygrad's
source code.

You do not need a tinygrad checkout for this chapter. Every required idea is
explained here before it is connected to source. The source links near the end
are a guided confirmation of the explanation, not a substitute for it.

By the end, you should be able to explain this sentence without treating any
word in it as magic:

> tinygrad records Tensor operations as a graph, rewrites and schedules the
> graph, lowers fused work into a target program, and asks a runtime to execute
> that program on a device.

More concretely, you will be able to:

- distinguish a mathematical tensor value, a Python `Tensor` object, a graph
  node, and a buffer containing bytes;
- read a small dependency graph and put its nodes in a valid order;
- explain lazy evaluation and realization;
- define IR, rewrite, lowering, scheduling, fusion, renderer, compiler,
  backend, runtime, kernel, launch, and thread;
- trace `(x * 2 + 1).relu()` through those ideas; and
- recognize what you understand—and what you have deliberately not learned
  yet—when you open the corresponding tinygrad source.

That is a substantial target. Read slowly, work the paper exercises, and
return to the quick reference when later chapters use this vocabulary.

## Start with the value we want

We will carry one expression through the whole chapter:

```python
from tinygrad import Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0])
twice = x * 2
shifted = twice + 1
y = shifted.relu()
```

Ignore implementation for a moment. Mathematically, each operation acts on
each element independently:

| Name | Rule | Value |
| --- | --- | --- |
| `x` | input | `[-2, -1, 0, 1]` |
| `twice` | `x[i] * 2` | `[-4, -2, 0, 2]` |
| `shifted` | `twice[i] + 1` | `[-3, -1, 1, 3]` |
| `y` | `shifted[i]` if positive, otherwise `0` | `[0, 0, 1, 3]` |

Here `i` means one position in the one-dimensional tensor. The tensor has
shape `(4,)`: one axis, containing four elements. Its elements have a dtype,
such as 32-bit floating point, which says how their values are represented.

The three operations are **elementwise**. Element `i` of an output depends
only on element `i` of its input. That fact will later make fusion possible.
A reduction such as `x.sum()` is different: one output depends on several
input elements.

The Python numbers `2`, `1`, and `0` are **scalars**: each represents one
value, with no tensor axes. tinygrad represents a scalar constant with shape
`()`. When a scalar and a tensor participate in an elementwise operation,
**broadcasting** lets the one scalar value apply at every tensor position; it
does not require the user to construct `[2, 2, 2, 2]`. More generally,
broadcasting defines when different but compatible shapes can act as though
they had a common shape.

The operands also need a compatible dtype. **Dtype promotion** is the rule for
choosing that common dtype—for example, combining a floating-point Tensor and
a Python integer scalar without turning the Tensor into integers. Exact
broadcasting and promotion rules matter later, but this example needs only the
idea that tinygrad resolves both before recording one typed elementwise
operation over shape `(4,)`.

### Four things that are easy to conflate

At this point, keep four distinct meanings in view:

1. **Mathematical value** — the abstract numbers shown in the table. The same
   value could be computed by tinygrad, PyTorch, NumPy, or pencil and paper.
2. **Python `Tensor` object** — an ordinary Python object referred to by a
   variable such as `shifted`. It has properties and methods and participates
   in Python operator overloading.
3. **Computation description** — tinygrad's record of how a value is obtained,
   such as “add the result of this multiplication to the constant 1.” In the
   first part of tinygrad this description is made of UOps.
4. **Buffer** — storage containing encoded elements. A buffer may live in host
   memory or device memory. Its bytes are physical representation, not the
   mathematical idea of the tensor.

A `Tensor` is therefore not simply “an array of bytes.” It is a Python wrapper
around a computation or stored value, together with enough information to
describe its shape, dtype, and device. A lazy intermediate can have a perfectly
usable `Tensor` object and computation description before it has its own
materialized buffer.

## What Python does immediately

Python does not know tensor multiplication as a built-in operation. When it
evaluates:

```python
twice = x * 2
```

it approximately performs this language-level interaction:

```python
twice = x.__mul__(2)
```

Calling `__mul__` immediately enters ordinary Python helper code. `__mul__`
itself delegates to `mul`; the call chain then wraps the scalar as a Tensor
constant, applies broadcasting and dtype-promotion rules, and constructs a new
`Tensor` whose underlying computation says “multiply these sources.” The call
can return without running the element-by-element multiplication on the target
device.

This separates two kinds of work:

- **Python/frontend work now:** execute methods, validate arguments, infer
  metadata, and build computation nodes.
- **Numerical/device work later:** allocate output storage if needed and
  compute the output elements.

This is **lazy evaluation**: tinygrad can defer numerical work until a value
must be materialized or observed. “Lazy” does not mean that the Python line did
nothing. It built a more detailed recipe.

An eager system could instead execute a multiplication immediately, write all
four results to a new buffer, then do the same for addition, then ReLU. Both
approaches can produce `[0, 0, 1, 3]`. Laziness gives tinygrad an opportunity
to see all three operations together before deciding how to run them.

### Information available before the elements are computed

For the running example, tinygrad can usually know all of the following while
`y` is still lazy:

| Information | For `y` | Why it can be known early |
| --- | --- | --- |
| Shape | `(4,)` | Elementwise rules preserve the input shape. |
| Dtype | floating point | Promotion rules combine the input dtype and scalar constants. |
| Device | the selected target | Inputs and operation rules determine placement. |
| Dependencies | compare, add, multiply, input, constants | They were recorded while Python evaluated the expression. |
| Numerical elements | eventually `[0, 0, 1, 3]` | These normally require executing the described work. |

This is why printing `y.shape` need not do the same work as calling
`y.tolist()`. A shape is metadata. `tolist()` asks for the actual elements in
host-accessible form, so it has to ensure the computation is complete and the
data is available to Python.

## Computations form a graph

The word **graph** here has a precise but small meaning. A graph consists of:

- **nodes**, which represent things such as an input, a constant, or an
  operation; and
- **edges**, which represent relationships between nodes.

In a computation graph, an edge points along a dependency relationship. If an
addition needs the result of a multiplication, the addition has the
multiplication as a source. In this chapter, a dataflow arrow points from a
source/producer to its consumer:

```text
input x ------> multiply
constant 2 ---> multiply
multiply -----> add
constant 1 ---> add
add ----------> ReLU
```

Later dependency listings use `CONSUMER <- SOURCE_1, SOURCE_2`; the arrow
still points from each source toward its consumer, but putting the consumer on
the left makes its input list easy to scan. In tinygrad code, the consumer's
`src` field stores references back to its source nodes. Starting at an output
and following `src` therefore walks opposite the dataflow arrows, from consumer
back toward producers.

Vocabulary for this drawing:

- The input and constants are **leaves** here: they do not depend on earlier
  arithmetic nodes.
- `multiply` is a **source** or **producer** for `add`.
- `add` is a **consumer** of `multiply`.
- `ReLU` is the **root** of the expression we are currently studying. Starting
  at it and repeatedly following sources reaches everything needed for `y`.
- The graph is **directed** because “A is a source of B” has a direction.
- It is **acyclic** because following dependencies cannot eventually lead back
  to the same operation. A graph that is both directed and acyclic is a
  **directed acyclic graph**, or **DAG**.

The word “root” is a viewing convention, not a claim that the root happened
first. We often begin at the desired output and walk backward to its sources.

### A DAG is not necessarily a tree

In a tree, every non-root node has only one consumer along that drawing. A DAG
can share a node. ReLU makes this visible in tinygrad because it is expressed
using smaller operations:

```text
ReLU(z) = choose z when z > 0, otherwise choose 0
```

For `z = shifted`, both `shifted` and the constant zero are reused. Listing
each node once makes that sharing unambiguous:

```text
MUL     <- input x, CONST(2)
ADD     <- MUL, CONST(1)
COMPARE <- CONST(0), ADD
WHERE   <- COMPARE, ADD, CONST(0)
```

`ADD` is consumed by both `COMPARE` and `WHERE`; the same `CONST(0)` is also
consumed by both. If we drew the graph as a tree, we would have to duplicate
those shared nodes or subtrees, which could misleadingly suggest extra
operations. Object identity and shared edges matter.

### Topological order

A **topological order** lists every source before any operation that consumes
it. One valid order for the simplified graph is:

```text
input, constant 2, multiply, constant 1, add,
constant 0, compare, choose
```

There can be several valid topological orders. The constants have no
dependency on one another, so their relative positions can move. This order is
invalid:

```text
add, multiply, input, ...
```

because `add` appears before a source it needs.

Topological ordering does not itself execute anything. It is a way to traverse
or arrange dependencies safely. Compilers use it for tasks such as evaluating
facts, applying transformations, and ordering work.

## tinygrad's first graph: UOps

tinygrad represents many program ideas with objects called **UOps**. For now,
think of a UOp as one labeled node with five useful fields:

| Field | Question it answers | Example |
| --- | --- | --- |
| `op` | What kind of node is this? | `MUL`, `ADD`, `CONST`, `BUFFER` |
| `dtype` | What type does the value have? | 32-bit float or boolean |
| `src` | Which UOps does this node depend on? | the two inputs of `ADD` |
| `arg` | What extra information belongs to this node? | a constant's value |
| `tag` | Is there optional annotation attached? | normally irrelevant here |

Do not try to memorize every possible `op`. The complete enumeration mixes
ideas from many stages of the compiler, so reading it now would expose names
without the models needed to interpret them. Chapter 5 builds a systematic UOp
vocabulary.

For the running expression, the important arithmetic portion of the UOp graph
is below. A name appearing on the right of more than one line still refers to
one node:

```text
MUL      <- BUFFER(x), CONST(2)
ADD      <- MUL, CONST(1)
CMPLT    <- CONST(0), ADD
WHERE(y) <- CMPLT, ADD, CONST(0)
```

Starting at `WHERE`, follow each node's `src` references backward toward its
producers:

1. `WHERE` needs a boolean condition, a value for the true case, and a value
   for the false case.
2. Its condition is `CMPLT(0, ADD)`, which means “is zero less than the
   shifted value?” This is the same question as `shifted > 0`.
3. Its true value is the shared `ADD`; its false value is zero.
4. `ADD` needs `MUL` and the constant 1.
5. `MUL` needs the input buffer and the constant 2.

There is no special `RELU` node in this graph. The Tensor method decomposes
ReLU into comparison plus selection. This is our first example of one
user-facing operation being represented by several simpler operations.

`BUFFER(x)` above is a deliberately simplified label. A real `BUFFER` node
also has a size source, which is why the runnable observation later shows a
seemingly surprising integer constant `4`. That `4` describes the input
buffer's element count; it is not a fourth arithmetic constant.

### One representation, changing roles

An **intermediate representation** (**IR**) is a program form designed for
software to analyze and transform. “Intermediate” means it lies between what
the user wrote and what the machine ultimately executes.

A UOp graph is an IR. It is more explicit than the original Python expression:
ReLU has become comparison and selection, dependencies are edges, and dtype
information is attached to nodes. It is still less explicit than a GPU
program: there are not yet thread indices, loads, stores, or launch sizes in
this frontend graph.

tinygrad continues using UOps as the graph moves through different compiler
stages. The set of operation kinds and the meaning of a graph region change.
Consequently, “this is a UOp” does not tell you where you are in the pipeline.
You must also ask:

- What is the root operation?
- Which operation kinds are present?
- Which transformation just ran?
- What invariant should this stage satisfy?

An **invariant** is a property expected to remain true. For example, a rewrite
may change how an expression is represented while being required to preserve
its output value for every valid input.

In compiler writing, **semantics** simply means the program's defined meaning
or behavior: which values and observable effects it must produce. A
“semantics-preserving” change may alter the representation or execution plan
without altering that required behavior.

## What a compiler contributes

A compiler is not limited to translating C into machine code. More generally,
it takes a program in one representation and produces another representation
that is suitable for a later purpose while preserving required behavior.

tinygrad has a **frontend**: the layer where Python `Tensor` operations are
validated and recorded. It also has a compiler pipeline that transforms those
recorded operations into executable work.

The following words describe different jobs in that pipeline.

### Rewrite and pass

A **rewrite** replaces one graph pattern with another. A simple algebraic
example is:

```text
ADD(x, 0)  ->  x
```

This replacement is valid only under the relevant semantic rules. Floating
point edge cases, dtypes, shapes, side effects, and special values can make an
apparently obvious identity less universal than it looks.

A **pass** is an organized traversal or transformation of an IR. A pass may
apply many rewrite rules, collect information, or **normalize** a graph—put
several equivalent forms into one preferred form expected by the next pass. A
pass has:

- an input form it expects;
- an output form it promises; and
- properties it must preserve.

“Optimization” is not a synonym for every rewrite. Some rewrites merely make
later work possible or replace a convenient high-level operation with supported
lower-level ones. An **optimization** specifically tries to improve a cost such
as execution time, memory traffic, or compilation time without violating
correctness.

Rewrite and lowering are not competing names for the same thing. A rewrite is
a **mechanism** for changing a representation; lowering describes the
**direction and purpose** of a change toward something more explicit. A
lowering pass can be implemented by a sequence of graph rewrites.

### Lowering

**Lowering** replaces a higher-level idea with a more explicit or
target-specific one. For example:

```text
elementwise operation over shape (4,)
```

can eventually become something resembling:

```text
for i from 0 through 3:
    z = input[i] * 2 + 1
    output[i] = z if z > 0 else 0
```

On a GPU, the loop may instead be distributed over threads. Lowering does not
mean “make the program worse” or simply “move it downward in a file.” It means
committing previously implicit semantics to more concrete mechanisms.

### Scheduling

**Scheduling** decides how required work is divided and ordered. There are two
scales that later chapters will keep separate:

- An **execution schedule** orders compute programs, copies, and other calls so
  dependencies are respected.
- A **kernel schedule** chooses how the work inside one program maps to loops,
  threads, vector-width choices (processing several scalar lanes together),
  and memory.

At this snapshot, tinygrad represents its ordered multi-call plan with a
`LINEAR` UOp. You only need the plain-language idea in this chapter: before
execution, tinygrad needs an ordered plan of calls. The name `LINEAR` is also
reused inside a compiled `PROGRAM` for an ordered sequence of lowered
instructions. Context distinguishes the outer call plan from the inner
instruction sequence; Chapter 3 shows both.

### Bufferization

Graph values are logical results; they do not all need their own stored array.
**Bufferization** chooses which values require buffers and introduces the
corresponding storage, read, and write boundaries. An output that must survive
between two programs needs storage. A short-lived value inside one fused
program may not. Bufferization is therefore where questions about fusion,
materialization, and memory start becoming an explicit execution plan.

## Why laziness enables fusion

Suppose the three operations in the running example execute eagerly. A simple
implementation might do this:

```text
program 1: read x, compute x * 2, write temporary A
program 2: read A, compute A + 1, write temporary B
program 3: read B, compute ReLU, write y
```

That approach may allocate two intermediate buffers, write them to memory,
read them back, and invoke three programs.

Because lazy tinygrad sees the whole elementwise chain, it can potentially
**fuse** the operations into one program:

```text
for each output position i:
    value = x[i]
    value = value * 2
    value = value + 1
    y[i] = value if value > 0 else 0
```

The fused version can keep `value` as a short-lived local value instead of an
entire stored tensor. It can avoid writing and rereading full intermediate
buffers and the overhead of invoking extra programs.

Fusion is not automatically legal or faster in every case. Boundaries can be
required or useful because of:

- a value explicitly requested in storage;
- a copy between devices;
- a reduction or operation with a different iteration structure;
- mutation or another observable side effect;
- hardware limits on fast per-thread or per-group storage, program size, or
  worker count; or
- reuse that makes recomputation more expensive than materialization.

So “fewer kernels is better” is a hypothesis, not a law. The compiler must
preserve semantics, and performance must eventually be measured.

### Materialization and realization

To **materialize** a lazy value is to compute it into storage. tinygrad calls
the transition that ensures a Tensor has such backing **realization**.

These operations can force or request observation:

```python
y.realize()  # request computation/backing storage, keep a Tensor
y.tolist()   # make the elements available as Python values
y.numpy()    # make the elements available as a NumPy array
y.item()     # obtain the element of a scalar Tensor
```

They do not all have exactly the same boundary. In particular, accelerator
work is commonly asynchronous: the host can submit work and continue before
the device finishes. Having scheduled or backed work is not always equivalent
to “the CPU waited until the GPU was done.” A host readback such as `tolist()`
must ultimately make the requested data safe for the host to read and therefore
introduces whatever completion and copying are necessary.

There are also special virtual cases, including constants and empty values,
that need not correspond to an ordinary allocated compute buffer. Treat
“realized?” as an implementation question about representation, not as a
universal stopwatch for device synchronization.

## The machine boundary: host, device, and backend

Now we can introduce the GPU vocabulary without assuming prior GPU knowledge.

The **host** is the CPU-side environment running Python and tinygrad's compiler
and runtime code. A **device** is an execution target, usually with a class or
region of memory that it can access, such as the CPU itself or an NVIDIA GPU.
“Host” and “device” describe roles; on a CPU backend they may use the same
physical machine, while on a discrete GPU they have separate memory and
execution machinery.

A tinygrad **backend** is the implementation selected for a device path. For
example, `CPU`, `CUDA`, and `NV` reach different renderer/runtime machinery.
The mathematical `ADD` operation does not specify which backend will execute
it.

The backend has to provide mechanisms such as:

- allocating and freeing buffers;
- copying data where necessary;
- converting a lowered program into a target-usable form;
- loading that program;
- invoking it with buffers and launch dimensions—the number and arrangement
  of parallel workers requested; and
- synchronizing when a caller needs completion.

Different layers own those jobs. A **renderer** converts lowered IR into target
source text or another target representation. A target **compiler** converts
that representation into executable bytes. **Runtime** is a useful broad name
for the execution-side machinery, but it is not one all-purpose object in this
tinygrad snapshot: `Buffer` and an allocator own storage and copies, a backend
`Program` loads and invokes one compiled program, and the compiled device
exposes synchronization. The backend bundles these contracts and can
specialize them differently. Chapter 12 draws their exact boundaries.

### Program, kernel, launch, and thread are not synonyms

A **kernel** is a program or function intended to run across an accelerator's
parallel workers. A **launch** is one invocation of that kernel with particular
arguments and dimensions. The distinction is the same as the distinction
between a Python function and calling that function:

```python
def f(x):       # function
    return x+1

f(3)            # one call
f(8)            # another call of the same function
```

Likewise, one compiled kernel can be launched many times. Informal compiler
logs sometimes use “kernel” as shorthand for one execution event, but when
reasoning about bugs or performance, ask whether you mean the program or its
invocation.

A **thread** is one logical worker in a GPU launch. Many threads run the same
kernel code but receive different index values. A simplified elementwise GPU
kernel might mean:

```text
kernel fused(x, y, number_of_elements):
    i = this_thread's_global_index
    if i < number_of_elements:
        z = x[i] * 2 + 1
        y[i] = z if z > 0 else 0
```

Read it line by line:

1. Every thread begins in the same kernel program.
2. Each thread obtains a different `i` from its position in the launch.
3. The bounds check prevents extra launched threads from accessing beyond the
   four-element tensor.
4. A valid thread loads one element, computes its result, and stores one
   element.

Actual GPUs group threads and have multiple memory spaces, synchronization
rules, and performance constraints. Chapters 9, 10, and 14 build those models.
For now, the key transition is from “apply this tensor operation elementwise”
to “each indexed worker performs scalar operations for one or more elements.”

### Calls, copies, and synchronization

An execution plan may contain more than compute work. It can include:

- a compute-program call;
- a copy between buffers or devices;
- a view or bookkeeping operation; or
- a wait or synchronization requirement.

Therefore, do not infer “one GPU launch” merely from “one scheduled call.” The
kind of call and selected backend matter. The `PYTHON` backend used in early
labs executes lowered work in Python; its scheduled compute call is useful for
understanding boundaries, but it does not create GPU threads.

On an accelerator, the host commonly queues a launch and returns before all
threads finish. **Synchronization** creates a required ordering or waits for
completion. Synchronization can be necessary for correctness, but unnecessary
waiting also destroys concurrency. Later runtime chapters distinguish enqueue
time, device execution time, and host-visible completion.

## Reassemble the entire route

We now have enough vocabulary to describe the journey without unexplained
gaps. Exact internal names belong to the pinned tinygrad snapshot and can move;
the questions in the middle column are the durable part.

| Stage | Question answered | Running example |
| --- | --- | --- |
| Python Tensor frontend | What operation did the user request, with what metadata? | `*`, `+`, and `relu()` methods construct Tensors. |
| Frontend UOp DAG | What values depend on what? | `BUFFER`, `MUL`, `ADD`, `CMPLT`, `WHERE`, and constants. |
| Graph rewrites | Can the graph be normalized, simplified, or decomposed while preserving semantics? | ReLU is already comparison plus selection; later rules simplify supported forms. |
| Bufferization and execution scheduling | Which results need storage, which operations can share a program, and in what order must calls occur? | The elementwise chain can become one compute call with input and output buffers. |
| Kernel lowering and optimization | How will positions map to explicit ranges, indices, loads, arithmetic, and stores? | Each output index loads `x[i]` and computes the fused expression. |
| Rendering | How is the lowered kernel spelled for this target? | C-like source for CPU/CUDA, or another target representation. |
| Target compilation | What executable bytes can the runtime load? | The rendered program is compiled for the selected backend. |
| Runtime execution | Which buffers and dimensions are used for this invocation? | Launch or call the fused program and eventually make `y` observable. |

At the pinned snapshot, some concrete artifact names you will meet are:

```text
Tensor-rooted UOp DAG
    -> normalized/fused kernel work
    -> LINEAR ordered call plan
    -> PROGRAM containing SOURCE and BINARY
    -> runtime invocation
```

This is only a signpost. Chapter 3 will generate these artifacts in a real
process, and later chapters will explain each transformation. You are not
expected to understand `LINEAR`, `PROGRAM`, `SOURCE`, or `BINARY` internals
from their names alone.

## Two neighboring graphs that answer different questions

Machine-learning libraries also build an **autograd graph** to compute
gradients. It is related to the forward computation but answers a different
question:

- the compiler graph asks how to represent and execute values; and
- the autograd relationship asks how an output's derivative propagates to
  inputs.

An operation can be decomposed or fused for execution without changing the
mathematical gradient it must implement. Conversely, `.detach()` affects
gradient tracking but does not mean “copy this Tensor to the CPU.” Chapter 4
separates these mechanisms carefully. In this tinygrad snapshot, `backward()`
traverses the forward UOp DAG and constructs new gradient UOps; “autograd
relationship” is a conceptual distinction, not the name of an unrelated graph
class maintained beside the UOps.

A **view** is another distinct idea. Reshape, permute, expand, and similar
operations can often reinterpret how indices map to the same underlying
storage without immediately moving data. Shape is therefore not just a buffer
dimension written in stone; it participates in indexing semantics. Chapter 8
develops that model.

## Guided source tour: confirm one causal chain at a time

Only start this section after the preceding model makes sense. These links
target the guide's pinned tinygrad commit, so they will not drift as `master`
changes.

For every stop, use the same method:

1. Read the question before opening the source.
2. Read only the linked range.
3. Translate it into the plain-language answer given here.
4. Ignore unfamiliar helpers unless they block that translation.

### Stop 1: why is the root `WHERE`, not `RELU`?

Read [`relu`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L669-L678).

Question: which smaller Tensor operations implement ReLU?

Translation: the method constructs a condition asking whether each value is
positive, then selects the original value when true and zero otherwise. The
comment concerns the derivative exactly at zero; you may record that fact and
defer its autograd consequences to Chapter 4.

Now read [`__gt__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L315-L319).

Question: why did our graph show `CMPLT(0, shifted)` for `shifted > 0`?

Translation: greater-than reuses a less-than operation with operand order
reversed. It records “zero is less than shifted.” This is a representation
choice, not a reversal of the mathematical answer.

Finally read [`where`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L420-L432).

Question: after input conversion and broadcasting, what operation is recorded?

Translation: the boolean Tensor becomes the first source of a `WHERE` UOp;
the true and false values become the other sources. You do not yet need to
unpack `ufix`, `_broadcasted`, or every dtype rule.

### Stop 2: how does `x * 2` reach a `MUL` UOp?

This stop has more links because it follows one causal chain across abstraction
boundaries. Each range has one job:

1. [`__mul__` lines 267–268](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L267-L268)
   merely delegates to `mul`.
2. [The return on line 138 of `mul`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L138)
   chooses `Ops.MUL` and enters the shared binary-operation helper.
3. [`ufix`, `_broadcasted`, and `_binop` lines 18–34](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L18-L34)
   wrap the non-Tensor operand, choose a common dtype, and call `alu`. On this
   first read, follow lines 18–24 and 30–34; the weak-dtype and invalid-value
   cases on lines 25–29 are later details. Despite the helper's name, the
   output-shape rules live deeper in UOp shape inference; this range mainly
   exposes operand wrapping and promotion.
4. [`UOp.ufix` lines 602–604](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L602-L604)
   sends an ordinary Python value to
   [`UOp.const` lines 623–629](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L623-L629).
   This is how `2` first becomes a `CONST` UOp; promotion may then replace it
   with a dtype-compatible constant.
5. [`Tensor._apply_uop` and `Tensor.alu` lines 105–117](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L105-L117)
   extract the UOps, perform UOp-level arithmetic, and wrap the result in a new
   Tensor.
6. [`UOp.alu` on line 622](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L622)
   constructs the requested operation with the participating UOps as sources.

The causal chain is:

```text
Python x * 2
  -> Tensor.__mul__
  -> elementwise mul chooses Ops.MUL
  -> ufix turns Python 2 into a CONST UOp
  -> broadcasting/type promotion make compatible operands
  -> Tensor.alu with operation MUL
  -> UOp.alu constructs UOp(MUL, sources=(x_uop, two_uop))
  -> a new Tensor wraps that UOp
```

This chain is worth understanding; individual helper names are not worth
memorizing. The important result is that Python operator overloading runs now,
puts the operands into a compatible Tensor/dtype form, creates a dependency
node, and returns a Tensor wrapper. It does not need to run the target's
elementwise multiplication at that moment.

### Stop 3: what is actually stored in a UOp?

Read only the field declaration at [`class UOp`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L243-L250).

You should now be able to identify `op`, `dtype`, `src`, `arg`, and `tag` from
the earlier table. Stop before the following methods. They solve problems that
have not yet been introduced.

Then read [`UOp.toposort`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L303-L315).

The implementation uses an explicit stack and a “visited” flag. A node is
added to the output only after its sources have been processed. That is the
code form of “sources before consumers.” You do not need to reproduce the
algorithm from memory.

### Stop 4: what causes data observation?

First read the tiny [`shape` and `dtype` properties](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L143-L150).
They obtain metadata from the UOp; they do not ask for a Python list of all
elements.

Next read [`Tensor.realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L191-L196).
At this point, translate it only as: select roots that genuinely need backed
work, build their plan, run it, and return the Tensor. The helper names are the
subjects of Chapters 3 and 7.

Finish with three small steps instead of one large file range:

1. In [`Tensor._buffer` lines 238–245](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L238-L245),
   skip the JIT-capture guard on lines 239–242. Lines 243–245 make the value
   contiguous where necessary, realize it, obtain its `Buffer`, and ensure the
   storage is allocated.
2. In [the core of `Tensor.data`, lines 258–265](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L258-L265),
   lines 258–260 are special cases for weak dtypes, empty tensors, and symbolic
   shapes. Focus on lines 261–265: obtain the buffer and expose a host-readable
   memory view with the right dtype format and shape. Buffer copying and
   synchronization stay behind that call and belong to Chapter 12.
3. [`tolist` line 288](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L288)
   converts that host-readable data to ordinary Python values. The preceding
   branches handle special dtypes and empty shapes; they are not required for
   this example.

Together the stops establish the broad sequence without requiring you to sift
through unrelated machinery: metadata can be read directly, realization
creates needed backing, and value observation obtains host-readable data.

### Source that is intentionally deferred

Do **not** use the following as Chapter 1 reading assignments:

- the complete `Ops` declaration, because it combines vocabulary from almost
  every compiler stage—Chapter 5 organizes it;
- the full device and `Compiled` classes, because allocator, renderer,
  compiler, program, and runtime contracts need Chapter 12;
- scheduling and rangeification internals, because their inputs are not yet
  meaningful—Chapters 7 and 10 build them; or
- NVIDIA runtime files, because register packets and queues do not teach the
  basic host/device model—Chapter 14 reaches them with the required hardware
  vocabulary.

If a guide link leaves you looking at unexplained declarations, the reading
path has jumped ahead. Close it and return only when the guide has supplied a
specific question that those declarations answer; reading more untargeted
lines rarely fixes the ordering problem.

## Paper lab: build and execute a DAG by hand

No installation is required. Consider:

```python
x = Tensor([-2.0, 0.0, 3.0])
a = x + 1
b = a * a
c = b - 2
y = c.relu()
```

Answer these questions before opening the worked answer:

1. Compute the mathematical values of `a`, `b`, `c`, and `y`.
2. Draw one node for each input, scalar constant, arithmetic operation,
   comparison, and selection. Draw an edge from each source to its consumer.
3. Which nodes or source references are shared? Why is this graph a DAG but
   not an ordinary tree?
4. Give one valid topological order.
5. Which facts could tinygrad know before computing `y`'s elements?
6. Sketch the naive eager work using intermediate buffers.
7. Sketch one fused loop or GPU-style indexed kernel.
8. If `a.realize()` is inserted immediately after `a = x + 1`, what boundary
   might that introduce? Does it change the mathematical answer?
9. Label these as Python object, IR node, storage, program, or invocation:
   `a`, `MUL`, an allocated array of three floats, fused kernel source, and one
   launch of that kernel.

??? success "Worked answer"

    **1. Values**

    The values are:

    ```text
    a = [-1, 1, 4]
    b = [1, 1, 16]
    c = [-1, -1, 14]
    y = [0, 0, 14]
    ```

    **2. Dependency graph**

    A simplified dependency listing is:

    ```text
    ADD(a)   <- x, CONST(1)
    MUL(b)   <- ADD(a), ADD(a)
    SUB(c)   <- MUL(b), CONST(2)
    COMPARE  <- SUB(c), CONST(0)
    WHERE(y) <- COMPARE, SUB(c), CONST(0)
    ```

    As in the running example, the exact tinygrad form can express `>` as
    reversed `CMPLT`, subtraction through more primitive arithmetic, and ReLU
    as `WHERE`. The dependency structure is the point of this drawing.

    **3. Sharing and the tree distinction**

    `ADD(a)` occupies both source positions of `MUL`; it is one addition, not
    two independent additions. More importantly for the tree distinction,
    `SUB(c)` is consumed by both `COMPARE` and `WHERE`, and `CONST(0)` is also
    consumed by both. An ordinary expression tree would have to duplicate
    those shared nodes or their subtrees. The DAG retains one node and lets
    several edges refer to it.

    **4. Topological order**

    One valid order is: `x`, constant 1, `ADD(a)`, `MUL(b)`, constant 2,
    `SUB(c)`, constant 0, comparison, selection. Other orders of independent
    constants are valid.

    **5. Early information**

    Shape `(3,)`, resulting dtype, device, and dependencies can normally be
    known before all numerical elements are computed.

    **6. Naive eager plan**

    A naive eager plan writes `a`, then reads it to write `b`, reads `b` to
    write `c`, and reads `c` to write `y`: four compute invocations and several
    complete intermediate buffers.

    **7. One fused form**

    ```text
    for i in 0..2:
        av = x[i] + 1
        bv = av * av
        cv = bv - 2
        y[i] = cv if cv > 0 else 0
    ```

    A GPU form gives different `i` values to threads and applies a bounds
    check.

    **8. Explicit realization**

    Explicitly realizing `a` requests storage for it and can split later work
    from the addition. The final value remains `[0, 0, 14]`; the execution plan
    and memory traffic may change.

    **9. Artifact labels**

    `a` is a Python `Tensor` object; `MUL` labels an IR operation/node; the
    allocated float array is storage/a buffer; the fused kernel source is a
    program representation; one launch is an invocation of that program.

You pass this lab when you can justify each edge and each ordering constraint,
not merely reproduce the answer's drawing.

## Optional runnable observation after Chapter 2

This observation belongs conceptually here but operationally depends on the
pinned environment created in [Chapter 2](02-setup.md). Skip it on the first
read if tinygrad is not installed; return after setup.

Run from the `tinygrad-study` checkout:

```bash
CACHEDB=/tmp/tinygrad-guide-orientation.db DEV=PYTHON DEBUG=2 \
  .venv/bin/python - <<'PY'
from tinygrad import Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0])
twice = x * 2
shifted = twice + 1
y = shifted.relu()

for name, tensor in (("x", x), ("twice", twice), ("shifted", shifted), ("y", y)):
  print(f"{name:7} root={tensor.uop.op.name:6} shape={tensor.shape} "
        f"dtype={tensor.dtype} realized={tensor.uop.is_realized}")

nodes = list(y.uop.toposort())
node_id = {u: f"N{i}" for i, u in enumerate(nodes)}
print("\ngraph:")
for u in nodes:
  sources = ",".join(node_id[s] for s in u.src) or "-"
  value = f" value={u.arg}" if u.op.name == "CONST" else ""
  print(f" {node_id[u]} {u.op.name:6} shape={u.shape} src={sources}{value}")

print("\nbefore:", y.uop.op.name, y.uop.is_realized)
y.realize()
print("after: ", y.uop.op.name, y.uop.is_realized)
print("value: ", y.tolist())
PY
```

`CACHEDB` isolates this study run's cache. `DEV=PYTHON` selects the portable
Python backend, and `DEBUG=2` asks tinygrad for a compact, timed execution
line. That debug level can request waiting in order to measure work, so never
use it as evidence that an ordinary accelerator `realize()` always
synchronizes. These are shell environment variables for this one command, not
Python assignments.

The stable structure at the pinned commit is:

```text
opened device PYTHON from pid:...
x       root=BUFFER shape=(4,) dtype=dtypes.float realized=True
twice   root=MUL    shape=(4,) dtype=dtypes.float realized=False
shifted root=ADD    shape=(4,) dtype=dtypes.float realized=False
y       root=WHERE  shape=(4,) dtype=dtypes.float realized=False

graph:
 ... CONST  shape=()   ... value=0.0
 ... CONST  shape=()   ... value=4
 ... BUFFER shape=(4,) ...
 ... CONST  shape=()   ... value=2.0
 ... MUL    shape=(4,) ...
 ... CONST  shape=()   ... value=1.0
 ... ADD    shape=(4,) ...
 ... CMPLT  shape=(4,) ...
 ... WHERE  shape=(4,) ...

before: WHERE False
*** PYTHON ...
after:  BUFFER True
value:  [0.0, 0.0, 1.0, 3.0]
```

Node numbers and timing text are incidental. Interpret the observations:

- The device-open line confirms the selected backend; its process identifier
  varies from run to run.
- `x` came from concrete Python-list data, so on this backend it begins as a
  backed `BUFFER`. The derived `twice`, `shifted`, and `y` begin as lazy
  operation roots.
- Shape and dtype were available before realization.
- `shape=()` identifies each scalar constant. `dtypes.float` is tinygrad's
  displayed default floating dtype here.
- The topological traversal put sources before their consumers.
- `ADD` and `CONST(0)` each appear once even though both `CMPLT` and `WHERE`
  use them.
- The constant `4` belongs to the input `BUFFER`'s size description; arithmetic
  constants are `2`, `1`, and `0`.
- `y.realize()` ran one scheduled Python-backend compute program and replaced
  the Tensor's lazy root with backed storage.
- The `*** PYTHON` line is a Python-backend program invocation, not a GPU
  launch and not evidence about GPU thread behavior. `E_4` is a generated
  program name and `arg 2` reports its two buffer arguments here; timing,
  memory, and rate fields are diagnostics that can vary.
- `tolist()` then observed the expected elements.

If the import fails or output differs, do not modify the code until you have
completed Chapter 2's identity checks. The most common cause is running from
the wrong directory or importing a different tinygrad commit.

## Checkpoint: explain, predict, distinguish

Before continuing, say a coherent version of the following aloud or in a
study notebook:

> When Python evaluates `y = (x * 2 + 1).relu()`, tinygrad's overloaded Tensor
> methods immediately build a UOp dependency DAG. ReLU becomes comparison and
> selection, so the root is `WHERE`. The output shape, dtype, and dependencies
> can be known while its elements remain lazy. Because the operations are
> elementwise, scheduling may fuse them into one program that reads `x` and
> writes `y` without full intermediate buffers. Lowering makes iteration,
> indexing, loads, and stores explicit; rendering and target compilation create
> a program the selected backend can use; the runtime invokes that program.
> On a GPU, a kernel is the program and a launch is one invocation across
> indexed threads. Observing the result from Python eventually requires the
> data to be complete and host-accessible.

You are ready for Chapter 2 if you can also answer:

1. Why does laziness create an opportunity for fusion?
2. Why is a UOp DAG not necessarily a tree?
3. What makes an ordering topological?
4. What is the difference between a Tensor and a buffer?
5. What is the difference between a rewrite and lowering?
6. What is the difference between a renderer and a runtime?
7. What is the difference between a kernel and a launch?
8. Why does `realized=True` not universally prove that the host just waited for
   all GPU work?

If an answer is uncertain, revisit the corresponding section. No external
reading is required to pass this checkpoint.

## Quick reference

| Term | Working meaning in this guide |
| --- | --- |
| Tensor | Python wrapper around a typed, shaped computation or stored value. |
| Buffer | Storage containing encoded elements in host or device-accessible memory. |
| Node / edge | One graph item / one dependency relationship. |
| Source / producer | A node whose result another node needs. |
| Consumer | A node that uses one or more source nodes. |
| Root | The output node from which the required dependency region is viewed. |
| DAG | Directed graph with no dependency cycle; nodes may be shared. |
| Topological order | Sources appear before their consumers. |
| Eager / lazy evaluation | Compute each result immediately / record work that can be computed later. |
| Frontend | Python-facing layer that validates Tensor operations and records their meaning. |
| UOp | tinygrad IR node carrying an operation, dtype, sources, argument, and optional tag. |
| IR | Machine-processable representation between user intent and executable code. |
| Rewrite | Replace one representation pattern with another under a preservation rule. |
| Pass | Organized analysis or transformation over an IR. |
| Lowering | Make higher-level semantics more explicit or target-specific. |
| Schedule | Divide and order work; at kernel scale, choose its mapping to hardware. |
| Bufferization | Choose stored values and introduce explicit storage/read/write boundaries. |
| Fusion | Put several compatible operations in one program. |
| Materialization / realization | Ordinarily compute a logical value into storage / ask tinygrad to ensure needed backing, with virtual cases that need no ordinary allocation. |
| Renderer | Turn lowered IR into target source or another target representation. |
| Target compiler | Turn target representation into executable program bytes. |
| Runtime side | Execution machinery; in this snapshot its storage/copy, program invocation, and synchronization responsibilities are split across several contracts. |
| Backend | Selected implementation bundling those contracts for a target device path. |
| Host / device | CPU-side controller / selected execution target and its accessible memory. |
| Program | Executable function or representation of work. |
| Call / invocation | One requested execution event; an execution plan can also contain non-program calls such as copies or views. |
| Kernel | Accelerator program/function executed by parallel workers. |
| Launch | One invocation of a kernel with arguments and dimensions. |
| Thread | One logical worker running the kernel for particular indices. |
| Synchronization | Enforce ordering or wait for required work to complete. |

## Optional reinforcement—not missing prerequisites

The chapter has already taught the concepts needed to proceed. These official
resources provide other presentations if one model remains uncomfortable:

- Python's [emulating numeric types](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)
  section explains why `x * 2` can call `x.__mul__`. Stop after the binary
  arithmetic methods; descriptors and metaclasses are not needed here.
- Python's [`graphlib.TopologicalSorter`](https://docs.python.org/3/library/graphlib.html)
  documentation reinforces dependencies and topological order. The API is an
  example, not part of tinygrad.
- NVIDIA's [programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
  introduces host, device, kernels, threads, blocks, and grids. On a first read,
  stop before memory-hierarchy optimization.
- LLVM's [introduction to its IR](https://llvm.org/docs/LangRef.html#introduction)
  gives a second example of an intermediate representation. LLVM syntax is not
  required for tinygrad and should not be memorized for this course.

## What is deliberately left for later

- [Chapter 3](03-first-trace.md) captures the real artifacts between scheduling,
  compilation, and execution.
- [Chapter 4](04-tensor-and-autograd.md) separates frontend construction from
  autograd.
- [Chapter 5](05-uops.md) teaches UOp identity, operation families, and graph
  inspection systematically.
- [Chapter 7](07-scheduling.md) explains realization, bufferization, fusion
  boundaries, and the ordered execution plan.
- Chapters [9](09-kernel-optimization.md), [10](10-lowering.md), and
  [11](11-rendering.md) develop kernel scheduling, lowering, rendering, and
  compilation.
- Chapters [12](12-runtime.md) and [14](14-nvidia.md) cover runtime contracts
  and NVIDIA execution on Ubuntu.

[Next: build a reproducible development environment →](02-setup.md)
