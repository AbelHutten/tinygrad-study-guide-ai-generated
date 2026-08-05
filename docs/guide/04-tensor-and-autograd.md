# 4. Tensor frontend and autograd

## Purpose

This chapter connects the API you already know to the representation that
tinygrad contributors change. After it, `Tensor.matmul`, a loss function, and
`.backward()` should no longer look like privileged framework machinery. They
are Python code that constructs or transforms UOp graphs.

That distinction unlocks frontend fixes, dtype and broadcasting work, new
composite operations, and autograd rules. It also prevents a common debugging
mistake: looking for a kernel when the bug already exists in the lazy graph.

This chapter describes the [recorded source snapshot](../reference/source-snapshot.md).
The model is durable; names and exact decompositions can move.

## Prerequisite gate

Continue if you can explain all three statements:

1. reverse-mode autodiff propagates a cotangent from an output toward its
   inputs by applying a local derivative at each operation;
2. if one value feeds two paths, its gradient is the sum of both paths; and
3. a typed operation graph records dependencies without executing them.

If statement 3 is unfamiliar, work through the focused part of the official
[JAX jaxpr guide](https://docs.jax.dev/en/latest/jaxpr.html): stop when you can
identify variables, literals, primitives, inputs, and outputs in a small
jaxpr. If statements 1 or 2 are shaky, read PyTorch's official
[autograd mechanics note](https://docs.pytorch.org/docs/stable/notes/autograd.html)
through the sections on the graph and saved tensors. Return when you can derive
`d(x*x + 2*x)/dx` and explain why fan-out requires addition.

You do not need compiler lowering, CUDA, or kernel optimization yet. The
[learning-resource router](../reference/learning-resources.md#compiler-bridge-for-an-ml-reader)
has more context, but reading all of it now would hide the small idea this
chapter needs.

## Mental model: two object layers, one shared vocabulary

A tinygrad `Tensor` is a Python-facing handle. Its important state in this
chapter is:

- `uop`: the root of the value's current UOp graph;
- `grad`: either `None` or another `Tensor` handle for a gradient graph; and
- `is_param`: frontend state used to distinguish parameter-like tensors in
  relevant workflows.

A `UOp` is an interned IR node. Chapter 5 studies its fields and identity in
detail. For now, read this expression as construction, not execution:

```python
x = Tensor([1.0, 2.0, 3.0])
loss = (x * x + 2 * x).sum()
```

The Python objects point into a graph resembling:

```text
input ─┬─ MUL(x, x) ─────────┐
       └─ MUL(2, x) ── ADD ── REDUCE(sum) ── loss
```

The drawing duplicates the label `x`, but the real representation shares one
source node. It is a DAG, not generally a tree.

### Tensor identity is not UOp identity

`Tensor._apply_uop` extracts each input's `.uop`, calls a UOp-building
function, then creates a fresh `Tensor` wrapper around the returned node. Thus:

```python
a = x * 2
b = x * 2

assert a is not b
assert a.uop is b.uop
```

The first assertion is normal Python object identity. The second holds while
the structurally identical UOps are live because UOps are interned. A `Tensor`
is hashed by its own Python identity, and its `.uop` can later be replaced by
realization, assignment, or graph-wide substitution. Do not use the terms
“Tensor” and “UOp” interchangeably in an investigation.

### The mixins are the frontend

`Tensor` and `UOp` both inherit the same operation mixins. The mixins express
large APIs in terms of a small set of required construction primitives:

- elementwise code eventually calls `alu`;
- movement code eventually calls `_mop`;
- reduction code eventually calls `_rop`; and
- composites call other mixin methods.

`Tensor.alu` bridges from wrappers to UOps through `_apply_uop`. `UOp.alu`
directly creates another UOp. This is why most of the implementation of `dot`,
`linear`, indexing, activations, and losses is shared rather than duplicated
for the two classes.

For example, `matmul` delegates to `dot`. At this snapshot, `dot` reshapes its
operands, transposes one, multiplies them, reduces the contraction axis, and
casts the result. `linear` chooses elementwise multiplication or `dot`, then
adds a bias. Sparse categorical cross entropy expands into `log_softmax`, a
one-hot construction, masks, multiplications, and reductions. There is no
one-to-one rule between a public Tensor method and an `Ops` member—or between a
public method and a kernel.

When debugging a frontend method, repeatedly ask:

1. Is this method a thin primitive wrapper or a Python composite?
2. Which UOps did it construct for this shape and dtype?
3. Is the first wrong fact already visible before scheduling?

### Laziness has a boundary, not a slogan

Ordinary tensor algebra constructs graphs. `realize`, `data`, `tolist`,
`numpy`, and `item` can cross into scheduling and execution. In simplified
form:

```text
Tensor method
  → UOp construction
  → more lazy Tensor methods
  → realization request
  → callification, scheduling, lowering, compilation, execution
```

Chapter 3 observed that boundary and Phase 3 will open it. The important rule
here is: inspecting `.uop` does not execute the requested algebra.

Be precise about the limits of that statement. Constructing a Tensor from a
Python list creates and fills input storage, random generation maintains a
stateful counter, and mutation APIs introduce effect dependencies. “Lazy” does
not mean “no allocation or state anywhere”; it means the represented tensor
computation can remain an unevaluated graph until a value or schedule is
requested.

## Autograd is a graph-to-graph transform

`output.gradient(targets...)` does not run the forward graph and does not
record imperative callbacks while the forward code executes. It:

1. seeds the scalar output with a constant one unless an explicit gradient is
   supplied;
2. finds the portion of the forward UOp DAG that can reach the requested
   targets, stopping at `DETACH`;
3. traverses that portion in reverse dependency order;
4. asks `pm_gradient`, a `PatternMatcher`, for the local derivative of each
   operation;
5. sums contributions when a source has multiple consumers;
6. reduces a broadcasted contribution back to its source shape; and
7. wraps the resulting gradient UOps in new `Tensor` objects.

For multiplication, the local rule is conceptually:

```python
# ret = left * right, ctx = d(output) / d(ret)
(right * ctx, left * ctx)
```

These multiplications are new UOps. The derivative is itself lazy tensor
algebra and will go through the same compiler pipeline when read.

`gradient` and `backward` have different Python-side effects:

```python
dx, = loss.gradient(x)  # returns a Tensor; x.grad remains None
loss.backward()         # finds live forward Tensor wrappers and fills .grad
```

At this snapshot, `backward` looks through the weak registry of live Tensor
objects and attaches gradients to in-scope, floating, non-`CONST` wrappers in
the forward graph. It does not use PyTorch's `requires_grad` leaf model as an
exact analogue. Existing gradients are accumulated, potentially by an
assignment graph. Object lifetime and wrapper identity therefore matter when
debugging `.grad`, even though differentiation itself operates on UOps.

Two useful consequences follow:

- A wrong derivative can often be reproduced and inspected without a GPU by
  calling `.gradient()` and walking its `.uop`.
- A correct gradient UOp with a missing or surprising `.grad` attachment may
  be a Tensor registry/wrapper problem rather than a calculus problem.

## Source tour

All links below are pinned to commit
`874d33128b4e4785beea736d97df6716e0321717`.

| Read this | What to extract |
| --- | --- |
| [`Tensor` construction and `_apply_uop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L45-L123) | Separate wrapper state from UOp construction. Notice the fresh wrapper and forwarded `shape`, `dtype`, and `device`. |
| [`linear_with_vars`, `schedule_linear`, and `realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L169-L196) | Locate the boundary from a Tensor root to callification, scheduling, and execution. Do not study the internals yet. |
| [`ElementwiseMixin.add`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L12-L96) | Follow broadcasting and promotion into the required `alu` primitive. |
| [`MovementMixin.reshape`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L12-L171) and [`ReduceMixin.sum`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/reduce.py#L9-L48) | See shape validation and reduction normalization before `_mop`/`_rop`. |
| [`dot`, `matmul`, and `gradient`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/op.py#L361-L468) | Contrast a composite forward operation with the entry point to reverse-mode transformation. |
| [`linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/op.py#L1249-L1266) | Confirm that an ML-layer operation can be a short composition rather than a dedicated IR operation. |
| [`pm_gradient` and `compute_gradient`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/gradient.py#L44-L129) | Read one local rule, the reverse traversal, broadcast reduction, and fan-out accumulation. |
| [`Tensor.backward`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L429-L449) | Identify the live-Tensor selection and `.grad` mutation layered on top of `gradient`. |

Read call sites with `rg` instead of reading these files from top to bottom:

```bash
rg -n 'def _apply_uop|def gradient|def backward' tinygrad
rg -n 'pm_gradient|compute_gradient' tinygrad test
rg -n 'def dot|def linear|sparse_categorical_crossentropy' tinygrad/mixin
```

## Lab: watch both graphs

**Hardware:** Portable. The Python backend is sufficient.

The lab is `labs/phase2/frontend_autograd.py` in this guide's repository.
Run it from the tinygrad study checkout at the recorded snapshot. Point
`TINYGRAD_DOCS` at this guide's repository:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
CACHEDB=/tmp/tinygrad-guide-phase2.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/frontend_autograd.py"
```

Before running, predict:

- whether two `x * 2` expressions share a Tensor or a UOp;
- which operation families appear in the forward graph;
- whether `.gradient(x)` changes `x.grad`; and
- the UOps needed to turn the scalar seed into a length-three gradient.

At the snapshot, the important observations are:

```text
different Tensor wrappers: True
same interned UOp:        True
loss realized before read: False
x.grad after gradient():  None
x.grad after backward():  attached
gradient values:          [4.0, 6.0, 8.0]
```

Save the full operation counts. Explain every `EXPAND`, `REDUCE`, and repeated
`MUL` by pointing to either the forward composite or an autograd rule. The
printed count is evidence, not the explanation.

### Exercises

1. Change the loss to `(x * x).mean()`. Predict and then explain both the
   numeric gradient and any extra UOps.
2. Keep a reference to the intermediate `square = x*x`, call `backward`, and
   inspect `square.grad`. Then delete that reference, reconstruct the example,
   and explain why the live-wrapper behavior can differ.
3. Build a two-dimensional broadcast, such as `(x.reshape(3, 1) * y).sum()`,
   and identify where `compute_gradient` reduces each gradient to its source
   shape.
4. Trace a `2x3 @ 3x4` expression. Find `dot` in the source and account for the
   movement, elementwise, and reduction UOps before realization.
5. Add `.detach()` to one path of a branched expression. Predict which target
   gradients become zero and connect the result to `_deepwalk`.

## Checkpoint

You are ready for the UOp chapter when you can answer these without running
the lab again:

- Why can two distinct Tensor objects point at one UOp?
- Why is `matmul` not evidence that an `Ops.MATMUL` must exist?
- What exact action first requests execution in your lab?
- Why does fan-out add gradient UOps?
- Where is a broadcasted gradient returned to the source shape?
- Why can `.gradient()` be correct while `.backward()` attaches an unexpected
  `.grad`?

The evidence standard is a small graph plus a named source location, not “the
framework handles it.”

## Quick reference

| Question | Look here / remember this |
| --- | --- |
| What is a `Tensor` here? | A Python handle with `.uop`, `.grad`, and frontend state. |
| What constructs an elementwise node? | Mixin method → broadcasting/promotion → `alu` → `Tensor._apply_uop` → UOp. |
| Does one API call mean one UOp? | No. Composites such as `dot`, `linear`, and losses build graphs of smaller operations. |
| Does graph construction execute? | Usually no; a read, `realize`, or schedule request crosses the boundary. Input/storage and stateful API details still matter. |
| What does `.gradient()` do? | Reverse-traverses the relevant UOp DAG and constructs gradient UOps. |
| What does `.backward()` add? | It finds live forward Tensor wrappers and attaches/accumulates `.grad` handles. |
| Where are local derivatives? | `tinygrad/mixin/gradient.py:pm_gradient`. |
| First debugging artifact? | The forward root's and gradient root's `toposort()`, with op, dtype, shape, sources, and args. |
