# 4. Tensor frontend and autograd

## The promise of this chapter

If you know how to train a model, you have used automatic differentiation.
That does not mean you have been shown how it works. This chapter derives
reverse-mode differentiation from scalar arithmetic, then maps every step to
tinygrad's Tensor and UOp graphs.

The carried loss is:

```python
x = Tensor([1.0, 2.0, 3.0])
loss = (x * x + 2 * x).sum()
```

It is small enough to differentiate by hand but rich enough to expose four
essential mechanisms: the chain rule, two uses of one value, a reduction from
vector to scalar, and accumulation of several gradient contributions.

No calculus terminology, autograd implementation knowledge, or compiler
background is assumed. By the end, you will be able to:

- perform the forward and reverse passes by hand;
- explain why one shared value receives a sum of gradient contributions;
- distinguish a Tensor wrapper, a forward UOp DAG, and a gradient UOp DAG;
- explain why constructing either graph does not execute its tensor algebra;
- distinguish `gradient()` from `backward()` and their Python-side effects;
- supply a cotangent for a non-scalar output and identify the result as a
  vector-Jacobian product;
- derive the reverse reduction required by broadcasting;
- explain what `detach()` blocks and what it does not change; and
- locate a frontend/autograd problem before looking at scheduled kernels.

## Start with one scalar derivative

A derivative measures how quickly an output changes when an input changes by
a small amount. For a scalar function `L=f(x)`, the notation `dL/dx` means the
rate of change of `L` with respect to `x`.

You need three local rules for the carried example.

### Addition

If:

```text
v = a + b
```

then changing `a` by a small amount changes `v` by the same amount, and the
same is true for `b`:

```text
dv/da = 1
dv/db = 1
```

### Multiplication

If:

```text
v = a * b
```

then, while holding `b` fixed, a small change in `a` is scaled by `b`:

```text
dv/da = b
dv/db = a
```

### A sum reduction

If:

```text
v = a0 + a1 + ... + an
```

then every input element has local derivative one:

```text
dv/dai = 1
```

These are *local* derivatives: they describe one operation's output with
respect to its immediate inputs. A computation with several operations needs
the chain rule.

## The chain rule as message passing

Suppose an intermediate `v` affects the final loss `L`. We will write:

```text
bar(v) = dL/dv
```

`bar(v)` is commonly called an **adjoint**, **cotangent**, or **upstream
gradient**. All three names answer the same question here: “If `v` changed,
how would the final loss change?”

If `v=f(a,b)`, the chain rule sends contributions from `v` to its inputs:

```text
contribution to bar(a) = bar(v) * dv/da
contribution to bar(b) = bar(v) * dv/db
```

The output gradient multiplies each local derivative. Reverse mode starts at
the loss and processes nodes toward the inputs, after all consumers that can
contribute to a node have been considered.

### Work the scalar example at `x=3`

Name the intermediates:

```text
a = x * x
b = 2 * x
L = a + b
```

The forward pass computes:

```text
x = 3
a = 9
b = 6
L = 15
```

The reverse pass seeds the final scalar with one:

```text
bar(L) = dL/dL = 1
```

Addition sends that upstream value to both branches:

```text
bar(a) = 1
bar(b) = 1
```

For `a=x*x`, `x` occupies both input positions. The multiplication rule sends
one contribution through each edge:

```text
left x contribution  = bar(a) * right x = 1 * 3 = 3
right x contribution = bar(a) * left x  = 1 * 3 = 3
```

For `b=2*x`, the contribution to `x` is:

```text
bar(b) * 2 = 1 * 2 = 2
```

All three paths refer to the same `x`, so their contributions add:

```text
bar(x) = 3 + 3 + 2 = 8
```

That agrees with the symbolic derivative `d(x*x+2*x)/dx = 2*x+2`, evaluated
at `x=3`.

### Why fan-out requires addition

“Fan-out” means one value is used by more than one consumer edge. It is not a
special autodiff convention. If changing `x` affects the loss through three
paths, the total first-order change is the sum of the changes through those
paths. Reverse mode therefore accumulates every incoming contribution to
`bar(x)`.

Even the single operation `x*x` has two source-edge positions pointing to the
same node. A graph traversal must preserve both edge contributions. Counting
only distinct consumer nodes would lose one of them.

## Generalize the result to a tensor

For:

```python
x = Tensor([1.0, 2.0, 3.0])
loss = (x * x + 2 * x).sum()
```

the forward values are:

| Expression | Value |
| --- | --- |
| `x*x` | `[1, 4, 9]` |
| `2*x` | `[2, 4, 6]` |
| `x*x + 2*x` | `[3, 8, 15]` |
| `.sum()` | `26` |

The scalar sum sends its seed `1` to every element. Each coordinate is then
the scalar derivation above:

```text
d loss/d x[i] = 2*x[i] + 2
```

so:

```text
d loss/dx = [4, 6, 8]
```

Shape is part of the derivative contract. The target `x` has shape `(3,)`, so
its gradient must also have shape `(3,)`.

## Why reverse mode starts from an output seed

For a scalar loss, `dL/dL=1` supplies an unambiguous implicit seed. A
non-scalar output is different.

Suppose:

```python
v = x * x
```

maps a three-element `x` to a three-element `v`. The full Jacobian would store
every `dv[i]/dx[j]` pair in a 3-by-3 matrix. Reverse mode normally avoids
materializing that matrix. Given an output-shaped seed `g`, it computes the
effect of the weighted output combination:

```text
sum_i g[i] * v[i]
```

with respect to the targets. This is a **vector-Jacobian product** (VJP). With:

```text
x = [1, 2, 3]
g = [1, 10, 100]
```

the result for `v=x*x` is:

```text
g * 2*x = [2, 40, 600]
```

tinygrad's `gradient=` argument supplies this output cotangent. Without one,
`gradient()` and `backward()` require a scalar output. The supplied seed must
be understood as part of the derivative question; it is not a batch of input
examples. The clearest seed has the output's exact shape, as above. At this
snapshot tinygrad also accepts a seed whose shape broadcasts through the local
gradient rules—for example, a scalar seed for a vector output—so exact shape
equality is not an API precondition.

## Broadcasting creates a reverse reduction

Broadcasting lets shapes with dimensions of length one participate in a
larger common shape without manually copying values. Consider:

```text
xb shape (2, 1):       [[1],
                         [2]]

w shape (1, 3):       [[10, 20, 30]]

xb * w shape (2, 3):  [[10, 20, 30],
                         [20, 40, 60]]
```

For `broadcast_loss=(xb*w).sum()`, the forward result is `180`.

Each `xb` row value was reused across three columns. Its gradient is the sum
of the three corresponding `w` values:

```text
d broadcast_loss/dxb = [[10+20+30],
                         [10+20+30]]
                      = [[60], [60]]
```

Each `w` column value was reused across two rows. Its gradient is the sum of
the two corresponding `xb` values:

```text
d broadcast_loss/dw = [[1+2, 1+2, 1+2]]
                    = [[3, 3, 3]]
```

The local multiplication rule initially produces contributions in the output
shape `(2,3)`. Reverse broadcasting must sum the axes that were added or
expanded, then reshape the contribution to the source shape. Merely slicing
one repeated contribution would be mathematically wrong.

## Paper lab: perform reverse mode without tinygrad

Answer these before opening the worked result.

1. Draw the scalar DAG for `a=x*x`, `b=2*x`, `L=a+b`. Draw two separate source
   edges from `x` into `a`.
2. At `x=3`, fill a table with every forward value, every upstream gradient,
   every local derivative, and every contribution to `bar(x)`.
3. For `x=[1,2,3]`, compute `(x*x+2*x).sum()` and its gradient.
4. For `v=x*x`, use seed `[1,10,100]`. State whether you computed a full
   Jacobian or a VJP.
5. For the `(2,1)` by `(1,3)` broadcast above, mark which axis must be summed
   to recover each source gradient.
6. If the `2*x` branch is detached before addition, compute the forward loss
   and the gradient with respect to `x`.
7. Predict what two successive calls to `loss.backward()` should leave in
   `x.grad` if it is not cleared between calls.

??? success "Worked answer"

    **1–2. Scalar graph and reverse table**

    The arrows below run from each dependency to the operation that consumes
    it. Repeating `a` on two rows is intentional: these are two edge positions
    entering one `MUL` node, not two copies of `a`.

    ```text
    x ──(source slot 0)──▶ a = MUL(x,x)
    x ──(source slot 1)──▶ a = MUL(x,x)
    2 ──(source slot 0)──▶ b = MUL(2,x)
    x ──(source slot 1)──▶ b = MUL(2,x)
    a ──(source slot 0)──▶ L = ADD(a,b)
    b ──(source slot 1)──▶ L = ADD(a,b)
    ```

    At `x=3`, `a=9`, `b=6`, and `L=15`. Seed `bar(L)=1`.
    Addition gives `bar(a)=bar(b)=1`. The two `a` edges contribute `3` and
    `3`; the `b` edge contributes `2`; total `bar(x)=8`.

    **3. Tensor loss**

    The elementwise terms are `[3,8,15]`, loss is `26`, and the gradient is
    `[4,6,8]`.

    **4. Explicit seed**

    The seeded result is `[2,40,600]`. This is the VJP for the chosen output
    cotangent, not a materialized 3-by-3 Jacobian.

    **5. Broadcast axes**

    The contribution to `xb` must sum columns, axis `1`, turning `(2,3)` into
    `(2,1)`. The contribution to `w` must sum rows, axis `0`, turning `(2,3)`
    into `(1,3)`.

    **6. Detach**

    Detach does not change the forward values, so the loss remains `26`. It
    blocks the `2*x` contribution. Only `x*x` remains, giving `[2,4,6]`.

    **7. Accumulation**

    One backward pass contributes `[4,6,8]`; the second adds the same values
    to the existing gradient, yielding `[8,12,16]`.

## Map the forward pass to tinygrad's frontend

Python still executes the overloaded `*`, `+`, and `.sum()` methods
immediately. What those methods do is construct UOps and return Tensor
wrappers; they do not immediately run the represented elementwise and
reduction algebra.

There are four layers to keep separate:

| Layer | Carried example | Role |
| --- | --- | --- |
| Python objects and calls | `x`, `loss`, `Tensor.__mul__` | User-facing handles and graph-construction code. |
| Forward UOp DAG | `BUFFER`, `MUL`, `ADD`, `REDUCE` | Lazy dependency representation of the requested values. |
| Gradient UOp DAG | `EXPAND`, `MUL`, `ADD`, and the input `BUFFER` | New lazy algebra that will compute the derivative. |
| Scheduled/compiled/runtime work | `CALL(SINK)` then `CALL(PROGRAM)` | Planning or execution machinery entered when scheduling or realization is requested. |

A Python-list Tensor does allocate and fill input storage. Random-number APIs
also maintain state, and mutation introduces effects. “The algebra is lazy”
does not mean “nothing in the process allocates or changes state.”

### Tensor wrapper versus UOp node

A `Tensor` is a Python-facing wrapper whose `.uop` points at a graph root.
Most operation methods extract input UOps, build a new UOp, then return a fresh
Tensor wrapper around it.

Consequently:

```python
a = x * 2
b = x * 2

assert a is not b
assert a.uop is b.uop
```

The wrappers are distinct Python objects. At the pinned snapshot, the
structurally identical live UOps are interned and shared. Chapter 5 explains
the cache key and lifetime rules; this chapter uses the observation only to
keep wrapper identity separate from graph-node identity.

### The exact forward graph

The carried graph has seven distinct nodes:

```text
N0 CONST(3)                 buffer extent metadata
N1 BUFFER(shape=(3,)) <- N0 input x
N2 MUL              <- N1, N1       x*x
N3 CONST(2.0)
N4 MUL              <- N3, N1       2*x
N5 ADD              <- N2, N4
N6 REDUCE(add)      <- N5            scalar loss
```

`N0 CONST(3)` records the buffer's three-element extent. It is bookkeeping,
not another scalar in `x*x+2*x`. The public `.sum()` method uses a `REDUCE` UOp
whose reduction operation is addition; there is no requirement for a dedicated
`SUM` operation.

`N1` appears in both source positions of `N2` and again in `N4`. Distinct-node
counts therefore differ from source-edge counts.

### One Tensor method is not one UOp or one kernel

The frontend is largely written as Python compositions. `square()` returns
`self*self`. `matmul()` delegates to `dot()`, and `dot()` reshapes, transposes,
multiplies, reduces the contraction axis, and casts. An ML-layer helper such as
`linear()` chooses multiplication or `dot()` and then adds bias.

This has two consequences for contributors:

- an API bug may already be visible as the wrong shape, dtype, source edge, or
  UOp before scheduling; and
- an API name such as `matmul` does not imply an `Ops.MATMUL` node or one
  runtime kernel. Scheduling decides program boundaries later.

## Autograd constructs another UOp graph

`loss.gradient(x)` performs a graph-to-graph transformation. It does not first
execute the loss and does not mutate `x.grad`.

At a high level, it:

1. uses a scalar constant `1` as the output gradient when no explicit seed is
   supplied;
2. keeps only forward paths relevant to the requested target, stopping reverse
   flow at `DETACH`;
3. visits those nodes in reverse dependency order;
4. selects a local derivative rule for each operation;
5. reduces shaped contributions back to each source shape after broadcasting;
6. adds a contribution to any gradient already accumulated for a source; and
7. wraps the requested gradient UOps in new Tensor objects.

For the carried loss, the gradient DAG is structurally equivalent to:

```text
seed = EXPAND(1.0, shape=(3,))
p = 2.0 * seed
q = x * seed
dx = (p + q) + q
```

The actual distinct-node counts are:

```text
{'ADD': 2, 'BUFFER': 1, 'CONST': 3, 'EXPAND': 1, 'MUL': 2}
```

Calculus produced three multiplication contributions: `2*seed`, `x*seed`,
and another `x*seed`. The two `x*seed` expressions are structurally identical,
so they share one UOp. That shared node is consumed twice by the accumulation
ADDs. Operation counts count distinct DAG nodes, not mathematical edge uses.

`EXPAND` represents the scalar sum seed being logically available at all three
element positions. It need not mean a physical three-element copy. Shape/view
mechanics are taught in Chapter 8.

The derivative is still lazy. Reading `dx.tolist()` requests planning and
execution of this new graph through the same compiler pipeline as an ordinary
forward Tensor.

## `gradient()` and `backward()` answer different API questions

### `gradient()` is explicit and functional

```python
dx, = loss.gradient(x)
```

It takes explicit targets, returns a list of gradient Tensors, and leaves
`x.grad` unchanged. If a requested floating target is not reachable through a
differentiable path, the returned entry is a zero-like Tensor. Output and
targets must be floating point at this snapshot.

For non-scalar output, pass `gradient=seed` to specify the cotangent/VJP.

### `backward()` attaches mutable wrapper state

```python
loss.backward()
assert x.grad is not None
```

`backward()` invokes the same gradient construction for live forward Tensor
wrappers, then attaches or accumulates their `.grad` fields. Its selection
model is not PyTorch's `requires_grad`/leaf model:

- `Tensor` has no `requires_grad` slot here;
- every live, floating, non-`CONST` Tensor wrapper whose UOp occurs in the
  forward toposort can receive a `.grad`;
- named live intermediates can therefore receive gradients too; and
- wrapper lifetime matters for attachment, even though the forward UOp remains
  in the loss graph after an intermediate wrapper is destroyed.

`is_param` is separate frontend state used by optimizers to select parameters
and buffers. It does not decide whether `backward()` differentiates a live
Tensor.

A bare scalar such as `Tensor(1.0)` is a `CONST` UOp and is skipped by
`backward()`'s attachment filter. `Tensor([1.0])` is buffer-backed and differs
for that reason.

### Gradients accumulate until cleared

If `.grad` is already present, another `backward()` adds the new contribution.
Two calls on the same carried loss therefore produce `[8,12,16]`, not
`[4,6,8]`.

Clearing means removing the current gradient reference:

```python
x.grad = None
```

`Optimizer.zero_grad()` performs exactly that assignment for each parameter;
despite its name, it does not fill a gradient buffer with numerical zeroes.

## `detach()` changes the reverse path, not the forward value

`detach()` constructs a `DETACH` UOp marker around the same value relationship.
It does not copy a Tensor to the CPU, force realization, or numerically change
the forward expression.

During gradient path discovery, reverse traversal does not cross that marker.
For:

```python
detached_loss = (x*x + (2*x).detach()).sum()
```

the forward loss remains `26`, but the gradient becomes `[2,4,6]` because only
the square branch contributes. If every path from an output to an explicitly
requested target is detached, `gradient(target)` returns a zero-like gradient.

`backward()` may still attach that zero result to a live source wrapper it
selected from the full forward graph. “No reverse contribution” does not
universally mean “the Python `.grad` field remains `None`.”

## Runnable lab: inspect forward and gradient graphs

Run from the guide root with the pinned study environment:

```bash
CACHEDB=/tmp/tinygrad-guide-frontend-autograd.db DEV=PYTHON DEBUG=0 \
  ../tinygrad-study/.venv/bin/python labs/phase2/frontend_autograd.py
```

The stable output is:

```text
different Tensor wrappers: True
same interned UOp:        True

forward shape/dtype:      () dtypes.float
forward op counts:        {'ADD': 1, 'BUFFER': 1, 'CONST': 2, 'MUL': 2, 'REDUCE': 1}
forward graph:
  N0 CONST shape=() <- [] arg=3
  N1 BUFFER shape=(3,) <- [N0]
  N2 MUL shape=(3,) <- [N1,N1]
  N3 CONST shape=() <- [] arg=2.0
  N4 MUL shape=(3,) <- [N3,N1]
  N5 ADD shape=(3,) <- [N2,N4]
  N6 REDUCE shape=() <- [N5]
loss realized before read: False

gradient shape/dtype:     (3,) dtypes.float
gradient op counts:       {'ADD': 2, 'BUFFER': 1, 'CONST': 3, 'EXPAND': 1, 'MUL': 2}
gradient graph:
  N0 CONST shape=() <- [] arg=2.0
  N1 CONST shape=() <- [] arg=1.0
  N2 CONST shape=() <- [] arg=3
  N3 EXPAND shape=(3,) <- [N1,N2]
  N4 MUL shape=(3,) <- [N0,N3]
  N5 BUFFER shape=(3,) <- [N2]
  N6 MUL shape=(3,) <- [N5,N3]
  N7 ADD shape=(3,) <- [N4,N6]
  N8 ADD shape=(3,) <- [N7,N6]
x.grad after gradient():  None
returned gradient realized: False
returned gradient values: [4.0, 6.0, 8.0]
x.grad after backward():  attached
gradient realized before read: False
gradient values:          [4.0, 6.0, 8.0]
gradient after two backward calls: [8.0, 12.0, 16.0]
gradient after explicit reset: None
seeded vector gradient:   [2.0, 40.0, 600.0]
gradient with detached 2*x branch: [2.0, 4.0, 6.0]
broadcast gradient shapes: (2, 1) (1, 3)
broadcast gradient dxb:    [[60.0], [60.0]]
broadcast gradient dw:     [[3.0, 3.0, 3.0]]
```

Read it in five passes.

### 1. Wrapper and forward graph

Two Python wrappers can point to one live interned UOp. The forward graph then
shows every edge needed by the paper derivation. `REDUCE shape=()` turns the
length-three expression into a scalar. No represented algebra has run yet.

### 2. Returned gradient graph

`gradient()` leaves `x.grad` at `None`. Its returned Tensor is initially lazy;
the edge listing shows the scalar seed, shape expansion, constant branch, and
twice-used `N6`. `tolist()` requests execution and returns the predicted
values. Do not assert that a read always changes the original wrapper's
`uop.is_realized` flag: internal contiguous/read paths can realize a related
wrapper. The defensible boundary is that value observation requests needed
work.

### 3. Attached and accumulated state

`backward()` attaches a separate gradient wrapper with the same mathematical
values. A second call accumulates, and assigning `None` clears the state.

### 4. Seed and detach

The explicit seed produces the hand-computed VJP. Detaching the `2*x` branch
preserves forward semantics while removing precisely that reverse
contribution.

### 5. Broadcast shape restoration

The output shapes `(2,1)` and `(1,3)` match the original sources, not the
broadcast product. The numerical values prove that repeated contributions were
summed along the correct axes.

All of these are portable semantic observations. They do not test CUDA, GPU
launches, or performance.

## Debug by separating five failure classes

| Observation | First question | Likely layer |
| --- | --- | --- |
| Forward shape/value formula is wrong before `gradient()` | Which public method or composite constructed the first wrong UOp? | Tensor frontend, promotion, broadcasting, or composite logic |
| Forward DAG is right; returned gradient DAG has a wrong local expression | Which operation's local derivative rule ran? | `pm_gradient` rule or target-path selection |
| Local contribution is right but final source shape/value is wrong | Which broadcast axes should have been summed, and which other consumers contribute? | Shape restoration or fan-out accumulation |
| `gradient()` returns the right Tensor but `.grad` is missing/doubled/surprising | Which wrappers are live, and was existing state cleared? | `backward()` attachment/accumulation |
| Gradient UOp DAG is right but realized values differ by backend | Where does the first compiled/runtime artifact diverge? | Later compiler or runtime layer |

Do not begin by reading a GPU kernel when the returned gradient DAG already
contains the wrong sources. Conversely, a correct UOp graph does not prove all
lowering and execution paths are correct.

## Guided source tour: behavior first, implementation second

These links target the pinned commit. Each stop starts with a behavior you can
now predict. Read only the linked range and answer its question.

### Stop 1: which contracts do upstream tests state?

Read [custom seed, broadcast, and non-scalar tests at lines 21–41](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_gradient.py#L21-L41).

Question: when is an explicit gradient required, and what source shapes must
broadcast gradients recover?

Translation: a supplied cotangent scales the reverse result; non-scalar output
without one raises; and broadcast gradients return to each input's original
shape. These are executable behavioral contracts, not implementation guesses.

Then read [accumulation, non-leaf, and detach tests at lines 60–85](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_gradient.py#L60-L85).
They confirm repeated accumulation and tinygrad's deliberate non-leaf/live-
wrapper behavior.

### Stop 2: how does a Tensor method construct a new wrapper?

Read [`Tensor._apply_uop` and its required mixin bridge lines 105–123](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L105-L123).

Question: where are input wrappers unwrapped, and where is the output wrapper
created?

Translation: `_apply_uop` extracts every input `.uop`, calls a UOp-building
function, then creates a fresh Tensor with that root. Focus on lines 105–113;
metadata and weak-registry bookkeeping support other concerns.

### Stop 3: where do seed, target, and zero fallback enter?

Read [`gradient` lines 447–467](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/op.py#L447-L467).

Question: what restriction permits an implicit seed, which dtypes are
accepted, and what is returned for an absent target path?

Translation: only a scalar output may omit `gradient`; output and targets must
be floating; the default seed is one; and a missing target entry becomes a
zero-like UOp before wrapping.

### Stop 4: what are the local rules for this loss?

Read [`pm_gradient` lines 44–65](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/gradient.py#L44-L65).

Question: find only `ADD`, `MUL`, `REDUCE`, `RESHAPE`, and `EXPAND`. How many
contributions does each rule return, and in which source order?

Translation: ADD returns the upstream value for both inputs. MUL returns the
other operand times the upstream value for each source position. ADD reduction
expands its gradient back to the input shape. Ignore power, max, where, calls,
and movement rules until an example needs them; Chapter 6 teaches the matcher
syntax.

### Stop 5: where do reverse order and detach appear?

Read [`_deepwalk` and the start of `compute_gradient` lines 83–94](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/gradient.py#L83-L94).

Question: what path is kept, what marker blocks flow, and in what order is the
walk processed?

Translation: `_deepwalk` selects only nodes on paths to targets and excludes
`DETACH`; `compute_gradient` seeds the root and iterates the path in reverse.

Confirm the marker's construction in the tiny
[`detach` method lines 39–43](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L39-L43).
It creates an IR marker; it does not copy or execute data.

### Stop 6: where are broadcasting and fan-out handled?

Read [`compute_gradient` lines 109–123](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/gradient.py#L109-L123).

Question: what happens when a local contribution shape differs from its
source, and what happens when the source already has a gradient?

Translation: lines 115–117 sum broadcast axes and reshape to the source shape.
Lines 118–123 add to an existing entry, preserving fan-out contributions.
Tuple/call handling is unrelated to the carried example.

### Stop 7: what mutable behavior does `backward()` add?

Read [`Tensor.backward` lines 439–449](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L439-L449).

Question: which live wrappers are targeted, and how does an existing `.grad`
change the action?

Translation: every live, floating, non-constant Tensor whose UOp is in the
forward graph becomes a target. A missing field receives a returned gradient;
an existing field accumulates by assignment.

Finish with [`Optimizer.zero_grad` lines 29–33](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/nn/optim.py#L29-L33).
It sets references to `None`; the name does not imply a numerical fill kernel.

### Optional stop: why is matmul a composite?

After the carried example is clear, read only [`dot` and `matmul` lines 380–401](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/op.py#L380-L401).

Question: which familiar Tensor operations implement the contraction?

Translation: reshape and transpose align axes, multiplication creates pairwise
products, sum contracts the final axis, cast establishes the result dtype, and
`matmul` delegates. Stop before `einsum`; it is a larger frontend composition.

## Controlled extensions with answers

1. Replace the loss with `(x*x).mean()`. Before running, derive the result.
2. Detach the square branch instead of the `2*x` branch, then detach both.
3. Keep `square=x*x` as a named live wrapper, call `square.sum().backward()`,
   and predict `square.grad`.
4. For matrix shapes `2x3 @ 3x4`, describe the frontend composition without
   claiming a kernel count.

??? success "Answers"

    1. `mean` divides the sum by three, so `d mean(x*x)/dx = 2*x/3`, yielding
       approximately `[0.6667,1.3333,2.0]` for `[1,2,3]`.
    2. With square detached, only `2*x` contributes: `[2,2,2]`. With both
       branches detached, an explicit `gradient(x)` returns zeros.
    3. The sum rule supplies one for every square element, so the live
       intermediate receives `square.grad=[1,1,1]`; `x.grad=[2,4,6]`.
    4. `dot` reshapes the left and right operands, transposes the right
       contraction axis, broadcasts a multiply over paired elements, reduces
       the shared size-three axis, and casts. Scheduling—not the frontend
       method—later chooses program boundaries.

## Checkpoint: explain the transform, not just the values

Record these artifacts from your own run:

```text
forward node/edge listing:
hand-computed forward values:
scalar reverse contributions to x:
gradient node/edge listing:
returned gradient and x.grad after gradient():
x.grad after first and second backward():
seeded VJP:
detached-branch result:
broadcast source shapes, reduced axes, and values:
```

You pass when you can answer:

1. Why does `x*x` contribute through two source edges?
2. Why are there only two distinct `MUL` nodes in the gradient DAG?
3. Why does `.sum()` introduce an `EXPAND` in its reverse graph?
4. What does a non-scalar output seed mean mathematically?
5. Why must broadcast gradients sum rather than slice repeated values?
6. Why can `.gradient()` be correct while `.grad` attachment is surprising?
7. Why does repeated `backward()` double the carried gradient?
8. What does `detach()` preserve, and where does it stop reverse traversal?
9. Why is a public `matmul` call not evidence for an `Ops.MATMUL` or one
   kernel?
10. Which artifact would you inspect first for a wrong derivative: forward UOp
    DAG, gradient UOp DAG, schedule, source, or runtime value—and why?

## Quick reference

| Term/API | Meaning here |
| --- | --- |
| local derivative | Sensitivity of one operation's output to one immediate input. |
| `bar(v)` / cotangent | `d loss/dv`, the upstream value propagated in reverse. |
| chain rule | Multiply upstream gradient by each local derivative. |
| fan-out accumulation | Add contributions from every source edge/consumer path. |
| VJP | Gradient of a chosen weighted combination of output elements; avoids a full Jacobian. |
| forward UOp DAG | Lazy value dependencies built by Tensor methods. |
| gradient UOp DAG | New lazy algebra built by reverse-mode transformation. |
| `gradient(*targets)` | Return gradient Tensors for explicit targets; do not mutate `.grad`. |
| `backward()` | Target live forward wrappers and attach/accumulate `.grad`. |
| `detach()` | Preserve forward value but block reverse traversal through that edge. |
| `Optimizer.zero_grad()` | Set each parameter's `.grad` to `None`. |
| broadcast reverse | Sum repeated contributions along expanded/added axes. |

## Optional reinforcement—not missing prerequisites

- Re-run the paper derivation with `x=-2`, then verify the lab in a temporary
  copy. Stop once every local contribution predicts the final value.
- Read the two narrow upstream test ranges in Source Stop 1 as examples of how
  calculus claims become regression tests. Do not read the whole backend test
  suite yet.
- Use a central finite difference for one scalar coordinate:

  ```text
  (f(x+h) - f(x-h)) / (2*h)
  ```

  Compare it with `2*x+2` for a modest `h` such as `1e-3`. This is an
  independent numerical sanity check, not a replacement for exact graph and
  shape tests.

## What is deliberately left for later

- Chapter 5 explains UOp fields, identity, interning, traversal, and shape
  derivation.
- Chapter 6 explains how `pm_gradient` selects and applies pattern rules.
- Chapter 7 explains when the lazy forward and gradient graphs materialize and
  fuse.
- Chapter 8 derives broadcasting, views, and symbolic shape/index mechanics in
  full.
- Chapters 9–12 explain how correct gradient algebra becomes optimized target
  programs and runtime invocations.
- Chapter 16 develops gradient test matrices, tolerances, properties, and
  independent oracles for a real contribution.

[← First trace](03-first-trace.md) · [Next: UOp graphs →](05-uops.md)
