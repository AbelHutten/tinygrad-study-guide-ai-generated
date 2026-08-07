# 5. UOps: reading tinygrad's intermediate graphs

## The promise of this chapter

Chapter 4 followed this Tensor expression into a forward UOp graph:

```python
loss = (x * x + 2 * x).sum()
```

This chapter removes the Tensor wrapper and constructs the same expression in
tinygrad's intermediate language. That gives us a small graph whose every
node, edge, dtype, shape, and identity relationship can be explained. We will
use it to learn the graph-reading skills needed throughout the rest of the
compiler.

No compiler course or graph-algorithms course is assumed. Terms such as
*intermediate representation*, *node*, *edge*, *DAG*, *topological order*, and
*hash-consing* are introduced before they are used.

By the end, you will be able to:

- translate a small formula into raw UOps and back again;
- read `op`, `dtype`, `src`, `arg`, and `tag` without treating them as five
  unrelated pieces of trivia;
- distinguish a stored dtype from a recursively derived shape;
- explain why equivalent live constructions can be the same Python object;
- distinguish object identity, the interning key, `.key`, and mathematical
  equivalence;
- update an immutable-style graph by rebuilding a path to its root;
- prove that a traversal is topological and account for shared edges;
- build a consumer map and separately account for repeated source positions;
  and
- classify an unfamiliar operation well enough to find its constructors,
  transformations, and tests.

These are not merely vocabulary goals. Chapter 6's rewrite rules receive UOp
graphs and return UOp graphs. If a rewrite misbehaves, you need to know exactly
which representation changed before you can reason about the rule.

## Why have an intermediate representation?

An **intermediate representation**, usually shortened to **IR**, is a data
structure that records a computation in a form a compiler can analyze and
transform. It is “intermediate” because it sits between the user's expression
and the final target program.

The Python text is a poor long-lived compiler representation. Consider:

```python
loss = (x * x + 2 * x).sum()
```

After Python evaluates this line, tinygrad needs facts the text does not state
directly:

- which three appearances of `x` refer to the same value;
- the dtype and shape of every intermediate result;
- which input positions belong to each multiplication;
- that the final sum combines one tensor axis; and
- which later operations depend on each earlier result.

It also needs to transform the computation many times. Autograd builds new
algebra. Scheduling chooses materialization boundaries. Lowering introduces
indexes, ranges, and memory operations. Rendering produces target source.
Keeping Python syntax as the compiler's truth would make every pass re-solve
Python's much larger language.

A UOp is tinygrad's compact unit of IR. Different UOp graphs appear at
different compiler stages. An early graph can describe tensor algebra; a
later graph can describe memory access or control flow; still later nodes can
carry rendered source or compiled bytes. “UOp” therefore does not mean “one
GPU instruction.” It means a node in tinygrad's shared compiler language.

## Build the carried expression directly

The raw construction for this chapter is:

```python
from tinygrad import dtypes
from tinygrad.uop import GroupOp, Ops
from tinygrad.uop.ops import UOp

x = UOp.param(0, dtypes.float32, shape=(3,), device="PYTHON", name="x")
square = x * x
scaled = 2 * x
summed = square + scaled
loss = summed.sum()
```

This does not supply the values `[1,2,3]`. `x` represents a three-element
floating input parameter in slot zero. The graph says what to compute when a
matching input exists. It is useful for studying IR because it has no Tensor
wrapper, no allocated input, and no need to execute a backend.

The names `square`, `scaled`, and `summed` are ordinary Python references that
make the construction readable. They are not stored variable names inside the
math nodes. The graph stores dependencies.

## From a formula to a graph

### The minimum graph vocabulary

A **node** represents one value or operation. A directed **edge** connects a
dependency to an operation that uses it. If node `A` supplies an input to node
`B`, we call `A` a **source** of `B` and `B` a **consumer** of `A`.

The arrows are directed because “depends on” has an orientation. We will draw
them from source to consumer:

```text
source ──► consumer
```

The object itself stores the relationship in the other reading direction:
the consumer's `src` tuple contains its source objects. If `add.src` is
`(left, right)`, the drawing has `left -> add` and `right -> add`.

A **path** is a sequence of connected edges. A **cycle** would be a path that
eventually returns to its starting node. A **directed acyclic graph**, or
**DAG**, is a directed graph without such a cycle. Normal UOp construction
builds new consumers from already existing sources, so computations are read
as DAGs under tinygrad's logical-immutability discipline.

The final node through which we inspect a computation is its **root**. `loss`
is our root. Walking recursively through `loss.src`, then those nodes' sources,
reaches everything needed to describe the loss.

### First draw the tempting tree

A direct syntax-tree drawing duplicates each use of `x`:

```text
                         REDUCE(ADD, one axis)
                                  │
                                 ADD
                               /     \
                            MUL       MUL
                           /   \     /   \
                          x     x   2     x
```

This is useful for recovering the formula, and it correctly shows three
source positions occupied by `x`. It is not an accurate inventory of unique
objects. The three `x` leaves refer to one input node.

### Now draw the DAG

Merge repeated values instead of copying them:

```text
                       ┌──────────────┐
                       │      x       │
                       └──────────────┘
                         │    │     │
                         │    │     └────────┐
                         ▼    ▼              ▼
                       ┌────────┐    2 ──► ┌────────┐
                       │ x * x  │          │ 2 * x  │
                       └────────┘          └────────┘
                              \              /
                               ▼            ▼
                                ┌──────────┐
                                │   ADD    │
                                └──────────┘
                                      │
                                      ▼
                                ┌──────────┐
                                │  REDUCE  │
                                └──────────┘
```

The `x*x` node has two edges from `x`, one for each input position. The
`2*x` node has a third edge from the same `x`. Consequently:

- there is one unique `x` node;
- `x` occupies three source positions; and
- `x` has two unique consumer nodes: the two `MUL` nodes.

This distinction mattered to autograd in Chapter 4. The square sends two
gradient contributions through two edge positions even though both positions
have the same source. A set of consumer nodes alone cannot represent that
multiplicity.

At the formula level the DAG has six conceptual nodes: `x`, `2`, two
multiplications, one addition, and one reduction. The actual UOp graph has a
seventh node. We will account for it rather than dismiss it as internal noise.

## The exact seven-node UOp graph

Number each unique node in topological order, with every source numbered
before its consumer:

```text
id  op       dtype             shape  src       arg
N0  CONST    dtypes.weakint    ()     []        3
N1  PARAM    dtypes.float      (3,)   [N0]      slot=0, name='x', device='PYTHON'
N2  MUL      dtypes.float      (3,)   [N1,N1]   -
N3  CONST    dtypes.weakfloat  ()     []        ConstFloat(2.0)
N4  MUL      dtypes.float      (3,)   [N3,N1]   -
N5  ADD      dtypes.float      (3,)   [N2,N4]   -
N6  REDUCE   dtypes.float      ()     [N5]      (Ops.ADD, 1)
```

`dtypes.float32` prints as `dtypes.float` in this snapshot; they are aliases,
not different dtypes.

Read each row as a complete sentence.

### `N0`: shape metadata is represented by a UOp

`N0` is the integer constant `3`. It does **not** mean the tensor contains the
number three, and it is not a numeric operand of `x*x+2*x`. It participates as
the shape dependency that describes the extent of `x`'s only axis.

Why put shape information in a source node? Shapes are not always fixed Python
tuples of integers. Later examples have symbolic sizes on which other graph
facts depend. Representing a shape component as a UOp lets normal dependency
traversals see it.

Its shape is `()` because it is itself a scalar value. Its dtype is
`dtypes.weakint`, a literal integer dtype that can participate in promotion
without prematurely committing to a storage width.

### `N1`: the input parameter

`N1` is a floating parameter named `x`, assigned to parameter slot zero for
the `PYTHON` device. Its first source is `N0`. The PARAM shape rule interprets
that source as shape data, producing `(3,)`.

The remaining parameter facts live in a `ParamArg` payload. For now, `slot`,
`dtype`, `name`, and `device` are enough. Address spaces, sharding axes, and
volatility become meaningful in later chapters.

### `N2`: one node, two source positions

`N2.src` is `(N1,N1)`. The repeated entry is intentional. Multiplication has
two input positions, and both are supplied by `x`. A graph printer that turns
`src` into a set would silently change `x*x` into a one-input operation.

The output has shape `(3,)` because multiplication is elementwise and both
sources have that shape. Its dtype is floating because its sources are
floating.

### `N3`: the literal coefficient

`N3` is the scalar coefficient `2.0`. The Python spelling used integer `2`,
but construction in the context of floating `x` creates the weak floating
constant shown here. “Weak” allows the literal to follow the surrounding
promotion rules instead of independently forcing a particular floating
width.

`ConstFloat(2.0)` is a stable internal wrapper visible in `repr`. The
mathematical value is still two.

### `N4`: scalar-vector multiplication

`N4.src` is `(N3,N1)`: coefficient first, vector second. Its derived shape is
`(3,)` because the scalar broadcasts over the vector. No separate `EXPAND`
node is required merely to derive this UOp's shape.

Multiplication is mathematically commutative, but the raw source tuple is
ordered. `2*x` and `x*2` can therefore begin as distinct UOp constructions.
A later rule may prove and apply a canonical ordering; the constructor itself
does not perform general algebra.

### `N5`: combine the branches

`N5.src` is `(N2,N4)`. Both inputs have shape `(3,)` and floating dtype, so the
elementwise result does as well. This node is the direct IR counterpart of
the vector `x*x+2*x` from Chapter 4.

### `N6`: reduce one normalized leading axis

`N6` has one source, the vector `N5`. Its argument `(Ops.ADD,1)` says to combine
one normalized leading axis using addition. Removing the one axis from `(3,)`
leaves scalar shape `()`.

The `Ops.ADD` in the argument is the reduction combiner. It is not a second
ADD node and not a source edge. The integer `1` is the count of reduced axes
in this normalized representation. More complicated axis normalization is
deferred to Chapter 8.

## A UOp's five interning-key fields

The compact constructor is conceptually:

```python
UOp(op, dtype, src, arg, tag)
```

Those five fields form the key tinygrad uses to reuse an equivalent live UOp.
They should be read together:

| Field | Question to ask | Carried example |
| --- | --- | --- |
| `op` | What kind of definition or transformation is this node? | `PARAM`, `MUL`, `ADD`, `REDUCE` |
| `dtype` | What kind of element or scalar value does it produce? | `dtypes.float`, `dtypes.weakint` |
| `src` | Which ordered dependency nodes does it use? | `N2.src` holds `(N1,N1)` |
| `arg` | What hashable, operation-specific payload is not an edge? | constant value, `ParamArg`, `(Ops.ADD,1)` |
| `tag` | What optional, context-specific marker distinguishes this form? | normally `None` in this graph |

They are *interning-key fields*, not every observable fact attached to a UOp.
Shape is derived. Metadata and realized buffers are managed separately. A
node also exposes many derived properties reached by following its sources.

### `op`: interpret it in its graph stage

`op` is a member of the `Ops` enum. It tells the relevant passes which rule to
apply. The name alone is not a complete specification. `PARAM` in symbolic
integer algebra, `PARAM` for an input buffer, and a late `PARAM` in a rendered
function can carry different surrounding conventions.

When an operation is unfamiliar, inspect its construction and consumer in the
current stage. Do not infer all semantics from an English reading of its enum
name.

### `dtype`: stored after any default is derived

`dtype` is stored on the node and participates directly in identity. A caller
may omit it. In that case the metaclass first calls the dtype-production logic,
then places the resulting dtype into the interning key. “Derived by the
constructor” does not mean “a recursively derived property forever”; once the
node exists, dtype is one of its stored five fields.

A rewrite that changes sources may need a correspondingly valid dtype. Do not
copy a dtype mechanically when an operation's result type should change.

### `src`: dependencies, not merely arithmetic children

`src` is an ordered tuple of UOps. It can contain numeric operands, but it also
contains shape descriptions, buffer/index relationships, control-flow inputs,
or ordering dependencies. The general rule is:

> If another UOp is a dependency that traversals and rewrites must see, it
> belongs in `src`, not hidden inside `arg`.

Source position can carry meaning. The condition, true value, and false value
of `WHERE` cannot be sorted. Even for a commutative operation, reordering is a
transformation that requires the operation's rules; it is not a generic graph
cleanup.

### `arg`: payload interpreted by the operation

`arg` carries non-edge data whose meaning depends on `op`:

- a `CONST` carries its literal value;
- a `PARAM` carries a frozen `ParamArg`;
- a `REDUCE` carries its combiner and normalized axis count; and
- rendered or compiled artifacts can carry stage-specific text or bytes.

There is no safe universal parser for `arg`. Search the constructor and the
passes that consume that exact operation.

Because `arg` participates in a dictionary key, it must be hashable and
effectively immutable. A list or dictionary is normally invalid. Tuples and
frozen dataclasses are common when a payload has several parts.

### `tag`: identity without universal semantics

`tag` is another optional payload. Its interpretation belongs to the pass that
sets or matches it; there is no global meaning such as “optimized” or
“scheduled.” Most nodes in the carried graph have `tag=None`.

A non-`None` tag changes the interning key. Reconstructing the same five-field
form reuses it, but the tagged and untagged forms are different objects. Tags
must also be hashable and effectively immutable. For example:

```python
tagged = loss.rtag(("phase2", "demo"))

assert tagged is not loss
assert loss.rtag(("phase2", "demo")) is tagged
```

A list tag fails because a list cannot be hashed. Do not choose a mutable
payload merely because the field accepts `Any` at the type level.

### Hashable is necessary for all five fields

The complete cache key is:

```text
(op, dtype, src, arg, tag)
```

Every component must be hashable. `src` is a tuple whose UOps have identity
hashes; `Ops` and dtypes are hashable values; structured arguments such as
`ParamArg` are frozen. This is part of the representation contract, not a
minor Python implementation detail. A changing key would make cache lookup
and graph identity incoherent.

## Stored dtype versus recursively derived shape

Dtype and shape answer different questions:

- dtype describes the kind of each produced value; and
- shape describes how many axes and positions the value has.

A scalar float has dtype `dtypes.float` and shape `()`. An integer vector can
have dtype `dtypes.int` and shape `(3,)`. Neither determines the other.

### How the table's dtypes arise

The constructor has operation-specific dtype rules. In our graph:

- `PARAM` obtains its dtype from `ParamArg`;
- `CONST` obtains a weak dtype from the literal when no explicit dtype is
  supplied;
- broadcastable operations promote their source dtypes; and
- `REDUCE` passes through its value dtype.

The resulting dtype is filled before cache lookup and then stored. Two nodes
with otherwise identical fields but different stored dtypes have different
interning keys.

### How the table's shapes arise

Shape is not a sixth stored identity field. The internal `._shape` property is
computed recursively from the operation, sources, and argument, then cached.
The public `.shape` returns that tuple or raises if the operation has no tensor
shape.

For the carried graph:

```text
N0 CONST        -> ()
N1 PARAM        -> read N0 as the shape (3,)
N2 MUL          -> broadcast source shapes -> (3,)
N3 CONST        -> ()
N4 MUL          -> broadcast () with (3,) -> (3,)
N5 ADD          -> broadcast (3,) with (3,) -> (3,)
N6 REDUCE       -> remove one leading axis -> ()
```

Movement operations have explicit shape transformations. For example:

```python
cropped = x.shrink(((0, 2),))

assert cropped.shape == (2,)
assert cropped.dtype == x.dtype
```

The value remains floating, while its visible coordinate domain changes.

### Scalar is not shapeless

The empty tuple `()` is a valid tensor shape with zero axes and one scalar
position. `None` means an operation has no tensor shape in this representation.
Those cases must not be conflated:

```python
from tinygrad.uop.ops import UOp

assert loss._shape == ()
sink = UOp.sink(loss)
assert sink._shape is None
```

Asking `sink.shape` raises `RuntimeError` because `SINK` is a dependency root,
not a tensor value with axes. Do not generalize that every late operation is
shapeless: for example, `BINARY` has a byte-vector shape in this snapshot.
Inspect the exact operation's rule.

### Derived does not mean unimportant

Shape is absent from the five-field tuple, but the fields determine it through
the graph. Changing a source shape or a movement argument can change the
derived shape and every downstream shape. Other important properties—device,
bounds, address space, buffer base, and more—are also derived in particular
graph stages.

The useful distinction is not “stored facts matter, derived facts do not.” It
is “which input facts determine this property, and which nodes must be rebuilt
if those facts change?”

## Interning turns repeated construction into sharing

**Interning** means returning an existing object when a construction request
has the same representation as a live object. The technique is also called
**hash-consing** when applied to expression nodes.

At the pinned snapshot, `UOpMetaClass.__call__` performs this sequence:

1. derive a dtype if the caller omitted it;
2. form `(op,dtype,src,arg,tag)`;
3. look for that tuple in a weak cache;
4. if its weak reference still points to a live UOp, return that object; or
5. create a UOp and store a weak reference under the tuple.

Consequently, rebuilding a live subexpression can return the exact same
Python object:

```python
again = x * x
again_loss = (x * x + 2 * x).sum()

assert again is square
assert again_loss is loss
```

The shared `square` object then appears once in a node inventory even if many
consumers use it. This gives tinygrad structural sharing without requiring a
separate common-subexpression pass for identical construction.

### Four notions that must stay separate

It is easy to use the word “same” for four different claims:

1. **Python object identity:** `a is b` asks whether both references point to
   one live object.
2. **Interning-key equality:** the five-field tuples match, so constructing one
   while the other is live returns that object.
3. **The `.key` digest:** this snapshot recursively hashes `op`, `dtype`,
   `arg`, and source keys into bytes.
4. **Mathematical equivalence:** two expressions produce equal results for
   every valid input.

These claims are related but not interchangeable.

`x+0` and `x` are mathematically equivalent under appropriate dtype and
floating-point assumptions, but their raw structures differ. `a+b` and `b+a`
may be mathematically equivalent, while their ordered `src` tuples give
different interning keys. The UOp cache is not an algebra optimizer.

Conversely, `.key` is **not** the metaclass's cache key. In this snapshot it
omits `tag`:

```python
tagged = loss.rtag(("phase2", "demo"))

assert tagged is not loss
assert tagged.key == loss.key
```

That is not a contradiction. The two APIs serve different internal purposes.
Never document `.key` as the five-field tuple simply because both contain the
word “key.”

### The comparison sharp edge

Use `is` and `is not` for UOp object identity. At this snapshot `==` is not
overridden, so it retains default object-identity behavior, but `!=` **is**
overloaded to construct an `Ops.CMPNE` comparison UOp. This can surprise code
that expects a Boolean:

```python
same_object = square is rebuilt_square
different_object = loss is not tagged
```

Do not teach `a != b` as an identity check. It asks tinygrad to build a value
comparison. Explicit `is` also states the intended identity question clearly
if equality behavior evolves.

### The cache is weak, not permanent storage

The cache holds weak references. A weak reference can observe an object
without keeping it alive. Strong references from a root keep all reachable
sources alive, because every consumer stores its source tuple. But if no
ordinary Python reference reaches a UOp, the cache alone does not promise its
lifetime.

After an isolated node dies, an equivalent construction creates a live node
again. Do not compare old and new `id()` integers: CPython can reuse a memory
address. Local labels such as `N2` and process-local `id()` values are useful
for one inspection, not persistent graph IDs.

The practical guarantee is narrow:

> Equivalent construction reuses an equivalent UOp while that cached UOp is
> still alive.

## Treat UOps as logically immutable

The UOp dataclass is not mechanically frozen in this snapshot because freezing
has a performance cost. That is not permission to assign to `op`, `dtype`,
`src`, `arg`, or `tag`.

In-place mutation can invalidate the assumptions behind:

- the weak interning-cache entry created from the original five fields;
- already cached recursive properties such as shape;
- analyses and maps that treat one live object as one stable node; and
- consumers that were constructed under the original meaning.

Treat every UOp as an immutable value. Construct a replacement instead.

### `replace` creates or reuses a node

`UOp.replace` fills unspecified fields from the old UOp. If all five requested
fields are unchanged, it returns the old object. If any differs, it calls the
normal UOp constructor, so interning still applies:

```python
assert loss.replace() is loss

tagged = loss.replace(tag=("phase2", "demo"))
assert tagged is not loss
```

This is UOp's own method, and it routes the result through UOp interning.

### Replacing an interior node does not find its consumers

Nodes point to sources; they do not store parent/consumer pointers. Suppose we
replace the square with `x*3`:

```python
tripled = square.replace(src=(x, UOp.const(3.0)))
```

`tripled` is new, but `summed` and `loss` still point to the old path. Nothing
mutates them automatically. To make a root that reaches the replacement,
rebuild every consumer on a path from the changed node to that root:

```python
updated_sum = summed.replace(src=(tripled, scaled))
updated_loss = loss.replace(src=(updated_sum,))

assert loss.src[0] is summed
assert updated_loss.src[0] is updated_sum
assert updated_sum.src[1] is scaled
assert updated_loss is not loss
```

The original graph remains valid and unchanged. Any branch that does not
depend on the changed node can be shared directly into the new graph. This is
called a **persistent update**: old and new versions coexist and reuse
unchanged structure.

Chapter 6 automates this pattern. A graph rewrite walks sources, obtains
rewritten versions, and rebuilds only nodes whose inputs or rule result change.

## Traverse a DAG without counting shared nodes twice

### What topological order guarantees

A **topological order** of a DAG is an order in which every source appears
before every consumer that uses it. For our graph one valid order is
`N0,N1,N2,N3,N4,N5,N6`.

Sibling order is not semantic. `N3` could appear before `N2` while still
satisfying every dependency. Code should rely on the source-before-consumer
invariant, not on an arbitrary relative order between independent branches.

`loss.toposort()` returns an insertion-ordered dictionary whose keys are the
unique UOps. Convert it to a list when a numbered sequence is convenient:

```python
order = list(loss.toposort())
position = {node: i for i, node in enumerate(order)}

assert order[-1] is loss
assert all(position[source] < position[node]
           for node in order for source in node.src)
```

The second assertion is a direct proof of the property for this graph. It is
more useful than assuming that the method name guarantees your later code is
using the result correctly.

The implementation uses an explicit stack and a visited dictionary rather
than naive recursive Python calls. This avoids Python's recursion limit on
large compiler graphs and ensures each shared node is inserted once.

### Unique nodes are not source-edge positions

Topological traversal deduplicates nodes. `x` occurs once in `order`, and the
square occurs once. The traversal does **not** remove repeated entries from a
node's `src`:

```text
unique x nodes in toposort:       1
unique consumers of x:            2  (N2 and N4)
source positions occupied by x:   3  (N2[0], N2[1], N4[1])
```

Choose the correct quantity for the question. A compiler cost inventory might
visit each operation node once. An autodiff local rule must process both
positions of `N2.src`.

### `backward_slice` starts below the root

`loss.backward_slice` contains every transitive source but excludes `loss`
itself. Our seven-node topological order therefore corresponds to a six-node
backward slice:

```python
assert len(loss.toposort()) == 7
assert len(loss.backward_slice) == 6
```

Use `backward_slice_with_self` when the root belongs in the set. State the
choice when reporting a node count; otherwise an unexplained off-by-one can
look like a traversal bug.

### Build consumers from a chosen root

A UOp does not permanently store its consumers. The helper
`consumer_map_from_toposort(order)` reverses the edges visible under one root:

```python
from tinygrad.uop.ops import consumer_map_from_toposort

consumers = consumer_map_from_toposort(order)
```

For `x`, this map contains `N2` and `N4` once each. It uses a dictionary as an
ordered set of consumer nodes, so the two appearances of `x` in `N2.src` do
not create two `N2` entries. To recover edge positions, inspect each
consumer's `src` tuple:

```python
positions = [(consumer, i)
             for consumer in consumers[x]
             for i, source in enumerate(consumer.src)
             if source is x]
```

This yields `(N2,0)`, `(N2,1)`, and `(N4,1)`. The consumer map and source
tuples answer complementary questions.

Advanced traversal controls such as `gate` and `enter_calls` are deliberately
deferred. First make the default root, uniqueness, and edge-multiplicity rules
automatic.

## `Ops` contains several compiler stages

The `Ops` enum is broad by design. The following families are navigation aids,
not a replacement type system:

| Role | Representative operations | Question to ask |
| --- | --- | --- |
| Values and definitions | `CONST`, `PARAM`, `BUFFER`, `BIND` | Is this a literal, an input/symbol, storage, or a bound value? |
| Tensor math | arithmetic/comparison ALU ops, `CAST`, `REDUCE`, `WMMA` | Is it elementwise, a reduction, or specialized matrix work? |
| Shape and movement | `RESHAPE`, `EXPAND`, `PERMUTE`, `PAD`, `SHRINK`, `FLIP` | How are logical coordinates transformed? |
| Iteration and launch symbols | `RANGE`, `SPECIAL` | Which loop or launch dimension is represented? |
| Memory and addressing | `INDEX`, `LOAD`, `STORE` | Which storage and coordinates are involved? |
| Ordering and control | `SINK`, `AFTER`, `GROUP`, `BARRIER`, `IF`, `END` | Does this edge carry order/control rather than pure algebra? |
| Late artifacts | `LINEAR`, `PROGRAM`, `SOURCE`, `BINARY`, `INS` | Is this an ordered plan, rendered program, compiled bytes, or instruction? |

Operation groups such as `GroupOp.Binary`, `GroupOp.Movement`, and
`GroupOp.Commutative` are exact sets used by implementations and patterns.
They are often better evidence than the visual position of a name in the enum.

Three examples prevent common classification mistakes:

- `SHRINK` is declared beside `INDEX`, but it is explicitly a member of
  `GroupOp.Movement` and derives a cropped tensor shape.
- `GroupOp.Reduce` is the set of legal reduction combiners—`ADD`, `MUL`, and
  `MAX`—not a set containing the `Ops.REDUCE` node.
- `UNSHARD` has movement-like shape handling but is handled alongside the
  movement group rather than being an exact member of `GroupOp.Movement`.

Likewise, `CAST` belongs to `GroupOp.Elementwise` separately; it is not in the
`GroupOp.Unary` set. Read the exact set used by the code you intend to change.

### A repeatable search protocol

When an unfamiliar operation appears, do not read the entire compiler. Search
for the qualified name across the likely layers:

```bash
rg -n 'Ops\.SHRINK' tinygrad/uop tinygrad/mixin tinygrad/schedule tinygrad/codegen test
```

Answer these questions in order:

1. Where is it constructed?
2. How are dtype and shape determined?
3. Which `GroupOp` sets contain it?
4. Which pass consumes, transforms, or removes it?
5. Which focused tests state its behavior?

This produces a bounded reading path. The meaning of a UOp is the contract
shared by its constructors, derived properties, transformations, consumers,
and tests—not a guess based on one declaration.

## Runnable lab: inspect identity and traversal

**Hardware:** portable. This lab constructs IR and performs no tensor
execution.

The command must be run with the tinygrad study checkout as the current
directory so its editable environment imports the intended source. Replace
both paths before running:

```bash
export TINYGRAD_STUDY=/absolute/path/to/tinygrad-study
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
cd "$TINYGRAD_STUDY"
pwd
CACHEDB=/tmp/tinygrad-guide-phase2.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/uop_walk.py"
```

`pwd` should print the study checkout, not your home directory. If Python says
it cannot open `labs/phase2/uop_walk.py`, check `TINYGRAD_DOCS`: the lab belongs
to this guide repository, not to upstream tinygrad.

Before running, predict:

1. the seven operation names in topological order;
2. the three source positions containing `x`;
3. the two unique consumer nodes of `x`;
4. whether a rebuilt live loss is identical with `is`; and
5. whether tagging only the root changes `.key`.

### Read the output in four passes

The first block is the exact seven-row table explained earlier. The lab asserts
the operation sequence, root position, shapes, repeated square sources,
topological invariant, and node counts. If an assertion fails at the pinned
snapshot, the representation contract or the lab environment has diverged.

The next block reports:

```text
same square rebuilt:     True
same loss rebuilt:       True
x positions in square:   [0, 1]
all x source positions:  ['N2[0]', 'N2[1]', 'N4[1]']
x consumer nodes:        ['N2', 'N4']
backward slice nodes:    6
toposort nodes:          7
```

The two counts differ because the backward slice excludes its root. The edge
position and consumer lists differ because a consumer map deduplicates nodes,
while `src` preserves each operand position.

The tag block reports five distinct facts:

```text
tag changes identity:     True
same tagged form reused:  True
tag omitted from .key:    True
list arg rejected:        True
list tag rejected:        True
```

There is no inconsistency. `tag` belongs to the metaclass interning tuple,
`.key` is a different recursive digest that omits it, and list payloads are
unhashable. The two rejection lines show that the restriction applies to both
`arg` and `tag`.

The persistent-update block shows that replacing an interior node does not
change the old root. It changes `x*x` to `x*3`, rebuilds the sum and root,
shares the unchanged `2*x` branch, and leaves both roots with scalar shape.
The final weak-cache block releases an isolated constant and confirms that an
equivalent node can later be constructed again.

The lab deliberately prints local `N` labels instead of Python `id()` values.
Its claims should be stable across processes even though memory addresses are
not.

## Guided source tour: one question per stop

All source links below target commit
`874d33128b4e4785beea736d97df6716e0321717`. Do not open them as isolated
declarations. At each stop, first answer the question from the graph and lab,
then use the narrow range to verify one mechanism. The “ignore” sentence keeps
unfamiliar neighboring machinery from becoming an accidental prerequisite.

### Stop 1: where did `shape=(3,)` go?

Read [`shape_to_shape_arg` at lines 97–103](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L97-L103),
then [`UOp.param` at lines 1161–1167](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1161-L1167).

Question: why is the first row `CONST 3`, and why is it `N1`'s source rather
than part of the parameter payload?

Translation: a one-dimensional shape becomes one scalar shape UOp.
`UOp.param` places that descriptor in `src[0]` and places slot, dtype, name,
device, and related attributes in `ParamArg`.

Ignore the multi-device `axis` adjustment. It changes sharded parameter shape
and is not active in this single-device example.

### Stop 2: why is `ParamArg` a valid payload?

Read [`ParamArg` at lines 22–37](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L22-L37).

Question: what makes this structured object safe inside the cache key?

Translation: it is a frozen value dataclass, giving stable field-based hashing
and equality. In the table, interpret only `slot`, `dtype`, `name`, and
`device`.

Ignore `addrspace`, `axis`, and `volatile` until memory lowering, sharding, or
runtime behavior gives them a concrete role.

### Stop 3: when does construction reuse a node?

Read [`UOpMetaClass.__call__` at lines 193–207](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L193-L207).

Question: in what order do dtype derivation, cache lookup, and construction
occur?

Translation: line 197 fills a missing dtype. Lines 205–206 form the five-part
tuple, dereference any cached weak reference, return a live hit, or create and
cache a weakly referenced UOp.

Ignore the optional `SPEC` validation block at lines 201–204 for now. It checks
additional invariants but is not the identity mechanism.

### Stop 4: where are the five fields and safe replacement defined?

Read [the UOp fields and `replace` at lines 243–265](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L243-L265).

Question: why can `replace()` return either the old object or an interned new
one?

Translation: lines 245–250 declare the five stored fields. `replace` fills
missing values from the old node, returns `self` when the five-tuple is
unchanged, and otherwise calls `UOp` with the new tuple. `rtag` is a convenience
wrapper around that mechanism.

Ignore buffer lifetime handling in `__del__`; the guide's graph has no realized
buffer.

### Stop 5: why can different identities have equal `.key`?

Read [the recursive `.key` property at lines 272–274](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L272-L274).

Question: which interning field is absent from the digest input?

Translation: the digest includes this node's `op`, `dtype`, and `arg`, then
each source's recursive `.key`. It omits `tag`, so tagging only the root changes
the metaclass cache identity without changing this digest.

Ignore what other passes use the digest for. The only contract needed here is
that it is not the five-field interning tuple.

### Stop 6: which comparison asks about identity?

Read [the comparison overloads at lines 327–337](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L327-L337).

Question: why should graph code use `is not` instead of `!=`?

Translation: `__eq__` is intentionally not overridden at this snapshot, while
`__ne__` delegates to `.ne()` and constructs an `Ops.CMPNE` UOp. `is` and
`is not` are Python's unambiguous object-identity operators.

Ignore the numerical comparison implementation below this range; Chapter 6
will introduce operation-building and matching rules.

### Stop 7: can you reproduce every dtype in the table?

First read [literal promotion in `_broadcasted` at lines 21–30](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/elementwise.py#L21-L30).
Then read [the start of `dtype_from_uop` at lines 112–118](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L112-L118),
find REDUCE in [the pass-through cases at lines 134–149](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L134-L149),
then read [the relevant PARAM, CONST, unary, broadcast, and movement cases at lines
169–190](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L169-L190).

Question: which rule explains each of `weakint`, `float`, `weakfloat`, and the
reduction's float?

Translation: `_broadcasted` computes a common output dtype and can lift a weak
integer constant to weak float before constructing the ALU node; that is why
source spelling `2` becomes `ConstFloat(2.0)` here. PARAM reads its `ParamArg`;
CONST classifies an untyped literal; broadcastable ops promote sources; and
REDUCE's pass-through case preserves its first source's dtype.

Ignore dtypes for calls, images, tuples, and machine instructions. None occurs
in this graph.

### Stop 8: can you reproduce every shape in the table?

Read [scalar, PARAM, and other base shape cases at lines 379–405](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L379-L405),
then [reduction, broadcasting, and public `.shape` at lines
457–482](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L457-L482).
For the scalar/shapeless contrast, also read [the `SINK` and `CALL` base cases
at lines 335–345](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L335-L345).

Question: why do the literal and loss both have `()`, while the operations
between them have `(3,)` and a `SINK` has no shape?

Translation: CONST is scalar; PARAM decodes its shape source; elementwise
operations broadcast; REDUCE removes its normalized leading axes; and public
`.shape` raises when the recursive result is `None`. `SINK` is one operation
whose base case returns `None`.

Ignore WMMA, bitcasts, and multi-device cases. Revisit them only with an
example that contains those operations.

### Stop 9: why does traversal list nodes once but preserve both square edges?

Read [`consumer_map_from_toposort` at lines 104–110](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L104-L110),
then [`backward_slice` and `toposort` at lines 285–315](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L285-L315).

Question: where is uniqueness enforced, and where is edge multiplicity still
available?

Translation: dictionaries serve as visited/ordered sets of nodes. The consumer
map also stores unique consumer nodes. Neither changes the original `src`
tuple, which retains `N1` in both positions of `N2`.

Ignore traversal gates and call-boundary controls until a concrete pass uses
them. The default traversal is sufficient for this chapter.

### Stop 10: is enum neighborhood a taxonomy?

Read [the `INDEX`/`SHRINK` declarations at lines 49–53](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/__init__.py#L49-L53),
[the exact operation groups at lines 117–135](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/__init__.py#L117-L135),
and [movement shape handling at lines 428–452](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L428-L452).

Question: should `SHRINK` be classified from the heading beside its enum
declaration or from the exact group and behavior?

Translation: `SHRINK` is in `GroupOp.Movement`; its shape rule validates a
crop and returns the cropped sizes. The enum's visual sections help humans
navigate, but exact sets and consumers define reusable semantic categories.

Ignore the other movement cases until Chapter 8 derives view and shape
mechanics in detail.

## Controlled extensions with worked answers

Attempt each change in a temporary copy or a Python prompt. Predict identity,
sources, dtype, and shape before inspecting the result.

1. Build `left=x*2`, `right=x*3`, then compare `left+right` with
   `right+left` using `is`.
2. Change `loss` to use `(Ops.MAX,1)` as its REDUCE argument with
   `loss.replace(arg=(Ops.MAX,1))`.
3. Replace `square` with `x*3` in the carried graph while keeping the scaled
   branch. Build a new root without mutating any old node.
4. Construct `x.shrink(((0,2),))` and classify it from `GroupOp`, dtype, and
   shape rather than enum position.
5. Compare `loss.shape`, `UOp.sink(loss)._shape`, and
   `UOp.sink(loss).shape`.
6. Count consumers and source positions for `x*x*x`, taking Python's
   left-associative evaluation into account.

??? success "Answers"

    1. The two ADD roots are distinct raw objects because their ordered source
       tuples are reversed. Their dtype and shape agree. Mathematical
       commutativity does not make the five-field keys identical.
    2. The replacement has the same source, dtype, tag, and scalar shape, but a
       different argument and identity. Its semantics change from a sum to a
       maximum over the one reduced axis. This illustrates why `arg` matters
       even when graph edges do not change.
    3. One valid persistent update is:

       ```python
       tripled = square.replace(src=(x, UOp.const(3.0)))
       updated_sum = summed.replace(src=(tripled, scaled))
       updated_loss = loss.replace(src=(updated_sum,))
       ```

       `tripled is x*3` is true at the pinned snapshot. `scaled` is shared;
       `square`, `summed`, and `loss` remain unchanged; the new nodes form a
       path to `updated_loss`.
    4. It is `Ops.SHRINK`, a member of `GroupOp.Movement`. It preserves
       `dtypes.float` and changes `(3,)` to `(2,)`.
    5. `loss.shape` is `()`. The SINK's internal shape is `None`; requesting
       public `.shape` raises `RuntimeError`. Scalar and shapeless are
       different states.
    6. Python first constructs `first=x*x`, whose two positions point to `x`,
       then `root=first*x`, whose second position points to `x`. There are two
       unique MUL nodes. `x` has two unique consumers and three edge positions,
       just as in the carried two-branch example but with different topology.

## Debugging map

When a UOp inspection is surprising, localize the category before reading a
large pass:

| Observation | First check | Likely issue |
| --- | --- | --- |
| A literal or parameter has an unexpected dtype | Was dtype explicit, derived, or weakly promoted? | constructor/dtype rule |
| Shape differs while stored dtype is right | Which source or movement/reduction argument determines `_shape`? | shape derivation or malformed source |
| Two simultaneously live constructions are not identical | Compare all five fields, including source order and tag | interning-key difference |
| `.key` agrees but `is` does not | Is the only difference a tag? | confusing digest with cache key |
| A node appears twice in a custom printout | Did the traversal deduplicate by UOp identity? | tree walk used on a DAG |
| A consumer count loses a square contribution | Did code need source positions rather than unique consumers? | edge multiplicity discarded |
| An interior replacement has no effect at the root | Were all consumers on the path rebuilt? | persistent update stopped early |
| `.shape` raises | Is `._shape` `None`, meaning this op is shapeless? | wrong property for this graph stage |
| An operation seems to be in the wrong family | Did you inspect exact `GroupOp` membership and behavior? | enum-neighborhood guess |

Do not infer execution or kernel behavior from this raw algebra graph. Chapter
7 introduces planning and realization; later chapters introduce lowering and
target code.

## Checkpoint: produce evidence, not just definitions

Save these four artifacts:

```text
1. a hand-drawn tree and DAG for (x*x + 2*x).sum()
2. the lab's seven-row table, annotated row by row
3. a topological-order proof plus x's consumer and edge-position lists
4. old and new roots from one persistent path rebuild
```

You are ready for rewrite rules when you can answer all of these without
guessing:

1. Why are there six conceptual math/value nodes but seven actual UOps?
2. Why does `x` occur once in `toposort`, twice in its consumer map, and three
   times among source positions?
3. Which five fields form the interning key, and why must every part be
   hashable and effectively immutable?
4. Why is dtype stored even when the constructor initially derives it?
5. Why is shape derived, and how does each row obtain its shape?
6. What is the difference between scalar shape `()` and shapeless `None`?
7. Why can tagged and untagged roots have equal `.key` digests but different
   identity?
8. What guarantee does a weak cache provide, and what does it not provide?
9. Why does replacing `x` alone not alter `loss`?
10. Why is `SHRINK` a movement operation despite its enum neighborhood?

If any answer is vague, return to the corresponding row or source stop. The
goal is not to memorize every `Ops` member. It is to have a reliable procedure
for an unfamiliar graph.

## Quick reference

| Need | Rule |
| --- | --- |
| Recover one node's meaning | Read `op`, stored `dtype`, ordered `src`, op-specific `arg`, and context-specific `tag` together. |
| Interning identity | Live nodes reuse `(op,dtype,src,arg,tag)` when all parts match. |
| Compare object identity | Use `is` / `is not`; `!=` constructs `CMPNE`. |
| Interpret `.key` | It is a recursive digest, not the five-field cache key, and omits tag here. |
| Change a UOp | Use a constructor or `u.replace(...)`; never assign fields in place. |
| Change an interior value | Rebuild every consumer path to the chosen root; share unchanged branches. |
| List unique dependencies first | `order = list(root.toposort())` |
| Prove topological order | Check every `source` has a lower position than its consumer. |
| Get sources excluding root | `root.backward_slice` |
| Get unique consumers | `consumer_map_from_toposort(order)` |
| Count edge positions | Enumerate each consumer's original `src` tuple. |
| Read shape safely | `._shape` can be `None`; public `.shape` raises in that case. |
| Classify an unfamiliar op | Search constructors, dtype/shape rules, exact groups, consumers/passes, and tests. |

## Optional reinforcement—not missing prerequisites

- Review Python's official sections on
  [objects, values, and types](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types)
  and [weak references](https://docs.python.org/3/library/weakref.html) if
  `is` or weak lifetime remains unclear. Stop once you can explain why a weak
  cache can reuse an object without owning its lifetime.
- Read `spec/tinyspec.pdf` in the pinned checkout only after the checkpoint.
  It is a compact formal inventory, not a beginner-first explanation.
- Run the lab once with `SPEC=2` added to the environment. This asks tinygrad
  to validate more IR invariants. It should not change the seven-node output.
- Choose one new op from a small Tensor expression and apply the five-question
  search protocol. Record what you can prove and what remains deferred.

## What is deliberately left for later

- Chapter 6 explains patterns, matchers, rewrite order, fixed points, and
  rewrite debugging.
- Chapter 7 explains how lazy value graphs become scheduled work and when
  buffers are realized.
- Chapter 8 derives broadcasting, symbolic shapes, views, movement, and index
  transformations in depth.
- Chapters 9–12 follow UOps through lowering, optimization, rendering,
  compilation, and runtime invocation.
- Chapter 14 returns to memory, ordering, and control operations with the
  stage context required to understand them.

You do not need those topics to proceed. You need the ability to inspect a
small UOp root without losing identity, dependency direction, derived shape,
or edge multiplicity. That is the foundation Chapter 6 assumes.

[← Tensor frontend](04-tensor-and-autograd.md) · [Next: Rewrites →](06-rewrites.md)
