# 5. UOp graphs

## Purpose

UOps are the language in which most tinygrad compiler questions are asked.
They represent lazy tensor expressions, symbolic shape arithmetic, scheduled
kernels, memory effects, lowered control flow, rendered source, binaries, and
execution plans at different moments in the pipeline.

After this chapter you should be able to take an unfamiliar UOp root and:

- inventory its nodes without confusing a DAG for a tree;
- explain each node's operation, dtype, sources, argument, tag, and derived
  shape;
- distinguish identity sharing from coincidentally similar printed text; and
- classify an unfamiliar `Ops` member well enough to find the next pass or
  test.

That is the minimum foundation for reading rewrites in Chapter 6.

## Prerequisite gate

You should already be able to draw the lazy forward and gradient graphs from
Chapter 4. You also need two Python concepts:

1. `a is b` asks whether two references designate the same object, while
   `a == b` may ask a type-defined value question; and
2. a weak reference does not keep its target alive.

If either is unfamiliar, read the official Python sections on
[objects, values, and types](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types)
and [weak references](https://docs.python.org/3/library/weakref.html) only until
you can explain those two sentences.

No graph-algorithm course is required. This chapter defines the only traversal
property you need: in a topological order, every source appears before the node
that uses it.

## Mental model: one compact node, many IR states

At the recorded snapshot, the core UOp fields are:

```python
UOp(op, dtype, src, arg, tag)
```

Treat a UOp as logically immutable even though the dataclass is not frozen for
performance. Build a changed node with `replace` or a constructor; do not
mutate `op`, `dtype`, `src`, `arg`, or `tag` in place. Interning and cached
derived properties rely on that discipline.

### The five identity fields

| Field | Question it answers | Typical examples |
| --- | --- | --- |
| `op` | What kind of node is this? | `ADD`, `RESHAPE`, `REDUCE`, `LOAD`, `STORE`, `RANGE`, `PROGRAM` |
| `dtype` | What kind of value does it produce? | `dtypes.float`, `dtypes.int`, `dtypes.bool`, `dtypes.void` |
| `src` | Which ordered UOps does it depend on? | operands, a value being moved, shape/range nodes, buffer/index nodes, effect dependencies |
| `arg` | What operation-specific non-edge payload is needed? | constant value, permutation, reduction description, parameter metadata, target/source text |
| `tag` | What optional pass-local marker distinguishes this form? | `None` normally; a rewrite-controlled marker when a pass assigns meaning |

All five participate in the interning key. `arg` is not a generic bag of
mutable state; it must support stable equality and hashing for that key. Its
meaning is defined by `op`, so inspect constructors and consumers instead of
guessing from a `repr`.

`src` means “edges to dependencies,” not merely “numeric operands.” A `PARAM`
can source a node encoding its shape. `AFTER` sources a passed-through value and
operations that must precede its consumers. `INDEX` connects storage with
index/validity expressions. Source position can be semantic, so do not sort a
`src` tuple unless a specific rewrite proves that the operation is
commutative.

`tag` has no project-wide interpretation. Patterns can match it and graph
rebuilds preserve it. Because it changes identity, a tag can make a one-pass
rewrite form distinct from its untagged form; Chapter 6 shows why this can be
useful and dangerous.

### Stored dtype, derived shape

`dtype` is a stored identity field. If a constructor omits it,
`dtype_from_uop` derives an expected dtype from the operation, sources, and
argument. Optional spec checking can reject disagreement.

Shape is different. A UOp has a recursively derived `_shape`; the public
`.shape` raises if that operation has no tensor shape. Examples:

- `CONST` is scalar, so its shape is `()`;
- elementwise operations broadcast their source shapes;
- `RESHAPE`, `PERMUTE`, and other movement operations transform source shape;
- `REDUCE` removes the reduced leading axes in its normalized UOp form; and
- late containers such as `SINK`, `LINEAR`, `PROGRAM`, and `SOURCE` do not have
  a tensor shape.

Device, bounds, address space, buffer identity, and other properties are also
derived through the graph. “It is not in the five fields” does not mean “it is
unimportant”; it means you must find the derivation before changing an input
that affects it.

## Interning makes a DAG

`UOpMetaClass.__call__` keeps a weak cache keyed by
`(op, dtype, src, arg, tag)`. If an equivalent live UOp already exists, the
constructor returns it:

```python
shared = x * 2
root = shared + shared

assert root.src[0] is root.src[1]
assert ((x * 2) + (x * 2)) is root
```

This is hash-consing. It supplies structural sharing and an important form of
common-subexpression reuse before a separate pass would need to discover it.
It also lets many algorithms use UOps as identity-keyed dictionary entries.

There are three qualifications:

1. The cache stores weak references. If no strong reference keeps a node
   alive, reconstructing it later is not promised to preserve its old Python
   `id`.
2. A different dtype, source order, argument, or tag is a different key, even
   if a pretty printer makes the nodes look similar.
3. Interning is representation equality, not a proof of mathematical
   equivalence. `x+0` and `x` need a semantics-preserving rewrite; the cache
   will not equate them.

The natural construction direction—from existing sources to a new consumer—
produces an acyclic graph under normal use. A shared source can have many
consumers, but a UOp stores only its sources. If an analysis needs consumers,
build that reverse map for the chosen root.

## Topological traversal

`root.toposort()` walks backward through `src`, deduplicates by UOp identity,
and returns a dictionary whose iteration order places sources before their
consumers. The root is last. A dictionary is used because insertion order gives
the sequence while membership gives an efficient visited set.

```python
order = list(root.toposort())
position = {u: i for i, u in enumerate(order)}

assert order[-1] is root
assert all(position[s] < position[u] for u in order for s in u.src)
```

For a diamond graph, the shared node occurs once in `order`, not once per path.
That property is essential for gradient accumulation, rewrite memoization,
cost accounting, and execution dependency analysis.

Useful traversal variants include:

- `toposort(gate=...)`, which can omit a rejected node and the sources reached
  only through it;
- `topovisit(visitor, cache)`, which computes a result only after source
  results exist;
- `backward_slice`, the root's transitive sources without the root; and
- `consumer_map_from_toposort`, which constructs source-to-consumer edges.

Do not casually use recursive Python functions for real compiler graphs. The
current implementations use explicit stacks to avoid the Python recursion
limit and to preserve sharing.

## `Ops` is a union of roles, not one abstraction level

The `Ops` enum deliberately covers nodes used across the whole pipeline. A
UOp is therefore not necessarily a “micro-instruction,” and an op valid in one
graph state may be illegal in another. The spec and the pass consuming the
graph define stage-specific legality.

Use these role families to orient yourself; they are a reading aid, not a new
type system:

| Role | Representative ops | First question |
| --- | --- | --- |
| Values and definitions | `CONST`, `PARAM`, `BUFFER`, `BIND` | Is this a literal, symbolic/input value, or storage identity? |
| Tensor math | `ADD`, `MUL`, `WHERE`, `CAST`, `REDUCE`, `WMMA` | Is it elementwise, a reduction, or specialized matrix work? |
| Shape and movement | `RESHAPE`, `EXPAND`, `PERMUTE`, `PAD`, `SHRINK`, `FLIP`, `UNSHARD` | Does it reinterpret tensor coordinates or device shards without immediately moving data? |
| Iteration and symbols | `RANGE`, `SPECIAL` | Which loop, launch dimension, or bounded symbolic value does it represent? |
| Memory and addressing | `GETADDR`, `INDEX`, `LOAD`, `STORE`, `COPY`, `STAGE` | Where is the value stored, indexed, or materialized? |
| Ordering and control | `SINK`, `AFTER`, `GROUP`, `BARRIER`, `IF`, `ENDIF`, `END` | Is the edge about an effect/order requirement rather than a pure value? |
| Functions and calls | `FUNCTION`, `CALL`, `TUPLE`, `GETTUPLE` | Is this reusable graph body, invocation, or multi-result plumbing? |
| Late artifacts | `LINEAR`, `PROGRAM`, `SOURCE`, `BINARY`, `INS` | Has the graph become an execution list, rendered program, compiled bytes, or instruction form? |

`GroupOp` collects semantic sets used by code and patterns: unary, binary,
ternary, elementwise, movement, commutative, associative, comparison, and
reduction-capable operations. Membership is often more informative than the
enum's visual section, but read the exact set before relying on it. For
example, a broad `GroupOp.ALU` pattern includes boolean and comparison behavior
that may invalidate an algebraic rule written with only floats in mind.

When an unfamiliar op appears, use this navigation sequence:

```bash
rg -n 'MY_OP' tinygrad/uop tinygrad/schedule tinygrad/codegen tinygrad/runtime test
```

Find its constructors, derived dtype/shape behavior, rewrite rules, final
consumer, and tests. The path through those locations tells you what the op
means more reliably than its name alone.

## Source tour

All links are pinned to commit
`874d33128b4e4785beea736d97df6716e0321717`.

| Read this | What to extract |
| --- | --- |
| [`Ops` and `GroupOp`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/__init__.py#L12-L140) | Inventory the roles and the exact reusable operation sets. |
| [`dtype_from_uop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L134-L191) | See how result dtype depends on op, sources, and arg, including void and weak dtypes. |
| [`UOpMetaClass`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L193-L218) | Verify the five-part weak-cache key and optional spec check. |
| [`UOp` fields, `replace`, and traversal](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L245-L327) | Read identity fields, logical replacement, `backward_slice`, `toposort`, and `topovisit`. |
| [Shape derivation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L333-L482) | Trace a scalar, movement, broadcastable, reduction, and shapeless late op. |
| [`spec/tinyspec.pdf`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/spec/tinyspec.pdf) | Use after this chapter as a compact formal inventory, not as the first explanation. |
| [`test/null/test_uop_graph.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_uop_graph.py#L92-L166) | See identity deduplication and small graph-rewrite assertions in executable form. |

## Lab: make sharing visible

**Hardware:** Portable. This lab creates IR only.

Run [`labs/phase2/uop_walk.py`](../../labs/phase2/uop_walk.py) from the tinygrad
study checkout:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
CACHEDB=/tmp DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/uop_walk.py"
```

Before running, draw `(x*2) + (x*2)` as both a tree and a DAG. Assign a local
number to each unique node and predict the topological order.

The snapshot prints a table like:

```text
local  op        dtype              shape       src       arg
N0     STACK     dtypes.void        ()          []        None
N1     PARAM     dtypes.float       ()          [N0]      ParamArg(...)
N2     CONST     dtypes.weakfloat   ()          []        ConstFloat(2.0)
N3     MUL       dtypes.float       ()          [N1,N2]   None
N4     ADD       dtypes.float       ()          [N3,N3]   None
```

The surprising `STACK` is not hidden arithmetic: it is the scalar shape source
of this symbolic `PARAM`. `N3` appears once, while the root refers to it twice.
The lab also shows that adding a tag changes the interned root and recreating
the same tagged form returns that tagged object.

### Exercises

1. Change the root to `(x*2) + (x*3)`. Mark which nodes remain shared and
   explain every changed cache-key field.
2. Use two `UOp.variable` calls with the same name, bounds, and dtype. Test
   identity. Then change only one bound and find where that difference lives
   inside the `PARAM` argument.
3. Build a consumer map from the topological order and print each node's local
   consumer numbers. Confirm the shared `MUL` has one consumer node but two
   source positions in that consumer.
4. Replace the root's tag, dtype, source order, and argument one at a time where
   construction is legal. Record which change affects identity, shape, and
   semantics.
5. Inspect the UOps for a small `Tensor` matmul and classify every operation by
   the role table. If one does not fit, refine the table from its consumers
   rather than forcing it into the nearest label.

## Checkpoint

Proceed when you can inspect an arbitrary root and answer:

- Which facts are stored in the interning key, and which are derived?
- Why is UOp identity useful but not a permanent node ID?
- Why does `src` include a scalar `PARAM`'s shape descriptor?
- How do you prove that a traversal is topological?
- Why does a diamond appear once in `toposort()`?
- Why can two mathematically equivalent expressions still be different UOps?
- Which role and graph stage does each unfamiliar `Ops` member belong to?

As evidence, save one numbered UOp table with at least one shared source and a
short explanation of every row.

## Quick reference

| Need | Rule of thumb |
| --- | --- |
| Core identity | `(op, dtype, src, arg, tag)` |
| Change a node | Use `u.replace(...)` or a constructor; treat live UOps as immutable. |
| Compare representation identity | Use `is`; UOps are identity-keyed and interned while live. |
| Compare semantics | Identity is insufficient; use a proof plus tests or execution. |
| Get dependencies first | `list(root.toposort())` |
| Get all transitive sources | `root.backward_slice`; add the root when needed. |
| Get consumers | Build a reverse map from one chosen root/toposort. |
| Read shape safely | `_shape` may be `None`; `.shape` raises for shapeless ops. |
| Interpret `arg` | Search constructors and consumers for that exact `op`. |
| Interpret `tag` | Find the pass that assigns/matches it; there is no universal meaning. |
| Classify an op | Check `GroupOp`, constructors, derived properties, pass consumer, and tests. |
