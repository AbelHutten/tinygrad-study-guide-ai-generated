# 6. Pattern matching and graph rewriting

## Purpose

Rewrites are tinygrad's main mechanism for simplification, canonicalization,
autograd, lowering, target decomposition, scheduling transformations, and
instruction selection. A five-line rule can remove a large class of work—or
silently miscompile a large class of programs.

This chapter teaches the whole local contract:

```text
UPat describes a matching subgraph
  → PatternMatcher chooses a rule
  → callback returns a replacement or None
  → graph_rewrite drives that decision across a UOp DAG
```

After the lab you will have a guarded simplification with positive, negative,
dtype, shape/broadcast, semantic, order, and termination tests. The example is
deliberately small; the validation discipline scales to renderer selection and
kernel rewrites.

## Prerequisite gate

Continue only if you can read the numbered UOp table from Chapter 5 and tell
which repeated labels are the same object. You must also be comfortable with
the distinction between:

- **syntactic match:** the graph has a requested form;
- **semantic precondition:** replacing that form preserves the property this
  pass promises; and
- **canonical preference:** among equivalent forms, this pass wants one
  particular representation.

If those blur together, read the official MLIR
[Pattern Rewriter documentation](https://mlir.llvm.org/docs/PatternRewriter/)
through pattern benefits, restrictions, and bounded recursion. Do not learn
MLIR's API. Return when you can explain why a matching pattern may still be
unsafe to replace and why two individually valid rules can loop together.

## A rewrite has three separate parts

Consider a teaching rule that removes zero from a concrete integer addition:

```python
def remove_integer_add_zero(add, x, zero):
  if add.dtype not in dtypes.ints: return None
  if zero.base.op is not Ops.CONST or zero.base.val != 0: return None
  if x.dtype != add.dtype or x.shape != add.shape: return None
  return x

pm = PatternMatcher([
  (UPat(Ops.ADD, name="add",
        src=[UPat.var("x"), UPat.var("zero")]),
   remove_integer_add_zero),
])
```

The `UPat` says “an `ADD` with these two bindings in either source order.” The
callback proves the narrower conditions: concrete integer output, a zero-like
source, and a replacement whose dtype and shape already equal the result. The
driver decides where and how often to try it.

Keeping those responsibilities separate makes reviews clearer. A broad pattern
plus a guarded callback is often simpler than encoding every semantic fact in
the pattern, but the guard must be tested directly.

## `UPat`: describe representation, bind identity

A `UPat` can constrain `op`, `dtype`, source patterns, `arg`, and `tag`. A
`name` binds the matched UOp and passes it to the rule callback.

The common constructors are:

```python
UPat(Ops.ADD, name="add")             # any ADD, sources unconstrained
UPat.var("x")                          # any UOp, bound as x
UPat.var("x", dtype=dtypes.int32)      # any matching dtype
UPat.cvar("c", arg=0)                  # CONST with value/arg zero
UPat.const(0, dtypes.int32)             # exact CONST pattern
UPat.any(pattern_a, pattern_b)          # alternatives
```

Patterns share UOp-like operator sugar:

```python
UPat.var("x") + 0
UPat.var("x") * UPat.var("y")
UPat.var("cond").where(UPat.var("a"), UPat.var("b"))
```

This constructs patterns, not UOps. Read the callback names to learn what each
subpattern bound.

### Source matching rules

The form of `src` changes the match:

| Pattern form | Meaning |
| --- | --- |
| `src=None` | Do not constrain sources. |
| `src=(p0, p1)` | Ordered sources with exactly this length by default. |
| `src=[p0, p1]` | Try permutations; useful for a deliberately commutative match. |
| `src=p` | Every source matches the repeated pattern; any length. |
| `allow_any_len=True` | Require at least the listed source prefix rather than exact length. |

Do not use list form merely because an op “looks symmetric.” Check
`GroupOp.Commutative` and the semantics of the current graph stage. `INDEX`,
`STORE`, `WHERE`, division, and effect dependencies are positional.

If one name occurs more than once in a pattern, both positions must bind the
same UOp object. With interning, this is how a pattern such as `x op x`
recognizes shared representation identity. It does not mean two unrelated
nodes that happen to print alike.

Top-level patterns in a `PatternMatcher` must constrain at least one `op`, so
the matcher can index candidates efficiently. Dtype matching also accepts a
vector dtype whose scalar dtype satisfies the pattern. `arg=None` and
`tag=None` mean unconstrained in `UPat`; read the implementation before trying
to distinguish an explicit `None` payload.

## `PatternMatcher`: ordered, first successful replacement

A matcher stores `(UPat, callback)` pairs in declaration order, indexed by
root `op`. For a candidate UOp it:

1. considers only patterns registered for that `op`;
2. cheaply rejects patterns whose required source operations are absent;
3. binds names by matching the pattern recursively; and
4. returns the first callback result that is neither `None` nor the original
   UOp.

For graph rewriting, a callback should return a replacement `UOp` or `None`.
`None` means “this representation matched broadly, but the semantic guard did
not prove replacement.” Returning the original object also counts as no
rewrite.

Rule order is therefore observable semantics. Adding matchers with `pm_a +
pm_b` concatenates their rule lists; it does not compute an unordered union.
Put specific/legality-preserving normalization before rules that would erase
the evidence it needs, and test overlaps explicitly.

A callback can request shared context by declaring a `ctx` parameter. Named
pattern variables cannot be called `ctx`. Pattern callbacks are compiled or
reconstructed by the matcher and closures are unsupported, so define
non-trivial callbacks at module scope and pass changing state through `ctx`.

`PatternMatcher.rewrite(node)` handles one node. It does not walk a graph. That
is the driver's job.

## `graph_rewrite`: traversal is part of semantics

`graph_rewrite(root, pm, ...)` preserves sharing with a replacement map and
rebuilds consumers whose sources changed. Its mode controls whether replacement
subgraphs are revisited and whether matching occurs before or after sources.

The exact snapshot behavior is:

| Invocation | Matching order | Replacement subgraph |
| --- | --- | --- |
| `graph_rewrite(root, pm)` | Greedy; sources are processed, the rebuilt node is matched, and newly produced nodes continue through the driver. | Re-entered; rules can compose to a stable form. |
| `graph_rewrite(root, pm, bottom_up=True)` | Greedy pre-order/fixed-point matching at a node before descent; a node rebuilt after source changes is reconsidered. | Re-entered. |
| `graph_rewrite(root, pm, walk=True)` | Single post-order walk: sources first, then at most one successful replacement at the rebuilt node. | Not entered. |
| `graph_rewrite(root, pm, bottom_up=True, walk=True)` | Single pre-order walk: try the node first; a successful match skips its original children. | Not entered. |

Some upstream comments call the default `walk=True` mode “top-down” despite
its observable post-order callback sequence. When order matters, state
“pre-order” or “post-order” and write a visit-order test rather than relying on
the informal label.

The optional `bpm` matcher supports bidirectional traversal: it is tried before
descent, while `pm` is tried after sources and rebuild. `enter_calls=False` is
the default and protects function or opaque call bodies from ordinary walks.
These options are useful, but a new contributor should choose the simplest
driver that expresses the pass contract and test that exact invocation.

Use greedy rewriting for interacting canonicalization rules that are intended
to compose. Use `walk=True` when replacements are final and must not themselves
be traversed—for example, an exact substitution map. It is not a switch to use
merely because a greedy rule loops; that would change the transform's meaning.

## Termination and ordering

A rewrite system needs a decreasing measure. Depending on the pass, a rule may
reduce node count, remove a forbidden operation, lower an abstraction level,
or move an expression toward one canonical ordering. Write that measure in the
rule comment or test plan.

Common non-termination shapes include:

```text
A → B and B → A
x + y → y + x with no strict ordering predicate
distribution → factorization → distribution
a replacement that recreates its own match under a new wrapper
```

The greedy driver detects repeated fixed-point nodes and has a stack limit, so
obvious cycles raise rather than hanging forever. That is a safety net, not a
termination proof: a rule can keep creating novel, ever-larger graphs until a
limit or memory is exhausted.

Tags can mark a form as already processed because `tag` participates in UOp
identity and pattern matching. Use them only when the pass defines their
lifetime and removal. An unexplained tag that merely suppresses a second match
can hide a badly oriented rewrite and leak into later passes.

Ordering also affects correctness without a loop. If an early broad rule
replaces a node, later rules for that original form never run. Tests should
cover at least one overlap whenever order is intentional.

## Correctness hazards: prove the real domain

Algebra learned over real numbers is not automatically valid over tinygrad
UOps. Before replacing a graph, audit these dimensions:

### Dtype and numeric semantics

- Concrete integers have finite-width overflow behavior.
- Weak dtypes defer a width/type commitment and can change promotion.
- Boolean `ADD`/`MUL` are normalized toward boolean OR/AND behavior.
- Floating values include NaN, infinities, and signed zero; reassociation and
  textbook identities can change observable results.
- Casts and bitcasts have different meanings.

For example, blindly treating floating `x + 0.0` as `x` needs an explicit
decision about signed-zero equivalence. The current symbolic matcher contains
a broader `x+0` rule; this chapter's lab stays integer-only to teach guards. It
is a pedagogical isolated matcher, not a proposal to duplicate or replace the
upstream rule.

### Shape and lane semantics

An operand may broadcast to the result. Replacing a matrix result with its
scalar source is wrong even if the other source is zero. Compare the proposed
replacement's shape with the matched root, and test scalar, vector, empty, and
broadcast cases relevant to the pass. Vector dtypes add another lane dimension
that patterns can match through scalar dtype rules.

### Invalid values, masks, and bounds

`Invalid` values and validity gates carry out-of-bounds semantics. A rule that
folds `0 * invalid` too early can erase the poison used to remove or guard a
load/store. Symbolic bounds are proofs with assumptions, not ordinary sample
integers.

### Effects and graph stage

Pure arithmetic can often be reordered; `STORE`, `AFTER`, barriers, calls, and
buffer reuse carry ordering. A structurally valid UOp can still be illegal at
the current or next stage. Identify the pass input invariant and downstream
consumer before widening a matcher.

## Test a rewrite as a small compiler change

A serious rewrite test matrix contains:

| Test | What failure it catches |
| --- | --- |
| Positive minimal form | Rule never matches or returns the wrong replacement. |
| Commuted/alternate positive form | Intended source-order coverage is missing. |
| Negative near-match | Pattern or callback is broader than the proof. |
| Dtype cases | Weak, bool, float, signed, unsigned, or vector behavior leaks across domains. |
| Shape/broadcast cases | Replacement drops expansion, reduction, or lane structure. |
| Semantic/differential execution | Representation assertion passes while values differ. |
| Driver/order test | A different rule wins or traversal changes composition. |
| Termination/idempotence test | Rules bounce, grow, or keep changing a stable graph. |
| Spec/downstream test | Replacement is locally plausible but illegal for its consumer. |

Prefer identity assertions when canonical identity is the contract:

```python
self.assertIs(graph_rewrite(x + 0, pm), x)
```

Also execute representative values when the rule claims numeric equivalence.
Neither test replaces the other. An identity test is precise about the chosen
form; a differential test samples semantics.

For an upstream change, first add the focused regression beside the matcher or
its closest unit tests. Then run the broader subsystem suite, spec modes used by
that subsystem, and process replay or hardware tests in proportion to the
pass's reach. Chapter 16 builds the full test-selection method.

## Source tour

All links are pinned to commit
`874d33128b4e4785beea736d97df6716e0321717`.

| Read this | What to extract |
| --- | --- |
| [`UPat`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1316-L1421) | Constructor constraints, named identity, tuple/list/repeated sources, and operator sugar. |
| [`PatternMatcher`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1454-L1478) | Op indexing, concatenation, declaration order, and first non-`None` replacement. |
| [`RewriteContext` and `graph_rewrite`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1630-L1753) | Compare one-pass walk with greedy rewrite, rebuilds, cycle detection, direction, and call boundaries. |
| [Core symbolic rules](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py#L65-L180) | Study rule ordering around invalid propagation, identities, bool normalization, constant folding, and floating caveats. |
| [Pattern matcher tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_pattern_matcher.py#L6-L203) | Copy the test style for op/dtype/arg constraints, repeated names, permutations, and source length. |
| [Driver cycle and walk tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py#L294-L528) | Treat these assertions as the authoritative traversal semantics for this snapshot. |
| [`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md) | Use current commands to inspect named passes and individual matches. |

Useful searches:

```bash
rg -n 'PatternMatcher\(\[' tinygrad test
rg -n 'graph_rewrite\(' tinygrad test
rg -n 'name="[^\"]+"' tinygrad | rg graph_rewrite
```

Follow a real matcher to every place it is added to another matcher. The local
rule list alone may not reveal its actual priority.

## Lab: a guarded simplification

**Hardware:** Portable. Only one semantic check uses the Python backend.

Run `labs/phase2/rewrite_lab.py` from this guide's repository:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
CACHEDB=/tmp/tinygrad-guide-phase2.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/rewrite_lab.py"
```

Before running, predict all eight tests. In particular, predict why:

- both `x+0` and `0+x` match;
- `x+1` and floating `x+0.0` remain `ADD`;
- a scalar plus a zero matrix cannot be replaced by the scalar; and
- a `3→4` plus `4→3` matcher raises greedily but returns `4` with
  `walk=True`.

The tests should pass:

```text
Ran 8 tests ...
OK
```

Read the lab after it runs. Map each line of `remove_integer_add_zero` to at
least one positive or negative test. Then temporarily remove each guard, one at
a time, and add a test that demonstrates the expanded match domain. Restore
the guard before continuing.

### Inspect the matches with VIZ

The lab names the pass `lab integer add zero`. Capture rewrite data without
letting an interactive terminal replace the process with the server:

```bash
VIZ=1 CACHEDB=/tmp/tinygrad-guide-phase2-viz.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/rewrite_lab.py" \
  TestIntegerAddZero.test_positive_both_orders | tee /tmp/phase2-viz.log

.venv/bin/python -m tinygrad.viz.cli -s TINY | rg 'lab integer add zero'
```

Use the matching timeline name reported by the first CLI command. `DEBUG=6`
shows UOp graphs and `DEBUG=7` includes individual rewrite steps:

```bash
DEBUG=7 .venv/bin/python -m tinygrad.viz.cli -s TINY '<timeline name>'
```

VIZ is evidence for “which rule matched which graph.” It does not establish
semantic correctness. Pair it with the test matrix.

### Compare with upstream tests

Run the focused matcher/driver suites at the snapshot:

```bash
CACHEDB=/tmp/tinygrad-guide-phase2.db DEV=PYTHON DEBUG=0 .venv/bin/python -m pytest -q \
  test/null/test_pattern_matcher.py test/null/test_graph_rewrite.py
```

Now find the existing upstream `x+0` rule and its surrounding ordering. Explain
why the lab's matcher is useful for learning but would be a duplicate, not a
contribution.

### Exit exercise: design one non-duplicate change

On a throwaway branch, choose a real failing test, issue, or missing
canonicalization on current `master`. Before editing the matcher, write a
one-page rule contract:

1. exact input graph stage and invariant;
2. matched form and callback guards;
3. semantic equivalence argument for every dtype/shape in scope;
4. decreasing termination measure;
5. overlap and priority relative to neighboring rules;
6. positive and negative representation tests;
7. dtype and shape/broadcast tests;
8. semantic, spec, replay, or hardware evidence proportional to reach; and
9. expected VIZ match count/location on a reproducer.

Implement only after the contract predicts the tests. If the proposed rule is
already present, use history and current tests to find a different case rather
than adding another spelling.

## Checkpoint

Phase 2 is complete when you can demonstrate all of the following:

- Translate a UPat expression into exact op/dtype/source/arg/tag constraints.
- Explain why repeated names test UOp identity.
- Predict which of two overlapping rules wins.
- Choose greedy versus one-pass walking and state pre- versus post-order
  behavior precisely.
- Give a termination measure, not merely report that the test finished.
- Reject a tempting rewrite using a dtype, signed-zero/NaN, broadcast, invalid,
  or effect counterexample.
- Add a guarded simplification with positive, negative, dtype, shape, semantic,
  and termination/order evidence.
- Use VIZ to show the actual match and a focused test to show the contract.

The phase exit artifact is the rule contract, patch, test output, and one VIZ
trace for your practice change. A green value test alone is not enough.

## Quick reference

| Need | API / rule |
| --- | --- |
| Any bound node | `UPat.var("x")` |
| Bound constant | `UPat.cvar("c", arg=...)` |
| Ordered exact sources | `src=(p0, p1)` |
| Permuted exact sources | `src=[p0, p1]` |
| Any number of same-kind sources | `src=p` |
| Semantic guard fails | Return `None`. |
| One-node match | `pm.rewrite(uop, ctx=...)` |
| Greedy post-source composition | `graph_rewrite(root, pm)` |
| Greedy pre-order/fixed-point | `graph_rewrite(root, pm, bottom_up=True)` |
| One-pass post-order | `graph_rewrite(root, pm, walk=True)` |
| One-pass pre-order | `graph_rewrite(root, pm, bottom_up=True, walk=True)` |
| Make a pass discoverable in VIZ | Supply a stable `name=` to `graph_rewrite`. |
| Minimum correctness matrix | Positive, negative, dtype, shape/broadcast, semantic, order, termination, downstream legality. |
| First ordering fact | First successful rule in matcher declaration/concatenation order wins. |
| First termination question | What strict measure decreases on every greedy replacement? |
