# 6. Pattern matching and graph rewriting

## The promise of this chapter

Chapter 5 treated a UOp graph as a representation that can be inspected and
rebuilt without mutating its old nodes. This chapter explains how tinygrad
performs those rebuilds systematically. Its main tool is a **graph rewrite**:
recognize a particular UOp arrangement, decide whether replacing it is legal,
and construct the preferred replacement.

You do not need prior compiler terminology. We will define pattern, match,
rule, pass, canonical form, traversal order, fixed point, and termination from
ordinary expression trees before reading tinygrad's APIs. The carried rule is
deliberately small:

```text
integer x + 0  →  x
```

The algebra looks obvious. Making the rule defensible exposes nearly every
question a contributor must ask of a larger compiler rewrite: Which graph
forms match? Which numeric types are in scope? Does broadcasting change the
shape? Which rule wins? Are replacements visited again? Why does rewriting
stop? What evidence distinguishes “the pattern fired” from “the result is
correct”?

By the end, you will be able to:

- translate an ordinary expression rule into `UPat`, a callback, a
  `PatternMatcher`, and a `graph_rewrite` invocation;
- distinguish syntactic matching from semantic legality and canonical
  preference;
- predict tuple, list, repeated-source, and repeated-name pattern behavior;
- explain first-successful-rule priority;
- trace source-first and root-first drivers, with and without replacement
  re-entry;
- state a decreasing measure that argues a greedy rewrite terminates;
- reject tempting rules using dtype, floating-point, shape, effect, or graph-
  stage counterexamples;
- test representation, semantics, order, termination, and downstream legality
  separately; and
- capture one rewrite with tinygrad's VIZ tooling and say what that trace does
  and does not prove.

## Begin with an ordinary expression tree

Consider:

```text
(a + 0) * 1
```

Before using compiler words, name its parts:

- `a`, `0`, and `1` are **leaves**: the drawing has no smaller expressions
  beneath them;
- `+` and `*` are **operations**;
- the values supplied to an operation are its **operands**;
- the entire multiplication is the **root**, the expression whose value we
  asked for; and
- `a+0` is a **subtree**, a complete expression inside the larger one.

Two familiar algebraic rules are:

```text
x + 0  →  x
x * 1  →  x
```

Here `x` is a **pattern variable**. It does not mean a particular Python
variable called `x`; it means “bind whatever expression occupies this
position.” If the first rule sees `a+0`, it binds `x` to `a` and returns `a`.
The second rule can then turn `a*1` into `a`.

A **pattern** describes the structure to recognize. A **match** is one
successful assignment of concrete graph nodes to the pattern's variables. A
**replacement** is the graph returned for that match. A pattern together with
the code that chooses a replacement is a **rewrite rule**.

This is structural matching, not source-text substitution. These Python
spellings might construct equivalent graph shapes:

```python
a + 0
Tensor.__add__(a, 0)
helper_returning_a_plus_zero(a)
```

A graph rule sees operations, sources, dtypes, shapes, and payloads after the
frontend has run. It does not search the user's Python file for the characters
`+ 0`.

## Algebra is always relative to a domain

The arrow in `x+0 → x` silently claims that both sides are equivalent. That
claim needs a **domain**: a set of values and operations whose semantics we
have fixed.

Over mathematical integers, addition by zero is an identity. In a tensor IR,
more facts can be observable:

- the result and replacement may have different shapes because one operand
  broadcast;
- the result may have a promoted dtype;
- a weak literal has not committed to the same type rules as a concrete
  integer;
- floating-point signed zero can distinguish calculating `-0.0+0.0` from
  returning `-0.0`;
- a source may carry ordering or invalid-value semantics; and
- an operation legal in one graph stage may be illegal in the next.

Three questions must therefore stay separate:

| Question | Meaning |
| --- | --- |
| **Syntactic match** | Does the graph have the structure described by the pattern? |
| **Semantic precondition** | Under this pass's promised equivalence, is replacement legal for this particular match? |
| **Canonical preference** | If several legal representations exist, which one should this pass choose? |

A pattern can match while its callback declines replacement. Two replacements
can both preserve values but disagree about the preferred form. This is why a
rewrite is more than a clever one-line pattern.

## Move from a tree to a UOp DAG

Chapter 5 established four facts we need here:

1. a UOp stores ordered references to its dependencies in `src`;
2. a reachable graph is usually a DAG because one node may be shared;
3. UOps are treated as immutable, so changing an interior expression means
   rebuilding its consumers up to the root; and
4. constructing the same live representation often returns the same interned
   object.

Suppose the graph is:

```text
root = MUL(ADD(a, 0), 1)
```

Replacing `ADD(a,0)` with `a` does not assign into the old `MUL.src`. The
driver constructs the representation `MUL(a,1)`. If another rule replaces
that multiplication, the new root is simply `a`. The old root remains a valid
description of the old expression.

A **compiler pass** is a bounded transformation from one valid graph state to
another. A pass may simplify, canonicalize, lower an abstract operation,
introduce explicit indices, select target instructions, or enforce a
downstream invariant. tinygrad uses the same rewrite machinery for many of
these jobs; “rewrite” does not mean only algebraic cleanup.

## The four parts of a tinygrad rewrite

Keep four responsibilities distinct:

```text
UPat pattern
  → callback and semantic guard
  → ordered PatternMatcher
  → whole-graph driver
```

| Part | Question it answers |
| --- | --- |
| `UPat` | Which local UOp arrangements should be considered, and which nodes get names? |
| callback | Given one binding, is replacement legal, and which UOp should replace the root? |
| `PatternMatcher` | Which rules are candidates for this root op, and which successful rule has priority? |
| `graph_rewrite` | Which reachable nodes are visited, in what order, and whether produced replacements are visited again? |

Confusing these layers causes common debugging mistakes. A callback may be
perfect but never run because its pattern does not match. A matcher may work
on one node while a whole-graph driver visits it in an unexpected order. Two
sound rules can loop because their canonical directions disagree.

## Build the carried rule one decision at a time

tinygrad offers UOp-like operator sugar for patterns. The shortest teaching
spelling is:

```python
PatternMatcher([
  (UPat.var("x") + 0, lambda x: x),
])
```

Read it as “match an addition containing a node named `x` and a constant whose
argument compares equal to zero; replace the result with `x`.” It is useful
for learning the syntax, but it has not stated a defensible domain. It can
encounter floats, bools, weak types, vectors, and broadcast sources.

The checked-in lab uses a broad structural pattern and an explicit guard:

```python
def remove_integer_add_zero(add, x, zero):
  if add.dtype not in dtypes.ints: return None
  if zero.base.op is not Ops.CONST or zero.base.val != 0: return None
  if x.dtype != add.dtype or x.shape != add.shape: return None
  return x

integer_add_zero = PatternMatcher([
  (UPat(Ops.ADD, name="add",
        src=[UPat.var("x"), UPat.var("zero")]),
   remove_integer_add_zero),
])
```

Translate every line:

1. The root must be `Ops.ADD`; bind it as `add`.
2. The list-valued `src` asks the matcher to try both two-source
   permutations. Bind one source as `x` and the other as `zero`.
3. Restrict the output to tinygrad's set of concrete scalar integer dtypes.
4. Follow `zero.base` through movement operations and `DETACH`, then require a
   literal `CONST` whose value is zero.
5. Require returning `x` to preserve both the result dtype and result shape.
6. Return the already existing `x` UOp. Returning `None` from any failed guard
   means “this binding did not prove a legal replacement.”

`zero.base` does not evaluate arbitrary algebra. It does not strip casts,
addition, multiplication, or a parameter merely because its known bounds are
zero. It makes an expanded or reshaped literal zero acceptable without turning
the rule into a general theorem prover.

### State the rule contract before trusting it

The complete local contract is:

| Contract part | Carried rule |
| --- | --- |
| Input representation | A spec-valid, promotion-consistent `ADD` whose sources have valid derived shapes. |
| Syntactic domain | Either source order; one source may be a movement view or `DETACH` over literal zero. |
| Numeric domain | Concrete signed or unsigned integer dtypes in `dtypes.ints`; any tensor shape is allowed when preserved. |
| Replacement | The other source, only when it already has the root's dtype and shape. |
| Value argument | Integer addition by zero preserves every value and cannot itself overflow. |
| Representation argument | Explicit equality checks preserve the observable result dtype and shape. |
| Purity argument | The removed source is a constant plus view-like wrappers, not an effectful computation. |
| Termination measure | The number of reachable matching `ADD` nodes decreases by one. |
| Project scope | An isolated teaching matcher; upstream already has a broader identity rule. |

Notice how much of the contract is absent from the arrow `x+0 → x`. Writing
this table is good preparation for proposing a real rule: a reviewer can
challenge each boundary independently.

The input-invariant row matters. A compiler pass normally receives IR already
validated by its producer: an ADD's stored result dtype agrees with promotion
from its sources. With validation disabled, Python lets you manually construct
malformed UOps—for example, an integer-typed ADD over an integer and a floating
zero. The callback is not a validator for every arbitrary five-field object;
its proof is conditional on the pass's well-formed-input contract. `SPEC=2`
rejects that malformed example before this teaching rule is relevant.

## `UPat`: describe and bind representation

A `UPat` can constrain the same local facts you learned to inspect on a UOp:

```python
UPat(op=..., dtype=..., src=..., arg=..., tag=..., name=...)
```

- `op` restricts the operation kind or kinds.
- `dtype` restricts the result dtype or a tuple/set-like family accepted by
  the constructor.
- `src` recursively describes dependencies.
- `arg` restricts the operation-specific payload.
- `tag` restricts the tag.
- `name` binds the matched UOp for the callback.

Common helpers are:

```python
UPat(Ops.ADD, name="add")           # any ADD; sources unconstrained
UPat.var("x")                        # any UOp, bound as x
UPat.var("x", dtype=dtypes.int32)    # any int32 UOp
UPat.cvar("c", arg=0)                # a CONST with argument zero
UPat.const(0, dtypes.int32)           # exact int32 zero CONST pattern
UPat.any(pattern_a, pattern_b)        # either alternative
```

Pattern operator overloads construct patterns, not executable UOps:

```python
UPat.var("x") + 0
UPat.var("x") * UPat.var("y")
UPat.var("cond").where(UPat.var("a"), UPat.var("b"))
```

Read pattern code inside out, just as you read a UOp DAG. Then list every name
the callback expects. Misspelled or unintentionally repeated names change the
match contract.

### Source containers have different meanings

The spelling of `src` is semantic:

| Pattern form | Meaning |
| --- | --- |
| `src=None` | Do not constrain sources. |
| `src=(p0, p1)` | Match these source patterns in this order, with exact length by default. |
| `src=[p0, p1]` | Generate source permutations and try them; exact length by default. |
| `src=p` | Require every source to match the repeated pattern; source count may vary. |
| `allow_any_len=True` | Permit additional sources beyond the listed prefix for the supported form. |

For the carried list pattern, candidate `0+x` may invoke the callback twice.
One permutation can bind `x` to zero and `zero` to the parameter, causing the
guard to return `None`; the other binding succeeds. A callback must therefore
be pure: logging, counters, mutation, and other side effects can happen more
times than the number of replacements.

A list blindly requests permutations. It does not consult an algebra system
to prove the root operation commutative. Use list form only after checking the
operation and graph stage. `WHERE`, `INDEX`, `STORE`, division, and effect
dependencies are positional. Even for a normally commutative operation,
special invalid or floating semantics may constrain what a particular pass is
allowed to exchange.

### Repeated names demand one identity

If one name occurs twice:

```python
UPat(Ops.MUL, src=(UPat.var("x"), UPat.var("x")))
```

both positions must bind the same UOp object. The implementation checks
identity, not merely equal printed text. Because live UOps with the same five-
field representation are normally interned, separately reconstructing that
same representation often does produce the required identity. In contrast,
two different-key nodes that happen to render similarly do not satisfy the
repeated binding.

This distinguishes two edges to one shared node from two merely similar
subgraphs. It also means a pattern can depend on canonicalization that happened
before the current pass; state that invariant if correctness relies on it.

### Three easy constraint footguns

1. `arg=None` and `tag=None` mean “unconstrained,” not “match an explicit
   `None`.” Check an explicit-`None` requirement in the callback or use a more
   specific representation invariant.
2. `arg=0` uses Python equality. Since `False == 0`, a zero-argument pattern can
   also match a false constant unless dtype or callback guards distinguish it.
3. Dtype families are exact project sets, not English categories.
   `dtypes.ints` excludes `bool` and weak integers at the pinned snapshot. The
   carried callback deliberately inherits those exclusions. Vector-shaped
   tensor values still use scalar element dtypes here; their positions are
   represented by shape (and some IR containers such as `STACK`), not a
   separate vector-DType class.

Top-level patterns in a `PatternMatcher` must constrain a root `op`, allowing
candidate indexing. A completely unconstrained `UPat.var("x")` is useful as a
source pattern but not as an indexed top-level rule.

## `PatternMatcher`: ordered rules and first success

A `PatternMatcher` stores `(UPat, callback)` pairs in declaration order and
indexes them by candidate root op. For one node it conceptually does:

1. select rules registered for the node's `op`;
2. reject impossible source-op combinations cheaply;
3. recursively generate name bindings for the pattern;
4. call the rule for each binding;
5. on `None`, try the next binding of that pattern;
6. on the original UOp, stop that pattern's remaining bindings and continue
   with the next declared rule; and
7. on a different UOp, return it immediately.

The observable result can therefore depend on declaration order:

```text
binding returns None          → try another binding of the same pattern
binding returns original root → stop this pattern; try the next rule
binding returns replacement   → stop everything and return it
```

The lab verifies all three paths. One test puts a `None`-returning rule before
a rule that returns constant `99`; the second rule wins. Another makes the
first rule return `x`; the later `99` rule is never reached. A third uses a
list pattern whose first permutation returns the original ADD: the matcher
does not try the otherwise-valid second permutation of that same pattern, but
it does continue to the next declared rule, which returns `99`.

Adding matchers with `pm_a + pm_b` concatenates their rule lists from left to
right. It is not an unordered set union. When extending an upstream matcher,
inspect where each component is added; the local file does not necessarily
show the rule's final priority.

### Callbacks, context, and closures

Callbacks should be deterministic functions of their bindings and explicit
context. A callback may request shared driver state through a parameter named
`ctx`; pattern variables may not also use that name. This supports lookups or
configuration without hiding state in globals.

Matcher callbacks are compiled or reconstructed by tinygrad, so Python
closures are unsupported. Define non-trivial callbacks at module scope. Pass
changing shared information through `ctx`. Besides satisfying the machinery,
this makes a rule's evidence visible to tests and reviewers.

`PatternMatcher.rewrite(node)` asks only about that node. It does not traverse
its sources or rebuild its consumers. Use it for a tight pattern/callback unit
test; use `graph_rewrite` when the pass contract concerns a reachable DAG.

## `graph_rewrite`: traversal changes the transformation

A whole-graph driver has three independent questions:

1. Is matching attempted at just one node or throughout the reachable graph?
2. Does a node get its main match before its sources (**root-first/pre-order**)
   or after them (**source-first/post-order**)?
3. When a rule creates a replacement, is that replacement entered again
   greedily or accepted as the final result for this visit?

Use “root-first” and “source-first” in your reasoning. Some implementation
comments use top-down/bottom-up labels in a counterintuitive way, and the
public `bottom_up` flag does not map cleanly onto every reader's convention.

For a root `ADD(CONST(1), CONST(2))`, instrumented callbacks observe:

```text
source-first greedy:    [1, 2, 'ADD']
root-first greedy:      ['ADD', 1, 2]
source-first one-pass:  [1, 2, 'ADD']
root-first one-pass:    ['ADD', 1, 2]
```

The pinned invocation map is:

| Invocation | Conceptual order | Replacement behavior |
| --- | --- | --- |
| `graph_rewrite(root, pm)` | Sources are made ready; main `pm` matches the node/rebuilt node. | Greedy: produced nodes are re-entered until stable for the driver. |
| `graph_rewrite(root, pm, bottom_up=True)` | Main `pm` is used before descent at each node. | Greedy fixed point before sources; rebuilt nodes are reconsidered. |
| `graph_rewrite(root, pm, walk=True)` | Sources first, then one main match at the rebuilt node. | One pass: do not enter the produced replacement subtree. |
| `graph_rewrite(root, pm, bottom_up=True, walk=True)` | Try the node first; a success skips its original children. | One pass: do not enter the produced replacement subtree. |

The optional `bpm` matcher is the pre-descent matcher when
`bottom_up=False`; the main `pm` remains the source-ready matcher. When
`bottom_up=True`, the main `pm` itself occupies the pre-descent role. Do not
describe `bpm` as universally separate from the direction flag.

`enter_calls=False` is the default. It protects opaque function/call bodies
from ordinary traversal. Entering such a boundary is a pass-level semantic
choice, not merely a way to find more matches.

### Greedy re-entry versus one-pass replacement

Use two rules:

```text
CONST 3 → CONST 4
CONST 4 → CONST 5
```

Then:

```text
greedy graph_rewrite(CONST 3)  → CONST 5
walk=True                       → CONST 4
```

The greedy driver re-enters the `4`, so the second rule composes. The one-pass
driver accepts `4` without visiting that new subtree. Neither result is
intrinsically better; the pass contract chooses.

Use greedy rewriting when a family of canonicalization or lowering rules is
meant to compose toward a stable representation. Use one-pass walking for an
exact substitution or transformation whose produced form must not be
traversed in this pass. Do not select `walk=True` merely to hide a loop. That
changes the meaning instead of proving the original rewrite system sound.

## Fixed points and termination

A **fixed point** is a form for which another application of the chosen
transformation produces no further change. For the carried rule:

```text
(x + 0) + 0
```

the source-first greedy trace is:

1. Visit `x` and the shared zero constant. Neither has root op `ADD`.
2. Match the inner `ADD`; bind `x` and `zero`; all guards pass; return `x`.
3. Rebuild the outer node with that rewritten source, producing `x+0`.
4. Match the rebuilt `ADD` and return `x`.
5. Rewriting the result again returns the same interned `x`.

The lab calls the transform twice and verifies identity on the second result.
That is an **idempotence** check: `rewrite(rewrite(g)) is rewrite(g)` for this
case.

### Give every greedy family a decreasing measure

A termination argument identifies something that cannot decrease forever:

- reachable count of a targeted operation;
- number of nodes illegal for the next stage;
- abstraction rank, such as high-level operation to lower-level operations;
- distance from a strict canonical operand ordering; or
- a lexicographic combination of several bounded measures.

The carried rule removes one matching `ADD` and creates none, so its reachable
matching-`ADD` count strictly decreases.

Rules without a shared orientation can fail:

```text
A → B        together with B → A
x+y → y+x   without a strict ordering predicate
distribute   together with factor
replacement that wraps itself in another matching node
```

The pinned drivers have safety mechanisms, but those are not proofs:

- root-first (`bottom_up=True`) greedy matching keeps a local `seen` set and
  raises `RuntimeError: infinite loop in fixed_point_rewrite` on an exact
  repeated node;
- the default greedy driver eventually raises
  `RuntimeError: infinite loop in graph_rewrite (stack too big)` for the
  verified `3↔4` cycle; and
- a rule that continually creates novel, larger graphs may consume substantial
  time or memory before any limit notices.

Tags can distinguish a processed form because `tag` participates in UOp
identity and pattern constraints. Use them only when the pass defines the
tag's lifetime and downstream removal. A mystery tag that merely suppresses a
second match often hides a badly oriented rule.

## Semantic legality: test the arithmetic you actually have

Textbook algebra is normally stated over exact real numbers. tinygrad operates
on finite integers, IEEE-like floating values, booleans, weak literals,
vectors, shapes, invalid values, and effectful nodes. This short standard-
Python/tinygrad probe makes several failures visible:

```python
import math
from tinygrad import Tensor, dtypes

print("int8 127 + 1:",
      Tensor([127], device="PYTHON", dtype=dtypes.int8).add(1).item())

a, b, c = 1e20, -1e20, 3.0
print("float regroup left:", (a + b) + c)
print("float regroup right:", a + (b + c))

calculated = -0.0 + 0.0
print("sign after -0.0 + 0.0:", math.copysign(1.0, calculated))
print("sign if -0.0 is returned:", math.copysign(1.0, -0.0))
print("inf * 0.0 is NaN:", math.isnan(float("inf") * 0.0))
```

Stable results are:

```text
int8 127 + 1: -128
float regroup left: 3.0
float regroup right: 0.0
sign after -0.0 + 0.0: 1.0
sign if -0.0 is returned: -1.0
inf * 0.0 is NaN: True
```

Each line rejects an over-broad intuition:

- fixed-width integer arithmetic wraps or truncates according to its defined
  semantics; it is not an unbounded mathematical integer;
- floating addition is not associative because rounding happens after each
  operation;
- returning `x` for floating `x+0.0` can preserve a different signed zero from
  performing the addition;
- `x*0 → 0` fails for infinity and NaN; and
- similarly, `x/x → 1` needs exclusions for zero, infinity, and NaN.

The project may deliberately choose a looser equivalence in a particular
pass—for example, an optimization policy that does not preserve signed-zero
distinctions. The contributor's job is to discover, state, and test that
policy, not silently assume mathematical-real semantics. The pinned upstream
symbolic matcher already contains a broader `x+0` rule. Our integer-only rule
is a teaching vehicle, not a claim that upstream should narrow its policy.

### Dtype and lane checklist

Before widening a rule, ask:

- Is the dtype concrete or weak?
- Is it bool, signed integer, unsigned integer, or float?
- Does promotion change the root dtype?
- Is the value vector-shaped or assembled through `STACK`, and which shape
  positions must the replacement preserve?
- Are casts value conversions or bit reinterpretations?
- Do finite bounds or overflow affect the proposed identity?

Do not infer the exact content of `dtypes.ints` from its name. Inspect or test
membership. At this snapshot it excludes boolean and weak-integer dtypes,
which is why the lab includes explicit negative cases for both. Vector-shaped
integer tensors retain an ordinary integer element dtype and are covered by
the separate shape contract.

### Shapes and broadcasting are semantics

Suppose a scalar `x` is added to an expanded `(2,3)` zero tensor. The root has
shape `(2,3)`. Returning scalar `x` would change the result shape even though
all six mathematical values equal `x`.

The carried callback therefore requires:

```python
x.shape == add.shape
```

It accepts an expanded zero when the other source already has the full result
shape, but rejects removing an expansion that the result needs. Test scalar,
vector, empty, and broadcast shapes appropriate to the pass. A value-only
test can miss a representation error if later consumers happen to broadcast
again.

### Bounds are evidence, not literal syntax

A symbolic `PARAM` can have verified bounds `[0,0]`. Its value is always zero
under those assumptions, but it is not an `Ops.CONST` literal. The carried
rule deliberately rejects it because its syntactic contract requires a
literal constant (possibly through movement). A different rule could use
bounds as proof, but it would need to state how those bounds were established
and remain valid.

This distinction matters when debugging a missed match: “known to equal zero”
and “represented by a zero constant” are different predicates.

### Invalid values, effects, and graph stage

`Invalid` values and validity gates carry out-of-bounds and masking semantics.
Folding `0*invalid` too early can erase poison used to guard or remove a load
or store. Read nearby rule ordering before moving ordinary identities ahead of
invalid propagation.

Pure arithmetic can often be reordered. `STORE`, `AFTER`, barriers, calls,
buffer reuse, and other effect dependencies carry execution order. Never drop
an effectful source merely because its numeric result looks neutral.

Finally, legality is stage-specific. A replacement can be a well-formed UOp
yet violate the next pass's expected operation set, shape form, or effect
ordering. Identify:

```text
producer invariant → this rewrite's promise → downstream consumer invariant
```

before calling a local graph aesthetically “simpler.”

## Test a rewrite as a compiler change

One green example proves very little. Separate the obligations:

| Test category | What it catches |
| --- | --- |
| Minimal positive | Pattern never matches, callback binds the wrong node, or replacement is wrong. |
| Alternate order/form | Intended permutation or movement-base support is absent. |
| Near-match negative | Broad pattern escapes its proof. |
| Dtype matrix | Bool, weak, float, signed, or unsigned behavior crosses the boundary. |
| Shape matrix | Replacement drops required broadcast or view structure. |
| Representation assertion | Values happen to agree while the chosen canonical UOp is wrong. |
| Semantic execution | Structural result looks right but computes different values. |
| Priority/traversal | Another rule wins, or replacement re-entry changes composition. |
| Termination/idempotence | Rules bounce, grow, or continue changing a stable result. |
| Spec/downstream | Local replacement violates a later graph-stage contract. |

Use identity when canonical representation is the claim:

```python
self.assertIs(rewrite(x + 0), x)
```

Also check boundary values when numeric equivalence is the claim. The lab
constructs the candidate ADD directly so the identity assertion definitely
exercises this matcher, then realizes the returned `x` and compares
`int32.min`, `int32.max`, negative, zero, and positive values with a hard-coded
oracle on the portable Python backend. It does not independently execute an
unreduced ADD: downstream symbolic processing would apply upstream's broader
`x+0` simplifier. For a less elementary real rule, add an independent
reference implementation or a mode that genuinely executes both sides.

Neither evidence substitutes for the other. A returned `x` identity test says
exactly which representation won; a value test samples behavior; a spec test
asks whether downstream code accepts the form.

## Runnable lab: guarded matching and driver behavior

**Hardware:** portable. One test realizes integer values on the Python backend.

Run from the pinned `tinygrad-study` checkout, not from your home directory and
not from the guide repository:

```bash
cd /absolute/path/to/tinygrad-study
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs

CACHEDB=/tmp/tinygrad-guide-phase2.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/rewrite_lab.py"
```

The stable summary is:

```text
Ran 15 tests

OK
```

The test names divide the evidence:

- both operand orders and several concrete integer dtypes are positive;
- an expanded zero is positive only when returning `x` preserves the result
  shape, and a detached literal zero confirms the documented base traversal;
- nonzero, float, bool, weak integer, broadcast collapse, and bounded-symbolic
  zero are negative;
- realized values include the `int32` endpoints;
- nested additions reach an idempotent fixed point;
- returning `None` permits another binding; returning the original root skips
  the rest of that pattern but permits the next rule; and a real replacement
  stops the matcher;
- greedy rewriting follows `3→4→5`, while `walk=True` stops at `4`; and
- a `3↔4` cycle raises in greedy mode but a one-pass walk applies once.

Read one positive and one negative test before modifying the rule. The pair
usually explains its boundary more reliably than the pattern alone.

### Compare with focused upstream behavior

After Chapter 2's testing dependencies are installed:

```bash
CACHEDB=/tmp/tinygrad-guide-phase2-upstream.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python -m pytest -q \
  test/null/test_pattern_matcher.py test/null/test_graph_rewrite.py
```

At the pinned commit the stable structure is:

```text
64 passed, 9 skipped
```

Do not add `UPAT_COMPILE=0` as a required second mode for this snapshot. The
default compiled matcher path passes these focused tests. Disabling it exposes
unrelated callback-reconstruction failures in the pinned upstream suite, so it
would test a different unstable configuration rather than strengthen this
lab's claim.

## Inspect a successful replacement with VIZ

Normal tests answer pass/fail questions. VIZ records named rewrite events and
lets you inspect a structural before/after diff. The lab's wrapper is
decorated so the named `graph_rewrite` call is associated with a tracked event:

```python
@track_rewrites()
def rewrite(root):
  return graph_rewrite(root, integer_add_zero,
                       name="lab integer add zero")
```

From the `tinygrad-study` checkout, capture the smallest positive test:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs

VIZ=1 CACHEDB=/tmp/tinygrad-guide-phase2-viz.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase2/rewrite_lab.py" \
  TestIntegerAddZero.test_positive_both_orders 2>&1 | \
  tee /tmp/tinygrad-guide-phase2-viz.log
```

Discover the event name rather than assuming its numeric suffix:

```bash
DEBUG=0 .venv/bin/python -m tinygrad.viz.cli -s TINY | rg 'rewrite n'
```

Use the discovered event—for example `rewrite n1`—to list its named passes:

```bash
DEBUG=0 .venv/bin/python -m tinygrad.viz.cli \
  -s TINY 'rewrite n1' --ls
```

The listing should include:

```text
lab integer add zero - 1
```

Then print a detailed successful match:

```bash
DEBUG=7 .venv/bin/python -m tinygrad.viz.cli \
  -s TINY 'rewrite n1' 'lab integer add zero'
```

Look for a diff that removes an `ADD` and its zero source while retaining the
other source. Event numbers, object addresses, timings, and local node labels
vary. The stable evidence is the pass name, a successful match, and the
structural removal.

VIZ proves that one positive graph matched and changed in the recorded way. It
does not prove negative guards, dtype coverage, numeric equivalence,
termination, or downstream legality. Keep the unit and semantic tests.

## Guided source tour: answer one question per stop

These links target commit
`874d33128b4e4785beea736d97df6716e0321717`. Read them after running the lab.
Each range has one job; ignore surrounding machinery until a later question
requires it.

### Stop 1: why does list order differ from tuple order?

Read [`UPat` source-container handling at lines 1332–1341](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1332-L1341).

Question: which branch turns a list into permutations, which preserves a tuple,
and which repeats one pattern?

Translation: the container selected by the rule author becomes explicit match
behavior. The matcher does not independently prove commutativity. Ignore
`early_reject` construction on the first pass.

### Stop 2: how does literal pattern sugar work?

Read [`var`, `cvar`, and `const` lines 1362–1369](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1362-L1369), then [operator sugar lines 1393–1402](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1393-L1402).

Question: when `UPat.var("x") + 0` is evaluated, how does integer `0` become a
constant pattern, and why are addition sources permuted?

Translation: `ufix` wraps a literal with `cvar`, and the ALU helper uses a list
for operations in the commutative group. This is pattern construction, not
UOp execution.

### Stop 3: where is repeated-name identity enforced?

Read [`UPat.match` lines 1404–1421](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1404-L1421).

Question: which condition rejects a second binding of the same name?

Translation: line 1407 uses `is not`. The surrounding checks reject op, dtype,
arg, tag, and source-length mismatches. Follow only the branches exercised by
the carried pattern.

### Stop 4: what exactly does “first successful rule” mean?

Read [`PatternMatcher` lines 1454–1478](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1454-L1478).

Question: if one callback returns `None`, does the next candidate run? What if
the callback returns the original UOp?

Translation: candidates remain in declaration order; matcher addition
concatenates lists. `None` tries the pattern's next binding. Returning the
original UOp breaks the remaining bindings for that pattern and advances to
the next declared rule. A different UOp returns immediately and ends search.

### Stop 5: where does one-pass walking avoid replacement re-entry?

Read [the walk driver lines 1651–1675](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1651-L1675).

Question: after a successful pre-descent or post-source match, where is the
replacement stored without pushing its subtree?

Translation: `walk=True` records the produced node directly. Relate that to
the lab's `3→4` result.

### Stop 6: where does greedy matching re-enter?

Read [pre-descent fixed-point handling lines 1681–1701](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1681-L1701), then [source rebuild and re-entry lines 1721–1748](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1721-L1748).

Question: how is a produced or rebuilt node placed back into the driver's work,
and which direction has the exact-repeat `seen` set?

Translation: greedy composition is explicit worklist behavior. The local
cycle detector belongs to the pre-descent fixed-point branch; the default
direction relies on the broader rewrite stack limit for the verified cycle.

### Stop 7: why can rule priority preserve invalid semantics?

Read [invalid-value rules lines 65–77](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py#L65-L77).

Question: why are poison/invalid transformations placed before ordinary
algebraic identities?

Translation: simplifying numeric-looking sources too early can erase the
evidence required to preserve masking and out-of-bounds behavior.

### Stop 8: is this lab rule novel upstream?

Read [existing identity rules lines 99–107](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py#L99-L107).

Question: does upstream already handle `x+0`, and is its domain identical to
the lab's?

Translation: upstream contains a broader rule, including floating forms. The
lab isolates matcher mechanics and conservative proof structure; it is not a
candidate duplicate contribution.

Optional executable confirmations are [permutation and repeated-source tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_pattern_matcher.py#L160-L180), [cycle tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py#L307-L323), [replacement re-entry tests](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py#L364-L387), [source-first order](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py#L417-L427), and [root-first order](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_graph_rewrite.py#L465-L476). Treat their assertions as the snapshot contract; do not read the whole test files yet.

## Controlled exercises with worked answers

Attempt these on paper before opening the answers.

1. For the tuple pattern `(x,0)`, predict matches for `x+0` and `0+x`. Repeat
   for list pattern `[x,0]`.
2. For the carried list pattern and candidate `0+x`, describe two possible
   callback bindings and why one can return `None` before the other succeeds.
3. Predict what happens when one binding returns `None`, the original node, or
   a replacement. Distinguish another binding from another rule.
4. Predict `CONST 3` under rules `3→4` and `4→5` for greedy and `walk=True`.
5. Add `5→3`. State why the greedy family has no decreasing measure.
6. Explain why `(a+b)+c → a+(b+c)` is not generally valid for floats using
   `1e20`, `-1e20`, and `3`.
7. Decide whether a scalar `x` may replace `x + EXPAND(0,(2,3))`.
8. State the decreasing measure for the guarded integer `x+0` rule.
9. Explain why a `PARAM` with bounds `[0,0]` fails the literal-zero syntax
   even though its known value is zero.
10. Explain why one green VIZ diff and one green value test answer different
    questions.

??? success "Worked answers"

    1. The tuple matches only `x+0` in that order. The list tries permutations
       and can match both. A list requests this behavior; it does not prove the
       operation safe to commute.
    2. One permutation may bind pattern `x` to literal zero and pattern `zero`
       to the real input. The base-constant guard fails. The other binds them
       as intended and returns the input. This is why callbacks should not have
       side effects.
    3. `None` tries another binding of the same pattern. The original node
       stops that pattern's remaining bindings but allows the next declared
       rule. A different UOp stops all ordered search and becomes the
       replacement.
    4. Greedy re-enters `4` and reaches `5`. `walk=True` accepts the first
       replacement and stops at `4` for that visit.
    5. `3→4→5→3` cycles. No ordered quantity strictly decreases around the
       loop; greedy rewriting cannot reach a fixed point.
    6. `(1e20 + -1e20) + 3` is `3`, while
       `1e20 + (-1e20 + 3)` rounds to `0`. Real-number associativity does not
       survive finite floating rounding.
    7. No. The root shape is `(2,3)` while scalar `x` has shape `()`. Returning
       it drops required broadcast representation. If `x` already has
       `(2,3)`, the carried guard can accept an expanded zero.
    8. The reachable number of ADD nodes satisfying the rule's domain drops by
       exactly one and the replacement creates no such node.
    9. Bounds are semantic evidence attached to `PARAM`; they do not change its
       root op to `CONST`. A bounds-aware rule would be a different contract.
    10. VIZ shows a successful structural match and replacement. A value test
        samples numeric behavior. Neither proves negative guards, all values,
        termination, or downstream legality.

## Checkpoint: produce a rule contract, not a clever arrow

Save these artifacts:

```text
hand trace for (x+0)+0:
pattern fields and callback bindings:
positive and negative domain table:
dtype and shape invariants:
decreasing termination measure:
greedy versus walk result for 3→4→5:
focused test summary:
one VIZ structural diff:
downstream/spec suite you would run for a real change:
```

You pass when you can answer:

1. What is the difference between a structural match and a proved legal
   replacement?
2. Why can a list-valued source pattern invoke a callback more than once?
3. Why do repeated names test UOp identity?
4. What do `None`, the original UOp, and a new UOp mean to
   `PatternMatcher`, and which level—binding or rule—continues?
5. When does a produced replacement get visited again?
6. What fixed point does nested integer addition-by-zero reach?
7. Which quantity proves the carried rule terminates?
8. Give one dtype, one floating, one shape, and one effect counterexample to an
   over-broad algebraic rule.
9. Why is a bounded-zero parameter not a literal-zero match?
10. Which evidence would you add before changing a matcher used by lowering or
    rendering?

## Quick reference

| Term/API | Meaning here |
| --- | --- |
| pattern | Structural description of candidate UOps. |
| binding | Mapping from a pattern name to one matched UOp. |
| semantic guard | Callback checks that prove replacement is legal in the promised domain. |
| canonical form | Preferred representation among forms considered equivalent by the pass. |
| `UPat` | Pattern description over op, dtype, sources, arg, tag, and names. |
| `PatternMatcher` | Ordered rule bundle; first real replacement wins. |
| `PatternMatcher.rewrite` | Try rules on one node only. |
| `graph_rewrite` | Traverse/rebuild a reachable graph under selected driver semantics. |
| source-first/post-order | Process dependencies before the consumer's main match. |
| root-first/pre-order | Try the consumer before descending into dependencies. |
| greedy | Re-enter produced/rebuilt nodes so rules can compose to a fixed point. |
| `walk=True` | One-pass walk; do not traverse a successful replacement subtree. |
| fixed point | Another application produces no further change. |
| decreasing measure | Bounded quantity that strictly moves toward termination. |
| VIZ rewrite trace | Evidence of recorded successful structural replacements, not a correctness proof. |

## Optional reinforcement—not a prerequisite gate

- Read the introduction, application-recursion, and walk-versus-greedy parts
  of [MLIR's Pattern Rewriting documentation](https://mlir.llvm.org/docs/PatternRewriter/).
  Translate its terms back to `UPat`, callback, matcher, and driver. Ignore
  MLIR-specific mutation APIs and pattern-benefit machinery.
- Read the rounding-error and special-quantity sections through signed zero in
  [Goldberg's floating-point tutorial](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html),
  then rerun the counterexamples above. This deepens the proof audit; it is not
  required before beginning the chapter.
- Search one real matcher with:

  ```bash
  rg -n 'PatternMatcher\(\[' tinygrad test
  rg -n 'graph_rewrite\(' tinygrad test
  ```

  Follow it to every matcher composition and call site. Record its actual
  priority, graph stage, driver flags, and closest behavioral tests.

## What is deliberately left for later

- Chapter 7 follows graph construction into scheduling and realization, where
  rewrite-selected forms begin to determine executable work.
- Chapter 8 develops shapes, views, broadcasting, symbolic expressions, and
  index validity in enough detail to prove movement/index rewrites.
- Chapters 9–11 show optimization, lowering, and rendering matchers with
  stricter stage and target contracts.
- Chapter 15 combines rewrite traces with schedules, generated source, and
  runtime evidence during debugging.
- Chapter 16 builds the full test-selection method for an upstream compiler
  change.
- Chapter 18 returns to choosing a genuinely novel improvement after the whole
  pipeline is visible. At this point your artifact is a complete local rule
  contract and verified teaching matcher—not a premature upstream proposal.

[← UOp graphs](05-uops.md) · [Next: Scheduling →](07-scheduling.md)
