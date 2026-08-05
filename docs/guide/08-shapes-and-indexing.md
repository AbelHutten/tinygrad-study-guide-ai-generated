# 8. Shapes, views, indexing, and symbolic values

## Purpose

Many compiler bugs that look like bad arithmetic are bad addresses. A reshape,
broadcast, slice, flip, or pad changes the relationship between a logical
output coordinate and the storage coordinate it reads. tinygrad must preserve
that relationship while combining views, fusing kernels, simplifying symbols,
and lowering to target index types.

This chapter gives you an address-first method for investigating shape and
indexing work. It prepares you to contribute to movement operations, symbolic
simplification, validity handling, rangeification, and the indexing side of
codegen.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

Before continuing, you should be able to:

- derive a row-major flat address from a multidimensional coordinate;
- state the broadcasting rule for an axis of size one;
- explain integer floor division and modulo; and
- distinguish a logical tensor view from a copied tensor.

If the first or last item is unclear, read NumPy's authoritative
[array memory-layout overview](https://numpy.org/doc/stable/dev/internals.html#internal-organization-of-numpy-arrays)
and create a few arrays whose shape, strides, and transpose differ. For
broadcasting, stop after the examples in the
[Python Array API broadcasting rules](https://data-apis.org/array-api/latest/API_specification/broadcasting.html).
For explicit loop/index notation, return to the
[TensorIR tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html).

## Mental model: a view is an index function

Ignore APIs for a moment. A tensor view can be described by:

```text
logical shape
output coordinate ──address function──▶ source coordinate / flat address
                  └─validity predicate▶ may this access occur?
```

For a contiguous row-major tensor with shape `(d0, d1, ..., dn)`, the element
address is:

```text
i0*(d1*d2*...*dn) + i1*(d2*...*dn) + ... + in
```

The multipliers are the familiar contiguous strides. After view operations,
“stride” is best understood as a coefficient in the resulting address
expression. It can be zero for a broadcast, negative before normalization for
a flip, or hidden inside division/modulo after a reshape.

In this snapshot, do not look for one persistent ShapeTracker object carrying a
final stride tuple. Tensor methods create movement UOps, and rangeification
re-expresses them as explicit `RANGE` and index algebra. Historical tinygrad
articles that center `ShapeTracker` can still teach the concept, but not the
current source path.

## Movement operations as coordinate maps

Suppose `o` is an output coordinate and `i` the source coordinate. The durable
meaning of the common movement operations is:

| Operation | Coordinate relationship |
| --- | --- |
| `RESHAPE` | Flatten `o` in the new shape, then recover source coordinates by repeated floor-divide/modulo using the old shape. Element order is preserved. |
| `PERMUTE` | Reorder axes; recover source coordinates with the inverse permutation. |
| `EXPAND` / broadcast | Added or expanded axes select source coordinate `0`; their effective source stride is zero. |
| `PAD` | `i[k] = o[k] - before[k]`, valid only inside the original extent; invalid positions produce the padding value. |
| `SHRINK` | `i[k] = o[k] + start[k]`. A Python slice with a non-unit step is normalized through a larger movement-op sequence. |
| `FLIP` | On a flipped axis, `i[k] = size[k] - 1 - o[k]`. |

These operations normally describe a zero-copy view. They cause memory traffic
only when a storage boundary such as `contiguous()` is required, or when a
consumer kernel eventually performs loads/stores through the derived address.
Padding is also representable without allocating a larger input: its
out-of-bounds region becomes a condition and a zero value.

### Snapshot-specific movement arguments

Public arguments are normalized before becoming movement UOps:

- public `pad((before, after))` stores `(before, new_total_size)` per axis;
- public `shrink((start, end))` stores `(start, length)` per axis; and
- high-level expansion squeezes the size-one axes being broadcast, injects
  dimensions with `EXPAND`, then permutes them into user-visible order.

When debugging, distinguish a Tensor method's user argument from the UOp's
normalized `marg`. Mixing them produces plausible but wrong formulas.

## From shapes to `RANGE` and `INDEX`

`run_rangeify` creates a `RANGE` for each iteration dimension that is not a
constant-one axis. It walks from consumers toward producers and records, for
each tensor UOp:

- the ranges indexing its input; and
- the ranges describing its output.

`apply_movement_op` transforms output ranges into source ranges according to
the table above. `_apply_reshape`, for example, computes one flat output index
and decomposes it into old-shape coordinates with `%` and `//`.

When a buffer-like source is reached, `UOp.index` builds:

```text
INDEX(buffer PARAM/BUFFER/SLICE, index expression...)
```

Later cleanup can combine nested indices, and codegen eventually inserts
`LOAD` or `STORE`. `RANGE` therefore answers “which logical iterations exist?”
while `INDEX` answers “where does this iteration access storage?”

For a scalar-address backend, multiple coordinates are ultimately flattened.
For an image or other shaped address space, more than one coordinate can
remain. Do not assume every `INDEX` has exactly one integer source.

## Validity masks are part of the address

An address expression alone cannot represent padding safely. tinygrad's
`UOp.valid(cond)` produces a `WHERE(cond, value, Invalid)` form. Two helpers
separate it again:

- `get_idx()` returns the value/address expression;
- `get_valid()` returns the condition; it is true for an ordinary ungated value
  and false for a bare `Invalid`.

For a padded input axis of length `sh` with `before = off`, rangeification
derives:

```text
source_index = output_range - off
valid = (output_range >= off) & (output_range < sh + off)
```

The padding rewrite turns invalid input positions into zero near the pad's
semantics. Other invalid forms propagate toward loads and stores, where later
rules gate or remove memory effects. This representation lets symbolic rules
simplify the address and condition together while preventing an out-of-bounds
load from being justified by arithmetic that ignored validity.

Whenever you change an index rewrite, test three regions:

1. the first invalid coordinate;
2. both valid boundaries; and
3. the first invalid coordinate after the valid region.

Off-by-one bugs often survive tests that sample only the middle.

## Symbolic values carry proofs, not guesses

Dynamic shapes and launch parameters are UOps too. `UOp.variable(name, min,
max, multiple_of=...)` creates an ALU-space `PARAM` with known bounds and an
optional divisibility fact. Expressions derive conservative `vmin` and `vmax`
from those facts.

Bounds support proofs such as:

- `0 <= col < width`, so `(row*width + col) // width == row`;
- the same condition gives `(row*width + col) % width == col`; and
- `n` is a multiple of 8, so `n % 8 == 0`.

`bind` associates a permitted concrete value with a symbolic parameter.
`linear_with_vars` later returns only bound values actually used by the
execution plan.

Bounds are not runtime checks by themselves. A rewrite is correct only if its
identity holds for every value allowed by the recorded constraints. If a
caller constructs incorrect bounds, a simplification can be locally justified
and globally wrong; tests should therefore cover constraint construction as
well as the rewrite.

## Why division and modulo deserve their own subsystem

Reshape and flattened indexing produce nested division/modulo expressions.
Without simplification, generated address arithmetic becomes expensive and
equivalent indices fail to match for fusion or coalescing.

`div_and_mod_symbolic` and `fold_divmod_general` use constants, bounds,
congruence, greatest common divisors, and `multiple_of` facts to perform
rewrites. Examples include extracting terms exactly divisible by a denominator,
removing redundant nested modulo, and reconstructing a quotient plus remainder.

The relevant UOps are `FLOORDIV` and `FLOORMOD`. Their behavior follows floor
division, including for negative operands. Do not import C/CUDA
truncating-division intuition into a symbolic proof. A candidate rule must also account
for a zero or variable-sign denominator and for the numerator's valid range.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Tensor movement APIs and argument normalization | [`MovementOps`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L103) |
| Local movement/index canonicalization | [`mop_cleanup`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/movement.py#L5) |
| Range creation and movement-to-index conversion | [`run_rangeify`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L180) |
| Exact movement coordinate definitions | [`apply_movement_op`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L161) |
| `INDEX`, `RANGE`, and validity helpers | [`UOp.index`, `UOp.range`, and `UOp.valid`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L574) |
| Symbol variables, binding, and divisibility | [`UOp.variable`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L983) |
| General symbolic rewrite sets | [`symbolic_simple` and `symbolic`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py#L99) |
| Division/modulo proof rules | [`fold_divmod_general`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/divandmod.py#L8) |
| Z3-backed symbolic fuzzing | [`fuzz_symbolic.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/fuzz_symbolic.py) and [`fuzz_symbolic_div.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/fuzz_symbolic_div.py) |
| Differential movement fuzzing | [`fuzz_shape_ops.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/fuzz_shape_ops.py) |

## Lab — Derive movement indices and prove simplifications

**Portable.** This lab calls the snapshot's coordinate definition directly so
the result is small enough to reason about. Predict every formula before
running it:

```bash
DEV=NULL DEBUG=0 CACHELEVEL=0 .venv/bin/python - <<'PY'
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, UOp

def text(x):
  return x.render() if isinstance(x, UOp) else repr(x)

def show(op, in_shape, arg, out_shape):
  out = tuple(UOp.range(size, axis, AxisType.WEAK)
              for axis, size in enumerate(out_shape))
  src = apply_movement_op(op, in_shape, arg, out)
  print(op.name, "source coordinates:", [x.render() for x in src])
  coord = tuple(size-1 for size in out_shape)
  replace = {rng: UOp.const(value) for rng, value in zip(out, coord)}
  print("  output", coord, "reads",
        tuple(text(x.substitute(replace).ssimplify()) for x in src))

show(Ops.PERMUTE, (2, 3), (1, 0), (3, 2))
show(Ops.RESHAPE, (2, 3), (3, 2), (3, 2))
show(Ops.FLIP, (2, 3), (False, True), (2, 3))
# Internal SHRINK arg is (start, length), not public (start, end).
show(Ops.SHRINK, (4, 5), ((1, 3), (2, 2)), (3, 2))

# Internal PAD arg is (before, new_total_size). This corresponds to
# public padding ((1, 1), (2, 1)) on an input with shape (2, 3).
out = tuple(UOp.range(size, axis, AxisType.WEAK)
            for axis, size in enumerate((4, 6)))
src = apply_movement_op(Ops.PAD, (2, 3), ((1, 4), (2, 6)), out)
for axis, idx in enumerate(src):
  print("PAD axis", axis, "idx=", idx.get_idx().render(),
        "valid=", idx.get_valid().render())
for coord in ((0, 0), (1, 2), (2, 4), (3, 5)):
  replace = {rng: UOp.const(value) for rng, value in zip(out, coord)}
  observed = [(text(x.get_idx().substitute(replace).ssimplify()),
               text(x.get_valid().substitute(replace).ssimplify())) for x in src]
  print("PAD", coord, "->", observed)

row = UOp.variable("row", 0, 7)
col = UOp.variable("col", 0, 15)
flat = row*16 + col
print("flat/bounds:", flat.render(), flat.vmin, flat.vmax)
print("recover row:", text((flat//16).simplify()))
print("recover col:", text((flat%16).simplify()))

n = UOp.variable("n", 8, 64, multiple_of=8)
quotient = (n//8).simplify()
bound = n.bind(24)
print("n % 8:", text((n%8).simplify()))
print("n // 8 bounds:", quotient.vmin, quotient.vmax)
print("binding:", bound.op, bound.src[1].val)
PY
```

Expected durable observations are:

- permutation inverts the requested axis order;
- reshape flattens with the output shape and decomposes with the input shape;
- flip reverses only its selected coordinate;
- shrink adds its starts;
- the padded interior starts at output `(1, 2)`, while the surrounding region
  has false validity; and
- bounds prove recovery of `row`/`col` and `n % 8 == 0`.

Rendered expression ordering is canonicalization detail; compare semantics, not
the exact placement of additions.

### Change and regress

Change the reshape, permutation, or pad widths and write the expected map first.
Then choose one proposed symbolic identity and exhaustively substitute all
values in small declared bounds. Only after that local oracle passes should you
express it as a rewrite.

Run the focused suites with the testing environment from setup:

```bash
.venv/bin/python -m pytest -q \
  test/unit/test_indexing.py \
  test/null/test_uop_symbolic.py \
  test/null/test_symbolic_failures.py \
  test/unit/test_symbolic_tensor.py
```

For a family of failures, use the
[Hypothesis workflow](../reference/learning-resources.md#testing-transformations)
or the external fuzzers. The Z3 fuzzers run hundreds or thousands of cases and
need the corresponding testing dependencies; keep the printed random seed so a
failure is reproducible. Begin a real fix with the minimized example, then use
fuzzing to establish breadth.

## Debugging method: keep three artifacts

For a wrong-value or out-of-bounds issue, save:

1. the public movement sequence and logical shapes;
2. the rangeified `INDEX` expression plus `get_valid()`; and
3. the final generated address/gate.

Find the first artifact that is wrong. If item 1 is wrong, inspect Tensor
argument normalization. If item 2 is wrong, inspect `apply_movement_op` and
symbolic rewrites. If item 2 is correct but item 3 is wrong, continue into
index lowering, devectorization, and the renderer rather than changing movement
semantics.

## Checkpoint

Continue when you can:

- derive contiguous strides and a flat address by hand;
- express reshape, permute, expand, pad, shrink, and flip as coordinate maps;
- explain how movement UOps become `RANGE` and `INDEX` expressions;
- separate an index from its validity condition;
- use bounds and divisibility to justify a div/mod simplification; and
- design an example regression plus a property/fuzz oracle for an indexing
  change.

## Quick reference

| Symptom | Inspect first |
| --- | --- |
| Wrong output shape | Tensor movement argument normalization and UOp `_shape` |
| Right shape, wrong element order | `apply_movement_op`; inverse permutation or reshape flatten/decompose |
| Broadcast reads wrong values | Expanded axes should map to source coordinate zero |
| Pad returns wrong edge value | `(output-before)` and both validity inequalities |
| Out-of-bounds load/store | `INDEX.get_valid()`, Invalid propagation, later gate lowering |
| Huge address expression | `symbolic`, then `div_and_mod_symbolic`; check whether bounds are strong enough |
| Simplification works only for one size | Variable bounds, `multiple_of`, denominator sign, and a fuzz/property test |
| Historical note names ShapeTracker APIs | Translate the index-function idea through current movement UOps and rangeification |
