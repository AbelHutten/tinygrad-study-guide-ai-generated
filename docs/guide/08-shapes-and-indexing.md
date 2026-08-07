# 8. Shapes, views, indexing, and symbolic values

## The promise of this chapter

A tensor program says things like “reshape this,” “take every other column,” or
“broadcast this row.” A kernel eventually needs integers: loop counters and
memory addresses. This chapter builds the bridge between those two descriptions
without assuming that you already know array memory layout or compiler indexing.

The central question is:

> For one coordinate in the result, which coordinate in the source supplies its
> value, and is that source coordinate valid?

We will carry one six-element tensor all the way from visible Python values to
tinygrad's `RANGE` and `INDEX` algebra:

```python
base = Tensor([[0, 1, 2],
               [3, 4, 5]])
view = base.reshape(3, 2).permute(1, 0)
```

The result is:

```text
[[0, 2, 4],
 [1, 3, 5]]
```

That small example is enough to expose coordinate systems, flattening,
strides, view composition, rangeification, division and modulo, and symbolic
proofs. We will also contrast it with broadcasting, slicing, padding, and
advanced indexing.

By the end, you will be able to:

- use shape, rank, axis, extent, coordinate, element count, stride, element
  index, and byte offset precisely;
- derive a row-major address rather than memorizing a formula;
- predict the values produced by reshape, permute, expand, shrink, flip, pad,
  and stepped slicing;
- describe a view as a composition of coordinate maps, with validity as part
  of the map;
- distinguish Python `tensor[...]` syntax from the IR operation `Ops.INDEX`;
- explain both implicit broadcasting and an explicit `EXPAND` UOp;
- follow movement UOps into `RANGE` and `INDEX` expressions;
- use bounds and divisibility facts to justify, or reject, a symbolic
  simplification;
- inspect the right source in small, question-led stops; and
- design focused regressions for a shape or indexing change.

The source references and checked output use tinygrad snapshot `874d331`
(2026-08-05). tinygrad changes quickly, so treat exact graph spellings as
snapshot facts and the coordinate reasoning as the durable skill.

## First vocabulary: shape describes coordinates

A **tensor** is a collection of values addressed by zero or more integer
coordinates. Its **shape** says how many coordinate choices exist along each
direction.

For our tensor:

```text
base = [[0, 1, 2],
        [3, 4, 5]]
shape = (2, 3)
```

The shape has two entries, so the tensor has two **axes** and **rank** 2. Axis
0 has **extent** 2; axis 1 has extent 3. “Dimension” is often used for either
an axis or its extent, which is why explicit words help when debugging.

A **coordinate** selects one element by giving one integer per axis. Coordinates
start at zero:

| Coordinate | Value | Spoken as |
| --- | ---: | --- |
| `(0, 0)` | 0 | row 0, column 0 |
| `(0, 2)` | 2 | row 0, column 2 |
| `(1, 0)` | 3 | row 1, column 0 |
| `(1, 2)` | 5 | row 1, column 2 |

For shape `(2, 3)`, valid coordinates satisfy:

```text
0 <= row < 2
0 <= column < 3
```

The upper bounds are excluded. There are `2 * 3 = 6` possible coordinates.
That product is the **element count**, called `numel` in tinygrad. Reshape may
change the shape but must preserve this product.

Rank and element count are different. Shape `(6,)` has rank 1 and six
elements. Shape `(2, 3)` has rank 2 and six elements. Shape `(1, 2, 3)` has
rank 3 and still six elements.

Two edge cases are worth naming early:

- A scalar has shape `()`, rank 0, and one value. Its sole logical coordinate
  is the empty tuple `()`.
- A one-element vector has shape `(1,)`, rank 1, and coordinate `(0,)`.

They contain the same number of values but have different shapes and broadcast
differently. The comma in `(1,)` is Python's spelling of a one-item tuple; it
is not decoration.

An axis of extent zero is also possible. It contributes no coordinates, so the
tensor has zero elements. Empty shapes matter in slicing and in proofs: a rule
that assumes a loop executes at least once can fail on an empty axis.

## Logical coordinates and flat storage

The nested-list display is a logical presentation. Ordinary memory is more like
one long row of locations. A contiguous row-major representation of `base` is:

```text
flat element index:  0  1  2  3  4  5
stored value:        0  1  2  3  4  5
```

**Row-major** means that the last coordinate changes fastest. Walk across all
columns of one row, then begin the next row.

To derive a flat element index for coordinate `(row, column)`, ask how many
elements come before it:

1. Every complete preceding row contains 3 elements, contributing `row * 3`.
2. Within the chosen row, `column` elements precede the chosen element.
3. Add the contributions.

Therefore:

```text
flat_index(row, column) = row * 3 + column
```

For `(1, 2)`, this is `1*3+2 = 5`, and flat position 5 contains value 5.

The multipliers are **contiguous strides**, measured here in elements. Shape
`(2, 3)` has strides `(3, 1)`: moving one step along axis 0 skips three
elements, while moving one step along axis 1 skips one.

For a contiguous shape `(d0, d1, ..., dn)`, work from the right:

```text
last stride       = 1
previous stride   = last extent * last stride
next previous     = next extent * previous stride
...
flat index        = i0*stride0 + i1*stride1 + ... + in*striden
```

For shape `(2, 3, 4)`, the strides are `(12, 4, 1)` because a complete axis-1
slice contains `3*4 = 12` elements and a complete last-axis row contains 4.
Coordinate `(1, 2, 3)` maps to `1*12 + 2*4 + 3 = 23`.

Do not silently confuse three quantities:

| Quantity | Example | Unit |
| --- | --- | --- |
| logical coordinate | `(1, 2)` | one index per axis |
| flat element index | `5` | elements |
| byte offset | `20` for a 4-byte dtype | bytes |

The indexing algebra in this chapter mostly uses element indices. A later
lowering can multiply by item size or use a target-specific address type.

## The carried example, first as visible values

Start with the logical row-major sequence of `base`:

```text
0, 1, 2, 3, 4, 5
```

`reshape(3, 2)` asks for three rows of two values while preserving that logical
sequence:

```text
[[0, 1],
 [2, 3],
 [4, 5]]                  shape (3, 2)
```

`permute(1, 0)` requests old axis 1 first and old axis 0 second. For a rank-2
tensor that is a transpose: columns become rows.

```text
[[0, 2, 4],
 [1, 3, 5]]               shape (2, 3)
```

Notice what did *not* happen. The values were not sorted. The operation changed
which logical coordinates refer to which source values.

### Derive the map backward

Let `(r0, r1)` be a coordinate in the final `(2, 3)` result. We work backward
because the question is “what source does this output read?”

The inverse of `permute(1, 0)` swaps the coordinates:

```text
final coordinate (r0, r1)
  -> coordinate in reshaped (3, 2) view: (r1, r0)
```

Now flatten `(r1, r0)` in shape `(3, 2)`. Its contiguous strides are `(2, 1)`:

```text
logical flat position = r1 * 2 + r0
```

To recover a coordinate in the original shape `(2, 3)`, divide that position
into groups of three:

```text
base row    = (r1*2 + r0) // 3
base column = (r1*2 + r0) % 3
```

Here `//` is integer floor division and `%` is remainder. For nonnegative
positions, quotient counts complete groups of three and remainder selects the
position inside the group.

The complete coordinate map is therefore:

```text
(r0, r1)
  -> (((r1*2+r0)//3), ((r1*2+r0)%3))
```

Check four corners instead of trusting the symbols:

| Output coordinate | `r1*2+r0` | Base coordinate | Value |
| --- | ---: | --- | ---: |
| `(0, 0)` | 0 | `(0, 0)` | 0 |
| `(0, 2)` | 4 | `(1, 1)` | 4 |
| `(1, 0)` | 1 | `(0, 1)` | 1 |
| `(1, 2)` | 5 | `(1, 2)` | 5 |

Flatten the recovered base coordinate again:

```text
((p // 3) * 3) + (p % 3) = p, where p = r1*2+r0
```

So the eventual source flat index simplifies to:

```text
r1*2 + r0
```

This round trip is why reshape produces so much division/modulo algebra and why
symbolic simplification is part of correct, practical indexing.

## A view is an index function

The carried example suggests a more durable definition than “a tensor with a
stride tuple.” A logical view can be modeled as:

```text
logical output shape
output coordinate --coordinate map--> source coordinate or address
                  \--validity map----> is that source access allowed?
```

Most ordinary views have validity `True` everywhere. Padding needs a conditional
validity map because some output coordinates have no corresponding source
element.

Movement operations compose like functions. If operation B maps its output
coordinate to an input coordinate and operation A maps that coordinate to an
earlier source, the combined map is `A(B(output))`. That is exactly what we did
by applying permute backward and then reshape backward.

People often call reshape, transpose, and slicing **zero-copy views**. The safe
snapshot-specific statement is narrower:

> Constructing these movement UOps does not itself copy the source values.

Realization can still materialize an output. Calling `contiguous()` explicitly
requests a layout boundary. Scheduling may also introduce storage when two
consumers cannot share compatible ranges. A final kernel still performs loads
and stores. “View” describes the logical relationship, not a promise that no
future memory traffic will occur.

Older tinygrad material often teaches these ideas through `ShapeTracker`. That
historical model is useful for intuition about strides and masks, but this
snapshot does not carry one persistent ShapeTracker through the current path.
Tensor operations construct movement UOps; rangeification later turns their
semantics into explicit ranges and index expressions. Search the current code
for those operations instead of assuming an older class is merely hidden.

## Six movement operations, one coordinate question

For every operation below, let `o` mean an output coordinate and `i` the source
coordinate it reads. The fastest debugging method is to draw a tiny source,
write the visible result, and then derive `i(o)`.

### `RESHAPE`: preserve the current logical order

Reshape requires equal element counts. To map backward:

1. flatten the output coordinate using the new shape;
2. decompose that flat position using the old shape.

For old shape `(2, 3)` and new shape `(3, 2)`, output `(a, b)` has flat position
`2*a+b`, then reads old coordinate:

```text
((2*a+b)//3, (2*a+b)%3)
```

The phrase “preserve element order” means the logical row-major order of the
*current view*. It does not mean “visit increasing physical addresses.” This
matters after a permutation. If a transposed view displays
`[[0, 3], [1, 4], [2, 5]]`, reshaping that view follows the displayed logical
sequence `0,3,1,4,2,5`. Do not jump back to the original buffer's address order
when predicting semantics.

A reshape may contain `-1` in the public API to infer one extent, but only one
such extent is allowed and the inferred result must preserve element count.
The movement UOp receives a resolved shape, not the unresolved user spelling.

### `PERMUTE`: rename and reorder axes

`permute(1, 0)` on shape `(2, 3)` produces shape `(3, 2)`. Output `(a, b)` reads
source `(b, a)`. For more axes, the public permutation states which old axis
occupies each new position; the backward coordinate map uses the inverse
permutation.

For shape `(2, 3, 4)`, `permute(2, 0, 1)` yields `(4, 2, 3)`. Output coordinate
`(a, b, c)` reads old coordinate `(b, c, a)`.

The permutation must contain every axis exactly once. Duplicating an axis would
not be a permutation; dropping one would change element count.

### `EXPAND`: repeat by selecting coordinate zero

Suppose a source has shape `(2, 1)` and is displayed as:

```text
[[10],
 [20]]
```

Expanding it to `(2, 3)` displays:

```text
[[10, 10, 10],
 [20, 20, 20]]
```

Output `(r0, r1)` always reads source `(r0, 0)`. The second output coordinate
does not affect the source address. In stride language, that source axis has an
effective stride of zero.

In this snapshot, a low-level `EXPAND` adds leading dimensions. The high-level
`expand` helper can make an arbitrary size-one axis expand by reshaping away the
axis, adding dimensions, and permuting them into the requested order. A graph
may therefore show more movement nodes than the single word “expand” suggests.
Treat the composed coordinate map as authoritative.

### `SHRINK`: add the starting offset

For a simple half-open slice `source[start:end]`, output coordinate `o` reads
`o+start`. On two axes, shrinking with starts `(s0, s1)` maps:

```text
(r0, r1) -> (r0+s0, r1+s1)
```

The public `shrink` argument uses `(start, end)`. Its UOp representation stores
`(start, length)`, where `length=end-start`. This is a recurring source of
plausible-looking mistakes: a debugger sees the second internal number and
mistakes it for an end coordinate.

### `FLIP`: count from the opposite edge

For an axis of extent `s`, a flip maps output coordinate `o` to:

```text
s - 1 - o
```

The `-1` is required because the last valid coordinate is `s-1`, not `s`.
Flipping the columns of our base gives:

```text
[[2, 1, 0],
 [5, 4, 3]]
```

Output `(r0, r1)` reads `(r0, 2-r1)`.

### Stepped slices are movement sequences

`base[:, ::2]` selects columns 0 and 2:

```text
[[0, 2],
 [3, 5]]
```

`base[:, ::-1]` reverses columns. The basic indexing parser first resolves
Python slice rules, including negative bounds and negative steps. It then
normalizes the operation through shrink, flip, pad/reshape/shrink steps for a
non-unit absolute stride, and a final reshape. Do not expect a dedicated
`STRIDE` operation in the graph.

That normalization is implementation detail, but it has a practical
consequence: when a stepped-slice bug appears, inspect the whole movement chain
and its composed map, not only the first `SHRINK`.

### `PAD`: subtract an offset and carry validity

Pad one value before and after each axis of the carried `(2, 3)` view:

```text
[[0, 0, 0, 0, 0],
 [0, 0, 2, 4, 0],
 [0, 1, 3, 5, 0],
 [0, 0, 0, 0, 0]]       shape (4, 5)
```

For a source axis of extent `s` with `before` padding values, output coordinate
`r` proposes source coordinate:

```text
r - before
```

That coordinate is valid only when:

```text
before <= r < before + s
```

Both pieces are required. At the left pad, the proposed source coordinate is
negative. At the right pad, it is too large. The value-producing rewrite turns
invalid positions into the pad value without authorizing an out-of-bounds
load.

Low-level `Ops.PAD` in this snapshot has zero-pad semantics. A public pad with a
nonzero constant constructs zero-pad data/validity and a `WHERE` that selects
the requested value outside. Reflect, replicate, and circular modes are also
composite operations rather than alternate meanings of the low-level `PAD`
node. This distinction matters when adding a rewrite: matching `Ops.PAD` does
not mean matching every high-level padding mode in one node.

## Broadcasting from first principles

Elementwise arithmetic needs one logical coordinate at which to read every
operand. If operand shapes differ, **broadcasting** decides whether such maps
exist.

Align shapes at the right. For each aligned axis, the extents are compatible
when they are equal or at least one is 1. Missing leading axes behave like
size-one axes. After ignoring size-one extents, there must be at most one
remaining extent; that unique non-1 extent becomes the result. This wording
also handles empty axes: zero is an extent, not something that loses to one by
numeric size.

Examples:

| Left | Right | Result | Why |
| --- | --- | --- | --- |
| `(2, 3)` | `(2, 3)` | `(2, 3)` | every extent equal |
| `(2, 1)` | `(1, 3)` | `(2, 3)` | each operand repeats on one axis |
| `(3,)` | `(2, 3)` | `(2, 3)` | missing leading axis repeats |
| `(0, 3)` | `(1, 3)` | `(0, 3)` | extent 1 yields to the unique non-1 extent 0 |
| `(2, 2)` | `(2, 3)` | error | last extents differ and neither is 1 |

For a `(2, 1)` column times a `(1, 3)` row, let `(r0, r1)` be the result
coordinate:

```text
column source coordinate = (r0, 0)
row source coordinate    = (0, r1)
```

This produces an outer-product-shaped result. Each size-one source axis maps
to constant zero.

### Implicit broadcasting is not necessarily an `EXPAND` node

It is tempting to search a failed elementwise multiply for `Ops.EXPAND`. In this
snapshot, broadcastable UOps derive the combined shape directly. During
rangeification, `broadcast_rngs` maps result ranges to each operand and replaces
broadcast axes with zero. The raw multiply can therefore have shape `(2, 3)`
without any `EXPAND` in its reachable graph.

An explicit call such as `column.expand(2, 3)` is different. At this snapshot,
the normalized movement chain from the explicit result back to its source is:

```text
PERMUTE -> EXPAND -> RESHAPE
```

Its final coordinate map is still `(r0, 0)`. These are two representations of
the same broadcast relationship in different contexts:

- implicit elementwise broadcasting: the consumer maps its ranges to each
  operand;
- explicit expand: movement nodes describe a standalone expanded value.

When testing a broadcast change, assert values and coordinate maps first. An
assertion that “an EXPAND node must exist” would encode a false implementation
assumption.

## Python indexing and IR indexing are different layers

The spelling `tensor[...]` appears to be an address operation, but it is a
frontend request with several possible meanings. `Ops.INDEX` is a later IR
operation that attaches one or more address expressions to a buffer-like
source. Keep those concepts separate.

### Basic Python indexing becomes movement operations

Basic indices include integers, slices, `None`, and `...`:

```python
base[1]       # [3, 4, 5]
base[:, ::2]  # [[0, 2], [3, 5]]
base[None]    # add a size-one axis
base[...]     # fill omitted full slices
```

The frontend normalizes ellipsis and missing axes, parses one descriptor per
axis, applies the view movement sequence, injects `None` axes, and collapses
integer-selected axes. At the pinned snapshot, `base[1]` has a `RESHAPE` at its
visible UOp root after this normalization; it is not an IR `INDEX` root.

A basic integer is checked against its known axis extent. `base[3]` on a
two-row tensor raises `IndexError`. Negative integers are translated relative
to the end when valid.

### Advanced indexing constructs selection algebra

A list, a non-scalar Tensor index, or a tuple used as one index descriptor
enters advanced indexing. Python's top-level indexing tuple needs care:
`base[(1,0)]` supplies two basic integer descriptors and returns one scalar,
whereas `base[(1,0),]` supplies one tuple-of-integer-data descriptor and is
advanced indexing.

```python
rows = Tensor([1, 0])
cols = Tensor([2, 1])
base[rows, cols]          # [5, 1]
```

The two index tensors broadcast to a common index shape. When more than one
advanced-index axis is consecutive and the selected base-axis extents are
concrete integers, this snapshot can combine them into a linear index, compute
a validity condition, select a safe fallback index, and then use `WHERE` to
make invalid selections zero. Other advanced arrangements construct one-hot
masks and reductions.

This means basic and advanced out-of-bounds behavior differs at the pin:

```text
base[3]       -> IndexError
base[[3,-4]]  -> two all-zero rows
```

Do not generalize this snapshot behavior to every framework or future tinygrad
version. More importantly, do not “fix” an advanced-index result by changing
low-level `Ops.INDEX` before confirming which frontend path built it.

### What `Ops.INDEX` means

At the IR layer, conceptually:

```text
INDEX(buffer_like_source, address_expression...)
```

means “this storage object at these index coordinates.” It does not itself mean
Python slicing. A scalar-address backend will usually end with one flattened
integer, while an image or other shaped address space may retain multiple
coordinates.

An index can later become a read when codegen's load-insertion pass sees it in a
value position. Stores already exist as effects earlier in the plan: their
destination is an indexed storage source and their value may contain input
indices. It is inaccurate to say that codegen invents every load and store
together. At this boundary it notably wraps eligible storage/index values in
`LOAD`, including the value supplied to a pre-existing `STORE`.

## How shape is represented in current UOps

You can inspect `uop.shape`, but shape is not merely an arbitrary tuple attached
once and trusted forever. The `_shape` property derives it according to the
operation:

- `RESHAPE` validates equal element counts and returns its resolved new shape;
- `EXPAND` prepends its stored extents;
- `PERMUTE` reorders the source shape;
- `PAD` and `SHRINK` validate normalized offset/size pairs and return stored
  output sizes;
- `FLIP` preserves shape; and
- broadcastable elementwise operations compute the common broadcast shape.

Movement arguments are encoded as UOp sources when needed. The convenient
`marg` property translates that representation back into the tuple consumed by
coordinate mapping:

| Public idea | Movement `marg` at this pin |
| --- | --- |
| reshape to new shape | the resolved new shape |
| expand | leading extents added by this low-level node |
| permute | axis permutation |
| flip | one Boolean per source axis |
| pad | `(before, new_total_size)` per axis |
| shrink | `(start, length)` per axis |

This validation is useful evidence. If the visible output shape is already
wrong, inspect frontend normalization and `_shape` before studying generated
addresses. If shape is right but values are shuffled, proceed to the coordinate
map.

## Turn a tensor computation into loops

Return to the carried expression, now adding 10:

```python
out = base.reshape(3, 2).permute(1, 0) + 10
```

The visible result has shape `(2, 3)`. A straightforward loop description is:

```python
for r0 in range(2):
  for r1 in range(3):
    out[r0, r1] = base_flat[r1*2 + r0] + 10
```

The two counters describe **iteration space**: which result coordinates exist.
The expression `r1*2+r0` describes an **access function**: where one iteration
reads. The expression `r0*3+r1` describes where it writes a contiguous result.

Compiler IR needs both ideas. In tinygrad here:

- `RANGE` represents a loop-like symbolic coordinate;
- `INDEX` associates an address expression with storage; and
- `STORE` represents the output write effect.

### `RANGE` bounds are half-open even though `vmax` is inclusive

`UOp.range(end, ...)` represents values from 0 through `end-1`, just like
Python `range(end)`. A range with end 3 therefore has:

```text
render: r1
vmin:   0
vmax:   2
```

`vmax` is an inclusive fact about possible values. Do not read it as the
half-open loop end. Confusing these conventions creates classic off-by-one
errors.

An extent-one axis needs only coordinate zero. `IndexingContext.new_range(1)`
returns constant `0`, not a `RANGE` that performs one iteration. An extent-three
axis returns a real range. Removing a size-one range in this way simplifies
broadcast and reshape expressions without changing the set of coordinates.

### Rangeification walks from consumers toward producers

It is also too simple to say “rangeification creates one fresh range for every
dimension.” It first determines which values must be realized and builds a
consumer map. Then it walks in reverse topological order:

- a realized output starts new ranges;
- a node with one ranged consumer normally inherits the consumer's compatible
  ranges;
- multiple consumers may share ranges or force new ranges and partial
  realization;
- broadcast axes map to constant zero for the source;
- movement nodes transform output ranges into source ranges; and
- reductions introduce reduction ranges for their reduced axes.

Range inheritance is how elementwise producers can stay in the same kernel as
their consumer. Fresh ranges are also scheduling decisions, not merely a
restatement of a shape tuple.

After those maps are recorded, another rewrite replaces movement semantics with
explicit indexed storage and local validity logic. The original movement nodes
can then disappear from the rangeified data path because their meaning now
lives in the access expressions.

### The carried plan's concrete artifact

At the pinned snapshot, a fresh call to `schedule_linear()` for the carried
expression produces one `CALL`. Inside its body, the important pieces are:

```text
RANGE r0: vmin=0, vmax=1
RANGE r1: vmin=0, vmax=2
output INDEX: r0*3 + r1
input INDEX:  r1*2 + r0
```

Read it against the loop above. The output index is ordinary contiguous
row-major storage for shape `(2, 3)`. The input index is the composed
reshape-then-permute map. The value at output `(0,2)` reads input position 4,
then adds 10.

`schedule_linear()` mutates connected Tensor wrappers as it replaces lazy
representations with planned storage. Use a fresh expression for inspection
and treat the returned plan as the final artifact; do not schedule a Tensor and
then expect to realize that same wrapper normally.

## Validity is part of an address

Consider public padding `((1, 1), (2, 1))` on source shape `(2, 3)`. That means:

- axis 0: one before, one after, output extent 4;
- axis 1: two before, one after, output extent 6.

The internal movement argument is:

```text
((1, 4), (2, 6))
```

Remember: each pair is `(before, new_total_size)`, not `(before, after)`.

For output ranges `(r0, r1)`, `apply_movement_op` produces proposed source
coordinates and validity conditions:

```text
axis 0 index = r0 - 1    valid when 1 <= r0 < 3
axis 1 index = r1 - 2    valid when 2 <= r1 < 5
```

UOp expression rendering currently prints equivalent canonical Boolean forms
such as:

```text
(((r0<1)!=True)&(r0<3))
```

Read semantics, not typography: `(r0<1)!=True` means `not (r0<1)`, hence
`r0>=1`.

Check boundary representatives:

| Output | Proposed source `(index, valid)` per axis | Overall |
| --- | --- | --- |
| `(0, 0)` | `(-1, False)`, `(-2, False)` | pad |
| `(1, 2)` | `(0, True)`, `(0, True)` | first source value |
| `(2, 4)` | `(1, True)`, `(2, True)` | last source value |
| `(3, 5)` | `(2, False)`, `(3, False)` | pad |

tinygrad represents a value carrying validity using a `WHERE(condition, value,
Invalid)` form. `Invalid` is a sentinel, not the numeric padding value. Helpers
separate the pair:

- `get_idx()` extracts the proposed value/address;
- `get_valid()` extracts its condition;
- an ordinary value has validity true; and
- bare `Invalid` has validity false.

During rangeification, `PAD` becomes local `WHERE` behavior: combine per-axis
validity, use the source value when valid, and use zero otherwise. This ordering
lets later rules simplify addresses and conditions without silently permitting
an invalid load.

For any validity change, test at least four points: one just before the valid
region, the first valid point, the last valid point, and one just after. A test
only in the interior will not catch either off-by-one boundary.

## Symbolic values are statements about sets

Real programs do not always know every extent or index when Python constructs
the graph. tinygrad represents symbolic integers as UOps too.

```python
row = UOp.variable("row", 0, 7)
col = UOp.variable("col", 0, 15)
```

These declarations mean:

```text
row may be any integer in {0, ..., 7}
col may be any integer in {0, ..., 15}
```

Both endpoints are inclusive. This convention differs from `RANGE(end)`, whose
end is excluded. The facts are useful precisely because they describe every
allowed value, not only one example.

Build a flat address for 8 rows of width 16:

```python
flat = row*16 + col
```

tinygrad derives conservative bounds `0 <= flat <= 127`. It can then prove:

```text
flat // 16  -> row
flat % 16   -> col
```

Why is the proof legal? Since `0 <= col < 16`, adding `col` never crosses into
the next group of 16. Quotient recovers the complete-group count and remainder
recovers the position within the group.

### A negative control: one wider bound breaks the proof

Now declare:

```python
wide_col = UOp.variable("wide_col", 0, 16)
```

The maximum 16 is allowed. At `wide_col=16`:

```text
(row*16 + 16) // 16 = row + 1
```

So replacing the quotient with `row` would be wrong. The pinned simplifier
retains:

```text
row + wide_col//16
```

This is an essential contributor habit: accompany a proof example with the
nearest counterexample. It checks that the rewrite responds to its precondition
rather than merely recognizing familiar syntax.

### Divisibility is another proof fact

```python
n = UOp.variable("n", 8, 64, multiple_of=8)
```

This says `n` is within the inclusive bounds and divisible by 8. Therefore:

```text
n % 8       -> 0
(n // 8) bounds -> 1 through 8
```

`n.bind(24)` creates a binding because 24 is in range and divisible by 8.
Binding 26 fails divisibility; binding 72 fails the maximum bound. A binding is
not permission to forget the declared domain during arbitrary rewrites. The
symbol remains the parameter, and later schedule logic collects concrete bound
values actually used by the plan.

Bounds are not automatic runtime checks on all external data. They are facts
attached during graph construction. A rewrite is sound only if it holds for
every value admitted by those facts. Tests must therefore cover both constraint
construction and symbolic transformation.

## Floor division and modulo: signs matter

For positive addresses and positive extents, quotient/remainder intuition is
simple. Symbolic rules are more general, and tinygrad's relevant operations are
`FLOORDIV` and `FLOORMOD`. Their semantics follow floor division.

The identity remains:

```text
quotient * denominator + remainder = numerator
```

but the sign of the remainder follows the denominator under Python-style floor
semantics. Two checked examples are:

| numerator | denominator | quotient | remainder | reconstruction |
| ---: | ---: | ---: | ---: | --- |
| -7 | 3 | -3 | 2 | `-3*3+2 = -7` |
| 7 | -3 | -3 | -2 | `-3*-3-2 = 7` |

Code written with C/CUDA truncation-toward-zero intuition would predict
different quotients for these cases. Before accepting a new div/mod rewrite,
state its denominator sign assumptions and its numerator range.

The div/mod simplifier uses several kinds of evidence: constant denominators,
bounds that trap an expression within one quotient interval, exact factors,
greatest common divisors, congruence, and `multiple_of` facts. It can remove
nested modulo, split divisible terms from a sum, or reconstruct a quotient and
remainder. These are proof-guided rules, not guesses that addresses are
“probably positive.”

## A paper lab before the runnable lab

Do these without tinygrad first. The point is to separate your coordinate
reasoning from whatever the current renderer happens to print.

### Problem 1: compose reshape and permute

For source shape `(2, 3)`, reshape to `(3, 2)`, then permute `(1, 0)`.

1. What is the final shape?
2. Which reshaped coordinate does final `(a, b)` read?
3. What is its logical flat position?
4. Which source coordinate does it read?
5. What source flat address remains after simplification?

??? success "Worked answer"

    The final shape is `(2, 3)`. Permute maps `(a,b)` back to `(b,a)` in the
    reshaped `(3,2)` view. Its flat position is `2*b+a`. Decomposing for source
    shape `(2,3)` gives `((2*b+a)//3, (2*b+a)%3)`. Flattening that coordinate
    reconstructs `2*b+a`.

### Problem 2: broadcast two operands

A column has shape `(4, 1)` and a row has shape `(1, 5)`. For result coordinate
`(a, b)`, write both source coordinate maps. Then name one incompatible shape
for an elementwise operand.

??? success "Worked answer"

    The result shape is `(4,5)`. The column reads `(a,0)` and the row reads
    `(0,b)`. Shape `(4,2)` is incompatible with `(4,5)` because the last
    extents 2 and 5 differ and neither is 1.

### Problem 3: pad validity

A one-dimensional source has length 4 and public padding `(2, 1)`. Give the
output length, source-index formula, valid interval, first valid output, and
last valid output.

??? success "Worked answer"

    The output length is 7. Output `r` proposes source index `r-2`. It is valid
    when `2 <= r < 6`. Output 2 is the first valid point and reads source 0;
    output 5 is the last valid point and reads source 3. Outputs 1 and 6 are
    useful adjacent invalid tests.

### Problem 4: reject an unsound symbolic rewrite

Someone proposes `(row*w+col)//w -> row` for all integer `row`, `col`, and
nonzero `w`. Give two missing preconditions and one concrete counterexample.

??? success "Worked answer"

    A common sufficient domain is `w > 0` and `0 <= col < w`. Without the
    upper bound, `row=2, w=4, col=4` gives quotient 3, not 2. Negative widths or
    negative/reordered remainder domains need separate floor-semantics analysis.

## Runnable lab: values, maps, ranges, and proofs

The checked-in lab turns each paper artifact into an assertion. Run it from the
guide repository root. The path deliberately points to the separate pinned
tinygrad study checkout created during setup:

```bash
CACHEDB=/tmp/tinygrad-guide-shapes.db DEV=PYTHON DEBUG=0 \
  ../tinygrad-study/.venv/bin/python labs/phase3/shapes_and_indexing.py
```

`DEV=PYTHON` makes the visible Tensor checks portable. `DEBUG=0` prevents an
ambient debug setting from flooding or changing the output. A task-local
`CACHEDB` keeps this exercise's cache separate.

The pinned output is:

```text
base shape/values:        (2, 3) [[0, 1, 2], [3, 4, 5]]
view shape/values:        (2, 3) [[0, 2, 4], [1, 3, 5]]
expanded shape:           (4, 2, 3)
expanded batches equal:   True
padded shape/values:      (4, 5) [[0, 0, 0, 0, 0], [0, 0, 2, 4, 0], [0, 1, 3, 5, 0], [0, 0, 0, 0, 0]]
reverse columns:          [[2, 1, 0], [5, 4, 3]]
stride-two columns:       [[0, 2], [3, 5]]
basic/advanced roots:     RESHAPE WHERE
basic/advanced values:    [3, 4, 5] [5, 1]
basic/advanced OOB:       IndexError [[0, 0, 0], [0, 0, 0]]

view movement chain:      ['PERMUTE', 'RESHAPE']
PERMUTE  -> source shape (3, 2): ('r1', 'r0')
RESHAPE  -> source shape (2, 3): ('((r1*2+r0)//3)', '((r1*2+r0)%3)')
source flat address:      (r1*2+r0)
view coordinate (0, 0) -> base coordinate (0, 0)
view coordinate (0, 2) -> base coordinate (1, 1)
view coordinate (1, 0) -> base coordinate (0, 1)
view coordinate (1, 2) -> base coordinate (1, 2)

implicit broadcast:       (2, 1) (1, 3) -> (2, 3)
implicit source maps:     ('r0', '0') ('0', 'r1')
implicit has EXPAND:      False
explicit expand chain:    ['PERMUTE', 'EXPAND', 'RESHAPE']
PERMUTE  -> source shape (3, 2): ('r1', 'r0')
EXPAND   -> source shape (2,): ('r0',)
RESHAPE  -> source shape (2, 1): ('r0', '0')
explicit final map:       ('r0', '0')
singleton/active axes:    CONST 0 RANGE r0 0 2

pad public/internal:      ((1, 1), (2, 1)) ((1, 4), (2, 6)) (4, 6)
pad axis 0 idx (r0+-1) valid (((r0<1)!=True)&(r0<3))
pad axis 1 idx (r1+-2) valid (((r1<2)!=True)&(r1<5))
pad coordinate (0, 0) -> ((-1, False), (-2, False))
pad coordinate (1, 2) -> ((0, True), (0, True))
pad coordinate (2, 4) -> ((1, True), (2, True))
pad coordinate (3, 5) -> ((2, False), (3, False))
zero/nonzero pad roots:   PAD WHERE

rangeified ranges:        [('r0', 0, 1), ('r1', 0, 2)]
rangeified output index:  (r0*3+r1)
rangeified input index:   (r1*2+r0)

symbolic flat/bounds:     (col+row*16) 0 127
recover row/column:       row col
insufficient bound:       (row+wide_col//16)
multiple-of proof:        0 1 8
valid binding:            24
floor div/mod examples:   [(-7, 3, -3, 2), (7, -3, -3, -2)]
exhaustive small oracle:  passed
```

### Read the output in five passes

First, compare visible values. If you cannot predict the tiny lists, do not move
to IR. The assertions establish the intended semantics independently of the
rendered coordinate formulas.

Second, trace the movement chain root-to-source. The lab calls the same
`apply_movement_op` definition used by rangeification. `PERMUTE` maps final
ranges to `(r1,r0)`; `RESHAPE` decomposes their flat position into the original
coordinate. Substituting four concrete coordinates guards against a pretty but
wrong formula.

Third, contrast implicit and explicit broadcasting. Both maps select coordinate
zero on the repeated source axis. Only the explicit value has the normalized
movement chain. The singleton line separately shows that size-one iteration
becomes constant zero.

Fourth, read each pad coordinate as `(proposed index, valid)`. Negative proposed
indices are harmless only because validity is retained. The root comparison
shows that zero and nonzero padding have different high-level graph forms.

Fifth, connect the final rangeified and symbolic blocks. The rangeified input
address matches the paper derivation. Bounds then prove a more general
flatten/decompose round trip, while `wide_col` demonstrates that the proof stops
when its strict remainder bound is absent. The exhaustive oracle checks all
`8*16` concrete row/column pairs.

## Focused upstream regressions

After the lab passes, run existing tests from the tinygrad checkout rather than
from the guide root:

```bash
cd ../tinygrad-study
DEV=PYTHON DEBUG=0 .venv/bin/python -m pytest -q \
  test/null/test_indexing.py \
  test/null/test_tensor_uop_mixin.py::TestTensorUOpGetitem \
  test/null/test_tensor_uop_mixin.py::TestTensorUOpPad \
  test/unit/test_symbolic_tensor.py \
  test/null/test_symbolic_tensor.py
```

If you change div/mod rules, add the focused reshape-roundtrip cases in
`test/null/test_uop_symbolic.py`, then run the wider symbolic suite available in
your configured development environment. Some external fuzzers require optional
NumPy, Hypothesis, or Z3 dependencies; a missing extra is an environment issue,
not evidence that a rewrite passed.

`test/external/fuzz_shape_ops.py` at this snapshot covers split, chunk, squeeze,
and unsqueeze comparisons. It is useful but not broad proof for every movement
operation discussed here. Likewise, `fuzz_symbolic_div.py` primarily targets
`CDIV`/`CMOD`; do not cite it as complete `FLOORDIV`/`FLOORMOD` reshape-index
coverage. Read a fuzzer's generators and oracle before deciding what evidence
it supplies.

## A debugging ladder

When values are wrong, preserve artifacts from the highest-level intent to the
lowest-level address. Stop at the first artifact that differs from expectation:

1. public operation sequence, arguments, shapes, and a tiny visible result;
2. normalized movement chain and each node's `marg`;
3. output-coordinate to source-coordinate map, including validity;
4. rangeified `RANGE`, input `INDEX`, output `INDEX`, and `STORE`;
5. codegen `LOAD`/`STORE` gates and final rendered address; and
6. comparison with a simple reference or exhaustive bounded oracle.

This ordering keeps you from repairing the renderer when the slice was parsed
wrong, or changing frontend semantics when only a late load gate was lost.

| Symptom | Ask first | Likely first source area |
| --- | --- | --- |
| Wrong output shape | Was the public argument normalized correctly? | movement mixin and UOp `_shape` |
| Right shape, shuffled values | What map does each movement node apply backward? | `apply_movement_op` |
| Broadcast reads one row/column incorrectly | Which operand axes map to zero? | `_shape`, `broadcast_rngs`, explicit `_broadcast_to` |
| Stepped or negative slice wrong | What full shrink/flip/stride-normalization chain was built? | `_parse_view_index`, `_apply_view_ops` |
| Pad edge wrong | Are internal pairs `(before,total)`, and are both inequalities strict in the right place? | pad normalization and validity conversion |
| Out-of-bounds memory effect | Does the index still carry `get_valid()` when the load/store is formed? | indexing, validity rewrites, codegen load insertion |
| Huge reshape address | Which quotient/remainder identity lacks a proof? | symbolic and div/mod rules |
| Simplification fails only at one size | Are bounds inclusive, and did a remainder reach the divisor? | variable bounds and min/max propagation |
| Advanced index differs from basic index | Which `_getitem` branch built the graph? | frontend indexing, before low-level INDEX |
| Old article mentions ShapeTracker APIs | What is the equivalent movement-UOp/rangeification path at this pin? | current source, not historical names |

## Question-led source tour

Opening a large file without a question makes declarations look arbitrary. Each
stop below gives you a prediction to make, a deliberately narrow source range,
and a plain-language translation. Use the pinned links so line drift does not
change the lesson.

### Stop 1: where does `Tensor.shape` come from?

**Question:** Does the public Tensor object calculate shape independently, or
delegate it to its UOp? Predict before opening the link.

Read [`Tensor.shape`, lines 143–147](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L143-L147), then the inherited
[`ndim` definition, lines 27–36](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L27-L36).

**Translation:** the Tensor property returns `self.uop.shape`; rank is the
length of that tuple. This tells you that a wrong Tensor shape can originate in
UOp shape derivation rather than a separate Tensor metadata tracker.

### Stop 2: how are basic indices parsed?

**Question:** What happens to `...`, `None`, negative integers, a negative
slice step, and a step of zero?

Read [`_normalize_indices`, `_parse_view_index`, and `_apply_view_ops`, lines
63–114](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L63-L114).

**Translation:** ellipsis becomes enough full slices to cover omitted axes;
`None` does not consume a source axis; a basic integer is range-checked and
normalized; Python resolves concrete slices; zero step is rejected; negative
step becomes flip; and an absolute step above one expands into movement
operations. The returned dictionaries are a normalized intermediate language
for `_getitem`.

### Stop 3: where do basic and advanced indexing diverge?

**Question:** Which inputs count as advanced, and why can two adjacent Tensor
indices become a linear address plus validity rather than ordinary slices?

Read [the beginning and consecutive-index path of `_getitem`, lines
74–117](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/op.py#L74-L117).

**Translation:** list/tuple data and non-scalar Tensor-like indices are marked
advanced. Basic descriptors build view operations first. Consecutive advanced
axes with concrete shapes get row-major strides, combine into one linear index,
and carry a validity check that controls zero output for out-of-range choices.
The later one-hot path is worth reading only after this smaller branch makes
sense.

### Stop 4: how are movement and broadcast shapes derived?

**Question:** What does each movement operation promise about output shape, and
can an elementwise op broadcast without an `EXPAND` source?

Read [`_broadcast_shape`, lines 68–83](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L68-L83), then [the movement and broadcastable cases of `UOp._shape`, lines
428–474](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L428-L474).

**Translation:** each movement case validates its normalized argument and
returns a derived tuple. `_broadcast_shape` right-aligns axes, ignores size-one
extents, and requires all remaining extents to agree. The broadcastable case
calls it over its sources, which explains why an implicit multiply can acquire
`(2,3)` shape without constructing an explicit expanded value.

### Stop 5: why does explicit expand have a surprising chain?

**Question:** If low-level `EXPAND` adds only leading axes, how can public
`expand(2,3)` repeat the second axis of `(2,1)`?

Read [`_broadcast_to`, lines 116–135](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L116-L135).

**Translation:** align shapes, validate equal-or-one extents, reshape away the
size-one axes that must repeat, add them as leading expansion extents, then
permute them back into visible order. The lab's
`PERMUTE -> EXPAND -> RESHAPE` chain is the reverse traversal of that
construction.

### Stop 6: what exactly is stored in a movement UOp?

**Question:** Are pad pairs public `(before,after)` pairs? Are shrink pairs
public `(start,end)` pairs?

Read [`marg` and `_mop`, lines 782–809](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L782-L809), then the small
[pad/shrink constructors, lines 174–200](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/mixin/movement.py#L174-L200).

**Translation:** shape-like arguments can live as sources and `marg` decodes
them. Public constructors convert pad to `(before,total)` and shrink to
`(start,length)` before `_mop` encodes the pairs. Always label which layer an
argument belongs to in a bug report.

### Stop 7: what is the exact coordinate definition?

**Question:** Can you predict all six cases before reading the match statement?
For reshape, where are output coordinates flattened and source coordinates
decomposed?

Read [`_apply_reshape` and `apply_movement_op`, lines
145–177](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L145-L177).

**Translation:** this is the executable definition behind the chapter's maps.
Shrink adds offsets, permute applies the inverse order, flip subtracts from the
last coordinate, expand removes injected leading ranges, pad attaches validity,
and reshape performs the flatten/decompose round trip. The final reshape rewrite
does substantial symbolic cleanup.

### Stop 8: when are ranges new, inherited, or zero?

**Question:** Does every shape axis receive a fresh `RANGE`? What happens to a
broadcast source axis and an extent-one axis?

Read [`IndexingContext.new_range` and `broadcast_rngs`, lines
45–61](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L45-L61), then the
[range-selection portion of `run_rangeify`, lines 195–271](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L195-L271).

**Translation:** size one becomes constant zero. Broadcast source axes also
receive zero. A realized node creates output ranges, a single-consumer producer
usually inherits mapped consumer ranges, and multiple consumers may share or
force new ones. The comments enumerate the policy before the code implements
it.

### Stop 9: how do maps become indices and pad values?

**Question:** When a buffer-like source is encountered, where are ranges
attached? When does pad validity become zero-producing behavior?

Read [`create_bufferize_and_index_srcs` and pad conversion, lines
63–104](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L63-L104), then the
[backward coordinate-mapping loop, lines 278–308](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L278-L308), the
[movement-removal rule and matcher, lines 123–137](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L123-L137), and its
[application at lines 310–315](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L310-L315).

**Translation:** the recorded range map determines which data sources receive
`.index(...)`; the backward loop transfers movement semantics into source
coordinate maps; staging/realization closes selected ranges; and pad combines
range validities and selects zero outside. The final matcher can remove a
movement node only after its coordinate meaning has been transferred, and the
bottom-up `graph_rewrite` applies that matcher.

### Stop 10: why isn't `uop[...]` always `Ops.INDEX`?

**Question:** Which UOps use the shared Python view path, and which storage
objects treat brackets as an actual scalar lookup?

Read [`UOp.index` and `UOp.__getitem__`, lines
574–591](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L574-L591).

**Translation:** `.index(...)` explicitly constructs `Ops.INDEX`. Bracket
syntax checks address space/device context: ordinary ALU/deviced UOps use the
same movement-mixin indexing path as Tensor, while storage-like UOps can create
an actual index. Slices of storage are themselves normalized before remaining
scalar coordinates are attached.

### Stop 11: how are validity and variables represented?

**Question:** How can one expression carry both a proposed index and a Boolean
condition? Are variable bounds inclusive, and what does `bind` validate?

Read [validity helpers, lines 652–660](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L652-L660), then
[`variable` and `bind`, lines 983–1005](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L983-L1005).

**Translation:** validity is encoded as a `WHERE` whose false branch is
`Invalid`, and helpers recognize that form. A variable is an ALU-space `PARAM`
with an inclusive min/max pair and a divisibility fact. Binding asserts both
range containment and divisibility.

### Stop 12: what licenses a div/mod rewrite?

**Question:** Which rules require a positive constant denominator? Which facts
can reduce a whole quotient interval to one constant or split divisible terms?

Read the opening of [`fold_divmod_general`, lines
8–83](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/divandmod.py#L8-L83).
Confirm the matcher composition in [`symbolic`, lines
217–295](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/symbolic.py#L217-L295).

**Translation:** zero is rejected; min/max can prove a constant result;
`multiple_of` can prove a parameter remainder zero; several rules explicitly
require a positive scalar constant; congruence and GCD transformations carry
additional range guards. `div_and_mod_symbolic` is already included in the
general symbolic matcher composition in this snapshot—do not describe it as a
separate pass that always runs afterward.

For a focused regression demonstrating the exact reshape identity, read
[`test_reshape_index_roundtrip`, lines 872–879](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/null/test_uop_symbolic.py#L872-L879).

## Controlled exercises for contributor readiness

### Exercise 1: change the carried shape

Use `Tensor.arange(12).reshape(3,4)`, then `reshape(2,6).permute(1,0)`.
Before running anything:

1. write the final shape and first two displayed rows;
2. map final `(r0,r1)` to the `(2,6)` coordinate;
3. derive the original `(3,4)` coordinate; and
4. simplify its flat address.

??? success "Answer"

    Final shape is `(6,2)` and the first two rows are `[[0,6],[1,7]]`.
    Permute maps `(r0,r1)` to `(r1,r0)` in `(2,6)`, whose flat position is
    `6*r1+r0`. The original coordinate is
    `((6*r1+r0)//4, (6*r1+r0)%4)`, and its original flat address reconstructs
    `6*r1+r0`.

### Exercise 2: separate shape from address

Create two expressions with output shape `(2,3)`: a contiguous base and the
carried reshaped/permuted view. Explain why equal shape does not imply equal
indexing or equal values.

??? success "Answer"

    Both iterate over ranges `r0 in [0,2)` and `r1 in [0,3)`. A contiguous
    value reads flat `r0*3+r1`; the carried view reads `r1*2+r0`. Shape defines
    which output coordinates exist, while the access function defines which
    source supplies each coordinate.

### Exercise 3: add an adversarial pad case

For source length zero or one, predict how a pad validity interval changes.
Write boundary assertions without assuming that there is a “middle” element.

??? success "Answer"

    For source length one and `before=b`, exactly output `b` is valid:
    `b <= r < b+1`. Test `b-1`, `b`, and `b+1` where those output coordinates
    exist. For source length zero, the interval `b <= r < b` is empty and every
    padded output is invalid. Avoid a test that computes “last valid” as
    `b+s-1` without first checking `s>0`.

### Exercise 4: design a symbolic property test

State a bounded oracle for `(row*w+col)//w == row` and
`(row*w+col)%w == col`. Include both valid and invalid precondition cases.

??? success "Answer"

    Enumerate small positive `w`, bounded rows, and `col` in `range(w)`; both
    identities must hold. Then include `col=w`, where quotient recovery must
    fail, and optionally negative `col`, where the claimed remainder equality
    is outside this domain. Compare simplified UOps by substitution or evaluate
    both sides rather than requiring one exact rendered tree.

### Exercise 5: plan a real bug report

Suppose `x[:, ::-2]` has the right shape but wrong values on CUDA only. List the
smallest artifacts you would attach before proposing a fix.

??? success "Answer"

    Include a tiny concrete input and expected/actual values on `PYTHON` and
    CUDA; public slice and shapes; normalized movement chain and arguments;
    composed coordinate map at representative endpoints; rangeified input
    index and validity; final CUDA address/gate; snapshot, device, environment,
    and exact reproduction command. This evidence distinguishes frontend,
    rangeification, and backend failures.

## Checkpoint artifact

Before continuing, create one short note—on paper or in your study log—with all
of the following for the carried example:

1. the three visible arrays (`base`, reshaped, final view);
2. the final output coordinate domain;
3. the two-step backward coordinate map;
4. the simplified source and output flat indices;
5. one implicit broadcast map for each operand;
6. one pad index plus validity interval;
7. one valid symbolic proof and its nearest invalid-bound counterexample; and
8. the exact lab and focused-test commands you ran.

You are ready for the next chapter when you can answer, without source open:

- Why does the carried input index use `r1*2+r0` while the output uses
  `r0*3+r1`?
- Why can implicit broadcasting contain no `EXPAND`?
- Why does a size-one axis become constant zero?
- Why is a padded negative proposed index safe only while validity survives?
- Why is `wide_col <= 16` insufficient to recover `row`?
- Where would you look first for right-shape/wrong-values, wrong-shape, and
  late out-of-bounds-store failures?

If any answer feels like memorized vocabulary, change one extent and recompute
the tiny example. Coordinate reasoning should survive different numbers.

## Quick reference

| Concept | Precise working meaning |
| --- | --- |
| shape | tuple of extents, one per logical axis |
| rank / `ndim` | number of axes, `len(shape)` |
| coordinate | one zero-based integer per axis |
| numel | product of extents; scalar `()` has one |
| contiguous stride | element-index change caused by advancing one axis |
| row-major | last logical axis changes fastest |
| view | output shape plus coordinate map and, when needed, validity |
| reshape | flatten in output/current logical order, decompose in source shape |
| permute | reorder axes; backward map uses inverse order |
| expand/broadcast | repeated source axes map to coordinate zero |
| shrink | add source start offset |
| flip | map `r` to `extent-1-r` |
| pad | subtract `before`; require `before <= r < before+source_extent` |
| `RANGE(end)` | loop-like values `0 ... end-1`; `vmax=end-1` |
| `Ops.INDEX` | storage-like source plus one or more address coordinates |
| `Invalid` | sentinel used to carry false validity, not a numeric pad value |
| symbolic bounds | inclusive set of allowed values used as proof facts |
| `multiple_of` | divisibility fact available to symbolic reasoning and binding |
| floor div/mod | quotient rounds down; remainder reconstructs with denominator |

## Optional background, only where you need it

You do not need to finish these before experimenting. Use the smallest resource
that resolves the specific gap you encounter:

- NumPy's [broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html)
  gives many visible shape examples; translate every example into source
  coordinate maps to connect it to this chapter.
- NumPy's [array internals overview](https://numpy.org/doc/stable/dev/internals.html#internal-organization-of-numpy-arrays)
  explains the conventional shape/stride/data-buffer model. Keep the current
  tinygrad movement-UOp path distinct from NumPy's object representation.
- Python's [expression reference for slicings](https://docs.python.org/3/reference/expressions.html#slicings)
  specifies the frontend syntax. Use `slice.indices` on small concrete lengths
  when negative start/stop/step behavior is surprising.
- The Python [numeric operations reference](https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations)
  is the right baseline when signs make floor division and modulo unintuitive.
- TVM's [TensorIR creation tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
  provides a second example of making loops and buffer accesses explicit. Learn
  the general distinction; do not copy its IR names into tinygrad.

## Deliberate deferrals

This chapter establishes scalar coordinate correctness. It deliberately does
not yet teach:

- how optimizations reshape iteration axes into local, group, upcast, or
  vectorized ranges;
- coalescing, shared/local memory, tensor cores, occupancy, or GPU launch
  tuning;
- image-texture coordinate constraints and backend-specific address types;
- the complete advanced-index one-hot and advanced-assignment machinery;
- multi-device sharding coordinate transformations; or
- how a renderer spells every final pointer expression.

Those topics become manageable only after shape, iteration space, access
function, and validity are separate in your head. Chapter 9 now takes the same
index-aware loop representation and asks how to transform it for faster
kernels without changing its meaning.

[← Scheduling and realization](07-scheduling.md) · [Next: Kernel optimization →](09-kernel-optimization.md)
