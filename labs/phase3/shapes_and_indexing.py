"""Trace shape and movement semantics from Tensor values to RANGE/INDEX algebra."""

from tinygrad import Tensor, dtypes
from tinygrad.schedule.indexing import IndexingContext, apply_movement_op, broadcast_rngs
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp


def movement_chain(root: UOp) -> list[str]:
  """Return movement operations from a view root back toward its first data source."""
  chain: list[str] = []
  node = root
  while node.op in GroupOp.Movement:
    chain.append(node.op.name)
    node = node.src[0]
  return chain


def map_to_source(root: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...], list[tuple[str, tuple, tuple[str, ...]]]]:
  """Apply each movement node in reverse, mapping output ranges to source coordinates."""
  output_ranges = tuple(UOp.range(size, axis, AxisType.WEAK) for axis, size in enumerate(root.shape))
  coordinates, node = output_ranges, root
  steps: list[tuple[str, tuple, tuple[str, ...]]] = []
  while node.op in GroupOp.Movement:
    coordinates = apply_movement_op(node.op, node.src[0].shape, node.marg, coordinates)
    steps.append((node.op.name, node.src[0].shape, tuple(coord.render() for coord in coordinates)))
    node = node.src[0]
  return output_ranges, coordinates, steps


def evaluate_coordinates(output_ranges: tuple[UOp, ...], source_coordinates: tuple[UOp, ...],
                         output_coordinate: tuple[int, ...]) -> tuple[int, ...]:
  replacements = {rng: UOp.const(value) for rng, value in zip(output_ranges, output_coordinate)}
  return tuple(int(coord.substitute(replacements).ssimplify()) for coord in source_coordinates)


def main() -> None:
  # ----- Visible Tensor semantics -----
  base = Tensor([[0, 1, 2], [3, 4, 5]], device="PYTHON", dtype=dtypes.int32)
  view = base.reshape(3, 2).permute(1, 0)
  expanded = view.reshape(1, 2, 3).expand(4, 2, 3)
  padded = view.pad(((1, 1), (1, 1)))
  reversed_columns = base[:, ::-1]
  stride_two_columns = base[:, ::2]

  base_values = base.tolist()
  view_values = view.tolist()
  first_expanded, last_expanded = expanded[0].tolist(), expanded[3].tolist()
  padded_values = padded.tolist()
  reversed_values = reversed_columns.tolist()
  stride_two_values = stride_two_columns.tolist()

  print("base shape/values:       ", base.shape, base_values)
  print("view shape/values:       ", view.shape, view_values)
  print("expanded shape:          ", expanded.shape)
  print("expanded batches equal:  ", first_expanded == last_expanded)
  print("padded shape/values:     ", padded.shape, padded_values)
  print("reverse columns:         ", reversed_values)
  print("stride-two columns:      ", stride_two_values)

  assert base_values == [[0, 1, 2], [3, 4, 5]]
  assert view_values == [[0, 2, 4], [1, 3, 5]]
  assert first_expanded == last_expanded == view_values
  assert padded_values == [[0, 0, 0, 0, 0], [0, 0, 2, 4, 0], [0, 1, 3, 5, 0], [0, 0, 0, 0, 0]]
  assert reversed_values == [[2, 1, 0], [5, 4, 3]]
  assert stride_two_values == [[0, 2], [3, 5]]

  # Basic indexing is normalized through movement operations. Advanced list/Tensor
  # indexing constructs selection algebra and has different out-of-bounds behavior.
  basic = base[1]
  rows = Tensor([1, 0], device="PYTHON", dtype=dtypes.int32)
  cols = Tensor([2, 1], device="PYTHON", dtype=dtypes.int32)
  advanced = base[rows, cols]
  advanced_oob = base[[3, -4]]
  basic_root, advanced_root = basic.uop.op.name, advanced.uop.op.name
  try:
    _ = base[3]
  except IndexError:
    basic_oob = "IndexError"
  else:  # pragma: no cover - this is a pinned behavioral assertion
    raise AssertionError("basic out-of-bounds indexing should raise")
  print("basic/advanced roots:    ", basic_root, advanced_root)
  print("basic/advanced values:   ", basic.tolist(), advanced.tolist())
  print("basic/advanced OOB:      ", basic_oob, advanced_oob.tolist())
  assert basic.tolist() == [3, 4, 5] and advanced.tolist() == [5, 1]
  assert advanced_oob.tolist() == [[0, 0, 0], [0, 0, 0]]

  # ----- Compose the carried movement map -----
  source = UOp.param(0, dtypes.int32, shape=(2, 3), device="PYTHON", name="base")
  raw_view = source.reshape(3, 2).permute(1, 0)
  output_ranges, source_coordinates, steps = map_to_source(raw_view)

  print("\nview movement chain:     ", movement_chain(raw_view))
  for op, source_shape, coordinates in steps:
    print(f"{op:8s} -> source shape {source_shape}: {coordinates}")

  source_flat_address = (source_coordinates[0] * 3 + source_coordinates[1]).simplify()
  print("source flat address:     ", source_flat_address.render())
  expected_maps = {
    (0, 0): (0, 0),
    (0, 2): (1, 1),
    (1, 0): (0, 1),
    (1, 2): (1, 2),
  }
  for output_coordinate, expected_source in expected_maps.items():
    observed_source = evaluate_coordinates(output_ranges, source_coordinates, output_coordinate)
    print("view coordinate", output_coordinate, "-> base coordinate", observed_source)
    assert observed_source == expected_source
  assert source_flat_address.render() == "(r1*2+r0)"

  # ----- Implicit broadcasting versus explicit EXPAND -----
  column = UOp.param(0, dtypes.float32, shape=(2, 1), device="PYTHON", name="column")
  row = UOp.param(1, dtypes.float32, shape=(1, 3), device="PYTHON", name="row")
  product = column * row
  product_ranges = tuple(UOp.range(size, axis, AxisType.WEAK) for axis, size in enumerate(product.shape))
  column_coordinates = tuple(coord.render() for coord in broadcast_rngs(product, column, product_ranges))
  row_coordinates = tuple(coord.render() for coord in broadcast_rngs(product, row, product_ranges))
  print("\nimplicit broadcast:      ", column.shape, row.shape, "->", product.shape)
  print("implicit source maps:    ", column_coordinates, row_coordinates)
  print("implicit has EXPAND:     ", any(node.op is Ops.EXPAND for node in product.toposort()))
  assert product.shape == (2, 3)
  assert column_coordinates == ("r0", "0") and row_coordinates == ("0", "r1")
  assert not any(node.op is Ops.EXPAND for node in product.toposort())

  raw_expand = column.expand(2, 3)
  _, expand_source_coordinates, expand_steps = map_to_source(raw_expand)
  print("explicit expand chain:   ", movement_chain(raw_expand))
  for op, source_shape, coordinates in expand_steps:
    print(f"{op:8s} -> source shape {source_shape}: {coordinates}")
  print("explicit final map:      ", tuple(coord.render() for coord in expand_source_coordinates))
  assert tuple(coord.render() for coord in expand_source_coordinates) == ("r0", "0")

  singleton_context = IndexingContext()
  singleton_range = singleton_context.new_range(1)
  active_range = singleton_context.new_range(3)
  print("singleton/active axes:   ", singleton_range.op.name, singleton_range.render(),
        active_range.op.name, active_range.render(), active_range.vmin, active_range.vmax)
  assert singleton_range.op is Ops.CONST and singleton_range.val == 0
  assert active_range.op is Ops.RANGE and (active_range.vmin, active_range.vmax) == (0, 2)

  # ----- PAD addresses carry a validity predicate -----
  pad_root = source.pad(((1, 1), (2, 1)))
  pad_ranges = tuple(UOp.range(size, axis, AxisType.WEAK) for axis, size in enumerate(pad_root.shape))
  pad_coordinates = apply_movement_op(Ops.PAD, source.shape, pad_root.marg, pad_ranges)
  print("\npad public/internal:     ", ((1, 1), (2, 1)), pad_root.marg, pad_root.shape)
  for axis, coordinate in enumerate(pad_coordinates):
    print("pad axis", axis, "idx", coordinate.get_idx().render(), "valid", coordinate.get_valid().render())

  expected_pad_maps = {
    (0, 0): ((-1, False), (-2, False)),
    (1, 2): ((0, True), (0, True)),
    (2, 4): ((1, True), (2, True)),
    (3, 5): ((2, False), (3, False)),
  }
  for output_coordinate, expected in expected_pad_maps.items():
    replacements = {rng: UOp.const(value) for rng, value in zip(pad_ranges, output_coordinate)}
    observed = tuple((int(coord.get_idx().substitute(replacements).ssimplify()),
                      bool(coord.get_valid().substitute(replacements).ssimplify())) for coord in pad_coordinates)
    print("pad coordinate", output_coordinate, "->", observed)
    assert observed == expected

  nonzero_pad = source.pad(((1, 1), (2, 1)), value=-1.0)
  print("zero/nonzero pad roots:  ", pad_root.op.name, nonzero_pad.op.name)
  assert pad_root.op is Ops.PAD and nonzero_pad.op is Ops.WHERE

  # ----- Inspect one rangeified plan -----
  # schedule_linear mutates connected Tensor wrappers. Use a fresh expression and
  # finish with the returned plan rather than trying to realize the planned Tensor.
  plan_base = Tensor([[0, 1, 2], [3, 4, 5]], device="PYTHON", dtype=dtypes.int32)
  planned = (plan_base.reshape(3, 2).permute(1, 0) + 10).schedule_linear()
  assert planned.op is Ops.LINEAR and len(planned.src) == 1
  body = planned.src[0].src[0]
  stores = [node for node in body.toposort() if node.op is Ops.STORE]
  assert len(stores) == 1
  store = stores[0]
  output_index = store.src[0].src[1]
  input_indices = [node.src[1] for node in store.src[1].toposort() if node.op is Ops.INDEX]
  assert len(input_indices) == 1
  input_index = input_indices[0]
  ranges = sorted((node for node in body.toposort() if node.op is Ops.RANGE), key=lambda rng: rng.arg[0])

  print("\nrangeified ranges:       ", [(rng.render(), rng.vmin, rng.vmax) for rng in ranges])
  print("rangeified output index: ", output_index.render())
  print("rangeified input index:  ", input_index.render())
  assert [(rng.render(), rng.vmin, rng.vmax) for rng in ranges] == [("r0", 0, 1), ("r1", 0, 2)]
  assert output_index.render() == "(r0*3+r1)"
  assert input_index.render() == "(r1*2+r0)"

  # ----- Bounds and divisibility justify symbolic rewrites -----
  symbolic_row = UOp.variable("row", 0, 7)
  col = UOp.variable("col", 0, 15)
  flat = symbolic_row * 16 + col
  recovered_row, recovered_col = (flat // 16).simplify(), (flat % 16).simplify()
  print("\nsymbolic flat/bounds:    ", flat.render(), flat.vmin, flat.vmax)
  print("recover row/column:      ", recovered_row.render(), recovered_col.render())
  assert recovered_row is symbolic_row and recovered_col is col

  wide_col = UOp.variable("wide_col", 0, 16)
  insufficiently_bounded = (symbolic_row * 16 + wide_col) // 16
  insufficiently_bounded = insufficiently_bounded.simplify()
  print("insufficient bound:      ", insufficiently_bounded.render())
  assert insufficiently_bounded is not symbolic_row

  n = UOp.variable("n", 8, 64, multiple_of=8)
  quotient, remainder = (n // 8).simplify(), (n % 8).simplify()
  bound = n.bind(24)
  print("multiple-of proof:       ", remainder.render(), quotient.vmin, quotient.vmax)
  print("valid binding:           ", bound.src[1].val)
  assert remainder.op is Ops.CONST and remainder.val == 0
  assert (quotient.vmin, quotient.vmax) == (1, 8) and bound.src[1].val == 24
  for invalid_binding in (26, 72):
    try: n.bind(invalid_binding)
    except AssertionError: pass
    else: raise AssertionError(f"binding {invalid_binding} should fail")

  floor_examples = []
  for numerator, denominator in ((-7, 3), (7, -3)):
    floor_quotient = (UOp.const(numerator) // denominator).simplify().val
    floor_remainder = (UOp.const(numerator) % denominator).simplify().val
    assert floor_quotient * denominator + floor_remainder == numerator
    floor_examples.append((numerator, denominator, floor_quotient, floor_remainder))
  print("floor div/mod examples:  ", floor_examples)

  assert all((r * 16 + c) // 16 == r and (r * 16 + c) % 16 == c for r in range(8) for c in range(16))
  print("exhaustive small oracle: ", "passed")


if __name__ == "__main__":
  main()
