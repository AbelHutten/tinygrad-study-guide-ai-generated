"""A guarded UPat rewrite with positive, negative, dtype, and shape tests."""

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite, track_rewrites


def remove_integer_add_zero(add, x, zero):
  """Return x only when removing zero preserves this ADD's dtype and shape."""
  if add.dtype not in dtypes.ints: return None
  if zero.base.op is not Ops.CONST or zero.base.val != 0: return None
  if x.dtype != add.dtype or x.shape != add.shape: return None
  return x


def return_root_for_first_permutation(root, x, zero):
  """Expose original-return behavior: stop this pattern's remaining bindings."""
  if x.base.op is Ops.CONST and x.base.val == 0: return root
  if zero.base.op is Ops.CONST and zero.base.val == 0: return x
  return None


integer_add_zero = PatternMatcher([
  (UPat(Ops.ADD, name="add", src=[UPat.var("x"), UPat.var("zero")]), remove_integer_add_zero),
])


@track_rewrites()
def rewrite(root):
  return graph_rewrite(root, integer_add_zero, name="lab integer add zero")


class TestIntegerAddZero(unittest.TestCase):
  def test_positive_both_orders(self):
    x = UOp.placeholder((2, 3), dtypes.int32, slot=0)
    self.assertIs(rewrite(x + 0), x)
    self.assertIs(rewrite(0 + x), x)

  def test_concrete_integer_dtypes(self):
    for slot, dtype in enumerate((dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.uint64)):
      with self.subTest(dtype=dtype):
        x = UOp.placeholder((2, 3), dtype, slot=slot)
        out = rewrite(x + 0)
        self.assertIs(out, x)
        self.assertEqual((out.dtype, out.shape), (dtype, (2, 3)))

  def test_expanded_zero_with_result_shape(self):
    x = UOp.placeholder((2, 3), dtypes.int32, slot=0)
    expanded_zero = UOp.const(0, dtypes.int32).expand(2, 3)
    self.assertIs(rewrite(x + expanded_zero), x)
    self.assertIs(rewrite(x + UOp.const(0, dtypes.int32).detach()), x)

  def test_nonzero_is_a_negative_case(self):
    x = UOp.placeholder((2, 3), dtypes.int32, slot=0)
    out = rewrite(x + 1)
    self.assertIs(out.op, Ops.ADD)

  def test_float_is_a_negative_case(self):
    # x + 0.0 is not blindly removed: IEEE signed-zero behavior needs an
    # explicit project-level decision, so this teaching rule stays integer-only.
    x = UOp.placeholder((2, 3), dtypes.float32, slot=0)
    out = rewrite(x + 0.0)
    self.assertIs(out.op, Ops.ADD)

  def test_bool_and_weak_integer_are_negative_cases(self):
    bool_x = UOp.placeholder((2,), dtypes.bool, slot=0)
    self.assertIs(rewrite(bool_x + False).op, Ops.ADD)

    weak_x = UOp.const(7)
    self.assertIs(weak_x.dtype, dtypes.weakint)
    self.assertIs(rewrite(weak_x + 0).op, Ops.ADD)

  def test_symbolic_bounds_do_not_make_a_literal(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    bounded_zero = UOp.variable("bounded_zero", 0, 0, dtypes.int32)
    self.assertEqual((bounded_zero.vmin, bounded_zero.vmax), (0, 0))
    self.assertIs(rewrite(x + bounded_zero).op, Ops.ADD)

  def test_broadcast_shape_is_not_collapsed(self):
    scalar = UOp.variable("scalar", -8, 8, dtypes.int32)
    zero_matrix = UOp.const(0, dtypes.int32).expand(2, 3)
    expr = scalar + zero_matrix
    self.assertEqual(expr.shape, (2, 3))
    out = rewrite(expr)
    self.assertIs(out.op, Ops.ADD)
    self.assertEqual(out.shape, (2, 3))

  def test_semantics_on_portable_backend(self):
    expected = [dtypes.int32.min, -1, 0, 1, dtypes.int32.max]
    values = Tensor(expected, device="PYTHON", dtype=dtypes.int32)
    # Build the candidate directly so the baseline does not pass through
    # tinygrad's broader upstream x+0 simplifier before this lab matcher runs.
    candidate = UOp(Ops.ADD, values.dtype, (values.uop, UOp.const(0, values.dtype)))
    rewritten = rewrite(candidate)
    self.assertIs(rewritten, values.uop)
    self.assertEqual(Tensor(rewritten).tolist(), expected)

  def test_nested_additions_reach_a_fixed_point(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    nested = (x + 0) + 0
    self.assertIs(rewrite(nested), x)
    self.assertIs(rewrite(rewrite(nested)), x)


class TestDriverBehavior(unittest.TestCase):
  def test_first_successful_rule_wins(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    matcher = PatternMatcher([
      (UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", arg=0))), lambda a, c: a),
      (UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", arg=0))), lambda a, c: UOp.const(99, a.dtype)),
    ])
    self.assertIs(graph_rewrite(x + 0, matcher), x)

  def test_none_allows_the_next_rule(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    pattern = UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", arg=0)))
    matcher = PatternMatcher([
      (pattern, lambda a, c: None),
      (pattern, lambda a, c: UOp.const(99, a.dtype)),
    ])
    self.assertIs(graph_rewrite(x + 0, matcher), UOp.const(99, x.dtype))

  def test_original_stops_bindings_but_allows_the_next_rule(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    matcher = PatternMatcher([
      (UPat(Ops.ADD, name="root", src=[UPat.var("x"), UPat.var("zero")]), return_root_for_first_permutation),
      (UPat(Ops.ADD, name="root"), lambda root: UOp.const(99, root.dtype)),
    ])
    # For 0+x, the first list permutation binds pattern x to zero. Returning
    # the original root skips the valid second permutation, then rule 2 runs.
    self.assertIs(graph_rewrite(0 + x, matcher), UOp.const(99, x.dtype))

  def test_greedy_reenters_a_replacement_but_walk_does_not(self):
    advancing = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(5, x.dtype)),
    ])
    self.assertIs(graph_rewrite(UOp.const(3), advancing), UOp.const(5))
    self.assertIs(graph_rewrite(UOp.const(3), advancing, walk=True), UOp.const(4))

  def test_cycle_is_rejected_but_walk_applies_once(self):
    # Adapted from tinygrad's MIT-licensed rewrite-cycle test; see
    # ../../THIRD_PARTY_NOTICES.md for the pinned source and attribution.
    bouncing = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(3, x.dtype)),
    ])
    with self.assertRaisesRegex(RuntimeError, "infinite loop"):
      graph_rewrite(UOp.const(3), bouncing)
    self.assertIs(graph_rewrite(UOp.const(3), bouncing, walk=True), UOp.const(4))


if __name__ == "__main__":
  unittest.main(verbosity=2)
