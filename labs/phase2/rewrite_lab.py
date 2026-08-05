"""A guarded UPat rewrite with positive, negative, dtype, and shape tests."""

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite


def remove_integer_add_zero(add, x, zero):
  """Return x only when removing zero preserves this ADD's dtype and shape."""
  if add.dtype not in dtypes.ints: return None
  if zero.base.op is not Ops.CONST or zero.base.val != 0: return None
  if x.dtype != add.dtype or x.shape != add.shape: return None
  return x


integer_add_zero = PatternMatcher([
  (UPat(Ops.ADD, name="add", src=[UPat.var("x"), UPat.var("zero")]), remove_integer_add_zero),
])


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

  def test_broadcast_shape_is_not_collapsed(self):
    scalar = UOp.variable("scalar", -8, 8, dtypes.int32)
    zero_matrix = UOp.const(0, dtypes.int32).expand(2, 3)
    expr = scalar + zero_matrix
    self.assertEqual(expr.shape, (2, 3))
    out = rewrite(expr)
    self.assertIs(out.op, Ops.ADD)
    self.assertEqual(out.shape, (2, 3))

  def test_semantics_on_portable_backend(self):
    values = Tensor([1, -2, 3], device="PYTHON", dtype=dtypes.int32)
    before = values + 0
    after = Tensor(rewrite(before.uop))
    self.assertEqual(after.tolist(), before.tolist())


class TestDriverBehavior(unittest.TestCase):
  def test_first_successful_rule_wins(self):
    x = UOp.variable("x", -8, 8, dtypes.int32)
    matcher = PatternMatcher([
      (UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", arg=0))), lambda a, c: a),
      (UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", arg=0))), lambda a, c: UOp.const(99, a.dtype)),
    ])
    self.assertIs(graph_rewrite(x + 0, matcher), x)

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
