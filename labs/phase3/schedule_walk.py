"""Inspect planning, fusion, ordering, and memory reuse without hiding execution."""

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.uop.ops import Ops, UOp


MATH_OPS = {Ops.ADD, Ops.MUL, Ops.REDUCE}


def math_ops(call: UOp) -> list[str]:
  """Return a compact dependency-first inventory, not a rendered instruction list."""
  return [u.op.name for u in call.src[0].toposort() if u.op in MATH_OPS]


def plan_math(boundary: bool) -> list[list[str]]:
  """Plan a fresh graph and return only the math inventory for each compute call."""
  x = Tensor.empty(3)
  mid = x*x + 2*x
  if boundary: mid = mid.contiguous()
  loss = mid.sum()
  linear = loss.schedule_linear()
  assert all(call.op is Ops.CALL and call.src[0].op is Ops.SINK for call in linear.src)
  return [math_ops(call) for call in linear.src]


def label_mutation_call(call: UOp) -> str:
  vals = [u.val for u in call.src[0].toposort() if u.op is Ops.CONST]
  if 10.0 in vals: return "read old (+10)"
  if 2.0 in vals: return "overwrite (*2)"
  if 100.0 in vals: return "read new (+100)"
  raise AssertionError(f"unrecognized mutation call constants: {vals}")


def fake_call(*bufs: UOp) -> UOp:
  # memory_plan_rewrite only needs a LINEAR whose calls name their buffers.
  return UOp(Ops.CALL, src=(UOp(Ops.SINK, src=bufs), *bufs))


# A virtual scalar needs no device call. Tensor.empty has a storage identity,
# but its bytes are deliberately unspecified and have not been allocated.
scalar_linear = Tensor(2.0).schedule_linear()
assert len(scalar_linear.src) == 0
print("scalar constant calls:", len(scalar_linear.src))

empty = Tensor.empty(3)
assert empty.uop.has_buffer_identity() and not empty.uop.is_realized
print("empty identity/allocated:", empty.uop.has_buffer_identity(), empty.uop.is_realized)

# Carry the same expression used in Chapters 4-6 through planning and execution.
x = Tensor([1.0, 2.0, 3.0]).realize()
loss = (x*x + 2*x).sum()
old_loss = loss.uop
assert old_loss.op is Ops.REDUCE and not old_loss.has_buffer_identity() and not old_loss.is_realized
print("before:", old_loss.op.name, old_loss.has_buffer_identity(), old_loss.is_realized)

linear, var_vals = loss.linear_with_vars()
assert loss.uop is not old_loss
assert loss.uop.op is Ops.RESHAPE and loss.uop.has_buffer_identity() and not loss.uop.is_realized
assert len(linear.src) == 1 and var_vals == {}
assert linear.src[0].op is Ops.CALL and linear.src[0].src[0].op is Ops.SINK
assert math_ops(linear.src[0]) == ["MUL", "MUL", "ADD", "REDUCE"]
print("after planning:", loss.uop.op.name, loss.uop.has_buffer_identity(), loss.uop.is_realized)
print("calls/variables:", len(linear.src), var_vals)
print("call forms:", [f"{call.op.name}({call.src[0].op.name})" for call in linear.src])
print("math nodes:", math_ops(linear.src[0]))

run_linear(linear, var_vals)
assert loss.uop.is_realized and loss.item() == 26.0
print("after execution:", loss.uop.is_realized, loss.item())

# These helpers create fresh graphs. Planning mutates their Tensor wrappers, so
# reusing one graph for both alternatives would not be a valid comparison.
fused_math = plan_math(False)
materialized_math = plan_math(True)
assert fused_math == [["MUL", "MUL", "ADD", "REDUCE"]]
assert materialized_math == [["MUL", "MUL", "ADD"], ["REDUCE"]]
print("fused math by call:", fused_math)
print("materialized math by call:", materialized_math)

base = Tensor.empty(3)
same = base.contiguous()
same_linear = same.schedule_linear()
assert same.uop is base.uop and len(same_linear.src) == 0
print("contiguous BUFFER same UOp/calls:", same.uop is base.uop, len(same_linear.src))

# Versioned buffer states require the old reader to precede the overwrite and
# the new reader to follow it. Explicit boundaries keep the three calls visible.
x = Tensor([1.0]).contiguous().realize()
before = (x + 10).contiguous()
x.assign(x * 2)
after = (x + 100).contiguous()
linear, var_vals = before.linear_with_vars(x, after)
mutation_order = [label_mutation_call(call) for call in linear.src]
assert mutation_order == ["read old (+10)", "overwrite (*2)", "read new (+100)"]
print("mutation order:", mutation_order)

run_linear(linear, var_vals)
assert before.tolist() == [11.0] and x.tolist() == [2.0] and after.tolist() == [102.0]
print("mutation values:", before.tolist(), x.tolist(), after.tolist())

# Three synthetic temporary buffers make the lifetime calculation inspectable:
# A is used by calls 0-1, B by 1-2, and C only by call 3. Each 64-element
# float32 buffer occupies one 256-byte planner block in this snapshot.
a = UOp.new_buffer("NULL", 64, dtypes.float32)
b = UOp.new_buffer("NULL", 64, dtypes.float32)
c = UOp.new_buffer("NULL", 64, dtypes.float32)
unplanned = UOp(Ops.LINEAR, src=(fake_call(a), fake_call(a, b), fake_call(b), fake_call(c)))
planned = memory_plan_rewrite(unplanned)

views: dict[UOp, UOp] = {}
for old_call, new_call in zip(unplanned.src, planned.src):
  for old, new in zip(old_call.src[1:], new_call.src[1:]): views.setdefault(old, new)

assert all(views[buf].op is Ops.SLICE for buf in (a, b, c))
assert views[a].src[0] is views[b].src[0] is views[c].src[0]
offsets = {name: views[buf].src[1].val for name, buf in (("a", a), ("b", b), ("c", c))}
assert offsets == {"a": 0, "b": 256, "c": 0}
assert views[a].src[0].max_numel() == 512
print("arena offsets:", offsets)
print("arena bytes:", views[a].src[0].max_numel())
print("same arena:", views[a].src[0] is views[b].src[0] is views[c].src[0])
