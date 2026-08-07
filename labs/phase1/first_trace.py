"""Trace one lazy expression through planning, compilation, and execution."""

from collections import Counter

from tinygrad import Device, GlobalCounters, Tensor
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.uop.ops import Ops


def call_summaries(linear):
  """Return stable fields from each CALL without dumping the full UOp graph."""
  return [
    (call.op.name, call.src[0].op.name, call.device, len(call.src) - 1)
    for call in linear.src
  ]


assert Device.DEFAULT == "PYTHON", "this portable teaching trace expects DEV=PYTHON"

x = Tensor([-2.0, -1.0, 0.0, 1.0])
y = (x * 2 + 1).relu()

frontend = y.uop
nodes = list(frontend.toposort())
node_id = {u: f"N{i}" for i, u in enumerate(nodes)}
op_counts = Counter(u.op.name for u in nodes)
print("device:", Device.DEFAULT)
print("frontend:", frontend.op.name, "realized=", frontend.is_realized)
print("frontend op counts:", dict(sorted(op_counts.items())))
print("frontend graph:")
for u in nodes:
  sources = ",".join(node_id[s] for s in u.src)
  detail = f" arg={u.arg}" if u.op is Ops.CONST else ""
  print(f"  {node_id[u]} {u.op.name} <- [{sources}]{detail}")
print("shared ADD/zero:", frontend.src[1] is frontend.src[0].src[1], frontend.src[2] is frontend.src[0].src[0])

linear, var_vals = y.linear_with_vars()
planned = call_summaries(linear)
scheduled_sink = linear.src[0].src[0]
print("planned tensor:", y.uop.op.name, "realized=", y.uop.is_realized)
print("planned calls:", planned)
print("planned body ops:", dict(sorted(Counter(u.op.name for u in scheduled_sink.toposort()).items())))

compiled = compile_linear(linear, beam=0)
compiled_calls = call_summaries(compiled)
print("compiled calls:", compiled_calls)

program = next(call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM)
children = [u.op.name for u in program.src]
source = next(u for u in program.src if u.op is Ops.SOURCE)
binary = next(u for u in program.src if u.op is Ops.BINARY)
print("program children:", children)
print("program buffer roles: outs=", program.arg.outs, "ins=", program.arg.ins)
print("program payload types:", type(source.arg).__name__, type(binary.arg).__name__)
print("before execution:", y.uop.is_realized)

# compile_linear already replaced SINK bodies with PROGRAM bodies. jit=True tells
# run_linear to dispatch this exact compiled plan rather than compiling it again.
GlobalCounters.reset()
run_linear(compiled, var_vals, jit=True)
print("after execution:", y.uop.is_realized)
print("execution calls:", GlobalCounters.kernel_count)
print("value:", y.tolist())

assert op_counts["MUL"] == op_counts["ADD"] == op_counts["CMPLT"] == op_counts["WHERE"] == 1
assert planned == [("CALL", "SINK", "PYTHON", 2)]
assert compiled_calls == [("CALL", "PROGRAM", "PYTHON", 2)]
assert children == ["SINK", "LINEAR", "SOURCE", "BINARY"]
assert program.arg.outs == (0,) and program.arg.ins == (1,)
assert isinstance(source.arg, str) and isinstance(binary.arg, bytes)
assert GlobalCounters.kernel_count == 1
assert y.tolist() == [0.0, 0.0, 1.0, 3.0]

# Compare the same semantics with and without an explicit intermediate
# realization. The input is warmed before the counters are reset so only the
# expression under study contributes to these counts.
x2 = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()

GlobalCounters.reset()
fused = (x2 * 2 + 1).relu().realize()
fused_count = GlobalCounters.kernel_count
print("fused calls/value:", fused_count, fused.tolist())

GlobalCounters.reset()
barrier = (x2 * 2 + 1).realize()
split = barrier.relu().realize()
barrier_count = GlobalCounters.kernel_count
print("barrier calls/value:", barrier_count, split.tolist())

assert fused_count == 1
assert barrier_count == 2
assert fused.tolist() == split.tolist() == [0.0, 0.0, 1.0, 3.0]
