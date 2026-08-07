"""Compare one scheduled kernel with its lowered and linear program forms."""

from collections import Counter

from tinygrad import Device, Tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import NOOPT
from tinygrad.uop.ops import AxisType, Ops, UOp


MARKERS = (Ops.RANGE, Ops.REDUCE, Ops.LOAD, Ops.SPECIAL)


def marker_counts(root: UOp) -> list[tuple[str, int]]:
  counts = Counter(node.op for node in root.toposort())
  return [(op.name, counts[op]) for op in MARKERS]


def extent(axis: UOp) -> int:
  value = axis.src[0].ssimplify()
  assert isinstance(value, int), value
  return value


def evaluate_address(address: UOp, row_axis: UOp, column_axis: UOp, row: int, column: int) -> int:
  value = address.substitute({row_axis: row_axis.const_like(row), column_axis: column_axis.const_like(column)}).ssimplify()
  assert isinstance(value, int), value
  return value


def main() -> None:
  device = Device.DEFAULT

  # This is the same calculation derived as nested loops in Chapter 10.
  x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32, device=device).realize()
  out = (x*x + 2*x).sum(axis=1)

  # Planning mutates out.uop to its planned buffer form. Keep and execute the
  # returned LINEAR plan; do not ask the mutated wrapper to plan the old work again.
  linear, var_vals = out.linear_with_vars()
  assert len(linear.src) == 1 and linear.src[0].op is Ops.CALL
  scheduled = linear.src[0].src[0]
  assert scheduled.op is Ops.SINK

  compiled = compile_linear(linear, beam=0)
  programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(programs) == 1
  program = programs[0]
  # Inspect the same lowered SINK that this PROGRAM will execute. Text/source
  # renderers preserve it as child zero; native-ISA routes may lower it further.
  lowered = program.src[0]
  scheduled_markers, lowered_markers = marker_counts(scheduled), marker_counts(lowered)

  print("target:", program.arg.target)
  print("calls/variables:", len(compiled.src), var_vals)
  print("applied opts:", lowered.arg.applied_opts)
  print("scheduled markers:", scheduled_markers)
  print("lowered markers:", lowered_markers)

  # The chapter's detailed structural assertions deliberately describe only
  # its controlled target. Other renderers can use different legal forms; for
  # those runs the probe still reports markers, launch metadata, and the result.
  controlled = bool(NOOPT) and str(program.arg.target) == "CUDA:PYTHON:sm_89"
  if controlled:
    accumulators = [node for node in lowered.toposort() if node.op is Ops.BUFFER and node.addrspace is AddrSpace.REG]
    global_axes = [node for node in lowered.toposort() if node.op is Ops.SPECIAL and node.arg == "gidx0"]
    reduction_axes = [node for node in lowered.toposort()
                      if node.op is Ops.RANGE and node.arg[-1] is AxisType.REDUCE]
    assert len(accumulators) == len(global_axes) == len(reduction_axes) == 1
    accumulator, global_axis, reduction_axis = accumulators[0], global_axes[0], reduction_axes[0]

    input_indices = [node for node in lowered.toposort()
                     if node.op is Ops.INDEX and node.src[0].op is Ops.PARAM and node.src[0].arg.slot == 1]
    assert len(input_indices) == 1
    input_address = input_indices[0].src[1]
    address_samples = [((row, column), evaluate_address(input_address, global_axis, reduction_axis, row, column))
                       for row in range(2) for column in range(3)]
    barriers = sum(node.op is Ops.BARRIER for node in lowered.toposort())

    ordered = next(node for node in program.src if node.op is Ops.LINEAR)
    controls = [(node.op.name,
                 node.arg if node.op is Ops.SPECIAL else node.arg[-1].name if node.op is Ops.RANGE else None)
                for node in ordered.src if node.op in (Ops.SPECIAL, Ops.RANGE, Ops.END)]

    assert scheduled_markers == [("RANGE", 2), ("REDUCE", 1), ("LOAD", 0), ("SPECIAL", 0)]
    assert lowered_markers == [("RANGE", 1), ("REDUCE", 0), ("LOAD", 3), ("SPECIAL", 1)]
    assert lowered.arg.applied_opts == () and barriers == 0
    assert accumulator.arg.dtype.name == "float" and accumulator.max_numel() == 1
    assert extent(global_axis) == 2 and extent(reduction_axis) == 3
    assert address_samples == [((0, 0), 0), ((0, 1), 1), ((0, 2), 2),
                               ((1, 0), 3), ((1, 1), 4), ((1, 2), 5)]
    assert program.arg.global_size == (2, 1, 1) and program.arg.local_size == (1, 1, 1)
    assert controls == [("SPECIAL", "gidx0"), ("RANGE", "REDUCE"), ("END", None)]

    print("accumulator:", accumulator.addrspace.name, accumulator.arg.dtype.name, accumulator.max_numel())
    print("global axis:", global_axis.arg, extent(global_axis))
    print("remaining loop:", reduction_axis.arg[-1].name, extent(reduction_axis))
    print("input addresses:", address_samples)
    print("barriers:", barriers)

  print("launch global/local:", program.arg.global_size, program.arg.local_size)
  if controlled: print("linear control:", controls)

  # The calls are already compiled, so jit=True prevents run_linear from
  # compiling the plan a second time.
  run_linear(compiled, var_vals, jit=True)
  result = out.tolist()
  print("result:", result)
  assert result == [26.0, 107.0]


if __name__ == "__main__":
  main()
