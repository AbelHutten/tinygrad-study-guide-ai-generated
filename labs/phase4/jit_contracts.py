"""Exercise TinyJit's current input, Python, output, and mutation contracts.

The observations are pinned to the study guide's tinygrad snapshot.  This is a
deterministic DEV=PYTHON source-study lab, not a performance benchmark.
"""

from __future__ import annotations

import os
from collections.abc import Callable


if not __debug__:
  raise RuntimeError("this teaching lab uses assertions; run Python without -O or PYTHONOPTIMIZE")

for key, value in {
  "ASSERT_COMPILE": "0",
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "1",
  "CCACHE": "0",
  "DEBUG": "0",
  "DEV": "PYTHON",
  "GRAPH_ONE_KERNEL": "0",
  "HCQ2": "0",
  "IGNORE_JIT_FIRST_BEAM": "0",
  "IMAGE": "0",
  "JIT": "1",
  "JITBEAM": "0",
  "JIT_BATCH_SIZE": "32",
  "NO_COLOR": "1",
  "NO_MEMORY_PLANNER": "0",
  "NOOPT": "1",
  "PROFILE": "0",
  "SCACHE": "0",
  "SPEC": "2",
  "TC": "0",
  "THREADS": "1",
  "UNSAFE_ALLOW_JIT_BUFFER": "0",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, TinyJit, Variable, dtypes  # noqa: E402
import tinygrad.engine.jit as jit_engine  # noqa: E402
from tinygrad.engine.jit import JitError  # noqa: E402
from tinygrad.uop.ops import UOp  # noqa: E402


def tensor(values, dtype=dtypes.float32) -> Tensor:
  return Tensor(values, device=Device.DEFAULT, dtype=dtype).realize()


def expect_jit_error(label: str, fragment: str, action: Callable[[], object]) -> None:
  """Require one specific TinyJit rejection; unrelated exceptions must escape."""
  try:
    action()
  except JitError as error:
    assert fragment in str(error), (label, str(error))
    print(f"expected failure — {label}: {error}")
  else:
    raise AssertionError(f"{label}: expected JitError containing {fragment!r}")


def show_realization_boundary() -> None:
  @TinyJit
  def twice(x: Tensor) -> Tensor:
    return (x * 2.0).realize()

  lazy = tensor([1.0, 2.0]) + 1.0
  before = (lazy.uop.is_virtual, lazy.uop.is_realized)
  assert before == (False, False)
  result = twice(lazy).tolist()
  after = (lazy.uop.is_virtual, lazy.uop.is_realized)
  assert result == [4.0, 6.0] and after == (False, True)
  print("ordinary lazy input (virtual, realized) before/after:", before, after, "result:", result)

  @TinyJit
  def add_one(x: Tensor) -> Tensor:
    return (x + 1.0).realize()

  virtual = Tensor(UOp.const(2.0).cast(dtypes.float32))
  assert virtual.uop.is_virtual
  expect_jit_error("device-less virtual UOp", "JIT inputs must be real buffers", lambda: add_one(virtual))


def show_shallow_containers() -> None:
  @TinyJit
  def add_from_list(x: Tensor, payload: list[Tensor]) -> Tensor:
    return (x + payload[0]).realize()

  values = []
  for value in (1.0, 2.0, 3.0):
    values.append(add_from_list(tensor([10.0]), [tensor([value])]).tolist())
  assert values == [[11.0], [12.0], [13.0]]
  assert add_from_list.captured is not None
  assert add_from_list.captured.expected_names == [0]
  assert len(add_from_list.captured.expected_input_info) == 2
  print("one-level list tensor values:", values, "captured tensor inputs:", 2)

  @TinyJit
  def nested_container(payload: list[list[Tensor]]) -> Tensor:
    return (payload[0][0] * 2.0).realize()

  nested_values = [nested_container([[tensor([value])]]).tolist() for value in (1.0, 2.0, 3.0)]
  assert nested_values == [[2.0], [4.0], [4.0]]
  assert nested_container.captured is not None
  assert nested_container.captured.expected_names == []
  assert nested_container.captured.expected_input_info == []
  print("nested list values (replay freezes capture tensor):", nested_values)

  @TinyJit
  def subtract_boxes(left: list[Tensor], right: list[Tensor]) -> Tensor:
    return (left[0] - right[0]).realize()

  ordered = [
    subtract_boxes(left=[tensor([10.0])], right=[tensor([1.0])]).tolist(),
    subtract_boxes(left=[tensor([10.0])], right=[tensor([1.0])]).tolist(),
  ]
  reordered = subtract_boxes(right=[tensor([2.0])], left=[tensor([20.0])]).tolist()
  assert ordered == [[9.0], [9.0]]
  assert reordered == [-18.0]  # Python execution would compute 20 - 2 == 18.
  assert subtract_boxes.captured is not None and subtract_boxes.captured.expected_names == []
  print("reordered container-valued kwargs (replay actual, Python result):", reordered, [18.0])
  print("documented footgun: shallow container tensors are unnamed and follow caller insertion order")


def show_frozen_python() -> None:
  python_calls = 0

  @TinyJit
  def choose(x: Tensor, square: bool) -> Tensor:
    nonlocal python_calls
    python_calls += 1
    if square:
      return (x * x).realize()
    return (x * 2.0).realize()

  branch_values = [
    choose(tensor([3.0]), True).tolist(),
    choose(tensor([3.0]), False).tolist(),
    choose(tensor([3.0]), True).tolist(),
  ]
  assert branch_values == [[9.0], [6.0], [6.0]] and python_calls == 2
  print("Python bool branches (ignore, capture, replay):", branch_values, "python_calls:", python_calls)

  scale = 1.0

  @TinyJit
  def scale_from_closure(x: Tensor) -> Tensor:
    return (x * scale).realize()

  closure_values = []
  for scale in (1.0, 2.0, 3.0):
    closure_values.append(scale_from_closure(tensor([10.0])).tolist())
  assert closure_values == [[10.0], [20.0], [20.0]]
  print("closure scalar values (ignore, capture, replay):", closure_values)


def show_contract_rejections() -> None:
  @TinyJit
  def add(a: Tensor, b: Tensor) -> Tensor:
    return (a + b).realize()

  base = tensor([1.0, 2.0, 3.0])
  assert base[:2].uop.base is base[1:].uop.base
  expect_jit_error("two views alias one base buffer", "duplicate inputs to JIT", lambda: add(base[:2], base[1:]))

  @TinyJit
  def one_input(x: Tensor) -> Tensor:
    return (x + 1).realize()

  one_input(tensor([1.0, 2.0]))
  one_input(tensor([3.0, 4.0]))
  expect_jit_error(
    "dtype changed after capture",
    "args mismatch in JIT",
    lambda: one_input(tensor([5, 6], dtype=dtypes.int32)),
  )

  @TinyJit
  def fixed_view(x: Tensor) -> Tensor:
    return (x + 1.0).realize()

  view_base = tensor([10.0, 20.0, 30.0])
  fixed_view(view_base[:2])
  fixed_view(view_base[:2])
  expect_jit_error("view offset changed after capture", "args mismatch in JIT", lambda: fixed_view(view_base[1:]))

  @TinyJit
  def calling_convention(a: Tensor, b: Tensor) -> Tensor:
    return (a + b).realize()

  calling_convention(tensor([1.0]), tensor([2.0]))
  calling_convention(tensor([3.0]), tensor([4.0]))
  expect_jit_error(
    "positional inputs changed to keyword inputs",
    "args mismatch in JIT",
    lambda: calling_convention(a=tensor([5.0]), b=tensor([6.0])),
  )

  @TinyJit
  def bad_return(x: Tensor):
    return (x + 1.0).realize(), 7

  bad_return(tensor([1.0]))  # Return validation happens when the second call captures.
  expect_jit_error("Python integer in returned tuple", "non-Tensor value of type int", lambda: bad_return(tensor([2.0])))

  @TinyJit
  def host_read_during_capture(x: Tensor) -> Tensor:
    return (x + x.item()).realize()

  host_read_during_capture(tensor([2.0]))
  expect_jit_error(
    ".item() attempts a host read during capture",
    "cannot access tensor data during JIT capture",
    lambda: host_read_during_capture(tensor([3.0])),
  )


def show_mutation_and_lifecycle() -> None:
  @TinyJit
  def increment(x: Tensor) -> Tensor:
    x += 1.0
    return x.realize()

  state = tensor([0.0])
  mutation_values = [increment(state).item() for _ in range(3)]
  assert mutation_values == [1.0, 2.0, 3.0]
  assert increment.captured is not None and increment.captured._written_uops == set()
  print("read/write input mutation across ignore, capture, replay:", mutation_values)
  print("pure-write set excludes a buffer that is both read and written:", len(increment.captured._written_uops))

  increment.reset()
  assert increment.cnt == 0 and increment.captured is None
  after_reset = increment(state).item()
  assert after_reset == 4.0 and increment.cnt == 1
  print("reset removed the capture; next call was a new ignore call:", after_reset)

  @TinyJit
  def add_one(x: Tensor) -> Tensor:
    return (x + 1.0).realize()

  assert add_one(tensor([0.0])).tolist() == [1.0]       # ignore
  captured_output = add_one(tensor([10.0]))             # capture
  assert captured_output.tolist() == [11.0]
  assert add_one.captured is not None
  assert captured_output.uop.base in add_one.captured._written_uops

  copied_inputs: list[UOp] = []
  original_copy_input = jit_engine._copy_input

  def observe_copy(input_uop: UOp) -> UOp:
    copied_inputs.append(input_uop)
    return original_copy_input(input_uop)

  # Feed the stored output back as the next explicit input.  It aliases a
  # pure-output buffer in the captured plan, so CapturedJit must copy it before
  # the same plan overwrites that persistent output buffer.
  jit_engine._copy_input = observe_copy
  try:
    feedback_result = add_one(captured_output).tolist()
    copies_after_feedback = len(copied_inputs)
    # A fresh, non-aliasing input does not match the captured pure-write set.
    fresh_result = add_one(tensor([20.0])).tolist()
  finally:
    jit_engine._copy_input = original_copy_input

  assert feedback_result == [12.0]
  assert copied_inputs == [captured_output.uop.base]
  assert copies_after_feedback == 1
  assert fresh_result == [21.0] and len(copied_inputs) == copies_after_feedback
  print("pure-output fed back as input; replay result:", feedback_result)
  print("_copy_input calls for the aliased feedback input:", len(copied_inputs))
  print("fresh replay input result; additional _copy_input calls:", fresh_result, 0)

  add_one.captured.free_intermediates()
  after_free = add_one(tensor([30.0])).tolist()
  assert after_free == [31.0]
  print("replay after free_intermediates reallocated what it needed:", after_free)


def show_prune_reachability() -> None:
  weight = tensor([10.0])
  scale = Variable("jit_contract_scale", 1, 9)

  @TinyJit(prune=True)
  def scale_weight(x: Tensor, bound_scale: UOp) -> Tensor:
    prepared = (weight * bound_scale).contiguous().realize()
    return (prepared + x).realize()

  values = [
    scale_weight(tensor([1.0]), scale.bind(1)).tolist(),
    scale_weight(tensor([1.0]), scale.bind(2)).tolist(),
    scale_weight(tensor([1.0]), scale.bind(3)).tolist(),
  ]
  assert values == [[11.0], [21.0], [21.0]]
  print("prune with changing symbolic-only preprocessing (ignore, capture, replay):", values)
  print("documented footgun: prune reachability starts from Tensor input buffers, not var_vals")


def main() -> None:
  assert Device.DEFAULT == "PYTHON"
  print("controlled route: DEV=PYTHON JIT=1; this lab makes no device-graph or accelerator-timing claim")
  show_realization_boundary()
  show_shallow_containers()
  show_frozen_python()
  show_contract_rejections()
  show_mutation_and_lifecycle()
  show_prune_reachability()
  print("claim: all printed observations match pinned source semantics on controlled DEV=PYTHON")
  print("non-claim: footgun outputs are observations, not recommended API contracts or accelerator evidence")
  print("all TinyJit contract observations passed")


if __name__ == "__main__":
  main()
