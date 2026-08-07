"""Observe TinyJit's ignore, capture, and replay states.

Run this file in separate processes with JIT=0, JIT=1, and JIT=2.  It defaults
to DEV=PYTHON when DEV is unset, but respects an explicit DEV so the repository
runner can still exercise this lab on selected hardware.
"""

from __future__ import annotations

import os


if not __debug__:
  raise RuntimeError("this teaching lab uses assertions; run Python without -O or PYTHONOPTIMIZE")

requested_jit = os.environ.get("JIT", "1")
if requested_jit not in {"0", "1", "2"}:
  raise RuntimeError(f"set JIT to 0, 1, or 2 for this lab, not {requested_jit!r}")
requested_device = os.environ.get("DEV") or "PYTHON"

# Fix the settings this structural lab inspects.  This is still not a hardware
# benchmark, even when the caller deliberately selects an accelerator backend.
for key, value in {
  "ASSERT_COMPILE": "0",
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "1",
  "CCACHE": "0",
  "DEBUG": "0",
  "DEV": requested_device,
  "GRAPH_ONE_KERNEL": "0",
  "HCQ2": "0",
  "IGNORE_JIT_FIRST_BEAM": "0",
  "IMAGE": "0",
  "JIT": requested_jit,
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

from tinygrad import Device, Tensor, TinyJit, dtypes  # noqa: E402
from tinygrad.helpers import JIT  # noqa: E402
from tinygrad.uop.ops import Ops  # noqa: E402


python_calls = 0


@TinyJit
def add_one(x: Tensor) -> Tensor:
  """The counter changes only when ordinary Python executes this body."""
  global python_calls
  python_calls += 1
  return (x + 1.0).realize()


def make_input(call_index: int) -> Tensor:
  start = float(call_index * 10)
  return Tensor([start, start + 1.0], device=Device.DEFAULT, dtype=dtypes.float32).realize()


def main() -> None:
  assert JIT.value in (0, 1, 2)
  print("selected device:", Device.DEFAULT)

  outputs: list[Tensor] = []
  immediate_values: list[list[float]] = []
  for call_index in range(3):
    phase = ("ignore", "capture", "replay")[call_index] if JIT else "ordinary Python"
    out = add_one(make_input(call_index))
    actual = out.tolist()
    expected = [float(call_index * 10 + 1), float(call_index * 10 + 2)]
    assert actual == expected
    outputs.append(out)
    immediate_values.append(actual)
    print(
      f"call={call_index + 1} phase={phase:<15} result={actual} "
      f"python_calls={python_calls} cnt_after={add_one.cnt} captured={add_one.captured is not None}"
    )

  print("immediate values:", immediate_values)
  if JIT == 0:
    assert python_calls == 3 and add_one.captured is None
    assert len({id(out) for out in outputs}) == 3
    print("JIT=0: Python ran three times; no captured plan or reused return wrapper exists")
    print("note: the decorated wrapper still advanced cnt; compare modes in separate processes")
    print("claim: ordinary Python execution and wrapper counter behavior passed with JIT disabled")
    print("non-claim: JIT=0 provides no capture, replay, linking, or device-graph evidence")
    return

  assert python_calls == 2
  assert add_one.captured is not None
  assert outputs[0] is not outputs[1]
  assert outputs[1] is outputs[2]
  assert outputs[1].uop.base is outputs[2].uop.base
  assert outputs[1].tolist() == immediate_values[2]

  captured = add_one.captured
  raw_bodies = [call.src[0].op.name for call in captured._linear.src]
  linked_bodies = [call.src[0].op.name for call in captured.linear.src]
  graph_calls = [
    call for call in captured.linear.src
    if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph"
  ]

  assert raw_bodies == ["PROGRAM"] and linked_bodies == raw_bodies
  assert captured.linear is captured._linear  # HCQ2=0 makes link_linear an identity.
  assert graph_calls == []  # One PROGRAM is not graphed when GRAPH_ONE_KERNEL=0.

  print("capture/replay returned the same Tensor wrapper:", outputs[1] is outputs[2])
  print("capture/replay used the same output base UOp:", outputs[1].uop.base is outputs[2].uop.base)
  print("capture output reference now reads replay data:", outputs[1].tolist())
  print("captured call bodies before/after default linking:", raw_bodies, linked_bodies)
  print("linked LINEAR cached on CapturedJit:", "linear" in captured.__dict__)
  print("selected backend advertises a graph runner:", Device[Device.DEFAULT].graph is not None)
  print("device graph calls:", len(graph_calls), "(a one-PROGRAM capture stays ungraphed here)")
  print("claim: phase, input rebinding, return-wrapper reuse, and one-call plan structure passed")
  print("non-claim: this lab does not measure performance or prove a multi-call device graph")


if __name__ == "__main__":
  main()
