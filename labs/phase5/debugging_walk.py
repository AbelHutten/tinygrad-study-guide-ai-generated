"""Localize a safe process-local CPU renderer fault by adjacent artifacts."""

from __future__ import annotations

import argparse
import os


if not __debug__:
  raise RuntimeError("debugging_walk.py requires assertions; do not run Python with -O")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=("control", "injected", "fixed"))
  return parser.parse_args()


args = parse_args()
required_dev = "PYTHON" if args.mode == "control" else "CPU:CLANG"
requested_dev = os.environ.get("DEV")
if requested_dev != required_dev:
  raise RuntimeError(
    f"mode {args.mode!r} requires the explicit environment DEV={required_dev!r}; received {requested_dev!r}"
  )


# DEV remains the caller's explicit route.  The remaining settings make the
# artifact comparison independent of shell defaults and disable persistent
# schedule/compiler caches for this small experiment.
for key, value in {
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "0",
  "CCACHE": "0",
  "CHECK_OOB": "1",
  "DEBUG": "0",
  "DEBUG_RANGEIFY": "0",
  "DISALLOW_BROADCAST": "0",
  "HCQ2": "0",
  "IMAGE": "0",
  "JIT": "0",
  "NO_COLOR": "1",
  "NOOPT": "1",
  "PROFILE": "0",
  "SCACHE": "0",
  "SPEC": "2",
  "TC": "0",
  "THREADS": "1",
  "TRACK_MATCH_STATS": "0",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402
from tinygrad.codegen import to_program  # noqa: E402
from tinygrad.engine.realize import run_linear  # noqa: E402
from tinygrad.renderer.cstyle import ClangRenderer  # noqa: E402
from tinygrad.uop.ops import Ops, UOp  # noqa: E402


ORACLE = [5.0, 6.0, 7.0]
INJECTED_SYMPTOM = [-3.0, -2.0, -1.0]
ORIGINAL_ADD = ClangRenderer.code_for_op[Ops.ADD]


def faulty_add(a: str, b: str, dtype) -> str:
  """Render only scalar float32 ADD as subtraction; preserve every other ADD."""
  return f"({a}-{b})" if dtype == dtypes.float32 else ORIGINAL_ADD(a, b, dtype)


class FaultyClangRenderer(ClangRenderer):
  """A local renderer type with an independent operation table."""

  code_for_op = {**ClangRenderer.code_for_op, Ops.ADD: faulty_add}


def make_raw_plan() -> tuple[Tensor, UOp, dict[str, int]]:
  """Return a one-kernel float32 plan and its unrealized output."""
  x = Tensor([1.0, 2.0, 3.0], device=Device.DEFAULT, dtype=dtypes.float32).realize()
  out = x + 4.0
  frontend_has_add = any(node.op is Ops.ADD for node in out.uop.toposort())
  linear, var_vals = out.linear_with_vars()
  sink_calls = [call for call in linear.src if call.src[0].op is Ops.SINK]

  assert frontend_has_add
  assert len(linear.src) == 1 and len(sink_calls) == 1
  assert any(node.op is Ops.ADD for node in sink_calls[0].src[0].toposort())
  assert var_vals == {}
  return out, linear, var_vals


def one_sink_call(linear: UOp) -> UOp:
  calls = [call for call in linear.src if call.src[0].op is Ops.SINK]
  assert len(calls) == 1
  return calls[0]


def source_of(program: UOp) -> str:
  sources = [node.arg for node in program.src if node.op is Ops.SOURCE]
  assert len(sources) == 1 and isinstance(sources[0], str)
  return sources[0]


def binary_of(program: UOp) -> bytes:
  binaries = [node.arg for node in program.src if node.op is Ops.BINARY]
  assert len(binaries) == 1 and isinstance(binaries[0], bytes)
  return binaries[0]


def store_operator(source: str) -> str:
  """Classify the one generated buffer store without depending on temp names."""
  assignments: list[str] = []
  for line in source.splitlines():
    stripped = line.strip()
    if " = " in stripped and stripped.endswith(";") and not stripped.startswith("for "):
      assignments.append(stripped.split(" = ", 1)[1].removesuffix(";"))
  assert assignments, "expected at least one generated assignment"
  # The controlled kernel's final assignment is its global output store.  Any
  # preceding assignments are renderer-introduced temporaries with free names.
  rhs = assignments[-1]
  if "+" in rhs and "-" not in rhs: return "ADD"
  if "-" in rhs and "+" not in rhs: return "SUB"
  raise AssertionError(f"expected one ADD or SUB in store expression, got {rhs!r}")


def prepare_program(linear: UOp, renderer) -> UOp:
  call = one_sink_call(linear)
  program = to_program(call.src[0], renderer)
  assert tuple(node.op for node in program.src) == (Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY)
  return program


def execute_prepared(out: Tensor, raw: UOp, var_vals: dict[str, int], program: UOp) -> list[float]:
  call = one_sink_call(raw)
  prepared_call = call.replace(src=(program, *call.src[1:]))
  prepared = raw.replace(src=(prepared_call,))
  run_linear(prepared, var_vals, jit=True)
  Device[Device.DEFAULT].synchronize()
  return out.tolist()


def require_python() -> None:
  assert Device.DEFAULT == "PYTHON", f"control mode requires DEV=PYTHON, got {Device.DEFAULT!r}"


def require_cpu_clang() -> ClangRenderer:
  assert Device.DEFAULT == "CPU", f"this mode requires DEV=CPU:CLANG, got {Device.DEFAULT!r}"
  renderer = Device[Device.DEFAULT].renderer
  assert type(renderer) is ClangRenderer, f"this mode requires ClangRenderer, got {type(renderer).__name__}"
  return renderer


def control_mode() -> None:
  """Establish the portable semantic oracle without the CPU fault surface."""
  require_python()
  out, raw, var_vals = make_raw_plan()
  program = prepare_program(raw, Device[Device.DEFAULT].renderer)
  result = execute_prepared(out, raw, var_vals, program)
  assert result == ORACLE

  print("mode: control-python")
  print("frontend/schedule ADD preserved:", True, True)
  print("compiled body:", program.op.name)
  print("result/oracle:", result, ORACLE)
  print("status: control-passed")


def cpu_program_pair(raw: UOp, correct_renderer: ClangRenderer) -> tuple[UOp, UOp]:
  """Build correct and faulty artifacts from the same scheduled SINK."""
  faulty_renderer = FaultyClangRenderer(correct_renderer.target)
  assert ClangRenderer.code_for_op[Ops.ADD] is ORIGINAL_ADD
  assert FaultyClangRenderer.code_for_op is not ClangRenderer.code_for_op

  correct = prepare_program(raw, correct_renderer)
  # Render and compile the exact already-lowered LINEAR with the local faulty
  # renderer.  Reusing src[:2] makes the adjacent boundary literal: SINK and
  # LINEAR are identical objects; SOURCE is the first changed child.
  faulty_source = faulty_renderer.render(list(correct.src[1].src))
  faulty_binary = faulty_renderer.compiler.compile_cached(faulty_source)
  faulty = correct.replace(src=correct.src[:2] + (
    UOp(Ops.SOURCE, arg=faulty_source), UOp(Ops.BINARY, arg=faulty_binary)))
  assert correct.src[0] is faulty.src[0]
  assert correct.src[1] is faulty.src[1]
  assert source_of(correct) != source_of(faulty)
  assert binary_of(correct) != binary_of(faulty)
  assert store_operator(source_of(correct)) == "ADD"
  assert store_operator(source_of(faulty)) == "SUB"
  return correct, faulty


def injected_mode() -> None:
  """Reproduce the known wrong value and make expected failure an exit-zero mode."""
  renderer = require_cpu_clang()
  out, raw, var_vals = make_raw_plan()
  correct, faulty = cpu_program_pair(raw, renderer)
  result = execute_prepared(out, raw, var_vals, faulty)

  # This mode succeeds only when the bounded defect reproduces exactly.  It
  # never converts an arbitrary exception or unrelated wrong value into success.
  assert result == INJECTED_SYMPTOM
  assert result != ORACLE
  assert ClangRenderer.code_for_op[Ops.ADD] is ORIGINAL_ADD

  print("mode: injected-cpu-renderer")
  print("frontend/schedule ADD preserved:", True, True)
  print("lowered SINK/LINEAR equal:", correct.src[0] == faulty.src[0], correct.src[1] == faulty.src[1])
  print("correct/faulty SOURCE store operator:", store_operator(source_of(correct)), store_operator(source_of(faulty)))
  print("BINARY changed downstream of SOURCE:", binary_of(correct) != binary_of(faulty))
  print("last good artifact: LINEAR")
  print("first bad artifact: SOURCE")
  print("result/oracle:", result, ORACLE)
  print("status: expected-defect-reproduced")


def fixed_mode() -> None:
  """Execute the standard renderer as the green regression run."""
  renderer = require_cpu_clang()
  out, raw, var_vals = make_raw_plan()
  correct, faulty = cpu_program_pair(raw, renderer)
  result = execute_prepared(out, raw, var_vals, correct)

  assert result == ORACLE
  assert store_operator(source_of(correct)) == "ADD"
  assert store_operator(source_of(faulty)) == "SUB"

  print("mode: fixed-cpu-regression")
  print("standard renderer store operator:", store_operator(source_of(correct)))
  print("fault removed from executed route:", True)
  print("result/oracle:", result, ORACLE)
  print("status: regression-passed")


def main() -> None:
  print("controlled env: BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=1 DEBUG=0 DEBUG_RANGEIFY=0 "
        "DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1 NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 "
        "THREADS=1 TRACK_MATCH_STATS=0 VALIDATE_WITH_CPU=0 VIZ=0")
  if args.mode == "control": control_mode()
  elif args.mode == "injected": injected_mode()
  else: fixed_mode()


if __name__ == "__main__":
  main()
