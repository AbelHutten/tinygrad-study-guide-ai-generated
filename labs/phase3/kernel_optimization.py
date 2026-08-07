"""Inspect equivalent kernel schedules on the pinned Ada-targeted Python route."""

from __future__ import annotations

import argparse
from dataclasses import replace

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops


N = 16
MANUAL_OPTS = (
  Opt(OptOps.UPCAST, axis=1, arg=4),
  Opt(OptOps.LOCAL, axis=0, arg=4),
  Opt(OptOps.UNROLL, axis=0, arg=4),
)
HEURISTIC_OPTS = (
  Opt(OptOps.TC, axis=0, arg=(-1, 0, 1)),
  Opt(OptOps.UPCAST, axis=0, arg=2),
)
PADDING_STRICT_OPTS = (Opt(OptOps.UNROLL, axis=0, arg=0),)
PADDING_ENABLED_OPTS = (
  Opt(OptOps.TC, axis=0, arg=(-1, 2, 1)),
  Opt(OptOps.UPCAST, axis=0, arg=2),
  Opt(OptOps.UPCAST, axis=0, arg=3),
)


def show_range_states() -> None:
  """Apply the manual recipe one step at a time and print the current axis namespace."""
  ast = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule_linear().src[-1].src[0]
  scheduler = Scheduler(ast, Device[Device.DEFAULT].renderer)
  scheduler.convert_loop_to_global()

  def show(label: str) -> None:
    state = list(zip(scheduler.shape_str(), scheduler.full_shape, (axis.name for axis in scheduler.axis_types)))
    print(f"{label}: {state}")

  print("range states:")
  show("start")
  for label, opt in zip(("after UPCAST", "after LOCAL", "after UNROLL"), MANUAL_OPTS):
    scheduler.apply_opt(opt)
    show(label)


def replace_schedule_opts(linear, opts: tuple[Opt, ...] | None):
  """Set the exact internal option recipe on the one computational SINK in LINEAR."""
  calls, changed = [], 0
  for call in linear.src:
    if call.src[0].op is Ops.SINK:
      sink = call.src[0].replace(arg=replace(call.src[0].arg, opts_to_apply=opts))
      call, changed = call.replace(src=(sink, *call.src[1:])), changed + 1
    calls.append(call)
  assert changed == 1, f"expected one computational SINK, found {changed}"
  return linear.replace(src=tuple(calls))


def run_case(name: str, opts: tuple[Opt, ...] | None) -> None:
  # These values make every expected result a small, exactly representable integer:
  # a[i,k] = 1+i%4, b[k,j] = 1+j%4, c[i,j] = N*(1+i%4)*(1+j%4).
  a = Tensor([[float(1 + i % 4) for _ in range(N)] for i in range(N)], dtype=dtypes.float32).realize()
  b = Tensor([[float(1 + j % 4) for j in range(N)] for _ in range(N)], dtype=dtypes.float32).realize()
  out = a @ b

  linear, var_vals = out.linear_with_vars()
  compiled = compile_linear(replace_schedule_opts(linear, opts))
  programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(programs) == 1, [call.src[0].op for call in compiled.src]
  program, kernel = programs[0], programs[0].src[0]
  wmma_count = sum(uop.op is Ops.WMMA for uop in program.src[1].src)

  expected_structure = {
    "baseline": ((), (16, 16, 1), (1, 1, 1), (8192, 33792, 3072), 0),
    "manual": (MANUAL_OPTS, (4, 4, 1), (4, 1, 1), (8192, 21504, 3072), 0),
    "heuristic": (HEURISTIC_OPTS, (1, 1, 1), (32, 1, 1), (8192, 3072, 3072), 2),
  }[name]
  estimates = kernel.arg.estimates
  observed_structure = (
    kernel.arg.applied_opts,
    program.arg.global_size,
    program.arg.local_size,
    (estimates.ops, estimates.lds, estimates.mem),
    wmma_count,
  )
  assert observed_structure == expected_structure, (name, observed_structure)

  print(f"\nmode: {name}")
  print("opts:", kernel.arg.applied_opts)
  print("launch:", program.arg.global_size, program.arg.local_size)
  print("estimates:", estimates)
  print("WMMA:", wmma_count)

  # The call is already compiled. jit=True prevents run_linear from compiling it again.
  run_linear(compiled, var_vals, jit=True)
  values = out.tolist()
  expected = [[float(N * (1 + i % 4) * (1 + j % 4)) for j in range(N)] for i in range(N)]
  assert values == expected
  print("samples:", values[0][0], values[0][3], values[3][0], values[3][3])
  print("checksum:", sum(map(sum, values)))


def require_target() -> None:
  renderer = Device[Device.DEFAULT].renderer
  target = renderer.target
  assert Device.DEFAULT == "PYTHON" and target.device == "CUDA" and target.arch == "sm_89", \
    f"this lab requires DEV=PYTHON::sm_89, got {Device.DEFAULT=} {target=}"


def run_core() -> None:
  require_target()
  target = Device[Device.DEFAULT].renderer.target

  print("target:", target)
  show_range_states()
  run_case("baseline", ())                 # Explicit empty tuple: apply no options.
  run_case("manual", MANUAL_OPTS)          # Replay exactly this option sequence.
  run_case("heuristic", None)              # None: allow the snapshot's default heuristic.


def run_padding_boundary(name: str) -> None:
  """Assert one fresh-process side of the pinned 17x17 TC_OPT boundary."""
  require_target()
  n = 17
  a = Tensor([[float(1 + i % 4) for _ in range(n)] for i in range(n)], dtype=dtypes.float32).realize()
  b = Tensor([[float(1 + j % 4) for j in range(n)] for _ in range(n)], dtype=dtypes.float32).realize()
  out = a @ b

  linear, var_vals = out.linear_with_vars()
  compiled = compile_linear(replace_schedule_opts(linear, None))
  programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(programs) == 1, [call.src[0].op for call in compiled.src]
  program, kernel = programs[0], programs[0].src[0]
  estimates = kernel.arg.estimates
  wmma_count = sum(uop.op is Ops.WMMA for uop in program.src[1].src)

  expected_structure = {
    "padding-strict": (PADDING_STRICT_OPTS, (17, 17, 1), (1, 1, 1), (9537, 21964, 3468), 0),
    "padding-enabled": (PADDING_ENABLED_OPTS, (1, 1, 1), (32, 1, 1), (37568, 6528, 3468), 6),
  }[name]
  observed_structure = (
    kernel.arg.applied_opts,
    program.arg.global_size,
    program.arg.local_size,
    (estimates.ops, estimates.lds, estimates.mem),
    wmma_count,
  )
  assert observed_structure == expected_structure, (name, observed_structure)

  run_linear(compiled, var_vals, jit=True)
  values = out.tolist()
  expected = [[float(n * (1 + i % 4) * (1 + j % 4)) for j in range(n)] for i in range(n)]
  assert values == expected

  print("target:", Device[Device.DEFAULT].renderer.target)
  print("mode:", name)
  print("opts:", kernel.arg.applied_opts)
  print("launch:", program.arg.global_size, program.arg.local_size)
  print("estimates:", estimates)
  print("WMMA:", wmma_count)
  print("samples:", values[0][0], values[0][3], values[3][0], values[3][3])
  print("checksum:", sum(map(sum, values)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=("core", "padding-strict", "padding-enabled"), default="core")
  args = parser.parse_args()

  # Override optimization knobs that a reader may have exported in their shell.
  tc_opt = 2 if args.mode == "padding-enabled" else 0
  with Context(ALLOW_TF32=1, BEAM=0, IMAGE=0, NOLOCALS=0, NOOPT=0, TC=1, TC_OPT=tc_opt, TC_SELECT=-1):
    run_core() if args.mode == "core" else run_padding_boundary(args.mode)
