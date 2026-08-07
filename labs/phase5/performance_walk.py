#!/usr/bin/env python3
"""Build a correctness-bracketed timing dossier without asserting a speedup."""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time


if not __debug__:
  raise RuntimeError("performance_walk.py requires assertions; do not run Python with -O")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=15)
args = parser.parse_args()
if not 5 <= args.samples <= 50:
  parser.error("--samples must be between 5 and 50 so the lab stays bounded")

requested_dev = os.environ.get("DEV")
if not requested_dev:
  raise RuntimeError("performance_walk.py requires one explicit DEV route; automatic device selection is not evidence")
requested_upper = requested_dev.upper()
device_segment = requested_upper.rsplit("+", 1)[-1]
expected_device = device_segment.split(":", 1)[0]
interface_segment = requested_upper.rsplit("+", 1)[0] if "+" in requested_upper else ""
interface_name_requested = interface_segment.split(":", 1)[0]
if (expected_device == "NV" and not interface_segment) or expected_device == "PCI" or interface_name_requested == "PCI":
  raise RuntimeError(
    "performance_walk.py rejects bare NV and direct PCI routes; select an explicit driver-backed route such as CUDA or NVK+NV"
  )


# This is a measurement-mechanics lab, not a benchmark of the selected device.
# Pin the scheduler/codegen/runtime instrumentation choices that define this
# bounded experiment. Toolchain/driver paths and versions remain host evidence.
CONTROLLED_ENV = {
  "ALIGNED": "1",
  "ALLOW_DEVICE_USAGE": "1",
  "ALLOW_TF32": "0",
  "ASSERT_COMPILE": "0",
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "0",
  "CAPTURE_PROCESS_REPLAY": "0",
  "CCACHE": "0",
  "CHECK_OOB": "0",
  "DEBUG": "0",
  "DEBUG_LINEARIZE": "0",
  "DEBUG_RANGEIFY": "0",
  "DEFAULT_FLOAT": "float32",
  "DEFAULT_INT": "int32",
  "DISABLE_FAST_IDIV": "1",
  "DISALLOW_BROADCAST": "0",
  "DMC": "0",
  "EMULATE": "",
  "EMULATED_DTYPES": "",
  "EXPAND_SSA": "0",
  "FLOAT16": "0",
  "FUSE_OPTIM": "0",
  "GRAB_PMA": "0",
  "HCQ2": "0",
  "IMAGE": "0",
  "IOCTL": "0",
  "IOCTL_PRINT": "0",
  "JIT": "0",
  "MAX_KERNEL_BUFFERS": "0",
  "MV": "1",
  "MV_BLOCKSIZE": "4",
  "MV_ROWS_PER_THREAD": "4",
  "MV_THREADS_PER_ROW": "8",
  "NO_COLOR": "1",
  "NOLOCALS": "0",
  "NO_MEMORY_PLANNER": "0",
  "NOOPT": "0",
  "OCCUPANCY_FLOOR": "4096",
  "PCONTIG": "0",
  "PRINT_MATCH_STATS": "0",
  "PROFILE": "0",
  "REDUCEOP_SPLIT_SIZE": "22",
  "REDUCEOP_SPLIT_THRESHOLD": "32768",
  "REWRITE_STACK_LIMIT": "250000",
  "SCACHE": "0",
  "SPEC": "2",
  "SPLIT_REDUCEOP": "1",
  "SUM_DTYPE": "float32",
  "TC": "0",
  "TC_OPT": "0",
  "TC_SELECT": "-1",
  "THREADS": "1",
  "TRACE": "0",
  "TRACK_MATCH_STATS": "0",
  "TRACEMETA": "1",
  "TRANSCENDENTAL": "1",
  "TUPLE_ORDER": "1",
  "TEST_PICKLE": "0",
  "UPAT_COMPILE": "1",
  "USE_ATOMICS": "0",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}
for key, value in CONTROLLED_ENV.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402
from tinygrad.engine.realize import compile_linear, estimate_uop, run_linear, time_call  # noqa: E402
from tinygrad.uop.ops import Ops, UOp  # noqa: E402


N = 8


def matrices() -> tuple[list[list[float]], list[list[float]]]:
  # Multiples of 1/8 make all products and these short sums exactly
  # representable in float32.  This lets the lab use an exact oracle without
  # smuggling a loose numerical tolerance into its timing experiment.
  left = [[float(((row * N + col) * 3) % 11 - 5) / 8 for col in range(N)] for row in range(N)]
  right = [[float(((row * N + col) * 5) % 13 - 6) / 8 for col in range(N)] for row in range(N)]
  return left, right


def oracle(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
  return [
    [max(0.0, sum(left[row][inner] * right[inner][col] for inner in range(N))) for col in range(N)]
    for row in range(N)
  ]


def percentile(sorted_values: list[float], fraction: float) -> float:
  """Linearly interpolate a percentile with an explicit, reviewable rule."""
  assert sorted_values and 0.0 <= fraction <= 1.0
  position = (len(sorted_values) - 1) * fraction
  lower, upper = int(position), min(int(position) + 1, len(sorted_values) - 1)
  weight = position - lower
  return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summary(samples: list[float]) -> dict[str, float]:
  ordered = sorted(samples)
  return {
    "min": ordered[0],
    "p10": percentile(ordered, 0.10),
    "median": statistics.median(ordered),
    "p90": percentile(ordered, 0.90),
    "max": ordered[-1],
  }


def timed_completed_run(compiled: UOp) -> float:
  # Drain previous work before the start and include completion before the end.
  Device[Device.DEFAULT].synchronize()
  start = time.perf_counter()
  run_linear(compiled, jit=True)
  Device[Device.DEFAULT].synchronize()
  return time.perf_counter() - start


def main() -> None:
  assert Device.DEFAULT.split(":", 1)[0] == expected_device, (requested_dev, Device.DEFAULT)

  left_values, right_values = matrices()
  expected = oracle(left_values, right_values)

  left = Tensor(left_values, device=Device.DEFAULT, dtype=dtypes.float32).realize()
  right = Tensor(right_values, device=Device.DEFAULT, dtype=dtypes.float32).realize()
  Device[Device.DEFAULT].synchronize()

  construction_start = time.perf_counter()
  out = (left @ right).relu()
  construction_s = time.perf_counter() - construction_start
  assert not out.uop.is_realized

  schedule_start = time.perf_counter()
  raw_linear = out.schedule_linear()
  schedule_s = time.perf_counter() - schedule_start
  sink_calls = [call for call in raw_linear.src if call.src[0].op is Ops.SINK]
  assert len(raw_linear.src) == 1 and len(sink_calls) == 1

  one_call = UOp(Ops.LINEAR, src=(sink_calls[0],))
  compile_start = time.perf_counter()
  compiled = compile_linear(one_call, beam=0)
  compile_s = time.perf_counter() - compile_start
  assert len(compiled.src) == 1 and compiled.src[0].src[0].op is Ops.PROGRAM

  compiled_call = compiled.src[0]
  estimates = estimate_uop(compiled_call)
  assert estimates.ops > 0 and estimates.lds > 0 and estimates.mem > 0

  # The normal path populates the runtime-object cache.  time_call(cache=False)
  # can then reuse that already-loaded object while avoiding insertion of a
  # missing one.  Keep this warm-up outside every reported sample.
  run_linear(compiled, jit=True)
  Device[Device.DEFAULT].synchronize()
  assert out.tolist() == expected

  for _ in range(3): timed_completed_run(compiled)
  wall_samples = [timed_completed_run(compiled) for _ in range(args.samples)]
  Device[Device.DEFAULT].synchronize()
  assert out.tolist() == expected

  # time_call is an internal one-call diagnostic with wait=True.  On this
  # controlled one-call LINEAR, its max-over-linked-calls result has one term.
  for _ in range(3): time_call(compiled_call)
  internal_samples = [time_call(compiled_call) for _ in range(args.samples)]
  assert all(sample >= 0 for sample in (*wall_samples, *internal_samples))

  # Check again after time_call so neither timing family can overwrite a bad
  # result from the other and make the final oracle pass accidentally.
  Device[Device.DEFAULT].synchronize()
  assert out.tolist() == expected

  program = compiled_call.src[0]
  backend = Device[Device.DEFAULT]
  renderer, compiler = backend.renderer, backend.renderer.compiler
  runtime_name = getattr(getattr(backend, "runtime_t", None), "__name__", "none")
  interface_name = type(backend.iface).__name__ if hasattr(backend, "iface") else "none"
  source_nodes = [node.arg for node in program.src if node.op is Ops.SOURCE]
  binary_nodes = [node.arg for node in program.src if node.op is Ops.BINARY]
  assert len(source_nodes) == len(binary_nodes) == 1
  assert isinstance(source_nodes[0], str) and isinstance(binary_nodes[0], bytes)
  source_hash = hashlib.sha256(source_nodes[0].encode()).hexdigest()[:16]
  binary_hash = hashlib.sha256(binary_nodes[0]).hexdigest()[:16]

  print("controlled env:", " ".join(f"{key}={repr(value) if value == '' else value}" for key, value in CONTROLLED_ENV.items()))
  print("requested DEV / canonical device:", requested_dev, Device.DEFAULT)
  print("backend/interface:", type(backend).__name__, interface_name)
  print("renderer/compiler/runtime:", type(renderer).__name__, type(compiler).__name__, runtime_name)
  print("renderer target:", renderer.target)
  print("workload: exact float32 8x8 matmul followed by relu")
  print("correctness: independent Python-loop oracle passed before timing, after completed-wall samples, and after time_call")
  print("raw stage observations (us):", {
    "lazy construction": round(construction_s * 1e6, 3),
    "schedule": round(schedule_s * 1e6, 3),
    "compile": round(compile_s * 1e6, 3),
  })
  print("program launch global/local:", program.arg.global_size, program.arg.local_size)
  print("program children:", [node.op.name for node in program.src])
  print("SOURCE/BINARY bytes and sha256 prefixes:", len(source_nodes[0].encode()), source_hash,
        len(binary_nodes[0]), binary_hash)
  print("static estimates ops/lds/mem:", estimates.ops, estimates.lds, estimates.mem)
  print("completed-wall raw samples (us):", [round(sample * 1e6, 3) for sample in wall_samples])
  print("completed-wall summary (us):", {key: round(value * 1e6, 3) for key, value in summary(wall_samples).items()})
  print("time_call raw samples (us):", [round(sample * 1e6, 3) for sample in internal_samples])
  print("time_call summary (us):", {key: round(value * 1e6, 3) for key, value in summary(internal_samples).items()})
  print("claim: this process exercised a one-call correctness and timing protocol on the named route")
  print("non-claim: the observations are not a speedup, hardware-counter result, model benchmark, or cross-revision comparison")
  print("status: performance-mechanics-passed")


if __name__ == "__main__":
  main()
