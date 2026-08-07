"""Inspect LINEAR, SOURCE, and BINARY artifacts without confusing their contracts."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import os
import pickle


# Lock the code-generation settings on which the structural assertions depend.
# DEV and CC remain caller choices by design; the Tensor dtype is explicit
# below rather than inherited from DEFAULT_FLOAT.
for key, value in {
  "BEAM": "0",
  "CACHELEVEL": "0",
  "DEBUG": "0",
  "IMAGE": "0",
  "NOOPT": "1",
  "TC": "0",
  "THREADS": "1",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402
from tinygrad.codegen import to_program  # noqa: E402
from tinygrad.engine.realize import compile_linear, run_linear  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.uop.ops import Ops, UOp  # noqa: E402


EXPECTED_CHILDREN = (Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY)
EXPECTED_MEMORY = (
  ("STORE", "REG"),
  ("LOAD", "GLOBAL"),
  ("LOAD", "REG"),
  ("STORE", "REG"),
  ("LOAD", "REG"),
  ("STORE", "GLOBAL"),
)
EXPECTED_SIGNATURE = (
  (None, 0, dtypes.float32, (2,)),
  (None, 1, dtypes.float32, (6,)),
)


def make_plan() -> tuple[Tensor, UOp, dict[str, int]]:
  """Build the expression carried from Chapter 10 and expose its scheduled call."""
  x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32, device=Device.DEFAULT).realize()
  out = (x*x + 2*x).sum(axis=1)
  linear, var_vals = out.linear_with_vars()
  assert len(linear.src) == 1 and linear.src[0].op is Ops.CALL
  assert linear.src[0].src[0].op is Ops.SINK
  assert var_vals == {}
  return out, linear, var_vals


def one_program(compiled: UOp) -> UOp:
  programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(programs) == 1, [call.src[0].op for call in compiled.src]
  return programs[0]


def program_parts(program: UOp) -> tuple[UOp, UOp, str, bytes]:
  assert tuple(node.op for node in program.src) == EXPECTED_CHILDREN
  sink, linear, source_node, binary_node = program.src
  assert isinstance(source_node.arg, str) and isinstance(binary_node.arg, bytes)
  assert program.arg.name == "r_2_3"
  assert program.arg.vars == ()
  assert program.arg.globals == (0, 1)
  assert program.arg.outs == (0,) and program.arg.ins == (1,)
  assert program.to_elf().signature == EXPECTED_SIGNATURE
  assert sink.arg.estimates is not None
  assert (sink.arg.estimates.ops, sink.arg.estimates.lds, sink.arg.estimates.mem) == (24, 32, 32)

  memory = tuple((u.op.name, u.src[0].addrspace.name) for u in linear.src if u.op in (Ops.LOAD, Ops.STORE))
  assert memory == EXPECTED_MEMORY
  return sink, linear, source_node.arg, binary_node.arg


def control_trace(linear: UOp) -> list[tuple[str, str | None]]:
  controls: list[tuple[str, str | None]] = []
  for u in linear.src:
    if u.op is Ops.SPECIAL: controls.append((u.op.name, u.arg))
    elif u.op is Ops.RANGE: controls.append((u.op.name, u.arg[-1].name))
    elif u.op is Ops.END: controls.append((u.op.name, None))
  return controls


def arithmetic_counts(linear: UOp) -> list[tuple[str, str, int]]:
  """Keep address integer arithmetic separate from semantic float arithmetic."""
  counts = Counter((u.dtype.name, u.op.name) for u in linear.src if u.op in (Ops.ADD, Ops.MUL, Ops.MULACC))
  return [(dtype, op, count) for (dtype, op), count in counts.items()]


def print_common(mode: str, program: UOp, renderer_name: str, compiler_name: str, runtime_name: str) -> tuple[UOp, str, bytes]:
  sink, linear, source, binary = program_parts(program)
  print("mode:", mode)
  print("target:", program.arg.target)
  print("renderer/compiler/runtime:", renderer_name, compiler_name, runtime_name)
  print("PROGRAM children:", [node.op.name for node in program.src])
  print("name/roles:", program.arg.name, program.arg.globals, program.arg.outs, program.arg.ins)
  print("signature:", [(name, slot, dtype.name, shape) for name, slot, dtype, shape in program.to_elf().signature])
  print("launch global/local:", program.arg.global_size, program.arg.local_size)
  print("estimates:", sink.arg.estimates)
  print("linear control:", control_trace(linear))
  print("linear memory:", list(EXPECTED_MEMORY))
  print("linear arithmetic by dtype:", arithmetic_counts(linear))
  return linear, source, binary


def execute_oracle(out: Tensor, linear: UOp, var_vals: dict[str, int]) -> list[float]:
  compiled = compile_linear(linear, beam=0)
  run_linear(compiled, var_vals, jit=True)
  result = out.tolist()
  assert result == [26.0, 107.0]
  return result


def live_mode() -> None:
  device = Device.DEFAULT
  assert device in {"CPU", "PYTHON"}, f"live mode requires DEV=PYTHON or DEV=CPU:CLANG, got {device!r}"
  backend = Device[device]
  renderer = backend.renderer
  if device == "CPU":
    assert type(renderer).__name__ == "ClangRenderer", "use DEV=CPU:CLANG for the documented CPU route"

  out, linear_plan, var_vals = make_plan()
  compiled = compile_linear(linear_plan, beam=0)
  program = one_program(compiled)
  linear, source, binary = print_common(
    f"live-{device.lower()}", program, type(renderer).__name__, type(backend.compiler).__name__, backend.runtime_t.__name__)

  if device == "PYTHON":
    decoded = pickle.loads(binary)
    assert base64.b64decode(source) == binary
    assert isinstance(decoded, list) and [u.op for u in decoded] == [u.op for u in linear.src]
    assert arithmetic_counts(linear) == [
      ("int", "MULACC", 1), ("float", "MUL", 1), ("float", "MULACC", 1), ("float", "ADD", 1)]
    assert control_trace(linear) == [("SPECIAL", "gidx0"), ("RANGE", "REDUCE"), ("END", None)]
    assert program.arg.global_size == (2, 1, 1) and program.arg.local_size == (1, 1, 1)
    print("SOURCE artifact: base64 text wrapping pickled LINEAR UOps")
    print("BINARY artifact: decoded pickle bytes")
    print("SOURCE decodes to BINARY:", True)
  else:
    witnesses = (
      "void r_2_3(",
      "for (int Lidx1 = 0; Lidx1 < 2; Lidx1++)",
      "for (int Ridx0 = 0; Ridx0 < 3; Ridx0++)",
      "(val0*val0)",
      "(2.0f*val0)",
    )
    assert all(witness in source for witness in witnesses) and len(binary) > 0
    assert arithmetic_counts(linear) == [
      ("int", "MUL", 1), ("int", "ADD", 1), ("float", "MUL", 2), ("float", "ADD", 2)]
    assert control_trace(linear) == [("RANGE", "WEAK"), ("RANGE", "REDUCE"), ("END", None), ("END", None)]
    assert program.arg.global_size == (1, 1, 1) and program.arg.local_size == (1, 1, 1)
    print("SOURCE artifact: Clang-compatible C text")
    print("SOURCE witnesses: function row-loop reduction-loop load/math/store")
    print("BINARY artifact: linked host machine-code image")

  run_linear(compiled, var_vals, jit=True)
  result = out.tolist()
  assert result == [26.0, 107.0]
  print("artifact executed: yes")
  print("result:", result)


def mock_ptx_mode() -> None:
  assert Device.DEFAULT == "PYTHON", "mock modes require DEV=PYTHON for the separately executed oracle"
  from tinygrad.renderer.ptx import PTXRenderer

  out, linear_plan, var_vals = make_plan()
  renderer = PTXRenderer(Target.parse("MOCK+CUDA:PTX:sm_89"))
  program = to_program(linear_plan.src[0].src[0], renderer)
  linear, source, binary = print_common("mock-ptx", program, type(renderer).__name__, type(renderer.compiler).__name__, "none")
  finalized = binary.decode("utf-8")

  assert source.startswith(".version VERSION\n.target TARGET\n.address_size 64")
  assert binary == source.replace("TARGET", "sm_89").replace("VERSION", "7.8").encode()
  assert finalized.startswith(".version 7.8\n.target sm_89\n.address_size 64")
  assert "VERSION" not in finalized and "TARGET" not in finalized
  witnesses = {
    "workgroup coordinate": "%ctaid.x",
    "global load": "ld.global.f32",
    "multiply-add": "fma.rn.f32",
    "global store": "st.global.f32",
  }
  assert all(token in finalized for token in witnesses.values())
  assert arithmetic_counts(linear) == [
    ("int", "MULACC", 1), ("float", "MUL", 1), ("float", "MULACC", 1), ("float", "ADD", 1)]
  assert control_trace(linear) == [("SPECIAL", "gidx0"), ("RANGE", "REDUCE"), ("END", None)]
  assert program.arg.global_size == (2, 1, 1) and program.arg.local_size == (1, 1, 1)

  print("SOURCE artifact: direct-PTX template text")
  print("BINARY artifact: placeholder-finalized PTX text, not native code")
  print("BINARY header:", ".version 7.8 | .target sm_89 | .address_size 64")
  print("PTX witnesses:", list(witnesses))
  print("artifact executed: no")
  print("oracle route: PYTHON")
  print("oracle result:", execute_oracle(out, linear_plan, var_vals))


def optional_mock_cuda_mode() -> None:
  assert Device.DEFAULT == "PYTHON", "mock modes require DEV=PYTHON for the separately executed oracle"
  out, linear_plan, var_vals = make_plan()

  # Importing/constructing the renderer initializes NVRTC.  Only tinygrad's
  # explicit "failed to load library nvrtc" state is an optional skip.  An
  # incompatible library, constructor regression, rejected kernel, or artifact
  # regression must fail this lab rather than masquerade as absence.
  try:
    from tinygrad.renderer.cstyle import CUDARenderer
    renderer = CUDARenderer(Target.parse("MOCK+CUDA:CUDA:sm_89"))
  except AttributeError as exc:  # Only the lazy loader's explicit missing-library state is optional.
    if "failed to load library nvrtc:" not in str(exc): raise
    result = execute_oracle(out, linear_plan, var_vals)
    print("mode: optional-mock-cuda")
    print("status: unavailable", type(exc).__name__)
    print("artifact executed: no")
    print("oracle route/result: PYTHON", result)
    return

  program = to_program(linear_plan.src[0].src[0], renderer)
  linear, source, binary = print_common(
    "optional-mock-cuda", program, type(renderer).__name__, type(renderer.compiler).__name__, "none")
  ptx = binary.decode("utf-8")
  assert binary.endswith(b"\x00"), "NVRTC's PTX result includes its terminating NUL byte"
  cuda_witnesses = (
    'extern "C" __global__ void __launch_bounds__(1) r_2_3',
    "blockIdx.x",
    "for (int Ridx0 = 0; Ridx0 < 3; Ridx0++)",
  )
  assert all(token in source for token in cuda_witnesses)
  assert ".target sm_89" in ptx and ".visible .entry" in ptx
  assert arithmetic_counts(linear) == [
    ("int", "MUL", 1), ("int", "ADD", 1), ("float", "MUL", 2), ("float", "ADD", 2)]
  assert control_trace(linear) == [("SPECIAL", "gidx0"), ("RANGE", "REDUCE"), ("END", None)]

  print("status: available")
  print("SOURCE artifact: CUDA C text")
  print("BINARY artifact: NUL-terminated NVRTC PTX bytes, not native code")
  print("CUDA C witnesses: kernel block-index reduction-loop")
  print("artifact executed: no")
  print("oracle route/result: PYTHON", execute_oracle(out, linear_plan, var_vals))


def main() -> None:
  if not __debug__:
    raise RuntimeError("render_walk.py requires assertions; unset PYTHONOPTIMIZE and do not run Python with -O")
  parser = argparse.ArgumentParser()
  modes = parser.add_mutually_exclusive_group()
  modes.add_argument("--mock-ptx", action="store_true", help="render deterministic direct PTX without opening a CUDA device")
  modes.add_argument("--optional-mock-cuda", action="store_true", help="try CUDA C to PTX through an installed NVRTC library")
  args = parser.parse_args()

  print("controlled env: BEAM=0 CACHELEVEL=0 DEBUG=0 IMAGE=0 NOOPT=1 TC=0 THREADS=1 VIZ=0")
  if args.mock_ptx: mock_ptx_mode()
  elif args.optional_mock_cuda: optional_mock_cuda_mode()
  else: live_mode()


if __name__ == "__main__":
  main()
