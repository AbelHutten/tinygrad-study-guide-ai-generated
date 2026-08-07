"""Inspect NVIDIA target, renderer, compiler, and runtime boundaries.

The default ``static`` mode is deterministic and needs no NVIDIA driver.  The
other modes are deliberately tiny physical-device probes.  A physical probe
reports a narrowly recognized initialization failure as ``unavailable``; an
error after initialization is never converted into a skip.

Examples (run from the pinned tinygrad checkout):

  DEV=PYTHON::sm_89 python /path/to/nvidia_paths.py --mode static
  DEV=CUDA          python /path/to/nvidia_paths.py --mode cuda
  DEV=CUDA:PTX      python /path/to/nvidia_paths.py --mode cuda-ptx
  DEV=NVK+NV        python /path/to/nvidia_paths.py --mode nvk-nv
  DEV=NVK+NV:PTX    python /path/to/nvidia_paths.py --mode nvk-nv-ptx

There is intentionally no physical ``DEV=NV`` mode: in the pinned snapshot a
bare NV target may fall back from NVK to the direct PCI interface.
"""

from __future__ import annotations

import argparse
import os


if not __debug__:
  raise RuntimeError("this teaching lab uses assertions; run Python without -O or PYTHONOPTIMIZE")


ROUTES = {
  "static": "PYTHON::sm_89",
  "cuda": "CUDA",
  "cuda-ptx": "CUDA:PTX",
  "nvk-nv": "NVK+NV",
  "nvk-nv-ptx": "NVK+NV:PTX",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=tuple(ROUTES), default="static")
  parser.add_argument(
    "--require-available", action="store_true",
    help="fail instead of returning normally when a physical mode is unavailable",
  )
  parsed = parser.parse_args()
  if parsed.require_available and parsed.mode == "static":
    parser.error("--require-available applies only to physical modes")
  return parsed


args = parse_args()
expected_dev = ROUTES[args.mode]
requested_dev = os.environ.get("DEV")
if requested_dev != expected_dev:
  raise RuntimeError(
    f"mode {args.mode!r} requires the explicit environment DEV={expected_dev!r}; "
    f"received {requested_dev!r}"
  )

# Freeze settings which could change lowering, compilation, graphing, timing,
# or output.  In particular this replaces an ambient value such as
# DEBUG=release before tinygrad imports helpers.ContextVar.
for key, value in {
  "ALLOW_TF32": "0",
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "0",
  "CCACHE": "0",
  "CHECK_OOB": "0",
  "DEBUG": "0",
  "DEBUG_RANGEIFY": "0",
  "DEFAULT_FLOAT": "float32",
  "DEFAULT_INT": "int32",
  "DEV": expected_dev,
  "DISABLE_FAST_IDIV": "1",
  "EMULATED_DTYPES": "",
  "HCQ2": "0",
  "IMAGE": "0",
  "JIT": "0",
  "NOLOCALS": "0",
  "NO_COLOR": "1",
  "NO_MEMORY_PLANNER": "0",
  "NOOPT": "1",
  "PROFILE": "0",
  "SCACHE": "0",
  "SPEC": "2",
  "TC": "0",
  "TC_OPT": "0",
  "TC_SELECT": "-1",
  "THREADS": "1",
  "TRANSCENDENTAL": "1",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402
from tinygrad.codegen import to_program  # noqa: E402
from tinygrad.engine.realize import compile_linear, run_linear  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.renderer.ptx import PTXRenderer  # noqa: E402
from tinygrad.uop.ops import Ops  # noqa: E402


def program_child(program, op: Ops):
  matches = [node for node in program.src if node.op is op]
  assert len(matches) == 1, (op, [node.op for node in program.src])
  return matches[0]


def make_linear():
  """Return one lazy, one-kernel expression and its schedulable LINEAR."""
  x = Tensor([1.0, 2.0, 3.0, 4.0], device=Device.DEFAULT, dtype=dtypes.float32).realize()
  out = x * 2.0 + 1.0
  linear, var_vals = out.linear_with_vars()
  assert linear.op is Ops.LINEAR and len(linear.src) == 1
  assert linear.src[0].op is Ops.CALL and linear.src[0].src[0].op is Ops.SINK
  return out, linear, var_vals


def compiled_program(linear):
  compiled = compile_linear(linear, beam=0)
  programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(programs) == 1, [call.src[0].op for call in compiled.src]
  program = programs[0]
  assert [node.op for node in program.src] == [Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY]
  return compiled, program


def artifact_kind(blob: bytes) -> str:
  """Return a format hint, not a parser or an artifact-validity verdict."""
  if blob.startswith(b"\x7fELF"): return "ELF-like bytes (magic only)"
  # NVRTC may prefix comments before the first PTX directive; direct
  # PTXCompiler output starts at .version.  Require both header directives.
  if b".version " in blob[:512] and b".target " in blob[:512]: return "PTX-like text (header directives found)"
  return f"opaque bytes ({len(blob)} bytes)"


def static_mode() -> None:
  # First prove syntax without constructing NV or CUDA devices.  In particular,
  # parsing bare NV is safe; instantiating it is intentionally absent.
  expected = {
    "PYTHON::sm_89": Target(device="PYTHON", arch="sm_89"),
    "CUDA": Target(device="CUDA"),
    "CUDA:PTX": Target(device="CUDA", renderer="PTX"),
    "NV": Target(device="NV"),
    "NVK+NV": Target(device="NV", interface="NVK"),
    "NVK+NV:PTX": Target(device="NV", renderer="PTX", interface="NVK"),
    "PCI+NV": Target(device="NV", interface="PCI"),
  }
  for spelling, target in expected.items():
    parsed = Target.parse(spelling)
    assert parsed == target and repr(parsed) == spelling
    print(f"parse {spelling:<15} -> device={parsed.device:<6} renderer={parsed.renderer or '-':<6} "
          f"arch={parsed.arch or '-':<6} interface={parsed.interface or '-'}")

  assert Device.DEFAULT == "PYTHON"
  backend = Device[Device.DEFAULT]
  renderer = backend.renderer
  assert type(backend).__name__ == "PythonDevice"
  assert type(renderer).__name__ == "PythonRenderer"
  assert renderer.target == Target(device="CUDA", renderer="PYTHON", arch="sm_89")

  out, linear, var_vals = make_linear()
  original_sink = linear.src[0].src[0]

  # Render the exact same high-level kernel AST as direct PTX without touching a
  # driver.  PTXCompiler only substitutes target/version placeholders here.
  ptx_renderer = PTXRenderer(Target.parse("CUDA:PTX:sm_89"))
  ptx_program = to_program(original_sink, ptx_renderer)
  ptx_source = program_child(ptx_program, Ops.SOURCE).arg
  ptx_binary = program_child(ptx_program, Ops.BINARY).arg
  assert type(ptx_renderer.compiler).__name__ == "PTXCompiler"
  assert ptx_program.arg.target == Target.parse("CUDA:PTX:sm_89")
  assert ptx_source.startswith(".version VERSION\n.target TARGET\n.address_size 64")
  decoded_ptx = ptx_binary.decode("utf-8")
  assert decoded_ptx.startswith(".version 7.8\n.target sm_89\n.address_size 64")
  assert ".visible .entry" in decoded_ptx and ".param .u64" in decoded_ptx

  compiled, python_program = compiled_program(linear)
  assert python_program.arg.target == Target(device="CUDA", renderer="PYTHON", arch="sm_89")
  run_linear(compiled, var_vals, jit=True, wait=True)
  actual = out.tolist()
  assert actual == [3.0, 5.0, 7.0, 9.0]

  print("backend/renderer/compiler/runtime:", type(backend).__name__, type(renderer).__name__,
        type(backend.compiler).__name__, backend.runtime_t.__name__)
  print("renderer target:", renderer.target)
  print("PROGRAM children:", [node.op.name for node in python_program.src])
  print("launch global/local:", python_program.arg.global_size, python_program.arg.local_size)
  print("direct PTX source header:", repr(ptx_source.splitlines()[:3]))
  print("compiled PTX header:", repr(decoded_ptx.splitlines()[:3]))
  print("result:", actual)
  print("status: passed")
  print("claims established: target parsing; Ada-targeted lowering; direct PTX text/version substitution; arithmetic result")
  print("claims not established: NVIDIA driver availability; GPU execution; warp/barrier behavior; SASS; occupancy; performance")


def exception_leaves(exc: BaseException):
  if isinstance(exc, BaseExceptionGroup):
    for child in exc.exceptions: yield from exception_leaves(child)
  else: yield exc


def known_preflight_unavailability(exc: BaseException, mode: str) -> bool:
  """Recognize only device/tool presence failures, and only during preflight."""
  leaves = list(exception_leaves(exc))
  if not leaves: return False

  cuda_codes = ("CUDA Error 34,", "CUDA Error 46,", "CUDA Error 100,", "CUDA Error 101,",
                "CUDA Error 802,", "CUDA Error 803,")
  device_paths = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0")
  missing_libraries = {
    "cuda": ("cuda", "nvrtc"),
    "cuda-ptx": ("cuda",),
    "nvk-nv": ("nvrtc", "nvjitlink"),
    "nvk-nv-ptx": ("nvjitlink",),
  }

  def known(leaf: BaseException) -> bool:
    message = str(leaf)
    if mode.startswith("cuda") and isinstance(leaf, RuntimeError) and message.startswith(cuda_codes): return True
    if mode.startswith("nvk-nv") and isinstance(leaf, (FileNotFoundError, PermissionError)):
      filename = str(getattr(leaf, "filename", "") or "")
      return filename in device_paths
    # tinygrad's DLL wrapper deliberately swallows the loader's OSError.  Its
    # first bound-symbol access then emits this exact AttributeError prefix.
    if isinstance(leaf, AttributeError) and any(
      message.startswith(f"failed to load library {library}: ") for library in missing_libraries[mode]
    ): return True
    # ctypes raises AttributeError, rather than OSError, when the SONAME loads
    # but the installed nvJitLink ABI lacks the exact symbol this snapshot calls.
    if (mode == "nvk-nv-ptx" and isinstance(leaf, AttributeError) and "libnvJitLink.so" in message
        and message.endswith("undefined symbol: nvJitLinkVersion")): return True
    return False

  return all(known(leaf) for leaf in leaves)


def physical_preflight(mode: str):
  """Construct the backend and compiler before creating the lab Tensor.

  NVDevice construction itself submits and synchronizes queue-setup commands;
  this boundary promises only that no lab arithmetic kernel has been compiled
  or launched yet.
  """
  backend = Device[Device.DEFAULT]
  renderer = backend.renderer
  compiler = backend.compiler

  if mode.startswith("cuda"):
    assert Device.DEFAULT == "CUDA" and type(backend).__name__ == "CUDADevice"
    assert backend.runtime_t.__name__ == "CUDAProgram"
  else:
    assert Device.DEFAULT == "NV" and type(backend).__name__ == "NVDevice"
    assert type(backend.iface).__name__ == "NVKIface"
    assert backend.runtime_t.__name__ == "NVProgram"

  if mode.endswith("ptx"):
    assert type(renderer).__name__ == "PTXRenderer"
    if mode == "cuda-ptx": assert type(compiler).__name__ == "PTXCompiler"
    if mode == "nvk-nv-ptx": assert type(compiler).__name__ == "NVPTXCompiler"

  assert renderer.target.arch == "sm_89", (
    f"this physical lab is scoped to the RTX 4090 target sm_89, got {renderer.target.arch!r}"
  )
  return backend, renderer, compiler


def physical_mode(mode: str) -> None:
  try:
    backend, renderer, compiler = physical_preflight(mode)
  except BaseException as exc:
    if not known_preflight_unavailability(exc, mode): raise
    print("requested route:", expected_dev)
    print("status: unavailable")
    print("preflight reason:", f"{type(exc).__name__}: {exc}")
    print("claims established: the explicitly selected route could not initialize on this host")
    print("claims not established: compilation; module/program load; launch; GPU result; synchronization; performance")
    print("note: unavailable is not passed; no lab arithmetic kernel was compiled or launched")
    if args.require_available:
      raise RuntimeError(f"required physical route {expected_dev} is unavailable") from exc
    return

  # From here onward every error is a real lab failure.  In particular, a
  # compiler, module-load, launch, synchronization, or result error propagates.
  out, linear, var_vals = make_linear()
  compiled, program = compiled_program(linear)
  source = program_child(program, Ops.SOURCE).arg
  binary = program_child(program, Ops.BINARY).arg
  assert isinstance(source, str) and isinstance(binary, bytes) and binary
  assert program.arg.target == renderer.target

  run_linear(compiled, var_vals, jit=True, wait=True)
  backend.synchronize()
  actual = out.tolist()
  assert actual == [3.0, 5.0, 7.0, 9.0]

  if mode == "cuda-ptx":
    assert source.startswith(".version VERSION")
    assert binary.startswith(b".version 7.8\n.target sm_89")
  if mode == "nvk-nv-ptx":
    assert source.startswith(".version VERSION")
    assert binary.startswith(b"\x7fELF")

  print("requested/canonical route:", expected_dev, Device.DEFAULT)
  print("backend/interface:", type(backend).__name__, type(getattr(backend, "iface", None)).__name__)
  print("renderer/compiler/runtime:", type(renderer).__name__, type(compiler).__name__, backend.runtime_t.__name__)
  print("renderer target:", renderer.target)
  print("PROGRAM children:", [node.op.name for node in program.src])
  print("source first line:", source.splitlines()[0])
  print("binary artifact:", artifact_kind(binary))
  print("launch global/local:", program.arg.global_size, program.arg.local_size)
  print("result:", actual)
  print("status: passed")
  print("claims established: selected backend/interface/renderer/compiler; compile and load; one synchronized GPU result")
  print("claims not established: race freedom for other kernels; peak performance; occupancy; general backend correctness")


if __name__ == "__main__":
  static_mode() if args.mode == "static" else physical_mode(args.mode)
