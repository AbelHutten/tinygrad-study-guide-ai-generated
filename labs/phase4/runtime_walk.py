"""Walk one tinygrad plan across device, buffer, program, and runtime boundaries."""

from __future__ import annotations

from array import array
import os
import pickle


if not __debug__:
  raise RuntimeError("runtime_walk.py requires assertions; do not run Python with -O")


# DEV remains the caller's choice.  Everything else that could change the
# carried plan is fixed before tinygrad is imported.
for key, value in {
  "BEAM": "0",
  "CACHELEVEL": "0",
  "DEBUG": "0",
  "HCQ2": "0",
  "IMAGE": "0",
  "NO_COLOR": "1",
  "NOOPT": "1",
  "SPEC": "2",
  "TC": "0",
  "THREADS": "1",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402
from tinygrad.device import Buffer, TinyELF  # noqa: E402
from tinygrad.engine.realize import compile_linear, resolve_params, run_linear, runtime_cache  # noqa: E402
from tinygrad.helpers import DEV, GlobalCounters, Target  # noqa: E402
from tinygrad.uop.ops import Ops, UOp  # noqa: E402


EXPECTED_RESULT = [26.0, 107.0]
EXPECTED_SIGNATURE = (
  (None, 0, dtypes.float32, (2,)),
  (None, 1, dtypes.float32, (6,)),
)


def buffer_state(buf: Buffer) -> tuple[bool, bool]:
  """Return underlying-allocation state and this object's handle state."""
  return buf.is_allocated(), buf.is_initialized()


def show_selection() -> None:
  """Separate device-string canonicalization from target selection."""
  device = Device.DEFAULT
  backend = Device[device]
  requested_target = DEV.target(device.split(":", 1)[0])
  renderer_target = backend.renderer.target

  canonical_samples = {
    "python": Device.canonicalize("python"),
    "python:0": Device.canonicalize("python:0"),
    "python:2": Device.canonicalize("python:2"),
  }
  parsed_example = Target.parse("PYTHON::sm_89")

  assert canonical_samples == {"python": "PYTHON", "python:0": "PYTHON", "python:2": "PYTHON:2"}
  assert parsed_example == Target(device="PYTHON", arch="sm_89")
  assert backend.device == device

  assertion_set = "common Compiled contracts"
  if device == "CPU" and requested_target.renderer == "CLANG":
    assert (type(backend).__name__, type(backend.allocator).__name__, type(backend.renderer).__name__,
            backend.runtime_t.__name__) == ("CPUDevice", "CPUAllocator", "ClangRenderer", "CPUProgram")
    assertion_set = "common contracts + optional CPU:CLANG classes"
  elif device == "CUDA":
    assert type(backend).__name__ == "CUDADevice"
    assert type(backend.allocator).__name__ == "CUDAAllocator"
    assert backend.runtime_t.__name__ == "CUDAProgram"
    assertion_set = "common contracts + optional CUDA device/allocator/runtime classes"

  print("selection")
  print("  Device.DEFAULT:", device)
  print("  canonical samples:", canonical_samples)
  print("  DEV target for selected device:", requested_target)
  print("  selected renderer target:", renderer_target)
  print("  backend:", type(backend).__name__)
  print("  allocator/renderer/compiler/runtime:", type(backend.allocator).__name__, type(backend.renderer).__name__,
        type(backend.compiler).__name__, backend.runtime_t.__name__)
  print("  assertion set:", assertion_set)


def show_buffer_lifecycle() -> None:
  """Observe a base allocation and an offset view without patching tinygrad."""
  device = Device.DEFAULT
  backend = Device[device]
  mem_before = GlobalCounters.mem_used
  base = Buffer(device, 4, dtypes.float32)
  view = base.view(2, dtypes.float32, dtypes.float32.itemsize)

  assert buffer_state(base) == (False, False)
  assert buffer_state(view) == (False, False)
  initial = (buffer_state(base), buffer_state(view), GlobalCounters.mem_used - mem_before)

  base.ensure_allocated()
  assert buffer_state(base) == (True, True)
  assert buffer_state(view) == (True, False)
  assert GlobalCounters.mem_used - mem_before == base.nbytes
  after_base = (buffer_state(base), buffer_state(view), GlobalCounters.mem_used - mem_before)

  view.ensure_allocated()
  assert buffer_state(view) == (True, True)
  assert base.allocated_views == 1
  assert GlobalCounters.mem_used - mem_before == base.nbytes
  after_view = (buffer_state(base), buffer_state(view), base.allocated_views, GlobalCounters.mem_used - mem_before)

  if device == "PYTHON":
    assert isinstance(base._buf, memoryview) and isinstance(view._buf, memoryview)
    assert base._buf.obj is view._buf.obj
    base._buf.cast("f")[:] = memoryview(array("f", [10.0, 20.0, 30.0, 40.0]))
    assert list(view._buf.cast("f")) == [20.0, 30.0]

  # This explicit order is deliberate.  The lab does not claim that every
  # allocator rejects deallocating a base while a view still exists.
  view.deallocate()
  assert buffer_state(view) == (True, False)
  assert base.allocated_views == 0
  after_view_free = (buffer_state(base), buffer_state(view), base.allocated_views,
                     GlobalCounters.mem_used - mem_before)

  backend.synchronize()
  base.deallocate()
  assert buffer_state(base) == (False, False)
  assert GlobalCounters.mem_used == mem_before
  final = (buffer_state(base), GlobalCounters.mem_used - mem_before)

  print("buffer lifecycle: (is_allocated, is_initialized)")
  print("  initial base/view, logical-byte delta:", initial)
  print("  after base allocation:", after_base)
  print("  after view initialization, view count:", after_view)
  if device == "PYTHON": print("  PYTHON view reads base elements 1:3:", [20.0, 30.0])
  print("  after view deallocation:", after_view_free)
  print("  after base deallocation:", final)


def make_plan() -> tuple[Tensor, UOp, dict[str, int]]:
  """Build the expression carried through the lowering and rendering chapters."""
  x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=Device.DEFAULT, dtype=dtypes.float32).realize()
  out = (x*x + 2*x).sum(axis=1)
  linear, var_vals = out.linear_with_vars()
  assert len(linear.src) == 1 and linear.src[0].op is Ops.CALL
  assert linear.src[0].src[0].op is Ops.SINK
  assert var_vals == {}
  return out, linear, var_vals


def one_program(compiled: UOp) -> tuple[UOp, UOp]:
  calls = [call for call in compiled.src if call.src[0].op is Ops.PROGRAM]
  assert len(calls) == 1, [call.src[0].op.name for call in compiled.src]
  return calls[0], calls[0].src[0]


def invoke_python_runtime(call: UOp, program: UOp, var_vals: dict[str, int]) -> float:
  """Invoke the already loaded PythonProgram with wait=False and no sync call."""
  device = Device.DEFAULT
  resolved = resolve_params(call, ())
  buffers = [node.buffer for node in resolved]
  program_buffers = [buffers[index].ensure_allocated() for index in program.arg.globals]
  global_size, local_size = program.arg.launch_dims(var_vals)
  runtime = runtime_cache[(program.key, device)]
  elapsed = runtime(
    *[buf.get_buf(device) for buf in program_buffers],
    global_size=global_size,
    local_size=local_size,
    vals=program.arg.vals(var_vals),
    wait=False,
  )
  assert isinstance(elapsed, float)
  return elapsed


def show_program_lifecycle() -> None:
  """Compile once, instantiate once, dispatch, and contrast run_linear modes."""
  device = Device.DEFAULT
  backend = Device[device]
  out, raw, var_vals = make_plan()
  raw_body = raw.src[0].src[0].op
  compiled = compile_linear(raw, beam=0)
  call, program = one_program(compiled)
  transport = program.to_elf()
  key = (program.key, device)
  resolved_before = resolve_params(call, ())

  assert raw_body is Ops.SINK and call.src[0].op is Ops.PROGRAM
  assert isinstance(transport, TinyELF) and isinstance(transport.lib, bytes)
  assert transport.name == program.arg.function_name and transport.target == program.arg.target
  assert transport.signature == EXPECTED_SIGNATURE
  assert program.arg.globals == (0, 1) and program.arg.outs == (0,) and program.arg.ins == (1,)
  assert not resolved_before[0].buffer.is_allocated() and resolved_before[1].buffer.is_allocated()
  assert key not in runtime_cache

  print("program lifecycle")
  print("  raw/compiled CALL bodies:", raw_body.name, call.src[0].op.name)
  print("  roles globals/outs/ins:", program.arg.globals, program.arg.outs, program.arg.ins)
  print("  output/input allocated before dispatch:",
        resolved_before[0].buffer.is_allocated(), resolved_before[1].buffer.is_allocated())
  print("  TinyELF fields: lib/name/target/signature =",
        type(transport.lib).__name__, transport.name, transport.target, len(transport.signature))
  print("  loaded runtime cached before dispatch:", key in runtime_cache)

  if device == "PYTHON":
    decoded = pickle.loads(transport.lib)
    assert isinstance(decoded, list) and all(isinstance(node, UOp) for node in decoded)
    assert backend.runtime_t.__name__ == "PythonProgram"
    print("  PYTHON transport payload decodes to ordered UOps:", True)

  # jit=True says this LINEAR is already compiled/linked.  TinyJit is not used.
  run_linear(compiled, var_vals, jit=True, wait=False)
  assert key in runtime_cache
  assert type(runtime_cache[key]).__name__ == backend.runtime_t.__name__
  assert resolved_before[0].buffer.is_allocated()

  if device == "PYTHON":
    immediate = list(resolved_before[0].buffer._buf.cast("f"))
    assert immediate == EXPECTED_RESULT
    elapsed = invoke_python_runtime(call, program, var_vals)
    assert list(resolved_before[0].buffer._buf.cast("f")) == EXPECTED_RESULT
    print("  PYTHON result present before explicit synchronize:", immediate)
    print("  PYTHON wait=False returned an elapsed-time float:", isinstance(elapsed, float))

  backend.synchronize()
  result = out.tolist()
  assert result == EXPECTED_RESULT
  print("  loaded runtime cached after dispatch:", key in runtime_cache, type(runtime_cache[key]).__name__)
  print("  result after device synchronize:", result)
  print("  TinyJit capture object created:", False)

  # The default jit=False route accepts an uncompiled SINK plan and performs
  # compile_linear + link_linear internally.
  raw_out = (Tensor([1.0, 2.0], device=device, dtype=dtypes.float32).realize() + 3.0)
  raw_default, default_vars = raw_out.linear_with_vars()
  assert raw_default.src[0].src[0].op is Ops.SINK
  run_linear(raw_default, default_vars)
  backend.synchronize()
  default_result = raw_out.tolist()
  assert default_result == [4.0, 5.0]
  print("  raw SINK accepted by run_linear default jit=False:", default_result)


def main() -> None:
  print("controlled env: BEAM=0 CACHELEVEL=0 DEBUG=0 HCQ2=0 IMAGE=0 NO_COLOR=1 NOOPT=1 SPEC=2 TC=0 THREADS=1 VALIDATE_WITH_CPU=0 VIZ=0")
  show_selection()
  show_buffer_lifecycle()
  show_program_lifecycle()


if __name__ == "__main__":
  main()
