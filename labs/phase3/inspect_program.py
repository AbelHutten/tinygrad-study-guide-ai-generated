"""Inspect one compiled tinygrad PROGRAM without hiding execution behind Tensor.realize."""

import os

from tinygrad import Device, Tensor
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.uop.ops import Ops


device = Device.DEFAULT
backend = Device[device]
x = Tensor([1.0, 2.0, 3.0, 4.0], device=device).realize()
out = (x + 1).square().sum()

# linear_with_vars mutates in-scope Tensor UOps to point at their planned buffers.
linear, var_vals = out.linear_with_vars()
compiled = compile_linear(linear, beam=0)

programs = [call.src[0] for call in compiled.src if call.src[0].op is Ops.PROGRAM]
assert len(programs) == 1, [call.src[0].op for call in compiled.src]
program = programs[0]

print("requested DEV:", os.environ.get("DEV", "<automatic>"))
print(f"canonical device: {device}")
print("backend:", type(backend).__name__)
print("renderer/target:", type(backend.renderer).__name__, program.arg.target)
print("compiler/runtime:", type(backend.compiler).__name__, backend.runtime_t.__name__)
print("execution calls:", len(compiled.src))
print("PROGRAM children:", [node.op.name for node in program.src])
print("launch global/local:", program.arg.global_size, program.arg.local_size)
print("estimated ops/bytes:", program.src[0].arg.estimates)

source = next(node.arg for node in program.src if node.op is Ops.SOURCE)
print("source first line:", source.splitlines()[0][:120])

# The calls are already compiled, so jit=True tells run_linear not to compile them again.
run_linear(compiled, var_vals, jit=True)
print("result:", out.item())
assert out.item() == 54.0
