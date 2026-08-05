# 11. Rendering and compilation

## Purpose

Once lowering produces a target-legal kernel, tinygrad must turn it into a
loadable program. This chapter separates responsibilities that are often blurred
together: target selection, instruction selection, linearization, register
allocation, rendering, vendor compilation, binary loading, and launch.

That separation tells you whether a failure belongs in codegen, a renderer, a
compiler wrapper, or a runtime.

**Source snapshot:** `874d331` (2026-08-05).

## Prerequisite gate

You should be able to explain the output of the previous chapter as explicit
control flow, ALU, loads/stores, and launch dimensions. You do not need to know
PTX or an ISA yet.

If terms such as ABI, module loading, or assembly are opaque, read the short
NVIDIA codegen/runtime route in
[Learning resources](../reference/learning-resources.md#nvidia-code-generation-and-runtime-work),
then return when you can distinguish source text, an intermediate assembly, a
compiled object, and a loaded function.

## Keep four selections separate

For `DEV=CUDA:PTX`, for example:

- `CUDA` selects the **runtime/backend** that allocates and launches through the
  CUDA Driver API;
- `PTX` requests a **renderer** for the target;
- the renderer's target includes an **architecture**, such as `sm_89`; and
- the renderer's **compiler** turns its source representation into the bytes
  expected by the runtime.

A physical RTX 4090 does not imply one unique path. `NV` and `CUDA` are different
backends; CUDA C and PTX are different rendering routes; a native ISA renderer
has instruction-selection and register-allocation steps a C-like renderer
delegates to a vendor compiler.

Always record all four choices in a bug report or benchmark.

## Source tour

| Responsibility | Snapshot source |
| --- | --- |
| Target capability contract and estimates | [`Renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/__init__.py#L59) |
| C-like languages including CUDA C | [`renderer/cstyle.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py) |
| Direct PTX rendering | [`PTXRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L137) |
| LLVM IR rendering | [`renderer/llvmir.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/llvmir.py) |
| Native instruction renderers | [`renderer/isa/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/isa) and [`renderer/amd/`](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/amd) |
| Linearize/render/assemble/compile state machine | [`pm_to_program`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L441) |
| Compiler cache and base interface | [`Compiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L303) |
| CUDA source compiler wrapper | [`runtime/support/compiler_cuda.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cuda.py) |
| CUDA module loading and launch | [`CUDAProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L37) |

## `PROGRAM` is an artifact container

For an ordinary compiled kernel in this snapshot, `to_program` builds one
`Ops.PROGRAM` whose children are:

```text
PROGRAM
├── SINK    fully lowered kernel graph
├── LINEAR  topologically ordered operations/instructions
├── SOURCE  rendered source or printable assembly form
└── BINARY  compiled/assembled bytes
```

`ProgramInfo` on the root records information needed later: function name,
target, globals, symbolic values, launch dimensions, and input/output roles.
Operation and memory estimates remain in the lowered `SINK`'s `KernelInfo`.

These children are successive artifacts, not four independently executed
programs. Keeping them together makes a compiler trace and cache entry
self-describing.

## Two program-construction branches

### Text renderer

For a C-like, PTX, LLVM IR, WGSL, or similar renderer:

1. linearize the lowered graph;
2. compute estimates;
3. call `renderer.render` to produce `SOURCE`;
4. call `renderer.compiler.compile_cached` to produce `BINARY`; and
5. let the runtime construct a device-specific `Program` from the resulting
   `TinyELF` description.

The “binary” may itself be an intermediate format accepted by the device stack,
such as PTX. Do not infer native machine code merely from the UOp name.

### Native instruction renderer

An `ISARenderer` adds target instruction selection before program construction.
During linearization it may run pre-register-allocation rewrites, calculate live
ranges, allocate registers, and run post-allocation rules. It then assembles the
instruction UOps directly into bytes rather than sending C-like source to an
external compiler.

This branch exposes more of a traditional backend inside tinygrad, so an error
can originate in instruction selection or register allocation even when the
lowered kernel graph is correct.

## Renderer capability is a contract, not decoration

A `Renderer` describes supported operations, types, vector behavior, local and
thread capabilities, tensor cores, shared-memory limits, and launch bounds.
Codegen reads those properties to decide what must be decomposed and which
optimizations are legal.

Therefore:

- adding syntax for an op without advertising/testing capability is incomplete;
- advertising an op without correct rendering can let invalid work pass farther;
- changing a capability can alter earlier generated graphs and cache keys; and
- renderer tests need negative cases for unsupported forms, not just golden
  strings for supported ones.

## Lab 1 — Name every artifact

From the tinygrad study checkout, run the probe from the previous chapter on
three routes:

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs
DEV=CPU DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
DEV=PYTHON DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
DEV=PYTHON::sm_89 DEBUG=0 CACHEDB=/tmp/tinygrad-guide-cache.db \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase3/inspect_program.py"
```

The third route uses the Python UOp executor with an NVIDIA-oriented target. It
is valuable for codegen correctness; it is not evidence about 4090 timing,
driver behavior, or native instruction performance.

For each run, record:

- selected device and renderer type;
- `PROGRAM` child roles;
- target and architecture;
- first line and language of `SOURCE`;
- whether `BINARY` is source-like, an intermediate format, or native bytes; and
- the runtime `Program` class that will consume it.

Confirm numerical equality across routes. If a target is unsupported on your
checkout, record the exact initialization/lowering failure rather than silently
substituting another backend.

## Lab 2 — CUDA C versus PTX on the 4090

**NVIDIA.** First verify both requested routes explicitly:

```bash
DEV=CUDA DEBUG=1 .venv/bin/python -c 'from tinygrad import Device; print(Device.DEFAULT, type(Device.default.renderer).__name__, Device.default.renderer.target)'
DEV=CUDA:PTX DEBUG=1 .venv/bin/python -c 'from tinygrad import Device; print(Device.DEFAULT, type(Device.default.renderer).__name__, Device.default.renderer.target)'
```

Then run the same small kernel with `DEBUG=4` and save the generated source.
Predict before running:

- which route emits CUDA C and which emits PTX;
- which operations the external NVIDIA toolchain still performs; and
- which differences should be semantic versus merely syntactic.

Compare results and program metadata, not variable names. If you measure the
two routes, use the benchmarking discipline from the kernel-optimization
chapter: warm up, synchronize, repeat, report a distribution, and prove output
correctness. One isolated timing is not a renderer result.

### Optional ISA step

Use the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) to
annotate parameters, global loads, arithmetic, and stores in one generated PTX
kernel. Raw PTX is not input to `nvdisasm`. With compatible `ptxas` and
`nvdisasm` tools in `PATH`, `DEBUG=7` asks the snapshot compiler's `disassemble`
method to assemble a PTX `BINARY` temporarily and print SASS. For a manual
artifact, save the finalized PTX bytes from the program's `BINARY` child—not
the pre-compilation `SOURCE` with target placeholders—then run:

```bash
ptxas -arch=sm_89 kernel.ptx -o kernel.cubin
nvdisasm kernel.cubin
# cuobjdump also needs an appropriate cubin, fatbin, or host binary.
```

State clearly that SASS is native code while PTX is a virtual ISA, and record
the assembler/toolchain version because native output is not fixed by the PTX
text alone.

## Contribution-shaped exercise

Choose one operation whose support differs between two renderer targets.

1. Find the target capability or `code_for_op` decision.
2. Capture the graph immediately before decompositions.
3. Predict the direct and decomposed forms.
4. Add a renderer-isolation test for supported input and a negative or
   decomposition test for unsupported input.
5. Run the same semantic case through `DEV=PYTHON` as an oracle.
6. Only then make a study-branch renderer change.

Your test should fail for the intended contract violation, not because an exact
temporary name or whitespace string changed.

## Failure localization

| Evidence | Likely owner |
| --- | --- |
| Unsupported UOp remains in lowered `SINK` | Decomposition/capability negotiation |
| Lowered `SINK` correct, linear order wrong | Linearizer or control-flow dependencies |
| Linear UOps correct, source semantics wrong | Renderer |
| Native instruction choice wrong | ISA selector or target matcher |
| Values corrupted only under pressure | Register allocation, ABI, or runtime arguments |
| Source is valid but compilation fails | Compiler wrapper, target flags, toolchain support |
| Binary loads but launch fails | Program metadata, ABI, launch dimensions, runtime |
| Second run unexpectedly recompiles | Program/compiler cache key or configuration |

## Checkpoint

Continue when you can:

- distinguish backend, renderer, compiler, architecture, and loaded program;
- explain each `PROGRAM` child and why `BINARY` need not be native code;
- describe the text-renderer and native-ISA branches;
- trace renderer capabilities backward into lowering decisions; and
- localize a failure using adjacent artifacts rather than the final error alone.

## Quick reference

```text
lowered SINK
  → optional instruction selection
  → LINEAR operations
  → optional register allocation
  → SOURCE / assembly text
  → compiler or assembler
  → BINARY bytes
  → runtime Program load
  → launch with buffers, scalar values, global/local dimensions
```
