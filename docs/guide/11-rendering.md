# 11. Rendering and compilation

## The promise of this chapter

Chapter 10 ended with an ordered, target-legal program. That is a large step,
but a list of UOps is not yet the representation that Clang, NVIDIA's compiler,
or a tinygrad runtime consumes. The next boundary is easy to describe vaguely
and easy to misunderstand: tinygrad **renders** an ordered program into some
target representation, a **compiler** turns that representation into bytes,
and a **runtime program** later consumes those bytes.

The words *source* and *binary* are especially dangerous here. On one route,
source is C and binary is host machine code. On another, source is a base64
string and binary is a Python pickle. On the direct-PTX route, both artifacts
are PTX text in different states. The UOp names `SOURCE` and `BINARY` describe
positions in the pipeline, not universal file formats.

This chapter teaches that boundary from the beginning. It carries the same
two-row calculation used in Chapter 10 through three controlled routes:

- ordinary `PYTHON`, where the ordered UOps are serialized and interpreted;
- `CPU:CLANG`, where C text is compiled into a host machine-code image; and
- a hardware-free mock target, where tinygrad directly renders PTX for `sm_89`.

An optional fourth route sends generated CUDA C through an installed NVRTC
library without opening a CUDA device. Each route has a deliberately narrow
evidence contract. By the end, you should be able to:

- distinguish device/backend, target, renderer, compiler, artifact, and runtime;
- explain every child of an `Ops.PROGRAM`;
- read a short C or PTX kernel without already knowing either language;
- trace a generated statement back to a `LINEAR` UOp;
- explain why target capability changes the UOps that reach a renderer;
- state whether `BINARY` contains a pickle, PTX, native code, or something else;
- tell compilation failure from rendering failure and loading failure;
- inspect an artifact without claiming that it executed; and
- choose a focused source location for a renderer contribution.

The exact observations and links target tinygrad commit `874d331` from
2026-08-05. Generated text, class names, and pipeline details can change on
current `master`; reproduce them before using this chapter as contribution
evidence.

## Route through the chapter

Read this chapter front to back once. The sequence is intentional:

1. recover the carried calculation and its semantic oracle;
2. separate five objects that are often all called “the backend”;
3. learn the `PROGRAM` artifact container;
4. understand why a graph becomes a `LINEAR` sequence;
5. follow `to_program` through rendering and compilation;
6. read the CPU C, Python serialization, and direct PTX artifacts;
7. run the lab and interpret its output without crossing evidence boundaries;
8. localize failures between adjacent artifacts; and
9. turn one observation into contribution-shaped work.

No C, assembly, PTX, ABI, or compiler-construction background is assumed. When
a term first matters, it is defined. The optional background section points to
deeper material after the core route is understandable.

This chapter starts after lowering and stops before runtime loading and launch.
Chapter 12 follows `BINARY` bytes, signatures, buffers, and launch dimensions
across the machine boundary. Chapter 14 develops the physical NVIDIA routes.

## Recover the calculation before reading generated code

The carried expression is:

```python
from tinygrad import Tensor, dtypes

x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]], dtype=dtypes.float32).realize()
out = (x*x + 2*x).sum(axis=1)
```

At each input coordinate it computes `x*x + 2*x`. It then adds the three
values in each row. The six elementwise values are:

```text
row 0: 1*1 + 2*1 = 3     2*2 + 2*2 = 8     3*3 + 2*3 = 15
row 1: 4*4 + 2*4 = 24    5*5 + 2*5 = 35    6*6 + 2*6 = 48
```

The independent result is therefore:

```text
[3 + 8 + 15, 24 + 35 + 48] = [26, 107]
```

Chapter 10 derived the row-major input address:

```text
input element index = row*3 + column
```

It also showed the important lowered roles:

- one output coordinate, either a program-instance coordinate or an ordinary
  loop depending on the target;
- one reduction loop of extent three;
- a register-address-space buffer that holds the accumulator;
- one global input load per dynamic reduction iteration;
- updates of the accumulator; and
- one global output store per dynamic row.

Keep those facts beside the generated artifact. A temporary name such as
`val0`, `%alu_f32_1`, or `Ridx0` has no meaning in isolation. It becomes useful
only after you connect it to `x[row*3+column]`, the accumulator, or the output.

### Static lines are not dynamic work

The controlled artifacts contain one textual global load instruction. At
runtime, the reduction loop executes that instruction three times for each of
two rows, so it performs six input loads. Likewise, one output-store statement
executes twice. Do not count source lines and call the result memory traffic.

The lab reports `Estimates(ops=24, lds=32, mem=32)` on every route. The modeled
24 arithmetic operations come from four operations for each of six elements:

```text
x*x              1 multiply
2*x              1 multiply
sum the terms    1 add
reduce the value 1 add
                  --------
                  4 operations * 6 elements = 24
```

The estimates describe the lowered program under tinygrad's model. They are
not instruction counts produced by Clang, NVRTC, `ptxas`, or hardware.
`lds=32` counts dynamic non-register traffic: six four-byte input loads plus
two four-byte output stores. `mem=32` is also 32 here because the same execution
touches all 24 input bytes and all eight output bytes exactly once. In a kernel
that rereads values, `lds` can exceed `mem`: `mem` caps repeated accesses at
the buffer footprint for each load/store role. Register-address-space
accumulator traffic is excluded from both figures. Address arithmetic is also
excluded from `ops` because the `do_estimates` program-construction rewrite
calls `Estimates.from_uops(..., ignore_indexing=True)`.

## Five objects that must stay separate

People often say “CUDA backend” or “CPU codegen” when they mean several
different choices. Use the more precise objects below.

| Object | Question it answers | Carried examples |
| --- | --- | --- |
| **Device/backend** | Who allocates memory and eventually loads and launches work? | `PythonDevice`, `CPUDevice`, later `CUDADevice` |
| **Target** | What device family, renderer choice, architecture, interface, and index mode are being compiled for? | `PYTHON:PYTHON`, `CPU:CLANG:x86_64,native`, `MOCK+CUDA:PTX:sm_89` |
| **Renderer** | How do target-legal ordered UOps become a source or assembly representation? | `PythonRenderer`, `ClangRenderer`, `PTXRenderer` |
| **Compiler** | How does that representation become the bytes stored in `BINARY`? | `PythonCompiler`, `ClangCompiler`, `PTXCompiler`, optionally `NVRTCCompiler` |
| **Runtime program** | How are those bytes loaded and invoked later? | `PythonProgram`, `CPUProgram`; none for the mock artifacts |

The separation is not bureaucracy. It lets two routes share some components
and differ in others. A physical CUDA backend can select a CUDA C renderer or a
direct PTX renderer. A mock target can use a renderer and compiler without any
runtime program. A Python runtime can execute Python-serialized UOps while its
renderer is configured with CUDA target capabilities, as Chapters 9 and 10
did with `PYTHON::sm_89`.

### Read a target from right to left

The pinned `Target` has fields for `device`, `renderer`, `arch`, `interface`,
and `indices`. A normal three-part target is printed as:

```text
DEVICE:RENDERER:architecture
```

For example:

```text
CPU:CLANG:x86_64,native
```

means a CPU target, the Clang/C renderer, and an x86-64 host architecture with
native CPU tuning. The mock lab constructs:

```text
MOCK+CUDA:PTX:sm_89
```

The part before `+` is the interface (`MOCK`). The remaining parts request a
CUDA-family target, direct PTX renderer, and Ada architecture `sm_89`. This is
a `Target` constructed for compilation; it is not a claim that `Device["CUDA"]`
was opened.

Here `sm_89` is NVIDIA/tinygrad's target tag for compute capability 8.9, the
architecture family used by the RTX 4090. It is not the PTX language version:
the later `.version 7.8` header names the PTX syntax/ISA revision, and neither
number proves that a physical GPU is present.

The ordinary `DEV` route chooses a device first. The device publishes candidate
renderer classes, and `Compiled._select_renderer` matches the requested
renderer name and fills an otherwise unset target architecture with the
device's detected architecture. Thus
`DEV=CPU:CLANG` selects `CPUDevice` and explicitly asks its renderer list for
`ClangRenderer`.

Do not infer any of these choices from the GPU model alone. Record the complete
target plus renderer, compiler, and runtime class in a reproducer.

## `PROGRAM` is an artifact container

For every controlled route, the lab finds exactly one `Ops.PROGRAM` with four
children:

```text
PROGRAM  arg=ProgramInfo(...)
├── SINK    target-legal lowered kernel graph
├── LINEAR  ordered UOps
├── SOURCE  renderer output stored as a Python str
└── BINARY  compiler output stored as Python bytes
```

These are successive views of one kernel, not four kernels. The `PROGRAM`
keeps adjacent artifacts together so a later call has the code bytes and the
metadata needed to use them.

### `SINK`: semantic and effect structure

The `SINK` child still has graph structure. Its UOps refer to sources and
dependencies. Ranges, memory effects, accumulator storage, target
decompositions, and launch-coordinate UOps have already been made explicit.
Its `KernelInfo` holds the function name, applied optimization recipe, and
static estimates.

If the `SINK` still contains an operation unsupported by the renderer, the
problem generally precedes string formatting. Inspect target capability and
decomposition before patching the renderer to accept an illegal form.

### `LINEAR`: an order suitable for emission

The `LINEAR` child contains a tuple of UOps in an emission order. The UOps still
refer to their sources, but their position now determines where declarations,
loop starts, loads, arithmetic, stores, and loop ends appear. The renderer can
walk this tuple once and build text or instructions.

The carried Python/direct-PTX line has this control skeleton:

```text
SPECIAL gidx0
RANGE REDUCE
END
```

The row is a global program coordinate and the column is the only source loop.
The CPU/Clang route has:

```text
RANGE WEAK
RANGE REDUCE
END
END
```

Here both row and column remain ordinary C loops, so the launch size is one.
Both organizations compute `[26, 107]`; target capability changes program
shape before rendering.

### `SOURCE`: whatever the renderer emits

`SOURCE.arg` is a string, but it is not guaranteed to be a programming language
a human would type:

- Clang: C text;
- direct PTX: PTX template text;
- Python: base64 text wrapping a pickle of the ordered UOps;
- native ISA renderer: a printable instruction/assembly-oriented form; and
- other targets: their own C-like, shader, IR, or assembly representation.

The correct question is “what contract does this renderer give its compiler?”
not “where is the C source?”

### `BINARY`: whatever bytes the runtime or next tool accepts

`BINARY.arg` is bytes. Bytes alone do not imply native machine code:

- `PythonCompiler` base64-decodes `SOURCE`, so `BINARY` is pickle bytes;
- `PTXCompiler` substitutes header placeholders, so `BINARY` is UTF-8 PTX;
- `NVRTCCompiler` can return PTX or cubin depending on its configuration; its
  PTX result in this snapshot is a byte buffer that includes a final NUL;
- `ClangCompiler` builds an ELF object internally, links/relocates it, and
  stores the resulting host machine-code image; and
- a native assembler stores its assembled bytes.

Name the format whenever you say “binary.” A hash of PTX bytes is not a hash of
the driver-generated native SASS that may later execute.

## Metadata travels beside the artifacts

Code bytes are insufficient to call a function. The caller also needs its
name, arguments, launch dimensions, and symbolic scalar values.

### `ProgramInfo`

The root `PROGRAM.arg` is `ProgramInfo`. For the carried kernel, the lab prints:

```text
name/roles: r_2_3 (0, 1) (0,) (1,)
launch global/local: (2, 1, 1) (1, 1, 1)   # Python and PTX
```

`r_2_3` is the name observed in a clean local process. Kernel names are debug
labels, not semantic identities: tinygrad's process-local collision counter can
append a suffix if the same generated base name was already used. The lab
therefore checks that the name is a valid nonempty function identifier and
uses the observed `program.arg.function_name` in source witnesses; it does not
require the literal spelling `r_2_3`.

The tuples after the name are:

- `globals=(0,1)`: buffer parameter slots referenced by the kernel;
- `outs=(0,)`: slot zero is written;
- `ins=(1,)`: slot one is read.

The CPU route has launch `(1,1,1)/(1,1,1)` because the two rows are in its C
loop. `ProgramInfo.from_sink` derives these fields from target-legal UOps. It
sees `PARAM`s, loads, stores, `SPECIAL` coordinates, and runtime scalar
variables; it does not reverse-engineer rendered text.

### The signature and `TinyELF`

Calling `program.to_elf()` creates a `TinyELF` description with `lib`, `name`,
`target`, and `signature`. Despite its name, `TinyELF` is a common transport
object and its `lib` is not necessarily an ELF file.

The carried signature is:

```text
[(None, 0, 'float', (2,)),
 (None, 1, 'float', (6,))]
```

At this snapshot, `dtypes.float32.name` is the short string `"float"`; the
printed name therefore still describes the explicitly constructed float32
input and output.

Each entry records optional name, slot, dtype, and flattened shape. Slot zero
is a two-float output; slot one is a six-float input. A backend runtime later
uses this description to build pointer/scalar arguments according to its ABI.

**ABI** means the low-level agreement between caller and callee: how arguments
are represented, ordered, aligned, and passed; which calling convention is
used; and what code format is loadable. This chapter only identifies the
contract. Chapter 12 follows its runtime implementation.

Do not confuse three related tuples:

- `globals` says which parameter slots are buffers;
- `outs` and `ins` classify effects on those buffers; and
- `signature` describes the ordered parameter types and shapes.

## Why a graph needs linearization

A graph answers dependency questions well: this add needs these values; this
store must happen after this update; this loop end depends on this loop body.
A source file or instruction stream needs an order.

The pinned `linearize` is a priority-aware topological sort. A **topological
order** places every dependency before a node that uses it. More than one such
order can be legal. The linearizer assigns preferences so parameters and
buffers appear early, loads tend to appear early, stores late, and ranges and
ends form usable control flow. It then chooses an order that respects both
dependencies and those priorities.

This is not arbitrary pretty-printing. Consider the accumulator:

```text
initialize register buffer
enter reduction range
load x
load old accumulator
calculate update
store new accumulator
leave reduction range
load final accumulator
store output
```

Moving the output store inside the reduction would change semantics. Moving
the accumulator initialization after its first load would use an undefined
value. Effect dependencies and range relationships constrain the valid order.

### `RANGE` and `END` delimit control flow

In a graph, a `RANGE` value represents the current loop coordinate. In the
ordered sequence, a renderer sees the `RANGE` and opens a loop. An associated
`END` closes it or emits its back edge. The C renderer uses braces; direct PTX
uses labels, an increment, comparison, and conditional branch; the Python
executor changes its instruction index.

These are three spellings of one control relation. A bug can live in:

- the graph dependency that associated operations with the wrong range;
- the linearizer's placement;
- the renderer's text for `RANGE`/`END`; or
- the runtime/executor's interpretation.

Compare adjacent artifacts instead of guessing from a wrong final value.

### Linear UOp counts still are not hardware counts

The Python/direct-PTX sequence contains one integer `MULACC` for the row-major
address `row*3+column`, plus one float `MUL`, one float `MULACC`, and one float
`ADD`. The Clang/C sequence decomposes those into one integer `MUL` and one
integer `ADD` for the address, plus two float `MUL` and two float `ADD` UOps.
The target decomposition contract differs: direct PTX can express
multiply-add directly, while this C-like path receives decomposed operations.
Keeping dtype beside the operation matters—the address calculation is not one
of the 24 modeled semantic arithmetic operations.

Those are static UOp counts. Loops multiply their dynamic execution. A vendor
compiler can also fuse C multiply/add expressions, split operations, remove
work, or introduce address instructions. Only the next artifact can establish
what that stage produced.

## Follow `to_program` without treating it as magic

The pinned program-construction path is compact enough to narrate precisely.

### Step 1: finish target-aware lowering

`do_to_program` accepts a scheduled `SINK` and a selected renderer. It calls
`full_rewrite_to_sink`, passing that renderer. This is why renderer capability
affects more than final syntax. Unsupported dtypes or operations may be
decomposed, global ranges may become launch coordinates, and target-specific
forms may be introduced before `SOURCE` exists.

The lab disables the scheduling heuristic (`NOOPT=1`) to keep this example
small. It does not disable required target decomposition or lowering.

### Step 2: derive `ProgramInfo`

`ProgramInfo.from_sink` examines the fully rewritten sink and records target,
launch dimensions, parameter slots, inputs, outputs, and scalar variables.
Metadata is derived here, before text rendering. A renderer should not need to
parse its own output to recover a launch size.

### Step 3: optionally select native instructions

If the renderer is an `ISARenderer`, tinygrad runs pre-instruction-selection
and instruction-selection matchers. The graph then contains target instruction
UOps. The text routes in the core lab are not `ISARenderer`s; their external
compiler or textual virtual ISA retains part of the later backend work.

### Step 4: append `LINEAR`

`do_linearize` calls the priority topological sort and applies line-level
cleanup. For an ISA renderer, this stage can also run pre-register-allocation
rewrites, calculate live ranges, map virtual to physical registers, introduce
spills/fills, and apply post-allocation rules.

### Step 5: render or assemble

The `pm_to_program` state machine examines the current children:

```text
SINK
  -> append LINEAR
SINK + LINEAR
  -> attach estimates if absent
LINEAR containing instruction UOps
  -> assemble and create SOURCE + BINARY
ordinary LINEAR
  -> renderer.render(...) and append SOURCE
SINK + LINEAR + SOURCE
  -> compiler.compile_cached(...) and append BINARY
```

The pattern-rewrite engine repeats until the complete four-child `PROGRAM`
exists. This explains why instruction renderers take an assembly branch while
text renderers take render-then-compile.

### Step 6: cache the result

`to_program` has an in-process cache keyed by the scheduled AST key, renderer
type, complete renderer target, and a tuple of compilation configuration
values. The compiler wrapper may have a persistent cache keyed by compiler
cache name and source text.

Caching is useful, but it changes experiments:

- a second call may not invoke the renderer or external compiler;
- changing a variable outside the cache key may not produce the experiment
  you think it does;
- a toolchain upgrade can interact with old cached bytes; and
- timing a cache hit is not compilation timing.

The lab forces `CACHELEVEL=0` so its artifact walk does not silently reuse a
persistent compiler entry. In a bug report, record cache policy and reproduce
with an isolated `CACHEDB` or disabled cache when compilation itself matters.

## A renderer is a capability contract plus an emitter

It is tempting to think a renderer is a dictionary from UOp names to strings.
It is more than that.

The base `Renderer` exposes properties such as:

- supported dtypes and operations;
- global, local, and shared-memory capabilities;
- maximum launch dimensions and shared bytes;
- tensor-core descriptions;
- target-specific rewrite matchers; and
- operation-to-code rules.

Earlier lowering and optimization read those properties. If a renderer says it
cannot represent a form, codegen's target-aware rewrite may decompose it before
rendering. The `Compiler` discussed in this chapter acts later, from `SOURCE`
to bytes; it does not perform that UOp decomposition. If the renderer advertises
a form, that form can reach `render` and must be emitted correctly.

This creates a two-sided obligation:

1. the capability contract must describe what the emitter/compiler/runtime
   combination can actually support; and
2. the emitter must handle every form that the contract allows to reach it.

Adding a string case without checking capability and decomposition can fix one
golden source test while leaving the pipeline inconsistent. Changing a
capability can alter `SINK`, `LINEAR`, launch dimensions, static estimates, and
cache keys before it changes one source line.

### Test semantics, not incidental spelling

Renderer-generated names are conveniences. In the C example, `Lidx1`,
`Ridx0`, `val0`, and `buf0` are derived from current naming policy. Whitespace,
declaration placement, and temporary numbering can change without changing the
program.

A robust renderer test normally checks one of these:

- a required semantic construct is present;
- an unsupported UOp is decomposed before rendering;
- parameter types/address spaces are correct;
- a control-flow or memory operation is ordered correctly;
- source compiles through the intended compiler wrapper; or
- executing the result matches an independent oracle.

Use an exact full-string golden only when exact spelling is the contract you
intend to protect.

## Route 1: read the Clang/CPU artifact as C

With `DEV=CPU:CLANG`, the controlled `LINEAR` sequence renders to this source at
the pinned snapshot:

```c
void r_2_3(float* restrict data0_2, float* restrict data1_6) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Ridx0 = 0; Ridx0 < 3; Ridx0++) {
      float val0 = (*(data1_6+((Lidx1*3)+Ridx0)));
      *(buf0+0) = ((*(buf0+0))+(val0*val0)+(2.0f*val0));
    }
    *(data0_2+Lidx1) = (*(buf0+0));
  }
}
```

You do not need C fluency to map each line back to the semantic loop.

### Function and parameters

```c
void r_2_3(float* restrict data0_2, float* restrict data1_6)
```

`void` means the function returns no scalar value. It writes output memory.
Each `float*` is a pointer to float elements. Slot zero is the output buffer and
slot one the input buffer. `restrict` is an aliasing promise used by the C
compiler; it is not a tensor shape.

The suffixes `_2` and `_6` reflect flattened buffer shapes in current naming.
The authoritative shape information for invocation is the signature, not the
temporary parameter spelling.

### Register-address-space accumulator

```c
float buf0[1];
```

This represents the lowered `AddrSpace.REG` buffer. C syntax makes it look like
a one-element array. The external compiler decides whether it lives in a host
register, stack slot, or another optimized form. Do not infer physical storage
from this declaration alone.

### Two loops

```c
for (int Lidx1 = 0; Lidx1 < 2; Lidx1++)
for (int Ridx0 = 0; Ridx0 < 3; Ridx0++)
```

`Lidx1` is the two-row `WEAK` range and `Ridx0` the three-column reduction. A C
`for` loop initializes the coordinate to zero, continues while it is below the
extent, and increments it after each iteration.

The CPU program's launch is one because this particular controlled schedule
keeps both ranges in the generated function. Other CPU schedules can use
target-thread facilities; do not generalize this one source shape to all CPU
kernels.

### Address, load, arithmetic, and store

```c
data1_6 + (Lidx1*3) + Ridx0
```

Pointer addition is scaled by the pointed-to type, so this is the float-element
address `row*3+column`, not a raw byte offset. Dereferencing with `*` loads the
float. The update spells the two elementwise products plus accumulator add.
After the inner loop, `data0_2+Lidx1` selects the row's output element.

### What Clang does next

`ClangCompiler.compile_to_obj` invokes Clang with target and optimization flags
to produce an ELF object internally. `ClangCompiler.compile` then calls
tinygrad's JIT loader to lay out sections and apply relocations. The final
`BINARY` in this route is the linked host machine-code image that `CPUProgram`
can place in executable memory.

An **ELF object** is a container: it groups code and data into named sections
and carries symbols plus relocation records saying which address-dependent
fields still need fixing. **Relocation** lays those sections out and patches
their references once concrete addresses are known. The bytes retained by
this tinygrad path are the relocated flat image, not the original ELF
container produced by Clang.

The generated C does not establish the final instruction sequence. Clang may
fold constants, vectorize, unroll, or fuse multiply and add. Inspect/disassemble
the actual `BINARY` when native instruction choice matters, and record host
architecture and Clang version.

## Route 2: understand the Python artifact

`DEV=PYTHON` is intentionally different. `PythonRenderer.render` pickles the
ordered UOp list and base64-encodes those bytes into a string. Therefore:

```text
SOURCE = base64 text of pickle(LINEAR UOps)
```

`PythonCompiler.compile` base64-decodes the string:

```text
BINARY = pickle(LINEAR UOps)
```

The lab asserts `base64.b64decode(SOURCE) == BINARY`. `PythonProgram` later
unpickles the list and interprets its UOps. The word *compiler* here names the
common interface stage; the implementation is a decoding operation, not a
native optimizing compiler.

This route teaches two important lessons.

First, `SOURCE` need not be human-readable source code. Printing its first line
produces an opaque base64 prefix and explains almost nothing. Decode or inspect
the `LINEAR` child instead.

Second, `BINARY` need not be machine code. The Python bytes are executable only
in the sense that `PythonProgram` knows how to deserialize and interpret them.

### What executing Python proves

For this small kernel, Python execution proves that:

- the controlled Python target lowered and ordered the calculation;
- the serialized artifact can be decoded by `PythonProgram`;
- its parameter/launch plumbing suffices for this route; and
- it computes exactly `[26.0, 107.0]` for the oracle input.

It does not prove CUDA C or PTX syntax, native instruction behavior, warp
synchronization, device memory ordering, races, cache behavior, register
allocation, driver loading, or GPU performance. The Python route is a semantic
and compiler-structure oracle within its modeled execution contract.

## Route 3: read direct PTX without pretending it ran

**PTX** is NVIDIA's documented virtual instruction-set architecture. It is
lower level than CUDA C and higher level than the final native SASS instructions
executed by a particular GPU. A driver or NVIDIA tool can translate valid PTX
to native code.

The lab directly constructs `PTXRenderer` with target:

```text
MOCK+CUDA:PTX:sm_89
```

No CUDA device is opened. The renderer produces PTX template text. At this
snapshot, `PTXCompiler` does only two substitutions:

```text
VERSION -> 7.8
TARGET  -> sm_89
```

Thus the route is deterministic and hardware-free, but no NVIDIA parser,
assembler, driver, or GPU validates or executes the result.

### Header and entry point

The `SOURCE` begins:

```ptx
.version VERSION
.target TARGET
.address_size 64
.visible .entry r_2_3 (
  .param .u64 data0,
  .param .u64 data1
)
.maxntid 1
```

The `BINARY` begins:

```ptx
.version 7.8
.target sm_89
.address_size 64
```

`.version` selects the PTX language version. `.target` states the target
architecture. `.address_size 64` selects 64-bit addresses. `.visible .entry`
declares a kernel entry point. The two `.u64` parameters carry buffer addresses.
`.maxntid 1` states the maximum threads per block for this launch shape.

The fact that the finalized PTX has these strings proves placeholder
finalization, not PTX validity according to NVIDIA's tools.

### Registers and launch coordinate

PTX declares typed virtual registers such as `.s32`, `.u64`, `.f32`, and
`.pred`. These are PTX virtual registers; final allocation to physical GPU
registers occurs later.

The carried output coordinate is read with:

```ptx
mov.u32 %gidx0, %ctaid.x;
```

`%ctaid.x` is the x dimension's cooperative-thread-array identifier: CUDA's
block index. Because global size is two, two workgroups/blocks select output
rows zero and one. Local size is one, so each block has one work item/thread in
this tiny teaching kernel.

### Parameters, addresses, and global memory

The kernel loads its two pointer parameters with `ld.param.u64`. It computes an
address by converting the element index to 64 bits, multiplying it by four
bytes per float, and adding the base address. The data operation is:

```ptx
ld.global.f32 ...
```

`ld` means load, `global` names the address space, and `f32` is a 32-bit float.
The final output uses:

```ptx
st.global.f32 ...
```

Address-calculation instructions are real work too. A single logical
`x[row,column]` can require integer conversion and multiply-add instructions
before the floating-point load.

### Arithmetic and reduction loop

The direct PTX contains witnesses for:

```ptx
mul.f32
fma.rn.f32
add.f32
```

`fma` is fused multiply-add and `.rn` requests round-to-nearest behavior for
this form. The source `x*x + 2*x` and reduction update have been organized into
these ordered operations. Do not assume the same association for every target
or optimization recipe.

The reduction loop uses labels, a branch to the loop test, an increment, a
less-than predicate, and a predicated branch back to the body. C braces are
gone, but the `RANGE REDUCE`/`END` relation is still visible.

### What the mock PTX route proves

It proves that, at the pinned snapshot and controlled settings:

- the PTX target contract accepted this scheduled kernel;
- lowering produced the asserted launch, control, memory, and arithmetic UOps;
- `PTXRenderer` emitted the expected semantic witnesses;
- `PTXCompiler` finalized its header placeholders; and
- a separately compiled Python oracle computed `[26.0, 107.0]` for the same
  scheduled calculation.

It does **not** prove that the emitted PTX parses, assembles, loads, launches,
or computes the result on NVIDIA hardware. The Python oracle did not execute
the PTX. It also proves nothing about SASS, registers, occupancy, races, timing,
or performance.

To validate direct PTX further, pass the finalized `BINARY` through compatible
NVIDIA tools or a real runtime and then run a semantic test. Chapter 14 covers
that escalation.

## Optional route: CUDA C through NVRTC, still without a GPU

The lab's `--optional-mock-cuda` mode constructs `CUDARenderer` for:

```text
MOCK+CUDA:CUDA:sm_89
```

Here `SOURCE` is CUDA C. It contains an `extern "C" __global__` kernel,
`blockIdx.x`, a reduction loop, pointer loads, arithmetic, and a store.
`NVRTCCompiler` asks an installed NVIDIA Runtime Compilation library to compile
that source for `sm_89`. For this target configuration it requests PTX, so
`BINARY` is NVRTC-produced, NUL-terminated PTX bytes—not cubin and not native
SASS. Decoding those bytes yields PTX text plus its terminating `\0` character.

This route adds meaningful evidence over direct mock PTX:

- an installed NVRTC library initialized;
- NVRTC accepted the emitted CUDA C with the configured options; and
- NVRTC produced PTX bytes for the requested architecture.

It still does not open a CUDA device, load a module, launch the kernel, or
validate numerical behavior of those generated bytes. The printed Python
result is a separate oracle.

NVRTC is optional. A machine can have Python and Clang but no discoverable,
loadable NVRTC shared library. Only tinygrad's explicit
`failed to load library nvrtc` state takes the optional path: the mode prints
`status: unavailable` and the exception type, then exits successfully after the
Python oracle. An incompatible library, another constructor error, or an NVRTC
compiler rejection fails. That boundary prevents a renderer or compiler
regression from masquerading as an honest skipped capability.

## Run the rendering walk

Run from the **guide repository root**. The tinygrad checkout may be elsewhere;
adjust `PYTHONPATH` and the Python executable accordingly. The lab internally
sets `BEAM=0 CACHELEVEL=0 DEBUG=0 IMAGE=0 NOOPT=1 TC=0 THREADS=1 VIZ=0` before importing
tinygrad so exported optimization knobs cannot change its asserted structure.
It also constructs the input with explicit `dtypes.float32`, so an exported
`DEFAULT_FLOAT` cannot change the signature. It deliberately leaves `DEV` and
`CC` to the command. The lab refuses optimized Python (`python -O` or a
nonzero `PYTHONOPTIMIZE`) because that mode removes Python `assert` statements,
including the evidence checks.

### 1. Python serialization and execution

```bash
cd /path/to/tinygrad_docs
PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python labs/phase3/render_walk.py
```

The deterministic core output is:

```text
controlled env: BEAM=0 CACHELEVEL=0 DEBUG=0 IMAGE=0 NOOPT=1 TC=0 THREADS=1 VIZ=0
mode: live-python
target: PYTHON:PYTHON
renderer/compiler/runtime: PythonRenderer PythonCompiler PythonProgram
PROGRAM children: ['SINK', 'LINEAR', 'SOURCE', 'BINARY']
name/roles: r_2_3 (0, 1) (0,) (1,)
signature: [(None, 0, 'float', (2,)), (None, 1, 'float', (6,))]
launch global/local: (2, 1, 1) (1, 1, 1)
estimates: Estimates(ops=24, lds=32, mem=32)
linear control: [('SPECIAL', 'gidx0'), ('RANGE', 'REDUCE'), ('END', None)]
linear memory: [('STORE', 'REG'), ('LOAD', 'GLOBAL'), ('LOAD', 'REG'), ('STORE', 'REG'), ('LOAD', 'REG'), ('STORE', 'GLOBAL')]
linear arithmetic by dtype: [('int', 'MULACC', 1), ('float', 'MUL', 1), ('float', 'MULACC', 1), ('float', 'ADD', 1)]
SOURCE artifact: base64 text wrapping pickled LINEAR UOps
BINARY artifact: decoded pickle bytes
SOURCE decodes to BINARY: True
artifact executed: yes
result: [26.0, 107.0]
```

The `linear memory` line lists ordered `LOAD`/`STORE` UOps and their compiler
address spaces. `REG` entries are accesses to the explicit accumulator buffer;
they are not claims about physical memory transactions. `GLOBAL` entries are
the input load and output store in the static program. Dynamic loop counts and
hardware caching are separate questions.

### 2. Clang C and live CPU execution

```bash
cd /path/to/tinygrad_docs
PYTHONPATH=../tinygrad-study DEV=CPU:CLANG CC=clang \
  ../tinygrad-study/.venv/bin/python labs/phase3/render_walk.py
```

On the intended Ubuntu x86-64 route, the important differences are:

```text
mode: live-cpu
target: CPU:CLANG:x86_64,native
renderer/compiler/runtime: ClangRenderer ClangCompiler CPUProgram
launch global/local: (1, 1, 1) (1, 1, 1)
linear control: [('RANGE', 'WEAK'), ('RANGE', 'REDUCE'), ('END', None), ('END', None)]
linear arithmetic by dtype: [('int', 'MUL', 1), ('int', 'ADD', 1), ('float', 'MUL', 2), ('float', 'ADD', 2)]
SOURCE artifact: Clang-compatible C text
SOURCE witnesses: function row-loop reduction-loop load/math/store
BINARY artifact: linked host machine-code image
artifact executed: yes
result: [26.0, 107.0]
```

The full target string varies with host architecture. The lab requires
`ClangRenderer` but does not assert an x86-only binary length or hash. Compiler
version and CPU features can legitimately change native bytes.

If this command fails before source exists, first check that the pinned
tinygrad version supports the host architecture and that `clang` is available.
Do not weaken the lab to call arbitrary bytes “machine code.”

### 3. Deterministic direct mock PTX

```bash
cd /path/to/tinygrad_docs
PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python labs/phase3/render_walk.py --mock-ptx
```

The important output is:

```text
mode: mock-ptx
target: MOCK+CUDA:PTX:sm_89
renderer/compiler/runtime: PTXRenderer PTXCompiler none
PROGRAM children: ['SINK', 'LINEAR', 'SOURCE', 'BINARY']
name/roles: r_2_3 (0, 1) (0,) (1,)
signature: [(None, 0, 'float', (2,)), (None, 1, 'float', (6,))]
launch global/local: (2, 1, 1) (1, 1, 1)
estimates: Estimates(ops=24, lds=32, mem=32)
linear control: [('SPECIAL', 'gidx0'), ('RANGE', 'REDUCE'), ('END', None)]
linear memory: [('STORE', 'REG'), ('LOAD', 'GLOBAL'), ('LOAD', 'REG'), ('STORE', 'REG'), ('LOAD', 'REG'), ('STORE', 'GLOBAL')]
linear arithmetic by dtype: [('int', 'MULACC', 1), ('float', 'MUL', 1), ('float', 'MULACC', 1), ('float', 'ADD', 1)]
SOURCE artifact: direct-PTX template text
BINARY artifact: placeholder-finalized PTX text, not native code
BINARY header: .version 7.8 | .target sm_89 | .address_size 64
PTX witnesses: ['workgroup coordinate', 'global load', 'multiply-add', 'global store']
artifact executed: no
oracle route: PYTHON
oracle result: [26.0, 107.0]
```

The two final lines must be read together. A Python artifact executed and
provided the oracle. The mock PTX artifact did not execute.

### 4. Optional CUDA C to PTX through NVRTC

```bash
cd /path/to/tinygrad_docs
PYTHONPATH=../tinygrad-study DEV=PYTHON \
  ../tinygrad-study/.venv/bin/python labs/phase3/render_walk.py --optional-mock-cuda
```

When NVRTC is available, look for:

```text
mode: optional-mock-cuda
target: MOCK+CUDA:CUDA:sm_89
renderer/compiler/runtime: CUDARenderer NVRTCCompiler none
status: available
SOURCE artifact: CUDA C text
BINARY artifact: NUL-terminated NVRTC PTX bytes, not native code
CUDA C witnesses: kernel block-index reduction-loop
artifact executed: no
oracle route/result: PYTHON [26.0, 107.0]
```

When it is unavailable, the bounded output is instead:

```text
mode: optional-mock-cuda
status: unavailable <exception type>
artifact executed: no
oracle route/result: PYTHON [26.0, 107.0]
```

Only import/renderer initialization can take that skip path. A failure while
building `PROGRAM`, rendering CUDA C, compiling it, or checking the artifacts
propagates as a failing process.

Do not make this optional mode a prerequisite for the chapter checkpoint. The
direct PTX route is deterministic and does not require NVRTC.

## Compare the route contracts directly

| Route | `SOURCE` | `BINARY` | Compiler action | Artifact executed? | Strongest supported claim |
| --- | --- | --- | --- | --- | --- |
| `DEV=PYTHON` | Base64 of pickled ordered UOps | Pickle bytes | Base64 decode | Yes, by `PythonProgram` | This Python-executed artifact computes the oracle result. |
| `DEV=CPU:CLANG` | C text | Linked host machine-code image | Clang object compilation plus tinygrad JIT linking | Yes, by `CPUProgram` | This host artifact compiles and computes the oracle result on the recorded CPU/toolchain. |
| Mock direct PTX | PTX with `VERSION`/`TARGET` placeholders | Finalized PTX text | String substitution | No | The pinned direct renderer emits these PTX witnesses and header. |
| Optional mock CUDA C | CUDA C text | NUL-terminated NVRTC PTX bytes | NVIDIA CUDA C compilation to PTX | No | Installed NVRTC accepts this source and emits PTX for `sm_89`. |
| Physical CUDA, deferred | CUDA C or PTX depending renderer | PTX bytes on the pinned CUDA candidates discussed here | NVRTC compiles CUDA C, `PTXCompiler` substitutes direct-PTX placeholders, or NVCC emits PTX; the driver later loads/JITs the selected result | Yes | Requires runtime oracle and recorded physical environment. |

The last row is intentionally not exercised by this lab. “Compiled for CUDA”
and “executed on CUDA” are separate events.

## Read generated artifacts in a disciplined order

Opening hundreds of lines of generated code without a question is rarely
productive. Use this bounded method.

### 1. State the semantic oracle

Write the input, output shape, dtype, and expected values independently. For
this kernel: two-by-three float32 input, two-element output, `[26,107]`.

### 2. Record the complete route

Capture device, full target, renderer, compiler, runtime class, tinygrad commit,
and relevant environment. `DEV=CUDA` alone does not identify a renderer or code
format.

### 3. Inspect `ProgramInfo`

Check name, buffer roles, signature, symbolic variables, and launch dimensions.
A wrong global size is not repaired by pretty source text.

### 4. Compare `SINK` and `LINEAR`

Ask whether required operations survived decomposition and whether control and
memory effects have a legal order. Count address spaces, not only op names.

### 5. Find semantic witnesses in `SOURCE`

Locate parameter declarations, coordinate acquisition, address computation,
load, arithmetic, loop/back edge, and store. Explain each in terms of row,
column, input, accumulator, or output.

### 6. Identify `BINARY` format before opening it

Use the selected compiler implementation and target configuration. Do not run
`nvdisasm` on PTX text or an ELF parser on Python pickle bytes. Save raw bytes
without text decoding when the format is opaque.

### 7. Validate the next boundary

If the artifact can run safely, load and execute it with an independent oracle.
If it is inspection-only, state that plainly and choose a separate validation
tool appropriate to the format.

### 8. Only then discuss performance

Generated text can motivate a performance hypothesis. It cannot establish
elapsed time, cache transactions, occupancy, or end-to-end benefit. Measure the
real target using the protocol from Chapters 9 and 17.

## Hashes and saved artifacts: useful but bounded

A source or binary hash answers “are these exact bytes equal?” It does not
answer “are these programs semantically equal?” Different whitespace or
temporary names change a source hash. Two identical PTX blobs can produce
different native code under different driver/toolchain versions.

The pinned NVRTC helper returns the terminating NUL as part of `BINARY`, while
the direct `PTXCompiler` result has no trailing NUL. Hash the raw bytes if the
transport object is your subject. If a text tool requires the terminator to be
removed, strip it deliberately and record that normalization; otherwise two
equivalent-looking PTX files can differ by one byte without explanation.

When saving artifacts, use format-aware names and metadata:

```text
kernel.cuda.c          generated CUDA C SOURCE
kernel.direct.ptx.in   direct PTX SOURCE with placeholders
kernel.direct.ptx      finalized direct PTX BINARY
kernel.nvrtc.ptx       NVRTC-produced PTX BINARY
kernel.host.bin        linked host image, plus architecture/toolchain record
kernel.python.pickle   Python BINARY, never unpickle from an untrusted source
metadata.txt           commit, target, renderer/compiler, settings, signature
```

Python pickle is code-capable serialization. Only unpickle artifacts you
created or trust. The lab unpickles its own in-process output.

For a regression report, save both adjacent artifacts around the suspected
boundary. A bad PTX blob with no `LINEAR` sequence is harder to localize; a bad
result with no exact input and signature is harder still.

## Failure localization by adjacent artifacts

| Observation | First likely boundary | Next comparison |
| --- | --- | --- |
| Unsupported high-level UOp remains in `SINK` | Target decomposition/capability | Scheduled `SINK` vs fully rewritten `SINK`; renderer contract |
| Fully rewritten `SINK` is right, loop or effect order in `LINEAR` is wrong | Linearizer/control dependencies | Range/`END` dependencies and ordered list |
| `LINEAR` is right, C/PTX operation or type is wrong | Renderer | One UOp and its specific emit rule |
| Renderer throws “failed to render” | Capability/renderer mismatch | Was the form advertised or supposed to decompose? |
| `SOURCE` is right, external compiler rejects it | Compiler wrapper/toolchain/options | Exact source, target flags, compiler log/version |
| Direct mock PTX looks right but `ptxas` rejects it | PTX renderer or virtual-ISA version contract | Finalized `BINARY`, not placeholder `SOURCE`; PTX diagnostic |
| CUDA C is accepted but generated PTX is surprising | NVRTC/toolchain optimization | Saved CUDA C, NVRTC flags/version, PTX output |
| `BINARY` format is wrong for runtime | Renderer/compiler selection or transport | Target, compiler return mode, `TinyELF` metadata |
| Bytes load but arguments are rejected | Signature/ABI/runtime | `ProgramInfo`, `TinyELF.signature`, runtime argument packing |
| Program launches but result is wrong | Renderer, compiler, ABI, synchronization, or runtime | Python/CPU oracle, generated addresses, args, launch, sync |
| Only native ISA fails under pressure | Instruction selection/register allocation/spills | Pre/post-regalloc instructions and resource case |
| Second compile unexpectedly does nothing | In-process or compiler cache | Cache keys, `CACHELEVEL`, isolated `CACHEDB` |

Do not automatically assign every compiler error to the renderer. A correct
source can be rejected because the wrapper selected an unsupported architecture
or omitted an include path. Conversely, a compiler diagnostic can expose a
renderer type error. Preserve both the diagnostic and source.

## Debug output without confusing observation and execution

At this snapshot, `DEBUG>=4` prints rendered source during the compile path.
Higher debug levels can also print earlier artifacts and, where implemented,
ask the compiler wrapper to disassemble `BINARY`. This is convenient but not a
format guarantee.

For example, the CUDA compiler's disassembly helper knows whether its bytes are
PTX. If they are, it first invokes `ptxas` to make a native object and then
invokes `nvdisasm`. That output depends on installed tools and architecture.
Calling the helper is additional compilation, not merely displaying the PTX
unchanged.

Prefer programmatic extraction in a focused probe:

```python
program = next(call.src[0] for call in compiled.src
               if call.src[0].op is Ops.PROGRAM)
sink, linear, source, binary = program.src
```

Then assert the child op kinds before relying on positional meaning. Inspect
`source.arg` as text and preserve `binary.arg` as bytes until its compiler
contract identifies the format.

`DEBUG` can change timing and may cause extra disassembly/tool invocations. It
is an inspection setting, not a benchmark protocol.

## The native-ISA branch, bounded

Some tinygrad renderers choose and assemble native instructions inside
tinygrad. They subclass `ISARenderer`. The branch introduces three compiler
ideas that text renderers often delegate externally.

### Instruction selection

**Instruction selection** maps lowered operations to target instructions. One
instruction may implement several UOps, or one UOp may require several
instructions. Legal choices depend on dtype, immediate encodings, addressing
modes, architecture, and surrounding operations.

The pinned path runs a pre-selection matcher and an instruction-selection
matcher before creating the final `PROGRAM` graph.

### Virtual and physical registers

Selected instructions initially refer to compiler-created virtual registers.
There can be more live virtual values than the machine has physical registers.
A value is **live** from its definition until its last required use.

The linear-scan allocator computes live ranges and assigns allowed physical
registers. It pays attention to instruction constraints and loop entry/exit.

### Spills and fills

If no physical register is available, allocation can **spill** a value to a
stack slot and later **fill** it back into a register. Spills can preserve
correctness while damaging performance. Wrong spill offsets, widths, or live
ranges can corrupt values only under register pressure, which is why a tiny
smoke kernel may pass while a larger one fails.

After register allocation and post-allocation rewrites, the assembler produces
bytes directly. In this branch, `SOURCE` is a printable instruction-oriented
record and `BINARY` is assembled output. Do not assume the printable record is
a standalone source file accepted by an external assembler unless that
renderer promises it.

The core lab does not exercise native-ISA rendering. Read this section so you
can recognize the branch and defer its detailed debugging until a contribution
requires it.

## Question-led source stops

Do not open these links as isolated declarations. Bring the stated question,
read only the bounded range, answer it in your own words, and return to the
carried artifact. Every tinygrad link is pinned to the chapter snapshot.

### Stop 1: How is a target string represented?

Question: which fields are preserved in `MOCK+CUDA:PTX:sm_89`, and which ones
are filled by a device? Read
[`Target` and `_DEV.target`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L192-L230).
Parse the mock string on paper, then explain why it does not itself open
`CUDADevice`.

### Stop 2: How does a device choose a renderer?

Question: where do renderer name and detected architecture meet? Read
[`Compiled._select_renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L333-L363).
Then read the short
[`CPUDevice` constructor](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cpu.py#L162-L179),
which supplies both the renderer candidates and detected architecture. For
`DEV=CPU:CLANG`, identify the device class, selected renderer class, and
architecture source.

### Stop 3: Where do the four program children come from?

Question: which transition appends `LINEAR`, `SOURCE`, and `BINARY`, and when is
the assembly branch selected? Read
[`do_linearize` through `pm_to_program`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L409-L449),
then
[`do_to_program` and its cache](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L451-L485).
Write the child sequence after every matching step.

### Stop 4: How is emission order chosen?

Question: why are parameters early, loads early, stores late, and loop ends
constrained? Read the complete bounded
[`linearize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/linearizer.py#L8-L51).
For the control dependency behind the last clause, read
[`CFGContext` and `pm_add_control_flow`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/linearizer.py#L53-L85)
and the call site that
[`adds control flow`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L373-L377).
Relate the resulting priority-aware topological order to the accumulator
lifecycle.

### Stop 5: How does C text arise from ordered UOps?

Question: where are loops, loads/stores, arithmetic, names, indentation, and
function parameters emitted? Start with the narrow
[`base_rewrite` rules](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L10-L72),
then read
[`CStyleLanguage._render`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L200-L254).
Finish with the small
[`render_kernel` parameter wrapper](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L149-L161)
and
[`ClangRenderer` specialization](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L256-L305).
Find the carried row loop, input load, accumulator update, and output store.

### Stop 6: Why is Python `SOURCE` opaque?

Question: what exact transformations relate `LINEAR`, `SOURCE`, `BINARY`, and
the executor's list? Read
[`PythonProgram`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L42-L80)
and
[`PythonCompiler`/`PythonRenderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_python.py#L202-L225).
Explain why neither artifact is native code.

### Stop 7: Which direct PTX rules produced the witnesses?

Question: where do block coordinates, parameters, address calculation, global
loads/stores, loops, and arithmetic become PTX? Read
[`asm_for_op`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L18-L36),
then
[`string_rewrite`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L81-L135)
and the
[`PTXRenderer` header/compiler choice](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/ptx.py#L137-L170).
Annotate one instruction from each category in the finalized artifact.

### Stop 8: What did each compiler actually do?

Question: which path compiles C to an object and links it, which merely replaces
PTX placeholders, and which can invoke NVRTC? Read
[`ClangCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cpu.py#L6-L27)
and
[`_get_bytes`, `NVRTCCompiler`, `PTXCompiler`, and `NVPTXCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cuda.py#L9-L91).
For every return value, write its actual byte format.

### Stop 9: Where do roles and the ABI signature originate?

Question: which metadata comes from loads/stores and `SPECIAL`s, and which
comes from ordered parameters? Read
[`ProgramInfo.from_sink`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1213-L1260)
and
[`UOp.to_elf`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1195-L1199).
Reconstruct the carried `(0,1)/(0)/(1)` roles and two signature entries.

### Stop 10: What changes for a native instruction renderer?

Question: where are instruction selection, live ranges, physical register
assignment, spills, and assembly inserted? Read the small
[`ISARenderer` contract](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/isa/__init__.py#L17-L50),
then the complete bounded
[`LinearScanRegallocContext` and rewrite](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/late/regalloc.py#L9-L132),
and finally the
[`do_linearize` through assembly branch](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L409-L448).
Do not attempt to learn a complete ISA yet; identify the added failure
boundaries.

## Exercises

Try each exercise before opening its answer.

### 1. Classify four artifacts

Classify each `BINARY`: Python route, Clang/CPU route, direct mock PTX route,
and optional mock CUDA C route.

??? answer
    Python stores pickle bytes. Clang/CPU stores a linked host machine-code
    image after internal ELF object compilation and JIT linking. Direct mock
    PTX stores UTF-8 PTX after placeholder substitution. Optional mock CUDA C
    stores NUL-terminated PTX bytes produced by NVRTC for this target
    configuration. Only the first
    two artifacts are executed by this lab.

### 2. Reconcile one load line with six loads

Why can the generated source contain one global-load statement while the
semantic calculation loads six input values?

??? answer
    The statement is inside a reduction loop of extent three. On the CPU it is
    also inside a row loop of extent two; on the PTX route two global program
    coordinates each execute the three-iteration loop. One static statement
    therefore executes `2*3=6` times. Static UOp/source counts are not dynamic
    operation counts.

### 3. Explain the different launch sizes

Why is the controlled CPU launch `(1,1,1)` while Python and PTX report
`(2,1,1)` even though all compute two outputs?

??? answer
    CPU keeps the size-two `WEAK` range as an ordinary generated C loop inside
    one invocation. The Python/PTX target converts that range to `SPECIAL
    gidx0`, so two program/workgroup coordinates each select a row. The
    mathematical output count is the same; target program organization differs.

### 4. Separate PTX `SOURCE` and `BINARY`

What exact change does `PTXCompiler` make for the mock `sm_89` route, and what
does that change fail to validate?

??? answer
    It replaces `VERSION` with `7.8` and `TARGET` with `sm_89`, then encodes the
    text as bytes. It does not parse PTX, invoke `ptxas`, load a module, launch
    a GPU, or check the numerical result. The finalized header proves string
    substitution only.

### 5. Decode the Python boundary

Write the transformations from `LINEAR` to Python `SOURCE`, `BINARY`, and
runtime UOps.

??? answer
    `PythonRenderer` pickles the ordered UOp list and base64-encodes it to make
    the string `SOURCE`. `PythonCompiler` base64-decodes that string to pickle
    bytes in `BINARY`. `PythonProgram.__init__` unpickles those bytes into its
    executable UOp list. No native machine code is produced.

### 6. Read roles and signature

For `globals=(0,1)`, `outs=(0,)`, `ins=(1,)`, and signature entries with shapes
`(2,)` and `(6,)`, describe the call without using generated parameter names.

??? answer
    The kernel takes two float buffer parameters. Slot zero addresses a
    two-element output and is written; slot one addresses a six-element input
    and is read. The roles do not state allocation addresses or a target ABI;
    the runtime supplies those later.

### 7. Localize a wrong loop boundary

The lowered `SINK` associates the output store with the completed reduction,
but `LINEAR` places the store before the reduction `END`. Which boundary should
you inspect first?

??? answer
    Inspect linearization and the control/effect dependencies used to order the
    store and `END`. The semantic lowered graph is already correct. A renderer
    will usually emit the wrong order it receives, so patching its spelling is
    premature.

### 8. Explain the arithmetic-count difference

Python/direct PTX show an integer `MULACC` for addressing and float
`MUL`/`MULACC`/`ADD` UOps. Clang instead shows integer `MUL`/`ADD` and two
float `MUL`/`ADD` pairs. Why does the target change those forms, why is the
integer pair absent from `Estimates.ops`, and does either list prove a native
instruction count?

??? answer
    Target capability controls whether multiply-add stays fused or is
    decomposed before rendering. The integer operation computes the flattened
    address, and these estimates deliberately ignore indexing, so it is not
    part of the 24 semantic operations. The UOp lists still prove no native
    count: Clang can fuse or otherwise optimize C expressions, and NVIDIA
    tools can lower PTX further. Inspect the actual native artifact for that.

### 9. Match a claim to evidence

Which route supports each claim: “the semantic oracle passes,” “tinygrad emits
`ld.global.f32`,” “NVRTC accepts the CUDA C,” and “the RTX 4090 executes this
correctly”?

??? answer
    Live Python and live CPU support their own semantic-oracle claims. Direct
    mock PTX supports the pinned renderer-text claim. Optional mock CUDA C
    supports NVRTC acceptance when it reports available. None supports the
    physical RTX execution claim; that requires a real CUDA route and its own
    correctness run.

### 10. Diagnose a cache surprise

You change compiler flags but compilation returns instantly and `BINARY` is
unchanged. Name two caches to inspect and one clean experiment.

??? answer
    Inspect the in-process `to_program_cache` and the compiler's persistent
    `compile_cached` entry. Run a fresh process with a fresh `CACHEDB` or
    `CACHELEVEL=0`, record the exact target/configuration, and verify the flags
    are represented in the intended compiler instance/cache key.

### 11. Design a renderer test

You add rendering for one dtype conversion. What should a focused test establish
beyond the exact output string?

??? answer
    Establish that capability/decomposition allows the intended form to reach
    the renderer, that source has the correct source/destination types and
    rounding/width contract, that the intended compiler accepts it, and that a
    semantic oracle covers representative and edge values. Add a negative or
    decomposition case for a target that does not support the form. Avoid
    relying only on temporary names or whitespace.

### 12. Build an evidence ladder

Starting from direct mock PTX, list the next three stronger NVIDIA checks
without collapsing them into one claim.

??? answer
    First pass the finalized PTX through a compatible NVIDIA parser/assembler
    such as `ptxas`; this establishes tool acceptance and can produce a cubin.
    Next load and execute the artifact on a recorded CUDA device with an
    independent numerical oracle; this establishes that runtime path for the
    tested case. Finally inspect native SASS/resources and measure repeated,
    synchronized hardware behavior; those establish native structure and
    performance evidence, not merely correctness.

## Contribution-shaped workflow

Renderer and compiler-wrapper changes are small in file count but broad in
possible target effects. Use this sequence:

1. Save a minimal semantic kernel with exact shape, dtype, values, target, and
   independent oracle.
2. Record device, full `Target`, renderer, compiler, runtime, commit, and cache
   policy.
3. Capture the scheduled and fully rewritten `SINK`; identify the capability or
   decomposition rule that should govern the form.
4. Capture `LINEAR` and state the required control, memory, dtype, and effect
   ordering.
5. Point to one renderer rule or compiler-wrapper option and predict its
   artifact change before editing.
6. Add a focused test that protects semantic constructs and includes a negative
   or decomposition case where appropriate.
7. Compile through the real intended compiler wrapper and preserve diagnostics,
   source, bytes format, target flags, and tool version.
8. Execute on a safe supported route with an independent oracle. Do not present
   a separate Python oracle as execution of mock PTX or CUDA C.
9. Test nearby dtypes, vectors, masked accesses, symbolic sizes, control flow,
   and address spaces affected by the rule.
10. Run existing renderer/backend tests and at least one representative
    workload to detect capability-wide regressions.
11. If performance motivates the change, inspect the actual native artifact and
    benchmark the real target separately from compilation time.
12. In the contribution, state exactly which boundary was wrong and why the
    fix belongs there rather than in lowering, scheduling, or runtime.

A good renderer contribution does not merely make one source string look
plausible. It repairs a stated contract and demonstrates that the adjacent
artifacts agree.

## Checkpoint

Continue when you can do all of the following without treating a class name as
an explanation:

- derive `[26,107]` and `row*3+column` for the carried kernel;
- distinguish backend, target, renderer, compiler, runtime program, and toolchain;
- explain `SINK`, `LINEAR`, `SOURCE`, and `BINARY` in order;
- state the actual `SOURCE` and `BINARY` formats for Python, Clang/CPU, direct
  mock PTX, and optional CUDA C/NVRTC;
- explain why the CPU has a row loop and launch one while PTX uses `gidx0` and
  global size two;
- map parameter, coordinate, load, arithmetic, loop, and store witnesses from
  either C or PTX back to the semantic loop;
- explain why one static load statement executes six times;
- reconstruct buffer roles and the two signature entries;
- state exactly what the mock PTX and optional NVRTC routes do not prove;
- identify the text-renderer and native-ISA branches of `to_program`;
- localize a failure by comparing adjacent artifacts; and
- design a renderer test that protects semantics rather than temporary names.

## Quick reference

| Observation | Ask next |
| --- | --- |
| `PROGRAM` lacks `SOURCE` | Did rendering fail, or is this an earlier partial artifact? |
| `SOURCE` is opaque base64 | Is this `PythonRenderer` serialization? Inspect `LINEAR` or trusted decoded pickle. |
| `BINARY` decodes as PTX | Which compiler returned PTX, and has any NVIDIA tool validated it? |
| `BINARY` is bytes | Do not call it native until the compiler contract identifies the format. |
| CPU launch is one | Are logical outputs represented by an internal loop? |
| PTX has `%ctaid.x` | Which `SPECIAL` and global dimension produced it? |
| C has multiply/add text | The external compiler may still fuse or rewrite it; inspect native output if relevant. |
| Direct PTX has `fma.rn.f32` | This is a virtual-ISA witness, not yet a SASS or performance claim. |
| Mock artifact plus Python result | Keep claims separate: mock structure, Python semantics. |
| Compiler rejects valid-looking source | Save exact source, target flags, diagnostic, wrapper, and tool version. |
| Exact source golden changes | Decide whether semantics changed or only naming/formatting. |
| Second build skips work | Inspect in-process and persistent compiler caches. |
| Failure only under register pressure | Inspect ISA selection, live ranges, spills/fills, and ABI. |

```text
scheduled SINK
  -> target-aware full rewrite
  -> lowered SINK + ProgramInfo
  -> optional native instruction selection
  -> priority-aware linearization
  -> LINEAR
  -> text render ---------------------> SOURCE -- compiler --> BINARY
       or native regalloc/assembly ---> SOURCE + BINARY
  -> TinyELF(name, target, signature, BINARY)
  -> runtime loading and launch (Chapter 12)
```

## Optional background

Use these after the core artifact walk, not as prerequisites:

- The bounded [NVIDIA code-generation and runtime route](../reference/learning-resources.md#nvidia-code-generation-and-runtime-work)
  introduces virtual ISA, native ISA, module loading, and driver terminology.
- NVIDIA's [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  is the primary reference for `.entry`, registers, address spaces, instructions,
  and PTX versions.
- The [CUDA compilation workflow](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/contents.html)
  helps separate CUDA C, PTX, cubin, fatbin, and native code.
- The [System V AMD64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
  is useful only when a CPU calling-convention or relocation question becomes
  concrete.
- Any compiler text on topological ordering, instruction selection, liveness,
  and linear-scan register allocation can deepen the native-ISA section. Learn
  those topics around a small failing instruction sequence rather than reading
  an entire compiler course first.

## Deliberate deferrals

This chapter establishes artifact identity and renderer/compiler boundaries. It
deliberately defers:

- allocation handles, module/program loading, argument packing, queues,
  synchronization, and launch to Chapter 12;
- capture and replay of compiled calls to Chapter 13;
- physical `CUDA`, `CUDA:PTX`, `NV`, driver, `ptxas`, cubin, and SASS workflows
  to Chapter 14;
- cross-layer debugging to Chapter 15;
- complete test-matrix design to Chapter 16;
- compiler/runtime/performance measurement to Chapter 17; and
- upstream contribution packaging to Chapter 18.

Also defer exhaustive C syntax, PTX instructions, ELF relocation formats,
calling conventions, register-allocation algorithms, and vendor compiler flags
until a concrete contribution needs them. You now know where each topic enters
the pipeline and which artifact to bring when you study it.

[← Lowering a kernel](10-lowering.md) · [Next: Devices and runtimes →](12-runtime.md)
