# Learning resources

Use this page as a router, not a reading list. Follow a resource when a chapter's
prerequisite gate identifies a concrete gap, stop at the stated outcome, and
return to tinygrad while the connection is fresh.

## tinygrad sources and references

These are the highest-value companions to the guide:

| Resource | Use it for | Caution |
| --- | --- | --- |
| [Official documentation](https://docs.tinygrad.org/) | Current Tensor/API signatures, dtypes, environment variables, runtime inventory, and short developer summaries | Developer pages are compressed references, not a beginner curriculum. |
| [Source tree at this guide's snapshot](https://github.com/tinygrad/tinygrad/tree/874d33128b4e4785beea736d97df6716e0321717) | Reproducing exact source tours and lab behavior | Return to live `master` before developing a contribution. |
| [`spec/tinyspec.pdf`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/spec/tinyspec.pdf) | Formal inventory of the UOp dialect, derived properties, common compositions, optimization vocabulary, sharding, and lowering | Dense and reference-oriented. Read after the UOp chapter, not before it. |
| [`docs/abstractions3.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/abstractions3.py) | A compact front-to-back training example that reaches `schedule_linear` and `run_linear` | Read it line by line only after tracing the smaller expression in this guide. |
| [`docs/abstractions4.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/abstractions4.py) | Advanced custom UOp, BEAM, HIP, and assembly kernels | AMD/RDNA3-specific in important sections; it is not a generic next step. |
| [`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md) | Current rewrite-trace and profiling CLI commands | Match the README to your checkout because output and flags evolve. |
| [`test/external/process_replay/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md) | Comparing generated kernels across a refactor or optimization | It complements focused correctness tests; it does not replace them. |

### Historical community notes

[Di Zhu's tinygrad notes](https://mesozoic-egg.github.io/tinygrad-notes/)
contain helpful explanations of kernels, ShapeTracker-era layouts, pattern
matching, fusion, BEAM, and JIT. Treat each article as a historical/conceptual
supplement. Many pages describe 2024–2025 source layouts and removed APIs, so
translate the idea through `rg`, current tests, and the
[recorded snapshot](source-snapshot.md) rather than copying code.

The same warning applies more strongly to `docs/tinygrad_intro.pdf` in upstream:
it is useful project history, but names such as `LazyBuffer`, `ScheduleItem`, and
older UOp families do not describe this guide's 2026 snapshot.

## Compiler bridge for an ML reader

### First: see a familiar computation as IR

[JAX's jaxpr documentation](https://docs.jax.dev/en/latest/jaxpr.html) gives the
shortest bridge from array programs to a typed equation graph.

Stop when you can take a small jaxpr and identify inputs, constants, primitive
operations, data dependencies, and outputs. You do not need to learn JAX
transformation internals. Then compare that representation with a lazy UOp DAG:
the concepts transfer, but tinygrad's node set and lifecycle differ.

### Next: transformations and lowering

Use the [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) to see why
a compiler uses multiple abstraction levels and lowers between them. Chapters
introducing an AST, a custom dialect, transformations, and lower-level code
generation are the useful portion.

Stop when you can explain:

- why one IR is not ideal for every analysis and target;
- the difference between canonicalization and lowering;
- why legality must be defined at each stage; and
- how source locations and diagnostics help debug a transformation.

tinygrad often reuses the `UOp` class where MLIR would use visibly distinct
dialects. Do not force a one-to-one analogy.

For rewrite mechanics and discipline, use MLIR's
[Pattern Rewriter documentation](https://mlir.llvm.org/docs/PatternRewriter/).
Focus on matching, replacing, rewrite benefits/order, bounded recursion, and
termination. Then return to tinygrad's much smaller `UPat` and `PatternMatcher`
implementation.

### For loop and tensor scheduling intuition

Apache TVM's
[TensorIR creation tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
makes loops, blocks, buffers, and indices explicit.

Stop when you can translate a simple elementwise operation and a reduction into
loop nests with buffer reads and writes. TVM's APIs and scheduling model are not
tinygrad's; the value is learning to see tensor algebra as iteration plus
memory access.

Keep the [LLVM IR language reference](https://llvm.org/docs/LangRef.html) as a
later lookup source. It is useful when inspecting LLVM output or ABI details,
but reading it front to back is not an onboarding prerequisite.

## GPU execution on the RTX 4090 path

### Execution and memory model

Use NVIDIA's
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
Learn only enough initially to explain:

- host versus device code and asynchronous launches;
- grids, thread blocks, threads, warps, and lane indices;
- global, shared, local, constant, and register storage;
- synchronization scope and why a barrier is required;
- coalescing and divergence; and
- compute capability as a target contract.

The RTX 4090 is Ada, compute capability 8.9 (`sm_89`). NVIDIA's
[Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/) describes the
architecture-specific limits and features. Use it after the general execution
model, and look up a limit only when a generated kernel makes it relevant.

### Write two kernels elsewhere first

The [Triton vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
and [fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
offer a productive bridge from Python tensor code to explicit program IDs,
blocked indexing, masks, fusion, and bandwidth reasoning.

Stop after you can predict each address a program instance touches and explain
why fusion reduces traffic. The objective is not to learn Triton's compiler or
adopt its programming model inside tinygrad.

## GPU performance

NVIDIA's
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
is the default reference for transfers, coalescing, shared memory, occupancy,
and measurement. Pair it with the
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
only after a benchmark identifies one slow kernel.

Before returning to tinygrad performance work, be able to distinguish:

- latency from throughput;
- useful bytes from actual memory traffic;
- bandwidth-bound from compute-bound behavior;
- occupancy from achieved performance;
- register pressure from shared-memory use;
- warm-up/compilation time from steady-state execution; and
- a measured bottleneck from an optimization hunch.

For roofline reasoning, the conceptual requirement is small: performance cannot
exceed either peak compute or bandwidth multiplied by arithmetic intensity.
Use measured or vendor-documented numbers for the actual device and dtype; do
not copy a headline throughput number into a benchmark claim.

## NVIDIA code generation and runtime work

These are references for specialized chapters, not prerequisites for the main
compiler path:

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — instruction
  semantics, state spaces, directives, and target versions when reading or
  changing PTX rendering.
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) — contexts,
  modules, memory, streams, events, and launches when reading the `CUDA`
  backend.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) —
  `cuobjdump`, `nvdisasm`, binary formats, and resource information when source
  output is no longer enough.

For the lower-level `NV` backend, these CUDA references explain the program
being launched but not tinygrad's direct queue and userspace-driver machinery.
That work additionally requires targeted study of command formats, virtual
memory, synchronization, and the relevant open or generated hardware headers.

## Testing transformations

Use the [Hypothesis documentation](https://hypothesis.readthedocs.io/) when a
bug describes a family of shapes, dtypes, or symbolic expressions rather than
one example. Stop when you can define a constrained generator, state an
invariant, reproduce a minimized failure, and keep the test deterministic.

For tinygrad, combine techniques according to the risk:

- focused example test for the reported regression;
- differential comparison against another backend or NumPy/PyTorch;
- property/fuzz test for the broader input family;
- spec validation for illegal IR states;
- process replay for unexpected generated-kernel changes; and
- real-hardware tests for backend behavior that emulation cannot establish.

## Specialized contribution branches

The common course teaches you to identify when these branches are necessary; it
does not attempt to replace their domain documentation.

| Contribution area | Begin with |
| --- | --- |
| Multi-GPU collectives | [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html), then tinygrad's `schedule/allreduce.py` and multi-device tests |
| MLPerf bounty | [MLCommons Training benchmarks](https://mlcommons.org/benchmarks/training/), the exact benchmark rules/version, then tinygrad's relevant `examples/mlperf/` path |
| OpenCL/backend portability | [Khronos OpenCL resources](https://www.khronos.org/opencl/), then compare `ops_cl.py` with a simpler tinygrad runtime |
| GPU ISA/emulator | Vendor ISA manual, binary utilities, tinygrad renderer/assembler tests, then mock-GPU infrastructure |
| Userspace driver, PCIe, or USB | Linux device/virtual-memory interfaces, transport specifications, hardware command formats, then `runtime/support/` and device tests |
| Large-model throughput | Model architecture and serving workload, memory-capacity accounting, JIT/graph path, model-level profiler evidence, then kernel hotspots |

## Resource quality checklist

Before investing in any tutorial about tinygrad internals, record:

1. the tinygrad commit or publication date it describes;
2. whether its named files and symbols still exist;
3. whether its observed output reproduces on your checkout;
4. whether it teaches a durable concept or prescribes a current API; and
5. which current test encodes the behavior.

A stale tutorial can still teach an excellent compiler idea. It becomes harmful
only when historical implementation detail is mistaken for current evidence.
