# Glossary

Definitions describe how terms are used in this guide and in the recorded
tinygrad snapshot. They are deliberately operational: each should help you know
what to inspect or measure.

| Term | Working meaning |
| --- | --- |
| ABI | The binary-level agreement for calling a program: argument layout, types, registers/memory, and related conventions. |
| Address space | A class of storage with different access and lifetime rules, such as global, local/shared, or register storage. |
| Affine expression | An index expression built from sums and constant multiples of variables. Division, modulo, and validity conditions make many real indices only partly affine. |
| Allocation | Obtaining physical or virtual storage for a `Buffer`; distinct from merely representing a buffer in a graph. |
| Arithmetic intensity | Useful arithmetic operations per byte moved. It helps predict whether compute throughput or memory bandwidth limits a kernel. |
| AST | “Abstract syntax tree,” used loosely in tinygrad for a kernel-rooted UOp graph even though the structure is normally a DAG. |
| Backend | Device/runtime implementation selected by names such as `CPU`, `CUDA`, `NV`, or `AMD`. A backend is not the same thing as a renderer. |
| Barrier | Workgroup/device synchronization inserted to make memory effects visible in the required order. |
| Beam search | Measured search over equivalent kernel schedules; candidates are compiled and timed rather than chosen only by a fixed heuristic. |
| Bounds | Known minimum and maximum values for a symbolic expression, used to simplify indices and prove validity. |
| Bufferization | Choosing where a computed value must become stored data and introducing explicit buffer/memory operations. |
| Callification | tinygrad's normalization step that turns lazy tensor expressions and concrete storage into parameterized `CALL`-oriented graphs. |
| Canonicalization | Rewriting multiple equivalent forms into a preferred form so later rules and comparisons see fewer cases. |
| Code generation | The broad process from kernel graph to executable program; it includes optimization, lowering, rendering, and compilation. |
| Coalescing | Arranging adjacent GPU lanes to access adjacent or suitably grouped memory addresses. |
| Command queue | An ordered device-visible stream of launches, copies, waits, and signals, often submitted asynchronously. |
| Common subexpression elimination | Reusing one representation or computation for repeated equivalent expressions. UOp interning provides an important form of structural reuse. |
| Compile time | Host time spent transforming, rendering, and compiling before execution; separate from kernel duration and launch overhead. |
| DAG | Directed acyclic graph. UOps share sources, so drawing them as a tree can duplicate nodes and hide identity. |
| Decomposition | Replacing an operation or dtype unsupported by a target with supported lower-level operations. |
| Differential test | Run equivalent work through two implementations or backends and compare their results or artifacts. |
| Fusion | Combining operations into one kernel, normally avoiding intermediate memory traffic but potentially increasing other costs. |
| Graph rewrite | Pattern-directed replacement of UOp subgraphs while preserving an intended semantic property. |
| Grid / workgroup | Hierarchical GPU launch dimensions. CUDA commonly calls a workgroup a thread block. |
| Hash-consing / interning | Returning the same object for structurally identical immutable nodes. In tinygrad this makes object identity meaningful for many UOps. |
| HCQ | tinygrad's hardware-command-queue abstraction used by lower-level device paths and graph replay. |
| IR | Intermediate representation: a machine-processable program form between user intent and executable code. tinygrad uses multiple UOp graph states rather than unrelated node classes for every stage. |
| JIT capture | Recording and parameterizing a realized execution plan so later calls can replay it with much less Python/compiler work. |
| Kernel | One accelerator program launch. A fused kernel may implement many user-visible tensor operations. |
| Kernel AST | The `SINK`-rooted UOp graph describing one kernel before it becomes a rendered/compiled `PROGRAM`. |
| Kernel schedule | Loop/layout optimization choices inside one kernel. Do not confuse this with the ordered multi-kernel `LINEAR` execution schedule. |
| Liveness | The interval during which a value or buffer may still be used. It informs register allocation and reusable memory planning. |
| Local memory | GPU workgroup-shared on-chip memory (CUDA “shared memory”), not thread-private spill memory despite vendor terminology differences. |
| Lowering | Replacing a higher-level valid representation with a more target-specific or explicit valid representation. |
| Materialization | Computing a lazy value into storage, creating a boundary that prevents fusion across that value. |
| Memory hierarchy | Storage levels with different capacity, latency, bandwidth, visibility, and management: registers, shared/L1, L2, device memory, and host memory. |
| Occupancy | How many warps/workgroups can reside on a GPU execution unit relative to its limit; registers and shared memory often constrain it. High occupancy is a means, not the goal itself. |
| Process replay | tinygrad's comparison of compiler-process inputs and generated kernels across revisions, especially useful for refactors and speed changes. |
| Rangeification | tinygrad's conversion of shape/movement semantics into explicit iteration ranges and index expressions while forming kernel work. |
| Realization | Triggering computation so a lazy tensor obtains backed storage. |
| Register allocation | Mapping temporary values to a finite target register set and, when necessary, spills. |
| Renderer | Target-specific conversion from lowered UOps to source text or native instruction representation. |
| Rewrite fixed point | A state in which applying the relevant rewrite rules produces no further replacement. Termination and rule ordering matter when seeking it. |
| Roofline model | A bound comparing peak compute with bandwidth times arithmetic intensity; useful for rejecting impossible performance expectations. |
| Runtime | Device-specific memory, program loading, launch, copy, synchronization, and optional graph machinery. |
| Side effect | Observable state change such as a buffer write or device submission; dependencies must remain ordered even when pure expressions are freely rewritten. |
| SIMT | Single-instruction, multiple-thread execution: GPU lanes execute in groups (warps/wavefronts) with masking/divergence behavior. |
| Symbolic value | A value represented by an expression plus constraints/bounds rather than one fixed integer, commonly used for shapes and launch parameters. |
| Tensor core | A hardware matrix multiply-accumulate unit with strict shape, layout, and dtype requirements. |
| Topological order | An ordering in which every dependency appears before its consumer. Several tinygrad stages rely on it for traversal or execution. |
| UOp | tinygrad's compact, interned IR node: operation, dtype, source tuple, argument, and tag. |
| Validity mask / gate | A boolean condition under which an indexed load/store or value is valid, used to represent padding and bounds safely. |
| Vectorization / upcast | Performing several scalar lanes as one vector-shaped value or instruction to improve reuse and instruction efficiency. |
| View / movement op | A shape/index reinterpretation such as reshape, permute, expand, pad, shrink, or flip; ideally no data moves until indexing is lowered. |
| Warp | NVIDIA's group of 32 lanes that issue together. Other vendors use related but not identical terms and widths. |
| WAR / RAW dependency | Write-after-read and read-after-write ordering hazards. Scheduling must preserve them when buffers are reused or mutated. |
