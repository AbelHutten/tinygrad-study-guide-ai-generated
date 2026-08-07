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
| Arithmetic intensity | A named operation count divided by bytes at a named traffic level. Algorithmic bytes, tinygrad's static estimates, cache traffic, and DRAM traffic produce different intensities, so state both numerator and denominator before using a roofline argument. |
| Artifact | A representation or output preserved at a pipeline boundary, such as a lazy UOp graph, scheduled `LINEAR`, kernel `SINK`, generated `SOURCE`, compiled `BINARY`, launch description, or measured sample set. |
| AST | “Abstract syntax tree,” used loosely in tinygrad for a kernel-rooted UOp graph even though the structure is normally a DAG. |
| Backend | Device/runtime implementation selected by names such as `CPU`, `CUDA`, `NV`, or `AMD`. A backend is not the same thing as a renderer. |
| Baseline / candidate | The comparison states before and with a proposed change. Identify both by revision, dirty diff, workload, route, and environment rather than treating two unlabeled runs as sufficient. |
| Barrier | Synchronization among the participating workers at the barrier's execution scope, ordinarily one GPU workgroup for a kernel barrier, together with the required memory ordering. It is not a device-wide synchronize, host wait, or cross-workgroup rendezvous. |
| Beam search | Measured search over equivalent kernel schedules; candidates are compiled and timed rather than chosen only by a fixed heuristic. |
| Bottleneck | The mechanism currently limiting a named metric, not merely the largest-looking isolated duration. Removing it can expose a different limit. |
| Bounds | Known minimum and maximum values for a symbolic expression, used to simplify indices and prove validity. |
| Bufferization | Choosing where a computed value must become stored data and introducing explicit buffer/memory operations. |
| Callification | tinygrad's normalization step that turns lazy tensor expressions and concrete storage into parameterized `CALL`-oriented graphs. |
| Canonicalization | Rewriting multiple equivalent forms into a preferred form so later rules and comparisons see fewer cases. |
| Code generation | The broad process from kernel graph to executable program; it includes optimization, lowering, rendering, and compilation. |
| Coalescing | Arranging adjacent GPU lanes to access adjacent or suitably grouped memory addresses. |
| Command queue | An ordered device-visible stream of launches, copies, waits, and signals, often submitted asynchronously. |
| Common subexpression elimination | Reusing one representation or computation for repeated equivalent expressions. UOp interning provides an important form of structural reuse. |
| Compile time | Not one universal interval. Name whether the timer includes Tensor construction, scheduling, rewrites/search, rendering, compiler invocation, cache lookup, or device/toolchain initialization. Keep it distinct from kernel duration and submission unless measuring deliberately cold end-to-end latency. |
| DAG | Directed acyclic graph. UOps share sources, so drawing them as a tree can duplicate nodes and hide identity. |
| Decomposition | Replacing an operation or dtype unsupported by a target with supported lower-level operations. |
| Differential test | Run equivalent work through two implementations or backends and compare their results or artifacts. |
| First bad artifact | The earliest downstream artifact that violates a required contract while the preceding artifact still satisfies it. It localizes the producing transition more reliably than the final exception frame. |
| First costly artifact | The earliest decision or artifact that evidence shows can account for a measured performance cost. A slow downstream kernel can be the consequence of an earlier fusion, scheduling, or submission decision. |
| Fusion | Combining operations into one kernel, normally avoiding intermediate memory traffic but potentially increasing other costs. |
| Graph rewrite | Pattern-directed replacement of UOp subgraphs while preserving an intended semantic property. |
| Grid / workgroup | Hierarchical GPU launch dimensions. CUDA commonly calls a workgroup a thread block. |
| Hash-consing / interning | Returning the same object for structurally identical immutable nodes. In tinygrad this makes object identity meaningful for many UOps. |
| HCQ | tinygrad's hardware-command-queue abstraction used by lower-level device paths and graph replay. |
| IR | Intermediate representation: a machine-processable program form between user intent and executable code. tinygrad uses multiple UOp graph states rather than unrelated node classes for every stage. |
| JIT capture | Recording and parameterizing a realized execution plan so later calls can replay it with much less Python/compiler work. |
| Kernel | A compute program or function. tinygrad uses the term for a fused compute unit across backends; on an accelerator, parallel workers execute it. A launch is one invocation, not the program itself. |
| Kernel AST | The `SINK`-rooted UOp graph describing one kernel before it becomes a rendered/compiled `PROGRAM`. |
| Kernel schedule | Loop/layout optimization choices inside one kernel. Do not confuse this with the ordered multi-kernel `LINEAR` execution schedule. |
| Launch | One invocation of an accelerator kernel with particular arguments and dimensions. One compiled kernel may be launched many times. |
| Latency | Elapsed time from a declared start to completion of one named unit. On asynchronous devices, completed latency requires an end condition that establishes device completion. |
| Liveness | The interval during which a value or buffer may still be used. It informs register allocation and reusable memory planning. |
| Local memory | GPU workgroup-shared on-chip memory (CUDA “shared memory”), not thread-private spill memory despite vendor terminology differences. |
| Lowering | Replacing a higher-level valid representation with a more target-specific or explicit valid representation. |
| Materialization | Computing a logical value into storage, ordinarily establishing a storage/fusion boundary. Virtual values can be special cases without an ordinary allocation. |
| Memory hierarchy | Storage levels with different capacity, latency, bandwidth, visibility, and management: registers, shared/L1, L2, device memory, and host memory. |
| Observation | The value, artifact, event, or interval a test or diagnostic actually exposes. An unobserved property is not established merely because the run passed. |
| Occupancy | How many warps/workgroups can reside on a GPU execution unit relative to its limit; registers and shared memory often constrain it. High occupancy is a means, not the goal itself. |
| Oracle | An independently justified expected answer or invariant against which an observation is compared. Another implementation is useful only to the extent that it does not share the suspected defect. |
| Process replay | Captured kernel-generation inputs from the change branch are regenerated on a comparison revision and generated `SOURCE` is diffed. It enumerates unexpected compiler-output changes for that corpus; it does not execute those programs or prove numerical correctness, runtime behavior, or performance. |
| Rangeification | tinygrad's conversion of shape/movement semantics into explicit iteration ranges and index expressions while forming kernel work. |
| Realization | tinygrad's operation for running required work and ensuring backing where needed. Virtual values may need no ordinary allocation, and realization alone does not prove host synchronization. |
| Register allocation | Mapping temporary values to a finite target register set and, when necessary, spills. |
| Regression test | A focused test that is shown to fail on the unmodified defective baseline, pass after the fix, and is retained to detect the behavior returning. Red/green alone does not prove the fix covers every input or is at the owning layer. |
| Renderer | Target-specific conversion from lowered UOps to source text or native instruction representation. |
| Rewrite fixed point | A state in which applying the relevant rewrite rules produces no further replacement. Termination and rule ordering matter when seeking it. |
| Roofline model | A bound comparing peak compute with bandwidth times arithmetic intensity; useful for rejecting impossible performance expectations. |
| Runtime | Broad name for device-execution machinery. In this snapshot, `Buffer`/`Allocator` own storage and copies, `Program` loads and invokes compiled work, and the compiled device exposes synchronization and optional graph support. |
| Side effect | Observable state change such as a buffer write or device submission; dependencies must remain ordered even when pure expressions are freely rewritten. |
| SIMT | Single-instruction, multiple-thread execution: GPU lanes execute in groups (warps/wavefronts) with masking/divergence behavior. |
| SPEC | tinygrad's structural-legality validation mode. Nonzero values check selected schedule/codegen boundaries; in this snapshot `SPEC=2` also strengthens per-UOp and boundary checks with documented exceptions. It is not a numerical oracle. |
| Spike | A time-boxed, explicitly disposable investigation used to answer a feasibility or ownership question before committing to a merge-quality design. |
| Submission | Host-side work that enqueues launches, copies, signals, or graph execution. Submission return need not mean device completion. |
| Symbolic value | A value represented by an expression plus constraints/bounds rather than one fixed integer, commonly used for shapes and launch parameters. |
| Tensor core | A hardware matrix multiply-accumulate unit with strict shape, layout, and dtype requirements. |
| Throughput | Completed useful units divided by elapsed time, such as tokens/s or images/s. Name the unit and completion boundary; batching and overlap can improve throughput without reducing single-item latency. |
| Topological order | An ordering in which every dependency appears before its consumer. Several tinygrad stages rely on it for traversal or execution. |
| UOp | tinygrad's compact, interned IR node: operation, dtype, source tuple, argument, and tag. |
| Validity mask / gate | A boolean condition under which an indexed load/store or value is valid, used to represent padding and bounds safely. |
| Vectorization / upcast | Performing several scalar lanes as one vector-shaped value or instruction to improve reuse and instruction efficiency. |
| View / movement op | A shape/index reinterpretation such as reshape, permute, expand, pad, shrink, or flip; ideally no data moves until indexing is lowered. |
| Warm-up | Deliberate work outside the measured sample region used to reach a declared allocation, compilation, JIT, cache, or device state. It is not permission to discard inconvenient measured samples. |
| Warp | NVIDIA's group of 32 lanes that issue together. Other vendors use related but not identical terms and widths. |
| WAR / RAW dependency | Write-after-read and read-after-write ordering hazards. Scheduling must preserve them when buffers are reused or mutated. |
