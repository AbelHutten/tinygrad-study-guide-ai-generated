# Curriculum map

The course moves from observable behavior toward progressively lower layers,
then returns upward through debugging, performance work, and contribution
design.

## Phase 1 — Establish a feedback loop

1. **Orientation and vocabulary** — relate familiar ML operations to compiler
   and runtime concepts.
2. **Development setup** — create reproducible CPU, Python, CUDA, and NV
   workflows; learn the test hierarchy.
3. **Trace one expression** — follow a tiny tensor expression from Python to a
   launched program before studying each subsystem in isolation.

Exit test: explain, using artifacts from your own run, why a lazy expression
eventually becomes one or more device launches.

## Phase 2 — Learn tinygrad's graph language

4. **Tensor frontend and autograd** — see how familiar API calls construct and
   transform UOps.
5. **UOp graphs** — understand node identity, dtype, shape, sources, graph
   traversal, and the roles represented by `Ops`.
6. **Pattern matching and rewriting** — read, test, and write a small rewrite;
   learn termination, ordering, and validation hazards.

Exit test: add a correct simplification rule with focused positive, negative,
and dtype/shape tests.

## Phase 3 — Turn tensor intent into kernels

7. **Scheduling and realization boundaries** — understand fusion, bufferization,
   kernel dependencies, memory planning, and the `LINEAR` execution plan.
8. **Shapes, views, indexing, and symbolic values** — follow movement operations
   as they become ranges and address calculations.
9. **Kernel optimization** — distinguish semantic rewrites from equivalent
   schedule choices; understand local memory, upcasting, unrolling, tensor
   cores, heuristics, and search.
10. **Lowering and linearization** — follow a kernel AST into control flow,
    loads/stores, register allocation, and a linear instruction-like form.
11. **Rendering and compilation** — see how target capabilities select rewrites
    and how a renderer's output becomes a loadable binary.

Exit test: take a generated kernel that is incorrect or slow, identify the
earliest pipeline stage at which the problem exists, and justify the next test
to run.

## Phase 4 — Cross the machine boundary

12. **Devices, allocators, programs, and runtimes** — understand the stable
    runtime interfaces before reading a backend.
13. **JIT and graph execution** — distinguish compilation caching, capture,
    replay, symbolic inputs, and batched command submission.
14. **NVIDIA path** — compare the CUDA backend with tinygrad's lower-level NV
    path; learn only the CUDA and hardware concepts needed to inspect an Ada
    execution.

Exit test: trace buffer allocation, compilation, argument packing, launch, and
synchronization for one kernel on your selected backend.

## Phase 5 — Work like a contributor

15. **Debugging and visualization** — reduce failures, inspect rewrite history,
    compare backends, and localize the first bad representation.
16. **Testing and fuzzing** — choose unit, differential, property, process
    replay, and hardware tests proportional to a change.
17. **Performance engineering** — make a falsifiable bottleneck claim, benchmark
    correctly, and account for compile, model, kernel, and launch speed.
18. **From idea or bounty to PR** — study project intent and history, propose a
    minimal change, measure it, and prepare evidence that survives review.

Exit test: produce a contribution brief for a real current issue or bounty,
including scope, subsystem map, missing prerequisites, reproducer, proposed
tests, performance or correctness evidence, and rollback criteria.

## What “complete” means

Completion is demonstrated, not timed. You are ready to begin a contribution
when you can pass the five phase exit tests and can navigate an unfamiliar
subsystem by following its data structures, tests, history, and interfaces.

You do not need to memorize every `Ops` member, optimization, renderer, or
device register. You do need to recognize when one matters and know how to
learn it without treating the rest of the compiler as a black box.

