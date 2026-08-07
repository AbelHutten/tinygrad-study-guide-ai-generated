# Curriculum map

The course moves from observable Tensor behavior toward progressively lower
compiler and machine layers, then returns upward through debugging, testing,
measurement, and contribution design. Read the chapters in order on the first
pass; use this page as a map afterward. The
[guide-usage contract](how-to-use.md) explains how to handle source stops,
background gaps, labs, and evidence.

The bundled course work uses the guide repository together with
`tinygrad-study`, detached at the recorded commit. A real candidate is always
reproduced again in the separate, current `tinygrad-work` checkout. Never make
the pinned study tree double as a contribution branch, and never treat a lab
result at the snapshot as proof about current upstream state.

## Phase 1 — Establish a feedback loop

1. **[Orientation and vocabulary](01-orientation.md)** — build compiler, GPU,
   graph, kernel, and runtime vocabulary from familiar Python and ML examples;
   learn what each artifact can and cannot tell you.
2. **[Development setup](02-setup.md)** — create the guide/study/work directory
   model, pin the detached study checkout, prove interpreter and source
   identity, and establish portable, CPU, CUDA, and NV feedback routes without
   collapsing driver, compiler, runtime, and hardware failures together.
3. **[Trace one expression](03-first-trace.md)** — follow one lazy Tensor
   expression through graph construction, planning, program artifacts, and
   execution before studying each subsystem separately.

Exit test: from the correct repository and interpreter, use artifacts from your
own pinned run to explain why a lazy expression becomes one or more executable
programs, which boundary triggers realization, and which parts of the result
are backend-specific.

## Phase 2 — Learn tinygrad's graph language

4. **[Tensor frontend and autograd](04-tensor-and-autograd.md)** — connect
   familiar Tensor calls, laziness, gradient construction, broadcasting, and
   mutation to the UOps they create.
5. **[UOp graphs](05-uops.md)** — understand node identity, dtype, shape,
   sources, interning, traversal, replacement, and the roles represented by
   `Ops`.
6. **[Pattern matching and rewriting](06-rewrites.md)** — learn matching,
   bindings, ordering, fixed points, termination hazards, semantic guards, and
   how to test a small rewrite rather than merely observe that it fires.

Exit test: state a simplification's contract, implement it at the owning graph
layer, and demonstrate focused positive, negative, dtype, shape, and semantic
checks—including a case that would catch an over-broad match.

## Phase 3 — Turn tensor intent into kernels

7. **[Scheduling and realization boundaries](07-scheduling.md)** — explain
   fusion, bufferization, dependencies, mutation ordering, memory planning, and
   the transition from lazy graph to executable calls.
8. **[Shapes, views, indexing, and symbolic values](08-shapes-and-indexing.md)**
   — translate movement and broadcasting into coordinate maps, validity
   predicates, ranges, and addresses, including bounded symbolic reasoning.
9. **[Kernel optimization](09-kernel-optimization.md)** — distinguish semantic
   rewrites from equivalent schedule choices and reason about local memory,
   upcasting, unrolling, tensor cores, padding, heuristics, and search.
10. **[Lowering and linearization](10-lowering.md)** — follow a scheduled
    kernel into ranges, addresses, loads/stores, accumulators, barriers,
    register-like values, and ordered `LINEAR` control.
11. **[Rendering and compilation](11-rendering.md)** — see how target
    capabilities constrain legal programs, how ordered operations become
    source, and how source becomes the payload a runtime can load.

Exit test: for one wrong or unexpectedly costly generated program, compare
adjacent artifacts and identify the earliest representation where a stated
invariant fails or the suspected cost first appears; justify the next focused
experiment without blaming the layer that merely reported the symptom.

## Phase 4 — Cross the machine boundary

12. **[Devices, allocators, programs, and runtimes](12-runtime.md)** — learn
    device selection, allocation and view lifetimes, compilation/loading,
    argument roles, dispatch, synchronization, and the shared contracts behind
    different backends.
13. **[JIT and graph execution](13-jit.md)** — distinguish ordinary calls,
    compilation caching, ignore/capture/replay phases, input rebinding,
    symbolic values, lifetime rules, and optional device-graph batching.
14. **[NVIDIA on Ubuntu](14-nvidia.md)** — build the required CUDA, PTX, driver,
    queue, and Ada hardware model before comparing tinygrad's CUDA and
    lower-level NV paths.

Exit test: trace one kernel's buffer allocation, compiled artifact, argument
packing, launch, completion, and synchronization on a named backend; separate
facts observed on a physical device from structural evidence produced by an
emulated or renderer-only route.

## Phase 5 — Build a reviewable contribution claim

15. **[Debugging across the pipeline](15-debugging.md)** — turn a symptom into
    a deterministic reproducer and independent oracle, freeze the experiment,
    compare adjacent frontend/schedule/`LINEAR`/`SOURCE`/`BINARY`/runtime/JIT
    artifacts, and name the last good artifact, first bad artifact, and owning
    layer. The lab uses a narrow artificial renderer fault rather than claiming
    an upstream bug.
16. **[Testing a contribution](16-testing.md)** — turn the localized contract
    into a detector, prove that the same focused test is red for the intended
    baseline defect and green after correction, add boundary cases and
    independent oracles, and choose broader, fuzz, replay, backend, and
    hardware checks in proportion to the claim.
17. **[Performance engineering](17-performance.md)** — define a falsifiable
    workload/layer/metric claim, bracket timing with correctness, control
    compilation, cache, JIT, synchronization, and device state, retain raw
    sample distributions, localize the first costly artifact, and reconnect a
    local improvement to end-to-end impact without inventing a speedup.
18. **[From idea or bounty to a reviewable contribution](18-contributing.md)**
    — select a current bounty, issue, or self-chosen improvement; recheck live
    policy, ownership, overlap, source, tests, and history; decide Ready,
    Research, Question, or Decline; isolate work; design atomic commits; bound
    performance and hardware claims; and prepare review, provenance,
    communication, and rollback evidence. Its lab contrasts an incomplete
    patch idea with a complete packet for the artificial Chapter 15 case; it
    neither edits tinygrad nor certifies a real candidate.

Exit test: in `tinygrad-work`, choose one current bounty, issue, **or
self-chosen observation** and complete the
[contribution brief](../reference/contribution-brief.md). Record, with UTC
timestamps where state is live:

- candidate origin, policy/ownership/issue/PR overlap, current base, intended
  contract, success criteria, non-goals, and a Ready/Research/Question/Decline
  decision;
- a minimal reproducer or benchmark, independent oracle, baseline result, and
  the last good plus first bad or costly artifact and owning layer;
- current source, neighboring tests, and relevant history;
- the nearest regression, exact red-before/green-after evidence when a defect
  is claimed, proportional wider validation, and explicit unrun checks;
- bounded performance and hardware claims, risks, rollback signals, atomic
  commit plan, communication/stop policy, license/provenance treatment, and AI
  disclosure.

Passing does not require forcing the candidate into a patch or PR. A well
supported Research, Question, or Decline decision demonstrates better
contribution judgment than a speculative implementation. If the decision is
Ready, another person should be able to reproduce the claim and understand
exactly what remains unproved before reviewing code.

## What “complete” means

Completion is demonstrated, not timed. You are ready to approach unfamiliar
tinygrad work when you can pass the five phase exit tests, move deliberately
between the pinned study system and current upstream, and navigate an unknown
subsystem through its data structures, artifacts, tests, history, and
interfaces.

You do not need to memorize every `Ops` member, optimization, renderer, device
register, or bounty-specific hardware fact. You do need to recognize when one
matters, locate a bounded authoritative resource, perform a return exercise,
and update or limit the contribution claim. Some candidates will legitimately
require hardware, domain knowledge, or maintainer clarification you do not yet
have; the workflow tells you how to acquire it or when to stop rather than
treating the rest of the compiler as a black box.

[Start Phase 1: Orientation and vocabulary →](01-orientation.md)
