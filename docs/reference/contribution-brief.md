# Contribution brief template

Complete this before changing a real issue or bounty. The brief is an
investigation artifact, not a promise that the first proposed fix is correct.

## Claim

```text
Issue/bounty URL:
Status checked at (UTC time):
Commit reproduced:
One-sentence correctness, performance, or maintainability claim:
Why this belongs in tinygrad core:
```

A useful claim is falsifiable. “Scheduling is slow” is not; “this graph creates
two global-memory round trips where one fused kernel is legal, increasing
steady-state model time by X under the recorded setup” is.

## Minimal evidence

```text
Reproducer or benchmark path:
Exact command and environment:
Expected result:
Actual result:
Smallest known shape/dtype/device:
Control backend or revision:
First bad or costly artifact:
```

Attach only the graph/source/profile fragments needed to support the claim.
Retain raw captures outside the commit unless they are stable test fixtures.

## Pipeline location

```text
Primary subsystem:
Transformation that owns the first divergence:
Relevant source symbols:
Nearest existing tests:
Recent commits/PRs that explain intent:
Maintainer guidance already given:
```

If the primary subsystem is still “somewhere in codegen,” continue reducing and
tracing before proposing a patch.

## Prerequisite gap

```text
Specific concept or hardware fact not yet understood:
Authoritative resource selected:
Return exercise:
Evidence that the gap is now closed:
```

Examples include one PTX instruction's corner cases, one MLPerf rule version,
one queue packet format, or one collective cost model. Do not turn the brief
into an undirected reading list.

## Proposed change

```text
Smallest behavior change:
Why this layer owns it:
Expected files/symbols:
Semantic preconditions:
Alternatives rejected and why:
New complexity or lines justified by:
Independent prerequisite refactor, if any:
```

A prerequisite refactor must be a clear win without the follow-up feature. If
it is not independently valuable, keep evaluating whether the feature can be
made smaller.

## Oracle and validation

```text
Failing regression added first:
Positive cases:
Negative/edge cases:
Dtype/shape/symbolic cases:
Differential or property oracle:
Targeted test command:
Broader subsystem command:
Other backends:
SPEC/CHECK_OOB/CPU validation:
Fuzzing:
Process replay:
Static checks:
Hardware-only evidence:
```

Mark non-applicable rows and explain why. “Full tests pass” is not a substitute
for a test that fails for the reported reason before the change.

## Performance protocol

Complete this section for every speed claim.

```text
Device, driver, backend, renderer, target:
Commit and all relevant environment flags:
Workload, shapes, dtypes, model state:
Correctness oracle/tolerance:
Warm-up and cache state:
Synchronization/timing mechanism:
Number of samples and reported distribution:
Baseline result:
Patched result:
Compile time:
Model schedule/kernel count and memory traffic:
Affected kernel time and generated-code difference:
Noise floor and repeatability:
Complexity tradeoff:
```

Report the layer that improved. A faster isolated kernel that slows the model is
not a model speedup; a warm compiler cache is not a device speedup.

## Risk and rollback

```text
Representations/backends affected:
Numerical behavior that may change:
Aliasing, mutation, or async lifetime risk:
Cache/process-replay impact:
Unsupported hardware risk:
Observable signal for regression:
Simplest rollback:
```

## Review-ready summary

```text
Why merge this:
What changed:
What did not change:
How correctness was established:
How performance was established (if claimed):
Known limitations:
AI/tool assistance disclosed as required by current upstream policy:
```

Before opening a PR, re-read the live upstream contribution section, re-check
issue/bounty ownership, update from `master`, and rerun the relevant evidence.
