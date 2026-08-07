# Contribution brief template

Complete this before substantial work on a real issue, bounty, or self-chosen
improvement. The brief is an investigation artifact, not a promise that the
first proposed fix is correct or that upstream wants it.

## Claim

```text
Candidate and origin (URL or self-chosen observation):
Current status, owner, bounty terms, and maintainer direction:
Policy checked at (UTC time and URL):
Open issue/PR overlap checked at (UTC time and queries):
Current base and reproduced commit hashes:
Falsifiable correctness, performance, or maintainability contract:
Preconditions:
Success criteria:
Explicit non-goals:
Why this belongs upstream in the affected project area now:
Evidence or upstream event that would make you stop:
Current decision (Ready | Research | Question | Decline):
Evidence that would change that decision:
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
Independent oracle and tolerance, if any:
Baseline-red result and failure reason:
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
Current source plus recent commits/PRs that explain intent:
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
Explicitly out-of-scope edits:
Atomic commit sequence and validation at each commit:
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
Explicitly unrun checks and why:
```

Mark non-applicable rows and explain why. “Full tests pass” is not a substitute
for a test that fails for the reported reason before the change.

## Performance protocol

Complete this section for every speed claim.

```text
Claimed metric and layer (compile, model, kernel, submission, other):
Acceptance threshold and observation that would falsify the claim:
Requested/canonical device, interface, backend, renderer, compiler, runtime, target:
Machine, OS, driver/toolchain/library versions, power and clock state:
Commit and exact cache/JIT/search/compiler environment controls:
CACHELEVEL/CACHEDB/CCACHE/SCACHE/BEAM/JIT state:
Workload, shapes, dtypes, model state:
Correctness oracle/tolerance:
Warm-up, allocation, compile, and cache state:
Exact timed endpoints and synchronization mechanism:
Baseline/candidate sample order or interleaving:
Number of samples and reported distribution:
Baseline result:
Patched result:
Compile/search time with stated boundaries:
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

## Communication and provenance

```text
Question or status update needed before more work:
When to ask, pause, hand off, or decline:
Copied or adapted code/data and exact source:
Third-party license and retained-notice treatment:
Employer ownership/confidentiality check:
AI/tool assistance and current-policy disclosure:
Credentials, private data, or generated artifacts excluded from the diff:
```

“I wrote the patch” is not a provenance audit. State whether an implementation
or test was copied or adapted, identify its license, and retain required notices
locally. Do not publish employer-owned or confidential material. Reopen current
upstream policy before deciding how AI/tool assistance must be disclosed.

## Review-ready summary

```text
Why merge this:
What changed:
What did not change:
Atomic commits and why each stands alone:
How correctness was established:
How performance was established (if claimed):
Known limitations:
Checks intentionally not run:
Current issue/PR overlap rechecked at:
Third-party/license/employer provenance:
AI/tool assistance disclosed as required by current upstream policy:
```

Before opening a PR, re-read the live upstream contribution section, re-check
issue/bounty ownership and overlapping PRs, update from `master`, rerun the
relevant evidence, and record what remains untested. If a premise, owner,
policy, or competing change has moved, pause and update the claim before
submitting.
