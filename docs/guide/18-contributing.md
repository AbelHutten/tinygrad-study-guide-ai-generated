# 18. From idea or bounty to PR

## Purpose

You are ready to begin contributing when you can turn an unfamiliar, current
problem into a small, test-backed change whose evidence survives review. You
do not need to know every subsystem in advance. You do need to recognize the
target branch of the codebase, close the specific prerequisite gaps it exposes,
find the first wrong or costly artifact, and resist changing more than the
claim requires.

This chapter provides that end-to-end workflow for either an issue/bounty or a
self-discovered improvement. It concludes with two concrete outputs: a
three-issue triage record and one real patch against current `master` with a
regression test. Opening a pull request is a separate decision made only after
rechecking ownership and live policy.

**Source snapshot:** `874d331` (2026-08-05).

**Live contribution policy last checked:** 2026-08-05. Re-read the
[current upstream contribution section](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) <!-- live-upstream -->
before acting; the date above is not a promise about today's rules.

## Prerequisite gate

Before selecting a real contribution, you should be able to:

- reproduce behavior from a clean command and record the exact commit;
- localize the first divergent artifact with the methods in
  [Debugging and visualization](15-debugging.md);
- choose focused, differential, property/fuzz, hardware, and process-replay
  evidence using [Testing and fuzzing](16-testing.md);
- write a falsifiable performance claim and comparison protocol using
  [Performance engineering](17-performance.md); and
- use Git branches or worktrees without mixing an experiment into `master`.

If one item is missing, close that bounded gap before claiming an issue. This
is not a requirement to finish every possible specialized topic. The branch
router below identifies what to learn once a candidate reveals its subsystem.

## Mental model: contribution as evidence compression

A good contribution makes a large investigation reviewable:

```text
live problem
  -> minimal reproducer and oracle
  -> first wrong/costly representation
  -> smallest owning change
  -> focused regression + proportional broader evidence
  -> concise explanation of why this belongs in tinygrad
```

The diff is the smallest part of the work. Source history, rejected hypotheses,
raw profiles, and exploratory scripts may be extensive, but the reviewer should
receive a compact claim, a focused test, a minimal patch, and enough evidence
to challenge each step.

Three kinds of information have different expiry dates:

| Kind | Examples | How to use it |
| --- | --- | --- |
| **Durable method** | Reproduce first, add a regression, change the owning layer, keep scope small | Carry this between revisions |
| **Snapshot policy** | The contribution rules and process-replay behavior at commit `874d331` | Use the pinned links to understand this guide; do not assume they are still current |
| **Live state** | Current policy, bounty terms, issue ownership, linked PRs, maintainer direction, CI | Recheck when selecting work and again immediately before opening a PR |

Never let a guide page convert live project state into folklore. Record a UTC
timestamp and exact URL whenever a contribution decision depends on it.

## Understand upstream's policy before choosing work

At the guide's pinned snapshot, upstream's
[contribution section](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/README.md#L165)
sets a deliberately high review bar. In practical terms:

- explain in a sentence or two why the change should merge and how it improves
  the project;
- disclose any AI use, for everyone, and expect especially strict treatment of
  AI-looking work from a new contributor;
- optimize for lower complexity and greater readability, not code golf or
  fewer physical lines;
- benchmark every speedup claim and justify its maintainability tradeoff;
- avoid changing non-core code merely because it could be cleaned up;
- keep diffs small enough to review, separating an independent clear-win
  prerequisite refactor when that genuinely makes the eventual feature tiny;
  and
- pair bug fixes and features with regression tests.

The snapshot explicitly says docs and whitespace changes from people who have
not already demonstrated deep codebase knowledge will be closed. This guide is
an independent learning resource, not a suggestion that a new reader submit a
documentation or cosmetic PR upstream. Demonstrate understanding with a real,
test-backed core problem.

The same snapshot identifies valuable categories: bug fixes with regressions,
high-quality bounty solutions, useful small features, clear-win refactors,
non-brittle tests/fuzzers, and dead-code removal from core. These are examples
of policy fit, not standing authorization. Verify the
[live README](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) <!-- live-upstream -->
because wording, priorities, and consequences can change.

### AI disclosure is an engineering requirement

The snapshot warns that new-contributor work which appears AI-written may be
closed without feedback and may lead to a ban; it also requires every
contributor to disclose AI use. Treat that literally. If an assistant helped
search, explain code, draft a test, write code, or prepare the PR text, state
what it did. Then personally verify every claimed source fact, execute every
reported command, understand every changed line, and rewrite anything you
cannot defend.

Disclosure does not turn generated output into evidence. A passing test you
did not see fail on the baseline is not a regression. A benchmark summary
without raw samples is not a speed claim. A confident explanation unsupported
by the checked-out source is not project knowledge.

## Find live work without freezing a bounty list

Do not copy bounty names, amounts, or status into a long-lived study guide.
Use these live surfaces each time:

- the [bounty spreadsheet linked by upstream](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing);
- a [live open-issue bounty-label query](https://github.com/tinygrad/tinygrad/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty);
- [all current issues](https://github.com/tinygrad/tinygrad/issues); and
- [current pull requests](https://github.com/tinygrad/tinygrad/pulls).

The spreadsheet is the bounty surface linked from the README. The label query
is a discovery and cross-check tool, not proof that an item is available, that
its terms are unchanged, or that every spreadsheet item has a matching label.
Search results can be empty or different from the last time this guide was
checked.

For each candidate, open the exact issue and verify:

1. it is still open and its current requested outcome is unambiguous;
2. the spreadsheet terms and issue text agree, if it is a bounty;
3. comments, assignees, linked branches/PRs, and recent maintainer messages do
   not show active or superseding work;
4. current `master` still reproduces the problem;
5. a merge or adjacent change has not made the proposed direction stale; and
6. the required evidence, hardware, data, and external rules are available to
   you.

Do not infer ownership from an empty assignee field. If live surfaces conflict
or the expected result is ambiguous, resolve that uncertainty before investing
in an implementation. Record what you checked and when in the
[contribution brief](../reference/contribution-brief.md).

## Triage before you claim

Triage is a bounded attempt to decide whether a problem is ready for focused
work, not a race to propose a fix. Start from current `master` in a clean
worktree and produce a one-line decision:

- **Ready:** reproduced, scope and oracle are clear, no visible conflict, and
  prerequisites are bounded;
- **Research:** potentially suitable, but a named source fact, specification,
  hardware result, or ownership question must be resolved first; or
- **Decline:** stale/non-reproducible, already active, outside available
  resources, poor policy fit, or too broad to make reviewable now.

For a candidate that survives the live-state check:

1. Reduce the report to the smallest shape, dtype, backend, and operation that
   preserves the failure or performance symptom.
2. State the expected result independently of the proposed implementation.
3. Find the nearest existing test and the first pipeline artifact that differs
   from a correct backend, reference implementation, prior commit, or expected
   representation.
4. Read source history with `git log -- <path>` and `git blame`; search current
   issues and PRs for the relevant symbol and failure, not only the title's
   wording.
5. List the missing prerequisite as one concrete question. Follow an
   authoritative resource, then return with a small exercise or source
   observation that closes it.
6. Sketch the smallest test and the smallest owning change. If the sketch needs
   unrelated cleanup, remove that cleanup or make a separately justified plan.

A plausible code edit is not a reason to mark a candidate Ready. The
reproducer, oracle, ownership state, and first wrong/costly artifact are.

## Route branch-specific prerequisites

The common course teaches the pipeline and investigation method. Some real
work deliberately branches into a narrower domain. Use only the route selected
by the candidate:

| Candidate area | Return to this guide | Bounded external/source prerequisite |
| --- | --- | --- |
| Tensor API, dtype, autograd | [Tensor frontend](04-tensor-and-autograd.md), [UOps](05-uops.md), [testing](16-testing.md) | Exact NumPy/PyTorch behavior when current policy expects compatibility; dtype and gradient edge cases |
| Symbolic or rewrite rule | [UOps](05-uops.md), [rewrites](06-rewrites.md), [shapes/indexing](08-shapes-and-indexing.md), [fuzzing](16-testing.md) | Prove the identity over all represented bounds/dtypes; Z3 or property-testing skills when the family is broad |
| Fusion, scheduling, memory planning | [Scheduling](07-scheduling.md), [shapes/indexing](08-shapes-and-indexing.md), [performance](17-performance.md) | Recompute/materialization cost model and alias/lifetime constraints for the exact graph |
| Kernel schedule or codegen optimization | [Kernel optimization](09-kernel-optimization.md), [lowering](10-lowering.md), [rendering](11-rendering.md), [performance](17-performance.md) | Target memory hierarchy, instruction path, resource limits, and hardware-profiler evidence |
| Renderer or compiler | [Lowering](10-lowering.md), [rendering](11-rendering.md), [testing](16-testing.md) | Target language/ISA semantics and the exact toolchain contract |
| Runtime, JIT, graph, or queues | [Runtime](12-runtime.md), [JIT](13-jit.md), [NVIDIA path](14-nvidia.md), [performance](17-performance.md) | Driver API, asynchronous lifetime, queue/event semantics, and hardware-only validation |
| NVIDIA PTX or direct NV path | [NVIDIA path](14-nvidia.md) | PTX ISA or relevant command/memory interface; use safe supported interfaces and dedicated hardware where required |
| Test or fuzzer improvement | [Testing and fuzzing](16-testing.md) | Stable oracle, constrained generator, deterministic reproduction, and failure minimization |
| Multi-GPU, MLPerf, another backend, low-level transport, or large-model throughput | [Specialized branches](../reference/learning-resources.md#specialized-contribution-branches) | Exact current standard/rules, hardware and dataset requirements, then the named tinygrad subsystem/tests |

Do not read every branch “just in case.” In the brief, write the exact unknown—
for example, “Does PTX instruction X flush subnormals for dtype Y on target Z?”—
then select the authoritative section and a return exercise. If the prerequisite
cannot be closed with available hardware, rules, or data, mark the candidate
Research or Decline instead of guessing.

## Regression-first implementation workflow

### 1. Freeze the claim, not the implementation

Create the [contribution brief](../reference/contribution-brief.md) before the
patch. Record the live URL/status timestamp, exact reproducing commit, minimal
command, expected/actual behavior, first bad or costly artifact, source symbols,
nearby tests, and rejected alternatives.

For performance work, complete the brief's entire performance protocol. For a
correctness issue, describe semantic preconditions and negative cases so the
test does not overfit one accidental shape.

### 2. Add the regression before the fix

Write the smallest stable test that expresses the public or internal contract.
Run it on unmodified current `master` and save the failure. Confirm it fails for
the reported reason—not an unavailable device, missing dependency, stale API,
random tolerance, or the assertion itself.

Then run the same test on a known-correct backend/reference or encode a
property/invariant. The before/after evidence should answer:

```text
baseline: this test fails here, for this reason
candidate: this same test passes
control: the test detects a deliberately wrong result/path
```

If the bug is understood but not yet fixable, the snapshot policy explicitly
welcomes useful non-brittle tests/fuzzers and even broken tests marked
`@unittest.expectedFailure`. Recheck live policy before choosing that route.

### 3. Change the first owning layer

Follow the first divergent artifact from Chapter 15. Make the smallest semantic
change that fixes it. Do not add downstream exceptions to hide an upstream
representation error, and do not bundle formatting, renames, or opportunistic
cleanup.

If a prerequisite refactor is needed, ask whether it is an obvious improvement
without the follow-on feature. If yes, it may deserve an independent change
with its own tests and rationale. If not, keep reducing the feature or explain
why the coupled complexity is unavoidable.

### 4. Expand validation in risk order

Run evidence outward from the changed contract:

1. the new regression alone;
2. adjacent positive, negative, dtype, shape, and symbolic cases;
3. the nearest subsystem test file;
4. differential/property/fuzz/spec checks appropriate to the representation;
5. other relevant backends and real hardware where semantics differ; and
6. broader upstream tests and pre-commit checks required by the live policy/CI.

Passing the full suite cannot compensate for a missing focused regression.
Conversely, the focused test cannot expose cross-backend or optimizer-wide
damage. Record exact commands and results; do not summarize an unrun test as
“should pass.”

For a speed claim, bracket the benchmark with correctness and follow Chapter
17's cache, synchronization, distribution, attribution, and complexity
protocol. Report compile, full-workload, and affected-kernel outcomes without
substituting one for another.

### 5. Use process replay for the change it can test

At the snapshot, [process replay](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md#L1)
captures inputs to kernel-generation processes on the contribution branch,
then replays them on `master` to diff generated kernels. It does not execute
those kernels as a numerical oracle, and by default a kernel diff is not an
assertion.

The snapshot says refactor and speedup PRs with no expected behavior change
must put `[pr]` in the PR title so process replay asserts unexpected kernel
differences. The [CI workflow](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1)
uses that title marker to enable capture for pull requests. `[pr]` means “this
change expects generated behavior to remain equivalent”; it is not a generic
badge meaning the author ran tests.

Follow the
[live process-replay instructions](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) <!-- live-upstream -->
because commands and marker semantics can change. In this snapshot, the local
sequence is:

1. optionally reset old process-replay rows;
2. run representative tests on the contribution branch with
   `CAPTURE_PROCESS_REPLAY=1`;
3. switch a clean worktree to `master`; and
4. run `test/external/process_replay/process_replay.py` there.

The provided `local.sh` switches branches. Do not run it with uncommitted work
you could lose; use a clean worktree or preserve the patch safely. Capture
representative processes, inspect every diff, and keep focused correctness
tests. A no-diff replay does not prove that runtime behavior is correct.

### 6. Rebase the evidence onto current state

Before preparing a PR, update from current `master`, resolve changes by
understanding them, and rerun the regression plus proportional broader tests.
Recheck the exact issue, bounty spreadsheet, linked PRs, and maintainer
direction. Re-run performance measurements when code, compiler, driver, or
environment changed.

Do not open a PR just because an old commit passes. The artifact under review
is the current diff and its current evidence.

## Prepare a reviewable PR

The PR title and body should let a reviewer reconstruct the claim without
reading the issue archaeology first. Include:

```text
Why this should merge and how it improves tinygrad
Exact issue/bounty link and current status
Root cause / first wrong or costly artifact
Smallest change and semantic preconditions
Focused regression: command and baseline/candidate result
Broader tests, backends, fuzz/spec/process replay as applicable
Performance protocol and raw-summary results, if claiming speed
Known limitations, complexity tradeoff, and rollback signal
AI/tool assistance: exactly what was used for
```

Keep generated traces, huge logs, and exploratory notebooks out of the diff
unless they are stable fixtures needed by the test. Link or summarize only the
evidence necessary for review. Read every changed line as if asked to explain
its invariant and failure mode.

The final gate is not “does this look polished?” It is:

- does the current live policy invite this class of change;
- is the issue still available and the requested outcome still current;
- does the baseline regression fail for the claimed reason;
- is the patch the smallest change at the owning layer;
- does the same regression pass, with proportional wider evidence;
- are performance and complexity claims reproducible; and
- is all AI/tool assistance disclosed accurately?

If any answer is no, keep investigating or do not submit.

## Source tour

### Pinned policy and process behavior

| Responsibility | Snapshot source |
| --- | --- |
| Contribution bar, wanted changes, tests, AI disclosure, and `[pr]` guidance | [`README.md` contribution section](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/README.md#L165) |
| Local process-replay model and commands | [`test/external/process_replay/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md#L1) |
| PR-title capture condition in CI | [`.github/workflows/test.yml`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1) |
| Replay comparison, assertion, and early-stop behavior | [`process_replay.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L1) |

### Live state—always reopen

| Decision | Current source |
| --- | --- |
| What upstream accepts and how to test | [Live contribution section](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) <!-- live-upstream --> |
| Bounty ledger linked by upstream | [Live bounty spreadsheet](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing) |
| Discover/cross-check open bounty-labeled issues | [Live bounty-label query](https://github.com/tinygrad/tinygrad/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty) |
| Ownership, current interpretation, and competing work | [Live issues](https://github.com/tinygrad/tinygrad/issues) and [pull requests](https://github.com/tinygrad/tinygrad/pulls) |
| Current replay commands and marker policy | [Live process-replay README](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) <!-- live-upstream --> |

## Lab — Triage three issues, then produce one real patch

This capstone happens on current `master`, not the guide's pinned study commit.
Keep the triage notes outside the tinygrad diff.

### Part 1: exactly three candidate records

Select exactly three currently open candidates from the live bounty sheet,
bounty-label query, or issue tracker. They need not all be bounties. Time-box
the first pass so you investigate rather than prematurely implement.

For each candidate, complete this record:

```text
URL and title:
UTC status check:
Bounty sheet entry/terms, if applicable:
Assignee, linked PR, and recent maintainer direction:
Current-master commit:
Minimal reproduce command and observed result:
Expected result / independent oracle:
First bad or costly artifact:
Owning subsystem and nearest tests:
Recent source/issue/PR history:
One bounded prerequisite gap and return exercise:
Required hardware/data/external rules:
Smallest plausible regression:
Smallest plausible patch and complexity risk:
Policy fit:
Decision: Ready / Research / Decline
Reason that would change this decision:
```

Do not choose the issue with the most exciting title. Choose a Ready candidate
whose reproduced evidence, available resources, and minimal patch give it the
best chance of becoming a clear win. If all three are Research or Decline,
that is successful triage, not failure. Close the named gaps or run another
three-candidate round; do not force one into implementation.

### Part 2: one test-backed patch

For the selected Ready candidate:

1. Recheck its live state and create a focused branch/worktree from current
   `master`.
2. Complete every applicable section of the
   [contribution brief](../reference/contribution-brief.md).
3. Add the minimal regression and run it unpatched. Save the command and failure
   showing the expected reason.
4. Implement the smallest fix at the first owning layer. Exclude unrelated
   cleanup.
5. Run the same regression, nearby edge cases, subsystem tests, and proportional
   differential/fuzz/spec/backend/hardware validation.
6. If performance is claimed, complete Chapter 17's clean baseline/candidate
   comparison and complexity assessment.
7. Run process replay when the live policy and change semantics call for it;
   inspect differences instead of recording only an exit code.
8. Update from `master`, recheck ownership and policy, and rerun the evidence.
9. Draft the review summary, including limitations, rollback signal, and exact
   AI/tool-assistance disclosure.

The capstone artifact is a real current patch, its failing-then-passing
regression, the completed brief, and reproducible validation. Do not submit it
upstream until every final gate above passes. If new upstream work supersedes
it, preserve the learning record and choose another current candidate rather
than competing with stale assumptions.

## Checkpoint

You are ready to start contributing when you can:

- distinguish snapshot contribution policy from policy and work that must be
  checked live;
- triage current bounties/issues for ownership, reproducibility, policy fit,
  prerequisites, scope, and evidence instead of choosing by title;
- route an unfamiliar branch to a bounded external prerequisite and return to
  tinygrad with a concrete result;
- produce a regression that fails on baseline for the right reason before
  changing code;
- make and defend a minimal owning-layer patch with proportional validation;
- explain when process replay and `[pr]` apply and why replay is not a numerical
  correctness oracle; and
- prepare an evidence-backed PR summary with an accurate AI disclosure.

Passing this checkpoint does not guarantee that a particular bounty is
available or a PR will merge. It means you can evaluate current work, recognize
what you still need to learn, and present a contribution that deserves review.

## Quick reference

```text
always live-check:
  README policy -> bounty sheet/query -> exact issue -> linked PRs/comments
  check at selection and again before PR

candidate:
  reproduce -> oracle -> first divergence -> source/tests/history
  -> bounded prerequisite -> Ready / Research / Decline

implementation:
  brief -> failing regression -> smallest owning change
  -> focused test -> edge/subsystem/backend/fuzz/spec/hardware evidence
  -> performance protocol if claimed -> process replay when applicable
  -> update master and rerun

[pr]: snapshot marker for refactor/speedup with no expected generated change;
      verify current policy, and never substitute replay for correctness tests

PR summary:
  why merge + root cause + change + before/after test evidence
  + broader validation + limitations/rollback + complexity + AI disclosure

newcomer snapshot warning:
  do not submit docs/whitespace or AI-looking, unverified work as an entry PR
```
