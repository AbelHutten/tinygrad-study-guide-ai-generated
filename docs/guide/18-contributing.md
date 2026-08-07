# 18. From idea or bounty to a reviewable contribution

## The promise of this chapter

Knowing how tinygrad works is necessary for contributing, but it is not enough.
An upstream contribution is a claim made to other people:

> Under these exact conditions, current tinygrad violates or could improve this
> contract; this small change at the owning layer addresses it; and this
> evidence is strong enough to review, reproduce, and safely undo.

That claim has several parts.  A patch can be technically clever and still be
the wrong contribution because the issue is stale, another pull request already
owns the work, the expected behavior was never agreed, the benchmark measures
the wrong layer, the required hardware was not tested, the diff carries
unrelated cleanup, or the author cannot establish the provenance of copied
code.  Conversely, a useful contribution can be tiny: one precise regression,
one corrected rewrite condition, or deletion of genuinely dead core code.

This chapter starts from first principles for a reader who knows Python and
machine learning but is new to compiler/GPU projects and upstream review.  It
teaches a method that applies to a bounty, an issue, or a self-chosen
improvement.  By the end, you should be able to:

- distinguish an idea, issue, bounty, spike, patch, commit, and pull request;
- read issue text as dated evidence rather than an eternal specification;
- determine whether work is Ready, needs Research, needs an upstream Question,
  or should be Declined;
- define a falsifiable contract, success criteria, and explicit non-goals;
- check live policy, ownership, comments, linked work, current source, tests,
  and history before choosing an implementation;
- reproduce and localize a problem before proposing a fix;
- choose evidence appropriate to frontend, compiler, runtime, JIT,
  performance, or physical-hardware claims;
- time-box a feasibility spike and keep its exploratory code separate from a
  merge-quality patch;
- use a branch or worktree and build conservative, coherent atomic commits;
- prove a regression red on the baseline and green after the change;
- interpret CI and process replay without treating either as a universal
  correctness oracle;
- communicate before, during, and after work without duplicating effort or
  consuming maintainer attention with unbounded speculation;
- prepare a small PR whose rationale, evidence, risks, limitations, provenance,
  and AI disclosure are reviewable;
- respond to review while controlling scope and rerunning invalidated evidence;
- define a rollback signal before merging; and
- stop or ask upstream when ownership, intended behavior, safety, licensing, or
  required evidence cannot be resolved locally.

The executable lab at `labs/phase5/contribution_walk.py` does not edit tinygrad,
open an issue, access the network, or pretend to solve a real upstream bug.  It
compares a tempting but incomplete patch idea with a complete evidence packet
for the artificial renderer fault from Chapter 15.  Both modes audit pinned
policy and process facts read-only.

All exact project and source observations target tinygrad commit
`874d33128b4e4785beea736d97df6716e0321717` from 2026-08-05.  The live
contribution and process-replay surfaces were rechecked on 2026-08-07, but
policy, bounty terms, issue ownership, code, CI, and maintainer direction can
change immediately.  Reopen the live sources at selection time and again
before submission.

## Route through the chapter

Read front to back once.  The workflow is cumulative:

1. learn what an upstream contribution is;
2. separate pinned teaching facts from live project state;
3. understand the project's stated review bar and this guide's independence;
4. find and score bounty, issue, and self-chosen candidates;
5. define contract, success, non-goals, and evidence before implementation;
6. inspect current source, tests, issue/PR overlap, and history;
7. reproduce and localize the first wrong or costly artifact;
8. decide whether to spike, patch, ask, defer, or decline;
9. isolate work in Git and design atomic commits;
10. implement regression-first with subsystem-appropriate validation;
11. understand performance, hardware, process-replay, and CI claims;
12. communicate, prepare a PR, iterate through review, and retain rollback;
13. work through realistic cases and the bounded lab; and
14. use the source stops, background ladders, exercises, and checkpoint.

This is a readiness method, not a guarantee of review, bounty payment, or
merge.  Upstream maintainers decide what belongs in their project.

## First principles: what are you actually producing?

### An idea is not yet a contribution claim

Examples of ideas:

- “fuse these two kernels”;
- “make this renderer cleaner”;
- “support a new dtype”;
- “this bounty looks interesting”;
- “CUDA is slower than expected”; or
- “remove this branch.”

An idea points toward a possible change.  It does not yet state required
behavior, current evidence, scope, ownership, risk, or value.

### An issue is a conversation record

An issue can contain:

- an initial symptom;
- assumptions that were true at the opening commit;
- later reproductions or counterexamples;
- revised maintainer expectations;
- links to competing or abandoned patches;
- a bounty label whose terms live elsewhere; and
- comments rendered obsolete by later source changes.

The title and first post are not automatically the current specification.  The
latest maintainer comment is not automatically sufficient either: it may answer
one subquestion while source or policy has moved.  Treat the whole issue as
dated evidence, then verify current behavior.

### A bounty adds terms, not certainty

A bounty is work for which the project advertises a reward under stated terms.
It does not mean:

- the title contains a complete engineering specification;
- the first proposed implementation will qualify;
- no one else is working on it;
- an open issue is unclaimed;
- the spreadsheet and issue are synchronized;
- partial progress earns payment; or
- the project owes review or acceptance for effort spent.

Resolve the current acceptance condition, ownership, and reward terms before a
large investment.  Never hard-code amounts or availability into a long-lived
guide.

### A reproducer and an oracle make the claim testable

A **reproducer** is the smallest controlled procedure that exhibits the
behavior.  An **oracle** independently defines what should happen.  For a
performance candidate, the oracle includes unchanged semantics and a timing
protocol; “looks faster in one run” is not one.

Chapter 15 teaches first-bad-artifact localization.  Chapter 16 teaches test
power and red-before-green evidence.  Chapter 17 teaches performance claims.
This chapter composes those into an upstream decision.

### A spike answers feasibility; a patch proposes a maintained change

A **spike** is disposable exploration designed to answer a bounded question:

```text
Can the target represent this operation?
Does this rewrite remove the extra realization?
Can this backend update a graph node's scalar argument?
Is the measured bottleneck large enough to justify complexity?
```

A spike may contain hard-coded inputs, diagnostic prints, and deliberately
local shortcuts.  Its result is knowledge.  A merge-quality **patch** must
express the general contract, fit architecture and style, handle error cases,
include stable tests, and carry proportional evidence.  Do not polish a failed
spike into a large diff merely to preserve sunk effort.

### A commit is a review unit; a PR is a merge proposal

A commit records one coherent change in history.  A pull request asks upstream
to merge one or more commits.  Neither is the investigation itself.  Notes,
failed approaches, giant captures, and local probes can remain outside the
diff while their conclusions appear concisely in the PR.

An **atomic commit** is not “one file” or “few lines.”  It has one purpose, no
unrelated changes, a comprehensible invariant, and a validation story that
makes sense at that commit.  A bug fix and its regression often form one atomic
commit because separating a deliberately red test into the final branch can
leave an intermediate commit that knowingly fails.  Preserve baseline-red
evidence in the investigation record instead.

## Three clocks: durable method, pinned snapshot, live state

Different facts expire at different rates.

| Clock | Examples | Correct use |
| --- | --- | --- |
| Durable method | Define a contract; reproduce; localize; test red/green; keep scope small | Carry it across revisions. |
| Pinned snapshot | What `874d331`'s README, LICENSE, CI, and process replay say | Learn concrete mechanisms and cite reproducible source. |
| Live state | Current policy, bounty terms, issue/PR ownership, `master`, CI, maintainer direction | Recheck when selecting, before substantial work, and before PR. |

Write timestamps in UTC because contributors and maintainers may be in
different time zones.  This is an illustrative record format, not a claim that
the checks have already happened:

```text
Policy checked:       <YYYY-MM-DDTHH:MM:SSZ after opening current policy>
Issue/PR overlap:     <YYYY-MM-DDTHH:MM:SSZ after searching the candidate>
Reproduced on commit: <full hash>
Rechecked before PR:  <later timestamp and hash>
```

“Latest” is not a reproducible version.  A branch name is not a commit hash.

### Live surfaces to reopen

Use the current upstream sources, not a copied list:

- [live contribution policy](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) <!-- live-upstream -->;
- [bounty spreadsheet linked by upstream](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing);
- [open bounty-label query](https://github.com/tinygrad/tinygrad/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty);
- [all current issues](https://github.com/tinygrad/tinygrad/issues);
- [all current pull requests](https://github.com/tinygrad/tinygrad/pulls); and
- [live process-replay instructions](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) <!-- live-upstream -->;
- [live test workflow](https://github.com/tinygrad/tinygrad/blob/master/.github/workflows/test.yml) <!-- live-upstream -->;
- [live replay implementation](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/process_replay.py) <!-- live-upstream -->; and
- [live replay action](https://github.com/tinygrad/tinygrad/blob/master/.github/actions/process-replay/action.yml) <!-- live-upstream -->.

A search result is a discovery surface, not proof of availability.  Open the
exact issue, every linked PR, relevant comments, and current source.

## Understand the upstream relationship before doing work

### Upstream owns its review budget and direction

Open-source availability means you may inspect, run, fork, and modify code
under its license.  It does not entitle anyone to maintainer tutoring, review,
merge, roadmap changes, or bounty payment.  A respectful contribution reduces
the cost of evaluating a useful claim.

That means:

- search before opening a duplicate issue or PR;
- do local source reading before asking broad “where should I start?” questions;
- ask a narrow question only after showing the evidence that creates ambiguity;
- do not repeatedly ping maintainers for status;
- do not claim a bounty merely by announcing interest unless the live process
  explicitly defines such a claim mechanism;
- stop when a maintainer says the direction is unwanted or already owned;
- do not turn review disagreement into public pressure; and
- accept that technically correct work can still be a poor project tradeoff.

### This guide is independent and unofficial

This repository is not tinygrad, tiny corp, or an upstream-maintained resource.
Do not:

- report an error in this guide to tinygrad's issue tracker;
- imply that completing these labs grants contributor status;
- call the guide “official tinygrad documentation”;
- use tinygrad or tiny corp branding in a way that suggests sponsorship,
  approval, partnership, or endorsement;
- claim an artificial lab fault is an upstream bug; or
- ask upstream maintainers to support this guide's setup or generated prose.

Questions and corrections about this guide belong in this guide's repository.
When discussing it publicly, use plain identification such as “an independent,
unofficial study guide about tinygrad.”

### The pinned contribution bar is unusually explicit

At `874d331`, the upstream README says, in substance:

- explain briefly why the PR should merge and improve tinygrad;
- disclose AI use, with especially strict treatment of AI-looking submissions
  from new contributors;
- reduce complexity and improve readability rather than code-golfing;
- do not submit documentation or whitespace changes as a new contributor;
- benchmark every speedup claim and weigh maintainability;
- avoid changing non-core code merely because it could be cleaned up;
- keep diffs small, splitting only independently valuable clear-win
  prerequisites;
- include regression tests for fixes and features;
- use process replay for qualifying refactors; and
- value high-quality bounty work, tests/fuzzers, and real dead-core-code removal.

Read the exact pinned
[`README.md` contribution section](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/README.md#L165-L205),
then the live section.  Do not soften or paraphrase away a rule when acting.

### AI disclosure and personal verification

At both the pinned snapshot and the live check recorded above, upstream
requires disclosure of AI use and warns strongly against AI-looking,
unverified submissions from new contributors.  If an assistant searched,
explained source, generated code, suggested a test, rewrote a benchmark, or
drafted PR text, disclose what it did with the specificity current policy
requires.

Disclosure is necessary but not sufficient.  You must be able to:

- explain every changed line and invariant;
- reproduce every before/after result yourself;
- distinguish facts read in source from assistant inference;
- remove invented APIs, false citations, and claims you did not test;
- identify code or text copied from third parties;
- answer review questions without delegating understanding; and
- abandon generated work you cannot personally validate.

“An LLM said this is correct” is not evidence.  “I ran command X at commit Y,
observed Z, and inspected boundary W” is evidence.

## Find candidates without choosing by excitement

### Candidate sources

A candidate can come from:

- a current bounty;
- a reproducible open issue;
- a bug found while using tinygrad;
- a failing or weak existing test;
- a fuzzer-generated minimized counterexample;
- an observed performance bottleneck;
- a small feature whose project value and API contract are clear;
- a clear-win core refactor; or
- proven dead code in core.

Self-chosen does not mean policy-free.  It may require more justification
because no existing issue has established demand.

### First-pass candidate card

For each candidate, fill this before implementation:

```text
Candidate URL or local discovery:
Origin: bounty / issue / bug / performance / feature / refactor / test / dead code
UTC live-state check:
Current base commit:
One-sentence observable symptom or opportunity:
Independent expected behavior or measurable objective:
Visible owner, assignee, linked PR, or competing work:
Required backend/hardware/data/toolchain:
Likely subsystem:
Smallest next experiment:
Largest unresolved risk:
Decision: Ready / Research / Question / Decline
```

Do not write a proposed code change in the symptom line.  “Graph scalar update
uses a stale value on replay three” is an observation.  “Change `GraphRunner`
to copy the scalar” is already an implementation hypothesis.

### Score constraints, not prestige

Use a qualitative matrix:

| Dimension | Good first candidate | Warning | Stop condition |
| --- | --- | --- | --- |
| Contract | Expected behavior is explicit and independently testable | Several plausible interpretations | Maintainer intent is contradictory or unknown and material |
| Reproduction | Deterministic minimal case on available route | Flaky or large but reducible | Cannot reproduce current behavior and no artifact remains |
| Ownership | No active overlapping work found | Old abandoned PR needs reading | Active maintainer/contributor work already covers it |
| Scope | One owning layer and focused test | Bounded prerequisite gap | Requires redesign across many subsystems |
| Resources | Available CPU/GPU/toolchain/data | Borrowable or emulatable evidence | Acceptance requires unavailable hardware/data/rules |
| Policy fit | Named wanted category and clear project value | New feature needs stronger value case | Explicitly discouraged class of change |
| Validation | Focused plus proportional matrix is affordable | Some slow/hardware checks need coordination | Core claim cannot be validated safely |
| Complexity | Small change or net simplification | Moderate code with strong payoff | Large diff whose value depends on later work |

A high bounty amount does not repair a missing contract or hardware route.

### Ready, Research, Question, or Decline

Use four outcomes:

| Decision | Meaning | Next action |
| --- | --- | --- |
| **Ready** | Contract, current reproduction, ownership, resources, and bounded evidence path are clear | Create a clean branch/worktree and add the regression. |
| **Research** | One or more local technical facts are missing | Name the exact question and run a bounded source/experiment loop. |
| **Question** | A material project-intent, ownership, bounty-term, or safety decision cannot be resolved locally | Ask upstream concisely with collected evidence; do not implement broadly while waiting. |
| **Decline** | Stale, duplicate, unavailable, unsafe, poor fit, or too broad | Record why; choose another candidate without framing it as project failure. |

“Research” is not “read everything about compilers.”  It is:

```text
Unknown: Does this target operation preserve NaN payloads for float16?
Resource: exact target ISA section and one current renderer test.
Return exercise: construct two payloads, render, execute on required hardware,
                 and explain the observed bits.
```

## Define the contract before designing the fix

### Contract structure

A useful contract specifies:

```text
Preconditions: inputs, dtype, shape, layout, device, state, call number
Operation:     public or internal action
Postcondition: value, structure, ordering, error, or performance objective
Tolerance:     exact or justified numeric/timing threshold
Exclusions:    unsupported or deliberately unchanged cases
```

Correctness example:

```text
Given two non-overlapping float32 views with the recorded offsets,
third-call TinyJit replay must read the new input buffers and match JIT=0
within rtol=1e-5, atol=1e-6 on CUDA.
```

Performance example:

```text
For the recorded model, shapes, dtype, RTX 4090 route, driver, and warm state,
candidate reduces median synchronized step time beyond the measured noise band,
without changing outputs, kernel count unexpectedly, compile time materially,
or complexity beyond the stated budget.
```

Refactor example:

```text
The two current lowering paths produce identical program artifacts for the
captured corpus before and after consolidation; focused unit semantics and
process replay remain unchanged; duplicated logic is removed.
```

### Success criteria are observable

Write at least two kinds:

- the primary contract passes;
- a negative or neighboring case remains unchanged;
- a baseline regression fails for the intended reason and then passes;
- broader subsystem checks pass;
- generated kernels remain unchanged when required;
- performance exceeds a predeclared threshold/protocol; and
- code complexity is bounded or reduced.

Avoid criteria such as “clean,” “fast,” “robust,” or “better” without an
observable definition.

### Non-goals prevent accidental project expansion

Examples:

- no new public API;
- no behavior change for integer dtypes;
- no claim about AMD or Metal;
- no optimizer redesign;
- no model-wide speedup claim;
- no formatting or naming cleanup;
- no direct-PCI support;
- no resolution of adjacent issue B; and
- no promise to preserve incidental generated variable names.

Non-goals are not excuses to ignore affected behavior.  If the change can
alter AMD even though you do not claim AMD support, that is a risk requiring
validation or upstream coordination.

### Define rollback before implementation

A rollback plan needs an observable trigger and a recoverable action:

```text
Signal: process replay changes kernels outside the intended matcher family.
Action: revert the matcher commit; no schema/data migration is involved.
```

“We can fix it later” is not rollback.  For a cache format, API, allocator,
runtime queue, or persistent artifact, rollback can be harder and must be
designed explicitly.

## Check current source, tests, issues, PRs, and history

### Source is the present implementation, not the issue's memory

From a clean current checkout:

```bash
git rev-parse HEAD
git status --short
rg -n 'suspect_symbol|relevant_op|error fragment' tinygrad test
```

Read the producer, transformation, consumer, and closest test.  Follow the
actual call path.  If the issue says a function owns behavior but current code
no longer calls it, the issue's implementation discussion is stale.

### Search semantic overlap, not only titles

Search current and closed issues/PRs for:

- the public operation name;
- internal symbols;
- error messages;
- backend and device name;
- dtype/shape pattern;
- prior issue numbers referenced in comments; and
- alternate descriptions of the same invariant.

A PR titled “scheduler cleanup” may already change the range behavior in a
“wrong reshape result” issue.  Read diffs and status, not titles alone.

### Read comments chronologically, then rebuild current meaning

Make a small ledger:

| Date/commit | Statement or artifact | Still true on current base? | Evidence |
| --- | --- | --- | --- |
| issue opened | fails on CUDA for shape X | yes/no | current command |
| later comment | expected dtype is Y | unresolved | API/source/policy question |
| linked PR | changes renderer path | merged/closed/active | current source and PR status |
| current base | first bad artifact is Z | yes | adjacent comparison |

Do not cherry-pick a convenient old comment while ignoring a later contract
change.

### Use history after locating current ownership

Useful read-only commands:

```bash
git log --oneline --decorate -- tinygrad/path.py test/path.py
git log -p -S 'relevant_symbol' -- tinygrad test
git blame -L START,END tinygrad/path.py
git show --stat --oneline COMMIT
git show COMMIT -- tinygrad/path.py test/path.py
```

History answers:

- why a condition exists;
- which regression accompanied it;
- whether a “cleanup” restores previously removed behavior;
- whether generated artifacts were expected to change; and
- which tradeoff maintainers accepted then.

History does not override current evidence or current maintainer direction.
`blame` identifies the last line change, not the root cause.

## Reproduce and localize before patching

### Freeze the experiment

Record:

```text
full commit hash
Python and dependency versions
OS and architecture
DEV, renderer target, GPU/driver/toolchain
input values, shape, dtype, layout, state, call number
optimizer/JIT/cache/debug flags
random seeds
exact command, stdout/stderr, exit status
frequency across fresh processes
```

Use Chapter 15's deterministic process envelope.  An issue's old screenshot is
not a current reproducer.

### Locate the first wrong or costly artifact

For correctness:

```text
oracle
  → frontend UOps
  → scheduled SINK / outer LINEAR
  → lowered SINK / program LINEAR
  → SOURCE
  → BINARY
  → runtime arguments/order/completion
  → JIT capture/graph/replay
  → result
```

For performance:

```text
workload semantics
  → schedule/kernel count
  → program operations and memory traffic
  → generated source/ISA
  → compile/launch/runtime events
  → synchronized affected-kernel distribution
  → end-to-end distribution
```

The proposed owning change should be adjacent to the first violated contract,
not merely the last place the symptom appears.

### Evidence ladder by subsystem

| Candidate area | Direct evidence | Useful controls | Evidence that is insufficient alone |
| --- | --- | --- | --- |
| Tensor API/dtype/autograd | Public oracle, shape/dtype/gradient contract, frontend UOps, focused regression | NumPy/PyTorch only when compatibility is intended; finite differences where valid | One backend agreeing with itself |
| UOp/rewrite | Input domain, before/after graphs, preserved dtype/shape/effects, counterexamples/property checks | `SPEC=2`, bounded matcher tests, VIZ | “Graph is smaller” without semantic invariant |
| Scheduling/indexing | Scheduled kernels, realization/order, symbolic bounds, memory effects | `DEBUG_RANGEIFY`, `SCACHE=0`, `CHECK_OOB=1`, schedule tests | Final result for one shape |
| Kernel optimization | Pre/post lowered program, opts, resource and traffic estimates, result oracle | `NOOPT`, `BEAM`, `TC`, emulated targets, real target benchmark | Generated code looks efficient |
| Renderer/compiler | Same lowered input, exact source, target/toolchain diagnostic, binary/disassembly when relevant | `DEBUG=4/7`, alternate renderer, compile-only tests | Compiler error automatically blamed on compiler |
| Runtime/queues | Exact program ABI, buffers, launch dimensions, order/events/completion, physical result | scoped waits, fresh process, runtime tests | Python backend passing a race-free arithmetic check |
| TinyJit/graphs | `JIT=0 → 2 → 1`, ignore/capture/replay outputs with changing buffers, captured/graphed plans | three-plus calls, graph support proof, lifetime stress | A single decorated call |
| Correctness test/fuzzer | Independent oracle/property, mutant or known counterexample, deterministic minimization | focused seed, bounded generator, shrink/replay | Test count or coverage percentage alone |
| Performance | Correctness-bracketed baseline/candidate distributions, same artifact scope, synchronized timing, attribution | Chapter 17 protocol, raw samples, profiler | best run, unsynchronized host time, cached compile comparison |
| Hardware-specific | Exact physical device/API/driver/toolchain, safe run, target artifact, hardware result | vendor tools, dedicated hardware when fault risk exists | `PYTHON::target`, `NULL`, or source inspection claimed as hardware validation |

Choose evidence that can falsify your exact claim.  Do not collect every log.

## Decide between a feasibility spike and a patch

### Spike when one bounded unknown dominates

Good spike questions:

- Does legal target code exist for this dtype and operation?
- Can a matcher express the rule without capturing an unsafe domain?
- Does removing one realization change the model bottleneck materially?
- Can graph replay update this parameter without rebuilding the graph?
- Does required hardware reproduce the issue at all?

Time-box the spike and predeclare its outputs:

```text
Question:
Maximum time/experiments:
Disposable modifications allowed:
Success observation:
Failure observation:
What evidence would promote this to a patch?
What result makes us stop?
```

Keep spike code in a separate branch/worktree or uncommitted local file.  Do
not stack production polish onto a hypothesis that has not survived.

### Promote only the knowledge, not accidental spike structure

After a successful spike:

1. return to a clean branch from current base;
2. write the stable regression from the contract;
3. reimplement the smallest general change in project style;
4. remove diagnostic prints, fixed paths, magic values, and broad exceptions;
5. run focused and proportional wider evidence; and
6. compare the new clean patch with what the spike actually established.

### Stop a spike when the premise fails

Stop when:

- the problem no longer reproduces;
- the expected behavior is not desired upstream;
- active work supersedes it;
- the effect is below noise or irrelevant end-to-end;
- required hardware/data is unavailable;
- the safe implementation needs a broad redesign not justified by value;
- the code came from uncertain/incompatible provenance; or
- the remaining decision belongs to maintainers.

Recording a negative result is progress.  It prevents an unjustified PR.

## Isolate work with Git

### Keep the study checkout, baseline, spike, and patch distinct

One conservative layout is:

```text
tinygrad-study/          pinned guide snapshot; never your upstream patch base
tinygrad-live/           clean current upstream/fork clone
tinygrad-spike-topic/    disposable worktree for feasibility
tinygrad-fix-topic/      reviewable patch worktree
```

Do not implement a live contribution in the pinned study checkout and then
assume it applies cleanly to current `master`.

### Verify remotes and base without changing them

```bash
git remote -v
git branch --show-current
git rev-parse HEAD
git status --short
```

Common convention is `upstream` for `tinygrad/tinygrad` and `origin` for your
fork, but names are local; inspect rather than assume.

Fetching contacts the configured remote and updates local objects and remote-
tracking refs; it does not edit tracked worktree files:

```bash
git fetch upstream
```

Run it only when the remote is verified.  Never paste credentials into a
command or commit.

### Branch or worktree

From a clean updated base, a branch in the same checkout:

```bash
git switch -c fix/descriptive-contract upstream/master
```

Or a separate worktree:

```bash
git worktree add ../tinygrad-fix-contract -b fix/descriptive-contract upstream/master
```

Before either, inspect `git status --short`.  A worktree is valuable when you
need baseline and candidate simultaneously, or process replay expects branch
switches.  It is not a backup for uncommitted work.

### Inspect every edit before staging

```bash
git status --short
git diff -- path/to/source.py path/to/test.py
git diff --check
```

Stage only the coherent paths or hunks you understand:

```bash
git add path/to/source.py path/to/test.py
git diff --cached
```

Interactive staging can separate accidentally interleaved concerns, but do not
use it to fabricate commits whose code does not work independently.

### Conservative atomic commit rules

Each commit should answer:

```text
What one contract or clear-win structure changes?
Why is this exact code needed?
Which test demonstrates it?
Does this commit pass its intended checks?
Can it be reverted without entangling unrelated work?
```

Prefer:

- one bug fix plus its regression;
- one independently valuable prerequisite refactor plus unchanged-behavior
  evidence, followed by a separate feature commit only when both are clear;
- one test/fuzzer improvement with a stable oracle; or
- one dead-code removal with proof of non-use and tests.

Avoid:

- “cleanup” mixed with a semantic change;
- formatting an entire file;
- splitting source and its required test merely by file type;
- dozens of checkpoint commits with broken intermediate states;
- one giant commit containing spike, refactor, feature, benchmark, and docs;
- generated files not required by the project; and
- rebasing or force-pushing shared work without coordination.

Commit messages should state the behavior, not the editing motion:

```text
renderer: preserve float16 cast for gated loads
jit: update scalar args during graph replay
schedule: avoid realizing no-op contiguous view
```

“changes,” “fix stuff,” and “cleanup” make archaeology harder.

### Safe rollback during local work

Before discarding anything, inspect exact paths and preserve valuable work as a
commit or patch.  Prefer recoverable operations:

- stop using a disposable spike branch;
- revert one committed experiment with `git revert <commit>`;
- restore only an explicitly inspected file you own; or
- close the worktree after confirming `git status --short` is clean.

Do not use `git reset --hard` or broad recursive deletion as routine cleanup.

## Regression-first implementation

### 1. Add the focused contract before the fix

Find the nearest test:

```bash
rg -n 'related_operation|related_symbol' test tinygrad
```

The regression should preserve the actual trigger: dtype, view, symbolic
bound, backend, call number, optimization, or queue order.  Assert public
semantics or a stable internal contract, not temporary names or full log text.

Run it on unmodified baseline and record:

```text
command
commit
expected failure
actual failure
why the failure proves the intended gap rather than setup breakage
```

If the new test passes on baseline, it is not a regression for the claimed
bug.  Strengthen the case or revise the claim.

### 2. Prove the test has power

Possible controls:

- exact known counterexample;
- independent reference backend or loop oracle;
- a deliberately wrong local mutant;
- negative case outside the rule domain;
- property/metamorphic relationship; or
- prior known-good revision when semantics are stable.

A test that merely executes code can stay green while checking nothing useful.

### 3. Change the owning layer

Use the first-bad-artifact result:

```text
good scheduled SINK → bad lowered program       target lowering owns transition
good program LINEAR → bad SOURCE                renderer owns transition
good SOURCE/BINARY → bad runtime arguments      runtime/ABI owns transition
ordinary and JIT=2 good → graphed JIT=1 bad     graph path owns transition
```

Do not add a public Tensor special case to hide a renderer error unless the
public contract genuinely owns it.

### 4. Re-run the identical focused test

The candidate run must use the same test, inputs, environment, and oracle.
Changing the test while implementing can move the goalposts.  If the contract
really changes, document why and re-establish baseline evidence.

### 5. Expand validation outward by risk

Order:

1. new focused regression;
2. neighboring positive and negative cases;
3. dtype, shape, symbolic, layout, and state boundaries;
4. nearest subsystem test file;
5. differential/property/fuzz/SPEC checks;
6. affected backends and physical hardware;
7. process replay when semantically applicable;
8. project static checks/pre-commit hooks; and
9. broader CI-equivalent commands appropriate to the change.

Do not claim tests you did not run.  Say “not run: requires AMD hardware” and
explain how that affects readiness.

### 6. Rebase evidence onto current upstream state

Before PR:

- fetch and integrate current upstream base using the project's preferred
  history style;
- resolve conflicts by re-reading invariants, not selecting “ours” or “theirs”
  mechanically;
- rerun focused and invalidated broader tests;
- repeat performance measurements if code/environment changed;
- reopen issue, bounty, linked PRs, and live policy; and
- inspect the final diff from the reviewer's perspective.

Evidence from an old base is not automatically evidence for the rebased diff.

## Correctness, performance, and hardware claims need different proof

### Correctness claim

Minimum useful packet:

```text
contract and non-goals
independent oracle
baseline-red focused regression
first bad artifact and owner
smallest fix
same test green
neighboring domain and broader subsystem checks
unrun affected routes and risk
```

Full-suite green without baseline red can miss the claimed behavior entirely.

### Performance claim

Every speedup claim needs Chapter 17's full protocol:

- same commit ancestry and isolated worktrees;
- same workload, inputs, dtype, model state, backend, target, device, driver,
  power/thermal conditions, environment, and cache policy;
- correctness before and after measurement;
- explicit warm-up;
- synchronization or device-event timing appropriate to the layer;
- multiple raw samples and distribution;
- noise floor/repeatability;
- compile, affected-kernel, and end-to-end results kept separate;
- artifact explanation for why performance changed; and
- complexity/maintainability tradeoff.

Use accurate claim language:

```text
Observed: median synchronized affected kernel fell from A to B under setup S.
Not established: model throughput, other shapes, other GPUs, or compile time.
```

Do not promote a microbenchmark result into “tinygrad is X% faster.”

### Hardware claim

State evidence class:

| Route | Establishes | Does not establish |
| --- | --- | --- |
| `DEV=PYTHON` | Semantics on Python interpreter route | Generated native code, driver, queue, GPU timing |
| `DEV=PYTHON::sm_89` | Ada-target lowering/interpreter behavior | RTX 4090 execution or concurrency |
| `DEV=NULL` | Planning/compilation structure where supported | Numeric device result |
| `DEV=CPU:CLANG` | CPU renderer/compiler/runtime behavior | CUDA/NV behavior |
| `DEV=CUDA` on named 4090 | That CUDA route on recorded driver/device/run | NV direct route, other GPUs, race freedom generally |
| `DEV=NVK+NV` on named hardware | That explicit lower-level route under recorded setup | CUDA route or unsafe PCI path |

After a device fault, use Chapter 15's safe discipline.  Do not repeatedly run
a known-faulting kernel on a display GPU merely to strengthen a PR narrative.

### Missing hardware can change the decision

Options:

- narrow the claim to evidence you do have, if the code truly cannot affect
  the missing route;
- use emulation for structural evidence while explicitly leaving hardware
  unproved;
- ask a collaborator/maintainer for a specific hardware check after providing
  a minimal reproducer;
- keep the candidate in Research; or
- decline it.

Never write “should work on AMD” as validation.

## Process replay and CI: exact pinned meaning

### What process replay compares

At the pinned snapshot, process replay:

1. captures inputs to selected kernel-generation work on the contribution
   branch into `CACHEDB` when `CAPTURE_PROCESS_REPLAY=1`;
2. checks out or runs against `master`;
3. regenerates source from captured inputs; and
4. diffs generated kernels.

The pinned
[`process replay README`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md#L1-L19)
says diffs do not assert by default and that comparison stops early when more
than 20% of kernels change.  Treat that as the documented intent, not as a
precise description of the pinned counter.

The implementation sets `MAX_DIFF_PCT=20`, but each worker invocation starts
`changed = 0` for one page and compares that raw exception count directly with
20; it never divides by the page size or total captured rows.  With
`ASSERT_DIFF=0`, an ordinary generated-source difference emits a warning and
does **not** increment `changed`; exceptions raised while unpickling or
replaying do.  With `ASSERT_DIFF=1`, the source-difference warning becomes an
exception, reaches the handler, increments `changed`, and the handler's own
warning is also promoted, so that task will normally abort instead of
accumulating toward the threshold.  `changed` starts over for every page.  The
pinned code therefore does not compute a corpus-wide changed-kernel
percentage.  Verify the controls in
[`process_replay.py` setup](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L20-L36),
the
[`per-page diff loop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L63-L97),
and the
[`mapping/top-level handler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L99-L128).

It is evidence about generated artifacts for captured inputs.  It is not:

- a public numerical oracle;
- physical device execution;
- a proof for uncaptured shapes/dtypes/contexts;
- a benchmark;
- proof that a changed kernel is wrong; or
- proof that an unchanged kernel preserves runtime behavior.

### The pinned `[pr]` convention

At `874d331`, refactor and speedup PRs with no expected behavior change are
instructed to include `[pr]` in the PR title.  CI's
[`CAPTURE_PROCESS_REPLAY` condition](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1-L9)
looks for that marker.  The replay script also derives assertion behavior from
the title/commit message.

There is a spelling/evaluator mismatch in this exact snapshot.  The README and
workflow expression contain lowercase `[pr]`, while
[`process_replay.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L7-L8)
tests for uppercase `[PR]` with Python's case-sensitive substring operator
when deriving `ASSERT_DIFF`.  By contrast, GitHub Actions documents
[`contains()` as case-insensitive](https://docs.github.com/en/actions/reference/workflows-and-actions/expressions#contains),
so either `[pr]` or `[PR]` in a PR title satisfies the workflow condition.  The
[`process-replay` action](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/actions/process-replay/action.yml#L6-L16)
exports the actual PR title and commit message.  Consequently:

- lowercase `[pr]` in the title enables workflow capture and the action, but
  does not by itself enable the Python assertion;
- uppercase `[PR]` in the title enables both; and
- uppercase `[PR]` in the exported commit message can enable the Python
  assertion after either case in the title enabled the action.

A direct local invocation with neither title nor commit-message environment
variable receives the script's uppercase default and starts with assertion
mode enabled unless `ASSERT_PROCESS_REPLAY=0` disables it.  These are pinned
implementation facts, not advice to add both markers.  The live README,
workflow, script, and action linked above still showed the same literal
spellings when rechecked on 2026-08-07.  Recheck all four before a real
submission; if the mismatch still affects the work, ask upstream which
convention is intended.

`[pr]` does not mean “this PR has tests” or “please run CI.”  Recheck live
instructions before using it.

### Safe local replay workflow

The pinned README's conceptual sequence is:

```text
contribution branch:
  CACHELEVEL=1 CACHEDB=<absolute isolated db> CAPTURE_PROCESS_REPLAY=1 run representative tests
  captured inputs enter isolated CACHEDB

clean master worktree:
  CACHELEVEL=1 CACHEDB=<the same absolute db> run process_replay.py
  inspect every relevant diff and assertion mode
```

`diskcache_put` is inert below `CACHELEVEL=1`, so capture requires a cache level
of at least one as well as a stable, shared `CACHEDB` path.  Isolate that path;
know which branch each process uses; preserve uncommitted work; and read
`local.sh` before executing because it switches branches.  A separate clean
worktree is safer than switching a dirty patch worktree.

### Read CI as a matrix of claims

Do not say “CI tests everything.”  At the pinned snapshot, jobs select specific:

- Python versions;
- dependency extras;
- backend/device environments;
- test files and exclusions;
- `SPEC`, `CHECK_OOB`, dtype, JIT, and optimization configurations;
- hosted or project hardware; and
- timeouts/skips.

Read the current workflow around the job relevant to your change.  Reproduce
the smallest failing command locally first.  A CI failure on a backend you do
not have may require a focused artifact comparison or maintainer help; blindly
pushing guesses consumes shared resources.

At the pinned snapshot, the README asks agents to run tests with `-n12` for
speed.  This is snapshot policy, not a reason to oversubscribe a reader's
machine or hide an order-dependent test.  Recheck current wording and rerun a
suspected concurrency-sensitive failure serially for diagnosis.

## Communication is part of the engineering

### Before work: when silence is fine

For a tiny, unambiguous, unowned, reproducible bug with a clear test and small
fix, local investigation may be more useful than a speculative “I am working
on this” comment.  Recheck ownership immediately before opening the PR.

### Before work: when to ask

Ask before substantial implementation when:

- issue comments conflict about expected behavior;
- bounty acceptance terms are ambiguous or inconsistent;
- an active PR overlaps materially;
- a public API or architecture direction must be chosen;
- required validation uses scarce/unsafe hardware;
- a prerequisite refactor is large;
- compatibility behavior is unclear;
- security-sensitive disclosure may need a private route; or
- maintainers must provide data, hardware, credentials, or policy interpretation.

### A good question contains local work

Template:

```text
I reproduced <exact symptom> on <commit> with <minimal command>.
The independent contract is <expected behavior>.
Artifacts remain correct through <last good>; <first bad> differs at <boundary>.
Current source/tests suggest interpretations A and B; issue comment <link/date>
appears to favor A, while <current source fact> suggests B.
Before implementing, should the contract be A or B?
```

This is answerable.  “Can someone explain the compiler and tell me what to
change?” is not bounded.

### During work: communicate changed premises

Update the issue/PR when:

- the original reproduction no longer holds;
- localization changes the owning subsystem;
- the patch needs materially larger scope;
- a performance claim disappears or reverses;
- required hardware validation is unavailable;
- you find competing work;
- provenance becomes uncertain; or
- a maintainer-requested direction creates a new tradeoff.

Do not post every exploratory thought.  Compress evidence into decisions.

### Avoid duplicate and adversarial claims

Use neutral language:

- “I could not reproduce on commit X under setup Y,” not “the issue is fake”;
- “This overlaps files and contract Z in PR N,” not “they stole the work”;
- “The measured change is within noise,” not “the optimization is useless”;
- “This artificial guide case is not upstream,” not “tinygrad has this bug”; and
- “I do not have evidence for AMD,” not “AMD probably works.”

Critique artifacts and contracts.  Do not speculate about contributor motives.

### When to stop after asking

If the answer is “not wanted,” “already handled,” or defines a contract your
approach cannot satisfy, stop or redesign.  If no answer arrives, that does not
automatically authorize a broad change.  Continue only with reversible local
research that does not presume the missing decision.

## Scope control while implementing

### Keep a scope ledger

```text
Required for contract:
  - focused regression
  - one matcher predicate change

Tempting but excluded:
  - rename nearby variables
  - reorganize matcher table
  - add unrelated dtype feature
  - reformat test file

Newly discovered risk:
  - symbolic upper bound shares rule; add one negative test
```

Review this whenever the diff grows.

### Complexity is a cost even when tests pass

Ask:

- Can an existing abstraction express this?
- Is the new branch part of the general rule or a symptom-specific exception?
- Does the change increase flags, mutable state, cache keys, or backend
  divergence?
- Can code be deleted rather than added?
- Does the prerequisite refactor improve the project without the feature?
- Can a reviewer verify it in one sitting?

The pinned policy values small, readable, lower-complexity changes.  Small line
count achieved through dense code golf is not simplicity.

### Split only independently valuable work

A prerequisite refactor deserves a separate PR/commit only if it has its own:

- clear value;
- unchanged-behavior contract;
- tests/process replay as applicable;
- comprehensible scope; and
- usefulness even if the follow-up never merges.

Otherwise the split asks reviewers to accept speculative churn.

### Watch the final diff, not just your intention

```bash
git diff --stat upstream/master...HEAD
git diff upstream/master...HEAD -- tinygrad test
git diff --check upstream/master...HEAD
```

Look for accidental generated files, debug output, cache databases, model
weights, private paths, secrets, formatting, and unrelated hunks.

## Licensing and provenance

### Read the current license; do not infer from project popularity

The pinned tinygrad checkout declares MIT in
[`pyproject.toml`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L1-L12)
and contains the standard permission and warranty text in
[`LICENSE`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/LICENSE#L1-L7).
Read the live files before contributing.  This chapter is engineering guidance,
not legal advice.

### Ideas, specifications, and copied expression are different

You may learn an algorithm from a paper or specification.  Copying an
implementation, comments, test vectors, generated tables, headers, or vendor
code can create different license/notice obligations.  Before importing
anything:

```text
Source URL and exact revision:
What was copied or adapted:
Source license and notice requirements:
Compatibility with tinygrad's current license/project policy:
Required attribution or generated-file process:
Permission/employer constraints:
```

If uncertain, do not paste it.  Reimplement from an authoritative algorithmic
description only when you can establish that doing so is appropriate, or ask
the relevant rights holder/project maintainer.  “It was on GitHub” is not a
license.

### Tests and constants also have provenance

Large expected-output arrays, ISA encodings, model fixtures, and copied tests
can be copyrighted or governed by separate terms.  Prefer minimal values you
derive and can explain.  Preserve required notices for material reused under a
compatible license.

### AI-generated output still needs provenance review

An AI system may reproduce recognizable code or invent a source.  Disclosure
does not resolve license compatibility.  Search suspicious distinctive
fragments, verify every cited source, and rewrite or remove anything whose
origin you cannot establish.  Follow the live upstream AI policy exactly.

### Employer and contributor authority

If employment, contract work, research agreements, or university policy may
claim code produced in that context, resolve authority before contributing.
Do not include confidential code, logs, datasets, credentials, customer
shapes, or internal benchmark results.

### Do not add affiliation claims or casual license headers

Your fork or learning project should not imply endorsement.  An upstream patch
should follow the project's existing header/notice conventions; do not add a
new personal license header to each file without direction.  In a PR, say what
you authored or adapted and from where.

## Prepare a reviewable pull request

### Title

Use the subsystem and behavior:

```text
jit: update scalar arguments on graph replay
renderer: preserve cast semantics for gated load
schedule: avoid redundant realization for no-op view
```

Use snapshot markers such as `[pr]` only for their exact live-defined purpose.

### Body

A compact structure:

```text
Why merge
  What current contract is broken or what measured value improves?
  Why does this belong in tinygrad now?

Reproduction / baseline
  Commit, environment, minimal command, expected, actual.

Root cause
  Last good artifact, first bad/costly artifact, owning transition.

Change
  Smallest semantic edit and important non-goals.

Validation
  Focused baseline-red/candidate-green result.
  Neighboring and broader commands.
  Process replay, performance, backends, and hardware where applicable.
  Explicitly unrun checks and why.

Risk / rollback
  Affected representations and observable rollback signal.

Provenance / tools
  Copied/adapted sources and license treatment.
  Exact AI/tool assistance disclosure under current policy.
```

Reviewers should not need a huge log to find the claim.  Link or summarize the
small relevant artifact fragment.  Retain raw benchmark samples where the
project can inspect them without putting unstable clutter in core.

### PR claims must match evidence strength

Use:

- “passes on `DEV=PYTHON` and `CPU:CLANG` under these commands”;
- “RTX 4090 CUDA result under driver X”;
- “no generated kernel diffs in captured corpus Y”; or
- “median affected-kernel time changed from A to B across N samples.”

Avoid:

- “works everywhere”;
- “fully tested”;
- “GPU verified” from emulation;
- “zero risk”;
- “fixes issue” when contract remains disputed; or
- “no behavior change” without defining observed behavior scope.

### Never hide limitations to improve acceptance odds

A precise limitation makes review cheaper.  Discovery during review is more
costly:

```text
Not run on AMD hardware; shared renderer rule is exercised under
PYTHON::gfx1201, but queue/runtime behavior is outside this patch's claim.
```

If the missing route is essential, the honest consequence may be “not ready.”

## Review iteration

### Treat each review change as a new evidence state

For every substantive requested edit:

1. restate the invariant affected;
2. make the smallest change;
3. rerun the focused test;
4. rerun broader tests invalidated by the edit;
5. rerun benchmarks if generated artifacts or timing path changed;
6. inspect process replay again if applicable; and
7. update PR claims/limitations.

Do not reply “done” if the evidence no longer corresponds to the diff.

### Answer with artifact and reasoning

Useful response:

```text
Changed the predicate to exclude vectorized BITCAST inputs.  The original rule
assumed scalar dtype; the new negative test is red on the prior revision and
green now.  Focused command X and subsystem command Y pass.  Generated source
for the original positive case is unchanged.
```

Less useful:

```text
Fixed, should be good now.
```

### Keep review scope bounded

When feedback reveals a distinct improvement:

- if required for the contract, include it and update risk/evidence;
- if independently useful but not required, propose a separate follow-up;
- if speculative, record it outside the PR; and
- if it invalidates the design, stop and return to Research rather than piling
  exceptions onto the patch.

### Handle disagreement and rejection professionally

Ask what artifact or tradeoff differs.  A maintainer may value simplicity,
project direction, or review cost differently from you.  If the PR is closed,
do not reopen duplicates or present the project as obligated by your effort.
Keep the learning, discard the merge assumption, and choose another candidate.

### Updating commits during review

Follow current maintainer preference.  Small follow-up commits can make review
changes visible; later cleanup/squash may be requested.  Rewriting history on
your own branch can be safe, but coordinate if others depend on it and use
force-with-lease rather than blind force.  Never rewrite upstream history.

## Rollback and stopping conditions

### Before PR

You may:

- abandon a spike branch;
- preserve a useful reproducer without submitting a patch;
- reset the decision to Research;
- choose a smaller candidate; or
- publish local notes without implying upstream endorsement.

Sunk time is not evidence for merge.

### During PR

Stop or close when:

- the bug is fixed independently;
- active work supersedes it;
- intended behavior differs from your contract;
- evidence no longer reproduces;
- complexity outweighs measured value;
- required validation cannot be obtained;
- provenance cannot be established; or
- maintainers reject the direction.

Explain the state concisely so future readers do not repeat archaeology.

### After merge

If regression signals appear:

- reproduce on the merged commit;
- identify whether reverting is safe and sufficient;
- communicate impact and affected versions/routes;
- prefer a reviewable revert PR or maintainer-directed action;
- preserve the failing case; and
- do not conceal the problem to defend the original contribution.

A rollback plan is a safety feature, not an admission of weak work.

Every worked case A–F below is fictional or composite.  None asserts that a
current tinygrad bug, bounty, issue, or pull request exists.  The cases teach
method; live candidate facts must come from a new, timestamped search.

## Worked case A: current correctness issue with a JIT-only symptom

This is a composite teaching case, not a claim about a current tinygrad issue.

### Initial report

```text
On CUDA, a TinyJit function returns the second call's scalar on the third call.
```

### Bad first reaction

Change a graph update function immediately because “third call means graph
bug.”  This assumes graphing occurred and ignores capture, parameterization,
memory planning, scalar binding, and ordinary semantics.

### Triage

```text
Origin: open issue
Live check: issue open, no assignee, but one linked draft PR exists
Current base: exact hash recorded
Contract: each replay uses the new scalar argument
Non-goals: no buffer-argument redesign; no non-JIT API change
Resources: CUDA 4090 available; no AMD claim
Decision: Research until linked draft scope and JIT=2 result are known
```

An empty assignee does not erase the linked draft.  Read it and ask only if
overlap remains unclear.

### Reproduction and localization

Run fresh changing inputs/scalars for at least three calls:

```text
JIT=0: pass
JIT=2: fail on first replay
JIT=1: fail on first replay
```

Because `JIT=2` skips graph splitting, graph-runner code is not yet the owner.
Compare captured parameter mapping and prepared calls.  Suppose the captured
program stores a scalar as a literal rather than a replay variable: the first
bad artifact is parameterization before graphing.

### Patch packet

```text
Baseline red: focused three-call test with scalars 1, 2, 7 returns 2 on call 3
Owner: captured scalar parameterization
Change: represent eligible scalar as replay-updated value
Negative: static compile-time constant remains constant
Validation: JIT=0/2/1, changing buffers and scalars, nearby JIT tests
Hardware: CUDA 4090 exact route; PYTHON structural control
Risk: cache key or graph update semantics for symbolic values
Rollback: revert one parameterization commit if replay mismatch appears
```

The PR should not claim “fix graph scalar updates” if the defect occurs before
graphing.

## Worked case B: performance bounty whose microbenchmark wins

This is also composite and not a statement that a particular bounty is
available.

### Candidate

A bounty asks for faster scan-like model execution.  A spike replaces one
kernel form and reports a 20% isolated-kernel improvement.

### Contract and non-goals

```text
Success: correct model outputs and lower synchronized model step distribution
         on the bounty's required shapes/hardware under current terms
Non-goal: no claim for arbitrary scans, other GPUs, compile latency, or memory
```

### Evidence reveals a different outcome

```text
affected kernel: 20% faster
kernel share:     2% of model time
new compile time: +0.8 s
model median:     unchanged within noise
code:             +140 lines and a new backend special case
```

The spike is technically successful but the contribution is not yet a clear
win.  Report the layer-specific result.  Do not call it a model speedup or
bounty solution.  Look for a simpler general transformation, a required shape
where attribution differs, or decline.

### What would change the decision?

- bounty terms explicitly target isolated kernel latency;
- affected kernels dominate required workloads;
- complexity can be reduced substantially; or
- broader model measurements show repeatable benefit above noise.

Until then: Research, not Ready.

## Worked case C: self-chosen renderer feature

### Idea

Add syntax for a target operation because the target language supports it.

### Contract questions before code

- Which public/internal operation requires it?
- What dtype and corner-case semantics apply?
- Does target hardware/compiler actually support every advertised architecture?
- Is decomposition currently correct but slower, or is functionality missing?
- What does NumPy/PyTorch compatibility require, if applicable?
- Is there current demand or an issue?
- Does the feature add a branch for one rare case or simplify general lowering?

If no current tinygrad computation reaches the new renderer operation and no
project need is established, “target supports it” is not sufficient value.

### Feasible but not submission-ready

A compile-only spike emits valid source.  That proves syntax feasibility, not:

- frontend semantics;
- lowering legality;
- physical runtime result;
- architecture coverage;
- performance value;
- maintainability; or
- policy fit.

Keep it as a spike until a real contract and evidence path exist.

## Worked case D: real bug found while using tinygrad

### Observation

A float16 nonzero-offset view returns wrong values only for a tail shape on
`CPU:CLANG`.

### Strong path

1. Derive a small independent loop oracle.
2. Freeze values, dtype, shape, offset, and optimizer settings.
3. Confirm `PYTHON` passes and `CPU:CLANG` fails.
4. Inspect frontend and scheduled graph: both correct.
5. Inspect lowered index/gate: correct.
6. Inspect source: pointer offset is scaled by the wrong item size.
7. Add the smallest semantic regression retaining float16, offset, and tail.
8. Prove red on baseline for the exact value, not a compiler/setup error.
9. Change the renderer's owning address calculation.
10. Test zero/nonzero offsets, float16/float32, tail/no-tail, Python/CPU,
    applicable SPEC/OOB checks, and nearest renderer tests.
11. Search issues/PRs/history and rebase before submitting.

### Scope trap

While reading, you dislike several pointer variable names.  Renaming them adds
review noise and violates the non-goal.  Leave them alone.

## Worked case E: unavailable low-level hardware bounty

### Candidate

A bounty requires proving queue behavior on a backend/device configuration you
do not own.  You can inspect packets and emulate some lowering on an RTX 4090,
but acceptance requires a different physical device and stress workload.

### Correct decision

Mark Research or Decline.  You may contribute a minimal safe reproducer or ask
whether a collaborator can run one exact check, but do not claim hardware
success from source reading or emulation.  Do not use unsafe direct interfaces
on a display GPU to compensate for missing equipment.

Resource mismatch is not a personal failure.  Choose a candidate you can prove.

## Worked case F: duplicate “cleanup” and affiliation risk

### Idea

Publish a large upstream documentation rewrite derived from this guide and
describe it as “the new tinygrad contributor docs.”

### Why to stop

- the pinned policy explicitly discourages newcomer docs PRs;
- this guide is AI-generated and independently licensed/published;
- upstream did not ask for or approve it;
- the title implies affiliation;
- another docs PR may already exist; and
- a giant rewrite is difficult to verify and review.

Keep the guide independent, label it unofficial, route its issues here, and
contribute to upstream only with a separate current-policy-fitting claim.

## Bundled lab: contribution-readiness gates

### Purpose

The lab does not automate judgment.  It makes the structure of judgment
visible.  A machine can detect blank fields and exact pinned facts; it cannot
prove that a contract is true, an oracle is independent, a source search is
complete, or maintainers want the change.

The script defines seventeen gates across four stages:

| Stage | Gates |
| --- | --- |
| Triage | candidate/origin, live state, contract, success/non-goals |
| Evidence | reproduction, oracle, baseline red, localization, current context |
| Patch | nearest test, smallest change, validation, claim limits, risk/rollback |
| Review | atomic history, communication, license/provenance/AI disclosure |

### Read-only checkout audit

Before either case, the script:

- rejects optimized Python because assertions are part of the lab;
- removes inherited Git repository, worktree, index, object-store, namespace,
  discovery, and configuration redirection from every subprocess environment;
- sets `GIT_NO_REPLACE_OBJECTS=1`, `GIT_OPTIONAL_LOCKS=0`, `LANG=C`,
  `LC_ALL=C`, and `NO_COLOR=1` for its read-only Git queries;
- requires `git rev-parse --show-toplevel` to resolve to the supplied directory
  and `git rev-parse HEAD` to equal the full pinned hash;
- runs `git status --porcelain=v1 --untracked-files=no` before and after, and
  rejects any tracked staged or unstaged difference across the checkout;
- requires every index path, mode, stage, and object ID to equal `HEAD` and
  rejects nonstandard `git ls-files -v` flags, including `assume-unchanged` and
  `skip-worktree`;
- enumerates every recursive `HEAD` blob with `git ls-tree`, independently
  computes the Git object ID of each worktree file, and compares its regular-
  file type and executable bit, so a hidden index flag cannot conceal a change;
- uses `git show HEAD:<path>` once per required path to compare the worktree
  bytes with the pinned blob;
- reads the pinned README, LICENSE, process-replay README/implementation, CI
  workflow, and composite replay action;
- asserts exact policy/process facts; and
- requires the selected-file hashes to be equal at two observations.

Equal observations do not prove that an unrelated process never made and
reverted a transient edit between them.  They do establish an exact tracked
tree—including content and executable bits—at both observations, exact
required blobs at the read point, and no observed selected-file change.
Untracked files are deliberately ignored.
There is no tinygrad import, network client, cache write, checkout edit, issue,
PR, or commit.

### Run the incomplete case

From the guide repository:

```bash
.venv/bin/python labs/phase5/contribution_walk.py \
  --tinygrad /absolute/path/to/tinygrad-study \
  --case incomplete
```

Expected essential output:

```text
case: incomplete-patch-idea
decision: RESEARCH
failed gates: ['T2-live-state', 'T3-contract', 'T4-boundaries', 'E1-reproduction', 'E2-oracle', 'E3-baseline-red', 'E4-localization', 'E5-current-context', 'P1-test', 'P3-validation', 'P5-recovery', 'R2-communication']
proposed edit present: True
lesson: a plausible edit is not a contribution-ready claim
limit: passing fields still require human verification; this checker cannot prove their truth
status: expected-incompleteness-detected
```

The mode exits zero only when that exact incompleteness is detected.  The
packet already contains a proposed renderer edit, commit idea, performance and
hardware non-claims, provenance sentence, and AI disclosure.  Those do not
compensate for missing contract, oracle, reproduction, localization, test,
current-state check, validation, recovery, and communication.

### Run the complete case

```bash
.venv/bin/python labs/phase5/contribution_walk.py \
  --tinygrad /absolute/path/to/tinygrad-study \
  --case complete
```

Expected essential output:

```text
case: complete-artificial-evidence-packet
decision: READY for the bounded teaching claim
passed gates: 17 of 17
  triage: 4 passed
  evidence: 5 passed
  patch: 5 passed
  review: 3 passed
upstream action: none; artificial case, not an upstream bug
status: complete-evidence-packet-passed
```

The complete packet uses Chapter 15's process-local renderer fault.  It states:

- hand arithmetic oracle;
- exact wrong and correct results;
- `SOURCE` as first bad artifact after shared `LINEAR`;
- local renderer mapping as owner;
- focused/broader validation;
- performance and GPU non-claims;
- risk and rollback;
- one atomic educational change;
- no upstream communication about the artificial case;
- MIT/provenance treatment; and
- AI disclosure.

It is complete for its bounded teaching claim, not proof that upstream has a
bug and not permission to submit a PR.

### Safe extensions

1. Add a gate requiring a concrete prerequisite question and return exercise.
2. Add separate `spike_ready` and `submission_ready` decisions.
3. Create a local packet for one real candidate, but keep URLs/notes outside
   committed guide output and do not make network calls from the lab.
4. Add a mutually exclusive performance-claim gate: either a full protocol or
   an explicit non-claim must be present.
5. Add a machine-readable packet export while keeping URLs and human judgments
   outside committed teaching output.

Do not change the lab to open issues, post comments, push branches, or submit
PRs.  External communication requires current human judgment and authority.

## A practical contribution brief

Use the fuller
[contribution brief](../reference/contribution-brief.md), or this compact form:

```text
LIVE STATE
Issue/bounty/self-chosen source:
UTC policy/ownership/PR check:
Current base commit:
Maintainer direction and conflicts:

CONTRACT
Preconditions:
Required behavior:
Success criteria:
Non-goals:

EVIDENCE
Minimal command/environment:
Oracle:
Expected/actual:
Baseline-red result:
Last good / first bad artifact:
Owning source symbol:
Current source/tests/history checked:

PATCH
Nearest regression location:
Smallest owning-layer change:
Alternatives rejected:
Atomic commit plan:

VALIDATION
Focused and negative cases:
Subsystem/differential/fuzz/SPEC:
Backends/hardware:
Process replay:
Performance protocol:
Static/CI checks:
Explicitly unrun:

RISK
Affected domains:
Complexity:
Observable regression signal:
Rollback:

COMMUNICATION / PROVENANCE
Question or status update needed:
Third-party code/data/license:
Employer/confidentiality check:
AI/tool disclosure:
```

Mark a row non-applicable only with a reason.  “N/A” by itself can hide an
unexamined requirement.

## Question-led pinned project stops

These are not isolated declarations to stare at.  Read the prediction and
question first, then the bounded lines, then answer in your own words.

### Stop 1: What does upstream ask a contributor to optimize for?

Prediction: the project values readability and lower deep complexity, not code
golf or indiscriminate low line count.  It wants a short why-merge explanation
and explicitly rejects several newcomer submission classes.

Read pinned
[`README.md` lines 165–186](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/README.md#L165-L186).

Question: list every discouraged and wanted category.  Which rule would apply
if material from this independent guide were proposed upstream?  Why is a
three-line feature not automatically useful?

### Stop 2: What testing instructions are policy rather than universal truth?

Prediction: the README gives install/test examples, points at CI, and contains
an agent-specific parallelism instruction at this snapshot.

Read pinned
[`README.md` lines 188–205](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/README.md#L188-L205).

Question: which commands are examples, what should be read for the full matrix,
and why might a flaky order-dependent test need serial diagnosis despite the
parallelism instruction?

### Stop 3: What does the project license actually permit and require?

Prediction: the pinned package declares MIT, and the LICENSE grants broad
rights conditioned on retaining its notice and disclaims warranties.

Read pinned
[`pyproject.toml` metadata](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L1-L12)
and the complete short
[`LICENSE`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/LICENSE#L1-L7).

Question: what notice condition appears?  Why does tinygrad's MIT license not
authorize copying code from a differently licensed third-party repository?

### Stop 4: What is process replay's explicit scope?

Prediction: it captures branch process inputs, regenerates on master, diffs
kernels, does not assert by default, and can early-stop.

Read the complete pinned
[`process replay README`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md#L1-L19).

Question: which branch captures, where data is stored, which branch replays,
what `[pr]` changes, and what numerical/runtime claims remain unproved?

### Stop 5: How does assertion mode arise?

Prediction: title/commit markers determine `ASSERT_DIFF`, but an environment
control can disable it; import failure exit behavior depends on assertion mode.

Read pinned
[`process_replay.py` setup](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L1-L40).

Question: distinguish `[PR]`, `[skip_process_replay]`, `ASSERT_PROCESS_REPLAY`,
and `REF=master`.  Which spellings/cases are used in implementation versus the
README/CI convention?

### Stop 6: What exactly is compared?

Prediction: replay converts program artifacts to source strings, compares
those strings, logs unified diffs, and turns warnings into errors only in
assertion mode.

Read pinned
[`replay and diff implementation`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L42-L97).

Question: where do `good` and `compare` come from?  Why can identical strings
still coexist with a runtime bug?  Why can a legitimate optimization produce a
diff that needs interpretation rather than automatic rejection?

### Stop 7: When can replay stop or convert infrastructure trouble to status?

Prediction: the README describes a changed-kernel percentage, but the pinned
script actually counts exceptions per page against a raw threshold.  Ordinary
diff warnings count differently depending on assertion mode, and the top-level
handler returns an assertion-dependent status for escaping replay errors.

Read pinned
[`mapping and main loop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L99-L128).

Question: where is `changed` reset?  Which events increment it with
`ASSERT_DIFF=0`?  Why does warning-to-error promotion usually abort an
asserting task before the advertised threshold?  Why must a contributor inspect
logs and captured coverage rather than reporting only exit zero?

### Stop 8: How does CI enable capture?

Prediction: only PR events whose title satisfies a GitHub Actions `contains()`
test set the global capture variable.  The expression contains a lowercase
`[pr]` literal, but GitHub evaluates `contains()` case-insensitively, so an
uppercase `[PR]` title also matches.

Read pinned
[`test workflow header`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1-L20).

Question: what event condition is checked, and how does its case behavior
differ from Python's uppercase substring check?  Why should the current
workflow be reopened immediately before relying on this convention?

### Stop 9: What do local hooks actually run?

Prediction: the pinned pre-commit configuration runs ruff, a tiny test, mypy,
device examples, and a selected comprehensive pytest command; it is not the
entire CI workflow.

Read the complete pinned
[`pre-commit configuration`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.pre-commit-config.yaml#L1-L34).

Question: which hooks ignore staged filenames and always run?  Why does hook
success not establish a GPU-specific claim?

### Stop 10: How are test groups named?

Prediction: the tiny `test/README` distinguishes backend, null, and unit by CI
execution role rather than ordinary English meanings.

Read all five lines of pinned
[`test/README`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/README#L1-L5).

Question: where should a renderer-independent UOp invariant likely live, and
what current neighboring tests must you inspect before deciding?

### Stop 11: What dependencies and Python versions does the snapshot declare?

Prediction: core has no required dependencies, Python is at least 3.11, and
testing extras introduce pytest, Hypothesis, Z3, Torch, and other packages.

Read pinned
[`project metadata and testing extras`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L1-L12)
and
[`optional dependencies`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L57-L103).

Question: which extra is the smallest fit for a focused test?  Why must a
reproducer record versions even when core declares no dependency?

### Stop 12: How is simplicity partially measured in CI?

Prediction: `sz.py` walks Python and JavaScript files throughout `tinygrad/`,
excluding only `tinygrad/runtime/autogen` and `tinygrad/viz/assets`.  It reports
per-file statistics and both a separately filtered `core lines` subtotal and a
total.  CI's ceiling applies to the total, not the core subtotal.  This metric
supplements rather than defines readability.

Read pinned
[`sz.py` traversal and report`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/sz.py#L17-L95),
then the pinned workflow's
[`line-count assertion`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L208-L223).

Question: which two paths are excluded from traversal, which directories are
excluded only from the core subtotal, what total-line threshold is passed by
CI, and why can a smaller but code-golfed patch still violate project goals?

### Stop 13: Which CI job is evidence for your subsystem?

Prediction: Python backend, NULL, unit, SPEC, fuzzing, hardware, and benchmark
jobs run different commands and configurations.

Read the workflow index locally:

```bash
rg -n '^  [a-zA-Z0-9_-]+:|name:|run:' .github/workflows/test.yml
```

Then open only the bounded job relevant to the candidate.

Question: what exact command, environment, dependencies, runner, timeout, and
exclusions apply?  Which part can you reproduce locally, and which evidence
still depends on CI/hardware?

### Stop 14: Recheck live state

After understanding the snapshot, reopen the live policy, exact candidate
issue, linked PRs, and current source.  Do not merely compare line numbers.

Question: what changed since `874d331`?  Does that change candidate readiness,
test commands, `[pr]` use, AI disclosure, or project fit?  Record the UTC time
and exact URLs.

## Background ladders

Use only the level that blocks the current candidate.

### Level 0: Git branches, diffs, and worktrees

Read the official Git documentation for
[`git status`](https://git-scm.com/docs/git-status),
[`git diff`](https://git-scm.com/docs/git-diff),
[`git switch`](https://git-scm.com/docs/git-switch),
[`git worktree`](https://git-scm.com/docs/git-worktree), and
[`git revert`](https://git-scm.com/docs/git-revert).

Stop when you can create a branch/worktree from an exact base, inspect staged
and unstaged changes, make one coherent commit, and recover it without broad
destructive commands.

### Level 1: issue and PR literacy

Read GitHub's official documentation on
[`issues`](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)
and
[`pull requests`](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

Stop when you can inspect timeline, edits, linked work, review status, base
branch, commits, checks, and merge/close state without equating “open” with
“available.”

### Level 2: regression and evidence design

Revisit [Debugging](15-debugging.md), [Testing](16-testing.md), and
[Performance](17-performance.md).  Stop when you can show baseline red,
candidate green, test power, first bad/costly artifact, and an evidence claim
no broader than the experiment.

### Level 3: licenses and provenance

Read the actual license of every source you may reuse and GitHub's official
guidance on
[`licensing a repository`](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
If employer/contract or compatibility questions remain, obtain qualified legal
guidance.  Stop when every nontrivial copied/adapted artifact has a known source
and compatible treatment.

### Level 4: subsystem specialization

Use the branch router in
[Learning resources](../reference/learning-resources.md#specialized-contribution-branches).
Write one exact unknown, select an authoritative specification/source section,
and define a return exercise.  Do not consume an entire ISA or compiler course
when one operation's contract is the blocker.

### Level 5: upstream communication

Study a few accepted and declined current PRs in the same subsystem.  Observe
the evidence, diff size, review questions, and maintainer tradeoffs without
copying personalities or assuming one case is policy.  Stop when you can ask
one bounded question and present a claim without overselling it.

## Common misconceptions, corrected

| Misconception | Correction |
| --- | --- |
| An open bounty is available to whoever starts coding. | Verify current terms, ownership, linked work, and acceptance process. |
| Issue text is the specification. | It is dated conversation evidence; rebuild the current contract from source and maintainer direction. |
| No assignee means no one is working on it. | Search linked and semantic PR/issue overlap and recent comments. |
| A plausible fix makes a candidate Ready. | Readiness begins with contract, reproduction, oracle, ownership, and evidence path. |
| A spike is almost a PR. | A spike answers feasibility; a patch must be general, maintainable, tested, and policy-fitting. |
| Full tests passing proves the bug fix. | The focused regression must fail on baseline for the intended reason and pass after. |
| Process replay proves correctness. | It diffs generated kernels for captured inputs; it does not execute a public numerical oracle. |
| `[pr]` is a generic CI marker. | It has a narrow snapshot-defined refactor/speedup no-expected-change meaning. |
| `PYTHON::sm_89` is RTX 4090 hardware evidence. | It is targeted lowering/interpreter evidence, not physical GPU execution. |
| One fast run proves a speedup. | Use correctness-bracketed synchronized distributions and attribution. |
| More commits are more atomic. | Atomicity is coherent purpose and validation, not fragmentation. |
| A prerequisite refactor should always be separate. | Separate it only if it is an independent clear win without the follow-up. |
| “MIT project” means any GitHub code can be copied into it. | Each source retains its own license/provenance obligations. |
| AI disclosure makes generated code acceptable. | You must still understand, test, source, and defend every line under live policy. |
| Contributing docs is the safest newcomer PR. | The pinned upstream policy explicitly says otherwise. |
| A rejected PR means the investigation was wasted. | Preserve the learning and evidence; merge is not the only useful outcome. |
| Maintainers owe review after substantial effort. | Review is scarce and upstream controls project direction. |
| This guide is associated with tinygrad. | It is independent, unofficial, and not endorsed or maintained upstream. |

## Exercises

Try each before opening the answer.

### 1. Idea or contract?

Classify “make reductions faster” and rewrite it as a bounded contract.

??? answer
    It is an idea.  A bounded performance contract names workload, shapes,
    dtype, device/backend/driver, warm/cache state, correctness oracle, timing
    method, distribution, baseline, acceptable complexity, and non-goals.  For
    example: reduce median synchronized time for one recorded reduction family
    on a named route beyond noise without changing values or model compile time
    materially.

### 2. Is an open issue current?

An issue is open, unassigned, and two years old.  May you mark it Ready?

??? answer
    No.  Check current reproduction, all comments/edits, linked and semantic
    PRs/issues, current source/tests/history, live policy, resources, and
    expected behavior.  Open plus unassigned is discovery evidence only.

### 3. Bounty conflict

The spreadsheet says one acceptance target; the issue's latest maintainer
comment appears to say another.  What is the decision?

??? answer
    Question.  Record both exact links/timestamps and a minimal current
    reproduction, then ask which target governs before major implementation.
    Do not silently choose the easier interpretation.

### 4. Define non-goals

A CPU renderer bug affects float16 nonzero-offset views.  Name three useful
non-goals.

??? answer
    Examples: no public Tensor API change; no performance claim; no CUDA/NV
    runtime claim.  These do not remove the need to test any shared renderer
    code that can affect other dtypes/routes.

### 5. Spike or patch?

You do not know whether the target compiler accepts the required instruction.
What should you build first?

??? answer
    A bounded compile/run spike with exact target, dtype, corner cases, and a
    stop condition.  Its result decides feasibility.  If successful, return to
    a clean branch and implement the general tested patch rather than polishing
    hard-coded spike code.

### 6. First bad artifact

Frontend, schedule, and lowered LINEAR are correct; SOURCE first drops a cast.
Where should the smallest change normally live?

??? answer
    In the LINEAR-to-SOURCE renderer transition that owns the dropped cast,
    unless current architecture reveals an earlier violated renderer
    precondition.  A downstream runtime exception or frontend special case
    would hide the first bad boundary.

### 7. Baseline-red control

Your new regression passes before the patch.  Can the PR still call it a
regression test for the issue?

??? answer
    Not as written.  It does not detect the baseline defect.  Preserve the real
    trigger, strengthen the oracle, or revise the claim.  A separate passing
    neighboring test can be useful, but it is not baseline-red evidence.

### 8. Atomic commits

Should a bug test and its two-line fix always be separate commits?

??? answer
    No.  One coherent commit containing regression and fix is often more atomic
    because it has one contract and remains green.  Record the baseline-red run
    outside final history.  Separate only independently useful changes with
    their own validation.

### 9. Prerequisite refactor

A feature becomes three lines after a 200-line refactor.  When should the
refactor be separate?

??? answer
    Only when the refactor is an independent clear win—readability or
    complexity improvement worth merging even if the feature never arrives—
    with unchanged-behavior evidence and reviewable scope.  Otherwise continue
    reducing or justify the coupled cost.

### 10. Process replay

Process replay exits zero with no printed diffs.  What has been proved?

??? answer
    Only what the exact mode, captured rows, logs, assertion setting, and
    early-stop behavior support.  It may show unchanged generated source for a
    captured corpus.  It does not prove public numeric correctness, runtime
    order, uncaptured inputs, or hardware performance.

### 11. `[pr]`

Should every PR include `[pr]` to get more CI?

??? answer
    No.  At the pinned snapshot it is specifically for refactor/speedup PRs
    with no expected generated behavior change.  A lowercase title marker
    enables capture/the action because GitHub Actions `contains()` ignores
    case, but it does not by itself satisfy the replay script's case-sensitive
    uppercase assertion check.  Uppercase in the title enables both; uppercase
    in the exported commit message can also enable assertion after the action
    runs.  Recheck live instructions and use the marker only when the semantics
    match.

### 12. Hardware claim

`PYTHON::sm_89` passes and rendered PTX looks correct.  May the PR say “tested
on RTX 4090”?

??? answer
    No.  That route provides Ada-targeted lowering/interpreter and artifact
    evidence, not physical GPU execution.  Name the route exactly.  Test on the
    recorded physical device/API or leave the hardware claim unproved.

### 13. Performance attribution

One kernel is 30% faster but end-to-end model time is unchanged.  What claim is
valid?

??? answer
    The affected kernel improved under the recorded protocol.  A model speedup
    is not established.  Report kernel share, compile/launch effects, raw
    distributions, and complexity; decide whether project value remains.

### 14. Missing hardware

A candidate's acceptance depends on an AMD device you do not have.  What are
honest options?

??? answer
    Narrow only if code/claim genuinely excludes AMD, obtain one bounded run
    from someone with hardware, keep it Research, or decline.  Emulation can
    support structural evidence but cannot be relabeled physical validation.

### 15. Communication

When is “I am working on this” insufficient?

??? answer
    Always as an engineering claim.  For substantial/ambiguous work, state the
    current reproduction, contract, first-bad evidence or bounded unknown,
    overlap checked, and exact question.  Follow the live ownership process;
    an announcement alone may not reserve anything.

### 16. Duplicate PR

You discover an active PR with the same contract but a different implementation.
What next?

??? answer
    Read its scope and status.  Avoid opening a competing duplicate by default.
    Offer a minimal reproducer, review evidence, or bounded alternative if
    useful and welcomed.  Ask when ownership or missing requirements are
    unclear.

### 17. AI use

An assistant generated the patch, but every test passes.  What remains?

??? answer
    Follow current disclosure policy; personally understand and verify every
    line and claim; establish provenance; check for copied/invented code;
    reproduce baseline red/candidate green and all reported evidence; and
    remove anything you cannot defend.  Passing tests alone is insufficient.

### 18. License

Tinygrad is MIT.  Can you paste a useful GPL implementation into a core file?

??? answer
    Do not assume so.  Tinygrad's license governs tinygrad, not third-party
    code.  Determine the source license and compatibility/notice obligations,
    project policy, and your authority.  If uncertain, do not copy and seek
    qualified guidance.

### 19. Affiliation

May this guide be described as “tinygrad's AI contributor course”?

??? answer
    No.  That suggests upstream ownership or endorsement.  Describe it as an
    independent, unofficial study guide about tinygrad, and direct guide
    support/corrections to this repository.

### 20. Review change

A review-requested rewrite changes generated kernels.  Can you retain the old
process replay and benchmark results?

??? answer
    Not as evidence for the new diff.  Rerun focused correctness, applicable
    process replay, and affected performance measurements; update claims,
    risk, and limitations.

### 21. Rollback

Why define rollback before merge if tests are strong?

??? answer
    Tests cover a finite domain.  A predeclared observable signal and simple
    recovery reduce response time and reveal irreversible/stateful risks during
    design.  Rollback planning complements rather than distrusts tests.

### 22. Stop decision

After a week, the patch works only through three backend-specific exceptions
and the measured benefit is within noise.  Continue because of sunk effort?

??? answer
    No.  Return to Research or decline.  The original value/complexity premise
    failed.  Preserve the negative result and avoid asking reviewers to merge
    complexity justified by time already spent.

## Checkpoint

You are ready to pursue a real candidate when you can:

- explain idea, issue, bounty, spike, patch, commit, and PR without conflating
  them;
- separate durable method, pinned source facts, and live state;
- recheck live policy, exact issue/bounty terms, comments, linked work, current
  source, tests, and history at recorded times;
- choose Ready, Research, Question, or Decline and name what changes it;
- write a falsifiable contract with preconditions, success criteria, non-goals,
  and rollback;
- reproduce on an exact current commit and define an independent oracle;
- locate the first wrong/costly artifact and its owning layer;
- choose evidence proportional to frontend/compiler/runtime/JIT/performance/
  hardware claims;
- time-box a feasibility spike and restart clean for a patch;
- create a branch/worktree and inspect every staged change;
- make conservative atomic commits with no unrelated cleanup;
- prove the focused test red on baseline and green after the same change;
- state tests/backends/hardware not run and how that limits readiness;
- explain exact pinned process replay and `[pr]` semantics without treating them
  as universal correctness;
- ask a bounded evidence-backed upstream question only when local work cannot
  resolve a material decision;
- prepare a PR body with why-merge, root cause, change, evidence, limitations,
  risk, rollback, provenance, and AI disclosure;
- rerun invalidated evidence during review;
- avoid duplicate work, entitlement, hostile language, and affiliation claims;
- verify licenses/provenance for every copied/adapted artifact; and
- stop when current direction, evidence, resources, safety, provenance, or
  complexity no longer supports submission.

If you cannot localize, return to Chapter 15.  If you cannot make a powerful
regression, return to Chapter 16.  If you cannot defend a speed claim, return
to Chapter 17.  If one compiler/GPU fact blocks the candidate, use the narrow
background branch rather than guessing.

## Quick reference

```text
LIVE CHECK
  policy → bounty terms → issue timeline → linked/semantic PRs
  → current source/tests/history → exact base commit

TRIAGE
  candidate → contract → success/non-goals → resources/ownership
  → Ready | Research | Question | Decline

EVIDENCE
  reproduce → oracle → baseline red → last good / first bad
  → owning layer → focused test → proportional matrix

IMPLEMENT
  bounded spike if needed
  → clean branch/worktree
  → smallest owning change
  → coherent atomic commit(s), no unrelated cleanup

CLAIMS
  correctness: same regression red/green + domain matrix
  performance: synchronized distributions + attribution + complexity
  hardware: exact physical route, never emulation relabeled
  replay: generated-kernel diff for captured inputs, not numeric runtime proof

PR
  why merge + current issue/bounty + root cause + change/non-goals
  + exact tests/perf/hardware + unrun limits + risk/rollback
  + license/provenance + AI disclosure

STOP / ASK
  ambiguous intent or terms | active overlap | unavailable essential hardware
  unsafe route | unproved provenance | stale reproduction | scope explosion
  complexity > measured value | maintainer says no

RELATIONSHIP
  upstream controls review and direction
  this guide is independent, unofficial, and not endorsed
  guide problems belong here, not in tinygrad's tracker
```
