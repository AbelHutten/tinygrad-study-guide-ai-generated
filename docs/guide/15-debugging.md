# 15. Debugging across the pipeline

## The promise of this chapter

A tinygrad failure often becomes visible far from the code that caused it.  A
wrong dtype created in the frontend may survive until a renderer rejects it.  A
correct kernel may receive the wrong runtime argument.  A correct ordinary run
may fail only when TinyJit reuses buffers.  A missing queue dependency may
disappear when debug timing forces synchronization.

That distance creates a common trap: starting from the loudest symptom and
changing the subsystem that happened to report it.  A compiler error does not
prove the compiler created malformed source.  A GPU fault does not prove the
runtime chose a bad address.  A wrong Tensor result does not prove Tensor
addition is wrong.

The useful question is:

> What is the first artifact whose required invariant is false?

This chapter develops that question from first principles.  It teaches how to
turn a report into a deterministic experiment, define an independent oracle,
compare adjacent pipeline artifacts, interpret tinygrad's debugging flags at
the pinned snapshot, use Python tracebacks and `pdb`, and convert the result
into a focused regression test.

The bundled lab injects one deliberately narrow, process-local CPU renderer
fault.  It does not edit or monkeypatch the tinygrad checkout.  Its portable
control, exact defect reproduction, and fixed regression are separate automated
modes, all with unambiguous exit status.  By the end, you should be able to:

- distinguish a **symptom** from a **cause**;
- define a value oracle or structural invariant before inspecting internals;
- make a reproducer minimal without accidentally removing the failing path;
- freeze process, environment, dtype, device, randomness, optimization, JIT,
  and cache state;
- read a traceback and navigate relevant frames with `pdb`;
- compare frontend, schedule, lowered `SINK`, ordered `LINEAR`, `SOURCE`,
  `BINARY`, runtime, and JIT artifacts in adjacent pairs;
- state what each pinned `DEBUG` level and validation flag does and does not
  establish;
- capture rewrite data with `VIZ` and inspect the exact saved file through the
  command-line viewer;
- use controlled backend and JIT comparisons without treating them as verdicts;
- recognize when timing or synchronization changed the experiment;
- debug wrong values, exceptions, compiler failures, process/GPU faults,
  lifetime races, view/dtype failures, optimization-only failures, and JIT-only
  failures with different decision trees;
- gather hardware evidence without repeatedly submitting a known-faulting GPU
  program; and
- prepare an issue and a red-before/green-after regression that another person
  can reproduce.

All exact source observations and links target tinygrad commit `874d331` from
2026-08-05.  Debug output, pass names, flags, and runtime behavior change.  Use
this chapter to understand the pinned system, then recheck the live source and
current contribution policy before reporting or patching a present-day issue.

## Route through the chapter

Read front to back once.  The order is intentional:

1. separate observation, expectation, and cause;
2. learn value oracles and structural invariants;
3. name every artifact in the ordinary and JIT paths;
4. build a deterministic minimal reproducer;
5. learn tracebacks and basic `pdb` navigation;
6. choose the smallest tinygrad observation tool;
7. understand validation, cache, compiler, and disassembly controls;
8. compare backends, synchronization modes, optimizers, and JIT states;
9. run the controlled renderer-fault lab;
10. use the decision tree matching the symptom class;
11. consult history only after current ownership is located; and
12. turn the evidence into a regression and reviewable report.

This chapter assumes Python and ML familiarity, not compiler or GPU-debugging
experience.  Terms are defined before source links.  The background ladders
near the end tell you what to study if tracebacks, floating-point comparison,
generated C, queues, or Git history become the blocker.

## Observation, expectation, and cause are different

### A symptom is what you can observe

Examples:

- result is `[4, 5]` instead of `[5, 6]`;
- output dtype is `float16` instead of `float32`;
- `compile_cached` raises a compiler exception;
- Python exits with a segmentation fault;
- the CUDA driver reports an illegal memory access;
- call three of a TinyJit function reuses call two's value;
- a test fails only with `NOOPT=0`; or
- a failure vanishes under `DEBUG=2`.

A useful symptom statement is concrete and repeatable:

```text
At commit X, DEV=CPU:CLANG, float32 input [1,2,3], expression x+4,
fresh process, returns [-3,-2,-1]; expected [5,6,7].
```

“Addition seems broken” is already a causal guess.  “Model is unstable” lacks
an input, output, and invariant.

### An expectation must come from an oracle or contract

An **oracle** tells you what should happen independently of the path under
investigation.  For values it can be:

- arithmetic simple enough to compute by hand;
- a small NumPy or PyTorch expression whose semantics match exactly;
- a trusted reference implementation;
- a high-precision implementation with an explicitly justified tolerance; or
- a second implementation that does not share the suspected component.

A **structural invariant** tells you what must remain true even when there is no
single expected numeric value.  Examples:

- an output shape is `(2, 3)` and dtype is `float32`;
- every `LOAD` index is gated or provably in bounds;
- `ProgramInfo.globals` agrees with the runtime argument order;
- a rewrite preserves dtype and maximum shape;
- `LINEAR` places a producer before its consumer;
- a view covers bytes `[offset, offset+nbytes)` inside its base; or
- a replay updates every buffer parameter named by the captured plan.

Write the oracle or invariant before reading the implementation.  Otherwise it
is easy to reinterpret the desired behavior to match the current output.

### A cause is the earliest violated required contract

Suppose `x+4` returns `x-4`.  Possible causes include:

- the frontend created `SUB`;
- scheduling changed `ADD` to `SUB`;
- lowering selected the wrong operation;
- the renderer spelled lowered `ADD` as `-`;
- the compiler miscompiled valid source;
- runtime arguments were swapped; or
- stale memory made the observed values unrelated to this call.

The final values alone do not choose among them.  If frontend, scheduled
`SINK`, and lowered `LINEAR` all still contain `ADD`, but `SOURCE` contains a
subtraction, `SOURCE` is the first bad artifact.  The renderer owns that
transition.  `BINARY` and the result are also bad, but they are downstream.

### Root cause and contributing condition

Sometimes the first violated contract explains correctness while another
condition explains why the failure appeared:

```text
root cause:          missing cross-queue dependency
trigger condition:  allocator reused a buffer quickly
masking condition:  DEBUG=2 waited after every call
```

Record all three without confusing them.  Fix the missing dependency, not the
debug level or the timing that happened to reveal it.

## Define the evidence contract before collecting evidence

For each experiment, write four lines:

```text
Question:   Does the CPU renderer change float32 ADD semantics?
Input:      exact program, shape, dtype, device, flags, commit
Invariant:  lowered LINEAR is identical; SOURCE store remains addition
Evidence:   structural comparison plus independently checked result
```

This prevents an experiment from silently expanding its claim.  A rendered
source comparison can localize a renderer difference.  It cannot prove the
source compiles, loads, executes, or races correctly.  A passing physical GPU
result proves one scoped execution, not race freedom for all schedules.

### Evidence strength is claim-specific

There is no universal single ladder where “hardware” always beats “static.”
Match evidence to the claim:

| Claim | Minimum direct evidence | What is still not proved |
| --- | --- | --- |
| Frontend created the requested operation | Inspect the exact lazy UOp graph and public shape/dtype | Scheduling or execution |
| A rewrite preserved an invariant | Compare its input/output graphs or a structural property | Numeric behavior on a runtime |
| Renderer is the first bad boundary | Same lowered input, semantically different `SOURCE`, correct oracle | Compiler correctness |
| Compiler accepts the artifact | Exact toolchain successfully compiles exact `SOURCE` for target | Runtime load or result |
| Runtime executes this program correctly | Correct result on the selected physical runtime with completion established | Other shapes, devices, or races |
| A bug is JIT-only | Same ordinary function passes with `JIT=0`; capture/replay route fails | Which JIT substage owns it |
| A queue dependency is sufficient | Producer/consumer test with scoped event/order on hardware | Unrelated queue patterns |
| An optimization improves performance | Repeated synchronized/event timing after warm-up with identical semantics/artifact scope | Other workloads or devices |

Agreement between two backends is stronger only to the extent that they do not
share the suspected bug.  Python and CPU share frontend and much of scheduling;
both can agree on the same wrong scheduled computation.

## Name the artifacts before comparing them

### The ordinary realization path

```text
public Tensor expression
  ↓ frontend construction
lazy Tensor UOp graph
  ↓ callify / scheduling
outer LINEAR of CALLs
  ├── CALL(SINK, buffer UOps...)      kernel not compiled yet
  ├── CALL(COPY, ...)
  └── CALL(SLICE, ...)
  ↓ target lowering of a kernel SINK
lowered SINK
  ↓ linearization
ordered program LINEAR
  ↓ renderer
SOURCE
  ↓ compiler
BINARY bytes
  ↓ runtime construction and dispatch
loaded Program + handles + dimensions + scalar values
  ↓ queue completion / copyout
public result
```

The two objects called `LINEAR` have different scopes:

- the **outer execution LINEAR** is an ordered list of `CALL`s; and
- the **program LINEAR** inside `Ops.PROGRAM` is an ordered list of lowered
  operations for one kernel.

State which one you inspected.

### The JIT branch

TinyJit intercepts ordinary realizations during capture, combines their outer
LINEAR calls, substitutes input parameters, memory-plans, compiles the plan,
and may replace graphable batches with device graph calls:

```text
ordinary first call (ignore)
  ↓ second call capture
captured outer LINEAR calls
  ↓ parameterize + memory plan + compile
prepared captured LINEAR
  ↓ with ordinary JIT=1, when the backend and batch support graphing
graph-split LINEAR
  ↓ replay with new buffers/scalars
result
```

A JIT-only failure is not automatically a graph-runner failure.  It can arise
before graphing during capture, parameterization, memory planning, or prepared
plan execution.

### Adjacent comparisons, not distant comparisons

Compare one transition at a time:

| Last good | Candidate first bad | Question |
| --- | --- | --- |
| Public expression/oracle | frontend graph | Did construction encode the requested semantics, dtype, shape, and dependencies? |
| frontend graph | scheduled outer LINEAR | Did fusion/materialization/copy ordering preserve semantics? |
| scheduled `SINK` | lowered `SINK` | Did target legalization preserve operations, indices, gates, effects, and dimensions? |
| lowered `SINK` | program `LINEAR` | Did linearization preserve control/effect order? |
| program `LINEAR` | `SOURCE` | Did rendering spell the same operations and ABI? |
| `SOURCE` | `BINARY` | Did the selected compiler accept and translate the exact artifact? |
| `PROGRAM` metadata/handles | runtime result bytes | Were name, arguments, dimensions, ordering, and completion correct? |
| ordinary prepared calls | JIT prepared/graphed calls | Did capture, parameterization, planning, grouping, or replay alter the contract? |

If A is good and B is bad, inspect the code that creates B from A.  Do not jump
two arrows and patch the later consumer.

### A compact investigation ledger

Keep a small table in the issue or notebook:

| Artifact | Invariant | Observation | Verdict |
| --- | --- | --- | --- |
| Oracle | `x+4 = [5,6,7]` | hand calculation | good |
| Frontend | contains float32 `ADD` | observed `ADD` | good |
| Scheduled `SINK` | contains same float32 `ADD` | observed `ADD` | good |
| Lowered `LINEAR` | same for standard/fault route | structural identity | good |
| `SOURCE` | output store uses addition | fault route uses subtraction | **first bad** |
| `BINARY` | downstream translation of good source | differs after bad source | bad downstream |
| Result | equals oracle | `[-3,-2,-1]` | bad downstream |

One sentence per artifact is better than a thousand-line log with no invariant.

## Build a deterministic minimal reproducer

### Minimal means smallest while preserving the same failure

A useful reduction sequence is:

```text
model → layer → operation chain → one Tensor expression → one scheduled kernel
```

After every reduction verify:

1. the external symptom still occurs;
2. the same oracle still applies; and
3. once located, the same first bad artifact remains first.

A smaller program that fails for a different reason is a different reproducer.

### Reduce one dimension at a time

| Dimension | Controlled experiment | Trap |
| --- | --- | --- |
| Shape | Reduce axes separately; retain boundary sizes such as 0, 1, vector width ±1 | Power-of-two shapes can remove masks/tails. |
| Values | Preserve one fixed counterexample, then vary signs, zero, extrema, NaN/Inf | Randomizing first makes comparison noisy. |
| Dtype | State concrete dtype; vary one dtype at a time | Promotion or tensor-core eligibility can change the path. |
| Layout | Remove one permute/pad/shrink/expand/view at a time | `contiguous()` creates a copy and can hide the bug. |
| Kernels | Remove one forced realization/materialization boundary at a time | Fusion and memory planning may change. |
| Optimizer | Vary `NOOPT`, `BEAM`, and `TC` separately | Disabling all at once does not identify which path mattered. |
| Backend | Hold input/config fixed; change one route | Different targets can legally lower differently. |
| JIT | Compare `JIT=0`, then `2`, then `1` | One-call tests never reach replay. |
| Cache | Fresh process, then isolate schedule/compiler/disk caches | Clearing everything hides the cache owner. |

Do not simplify away the dtype, view, symbolic bound, tail, or replay that makes
the bug possible merely to obtain a shorter file.

### Freeze the process envelope

Many tinygrad settings are read into `ContextVar`s at import time.  Opened
devices, renderers, compiled programs, runtime programs, and several caches are
process-local.  Changing `os.environ` after importing tinygrad is not equivalent
to starting with that environment.

Launch each comparison as a fresh process and record:

```text
tinygrad commit
Python version
OS/architecture
DEV target and selected renderer target
explicit dtype and exact input values
shape/layout/contiguity
DEBUG, VIZ, SPEC, CHECK_OOB
NOOPT, BEAM, TC
JIT
SCACHE, CCACHE, CACHELEVEL, CACHEDB
random seeds, if any
driver/device/toolchain versions for hardware routes
exact command and exit status
```

Set plan-affecting environment variables before the first tinygrad import.  Do
not run evidence-bearing labs under `python -O`; optimized Python removes
`assert` statements.  The bundled lab rejects that mode explicitly.

### Use explicit values, dtype, shape, and tolerance

Prefer:

```python
x = Tensor([1.0, 2.0, 3.0], device=Device.DEFAULT,
           dtype=dtypes.float32).realize()
actual = (x + 4.0).tolist()
assert actual == [5.0, 6.0, 7.0]
```

over an implicit default dtype and randomly generated values.  For approximate
operations, state both tolerances and why they are appropriate:

```python
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

`rtol` scales with magnitude; `atol` matters near zero.  Do not loosen them
until a failure disappears and call that a fix.  First characterize error by
dtype, magnitude, and operation.

### A reproducer should fail loudly for the intended reason

Avoid:

```python
try:
  run_everything()
except Exception:
  print("bug reproduced")
```

That converts import errors, missing dependencies, compiler crashes, and wrong
exception types into false success.  Assert the exact value/invariant or catch
only the expected exception type and verify its stable semantic property.

For an expected wrong-value mode, automation can still exit zero safely:

```python
actual = run_fault_route()
assert actual == exact_known_symptom
assert actual != oracle
print("status: expected-defect-reproduced")
```

The mode fails if the defect disappears or changes.  It does not make the
wrong result “correct.”

## Cache isolation without destroying useful state

### The pinned caches have different scopes

| Control/cache | Default | Scope and content |
| --- | ---: | --- |
| `SCACHE` | `1` | Process-local schedule cache keyed by the function UOp key. |
| `CCACHE` | `1` | Gives a `Compiler` its persistent cache key; `0` disables that compiler cache. |
| `CACHELEVEL` | `2` | Universal SQLite reads/writes occur only when level is at least one; `0` bypasses them.  Beam search also consults this control. |
| `CACHEDB` | user cache DB path | Selects the SQLite database file used by universal disk caches. |
| `to_program_cache` | no environment off switch here | Process-local compiled `PROGRAM` artifact cache. |
| `runtime_cache` | no environment off switch here | Process-local loaded/interpretable runtime program cache. |
| `Device[...]`/renderer caches | no simple global off switch | Opened device and selected renderer objects. |

A new process is the most important isolation boundary.  For the smallest
uncached experiment use:

```bash
SCACHE=0 CCACHE=0 CACHELEVEL=0 DEV=CPU:CLANG \
  .venv/bin/python reproducer.py
```

At this pinned snapshot, the universal disk-cache implementation distinguishes
only `CACHELEVEL < 1` from `CACHELEVEL >= 1`; levels `1` and `2` have no separate
behavior there.  `SCACHE=0` bypasses schedule-cache reads and writes but does
not erase entries already accumulated in the process.  `CCACHE` is captured
when a compiler object is constructed, so neither switch is a substitute for a
fresh process when making clean comparisons.

If beam search is part of the failing route, `IGNORE_BEAM_CACHE=1` forces a
fresh beam-cache read decision but still writes the newly searched result when
disk caching is enabled.  Its function default is captured at import.  For an
uncached localization experiment, the simpler envelope is normally
`BEAM=0 CACHELEVEL=0`; when studying beam itself, set the controls before launch
and preserve them in the report.

When testing cache behavior itself, keep caching enabled but isolate the DB:

```bash
debug_cache_dir=$(mktemp -d)
CACHEDB="$debug_cache_dir/cache.db" DEV=CPU:CLANG \
  .venv/bin/python reproducer.py
```

Record the temporary path while investigating.  Do not clear the user's entire
shared cache as a first move.  If a fresh uncached process passes and the cached
one fails, isolate schedule, compiler, disk, program, and runtime caches one at
a time.

### “Works after restart” is evidence

It narrows attention toward:

- process-local cache keys;
- mutable global/class state;
- opened device/context state;
- allocator reuse and lifetimes;
- captured TinyJit state; or
- environment values frozen at import.

It is not a sufficient bug report.  Record which restart and which state reset
changed the outcome.

## Read Python exceptions before adding instrumentation

### Traceback anatomy

A Python traceback has:

```text
Traceback (most recent call last):
  older caller frame
  ...
  newest frame where exception was raised
ExceptionType: message
```

Start at the final exception type and message.  Then read frames from the
bottom upward until you reach the first frame owned by the code or artifact
boundary under investigation.  The final frame may be a generic assertion or
driver wrapper; the caller may reveal the invalid object it received.

For each relevant frame record:

- file and line;
- function;
- important arguments and local values;
- which artifact was entering the function; and
- which invariant the exception states or implies.

Do not paste only the last line.  Do not assume the deepest tinygrad frame is
the root cause.

### Run under `pdb`

From the pinned checkout:

```bash
DEV=PYTHON .venv/bin/python -m pdb reproducer.py
```

On an uncaught exception, `pdb` gives post-mortem access to the traceback.  The
small command set needed initially is:

| Command | Meaning |
| --- | --- |
| `where` or `w` | Show the stack and current frame. |
| `up` / `down` | Move toward callers / newer frames. |
| `list` / `longlist` | Show source around the current line / whole function. |
| `p expr` / `pp expr` | Evaluate and print an expression in this frame. |
| `args` | Show current function arguments. |
| `break path:line` | Set a breakpoint before the bad transition. |
| `next` | Execute the next line in this frame. |
| `step` | Enter a called Python function. |
| `return` | Continue until this function returns. |
| `continue` | Continue to the next breakpoint/exception. |
| `quit` | Exit the debugger. |

Start with `where`, move to the producer/consumer boundary, and print a small
property such as `u.op`, `u.dtype`, `program.arg`, or the selected target.  Do
not print an enormous graph before you know which node matters.

### Break before the first bad transition

Once adjacent comparison says `LINEAR` is good and `SOURCE` is bad, break in
the render transition, not in Tensor construction or runtime copyout.  Inspect:

```text
input ordered UOps
selected renderer type and target
operation mapping used for the suspect UOp
returned source fragment
```

The debugger explains how the artifact was produced.  A separate oracle still
decides whether it is correct.

### Exceptions versus process crashes

`pdb` can inspect Python exceptions.  It cannot recover Python frames after a
hard process termination, native segmentation fault, driver reset, or machine
hang.  For native crashes, enable Python's standard
[`faulthandler`](https://docs.python.org/3/library/faulthandler.html) early:

```bash
PYTHONFAULTHANDLER=1 DEV=CPU:CLANG .venv/bin/python reproducer.py
```

It can print Python stacks around some fatal signals; it does not prove the
native cause or make continued GPU submissions safe.

## Choose the smallest tinygrad observation tool

Higher verbosity is not automatically stronger evidence.  Start with the
lowest level that answers one question, save the relevant fragment, then move
deeper only if the adjacent boundary remains unknown.

### Runtime `DEBUG` levels at the pinned snapshot

The checks are mostly cumulative `DEBUG >= N`, but output is backend- and
workload-dependent.

| Level | Important pinned behavior | Limit/cost |
| ---: | --- | --- |
| `0` | No general debug stream. | Your own assertions still run. |
| `1` | Prints opened devices; schedule summaries when an outer LINEAR has more than one call; JIT capture/prune summaries; some external-command/backend notices. | A one-kernel schedule may print no schedule summary. |
| `2` | Adds per-call execution statistics and JIT graph-batch notices; `run_linear` requests waiting, and stats synchronize if a runtime returned no duration. | Changes synchronization/timing and can mask races. |
| `3` | Prints schedule summary even for one call, selected/applied optimization information when present, and some library/tool initialization details. | Does not show every rewrite. |
| `4` | Prints generated `SOURCE` or printable ISA form before compilation; adds renderer/optimizer-specific detail. | Large models produce large output. |
| `5` | Prints `pyrender` of the base kernel AST before lowering; CUDA runtime can pretty-print loaded PTX; some backends add their own detail. | Not a universal pass-by-pass trace. |
| `6` | No single new universal core stream; some backends add detail.  In the **VIZ CLI**, level 6 renders all captured UOp graphs. | Do not describe it as automatic rewrite logging without VIZ data. |
| `7` | On rendered-SOURCE paths, calls the selected compiler's `disassemble` after successful compilation; it also logs Buffer allocation/deallocation.  Direct `ISARenderer` assembly bypasses that hook, and some compilers implement no disassembler.  In the **VIZ CLI**, level 7 reconstructs individual rewrites. | Very noisy; optional tools may be missing. |
| `8` | Adds SQLite trace output and a few backend-specific diagnostics. | Rarely the right first tool. |

Two important corrections follow:

- `DEBUG=2` is not a neutral print setting.  It changes `ExecContext.wait` and
  can call device synchronization for timing.
- `DEBUG=6/7` only have the graph/rewrite meanings in the VIZ CLI over captured
  rewrite data.  Ordinary execution at those levels has different, limited or
  backend-specific output.

### A practical progression

```text
DEBUG=0   reproduce with your own oracle
DEBUG=1   confirm selected device and multi-call/JIT summary
DEBUG=2   inspect executed calls; note changed waits
DEBUG=3   inspect schedule/optimization summary
DEBUG=4   capture exact SOURCE before compiler failure
VIZ=1     save rewrite boundaries
CLI 5/6  list passes, then render graphs around one transition
CLI 7    inspect individual matches only after the pass is known
DEBUG=7  optional disassembly after successful rendered-SOURCE compilation
```

## VIZ: capture first, inspect the exact file second

### What `VIZ=1` changes

At import time, nonzero `VIZ` supplies the **defaults** for two other settings:
`PROFILE` defaults to `abs(VIZ)`, and `TRACK_MATCH_STATS` defaults to `2`.
An explicitly exported `PROFILE=0` or `TRACK_MATCH_STATS=0` wins over those
defaults and can prevent the corresponding capture.  With profiling and match
tracking enabled, named `graph_rewrite` calls and explicit “View …” checkpoints
become inspectable.  At normal process exit the pinned code writes per-user
`rewrites.pkl` and `profile.pkl` files in the system temporary directory.  The
rewrite saver reports its exact path.  A later VIZ run by the same user
overwrites these defaults.

With an interactive TTY, positive `VIZ` can replace the finished workload
process with the web viewer.  `VIZ=-1` enables the same software
capture/profiling path while suppressing that automatic launch.  Use the latter
for a capture-only experiment; do not use `with Context(VIZ=1)` as an
equivalent because matcher substitution and exit-hook registration happen at
import time.

This means VIZ is instrumentation, not a neutral performance run.  A hard
process crash can also prevent orderly atexit capture.

### Capture a minimized run

```bash
VIZ=-1 PROFILE=1 TRACK_MATCH_STATS=2 DEV=PYTHON DEBUG=0 \
  SCACHE=0 CCACHE=0 CACHELEVEL=0 \
  .venv/bin/python reproducer.py
```

Setting all three instrumentation variables explicitly makes this command
independent of hostile values inherited from the shell.  If you instead rely
on `VIZ`'s defaults, first `unset PROFILE TRACK_MATCH_STATS` in that shell.

The rewrite saver prints its exact output path.  Query both exact pinned
defaults rather than guessing `/tmp` or a username:

```bash
DEBUG=0 .venv/bin/python -c 'from tinygrad.helpers import temp; print(temp("rewrites.pkl", append_user=True)); print(temp("profile.pkl", append_user=True))'
```

Copy both files to uniquely named evidence files before another VIZ run.  The
CLI defaults to the latest per-user files, but explicit copied paths avoid
reading data from a concurrent or later process:

```bash
DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path /exact/reported/rewrites.pkl \
  --profile-path /exact/reported/profile.pkl \
  --list
```

### Find the event; do not guess its generated name

Use the pinned README workflow:

```bash
DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path /exact/reported/rewrites.pkl \
  --profile-path /exact/reported/profile.pkl \
  -s TINY | rg 'Schedule|Kernel'
```

Copy the actual event name.  Event names can contain process-local counters.
Then list its named passes:

```bash
DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path /exact/reported/rewrites.pkl \
  --profile-path /exact/reported/profile.pkl \
  -s TINY 'copied event name' --ls
```

First survey graph inputs/outputs for every captured step in the selected
event.  Then reconstruct only one copied pass name at low `DEBUG`:

```bash
DEBUG=6 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path /exact/reported/rewrites.pkl \
  --profile-path /exact/reported/profile.pkl \
  -s TINY 'copied event name'

DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path /exact/reported/rewrites.pkl \
  --profile-path /exact/reported/profile.pkl \
  -s TINY 'copied event name' 'copied pass name'
```

At CLI `DEBUG=3`, base AST data is included; `4` adds source; `5` lists rewrite
steps and the kernel graph; `6` shows all UOp graphs; `7` reconstructs all
individual rewrites/diffs for every step in the selected event.  Supplying one
pass name already reconstructs that pass even at `DEBUG=0`; raising the level
to seven does **not** scope reconstruction to that name and can produce
thousands of lines.  Use level seven only when you intentionally want every
match after the event has already been narrowed.

### What VIZ establishes

VIZ can show:

- the graph before and after a named rewrite;
- the pattern rule and changed nodes when fully reconstructed;
- captured and graphed JIT LINEAR checkpoints; and
- runtime/profile events when the process completed capture.

It does not independently decide semantic correctness.  Bring the invariant to
the graph.  “The graph changed here” is expected for every rewrite; “this pass
dropped the store gate required by the input invariant” is a diagnosis.

## Structural validation flags

### `SPEC`

The pinned default is `SPEC=1`.  At schedule and codegen boundaries, nonzero
SPEC runs the relevant `type_verify` matcher over the graph/program.  The
schedule-side check occurs while constructing a schedule; an in-process
`SCACHE` hit reuses its `LINEAR` without repeating that check.

`SPEC=2` additionally makes UOp construction perform the pinned per-node full
spec check, and boundary verification tests Python rendering.  It also checks
an inferred dtype at construction for most non-constant UOps, but deliberately
skips nodes with invalid sources, operations without an inferred dtype, and
weak-equivalent `INDEX` access dtypes.  It is a useful stricter debugging/CI
experiment, not a claim that every possible dtype relation was checked:

```bash
SPEC=2 DEV=PYTHON .venv/bin/python reproducer.py
```

It can move the symptom earlier—from generated-code failure to illegal-UOp
construction—which is useful localization.  It also costs time and is not a
numeric oracle.  Higher SPEC modes include additional experimental checks;
the snapshot itself notes broken `SPEC=3` shape interaction in rangeify, so do
not recommend them as the ordinary first escalation.

### `CHECK_OOB`

The pinned default is `0`.  With `CHECK_OOB=1`, index validation first accepts
bounds proven by interval min/max.  For applicable uncertain cases it asks Z3
whether the gated index can exceed the buffer range.  It requires
`z3-solver>=4.12.4` when that proof path is reached.  For this pinned snapshot,
also keep the version below `4.15.4`: its `testing_minimal` dependency group
sets `z3-solver<4.15.4` because the adjacent comment records a 4.15.4
segmentation fault while creating many Z3 contexts.  In other words, the
snapshot's supported study interval is `>=4.12.4,<4.15.4`, not merely “any
recent Z3.”

Important limits in the pinned implementation:

- image shapes are skipped;
- indices involving `BITCAST` or `STACK` are skipped;
- overflow validation is a TODO; and
- it validates the modeled index/gate, not physical runtime pointer ownership.

Use it with boundary checking enabled, commonly:

```bash
SPEC=2 CHECK_OOB=1 DEV=PYTHON .venv/bin/python reproducer.py
```

A pass is strong static evidence for supported modeled indices, not proof that
an allocator, view, or runtime passed the correct base pointer.

### `DEBUG_RANGEIFY`

`DEBUG_RANGEIFY=1` passes a debug Boolean to the ordinary rangeify algorithm.
For each considered tensor node it prints:

```text
realization marker (***)
consumer count
operation
shape
ending-range count
input/output range rendering
```

It does not select a different scheduling algorithm at this snapshot.  Use it
to answer why ranges were inherited, created, ended, or forced to materialize:

```bash
DEBUG_RANGEIFY=1 SCACHE=0 DEV=PYTHON DEBUG=0 \
  .venv/bin/python reproducer.py
```

`SCACHE=0` matters: a scheduler-cache hit skips the rangeify work whose rows you
want to inspect.  The output describes a choice.  It does not prove that the
chosen graph computes the oracle.

### `VALIDATE_WITH_CPU`

In the ordinary default `run_linear(..., jit=False)` path,
`VALIDATE_WITH_CPU=1` rewrites each raw kernel call into:

```text
copies into CPU shadow buffers
original device call
CPU-rendered execution of the same scheduled SINK
NumPy assert_allclose on output buffers (rtol=1e-3, atol=1e-3)
```

This is useful when a device target/runtime is suspect:

```bash
JIT=0 VALIDATE_WITH_CPU=1 DEV=CUDA \
  .venv/bin/python reproducer.py
```

It does **not** provide an independent frontend or scheduler oracle.  The CPU
shadow renders the same scheduled kernel; both routes can agree on the same
wrong `SINK`.  It also adds copies, CPU compilation, memory, and completion
effects.

Prepared `run_linear(..., jit=True)` skips the compile step where validation is
inserted.  TinyJit's captured-plan lowering calls `compile_linear` without the
validation option in this snapshot.  Use `JIT=0` and retain a separate public
oracle when invoking this diagnostic.

## Compiler diagnostics and optional disassembly

### Localize a compile failure first

Classify these cases:

```text
no SOURCE exists
  → failure is earlier than rendering completion

SOURCE exists, compiler rejects it
  → compare source legality, target, compiler/tool version and invocation

compiler returns BINARY, runtime load rejects it
  → artifact format/target/runtime-loader boundary

module loads, launch rejects
  → symbol, signature, arguments, dimensions, resources or driver state
```

At the pinned snapshot, `DEBUG=4` prints generated source immediately before
`compile_cached`.  Capture the smallest exact source, target, renderer/compiler
class, environment, and full diagnostic:

```bash
DEV=CPU:CLANG DEBUG=4 SCACHE=0 CCACHE=0 CACHELEVEL=0 \
  .venv/bin/python reproducer.py > /tmp/repro.stdout 2> /tmp/repro.stderr
```

Redirection is useful only if you preserve both streams and exit status.  Do not
paste a megabyte of unrelated source when one kernel fails.

The Clang compiler at this snapshot invokes the selected `CC` (default
`clang`) on C from standard input with target/optimization/freestanding flags,
then JIT-links the object.  Other compilers have different diagnostics and
artifact formats.  Reproduce through the selected compiler first.  A standalone
compiler command is useful only when it faithfully preserves flags, target,
source bytes, and artifact stage.

### Disassembly is downstream evidence

On the ordinary rendered-`SOURCE` path, `DEBUG>=7` calls
`renderer.compiler.disassemble(lib)` after successful compilation.  The base
method is a no-op; backend compilers may require optional tools.  The pinned CPU
Clang route uses Capstone when installed and otherwise prints that it is
unavailable.  CUDA/AMD routes can invoke their own tooling.  A direct
`ISARenderer` instead takes `do_assemble`: `DEBUG>=4` can print its textual
instruction form, but that branch creates `BINARY` directly and never invokes
the `DEBUG>=7` compiler-disassembly hook.

Use disassembly when:

- `SOURCE` is already semantically correct;
- compilation succeeds;
- the question concerns actual machine instructions, widths, branches, or
  resource usage; and
- the disassembler matches the artifact/architecture.

Do not start from disassembly for a frontend shape bug.  Preserve a mapping
from relevant `LINEAR` operation to source statement to instruction region.

## Controlled backend comparisons

### Change one route dimension at a time

Use the same input values, concrete dtype, shape/layout, oracle, optimization,
JIT, and cache state:

```bash
DEV=PYTHON JIT=0 DEBUG=0 .venv/bin/python reproducer.py
DEV=CPU:CLANG JIT=0 DEBUG=0 .venv/bin/python reproducer.py
DEV=PYTHON::sm_89 JIT=0 DEBUG=0 .venv/bin/python reproducer.py
DEV=CUDA JIT=0 DEBUG=0 .venv/bin/python reproducer.py
DEV=CUDA:PTX JIT=0 DEBUG=0 .venv/bin/python reproducer.py
```

Interpret them as hypotheses:

| Split | Initial hypothesis | Required confirmation |
| --- | --- | --- |
| Python and CPU both wrong | Oracle, frontend, shared scheduling/lowering | Compare those artifacts; both can share the bug. |
| Python right, CPU wrong | CPU target lowering, C renderer/compiler/runtime | Prove where artifacts first differ. |
| `PYTHON` right, `PYTHON::sm_89` wrong | Target-capability lowering/optimization | Compare targets and lowered programs; runtime is still Python. |
| CUDA default wrong, `CUDA:PTX` right | Renderer/compiler route | Confirm schedule/lowered semantics and CUDA runtime are held sufficiently constant. |
| CUDA right, safe NV route wrong | Runtime/queue/interface path | Confirm renderer/artifact and completion contract first. |

`PYTHON::sm_89` is a target-lowering and interpreter experiment, not physical
GPU concurrency evidence.  `NULL` can test compilation/planning structure but
does not validate result bytes.

### Backend agreement is not independence

Draw shared components:

```text
frontend ─ schedule ─ shared lowering ┬─ Python renderer/runtime
                                     └─ CPU renderer/compiler/runtime
```

Agreement after the branch says the branch-specific pieces agree on the shared
input.  It does not prove the shared input is correct.  Keep the external oracle.

## Synchronization and timing can change the bug

### `DEBUG=2` is a diagnostic intervention

The pinned `run_linear` sets:

```text
ExecContext.wait = explicit wait OR DEBUG >= 2
```

If the runtime does not return a duration, stats synchronize the device and
measure host time.  Therefore:

```text
DEBUG=0 fails, DEBUG=2 passes
```

is evidence for timing, completion, lifetime, or race sensitivity.  It is not
evidence that print statements fixed arithmetic.

### Use waits to narrow, not to finish the patch

If a full device synchronize removes the failure:

1. name the producer command;
2. name the consumer command;
3. identify their queues/devices;
4. identify the shared buffer byte range;
5. locate the event/signal/ordered-queue edge that should connect them; and
6. replace the diagnostic global wait with the narrow correct dependency.

A global barrier can serialize unrelated work and conceal a resource-lifetime
bug.

### Race reproduction needs repetition and variation

Record:

- number of fresh-process trials;
- per-process iterations;
- input/shape/seed variation;
- queue and graph mode;
- explicit waits/events;
- buffer allocation/reuse state; and
- failure rate, not merely “flaky.”

Python's synchronous runtime can validate arithmetic and lowered semantics; it
cannot prove accelerator race freedom.

## Isolate JIT in the order `0 → 2 → 1`

On Linux at this snapshot the default `JIT` value is `1` (`2` is the default on
macOS x86).  The meanings relevant to debugging are:

| Setting | Behavior |
| ---: | --- |
| `JIT=0` | TinyJit wrapper executes the Python function normally on every call; no capture/replay. |
| `JIT=2` | Ignore, capture, and replay still occur, but `jit_lower` skips device graph splitting. |
| `JIT=1` | Capture/replay plus graph splitting where a backend graph class supports the call/batch. |

Run at least three calls—ignore, capture, first replay—and pass fresh realized
input buffers with changed values.  Record all three outputs.

```text
JIT=0 fails
  → ordinary path; leave JIT debugging

JIT=0 passes, JIT=2 fails
  → capture, parameterization, memory planning, compiled prepared calls,
    pointer/scalar update, or replay lifetime

JIT=2 passes, JIT=1 fails
  → graph eligibility/grouping, graph construction, node update,
    graph launch dimensions/arguments, or graph-specific ordering
```

`JIT=1` does not guarantee a graph call; the backend and batch must support it.
Inspect the captured outer LINEAR call bodies or VIZ “View captured linear” and
“View graphed linear” checkpoints before claiming graphing occurred.

## Bundled lab: localize one renderer fault

### What the lab changes—and what it does not

The repository file `labs/phase5/debugging_walk.py` defines a local
`FaultyClangRenderer` subclass.  It copies `ClangRenderer.code_for_op` and
changes only scalar `float32` `ADD` rendering from `a+b` to `a-b`.

It does not:

- edit the pinned tinygrad checkout;
- mutate `ClangRenderer.code_for_op`;
- change integer address arithmetic;
- catch arbitrary exceptions;
- depend on temporary variable or function names;
- rely on an implicit default dtype;
- retain schedule/compiler disk cache state; or
- run with assertions removed.

The injected program is built in a particularly strong way:

1. schedule `x+4.0` once;
2. lower, linearize, render, and compile it normally;
3. feed the exact already-lowered program `LINEAR` to the local faulty renderer;
4. compile that faulty `SOURCE`; and
5. replace only the `SOURCE`/`BINARY` children of a prepared `PROGRAM`.

The correct and faulty programs therefore share identical `SINK` and `LINEAR`
objects.  `SOURCE` is literally their first differing child.

### The three automation modes

| Mode | Route | Success condition | Exit behavior |
| --- | --- | --- | --- |
| `control` | `PYTHON` | Independent portable result equals `[5,6,7]` | Zero only on control pass. |
| `injected` | `CPU:CLANG` | Same lowered program renders `SUB` and produces exact known symptom `[-3,-2,-1]` | Zero only when the expected defect reproduces exactly. |
| `fixed` | `CPU:CLANG` | Standard renderer executes `ADD` and result equals `[5,6,7]` | Zero only on green regression. |

An exit-zero injected mode does not declare the wrong output correct.  It tells
automation that the requested expected-defect experiment completed and all of
its exact assertions held.  An unrelated exception or different wrong value
still exits nonzero.

### Run from the pinned tinygrad checkout

```bash
export TINYGRAD_DOCS=/absolute/path/to/tinygrad_docs

DEV=PYTHON \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase5/debugging_walk.py" \
  --mode control

DEV=CPU:CLANG \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase5/debugging_walk.py" \
  --mode injected

DEV=CPU:CLANG \
  .venv/bin/python "$TINYGRAD_DOCS/labs/phase5/debugging_walk.py" \
  --mode fixed
```

The script controls:

```text
BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=1 DEBUG=0
DEBUG_RANGEIFY=0 DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1
NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 THREADS=1 TRACK_MATCH_STATS=0
VALIDATE_WITH_CPU=0 VIZ=0
```

`DEV` remains the explicit caller choice.  `DISALLOW_BROADCAST=0` keeps the
controlled scalar addition legal, while `TRACK_MATCH_STATS=0` prevents an
inherited tracker from writing rewrite data despite `VIZ=0`.  Every Tensor
input uses `dtypes.float32`.  The script fails immediately under `python -O`
because the experiment relies on assertions.

### Portable control output

```text
controlled env: BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=1 DEBUG=0 DEBUG_RANGEIFY=0 DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1 NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 THREADS=1 TRACK_MATCH_STATS=0 VALIDATE_WITH_CPU=0 VIZ=0
mode: control-python
frontend/schedule ADD preserved: True True
compiled body: PROGRAM
result/oracle: [5.0, 6.0, 7.0] [5.0, 6.0, 7.0]
status: control-passed
```

This proves the public arithmetic oracle through the pinned Python route.  It
does not exercise the CPU renderer fault surface.

### Injected reproduction output

```text
controlled env: BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=1 DEBUG=0 DEBUG_RANGEIFY=0 DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1 NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 THREADS=1 TRACK_MATCH_STATS=0 VALIDATE_WITH_CPU=0 VIZ=0
mode: injected-cpu-renderer
frontend/schedule ADD preserved: True True
lowered SINK/LINEAR equal: True True
correct/faulty SOURCE store operator: ADD SUB
BINARY changed downstream of SOURCE: True
last good artifact: LINEAR
first bad artifact: SOURCE
result/oracle: [-3.0, -2.0, -1.0] [5.0, 6.0, 7.0]
status: expected-defect-reproduced
```

Interpret it in order:

1. public oracle is `[5,6,7]`;
2. frontend and scheduled kernel preserve `ADD`;
3. correct and faulty lowered `SINK`/program `LINEAR` are the same objects;
4. the standard source's final output store uses `ADD`;
5. the local renderer's final output store uses `SUB`;
6. binary changes after that source change; and
7. executing the faulty program yields the exact subtraction symptom.

`LINEAR` is last good; `SOURCE` is first bad; the renderer's operation-to-text
mapping owns the transition.  Runtime result evidence confirms the downstream
effect but is not needed to guess ownership.

### Fixed regression output

```text
controlled env: BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=1 DEBUG=0 DEBUG_RANGEIFY=0 DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1 NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 THREADS=1 TRACK_MATCH_STATS=0 VALIDATE_WITH_CPU=0 VIZ=0
mode: fixed-cpu-regression
standard renderer store operator: ADD
fault removed from executed route: True
result/oracle: [5.0, 6.0, 7.0] [5.0, 6.0, 7.0]
status: regression-passed
```

The standard renderer is the controlled “fix.”  The value assertion is the
regression.  Source operator classification is localization evidence, not a
recommended whitespace/name golden test.

### Predict before modifying the lab

Try these safe extensions:

1. Change input values but retain float32 and predict both correct and faulty
   results before running.
2. Change the expression to multiplication.  The local ADD fault should not
   affect its source or value.
3. Change the input dtype to `int32`.  The narrow fault delegates integer ADD
   to the original mapping, so the result should remain correct.
4. Add a second float32 ADD.  Decide whether one static mapping fault affects
   one or two source sites and how to avoid exact temporary names.
5. Add a `SOURCE`-only mode that does not execute the faulty binary; state which
   evidence becomes weaker.

Do not broaden the mapping to all subtraction or all ALU operations.  A fault
that corrupts index arithmetic can cause native out-of-bounds access instead of
the intended safe wrong value.

## Decision tree: wrong numerical value

```text
wrong value
  ├─ Is expected value/dtype/shape independently defined?
  │    no → build oracle first
  ├─ Does failure reproduce with explicit input/dtype in fresh process?
  │    no → characterize seed/cache/timing/state
  ├─ Is frontend graph correct?
  │    no → frontend/promotion/movement/autograd construction
  ├─ Is scheduled SINK correct?
  │    no → schedule/fusion/materialization/dependency
  ├─ Is lowered SINK/LINEAR correct for target?
  │    no → target lowering/index/gate/optimization
  ├─ Is SOURCE semantically correct?
  │    no → renderer
  ├─ Does compiler accept and preserve it?
  │    no → compiler/toolchain/artifact
  ├─ Are runtime arguments/dimensions/order/completion correct?
  │    no → runtime/ABI/queue/lifetime
  └─ Ordinary correct, replay wrong?
       → JIT decision tree
```

Always check dtype, shape, tolerance, NaN/Inf, signed zero, and overflow before
interpreting a numeric delta as code generation.

## Decision tree: Python exception or assertion

```text
exception
  ├─ Record exact type/message and complete relevant traceback
  ├─ Find newest frame, then first producer/consumer boundary frame
  ├─ Is it an expected contract rejection?
  │    yes → minimize the illegal input and verify the contract/source
  ├─ Did SPEC/CHECK_OOB move failure earlier?
  │    yes → inspect the first rejected UOp/index
  ├─ Does same artifact fail without higher validation?
  │    no → validation configuration may expose latent illegality
  └─ Break before producer; inspect small arguments and invariant with pdb
```

Do not catch `Exception` in the final reproducer.  If testing a rejection, use
the narrow exception type and a stable semantic assertion.  Exact full messages
and temporary node names are usually brittle unless they are the public
contract under test.

## Decision tree: render or compile failure

```text
compile-like failure
  ├─ Was lowered PROGRAM constructed?
  ├─ Was SOURCE constructed?
  │    no → renderer did not complete; inspect LINEAR → SOURCE
  ├─ Save exact SOURCE, target, renderer/compiler, tool version, stderr
  ├─ Is SOURCE invalid by target language/ABI?
  │    yes → renderer or earlier illegal input
  ├─ Does same compiler invocation reject minimal exact SOURCE?
  │    yes → tool/flags/target/support boundary
  ├─ Did BINARY exist but loader reject it?
  │    yes → artifact format/architecture/runtime loader
  └─ Did launch reject after load?
       → name/signature/arguments/dimensions/resources
```

Use `DEBUG=4` before `7`.  Disassembly cannot help when compilation never
produced a binary.

## Decision tree: native crash or GPU fault

```text
hard crash / device fault
  ├─ Stop broad workload and record first minimal command/exit/signal
  ├─ Can PYTHON reproduce the semantic failure safely?
  ├─ Can CPU reproduce without native/GPU fault?
  ├─ Enable PYTHONFAULTHANDLER for native Python process stacks
  ├─ Fresh process/context for each hardware attempt
  ├─ Save smallest SOURCE/BINARY metadata, arguments, dimensions, buffers
  ├─ Collect driver/vendor diagnostics without repeated submissions
  └─ Move to mock/dedicated hardware before low-level interface experiments
```

After a CUDA illegal access, later API errors in the same process can be
cascading context state.  Do not treat each as an independent symptom.

For the lower-level NVIDIA path, use the safety boundary from
[Chapter 14](14-nvidia.md#interface-safety-boundary): routine work should use
`DEV=NVK+NV` when that path is explicitly required.  Do not switch to
`PCI+NV`, unbind a driver, reset a production/display GPU, or escalate
privileges merely to continue debugging.  A known-faulting program should not
be submitted repeatedly on the desktop display GPU.

## Decision tree: race, synchronization, or lifetime

Indicators include nondeterminism, failure-rate sensitivity, success under
`DEBUG=2`, success after synchronization, stale copies, or corruption after
allocation reuse.

```text
race/lifetime suspicion
  ├─ Establish deterministic arithmetic on PYTHON
  ├─ Record producer, consumer, queues/devices, byte range
  ├─ Compare DEBUG=0 and explicit scoped waits; don't benchmark DEBUG=2
  ├─ Disable/characterize JIT graphing and allocator reuse separately
  ├─ Track base/view/staging ownership through completion
  ├─ Add one targeted event/dependency
  └─ Stress fresh buffers, changed inputs, repeated runs on hardware
```

Python object lifetime is not device completion.  Logical `mem_used` is not
physical allocator release.  Revisit Chapter 12 before changing resource
ownership.

## Decision tree: view, indexing, or dtype

```text
view/dtype symptom
  ├─ Record logical shape, dtype, itemsize, base, byte offset, byte extent
  ├─ Compare view.is_allocated and view.is_initialized
  ├─ Inspect frontend movement UOps and rangeify output
  ├─ Run SPEC=2 and applicable CHECK_OOB=1
  ├─ Compare contiguous copy only as a localization experiment
  ├─ Inspect lowered INDEX/gate/cast/bitcast
  └─ Verify runtime receives correct base/offset handle and dtype ABI
```

If `contiguous()` makes the bug disappear, it may have removed a view or
changed fusion, scheduling, and allocation.  Do not submit “force contiguous”
as a fix until the violated view/index contract is identified.

Dtype investigation must separate:

- weak versus concrete dtype;
- promotion at the frontend;
- casts versus bitcasts;
- vector dtype versus scalar dtype;
- storage item size and buffer byte extent;
- compiler ABI width/alignment; and
- numeric precision/overflow semantics.

## Decision tree: optimization-only failure

First reproduce under the original settings.  Then vary one control:

```text
original
  → NOOPT=1 only
  → restore, BEAM=0 only
  → restore, TC=0 only
  → vary one shape boundary
```

If `NOOPT=1` passes, that localizes the configuration branch; it does not prove
“the optimizer” generically.  Compare:

- scheduled kernel boundaries;
- target and renderer capability;
- applied options;
- pre/post named VIZ passes;
- lowered loads/stores/indices/gates;
- launch dimensions; and
- source and semantic oracle.

An optimization can expose a pre-existing renderer/runtime bug by generating a
form that the unoptimized path never used.  The first bad artifact still decides
ownership.

## Decision tree: JIT-only failure

```text
ordinary/JIT split
  ├─ Three+ calls with changing fresh inputs and oracle each call
  ├─ JIT=0
  │    fail → ordinary pipeline
  ├─ JIT=2
  │    fail → capture/parameterize/plan/prepared replay
  ├─ JIT=1
  │    fail only here → graph split/runner/update/order
  ├─ Compare captured versus graphed LINEAR call bodies
  ├─ Check pointer/scalar/symbolic dimension updates
  └─ Check returned-buffer reuse and written-input copies
```

Record ignore, capture, and first replay separately.  Reusing identical values
can hide a stale argument.  Retaining outputs can expose captured storage reuse;
decide whether that is the documented contract before calling it corruption.

## Safe GPU debugging discipline

1. Prove the public oracle on Python or CPU first when semantics permit.
2. Print the explicit device, renderer, target, driver, and physical GPU.
3. Use a small allocation and one kernel before a model.
4. Preserve exact launch dimensions, signature, and buffers.
5. Synchronize once diagnostically to classify completion sensitivity.
6. After a device fault, terminate and use a fresh process/context.
7. Do not loop a known-faulting kernel on the display GPU.
8. Prefer CUDA Driver API route for ordinary NVIDIA runtime work.
9. Require `NVK+NV` for routine lower-level NV study; never permit interface
   fallback to become an accidental PCI experiment.
10. Use mock/emulation or dedicated hardware for driver/queue packet work.

Vendor sanitizers, debuggers, and driver logs can strengthen hardware evidence,
but use the tool matching the selected API/artifact and keep the reproducer
minimal.  A tool timeout or unavailable optional dependency is not a passing
test.

## Use history only after current behavior is understood

### Find present ownership first

Use `rg` to find definitions, call sites, and tests:

```bash
rg -n 'def run_linear|def full_rewrite_to_sink|class _TinyJit' tinygrad test
rg -n 'CHECK_OOB|DEBUG_RANGEIFY|VALIDATE_WITH_CPU' tinygrad test
rg -n 'code_for_op.*Ops.ADD|Ops.ADD:' tinygrad/renderer test
```

Read the current producer, consumer, and closest test.  State the first bad
artifact before opening history.

### Then use log and blame as questions, not answers

```bash
git log --oneline --follow -- tinygrad/renderer/cstyle.py
git log -p -S 'VALIDATE_WITH_CPU' -- tinygrad test
git blame -L 63,66 tinygrad/renderer/cstyle.py
git log -L 63,66:tinygrad/renderer/cstyle.py
```

`blame` tells you which commit last touched lines, not why they are correct.
Read the complete introducing diff, commit message, linked discussion where
available, and added/removed tests.  A refactor that moved the line may not have
introduced the behavior.

### Bisect in a clean dedicated checkout

Do not bisect in a dirty working tree containing your documentation or fix.
Use a disposable clean clone/worktree, a deterministic reproducer, and stable
dependencies:

```bash
git bisect start BAD_COMMIT KNOWN_GOOD_COMMIT
git bisect run env DEV=CPU:CLANG JIT=0 SCACHE=0 CCACHE=0 CACHELEVEL=0 \
  .venv/bin/python -m pytest path/to/test_regression.py -q
git bisect reset
```

`git bisect run` interprets exit zero as good, exit 125 as untestable/skip, and
other ordinary nonzero statuses as bad.  Ensure the test's nonzero result means
the same semantic regression, not missing dependencies on older commits.

The bundled lab's injected mode intentionally exits zero when the artificial
defect reproduces, so it is not itself a good/bad bisect predicate.  A real
regression test should exit zero on correct behavior and nonzero on the bug.

A bisect identifies an introducing commit.  Adjacent artifacts identify the
broken contract.  You need both before deciding whether to repair, revert, or
update an expectation.

## Turn a reproducer into a regression test

### Red before green

1. Find the nearest existing test and run it on the bad revision.
2. If it already fails for the intended reason, improve/fix that path instead
   of adding a duplicate test.
3. If coverage is missing, add the smallest case encoding the missing domain or
   invariant.
4. Run it on the bad revision and record the expected failure.
5. Make the smallest justified implementation change.
6. Run the focused test and prove it turns green.
7. Run nearby tests and the backend/config matrix implied by the contract.
8. Run the repository-prescribed broader validation before review.

The regression should normally assert public semantics or a stable structural
contract.  Exact generated temporary names, whitespace, program counters,
timings, and full exception strings are incidental unless the public contract
explicitly promises them.

### Test the trigger, not the story

If the bug requires:

```text
float16 view + nonzero byte offset + tail mask + JIT replay
```

the regression must retain those conditions.  A generic float32 contiguous add
test does not cover it even if both stories say “wrong add.”

### The artificial lab fault is not an upstream issue

The lab creates a renderer subclass in the guide.  Upstream tinygrad is not
broken.  Do not report or contribute its expected subtraction result.  Use its
method—same lowered input, first changed source, exact oracle—for a real defect.

## Prepare a useful issue or handoff

Include:

```text
Summary: one sentence symptom, no unproved cause
Snapshot: commit, Python, OS, device/driver/toolchain
Command: exact fresh-process invocation
Reproducer: smallest self-contained code
Expected: value/shape/dtype/tolerance or structural invariant
Actual: exact value/exception/exit/fault
Frequency: deterministic or trials/failures
Route matrix: only controlled comparisons actually run
Artifact ledger: last good and first bad with concise evidence
Validation flags: DEBUG/VIZ/SPEC/OOB/JIT/cache settings
Safety: whether physical GPU faulted and recovery/fresh-process steps
Regression: existing test or proposed focused red case
```

Avoid:

- claiming a cause based only on the final traceback frame;
- attaching private data, model weights, secrets, or unneeded giant binaries;
- posting a full VIZ pickle when a small graph/source fragment suffices;
- saying “latest” without a commit;
- omitting dtype, layout, target, or call number;
- hiding an expected failure behind a broad catch; or
- continuing to submit a known-faulting GPU workload for more logs.

Before filing, recheck current upstream policy, existing issues/PRs, and whether
the bug still reproduces on the live commit.  Chapter 18 covers contribution
coordination.

## Question-led source stops

Open these only after the preceding model makes sense.  Each stop gives a
prediction and a bounded question so the source is meaningful in isolation.
Every link targets the recorded snapshot.

### Stop 1: Which controls are frozen at import?

Prediction: Linux defaults include `DEBUG=0`, `JIT=1`, `CACHELEVEL=2`,
`VALIDATE_WITH_CPU=0`, `SPEC=1`, `CHECK_OOB=0`, `CCACHE=1`, and `SCACHE=1`.
Read the bounded
[`ContextVar` definitions](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L232-L280).
Then read the small
[`getenv`, `Context`, and `ContextVar` implementation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L153-L190).

Question: which controls change semantics/plans, validation, instrumentation,
and caching?  Why must a shell environment comparison start a new process?

### Stop 2: What does the schedule cache hide?

Prediction: `SCACHE=0` recomputes the schedule, while `DEBUG=1` prints a schedule
summary only for multi-call results and `DEBUG=3` prints even one call.  Read
[`schedule cache lookup and debug summary`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L107-L131).

Question: what forms the cache key, where is `type_verify` run on a miss, and
which condition prints `CACHE MISS` versus `cache hit`?

### Stop 3: What exactly does `DEBUG_RANGEIFY` print?

Prediction: it changes the debug Boolean, not the selected rangeify algorithm.
Read
[`run_rangeify` setup, range propagation, debug row, and range renderer`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/indexing.py#L179-L323),
then the narrow
[`caller and VIZ checkpoints`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L554-L577).

Question: explain every printed column and the leading `***`.  Which named VIZ
checkpoints surround later rangeify/kernel-graph transitions?

### Stop 4: How do SPEC and OOB checking differ?

Prediction: `SPEC=2` strengthens UOp construction/boundary legality; OOB proof
is a particular index rule with explicit skips.  Read
[`per-UOp SPEC behavior`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L193-L221)
and
[`index validation and boundary type verification`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/spec.py#L8-L44).

Question: which checks require exactly `SPEC==2`, which require `SPEC>1`, when
does Z3 run, and which index forms are skipped?

### Stop 5: What does runtime `DEBUG=2` change?

Prediction: it asks runtimes to wait and can synchronize for timing.  Read
[`track_stats`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L51-L84)
and
[`run_linear` context construction](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L268-L281).

Question: when is `wait` forced true, and when does the fallback call
`Device.synchronize()`?  Why can this mask a race?

### Stop 6: Where is CPU validation inserted?

Prediction: validation rewrites raw `CALL(SINK,...)` before normal compilation,
then compares device and CPU shadow outputs.  Read
[`validation rewrite and compile/dispatch matchers`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L234-L281)
and the executor's
[`CPU shadow comparison`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L188-L197).

Question: why can a wrong scheduled `SINK` pass both executions, and why does a
prepared `jit=True` plan skip insertion?

### Stop 7: What do DEBUG 4, 5, and 7 expose?

Prediction: 5 prints the base AST before lowering, 4 prints source before
compile, and 7 calls optional disassembly after compile only on the
rendered-SOURCE/compiler branch.  Read the start of
[`full_rewrite_to_sink`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L284-L310)
and
[`linearize/render/compile transitions`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L409-L441).

Question: place each output before or after its corresponding transformation.
Which one is unavailable when compilation fails?

### Stop 8: How does VIZ capture and launch?

Prediction: nonzero VIZ supplies profiling/match-tracking defaults at import;
when explicit overrides leave those enabled, exit saves the paired profile and
rewrite data and launches a viewer or prints the CLI instruction.  Read
[`tracking structures and enable condition`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1482-L1515)
and
[`rewrite save plus viewer launch`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L1595-L1627),
the separate
[`profile finalization and save`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L389-L400),
then the CLI's bounded
[`DEBUG rendering and path arguments`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/cli.py#L203-L230).

Question: why must explicit `--rewrites-path` and `--profile-path` remain paired
instead of mixing one copied artifact with the latest default?  How does naming
one pass at low DEBUG differ from setting CLI `DEBUG=7`?

### Stop 9: What separates JIT 2 from JIT 1?

Prediction: both capture and replay; only settings below 2 enter graph splitting.
Read
[`jit_lower`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L67-L77)
and the full bounded
[`TinyJit ignore/capture/replay state machine`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L222-L289).

Question: which call ignores, which captures, when is graph splitting selected,
and why do changing inputs need at least a third call?

### Stop 10: What does compiler caching/disassembly promise?

Prediction: `CCACHE=0` removes the compiler cache key; the common disassembler
can be a no-op.  Read
[`Compiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L301-L312)
and
[`ClangCompiler`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/compiler_cpu.py#L6-L27).
Follow its loader to the bounded
[`jit_loader` image/relocation implementation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/support/elf.py#L52-L82).
Follow only the disassembly call to the bounded
[`Capstone helper`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/helpers.py#L512-L524).

Question: which source string keys compilation, which command produces the CPU
object, what does linking return, and which optional tool backs CPU disassembly?

### Stop 11: Why is the lab's first bad artifact SOURCE?

Prediction: the generic C-style ALU rule delegates an `ADD` UOp to the selected
renderer mapping, after LINEAR already exists.  Read
[`C-style STORE and ALU rendering rules`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L57-L66)
and the base
[`ADD`/`SUB` source spellings](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L137-L145),
then
[`ClangRenderer` operation table and compiler construction](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/renderer/cstyle.py#L256-L304).

Question: why can a copied mapping change only source spelling while scheduled
and lowered UOps remain `ADD`?  Which dtype restriction keeps address arithmetic
outside the fault?

### Stop 12: Where is the frontend-to-execution boundary?

Prediction: `linear_with_vars` transforms the Tensor sink and resolves buffers;
`realize` passes the resulting LINEAR to `run_linear`.  Read the complete
[`Tensor.linear_with_vars` and `Tensor.realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L175-L196).

Question: where does weak dtype rejection occur, where are Tensor UOps replaced
with bufferized forms, and which call begins ordinary compile/dispatch?

## Background ladders

Use only the level blocking the current investigation.

### Level 0: Python exceptions and `pdb`

Read the official
[`pdb` documentation](https://docs.python.org/3/library/pdb.html) and
[`faulthandler` documentation](https://docs.python.org/3/library/faulthandler.html).
Stop when you can:

- identify exception type/message and relevant frames;
- use `where`, `up`, `down`, `args`, `p`, `next`, `step`, and `continue`;
- break immediately before a producer/consumer boundary; and
- explain why a fatal native signal differs from a Python exception.

### Level 1: numeric oracles

Learn exact versus approximate equality, `rtol`/`atol`, dtype precision,
overflow, NaN/Inf, and signed zero.  Stop when you can justify the expected
result and tolerance for the smallest reproducer without referring to tinygrad's
actual output.

### Level 2: compiler artifact model

Review [UOps](05-uops.md), [rewrites](06-rewrites.md),
[scheduling](07-scheduling.md), [lowering](10-lowering.md), and
[rendering](11-rendering.md).  Stop when you can name the object on both sides
of the suspected transition and write its invariant.

### Level 3: GPU execution and synchronization

Use the execution model and Driver API routes in
[Learning resources](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
and
[Learning resources](../reference/learning-resources.md#nvidia-code-generation-and-runtime-work).
Stop when you can distinguish host submission, queue order, event completion,
device synchronization, and host/device memory lifetime.

### Level 4: tests and history

Review [Testing](16-testing.md) and the official
[`git log`](https://git-scm.com/docs/git-log),
[`git blame`](https://git-scm.com/docs/git-blame), and
[`git bisect`](https://git-scm.com/docs/git-bisect) documentation.  Stop when
you can write a deterministic good/bad predicate and explain why history follows
current artifact localization rather than replacing it.

## Common misconceptions, corrected

| Misconception | Correction |
| --- | --- |
| The final exception frame is the root cause. | It is where a contract was detected; inspect the producing caller and first bad artifact. |
| A second backend is an independent oracle. | It can share frontend, scheduling, lowering, or even renderer components. |
| `DEBUG=2` only prints more. | It forces wait behavior and can synchronize for timing. |
| `DEBUG=6` automatically prints all rewrites. | All-graph/rewrite meanings belong to the VIZ CLI over captured data. |
| `SPEC=2` proves numerical correctness. | It checks structural legality/dtypes; retain an external oracle. |
| `CHECK_OOB=1` is a runtime memory sanitizer. | It statically validates supported modeled index forms and has explicit skips. |
| CPU validation proves scheduling correct. | CPU and device execute the same scheduled `SINK`; both can agree on a shared error. |
| `JIT=2` disables TinyJit. | It captures/replays but skips graph splitting; `JIT=0` disables capture. |
| A full synchronize fixing the bug is the final patch. | It localizes completion sensitivity; find the narrow dependency/lifetime rule. |
| `NOOPT=1` passing proves an optimizer bug. | It identifies a path difference; the first bad artifact can still be renderer/runtime. |
| `mem_used` returning to zero proves physical memory was freed. | It is logical active-buffer accounting; allocator/driver retention is separate. |
| Disassembly is always available at `DEBUG=7`. | Only the rendered-SOURCE compile branch calls the hook; direct ISA assembly bypasses it, and the selected compiler may no-op or require optional tools. |
| Clearing every cache is good isolation. | It proves state mattered but hides which cache; isolate scopes one at a time. |
| A broad `except Exception` makes an expected-failure test robust. | It turns unrelated failures into false success. |
| A bisect identifies the broken invariant. | It identifies an introducing commit; artifact comparison identifies the contract. |

## Exercises

Try each before opening its answer.

### 1. Symptom or cause?

Classify: “call three returns call two's values” and “graph pointer update is
broken.”

??? answer
    The first is a symptom: it states observable call behavior.  The second is
    a causal hypothesis.  It becomes justified only after ordinary/captured
    artifacts and pointer-update evidence localize the first violation there.

### 2. Build an independent oracle

For `x=[1,2,3]`, `out=x*x+2*x`, state an oracle without executing tinygrad.

??? answer
    Elementwise arithmetic gives `[1+2, 4+4, 9+6] = [3,8,15]`.  Also state the
    intended concrete dtype and shape `(3,)`; for float32 these small integers
    are exactly representable.

### 3. Identify the first bad artifact

Frontend, scheduled SINK, and lowered LINEAR contain float32 `ADD`.  Generated
source stores `a-b`; binary and result implement subtraction.  What is first
bad, and who owns it?

??? answer
    `SOURCE` is first bad.  The renderer transition from ordered lowered UOps to
    source owns the violation.  Binary and result are downstream evidence.

### 4. Preserve the bug while minimizing

A view failure disappears after adding `contiguous()`.  Is the contiguous
version a valid minimal reproducer?

??? answer
    Not yet.  `contiguous()` creates a new storage/copy boundary and can remove
    the faulty view/index path.  Keep the view while reducing shape and
    operations, and verify the same first bad artifact.

### 5. Interpret `DEBUG=2`

Why can `DEBUG=0` fail while `DEBUG=2` passes, and what should you do next?

??? answer
    Level two requests waiting and can synchronize when timing a runtime that
    returned no duration.  Suspect completion, queue order, or lifetime.  Name
    producer/consumer and test a scoped dependency; do not keep the global
    debug-induced wait as the fix.

### 6. Choose VIZ levels

You know the suspect schedule event but not the pass.  Which CLI progression is
appropriate?

??? answer
    Use `--ls` to list named passes, `DEBUG=6` to inspect captured graph states,
    then use low `DEBUG` with the copied event and one copied pass name to
    reconstruct only that pass.  `DEBUG=7` reconstructs every step and is the
    intentionally noisy fallback.  Keep the copied rewrite/profile paths
    paired in every command.

### 7. Separate SPEC and OOB

What does `SPEC=2 CHECK_OOB=1` add, and what does it still not prove?

??? answer
    SPEC strengthens per-UOp/boundary legality and dtype checks.  OOB validates
    supported modeled indices, using interval proof or Z3.  It skips documented
    forms and does not prove runtime pointer ownership, physical allocation, or
    numeric correctness.

### 8. Interpret CPU validation agreement

Public result is wrong, but `VALIDATE_WITH_CPU=1` reports device and CPU agree.
Where do you go?

??? answer
    Return before their branch: verify the public oracle, frontend, scheduled
    SINK, copies, and output interpretation.  Both routes execute the same
    scheduled kernel, so agreement cannot validate shared semantics.

### 9. Interpret CPU validation disagreement

The validation route reports a device/CPU mismatch for one scheduled kernel.
What is the next minimal comparison?

??? answer
    Preserve that one SINK, then compare target-lowered programs, source,
    signature, arguments, dimensions, and completion for CPU and device.  Keep
    a public result oracle because validation itself adds copies and waits.

### 10. Isolate JIT

`JIT=0` passes, `JIT=2` fails, and `JIT=1` also fails.  Can you blame graph
batching?

??? answer
    No.  JIT=2 skips graph splitting and already fails, so investigate capture,
    parameter substitution, memory planning, compiled prepared calls, replay
    argument updates, and lifetimes before graph-specific code.

### 11. Isolate graphing

`JIT=0` and `JIT=2` pass; `JIT=1` fails on the third call.  What must you prove
before inspecting a graph runner?

??? answer
    Prove the backend actually grouped the calls into a custom graph call by
    inspecting captured versus graphed LINEAR or VIZ checkpoints.  Then inspect
    graph eligibility, node creation/update, arguments, dimensions, and order.

### 12. Localize a compiler error

`DEBUG=4` shows malformed C before Clang rejects it.  Is Clang the first bad
artifact?

??? answer
    No.  The malformed `SOURCE` already violates the target language/ABI.
    Inspect the renderer or the illegal lowered input that produced it.  The
    compiler correctly detected a downstream error.

### 13. Use disassembly appropriately

When is `DEBUG=7` disassembly useful, and why might it print nothing useful?

??? answer
    Use it after source is correct and compilation succeeds, when the claim
    concerns generated machine instructions on a rendered-SOURCE/compiler
    path.  The direct ISA assembly branch never calls the hook; on eligible
    paths the common method is a no-op and backend implementations may require
    optional tools such as Capstone.

### 14. Diagnose cache sensitivity

Fresh `SCACHE=0 CCACHE=0 CACHELEVEL=0` passes, while a normal cached run fails.
What has been proved?

??? answer
    Some removed state/cache affects the symptom.  It has not identified which
    cache.  Use fresh processes and restore schedule, compiler, and disk caching
    one at a time; also consider process-only program/runtime/device caches.

### 15. Handle an expected failure in automation

Why does the lab's injected mode exit zero, and what prevents false success?

??? answer
    The requested outcome is to reproduce one exact artificial symptom, so
    that experiment is successful when its exact structural and numeric
    assertions hold.  It catches nothing broadly; a different value,
    exception, missing compiler, or absent boundary change exits nonzero.

### 16. Respond to a GPU illegal access

What should happen after the first minimized illegal access on a display GPU?

??? answer
    Stop broad/repeated submission, save the smallest artifact/arguments and
    driver diagnostic, terminate the process, and use a fresh context for any
    justified next run.  Prefer Python/CPU/mock or dedicated hardware while
    localizing.  Do not switch to direct PCI or reset/unbind the production GPU
    casually.

### 17. Use history correctly

`git blame` points at a renderer refactor.  What evidence is needed before
calling it the regression?

??? answer
    Reproduce current behavior, locate first bad artifact/owner, read the full
    diff and tests, and bisect with a deterministic semantic regression if the
    interval remains unknown.  Blame identifies the last line change, not the
    causal contract violation.

### 18. Design a regression

A failure requires a float16 view at nonzero offset during third-call JIT
replay.  Which properties must the focused test retain?

??? answer
    Retain float16, the view/base and byte offset, the relevant shape/range,
    at least ignore/capture/replay calls with changing fresh inputs, and the
    public oracle.  A contiguous float32 ordinary add test does not cover the
    trigger.

## Checkpoint

Continue to the testing chapter when you can:

- write a concrete symptom without embedding an unproved cause;
- state an independent numeric oracle or structural invariant;
- minimize a reproducer while preserving the same first bad artifact;
- freeze commit, process, environment, dtype, device, optimizer, JIT, seeds,
  and cache state;
- read a Python traceback and navigate to a producer/consumer boundary in
  `pdb`;
- distinguish outer execution LINEAR from one program's ordered LINEAR;
- compare every adjacent artifact from frontend through runtime;
- explain exact pinned runtime DEBUG levels separately from VIZ CLI levels;
- capture VIZ data and keep the explicit rewrite/profile pickle paths paired;
- explain SPEC, CHECK_OOB, DEBUG_RANGEIFY, and CPU validation limits;
- isolate schedule, compiler, disk, program, runtime, and device caches;
- classify a compiler, loader, launch, race, view/dtype, optimization, or JIT
  failure with the matching decision tree;
- explain why `JIT=2` still captures but skips graph splitting;
- use synchronization as a diagnostic without making a global wait the fix;
- stop safely after a physical GPU fault;
- use history after locating current ownership; and
- prove a focused test red before the change and green afterward.

If you cannot name the artifacts, review Chapters 10–13.  If queue completion
is unclear, review Chapters 12 and 14 plus Background Level 3.  If the test
asserts only log text or an incidental generated name, revise it before Chapter
16.

## Quick reference

```text
symptom ≠ cause
oracle/invariant first

reproduce → freeze process/config → minimize without path change
          → compare adjacent artifacts
          → last good / first bad
          → current owner → focused red/green test → wider validation

ordinary artifacts:
  public expression
  → frontend UOps
  → outer LINEAR CALLs / scheduled SINK
  → lowered SINK
  → program LINEAR
  → SOURCE
  → BINARY
  → runtime args/dimensions/order/completion
  → result

JIT isolation:
  JIT=0  ordinary Python every call
  JIT=2  capture/replay, no graph splitting
  JIT=1  graph splitting when backend/batch supports it

validation:
  SPEC=2          structural legality/dtype, not values
  CHECK_OOB=1     supported modeled index bounds, not runtime sanitizer
  DEBUG_RANGEIFY  scheduling explanation, not correctness
  VALIDATE_WITH_CPU same scheduled SINK on CPU/device, not frontend oracle

debug:
  runtime DEBUG=2 changes waiting
  runtime DEBUG=4 prints source
  runtime DEBUG=5 prints base kernel AST
  runtime DEBUG=7 attempts disassembly on rendered-SOURCE compile paths
  VIZ CLI 5/6/7 lists passes / graphs / individual rewrites

safe GPU rule:
  minimize → record first fault → stop repeated submission → fresh process
```
