# Command quick reference

Commands assume the tinygrad repository root and an editable installation. Set
`DEV` explicitly in investigations so logs and results identify the path used.

## Record the environment

```bash
git rev-parse HEAD
python3 --version
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
DEV=CUDA DEBUG=0 .venv/bin/python - <<'PY'
import os
from tinygrad import Device

requested, canonical = os.environ["DEV"], Device.DEFAULT
backend = Device[canonical]
renderer = backend.renderer
print("requested / canonical:", requested, canonical)
print("backend / interface:", type(backend).__name__, type(backend.iface).__name__ if hasattr(backend, "iface") else "none")
print("renderer / compiler:", type(renderer).__name__, type(renderer.compiler).__name__)
print("runtime:", getattr(getattr(backend, "runtime_t", None), "__name__", "none"))
print("target:", renderer.target)
PY
```

Repeat the last command with `DEV=NVK+NV` if investigating tinygrad's lower-level
NVIDIA backend. An initialization failure is useful evidence; do not hide it by
falling back to automatic device selection.

## Device and renderer routes

| Setting | Use |
| --- | --- |
| `DEV=PYTHON` | Execute lowered UOps with the readable Python interpreter. This is a useful control for lowered-program semantics, but it shares frontend, scheduling, and lowering with other routes and is not an independent oracle for those stages. |
| `DEV=NULL` | Exercise scheduling/codegen without meaningful numerical execution. |
| `DEV=CPU` | Compiled execution through the host CPU path. Record the concrete renderer/compiler selected on the host. |
| `DEV=CUDA` | NVIDIA execution through the CUDA Driver API and default renderer selection. |
| `DEV=CUDA:PTX` | CUDA runtime with the direct PTX renderer. |
| `DEV=NVK+NV` | NVIDIA execution through tinygrad's HCQ path, requiring the ordinary NVIDIA kernel-driver interface. |
| `DEV=NVK+NV:PTX` | The same explicit NV kernel-driver interface with the direct PTX renderer. |
| `DEV=PYTHON::sm_89` | Hardware-free Ada-targeted lowering executed by Python. It can check modeled structure and small numerical results, not NVIDIA compilation, launch behavior, physical tensor-core use, or timing. |

The complete recorded-snapshot grammar is
`interface[:indices]+device[:renderer[:architecture]]`; the entire interface
part and individual colon-separated components can be absent. Thus
`PYTHON::sm_89` leaves the renderer empty, while `NVK:0+NV` selects interface
index zero. Target syntax is snapshot-sensitive. Confirm it in `Target.parse`,
the selected runtime, and current CI examples before publishing a command.

Bare `DEV=NV` allows interface fallback in this snapshot. Do not use it when a
course command is meant to guarantee the kernel-driver route. `DEV=PCI+NV` is a
specialized direct-PCI path for dedicated driver work, not a routine fallback.
The bundled performance lab rejects both forms because neither establishes its
required explicit driver-backed route.

## Debug output

| Level | Adds in the recorded snapshot |
| ---: | --- |
| `DEBUG=1` | Opened devices, multi-call schedule summaries, JIT capture/prune summaries, and selected backend notices. A one-call schedule need not print a schedule summary. |
| `DEBUG=2` | Per-call execution statistics, static operation/memory estimates, and JIT graph-batch notices. `run_linear` requests waiting, and statistics synchronize when a runtime supplies no duration. Calls can be copies, views, graphs, or other bodies—not only physical kernels. |
| `DEBUG=3` | Schedule summaries even for one call, applied optimization information when present, and selected library/tool initialization details. It does not print the base kernel AST. |
| `DEBUG=4` | Generated `SOURCE` or printable ISA form before compilation, plus renderer/optimizer-specific detail. |
| `DEBUG=5` | `pyrender` of the base kernel AST before lowering; some backends add detail such as pretty-printed loaded PTX. It is not a universal pass trace. |
| `DEBUG=6` | No new universal core execution stream; some backends add detail. In the VIZ CLI only, level 6 renders all captured UOp graphs. |
| `DEBUG=7` | On rendered-SOURCE paths, request the selected compiler's optional disassembly after successful compilation; also log buffer allocation/deallocation. Direct `ISARenderer` assembly bypasses this compiler hook. In the VIZ CLI only, level 7 reconstructs individual rewrites. |
| `DEBUG=8` | SQLite trace output and a few backend-specific diagnostics. |

Most runtime checks are cumulative and can produce enormous output. Begin at 2
or 3, minimize the workload, and capture output to a file when moving higher.
Runtime DEBUG and VIZ CLI DEBUG are separate consumers: the graph/rewrite
meanings at levels 6 and 7 apply only when the CLI reads captured VIZ data.
Inspect the guards in the exact renderer/compiler/backend under study.

Examples:

```bash
DEV=CPU DEBUG=3 .venv/bin/python your_reproducer.py
DEV=CUDA DEBUG=4 .venv/bin/python your_reproducer.py
DEV=CUDA DEBUG=2 .venv/bin/python your_benchmark.py
```

`DEBUG=2` changes timing behavior by waiting for execution and may remove normal
asynchronous overlap. It is useful for attribution, but the final benchmark
needs `DEBUG=0`, an explicit correctness oracle, and synchronization around the
boundary actually claimed.

## Rewrite and profile visualization

```bash
VIZ=-1 PROFILE=1 TRACK_MATCH_STATS=2 DEV=CPU DEBUG=0 \
  SCACHE=0 CCACHE=0 CACHELEVEL=0 \
  .venv/bin/python your_reproducer.py

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 DEBUG=0 \
  .venv/bin/python - <<'PY'
from tinygrad.helpers import temp
print("rewrites:", temp("rewrites.pkl", append_user=True))
print("profile:", temp("profile.pkl", append_user=True))
PY
```

`VIZ=-1` enables capture without replacing an interactive process with the web
viewer. The explicit `PROFILE` and `TRACK_MATCH_STATS` values win even if the
shell inherited hostile overrides. `SCACHE=0` ensures the scheduler work of
interest is not hidden by an in-process hit; the other cache settings make this
a fresh diagnostic capture rather than a performance run.

The rewrite saver prints its path. Resolve both per-user defaults as above,
then copy both files to uniquely named evidence paths before another VIZ run.
Never combine one copied artifact with the latest default for the other. Analyze
the explicit pair:

```bash
rewrite_capture=/exact/copied/rewrites.pkl
profile_capture=/exact/copied/profile.pkl

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" --list

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" \
  -s TINY | rg 'Schedule|Kernel'

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" \
  -s TINY 'copied event name' --list

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=6 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" \
  -s TINY 'copied event name'

VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" \
  -s TINY 'copied event name' 'copied pass name'
```

Event names can contain per-process counters, so copy the actual event and pass
names from the listings. Supplying one pass name reconstructs that pass even at
`DEBUG=0`. CLI `DEBUG=7` instead reconstructs every individual match in the
selected event; adding a pass name does not narrow that behavior and can emit
thousands of lines.

For machine processing, keep the same explicit pair:

```bash
VIZ=0 PROFILE=0 TRACK_MATCH_STATS=0 CAPTURE_PROCESS_REPLAY=0 \
  DEBUG=0 NO_COLOR=1 .venv/bin/python -m tinygrad.viz.cli \
  --rewrites-path "$rewrite_capture" --profile-path "$profile_capture" \
  --json > /tmp/tinygrad-events.jsonl
```

See the snapshot-pinned
[`tinygrad/viz/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/viz/README.md)
for filters, intervals, markers, and backend-specific profiling.

## Focused validation modes

| Setting | Question it helps answer |
| --- | --- |
| `SPEC=2` | Do the applicable per-UOp and boundary checks satisfy the recorded specification? Constructor-side checks have explicit exceptions, and this is not a numerical oracle. |
| `CHECK_OOB=1` | Can the supported modeled index/gate forms be proven in bounds? This needs the snapshot's supported Z3 dependency and is not a runtime memory sanitizer. |
| `JIT=0 VALIDATE_WITH_CPU=1` | Does a device execution agree with a CPU rendering of the same scheduled `SINK` on the ordinary compile path? Shared frontend/scheduler errors can agree, and prepared `jit=True` plans skip insertion. |
| `DEBUG_RANGEIFY=1` | Why did range inheritance/materialization make this scheduling decision? |
| `NOOPT=1` | Does the problem disappear without kernel optimization? |
| `BEAM=0` / `BEAM=2` | Does heuristic versus measured kernel scheduling change the outcome? |
| `SCACHE=0` | Recompute scheduling instead of reading or writing the process-local schedule cache. Existing entries remain in memory. |
| `CCACHE=0` | Construct compilers without their persistent compiler-cache key. Use a fresh process because a compiler object retains the choice made at construction. |
| `CACHELEVEL=0` | Bypass universal disk-cache get/put operations. A supplied `CACHEDB` path is inert for those operations at this level; direct database connections and explicit cache clearing are separate. |
| `CACHEDB=/absolute/path.db` | Select and isolate the SQLite database; this chooses a location but does not disable caching. |
| `CAPTURE_PROCESS_REPLAY=1` | Capture selected kernel-generation inputs for cross-revision replay. Set it before import and retain one stable `CACHEDB` with `CACHELEVEL>=1`. |

Change one variable at a time and retain the correctness oracle. A flag that
makes a failure disappear narrows the stage; it does not prove the disabled
feature owns the root cause.

Some controls can also be scoped inside Python when the relevant code reads
them while the context is active:

```python
from tinygrad import Context

with Context(NOOPT=1, BEAM=0):
  result = workload()
```

Environment values are read when their `ContextVar` is created, usually during
import. Import-time setup makes shell variables and a fresh process mandatory
for several important experiments: VIZ profiling/rewrite hooks and process-
replay capture are installed conditionally, while compiler objects retain their
`CCACHE` choice. A later `Context(VIZ=1)` is therefore not equivalent to a
capture process launched with VIZ enabled.

## Test routing

| Area | Typical location |
| --- | --- |
| Backend-neutral UOps, symbolic algebra, schedules | `test/null/` |
| Same operation semantics across runtimes | `test/backend/` |
| Tests run on one backend in CI | `test/unit/` |
| Kernel optimization and tensor cores | `test/opt/` |
| Allocators, queues, runtimes, hardware behavior | `test/device/` |
| Mocked/emulated driver and device support | `test/mockgpu/` is primarily infrastructure; run the test that consumes the required `MOCK...` route. |
| Model behavior | `test/models/` plus model-specific cases under `test/external/` |
| Performance regression tests | `test/speed/`; benchmark and diagnostic scripts also live under `test/external/` |
| Fuzzers, process replay, and large integrations | `test/external/` |

Start with the exact regression and its nearest existing file:

```bash
DEV=CPU DEBUG=0 .venv/bin/python -m pytest \
  test/unit/test_jit.py::TestJit::test_jit_zero_does_not_jit -x -q
SPEC=2 DEV=NULL DEBUG=0 .venv/bin/python -m pytest \
  test/null/test_schedule.py::TestSimpleSchedule::test_reduce_doesnt_split -x -q
DEV=CUDA DEBUG=0 .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -x -q
```

At the recorded snapshot, upstream asks agents to use twelve workers when
`pytest-xdist` is installed:

```bash
DEV=CPU DEBUG=0 .venv/bin/python -m pytest test/unit/ -x -q -n12
```

Do not oversubscribe the local machine merely to copy policy, and rerun a
suspected order/concurrency failure serially. Broaden only after the focused
test fails for the intended reason before the fix and passes after it. Consult
`.github/workflows/test.yml` for the current CI matrix rather than assuming a
directory runs on every backend.

## Static checks

After installing the current checkout's `linting` extra, invoke every tool from
the same explicit environment:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy tinygrad/
PATH="$PWD/.venv/bin:$PATH" .venv/bin/pre-commit run --all-files
```

Use versions from the checkout's `pyproject.toml` extras. Formatting unrelated
code or fixing unrelated warnings makes a contribution harder to review.

## Find ownership and history

```bash
rg -n 'def create_linear_with_vars|class PatternMatcher' tinygrad test
rg -n 'Ops\.AFTER|DEBUG_RANGEIFY' tinygrad test
git log --oneline --follow -- tinygrad/schedule/rangeify.py
git blame -L 555,580 tinygrad/schedule/rangeify.py
git log -S 'PatternMatcher' --oneline --all -- tinygrad
```

Use `rg` to find definitions, callers, and tests before browsing directories.
Use history to recover intent after you understand the current behavior, not as
a replacement for reproducing it.

## Process replay outline

For a refactor or speed change, follow the exact current upstream instructions.
At this snapshot the local shape is:

1. choose an isolated absolute `CACHEDB` path and keep `CACHELEVEL>=1`;
2. on the change branch, start a fresh process with
   `CAPTURE_PROCESS_REPLAY=1` and run representative focused/broad tests;
3. preserve the change branch's `test/external/process_replay/process_replay.py`,
   then use a separate clean checkout at the comparison revision, pointing it at
   the same database; and
4. run that preserved candidate script with the comparison checkout's tinygrad,
   classify every relevant generated-source difference, and retain ordinary
   correctness/performance evidence.

For example, after creating the database directory outside either checkout:

```bash
# change checkout: preserve the replay implementation used for this comparison
cp test/external/process_replay/process_replay.py \
  /absolute/replay-evidence/process_replay.py

# change checkout: capture selected kernel-generation inputs
CACHEDB=/absolute/replay-evidence/cache.db CACHELEVEL=1 \
  CAPTURE_PROCESS_REPLAY=1 DEV=CPU DEBUG=0 \
  .venv/bin/python -m pytest \
  test/backend/test_ops.py::TestOps::test_add -x -q

# clean comparison checkout: inspect diffs without asserting them
CACHEDB=/absolute/replay-evidence/cache.db CACHELEVEL=1 \
  CAPTURE_PROCESS_REPLAY=0 ASSERT_PROCESS_REPLAY=0 DEV=CPU DEBUG=0 \
  .venv/bin/python /absolute/replay-evidence/process_replay.py
```

This mirrors the pinned composite action: it copies the candidate replay script
before checking out `origin/master`, then runs that copy against the comparison
code. If you instead invoke the script inside the comparison checkout, first
prove that it is identical to the candidate script; otherwise script changes are
confounded with generated-code changes.

The lowercase/uppercase marker behavior is unusually subtle in this commit.
The README and workflow contain lowercase `[pr]`, and GitHub Actions'
[`contains()`](https://docs.github.com/en/actions/reference/workflows-and-actions/expressions#contains)
is case-insensitive, so either `[pr]` or `[PR]` in a PR title enables the
capture/action condition. The Python replay script uses a case-sensitive
uppercase `[PR]` test for `ASSERT_DIFF`: lowercase alone does not enable
assertion, uppercase in the exported title does, and uppercase only in the
exported commit message can also enable it after the action runs. A direct local
invocation with both variables absent receives the script's uppercase default
and begins in assertion mode unless `ASSERT_PROCESS_REPLAY=0` is explicit.

The README describes an early stop after more than 20% of kernels change, but
the pinned loop does not compute a corpus-wide percentage: its `changed` value
is a raw per-page exception count, and assertion mode can abort on a promoted
warning before that threshold. Treat the README phrase as intended shorthand,
not a measured global percentage in this implementation.

The authoritative details are in
[`test/external/process_replay/README.md`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/README.md).
Before real contribution work, reopen the
[live upstream instructions](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) <!-- live-upstream -->
because the marker and replay contract can change.
Also inspect the pinned
[`workflow condition`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L1-L7),
[`composite action`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/actions/process-replay/action.yml#L1-L16),
and
[`replay implementation`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/external/process_replay/process_replay.py#L1-L128).
Do not switch revisions in a dirty working tree. Captures are tied to the
snapshot's serialization/cache schema, and unchanged generated source is not a
runtime or numerical proof.
