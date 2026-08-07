# 2. Development setup: make every observation reproducible

## The promise of this chapter

Compiler work produces a great deal of output: graphs, generated programs,
cache hits, timings, backend errors, and test failures. None of it is useful if
you cannot say which source tree, Python interpreter, device path, and command
produced it.

This chapter builds that foundation. It assumes ordinary terminal use, but it
does not assume that you already understand Git's detached-`HEAD` state,
editable Python installs, tinygrad backends, shell environment variables, or
GPU-toolkit layering. Each is explained before it appears in a command.

Ubuntu with an RTX 4090 is the primary hardware route. The required path starts
with `PYTHON` and `CPU`, however, so a driver problem cannot prevent you from
learning the compiler.

By the end, you will be able to:

- distinguish this guide's repository from a tinygrad study checkout and a
  contribution checkout;
- prove which tinygrad commit and Python package a command imported;
- select `PYTHON`, `CPU`, `CUDA`, or `NVK+NV` deliberately;
- separate package, compiler, driver, runtime, and hardware failures;
- run one reproducer and one focused test through a backend ladder;
- run the bundled guide labs from the correct directory; and
- record an environment card that makes a later observation reproducible.

## First orient yourself in the filesystem

A shell command runs relative to the shell's **current working directory**.
That small fact explains many apparent Python problems.

The prompt often shows the current directory. For example:

```text
abel@computer:~$
```

The `~` means the user's home directory. The prompt itself is not part of the
command. Confirm the location with:

```bash
pwd
```

If you run:

```bash
python3 scripts/run_labs.py
```

from `/home/abel`, Python looks for `/home/abel/scripts/run_labs.py`. It does
not search all projects for a file with that name. You must either change into
the guide repository or give an absolute path.

This course works best with three sibling directories:

```text
projects/
├── tinygrad_docs/     this guide, including scripts/run_labs.py
├── tinygrad-study/    exact tinygrad snapshot used by the guide
└── tinygrad-work/     current fork or branch used for real contributions
```

The public guide checkout may have a longer directory name; its role is what
matters. Use these commands inside either Git repository to establish identity:

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
```

`git rev-parse --show-toplevel` prints the repository root. When a chapter says
“run from the guide root,” that is the directory containing this guide's
`mkdocs.yml`, `scripts/`, and `labs/`. When it says “run from the tinygrad
root,” that is the directory containing tinygrad's `tinygrad/`, `test/`, and
`pyproject.toml`.

### Relative and absolute paths

A **relative path** is interpreted from the current directory:

- `scripts/run_labs.py` means a `scripts` child of the current directory;
- `../tinygrad-study` means go to the parent, then enter `tinygrad-study`;
- `.venv/bin/python` means a Python executable inside the current directory's
  `.venv` child.

An **absolute path** begins at `/` and names the same location regardless of
the current directory. `realpath -e` turns an existing relative path into an
absolute one and fails if any path component does not exist:

```bash
realpath -e scripts/run_labs.py
realpath -e ../tinygrad-study
```

The `-e` matters on GNU/Linux: plain `realpath` can print a normalized path even
when its final component is missing. If either command fails, fix the current
directory or argument before debugging Python.

### Multi-line shell commands

A backslash at the very end of a shell line means “continue this command on
the next line”:

```bash
python3 scripts/run_labs.py \
  --tinygrad ../tinygrad-study \
  --device CUDA
```

There must be no space after the backslash. `\ ` escapes a space; it does not
continue the line. When copying a command, also omit the displayed shell
prompt.

## Why there are two tinygrad checkouts

A Git **commit** identifies one exact repository state. A **branch** is a named
pointer that normally moves forward as new commits are made. `HEAD` means the
commit currently checked out.

This guide's source links and expected outputs target one exact commit:

```text
874d33128b4e4785beea736d97df6716e0321717
```

Checking out that commit directly creates a **detached HEAD**. You can read,
run, and even edit files there, but new commits are not automatically attached
to a normal branch name. That is desirable for a disposable study checkout:
the guide cannot silently drift when upstream `master` changes.

A real contribution has the opposite need. It should start from current
upstream state, live on a named branch, and eventually be pushed to your fork.
Trying to make one checkout serve both roles creates two recurring mistakes:

- updating the study tree invalidates the guide's line numbers and expected
  output; or
- experiments and temporary logging contaminate the contribution branch.

Keep the roles separate:

| Checkout | Git state | Purpose | May intentionally drift? |
| --- | --- | --- | --- |
| `tinygrad-study` | detached at `874d331` | Follow this guide exactly and run its labs. | No. |
| `tinygrad-work` | named branch based on current upstream | Investigate and submit actual changes. | Yes, deliberately. |

Chapter 18 creates the contribution workflow. This chapter only needs the
study checkout.

## Four layers that can be “the wrong Python”

When `import tinygrad` behaves unexpectedly, distinguish these layers:

1. **Python executable** — the actual interpreter process, such as
   `/usr/bin/python3` or `tinygrad-study/.venv/bin/python`.
2. **Virtual environment** — an isolated set of installed Python packages
   associated with one interpreter.
3. **Editable installation** — a package entry that points imports at a source
   checkout instead of copying that checkout into site-packages.
4. **Imported module path** — the file Python actually loaded for
   `import tinygrad`.

Activating a virtual environment is optional. Calling its interpreter by
explicit path is less ambiguous in documentation:

```bash
.venv/bin/python -c 'import sys; print(sys.executable)'
```

An editable install created with `pip install -e ...` means source edits in
the checkout are visible to that environment. It does not guarantee that a
different `python3` command imports the same checkout. Always verify both the
interpreter and module path when identity matters.

## Preflight the host tools

Check, but do not change, the system first:

```bash
python3 --version
git --version
clang --version
nvidia-smi
```

Interpret the results one layer at a time:

- The pinned tinygrad package requires Python 3.11 or newer.
- Git is needed to obtain and identify the source.
- `clang` is needed by the default `CPU` route; it is not needed by the
  `PYTHON` route.
- `nvidia-smi` asks the installed NVIDIA driver whether it can see a GPU. It
  does not prove that Python can load NVRTC, that tinygrad can compile a
  program, or that a kernel returns correct values.

On Ubuntu, failure to create a virtual environment often means the
distribution's package matching the chosen Python version is missing (commonly
`python3-venv`). Driver and CUDA installation are system-administration tasks.
Do not replace a working driver merely because one tinygrad backend probe
failed. First establish the portable route and locate the failing layer.

When installation really is required, use NVIDIA's current
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
rather than a version-specific command copied into this snapshot guide.

## Create the pinned study checkout

Run the clone from the directory that should contain `tinygrad-study`. If that
name already exists, inspect it; do not delete or overwrite an unknown
checkout.

```bash
git clone https://github.com/tinygrad/tinygrad.git tinygrad-study
cd tinygrad-study
git switch --detach 874d33128b4e4785beea736d97df6716e0321717

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[testing_minimal]'
```

Read the commands causally:

1. `git clone` creates a new source checkout and its Git history.
2. `cd` makes that checkout the current directory.
3. `git switch --detach` selects the exact commit without moving a branch.
4. `python3 -m venv .venv` creates an isolated interpreter environment in the
   checkout.
5. invoking `pip` through `.venv/bin` ensures packages enter that environment;
6. `-e` points the installed package at this source tree; and
7. `testing_minimal` adds NumPy, PyTorch, pytest, Hypothesis, Z3, and related
   test tools used by core tests.

The larger `testing` extras are not a badge of completeness. Install an
additional dependency group when a selected test establishes that it needs
one. A minimal environment makes hidden dependencies and reproduction easier
to reason about.

### Prove checkout and import identity

Do not proceed on the basis of the directory name alone:

```bash
git rev-parse HEAD
git status --short --branch

DEV=PYTHON DEBUG=0 .venv/bin/python - <<'PY'
import sys, tinygrad
from tinygrad import Device

print("python:", sys.executable)
print("tinygrad:", tinygrad.__file__)
print("device:", Device.DEFAULT)
PY
```

Expected structural facts:

```text
874d33128b4e4785beea736d97df6716e0321717
## HEAD (no branch)
python: .../tinygrad-study/.venv/bin/python
tinygrad: .../tinygrad-study/tinygrad/__init__.py
device: PYTHON
```

The exact path prefix depends on where you cloned. A dirty status is not
automatically wrong—temporary study edits may be intentional—but it must be
recorded because it means you are no longer observing the pristine snapshot.

## Select a backend deliberately

Chapter 1 separated mathematical semantics from the mechanism used to execute
them. tinygrad's `DEV` environment variable selects that mechanism for a
process.

An environment assignment placed immediately before a command applies only to
that command:

```bash
DEV=PYTHON DEBUG=0 .venv/bin/python your_script.py
```

It does not permanently change the shell. By contrast, `export DEV=PYTHON`
affects later child processes until it is changed or unset. This guide usually
uses per-command assignments because every observation then records its own
backend.

The useful study routes are:

| `DEV` | What it exercises | What a success does **not** prove |
| --- | --- | --- |
| `PYTHON` | Runs lowered work through a portable Python implementation; a strong semantic control. | Native code generation, GPU behavior, or performance. |
| `CPU` | Renders, compiles with `clang`, allocates, and invokes a native CPU program. | GPU indexing, GPU runtime behavior, or GPU speed. |
| `CUDA` | Uses tinygrad's conventional NVIDIA route through the CUDA Driver API and a selected NVIDIA renderer/compiler. | Correctness of the separate lower-level NV runtime. |
| `NVK+NV` | Uses tinygrad's lower-level NVIDIA path with the explicit ordinary NVIDIA kernel-driver interface selected for this course. | Correctness of the CUDA Driver API route. |
| `NULL` | Exercises selected compiler/scheduling paths without meaningful device computation. | Numerical correctness or copyout. |

The study targets use a small grammar rather than arbitrary labels:

```text
[INTERFACE+]DEVICE[:RENDERER[:ARCH]]
```

The pieces select progressively more specific layers. `NVK+NV` therefore means
“use the `NV` device with the `NVK` interface,” not a device whose literal name
contains a plus sign. `CUDA:PTX` means “use the `CUDA` device and explicitly
select its `PTX` renderer.” The full grammar also permits interface indices
before the plus sign; they are not needed here. Most commands should name only
the device and let tinygrad choose sensible defaults. Chapter 14 returns to
every field after the runtime and renderer concepts are in place.

Here `NVK` is tinygrad's name for its `NVKIface`, an interface to the ordinary
NVIDIA kernel driver. It does **not** mean Mesa's NVK Vulkan driver. Also note
that `Device.DEFAULT` reports the device portion only: with `DEV=NVK+NV`, it
prints `NV`. Print the parsed `DEV` value when the interface choice is part of
the evidence.

The RTX 4090 is an Ada device with compute capability `sm_89`. It is within the
supported range of the pinned CUDA and NV paths. Agreement across `PYTHON`,
`CPU`, `CUDA`, and `NVK+NV` is useful differential evidence. It is not four
independent proofs: the routes share frontend and compiler layers.

`DEBUG` controls observation verbosity. At the levels used early in the guide:

| Setting | Main observation | Important effect |
| --- | --- | --- |
| `DEBUG=0` | Quiet result-oriented run. | Best default for correctness checks. |
| `DEBUG=1` | Device openings and compact planning information. | Output varies with device initialization. |
| `DEBUG=2` | One timed line per tracked execution call. | Timing can request synchronization, changing asynchronous behavior. |
| `DEBUG=4` | Generated target program text. | Large output; scope it to the operation under study. |

Later chapters introduce other levels and focused debug variables only when
their output is interpretable.

## Understand the two cache layers

Repeated compilation should normally be cached. Two kinds of reuse can affect
an experiment:

- **in-process state** disappears when the Python process exits; and
- **persistent cache state** can survive in a SQLite database selected by
  `CACHEDB`.

Use the normal cache for ordinary work. When the question is specifically
whether compilation or a compiler pass ran, give the command a dedicated
database path:

```bash
CACHEDB=/tmp/tinygrad-guide-setup.db DEV=CPU DEBUG=2 \
  .venv/bin/python your_script.py
```

A new pathname plus a new Python process creates a controlled cold start
without deleting a normal cache. Do not infer a semantic difference merely
because a generated name, cache label, or timing changed.

## Establish the portable feedback loop

Predict that both routes compute the same list, while the selected backend and
implementation differ:

```bash
for dev in PYTHON CPU; do
  echo "--- $dev ---"
  CACHEDB="/tmp/tinygrad-guide-setup-$dev.db" DEV="$dev" DEBUG=1 \
    .venv/bin/python - <<'PY'
from tinygrad import Device, Tensor
from tinygrad.helpers import DEV

out = (Tensor([-2.0, -1.0, 0.0, 1.0]) * 2 + 1).relu()
print("target:", DEV)
print("selected:", Device.DEFAULT)
print("value:", out.tolist())
PY
done
```

The stable result is:

```text
target: PYTHON
selected: PYTHON
value: [0.0, 0.0, 1.0, 3.0]
...
target: CPU
selected: CPU
value: [0.0, 0.0, 1.0, 3.0]
```

Additional `opened device ...` lines are expected with `DEBUG=1`. CPU may also
open `PYTHON` while turning a Python list into input storage. That staging work
does not mean the fused output computation executed on the Python backend.

The two routes answer different questions:

- If both return the wrong value, suspect the expression, frontend semantics,
  or shared compiler before suspecting `clang`.
- If `PYTHON` passes and `CPU` fails during compilation, inspect generated
  source and the compiler boundary.
- If CPU compiles but returns the wrong value, compare the lowered program,
  arguments, and runtime behavior.

Now run one exact test on each backend:

```bash
CACHEDB=/tmp/tinygrad-guide-test-python.db DEV=PYTHON DEBUG=0 \
  .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus

CACHEDB=/tmp/tinygrad-guide-test-cpu.db DEV=CPU DEBUG=0 \
  .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus
```

In pytest syntax, `file.py::Class::test_name` selects one test method. `-q`
reduces reporting noise; it does not change the assertion. A passing broad
suite is reassuring, but a focused test is a faster and clearer first answer
to a narrow claim.

## Add the RTX 4090 feedback loop

First ask whether the operating system and driver can identify the device:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
```

Record the output. Then run the same semantic workload independently on each
NVIDIA route:

```bash
for dev in CUDA NVK+NV; do
  echo "--- $dev ---"
  CACHEDB="/tmp/tinygrad-guide-$dev.db" DEV="$dev" DEBUG=1 \
    .venv/bin/python - <<'PY'
from tinygrad import Device, Tensor
from tinygrad.helpers import DEV

out = (Tensor([-2.0, -1.0, 0.0, 1.0]) * 2 + 1).relu()
print("target:", DEV)
print("selected:", Device.DEFAULT)
print("value:", out.tolist())
PY
done
```

Each successful route should return `[0.0, 0.0, 1.0, 3.0]`. The second route's
identity lines are `target: NVK+NV` and `selected: NV`; that is expected because
the latter omits the interface. Run routes separately when debugging. A CUDA
success does not make an NV failure impossible or unimportant; it narrows the
first divergent layer.

The driver, compiler library, renderer, and runtime are separate layers. If
the CUDA device opens but NVRTC cannot be loaded, this diagnostic asks the CUDA
runtime to use tinygrad's PTX renderer instead:

```bash
DEV=CUDA:PTX DEBUG=1 .venv/bin/python -c \
  'from tinygrad import Tensor; print((Tensor([1, 2, 3]) + 1).tolist())'
```

If `CUDA:PTX` works while plain `CUDA` fails to find NVRTC, runtime access and
source-compiler discovery have been separated. Do not silently make the
diagnostic a permanent workaround; retain the original exception and decide
which route the intended contribution actually requires.

Finally, run the same focused test on the primary accelerator:

```bash
CACHEDB=/tmp/tinygrad-guide-test-cuda.db DEV=CUDA DEBUG=0 \
  .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus

CACHEDB=/tmp/tinygrad-guide-test-nv.db DEV=NVK+NV DEBUG=0 \
  .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus
```

## Use tests in widening rings

Testing is not “small suite versus big suite.” Each ring answers a different
question:

| Ring | Command shape | Question |
| --- | --- | --- |
| Reproducer | Ten or fewer lines of Python | Does the behavior exist, and on which backend? |
| One test | `pytest path.py::Class::test` | Does the nearest explicit contract pass? |
| Nearby file/subsystem | `pytest test/backend/test_ops.py` | Did adjacent behavior change? |
| Backend comparison | Repeat the same selection with controlled `DEV` values | Is the first disagreement frontend/compiler or backend-specific? |
| Relevant CI job | Reproduce the upstream workflow's exact selection and flags | Does the evidence match the project's integration contract? |
| Broader suites | Chosen according to the change's risk | Did the change disturb less-local behavior? |

Start at the smallest ring capable of falsifying the claim. If it fails, make
the case smaller before adding concurrency and unrelated tests.

At the pinned snapshot, the main test step of the Python-backend CI job selects
several backend files rather than every test in the repository. The following
reproduces that **test selection** in the guide environment:

```bash
SKIP_SLOW_TEST=1 DEV=PYTHON .venv/bin/python -m pytest -n=auto \
  test/backend/test_dtype.py \
  test/backend/test_dtype_alu.py \
  test/backend/test_ops.py \
  test/backend/test_uops.py \
  test/backend/test_symbolic_ops.py \
  test/backend/test_renderer_failures.py::TestRendererFailures \
  --durations=20
```

`-n=auto` asks pytest-xdist to use multiple worker processes. Do not introduce
parallel workers while reducing a flaky or stateful failure; reproduce it
alone first. The CI job installs `testing_unit`, a superset of the
`testing_minimal` environment created in this chapter, and it has additional
image and emulated-tensor-core steps. Exact CI reproduction therefore requires
that larger dependency group and those commands; the block above is not the
whole job. Current upstream CI can change, so Chapter 16 teaches how to read
the live workflow before submitting a patch.

## Run this guide's bundled labs from the guide repository

The bundled runner belongs to the **guide**, not to tinygrad. Change into the
guide root before invoking it:

```bash
cd /path/to/tinygrad_docs
pwd
realpath -e scripts/run_labs.py
realpath -e ../tinygrad-study
test -f scripts/run_labs.py
test -d ../tinygrad-study/tinygrad

python3 scripts/run_labs.py \
  --tinygrad ../tinygrad-study \
  --python ../tinygrad-study/.venv/bin/python
```

Replace `/path/to/tinygrad_docs` with the actual guide checkout. If the guide
and study checkout are not siblings, pass the study checkout's absolute path.

On the Ubuntu/RTX 4090 route, add the two hardware backends:

```bash
python3 scripts/run_labs.py \
  --tinygrad ../tinygrad-study \
  --python ../tinygrad-study/.venv/bin/python \
  --device CUDA \
  --device NVK+NV
```

The runner executes a controlled selection of guide labs with fresh temporary
cache paths. The Phase 1 trace, Phase 2 frontend/UOp/rewrite labs, and Phase 3
scheduling and shape/indexing labs always run on `PYTHON`. Runtime-oriented
labs run on `PYTHON`, on a hardware-free, Python-executed CUDA-targeted
`PYTHON::sm_89` structural route where applicable, and again on every added
`--device`.
The Phase 3 kernel-optimization lab runs its core, strict-padding, and
padding-enabled modes only on `PYTHON::sm_89`. They check pinned Ada-targeted
options, `WMMA` structure, and complete small results with the Python executor,
never on an added hardware backend, and make no physical-GPU or timing claim.
The Phase 3 lowering lab also runs only on that structural route, with
`NOOPT=1` and `SPEC=2`, so its range, accumulator, address, barrier, and control
assertions describe one intentionally controlled lowered program.
The Phase 3 rendering walk executes its serialized artifact on `PYTHON`, checks
a hardware-free direct-PTX artifact, and tries CUDA C-to-PTX only when an NVRTC
library can be loaded. An NVRTC compile rejection fails; only the explicit
missing-library state skips. When an added device is `CPU` or `CPU:CLANG`, the
runner also compiles and executes the Clang route. It does not run this
route-specific lab on added CUDA or NVK devices. The runner sets
`PYTHONOPTIMIZE=0` so Python cannot strip the labs' assertion-based evidence
checks.
The Phase 4 runtime walk runs first on the synchronous `PYTHON` baseline and
again on every explicitly added `--device`. It checks common device, buffer,
program-loading, dispatch, synchronization, and result contracts, while making
backend-class assertions only for the routes it names. On this guide's Ubuntu
host it has also been exercised successfully on the RTX 4090 through both
`CUDA` and `NVK+NV`; that local observation is not a portability guarantee for
another driver or machine.
The Phase 4 TinyJit state walk runs in three fresh `PYTHON` processes with
`JIT=0`, `JIT=1`, and `JIT=2`; the contract lab runs once with `JIT=1`.  Both
runner entries select the readable Python backend, and the contract lab also
pins that route internally.  They prove capture, replay, input, return,
mutation, and lifecycle observations without claiming a device graph or
accelerator timing result, and they are not repeated for added hardware
`--device` values.  Chapter 13 gives a separate, explicitly scoped way to run
the state walk on a selected accelerator.
The Phase 4 NVIDIA-path lab always runs its hardware-free `static` mode on
`PYTHON::sm_89`.  For the exact added routes `CUDA`, `CUDA:PTX`, `NVK+NV`, or
`NVK+NV:PTX`, the runner also selects the matching physical mode and requires
it to be available.  A recognized missing driver, device node, compiler
library, or exact ABI symbol therefore fails an explicitly requested runner
route instead of being counted in the final success line.  The ordinary manual
lab command remains non-failing on a recognized unavailable host unless
`--require-available` is supplied.  Chapter 14 explains the route-specific
evidence and the NV initialization commands which can precede renderer
preflight.
The Phase 5 debugging walk always runs its independent control on the exact
`PYTHON` route.  If `--device CPU` or `--device CPU:CLANG` was supplied, it
also runs the expected-defect and fixed-regression modes in fresh processes,
both with the exact `CPU:CLANG` spelling.  Those modes reuse one already
lowered program and prove that `SOURCE` is the first differing artifact after
an identical `LINEAR`; they do not edit tinygrad or turn arbitrary failures
into passing results.  Added accelerator devices do not cause this deliberately
CPU-renderer-specific experiment to be replayed on a GPU.
The Phase 5 testing walk then runs its deliberate-red mutant and green
regression in separate processes on exact `PYTHON`.  Both modes use one
unchanged five-test semantic contract.  The red mode succeeds only for its
exact set and count of assertion failures with zero unexpected errors; the
green mode must pass the same contract through tinygrad.  Neither mode is
repeated for added devices because this lab teaches test power and portable
oracle design, not backend coverage.
The Phase 5 performance walk always runs five samples on exact `PYTHON` and
then repeats on every additional device except a duplicate `PYTHON` entry.  It
checks an independent exact oracle before timing, immediately after its
synchronized wall samples, and again after its internal `time_call` samples.
It reports raw distributions and concrete backend/interface,
renderer/compiler/runtime, target, and artifact identities but asserts no speed
threshold.  Bare `NV` and direct `PCI` spellings are rejected before tinygrad
imports; use an explicit driver-backed route such as `NVK+NV` or `CUDA`.
Thus `--device CUDA` does not replay every lab on CUDA. The final success line
is:

```text
All selected labs passed.
```

That proves only the selected lab processes completed on the named checkout
and backends. It is not a full tinygrad test suite or a technical review of
every prose exercise.

If Python says:

```text
can't open file '/home/you/scripts/run_labs.py'
```

the path in the exception is the evidence: the command ran from
`/home/you`, not the guide root. Fix `cd` or use the script's absolute path.

## Guided source tour: confirm the configuration contract

Setup needs only three small source questions. Runtime implementation files
are deliberately deferred until the guide has taught their contracts.

### Stop 1: which Python and test dependencies does the snapshot declare?

Read [`pyproject.toml` lines 1–16](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L1-L16).

Question: what minimum Python version and mandatory runtime dependencies are
declared?

Translation: the package requires Python 3.11 or newer and has no mandatory
third-party runtime dependency list at this snapshot.

Then read only [`testing_minimal` lines 72–80](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L72-L80).
Those names explain what the editable-install command added. The larger groups
below it are not required reading.

### Stop 2: why does setting `DEV` avoid automatic probing?

Read [`Device.DEFAULT` and `_select_device` lines 38–55](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L38-L55).

Question: which branch supplies the default when `DEV` names a device, and
what happens otherwise?

Translation: an explicit parsed device value is returned directly. Without
one, tinygrad tries available device implementations and records the first
usable route. This is why every course command sets `DEV`: installing a driver
or changing probe order cannot silently change the experiment.

Ignore the class-loading and device-construction methods above this range.
Chapter 12 explains them after allocator and runtime concepts exist.

### Stop 3: what does upstream's Python-backend CI actually run?

Read [the Python-backend job lines 89–104](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L89-L104).

Question: which dependency group, environment variables, test files, and
special extra command define the job?

Translation: CI is an executable selection, not a vague instruction to “run
tests.” Your local ring should mirror the portions relevant to the change.

Do not read `ops_cuda.py`, `ops_nv.py`, or the full compiled-device contract as
setup material. A successful smoke test is enough now; Chapters 12 and 14
provide the questions needed to interpret those files.

## Lab checkpoint: write an environment card

Save this in a study notebook and replace every blank with observed output:

```text
guide repository root:
tinygrad study repository root:
tinygrad work repository root (or “not created yet”):
study commit:
study git status:
Python version and executable:
imported tinygrad path:
OS and kernel:
CPU and clang version:
GPU, driver, compute capability:
CACHEDB used for controlled runs:
DEV=PYTHON value / focused test:
DEV=CPU value / focused test:
DEV=CUDA value / focused test:
DEV=NVK+NV value / focused test:
bundled runner selection and result:
first failing layer, if any:
```

You pass when another person could reproduce the same observation without
guessing a directory, interpreter, commit, cache, or backend.

Also be able to narrate this result:

> I ran the pinned commit with its own virtual-environment interpreter. I set
> the backend explicitly. The portable semantic control passed. The next route
> first failed while opening the device / rendering / compiling / allocating /
> copying / invoking / synchronizing / comparing values.

“The GPU failed” is not yet a location.

## Troubleshooting by first failing layer

| Symptom | Evidence to gather before changing anything |
| --- | --- |
| `scripts/run_labs.py` not found | `pwd`, guide repository root, and `realpath -e scripts/run_labs.py`; run from the guide, not home or tinygrad. |
| `.venv/bin/python` not found | `pwd`, `ls -la .venv/bin`, and the environment-creation command's first error. |
| Import resolves outside `tinygrad-study` | Print `sys.executable` and `tinygrad.__file__`; rerun the editable install with the study environment's pip. |
| Commit differs from the guide | Record `git rev-parse HEAD` and status. Use the pinned study checkout rather than forcing a work branch backward. |
| `CPU` cannot initialize or compile | Confirm `clang` is discoverable. Retain `PYTHON` as the semantic control and capture generated/compiler errors. |
| `nvidia-smi` cannot see the 4090 | Fix OS/driver visibility outside tinygrad first. A container must also be given device access. |
| Device opens but NVRTC cannot load | Separate driver access from compiler-library discovery; use `CUDA:PTX` only as the documented diagnostic. |
| `CUDA` passes and `NVK+NV` fails | Keep the exact same reproducer. Inspect the first NV initialization/runtime boundary rather than changing Tensor math. |
| Values differ across backends | Freeze input, dtype, shape, commit, and cache/debug context; find the earliest differing graph/program/runtime artifact. |
| Timings or generated names change | Use a fresh process and controlled `CACHEDB`; compare structure and values before performance. |
| A broad suite fails first | Rerun the first failure alone with one backend and no worker concurrency. |

## Quick reference

### Where commands run

| Task | Required current directory |
| --- | --- |
| `scripts/run_labs.py` | Guide repository root |
| `.venv/bin/python`, tinygrad tests, source experiments | `tinygrad-study` root |
| Real patch, branch, and upstream comparison | `tinygrad-work` root |

### Identity

```bash
pwd
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short --branch
.venv/bin/python -c 'import sys, tinygrad; print(sys.executable); print(tinygrad.__file__)'
```

### Backend ladder

```bash
DEV=PYTHON DEBUG=0 .venv/bin/python reproducer.py
DEV=CPU    DEBUG=0 .venv/bin/python reproducer.py
DEV=CUDA   DEBUG=0 .venv/bin/python reproducer.py
DEV=NVK+NV DEBUG=0 .venv/bin/python reproducer.py
```

### Smallest test

```bash
DEV=PYTHON DEBUG=0 .venv/bin/python -m pytest -q \
  path/to/test.py::Class::test_name
```

### Controlled cache and debug

```bash
CACHEDB=/tmp/tinygrad-investigation.db DEV=CPU DEBUG=2 \
  .venv/bin/python reproducer.py
```

## Optional reinforcement—not missing prerequisites

- Git's [branches in a nutshell](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
  provides a visual model of branch pointers. Stop after creating and switching
  branches; rebasing belongs to Chapter 18.
- Python's [`venv` documentation](https://docs.python.org/3/library/venv.html)
  explains environment creation and activation. For this guide, explicit
  `.venv/bin/python` paths are sufficient.
- pytest's [usage guide](https://docs.pytest.org/en/stable/how-to/usage.html)
  lists node-selection and reporting flags. Stop after selecting tests; test
  design is taught in Chapter 16.

## What is deliberately left for later

- Chapter 3 turns the verified environment into an end-to-end artifact trace.
- Chapter 12 explains `Device`, `Buffer`, `Allocator`, `Program`, and
  synchronization contracts.
- Chapter 14 opens the CUDA and NV implementations after teaching the required
  GPU/runtime model.
- Chapter 16 derives a test matrix from a particular change rather than running
  broad suites by habit.
- Chapter 18 creates the current contribution branch or worktree and translates
  snapshot knowledge to live upstream.

[← Orientation](01-orientation.md) · [Next: trace one expression end to end →](03-first-trace.md)
