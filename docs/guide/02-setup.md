# 2. Development setup: make every observation reproducible

## Purpose

A compiler investigation is only useful when you know which source, backend,
cache, and test produced it. This chapter creates a pinned study checkout and
a small backend matrix that lets you separate Tensor semantics from CPU,
CUDA, and NV runtime behavior.

Ubuntu with an RTX 4090 is the primary path. The first loop nevertheless uses
`PYTHON` and `CPU`, so almost every reader has a known-good route before
introducing the GPU.

## Prerequisite gate

You should be comfortable creating a virtual environment and reading basic Git
state. In particular, know that a detached `HEAD` names an exact commit but is
not a branch on which to keep contribution work. If that distinction is new,
read [Git's branches-in-a-nutshell chapter](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
through “Creating a New Branch.”

Check the local tools:

```bash
python3 --version       # tinygrad 0.13.0 requires Python >= 3.11
git --version
clang --version         # needed by the default CPU renderer, not by PYTHON
nvidia-smi              # NVIDIA branch only
```

On Ubuntu, a missing `venv` module is supplied by the distribution package
matching your Python installation (commonly `python3-venv`). Driver and CUDA
installation are system-administration tasks; do not change them merely
because a tinygrad backend probe failed until the portable checks below pass.
When installation really is the missing layer, use NVIDIA's current
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
rather than a version-specific command copied from this guide.

## Mental model: two checkouts and five useful backends

Use two independent tinygrad checkouts once you begin real work:

```text
tinygrad-study  ── detached at 874d331; matches this guide line for line
tinygrad-work   ── your fork/branch based on current master; receives changes
```

Do experiments in the first and contributions in the second. This prevents a
guide exercise from dirtying a real branch and makes “does this differ from the
snapshot?” an explicit comparison.

`DEV` selects the execution target. These are the useful Phase 1 roles:

| `DEV` | Role in the feedback loop | Important limitation |
| --- | --- | --- |
| `PYTHON` | Most portable value-producing backend; excellent for semantic reduction and differential checks. | It interprets lowered work and is not a performance proxy. |
| `CPU` | Exercises real rendering, compilation, allocation, and execution without a GPU. Its generated C is approachable. | The default path needs `clang`; CPU-specific lowering is not GPU behavior. |
| `CUDA` | NVIDIA path through the CUDA Driver API; the default renderer compiles CUDA source with NVRTC. | A working NVIDIA driver and discoverable CUDA compiler libraries are separate requirements. |
| `NV` | tinygrad's lower-level NVIDIA/HCQ path, supported on Ampere, Ada, and Blackwell at this snapshot. | It has a different runtime and failure surface from `CUDA`; do not use its direct `PCI` interface casually. |
| `NULL` | Runs compiler and scheduling tests without meaningful device computation. | It cannot establish numerical correctness; copyout is disabled by default. |

The RTX 4090 is Ada (`sm_89`) and is in the supported range for both NVIDIA
paths. Begin with `CUDA` as the familiar control and add `NV` as a second
implementation. Agreement between two backends is useful evidence, not proof:
they share much of the compiler above the runtime.

## Create the pinned study checkout

Run these commands in the directory that should contain the checkout:

```bash
git clone https://github.com/tinygrad/tinygrad.git tinygrad-study
cd tinygrad-study
git switch --detach 874d33128b4e4785beea736d97df6716e0321717

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[testing_minimal]'
```

`testing_minimal` is intentionally enough for core tests: NumPy, PyTorch,
pytest, xdist, Hypothesis, and Z3. Add a larger optional dependency group only
when a selected test requires it; installing `.[testing]` up front makes the
environment slower to reproduce and does not tell you which dependency your
work actually needs.

Verify identity before debugging anything:

```bash
test "$(git rev-parse HEAD)" = 874d33128b4e4785beea736d97df6716e0321717
DEV=PYTHON DEBUG=0 .venv/bin/python - <<'PY'
import sys, tinygrad
from tinygrad import Device
print("python:", sys.version.split()[0])
print("tinygrad package:", tinygrad.__file__)
print("selected control device:", Device.DEFAULT)
PY
```

The package path should be inside `tinygrad-study`. All guide commands set
`DEV` explicitly so adding a driver or changing probe order cannot silently
change an experiment (and so setup does not probe every available runtime).

### Cache discipline

tinygrad stores compiled programs and other cached results in SQLite. Normal
development should use the default cache. For a controlled guide experiment,
give `CACHEDB` a dedicated path under `/tmp` and record it with the command:

```bash
export CACHEDB=/tmp/tinygrad-guide-phase1.db
```

A new pathname gives a cold disk cache without touching your normal cache. A
fresh Python process also clears in-process caches. Cache hits are expected in
real use; they become a confounder only when the question is specifically
about compilation or pass execution.

## Establish the portable loop

Predict that both commands print the same list, while `Device.DEFAULT` and the
implementation below it differ:

```bash
for dev in PYTHON CPU; do
  echo "--- $dev ---"
  DEV="$dev" DEBUG=1 .venv/bin/python - <<'PY'
from tinygrad import Device, Tensor
out = (Tensor([-2.0, -1.0, 0.0, 1.0]) * 2 + 1).relu()
print("selected:", Device.DEFAULT)
print("value:", out.tolist())
PY
done
```

Checkpoint output:

```text
selected: PYTHON
value: [0.0, 0.0, 1.0, 3.0]
...
selected: CPU
value: [0.0, 0.0, 1.0, 3.0]
```

Extra `opened device ...` lines are expected with `DEBUG=1`. CPU may also open
`PYTHON` to stage a Python list before copying it to CPU storage.

Run one exact test on both backends:

```bash
DEV=PYTHON DEBUG=0 .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus
DEV=CPU DEBUG=0 .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus
```

The important skill is selecting the smallest test that falsifies your claim,
not running the largest suite first.

## Add the RTX 4090 loop

First establish whether the operating system can see the device:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
```

Then run the same semantic smoke test independently on each NVIDIA backend:

```bash
for dev in CUDA NV; do
  echo "--- $dev ---"
  CACHEDB="/tmp/tinygrad-guide-$dev.db" DEV="$dev" DEBUG=1 \
    .venv/bin/python - <<'PY'
from tinygrad import Device, Tensor
out = (Tensor([-2.0, -1.0, 0.0, 1.0]) * 2 + 1).relu()
print("selected:", Device.DEFAULT)
print("value:", out.tolist())
PY
done
```

Both successful paths should report `[0.0, 0.0, 1.0, 3.0]`. A failure in one
does not invalidate the other. Record the full exception, `DEV`, driver
version, and whether `PYTHON` and `CPU` pass before investigating.

Finally, run the same focused test on the accelerator you intend to study:

```bash
DEV=CUDA DEBUG=0 .venv/bin/python -m pytest -q \
  test/test_tiny.py::TestTiny::test_plus
# Repeat with DEV=NV once the NV smoke test passes.
```

The default `CUDA` and `NV` renderers use NVRTC at this snapshot. If the driver
works but NVRTC is unavailable, this diagnostic chooses tinygrad's PTX
renderer on the CUDA runtime:

```bash
DEV=CUDA:PTX DEBUG=1 .venv/bin/python -c \
  'from tinygrad import Tensor; print((Tensor([1, 2, 3]) + 1).tolist())'
```

Use that as evidence that runtime access and renderer/compiler discovery are
different layers. Do not silently make it the permanent workaround; capture
the original error and decide which layer your intended contribution needs.

## Learn the test hierarchy

Use tests in widening rings. Stop at the first ring that fails and reduce
there.

| Ring | Command shape | Question answered |
| --- | --- | --- |
| Reproducer | A short `python - <<'PY'` program | Does the behavior exist at all, and on which backend? |
| One test | `pytest file.py::Class::test_name` | Does the nearest regression contract pass? |
| One subsystem/file | `pytest -n12 test/backend/test_ops.py` | Did the change disturb adjacent behavior? |
| Backend comparison | Repeat the same selection with `DEV=PYTHON`, `CPU`, then the target | Is the first disagreement semantic, codegen, or runtime-specific? |
| Relevant CI jobs | Mirror commands from `.github/workflows/test.yml` | Does the evidence match the project's actual matrix? |

For example, the snapshot's Python-backend CI ring is:

```bash
SKIP_SLOW_TEST=1 DEV=PYTHON .venv/bin/python -m pytest -n12 \
  test/backend/test_dtype.py \
  test/backend/test_dtype_alu.py \
  test/backend/test_ops.py \
  test/backend/test_uops.py \
  test/backend/test_symbolic_ops.py \
  test/backend/test_renderer_failures.py::TestRendererFailures
```

Use fewer than 12 workers if the machine is memory-constrained. “All of
`test/`” is not a faithful substitute for CI: the repository has hardware,
model, external, lint, and process-replay jobs with distinct dependencies and
environment variables. Read the live workflow before contribution work.

## Source tour

| Landmark | Why it matters | Source at `874d331` |
| --- | --- | --- |
| Package metadata | Records Python `>=3.11` and the minimal testing dependency set used above. | [`pyproject.toml`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L1-L16), [`testing_minimal`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/pyproject.toml#L57-L81) |
| Device discovery | Shows explicit `DEV` selection, canonicalization, probing, and lazy device construction. | [`tinygrad/device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L14-L55) |
| Device contract | Separates allocator, renderer/compiler, runtime program, and synchronization. | [`tinygrad/device.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/device.py#L303-L377) |
| CUDA runtime | Uses the CUDA Driver API for context, memory, module load, and launch. | [`tinygrad/runtime/ops_cuda.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_cuda.py#L37-L125) |
| NV runtime | Selects an NV interface, establishes hardware queues, discovers architecture, then supplies allocator/renderer/runtime types. | [`tinygrad/runtime/ops_nv.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/ops_nv.py#L585-L641) |
| Python CI | Gives an authoritative backend test selection rather than an invented “quick suite.” | [`.github/workflows/test.yml`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/.github/workflows/test.yml#L89-L125) |

## Lab checkpoint: write an environment card

Save this information in the study notebook:

```text
tinygrad commit:
Python version and executable:
editable package path:
OS/kernel:
CPU and clang version:
GPU, driver, compute capability:
DEV=PYTHON smoke/test:
DEV=CPU smoke/test:
DEV=CUDA smoke/test:
DEV=NV smoke/test:
CACHEDB used:
first failing layer, if any:
```

You pass the checkpoint when:

- the commit and import path are unambiguous;
- `PYTHON` and either `CPU` or a chosen accelerator produce the expected
  value;
- the focused `test_plus` passes on the portable backend and your primary
  backend; and
- given a backend failure, you can say whether it happened during device
  opening, rendering/compilation, allocation/copy, launch, synchronization, or
  numerical comparison.

### Troubleshooting map

| Symptom | First check |
| --- | --- |
| Import resolves outside the study checkout | Use `.venv/bin/python`, rerun the editable install from the repository root, and print `tinygrad.__file__`. |
| Snapshot output does not match | Check `git rev-parse HEAD` and `git status --short`; do not debug current `master` against snapshot line numbers. |
| `CPU` cannot initialize | Confirm `clang` is in `PATH`; retain `PYTHON` as the semantic control. |
| `nvidia-smi` cannot see the 4090 | Fix driver/device visibility outside tinygrad first. Containers also need the GPU passed through. |
| `libnvrtc` cannot be loaded | Distinguish CUDA toolkit/compiler-library discovery from driver access; try `CUDA:PTX` only as the diagnostic above. |
| `CUDA` passes, `NV` fails | Keep the same reproducer and inspect the NV initialization/runtime layer; do not rewrite Tensor/compiler code merely to make the paths agree. |
| Timings or cache labels change | Use a fresh process and dedicated `/tmp` `CACHEDB`; compare structure and values before performance. |
| A broad suite fails first | Re-run the exact first failure alone with one backend and no worker concurrency. |

## Quick reference

```bash
# identity
git rev-parse HEAD
.venv/bin/python -c 'import tinygrad; print(tinygrad.__file__)'

# portable controls
DEV=PYTHON DEBUG=0 .venv/bin/python your_reproducer.py
DEV=CPU    DEBUG=0 .venv/bin/python your_reproducer.py

# RTX 4090 paths
DEV=CUDA DEBUG=0 .venv/bin/python your_reproducer.py
DEV=NV   DEBUG=0 .venv/bin/python your_reproducer.py

# smallest pytest selection
DEV=PYTHON DEBUG=0 .venv/bin/python -m pytest -q path/to/test.py::Class::test_name

# controlled cache and debug summary
CACHEDB=/tmp/tinygrad-investigation.db DEV=CPU DEBUG=2 \
  .venv/bin/python your_reproducer.py
```

[← Orientation](01-orientation.md) · [Next: trace one expression end to end →](03-first-trace.md)
