# 13. TinyJit and graph replay

## The promise of this chapter

Suppose a training step has already run once.  Its kernels may already be in
tinygrad's program cache, and the runtime may already have loaded them.  A
normal second Python call can nevertheless repeat a surprising amount of host
work: execute the model's Python, build lazy Tensor UOps, schedule them, resolve
buffers, plan temporary memory, look up compiled programs, and submit every
kernel separately.

`TinyJit` is tinygrad's answer for a computation whose execution structure is
stable.  It runs the Python function twice, captures the scheduled calls on the
second run, turns their input buffers into replaceable parameters, and replays
that prepared `LINEAR` plan on later calls.  On a backend with graph support,
it may additionally combine compatible calls into a device graph so one graph
submission replaces several individual submissions.

Those are several different mechanisms.  This chapter builds them one at a
time for a reader who knows Python and ML but has not studied JIT compilers or
GPU graph APIs.  By the end, you should be able to:

- narrate ordinary execution, ignore, capture, and replay without calling all
  of them “compilation”;
- identify which Python work disappears on replay and which validation still
  occurs;
- separate the schedule cache, program/compiler caches, captured `LINEAR`,
  `link_linear`, and optional device graph;
- state TinyJit's exact current input and return contracts, including shallow
  containers, symbolic bindings, dtype/device/view matching, and base-buffer
  alias rejection;
- predict what happens to Python scalar branches, closure values, side effects,
  returned Tensor objects, and retained output storage;
- reason about read/write mutation and the narrower pure-write copy rule;
- use `prune`, `reset`, `free_intermediates`, and `JIT=0/1/2` deliberately;
- measure host submission and device completion without confusing them; and
- localize a failure to ordinary execution, capture lowering, replay binding,
  or backend graph execution.

All implementation claims and source links in this chapter target tinygrad
commit `874d33128b4e4785beea736d97df6716e0321717` from 2026-08-05.  TinyJit is
active development code.  Recheck live source and tests before proposing a
change.

## Route through the chapter

Read this chapter front to back once.  The order is deliberate:

1. recover what an ordinary Python call does;
2. separate four kinds of reuse;
3. walk ignore, capture, and replay one statement at a time;
4. inspect the captured `LINEAR` lifecycle;
5. learn input discovery before memorizing failure messages;
6. treat Python control flow and closures as capture-time specialization;
7. learn return-object and storage reuse before retaining outputs;
8. add mutation, pure-write buffers, pruning, reset, and freeing;
9. add device graphs only after plain replay is clear;
10. run deterministic `PYTHON` labs;
11. learn honest timing and failure localization; and
12. use bounded, question-led source stops to prepare a contribution.

The quick reference at the end is for later recall.  It is not a substitute
for the first front-to-back pass.

## Prerequisite and background gate

### What this chapter assumes

You should already be able to explain these Chapter 12 facts:

- a `LINEAR` UOp contains ordered `CALL`s;
- a compiled call can contain an `Ops.PROGRAM` plus buffer arguments;
- `run_linear(..., jit=True)` means “this plan is already compiled and linked,”
  not “start TinyJit capture”;
- `wait=False` may mean submission rather than completion on an accelerator;
  and
- a Tensor view and its base allocation are different identities.

Review [Devices, memory, and runtime execution](12-runtime.md) if any item is
unclear.  No CUDA C, compiler course, or vendor graph-API experience is needed
to begin this chapter.

### Three Python ideas you need

**A decorator replaces a name with an object.**  After:

```python
@TinyJit
def step(x):
  return (x + 1).realize()
```

`step` names a `_TinyJit` instance that holds the original function in
`step.fxn`, a call counter in `step.cnt`, and eventually a capture in
`step.captured`.  Calling `step(x)` invokes `_TinyJit.__call__`; it does not
necessarily call the original Python function.

**A closure is data remembered by a Python function.**  If `step` reads
`learning_rate` from an enclosing scope, that value is not automatically a
Tensor input.  Once replay begins, the function body is not executed and the
new closure value is not read.

**Object identity differs from value equality.**  Two Tensor wrappers can show
equal numbers while owning different buffers.  Conversely, one captured output
Tensor wrapper can show different numbers after each replay because its storage
is overwritten.  The labs print both values and identity relationships.

### Background ladders: learn only when the chapter asks for it

If “host versus device” or asynchronous submission is new, read the execution
model portion of [Learning resources](../reference/learning-resources.md#gpu-execution-on-the-rtx-4090-path)
until you can distinguish a Python thread issuing work from a device completing
work.  Return here; grids and warps are not yet necessary.

If a dependency graph is new, use this minimal model:

```text
A produces y ──► B consumes y ──► C writes result
```

The arrows say what must precede what.  A **graph replay** stores a previously
constructed set of operations and dependencies, then launches that structure
again after updating allowed parameters such as input addresses.  You do not
need graph theory beyond nodes, edges, and topological order for this chapter.

## Begin with an ordinary Python call

Consider a function without `TinyJit`:

```python
def add_one(x):
  return (x + 1.0).realize()
```

Every call executes the Python body.  At a high level:

```text
Python evaluates x + 1.0
  ↓
Tensor methods construct lazy UOps
  ↓
.realize() turns the required graph into a LINEAR schedule
  ↓
buffers are resolved and temporary memory is planned
  ↓
kernels become PROGRAMs, reusing caches where keys match
  ↓
runtime calls are submitted/executed
  ↓
the Python function returns its new Tensor wrapper
```

A cache can shorten one arrow without removing the arrows above it.  A compiler
cache hit does not mean Python stopped running.  A runtime program cache hit
does not mean scheduling stopped.  This distinction is the reason TinyJit is
useful even after kernel compilation is warm.

## Four layers of reuse, kept separate

The word *cache* is too vague for useful diagnosis.  At this snapshot, the
important layers are:

| Layer | Key idea | What it can reuse | What it does not by itself remove |
| --- | --- | --- | --- |
| Schedule cache | `lower_sink_to_linear` caches a scheduled form by the function UOp key when `SCACHE` is enabled | Work in `create_schedule(...)` for the same key | Python frontend, resolving fresh buffers, memory planning, program lookup, execution |
| Program/compiler caches | `to_program_cache` reuses an `Ops.PROGRAM`; compiler wrappers may also reuse compiled bytes | Lowered/rendered/compiled kernel artifact | Python frontend, scheduling, plan construction, per-call submissions |
| TinyJit capture | Stores a parameterized, memory-planned, compiled `LINEAR` plus the return object and input contract | The execution-plan structure across compatible calls | Input preparation/validation and execution of the captured calls |
| Device graph | Encodes a compatible batch of program/copy calls in a backend graph object | Backend command structure and one or more individual submissions | Input/scalar/dimension updates, graph launch, and device computation |

There is also a runtime-object cache that can reuse the object which loads or
interprets an already compiled program.  It is another useful boundary, not a
replacement for the four rows.

### Why submission overhead matters

**Host submission overhead** is CPU-side time spent preparing and issuing work:
Python calls, argument packing, driver/API calls, queue bookkeeping, and related
work.  It is not the time the GPU spends executing the kernel.  If one kernel
takes 5 microseconds and the host spends 10 microseconds arranging its launch,
improving kernel arithmetic alone cannot remove the 10 microseconds.  A model
with many tiny kernels can become submission-bound after compilation is warm.

TinyJit can remove repeated Python/frontend/plan work.  A device graph may
further reduce per-kernel submission work.  Neither makes the mathematical
operations free.

## TinyJit's state lives on the decorated object

A new `_TinyJit` begins with:

```text
fxn       = original Python function
cnt       = 0
captured  = None
prune     = decorator option
```

The `cnt` value tested at the start of a JIT-enabled call selects a phase.  The
counter increments at the end of every successful call.

### Call 1: ignore

“Ignore” means **ignore this call for capture**, not ignore the computation.
TinyJit first prepares the inputs, then executes the original Python function
normally.  It finds Tensor parameters reachable from the return value and
realizes them.  Therefore this works:

```python
@TinyJit
def add(a, b):
  return a + b             # no explicit .realize()
```

The return is auto-realized by the wrapper.  Explicit `.realize()` remains
useful when the function needs a boundary before its return, but it is not a
blanket return requirement.

The first call allows ordinary lazy initialization and one-time Python setup to
happen outside the recorded plan.  Its result must still be tested: if call 1
is wrong, capture has not caused that error.

### Call 2: capture

TinyJit prepares inputs again, creates an empty list for captured linears, and
places itself in the global `capturing` list.  The original Python function runs
a second time.

When a returned or explicitly realized Tensor reaches scheduling, the scheduler
still builds and resolves a `LINEAR`.  The behavior changes at the execution
boundary:

```text
ordinary realization:
  schedule → resolve → memory-plan → run the LINEAR

capture realization:
  schedule → resolve → append LINEAR to TinyJit → return an empty LINEAR
```

The ordinary execution path therefore does not run each just-recorded plan.
After the Python function returns, TinyJit realizes returned Tensors, which
records any remaining required plans, then clears capture mode.

TinyJit next:

1. rejects an empty capture with `JitError("didn't JIT anything!")`;
2. rejects unsupported non-Tensor return values;
3. flattens all captured `LINEAR` children into one ordered `LINEAR`;
4. optionally separates input-independent calls with `prune=True`;
5. replaces captured explicit input-buffer UOps with numbered `PARAM`s;
6. memory-plans the combined calls while protecting buffers reachable from
   live Tensors;
7. compiles each kernel, using `JITBEAM` when configured;
8. under `JIT=1`, asks graph splitting to batch supported calls;
9. constructs `CapturedJit(ret, linear, names, input_info)`; and
10. executes that prepared capture once.

The returned value for call 2 therefore comes from executing the new captured
plan.  It is not merely the lazy object created while Python was recording.

### Call 3 and later: replay

TinyJit prepares the new inputs but does not call the original Python function.
It compares argument names and the recorded input descriptions, substitutes
the new concrete buffers and symbolic values for parameters, and executes the
prepared plan.

```text
new call
  │
  ├─ discover/prepare Tensor and symbolic inputs
  ├─ validate names, views, symbolic variables, dtype, device
  ├─ bind current base buffers and current variable values
  ├─ execute already compiled/linked LINEAR with run_linear(jit=True)
  └─ return the stored capture-time return object
```

The Python function body runs twice in total: once for ignore and once for
capture.  A counter or print statement inside it does not run on replay.

### A state diagram worth memorizing

```text
new _TinyJit (cnt=0, captured=None)
        │ call 1: input preparation + Python + ordinary execution
        ▼
ignore complete (cnt=1, captured=None)
        │ call 2: input preparation + Python + record + lower + execute
        ▼
capture complete (cnt=2, captured=CapturedJit)
        │ call 3+: input preparation + validate/bind + execute capture
        └──────────────────────────────────────────────────────────────► replay
```

`reset()` returns a function-backed `_TinyJit` to the top state.  It does not
turn the next call directly into capture; the next call is another ignore call.

## Follow the captured `LINEAR` precisely

### Frontend Python and captured calls are different artifacts

The decorated Python function is not serialized.  TinyJit records the ordered
plans produced by realization.  A Python loop with a fixed iteration count may
construct several operations which become one captured plan.  A Python branch
chooses which operations are present; the branch itself does not become a
runtime branch merely because the function is decorated.

The artifact before JIT lowering can conceptually look like:

```text
LINEAR(
  CALL(SINK, BUFFER output0, BUFFER input0),
  CALL(SINK, BUFFER output1, BUFFER output0))
```

`jit_lower` substitutes explicit input bases with `PARAM(slot=...)`, memory
plans internal buffers, and compiles `SINK` bodies into `PROGRAM` bodies:

```text
LINEAR(
  CALL(PROGRAM, BUFFER temporary, PARAM input_slot_0),
  CALL(PROGRAM, BUFFER output, BUFFER temporary))
```

The exact buffers and number of calls depend on fusion, realization boundaries,
and the backend.  The important invariant is that `PARAM` marks a place where
replay can supply a new compatible input UOp.

### `_linear` versus `linear`

`CapturedJit._linear` stores the result of JIT lowering.  Its `linear` property
is a `functools.cached_property` which calls `link_linear(_linear)` on first
access.  This is **lazy linking relative to constructing `CapturedJit`**.

There is a subtle timing detail: the capture call immediately invokes the new
`CapturedJit`.  Computing its pure-write set accesses `self.linear`, so in the
ordinary path the linked property is already cached by the time call 2 returns.
With `HCQ2=0`, `link_linear` is an identity and `captured.linear is
captured._linear`.  With experimental `HCQ2=1`, linking can transform the plan.

Do not use “lazy” to claim that no linking occurred during call 2.  It means the
property defers linking until the capture is first executed, which call 2 does.

### Why replay calls `run_linear(..., jit=True)`

Chapter 12 established this contract:

```python
run_linear(plan, input_uops=current_inputs, jit=True)
```

means that `plan` is already compiled and linked.  `run_linear` resolves its
`PARAM`s and executes each call; it skips its normal `compile_linear` and
`link_linear` step.  The flag does not select ignore/capture/replay.  The
`_TinyJit` counter has already selected replay before this call occurs.

This naming collision is a common source of confused bug reports:

| Phrase | Correct meaning here |
| --- | --- |
| TinyJit capture | `_TinyJit` executes Python while realization records plans |
| JIT lowering | Parameter substitution, memory planning, compilation, optional graph splitting |
| `run_linear(jit=True)` | Execute a plan assumed already compiled/linked |
| Runtime/compiler JIT | A backend or driver may compile/load code; a separate concern |

## The input contract, from discovery to validation

TinyJit does not accept “anything Python can pass.”  It recognizes a narrow set
of replaceable inputs and stores the structural facts needed for safe replay.
Understanding the sequence explains both accepted calls and protective errors.

### Step 1: discover direct Tensor inputs

`_prepare_jit_inputs(args, kwargs)` first considers top-level positional and
keyword values whose exact class is `Tensor`:

```text
positional Tensor: name is 0, 1, 2, ...
keyword Tensor:    name is its keyword string; keyword items are sorted by name
```

The exact-class check is `t.__class__ is Tensor`, not recursive duck typing.
This prevents arbitrary model objects and Tensor-like wrappers from silently
becoming replay parameters.

The recorded names make calling convention part of the contract.  Capturing
`f(a, b)` and replaying `f(a=a, b=b)` fails because `[0, 1]` differs from
`['a', 'b']`.  **Direct Tensor kwargs** are sorted by name, so reversing the
textual order of the same direct Tensor keyword names is accepted.  That
guarantee does not extend to Tensors found inside containers; Step 2 explains
their different ordering rule.

### Step 2: inspect containers exactly one level deep

For every top-level argument value, TinyJit checks:

- elements if it is a `tuple` or `list`; or
- values if it is a `dict`.

Exact Tensor elements not already present by object identity are appended as
inputs.  The walk is deliberately **shallow**.  It does not descend into a list
inside a list, a dict inside a tuple, a dataclass, a model, or an arbitrary
iterator.

There are two easy-to-miss ordering details in the actual loop:

- positional argument values are inspected first;
- keyword **values** are then inspected in the caller's insertion order, not
  sorted keyword-name order.

Container-discovered Tensors are appended only to the input-info list; they do
not add their outer keyword names to `expected_names`.  Consequently, changing
the order of two container-valued kwargs can silently swap two structurally
identical replay slots instead of raising a names mismatch:

```python
@TinyJit
def subtract_boxes(left, right):
  return (left[0] - right[0]).realize()

subtract_boxes(left=[Tensor([10])], right=[Tensor([1])])  # ignore:  9
subtract_boxes(left=[Tensor([10])], right=[Tensor([1])])  # capture: 9
subtract_boxes(right=[Tensor([2])], left=[Tensor([20])])  # replay: -18, not 18
```

On replay, discovery sees `right`'s Tensor first and binds it to the captured
`left` slot.  Python never runs to look the values up by keyword name.  This is
a documented current footgun, not a calling convention to exploit.  Keep the
outer and inner insertion order stable, or pass semantically distinct inputs as
direct Tensor kwargs so their names are part of the contract.

```python
f(x, [y])       # x and y can be captured inputs
f(x, [[y]])     # x is discovered; y is not discovered through the nested list
```

This is more dangerous than a clean rejection: Python executes the nested
lookup on the capture call, so the captured plan can retain capture-time `y`.
Replay then ignores a new nested `y` because the function body is not called.
The contract lab demonstrates `[2.0, 4.0, 4.0]` for ignore, capture, and replay.

Flatten model inputs yourself into direct or one-level Tensor values.  Do not
rely on an undocumented recursive walk.

### Step 3: flatten sharded inputs and reject virtual UOps first

An `UNSHARD` input contributes its first source UOp.  Other Tensors contribute
their current `uop`.

TinyJit then checks `u.is_virtual` **before** trying to realize anything.  In
this snapshot, a UOp is virtual when it has no device or has a weak dtype.  Such
a UOp does not identify storage with a concrete device and width, so TinyJit
cannot replace its base buffer during replay.  It raises:

```text
JIT inputs must be real buffers; use .clone()
```

A device-less scalar constant is the canonical example.  Calling `.realize()`
on that virtual constant does not help: `Tensor.realize` deliberately excludes
virtual UOps.  A clone onto a real device with a concrete dtype creates storage
that can become an input.

### Step 4: auto-realize ordinary lazy inputs

Do not merge *virtual* with *unrealized*.  A normal lazy expression can have a
real device and concrete dtype but no allocated result yet:

```python
base = Tensor([1.0, 2.0], device="PYTHON", dtype=dtypes.float32).realize()
lazy = base + 1.0

assert not lazy.uop.is_virtual
assert not lazy.uop.is_realized
```

After the virtual check passes, `_prepare_jit_inputs` calls `Tensor.realize` on
such ordinary unrealized input Tensors.  The lab observes the state changing
from `(virtual=False, realized=False)` to `(False, True)` before the decorated
function runs.

The accurate rule is therefore:

```text
device-backed, concrete-dtype lazy Tensor → may be auto-realized and accepted
device-less or weak-dtype virtual UOp     → rejected before auto-realization
```

“All unrealized inputs fail” is incorrect for this snapshot.

### Step 5: reduce each input to a realized base buffer

TinyJit collects the base UOp of every input whose base has realized storage.
Views which differ in shape, offset, strides, or mask can share this same base.
The realized-storage qualification is consequential: an allocation-shaped UOp
such as a fresh `Tensor.empty` can already have buffer identity while its base
still has no realized storage.  `Tensor.realize` skips an object which already
has buffer identity, so that base can remain absent from `input_buf_uops`.

Such a Tensor still contributes an input description, and replay may accept a
structurally matching replacement, but it is not turned into a `PARAM`.  The
captured plan can therefore retain the capture-time allocation while ignoring
the replacement.  Duplicate detection also operates only on the collected
realized bases.  Do not use a fresh, unallocated `Tensor.empty` argument as
evidence that caller-provided destination storage is rebound on replay; realize
and test the intended destination contract explicitly.

If the same base appears twice, TinyJit rejects the entire call with:

```text
duplicate inputs to JIT
```

The obvious failure is `f(x, x)`.  Two different view wrappers over the same
base also fail:

```python
base = Tensor([1.0, 2.0, 3.0]).realize()
f(base[:2], base[1:])       # two inputs, one base allocation → rejected
```

The issue is base-buffer aliasing, not Python object equality.  Cloning one
input to independent storage can remove the alias when that is semantically
correct.  Do not clone blindly if the intended computation depends on aliasing.

### Step 6: store an unbound structural description

For each input, TinyJit replaces the base with a `NOOP`, cleans up movement
UOps, unbinds symbolic values, and records:

```text
(unbound view UOp, sorted symbolic Variables, dtype, device)
```

It separately records the direct positional/keyword Tensor names.  The view
UOp carries more than a shape tuple: movement structure, offset, strides, and
mask can matter.  That is why two equal-size slices with different offsets can
fail replay.

On call 3+, both must match:

```text
expected_names       == newly discovered names
expected_input_info  == newly constructed structural descriptions
```

A new Tensor wrapper and a new base allocation are expected.  The structure,
dtype, and device must match; the base address itself is the parameter that is
allowed to change.

### Step 7: carry symbolic values separately

Unbinding a view produces both a structure containing Variables and a mapping
from those Variables to current integer values.  Top-level positional or
keyword `UOp` arguments also contribute `Variable.bind(...)` values.  TinyJit
merges those bindings into `var_vals` and supplies them at replay.

This supports a bounded symbolic value changing while the unbound expression,
Variable identities/bounds, dtype, device, and captured call structure remain
compatible.  It does not make arbitrary Python shapes dynamic, and it does not
turn a Python integer into a symbolic parameter.

Graph runners additionally locate which compiled scalar argument slots and
launch dimensions depend on these Variables so a backend graph can update them.
Plain replay supplies the same `var_vals` to ordinary program calls.

### Input outcomes at a glance

| Input change | Current outcome | Reason |
| --- | --- | --- |
| New Tensor with a realized base, same view/dtype/device | Accepted | The realized base buffer is a replay parameter |
| Fresh buffer-identity Tensor whose base storage is still unrealized, such as some `Tensor.empty` inputs | Structurally described, but its base is omitted from replay parameters | Only bases with `base.realized is not None` enter `input_buf_uops` |
| Ordinary lazy Tensor with device and concrete dtype | Auto-realized, then checked | It can acquire real storage |
| Device-less or weak-dtype virtual UOp | Rejected before realization | No replaceable concrete buffer identity |
| Same Tensor with a realized base in two direct argument positions | Rejected | Duplicate realized base buffer |
| Two different views sharing one realized base | Rejected | Duplicate realized base buffer |
| Same direct Tensor keyword names in different order | Accepted | Direct Tensor kwargs are sorted by name |
| Structurally identical Tensors inside reordered container-valued kwargs | May silently bind in the wrong order | Container discovery uses caller insertion order and records no names |
| Positional capture, keyword replay | Rejected | Recorded input names differ |
| Same shape but different view offset | Rejected | Unbound view structure differs |
| Different dtype or device | Rejected | Recorded dtype/device differs |
| Tensor one level inside list/tuple/dict | Discovered | Explicit shallow-container support |
| Tensor nested two or more levels | Not discovered | Input walk is intentionally non-recursive |

## Python values specialize the capture

### Scalars are not replay parameters

Ordinary `bool`, `int`, `float`, strings, enums, and other Python objects do not
enter `input_buf_uops` or `var_vals`.  They can influence the two Python runs,
but replay does not call Python again.

```python
@TinyJit
def choose(x, square):
  if square:
    return (x*x).realize()
  return (x*2).realize()
```

If call 1 passes `True`, call 2 passes `False`, and call 3 passes `True`, the
results are square, double, double.  The False branch selected during capture
is the plan that replays.  TinyJit does not reject the changed bool because the
bool was never recorded as a contract input.

Treat a Python scalar as a **specialization constant**.  If it must change:

- create and retain a separate TinyJit specialization for each meaningful
  scalar configuration;
- call `reset()` and recapture after the configuration changes;
- express data selection as Tensor operations when both paths have a stable
  captured shape; or
- use supported symbolic UOps for bounded integer dimensions rather than an
  arbitrary Python branch.

Which choice is correct depends on whether the value changes computation
structure or only data.

### Closure values freeze the same way

```python
scale = 1.0

@TinyJit
def f(x):
  return (x * scale).realize()
```

Changing `scale` from 1 on ignore, to 2 on capture, to 3 on replay yields
`10, 20, 20` for input 10.  Capture built multiplication by 2.  A Tensor stored
in a model or closure is also not automatically a direct input; if its buffer
must be replaced, pass it explicitly.  If it is persistent model state, ensure
its realization and mutation semantics are intentional.

A particularly subtle instance-method footgun follows from Python descriptors:
decorating a method at class definition time places one `_TinyJit` on the
class.  Different instances can therefore share that capture.  Instance-specific
state read through `self` may be frozen from the instance used for capture.

### Side effects execute twice, not on every call

Logging, counters, list appends, random calls performed by Python rather than
captured Tensor operations, and other Python side effects happen on ignore and
capture.  They stop on replay.  Do not put required bookkeeping in the
decorated function unless it is represented as captured Tensor/runtime work or
performed outside the TinyJit wrapper.

Tensor random operations are a different case: their state updates can be part
of the captured computation, and upstream tests require random values to
regenerate.  “Python does not run” does not mean every numeric result is fixed.

### Host reads are rejected during capture

`.item()`, `.tolist()`, `.numpy()`, and other data access eventually need a
buffer read on the host.  During capture, `Tensor._buffer` raises:

```text
cannot access tensor data during JIT capture, the value will be baked in
```

The first ignore call may succeed, then the second capture call fails.  Without
the guard, Python could read one capture-time value and construct a permanently
specialized plan without an explicit contract.

Use static-size forms of data-dependent operations where the API supports
them.  For example, current tests exercise `masked_select(..., size=...)` and
`nonzero(size=...)`; their fixed output size avoids a host read to determine an
arbitrary Python allocation size during capture.

### Nested TinyJit is not supported

An outer and inner decorated function can appear to work on the outer ignore
call because no capture is active.  On call 2, the outer capture encounters an
already active `capturing` list when the inner tries to capture and raises a
runtime error.  Flatten the intended capture boundary or keep the inner helper
as an ordinary function called inside one outer TinyJit.

## Return values and persistent output storage

### What may be returned

On the capture call, `_check_no_non_tensor_return` recursively accepts:

- `None`;
- a `Tensor`; and
- nested `tuple`, `list`, or `dict` values whose values eventually contain only
  accepted objects.

It rejects an ordinary Python scalar, string, or custom object inside that
structure.  Dictionary values are checked recursively.  This fails on call 2:

```python
@TinyJit
def bad(x):
  return (x + 1).realize(), 7
```

Call 1 can appear to return the tuple successfully because return-type
validation occurs during capture.  Always drive a new TinyJit through at least
three calls in a test.

`None` being allowed does not mean an empty function can be captured.  At least
one realization must have supplied a `LINEAR`, perhaps for a mutation side
effect.  Otherwise the earlier “didn't JIT anything” error wins.

### The stored return object is reused

`CapturedJit` stores the exact `ret` produced while Python ran for capture.  Its
`__call__` executes the plan, then returns `self.ret`.  It does not reconstruct
new Tensor wrappers on every replay.

For the common case where the captured plan writes a dedicated output buffer,
as `add_one` does in Lab 1:

```text
call 1 output wrapper: A, ordinary storage
call 2 output wrapper: B, captured-plan storage
call 3 output wrapper: B, same captured-plan storage with new contents
call 4 output wrapper: B, same captured-plan storage with newer contents
```

Therefore reading a reference saved from call 2 after call 3 can show call 3's
value.  This is observable in the first lab:

```text
immediate call-2 value: [11.0, 12.0]
same call-2 reference after replay: [21.0, 22.0]
```

This behavior is excellent for a training loop that consumes an output before
the next step.  It is wrong for code that expects every returned Tensor to be a
historical snapshot.

When a snapshot is required, copy at the ownership boundary:

```python
saved = jitted_step(x).clone().realize()
```

The `clone` creates independent storage.  `realize` submits/materializes the
copy before the next replay is submitted, so the normal ordered dependency path
can copy from the captured source before that source is overwritten.  It does
**not** necessarily mean the accelerator has completed the copy when Python
returns.  Synchronize when host ownership, another queue, or another thread
requires completion rather than merely ordered submission.

Do not infer safe retention merely because two output wrappers printed
differently at the time they were first observed.  Check object and base-storage
identity or use an explicit copy.

Wrapper reuse is universal for replay because `CapturedJit` returns `self.ret`;
“the stored output buffer is overwritten with the new result” is not universal.
If the capture-time return wrapper aliases an input rather than a persistent
plan output, replay can mutate the current input but still return the old
capture-time input wrapper.  For example, calling a read/write `x += 1; return
x.realize()` JIT with a *fresh* `x` each time can mutate the third input while
returning the second call's wrapper.  Test both the current caller-owned object
and the returned object when an API returns an input or destination alias.

## Mutation and the pure-write rule

Mutation requires more precision than “TinyJit copies written inputs.”

For every captured call, tinygrad obtains output and input argument-slot roles
from the call's program/copy metadata.  `CapturedJit._written_uops` collects an
argument only when its slot is in:

```text
outputs − inputs
```

and the argument is a `BUFFER` or `SLICE`.  This is a **pure-write** set: the
call writes the buffer and does not read it.  A buffer appearing in both input
and output roles is deliberately excluded.

Before execution, `CapturedJit.__call__` replaces a current input UOp which is
in that pure-write set with `_copy_input(u)`, an explicit device copy to a new
buffer.  It then passes the resulting concrete input tuple to
`run_linear(..., jit=True)`.

### Read/write input mutation

The upstream behavior test uses the same realized Tensor repeatedly:

```python
@TinyJit
def increment(x):
  x += 1
  x.realize()
  return x
```

The values can advance `1, 2, 3, ...` through replay.  Because the program both
reads and writes `x`, it is not in `outputs − inputs` and is not protected by
the pure-write copy rule.  This is intentional mutation of caller-visible
state.

### When `_copy_input` actually runs

The copy rule is conditional on **UOp identity**.  Merely passing an argument
which the Python function uses as a write-only destination does not establish
that the new destination is copied or rebound.  `CapturedJit.__call__` copies a
current explicit input only when that input UOp is also a `BUFFER` or `SLICE` in
the captured plan's pure-write set.

A direct way to create that identity overlap is output feedback:

```python
@TinyJit
def add_one(x):
  return (x + 1).realize()

add_one(Tensor([0]))             # ignore
captured_output = add_one(Tensor([10]))  # capture; result is 11
add_one(captured_output)         # replay; result is 12
```

The capture-time result buffer is a pure output of the captured program, so it
is in `_written_uops`.  Passing that exact buffer as the replay input would make
the program read from storage it is also about to overwrite.  Tinygrad first
calls `_copy_input(captured_output.uop.base)`, binds the copy to the input
`PARAM`, and leaves the persistent captured output buffer as the output.  The
contract lab instruments this implementation-private helper and observes one
copy for the feedback call and no copy for a fresh non-aliasing input.

This is the rule exercised by upstream feedback and multiple-output alias tests.
It is not a promise that arbitrary new write-only destination arguments are
copied.  In particular, a fresh unallocated `Tensor.empty` can be structurally
accepted yet omitted from `input_buf_uops`, as Step 5 explained.  Replay then
cannot substitute it at all and may keep writing capture-time storage.

The stable source facts are:

- pure-write classification is `outs - ins` and only retains `BUFFER`/`SLICE`
  arguments;
- a current explicit input is copied only when its UOp is in that set;
- read/write arguments are not included by this rule; and
- TinyJit returns its stored capture return object, not necessarily a new input
  or destination wrapper supplied by the caller.

If an API promises that a caller-provided destination object itself is mutated
and returned on every call, test ignore, capture, plain replay, and device-graph
replay explicitly.  Use a realized destination, check that its base became a
`PARAM`, and trace `STORE`/`AFTER`, call argument roles, base aliases, and output
wrapper identity.

### Why graphs make alias tests stricter

A plain ordered `LINEAR` launches calls in order.  A device graph additionally
constructs dependency edges from buffer reads and writes.  An incorrect role,
offset, or alias can produce a missing edge or an intra-kernel hazard.  A
mutation test that passes under `JIT=2` but fails under `JIT=1` first points
toward the `graph_split_rewrite` path, not automatically toward the Tensor
frontend.  That pass also omits `SLICE` calls even when no backend graph object
is formed.  Confirm that the post-split plan actually contains a
`CUSTOM_FUNCTION(arg="graph")` before narrowing the cause to graph dependency
construction or backend graph parameter updates.

## Pruning, reset, and freeing are three different operations

### `TinyJit(..., prune=True)`

Capture normally retains every recorded call.  `prune_linear` begins with the
explicit input base buffers as `needed`.  It walks calls in execution order:

- if a call touches a currently needed buffer, keep it and add all of its
  buffers to the needed set;
- otherwise, place it in a one-time `LINEAR`.

TinyJit executes the one-time plan during capture, then lowers only the kept
input-connected calls for replay.  A common use is preprocessing a persistent
weight once before combining it with a changing input.

This is a buffer-connectivity rule, not a general semantic optimizer or dead
code eliminator.  Enable it only with a test that proves numerical equivalence
and the intended captured call count.  A hidden changing value which was not
recognized as an input can make “one-time” the wrong classification.

In particular, `var_vals` does not seed `prune_linear`'s `needed` set.  A
top-level bound symbolic UOp can update replay scalar slots, but that fact alone
does not keep preprocessing which is disconnected from every explicit Tensor
input buffer.  The contract lab makes a weight-preparation call depend on a
changing symbolic scale and observes `11, 21, 21`: ignore uses scale 1, capture's
one-time work uses scale 2, and replay cannot apply scale 3.  Pass a changing
Tensor buffer through that work or deliberately recapture; do not assume a
symbolic binding proves pruning reachability.

### `reset()`

`reset()` sets `cnt=0` and `captured=None` while preserving the original
function.  Use it when a legitimate specialization constant, model structure,
or capture boundary changes.  The next two calls are again ignore and capture.

A `_TinyJit` reconstructed from only a pickled `CapturedJit` has no original
function and cannot reset.  The method asserts that `fxn` exists.

### `captured.free_intermediates()`

This method keeps the capture description but releases selected runtime state:

- it removes cached graph-runner objects for top-level graph calls; and
- it deallocates initialized buffers found through the pure-write set, plus an
  eligible base when it has no allocated views.

The next replay can recreate/reallocate what it needs.  Upstream tests call the
method twice and replay again.  It is not equivalent to `reset`: Python does
not run again and the input contract is unchanged.  It is also not a promise
to free every allocation reachable from a capture; live Tensor ownership,
views, allocator caching, and backend retention still matter.

| Operation | Keeps captured structure? | Next call runs Python? | Main purpose |
| --- | --- | --- | --- |
| `prune=True` at construction | Builds a smaller replay plan | Only normal ignore/capture phases | Move input-disconnected recorded calls to one-time capture work |
| `reset()` | No | Yes, as a new ignore call | Recapture a changed specialization/structure |
| `free_intermediates()` | Yes | No | Drop graph runtime and eligible captured allocations |

## `JIT=0`, `JIT=1`, and `JIT=2`

At this snapshot:

| Setting | Original Python function | TinyJit capture/replay | Device-graph splitting |
| --- | --- | --- | --- |
| `JIT=0` | Runs on every call | Disabled | Disabled because there is no capture |
| `JIT=1` | Runs on ignore and capture | Enabled | The graph-splitting rewrite runs; actual graph batching requires backend/call support |
| `JIT=2` | Runs on ignore and capture | Enabled | Skipped in `jit_lower` |

Two edge details matter.

First, `_prepare_jit_inputs` runs before the `if not JIT` branch.  A decorated
function under `JIT=0` still performs TinyJit's narrow input discovery,
auto-realization, virtual-input rejection, and duplicate-base rejection.  The
original function runs every time, but the wrapper is not identical to calling
the undecorated function directly.

Second, `cnt` still increments under `JIT=0`.  Do not call one decorated object
several times with JIT disabled and then flip the global setting in the same
process.  Its counter can no longer describe a capture it never made.  Compare
`JIT=0/1/2` in separate processes, as the lab does.

`CAPTURING=0` is a more surgical experimental setting: scheduling during the
capture phase does not append those plans.  It can exclude deliberate setup
work, but a function with no remaining captured plan fails.  It is not the
normal first localization switch; start with `JIT=0`, `JIT=2`, and `JIT=1`.

## Device graphs, built from the beginning

### What graph replay means here

A device graph is a backend object representing several already compiled
operations and their dependencies.  Instead of asking the runtime to submit
each call afresh:

```text
host: submit A
host: submit B after A
host: submit C after B
```

the host instantiates a reusable graph once and later does roughly:

```text
update allowed input addresses/scalars/dimensions
submit graph(A → B → C)
```

The graph still launches work and the device still computes A, B, and C.  Its
possible benefit is less repeated host/driver command construction and more
backend knowledge of the dependency batch.

### Graph creation is conditional

With `JIT=1`, `jit_lower` calls `graph_split_rewrite`.  For each captured call:

1. `SLICE` view calls are omitted from the rewritten plan by this pass;
2. tinygrad finds the participating compiled devices;
3. the device must advertise a non-`None` `.graph` class;
4. that graph class must report that it supports the call and current batch;
5. the batch must respect the current size bound; and
6. a batch of one remains an ordinary call unless `GRAPH_ONE_KERNEL` is set.

A graphable batch becomes:

```text
CALL(
  CUSTOM_FUNCTION(arg="graph", src=(LINEAR(original calls),)),
  PARAMs used by the batch)
```

The generic `GraphRunner` supports `PROGRAM` calls on one device.  The
`MultiGraphRunner` marker also permits `COPY` calls when all devices have the
same backend type.  Concrete graph classes can impose stricter rules.

`JIT_BATCH_SIZE` supplies the initial maximum batch size; after a flushed graph
batch, this implementation doubles the bound.  Do not describe it as a fixed
number of kernels per graph without inspecting the current rewrite.

### Graph runtime creation and updates

The graph custom-function UOp is part of the captured `LINEAR`, but the backend
graph runtime is created on first execution and stored in a weak-key
`graph_cache`.  `GraphRunner.__init__` resolves the original calls and records:

- which call argument positions correspond to input `PARAM` slots;
- which compiled scalar positions depend on symbolic Variables;
- which global/local launch dimensions are symbolic;
- program runtime objects and concrete buffers; and
- read/write dependencies used by concrete graph implementations.

On replay, a concrete graph runner can patch the current input addresses,
current scalar values, and current launch dimensions before launching the graph.
For example, `CUDAGraph` updates encoded argument structures and CUDA graph node
parameters, then calls `cuGraphLaunch`.

### Backend support is evidence, not an assumption

`PythonDevice` calls `Compiled.__init__` without a graph class, so
`Device["PYTHON"].graph is None`.  The portable labs must show zero graph calls
under both `JIT=1` and `JIT=2`.  They prove TinyJit structure and numerical
replay, not accelerator graph behavior or asynchronous timing.

At this snapshot, a non-mock `CUDADevice` supplies `CUDAGraph`; HCQ-backed
devices supply their own graph route.  That source fact does not prove a
particular batch works on an RTX 4090.  A hardware claim requires the exact
device, renderer, driver/runtime environment, captured calls, `JIT` mode,
output oracle, and timing/synchronization method.

## Lab 1 — See ignore, capture, replay, and output reuse

This lab defaults to `DEV=PYTHON` when `DEV` is absent and respects an explicit
device so the repository runner can still use its hardware matrix.  The commands
below select `DEV=PYTHON`, and the lab fixes the debug, JIT-lowering, cache,
color, profiling, and safety settings which its assertions inspect before
importing tinygrad.  It also rejects optimized Python because its checks rely on
`assert`.  This is controlled structural evidence, not a claim that every
tinygrad environment variable has been neutralized.

Run from the **guide repository root**, not from your home directory.  Point
`TINYGRAD_STUDY` at the pinned tinygrad checkout:

```bash
cd /absolute/path/to/tinygrad_docs
TINYGRAD_STUDY=/absolute/path/to/tinygrad-study

DEV=PYTHON JIT=0 PYTHONPATH="$TINYGRAD_STUDY" "$TINYGRAD_STUDY/.venv/bin/python" labs/phase4/jit_three_calls.py
DEV=PYTHON JIT=1 PYTHONPATH="$TINYGRAD_STUDY" "$TINYGRAD_STUDY/.venv/bin/python" labs/phase4/jit_three_calls.py
DEV=PYTHON JIT=2 PYTHONPATH="$TINYGRAD_STUDY" "$TINYGRAD_STUDY/.venv/bin/python" labs/phase4/jit_three_calls.py
```

Each line launches a separate process.  There is no trailing backslash to copy
incorrectly, and no relative `scripts/run_labs.py` lookup from `/home/you`.

Before running, predict:

1. the Python counter after each call in each mode;
2. when `captured` changes from `None`;
3. whether the call-2 and call-3 output wrappers are identical;
4. what the saved call-2 reference reads after call 3; and
5. how many graph custom-function calls `PYTHON` can contain.

The important `JIT=1` excerpt is:

```text
call=1 phase=ignore          ... python_calls=1 ... captured=False
call=2 phase=capture         ... python_calls=2 ... captured=True
call=3 phase=replay          ... python_calls=2 ... captured=True
capture/replay returned the same Tensor wrapper: True
capture output reference now reads replay data: [21.0, 22.0]
captured call bodies before/after default linking: ['PROGRAM'] ['PROGRAM']
selected backend advertises a graph runner: False
device graph calls: 0 (a one-PROGRAM capture stays ungraphed here)
```

Interpret the last two lines carefully:

- `PROGRAM` is structural evidence that JIT lowering compiled the captured
  call;
- identical before/after objects are expected because the lab fixes `HCQ2=0`,
  making default `link_linear` an identity; and
- zero graph calls are consistent both with `PythonDevice.graph is None` and
  with the general one-call rule.  They do not imply graph splitting is broken.

`JIT=2` has the same visible one-program structure because this workload has
only one call and `PYTHON` has no graph support anyway.  The mode difference
becomes observable on a graph-capable backend with a graphable multi-call plan.

## Lab 2 — Exercise the actual contract

Run the second deterministic lab:

```bash
cd /absolute/path/to/tinygrad_docs
TINYGRAD_STUDY=/absolute/path/to/tinygrad-study
PYTHONPATH="$TINYGRAD_STUDY" "$TINYGRAD_STUDY/.venv/bin/python" labs/phase4/jit_contracts.py
```

This script uses only specific `JitError` catches; an unrelated exception is a
real failure.  It asserts all of these observations:

- a normal device-backed lazy input is auto-realized;
- a device-less virtual UOp is rejected before realization;
- a Tensor one level inside a list updates across replay;
- a Tensor two levels deep is not an input and freezes at capture;
- reordered container-valued kwargs can silently swap unnamed, structurally
  identical Tensor slots;
- a changed Python bool replays the capture-time branch;
- a changed closure scalar replays the capture-time value;
- two views of one base are rejected as duplicate inputs;
- dtype, view offset, and positional/keyword convention changes are rejected;
- a Python integer in a returned tuple is rejected on capture;
- `.item()` is rejected on capture;
- a read/write input mutates through replay and is absent from the pure-write
  set;
- `reset()` makes the next call a new ignore call;
- feeding a captured pure-output buffer back as an explicit input invokes
  `_copy_input`, while a fresh non-aliasing input does not;
- replay still works after `free_intermediates()`; and
- a symbolic value alone does not keep input-buffer-disconnected work under
  `prune=True`.

Do not merely look for the final “passed” line.  For each printed observation,
point to the stage of `_prepare_jit_inputs`, `_TinyJit.__call__`,
`CapturedJit.__call__`, or `prune_linear` that explains it.  The final two lines
also separate what the lab claims from what its footgun outputs and `PYTHON`
backend cannot establish.

## Optional hardware workflow — prove graphing separately

The portable labs intentionally cannot prove a CUDA graph claim.  On the Ubuntu
RTX 4090 route, make a separate minimal probe with an explicit realization
between two computations:

```python
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.uop.ops import Ops

@TinyJit
def two_stage(x):
  middle = (x + 1.0).contiguous().realize()
  return (middle * 2.0).contiguous().realize()

for value in (1.0, 2.0, 3.0):
  out = two_stage(Tensor([value] * 1024, dtype=dtypes.float32, device=Device.DEFAULT).realize())
  assert out[0].item() == (value + 1.0) * 2.0

assert two_stage.captured is not None
for call in two_stage.captured._linear.src:
  body = call.src[0]
  print("top-level:", body.op.name, body.arg if body.op is Ops.CUSTOM_FUNCTION else "")
  if body.op is Ops.CUSTOM_FUNCTION and body.arg == "graph":
    print("graph members:", [member.src[0].op.name for member in body.src[0].src])
```

Run the probe in separate fresh processes under `DEV=CUDA JIT=1` and
`DEV=CUDA JIT=2`.  Fix all other optimization settings and record them.  Your
questions are:

- Under `JIT=2`, where graph splitting is skipped, did both boundaries remain
  two top-level `PROGRAM` calls?
- Under `JIT=1`, did the top level become a `CUSTOM_FUNCTION` whose `arg` is
  `graph`, and does its inner `LINEAR` contain the two `PROGRAM` members?
- Were all three numerical results correct?
- Does the selected `Device` actually advertise a graph class?

`captured._linear` is already the result of `jit_lower`; under `JIT=1` it is a
**post-split** artifact, not the captured pre-graph plan.  The graph custom
function retains its original batch in an inner `LINEAR`, while `JIT=2` provides
the no-split comparison.  If either route shows only one program, the probe did
not establish a multi-call graph case.  Inspect the artifacts and strengthen
the realization boundary instead of asserting that a graph must exist.

Use `VIZ=1` only after the textual probe works.  The rewrite stages named “View
captured linear” and “View graphed linear” show the plan before and after graph
splitting.  A visualization is evidence about structure, not timing.

## Timing without lying to yourself

### Three intervals, three questions

Keep these intervals distinct:

```text
capture time
  Python + scheduling + memory planning + compilation + optional graph creation

steady-state host submission time
  validate/bind inputs + patch/submit ordinary calls or graph; may return early

device completion time
  how long until the device has completed the submitted work
```

Timing the first three calls together mostly measures initialization and capture.
It does not answer steady-state replay cost.

### A sound asynchronous measurement shape

For a graph-capable accelerator:

1. pre-create and realize a pool of compatible inputs;
2. drive the TinyJit through ignore and capture;
3. replay several untimed times;
4. synchronize the device to start from an empty queue;
5. start a host timer;
6. issue a fixed number of replay calls without `.item()`, `.tolist()`,
   `.numpy()`, or synchronization in the loop;
7. stop the host timer: this is a host-side issue interval;
8. synchronize once;
9. stop a second timer: this includes completion of the issued device work; and
10. validate outputs outside the timed interval.

The host interval still includes TinyJit's Python wrapper and input binding;
that is often the intended quantity.  It is not kernel duration.  For device
time, use backend events or a profiler which records device timestamps.

### `DEBUG=2` changes waiting behavior

`run_linear` constructs its execution context with:

```text
wait = explicit_wait or DEBUG >= 2
```

Thus `DEBUG=2` is useful for synchronized per-call diagnostics, but it changes
the asynchronous behavior being measured.  Do not compare a `DEBUG=2` JIT run
against a `DEBUG=0` run and attribute the entire difference to graphing.

For a graph comparison, use the same workload, inputs, warmup, debug level,
cache state, synchronization points, and correctness oracle in separate
`JIT=1` and `JIT=2` processes.  Record at least:

| Evidence | Why it is needed |
| --- | --- |
| Pre-graph captured call count/types | Shows what graph splitting was offered |
| Post-split call count/types | Shows whether a graph was actually formed |
| Device/backend/renderer/driver | Makes the graph implementation reproducible |
| Host issue interval | Tests submission overhead |
| Device-event or synchronized completion interval | Tests device work/critical path |
| Numerical oracle | Prevents a fast wrong graph from looking successful |

On `DEV=PYTHON`, program execution is synchronous Python interpretation.  Its
wall time is useful for deterministic structure testing but cannot establish
asynchronous submission savings or CUDA graph performance.

## Failure localization by phase

Start with the earliest phase that differs.  Do not read the entire JIT file
for every symptom.

| Observation | First boundary to investigate | Useful next comparison |
| --- | --- | --- |
| Undecorated function and call 1 are wrong | Frontend, schedule, codegen, runtime | Reproduce without `TinyJit` |
| Call 1 right, call 2 raises while reading data | Capture-time host read or unsupported dynamic Python | Find `.item()`/`.tolist()`/implicit `_buffer` |
| Call 1 right, call 2 captures no calls | Realization boundary or `CAPTURING` | Confirm returned/side-effect Tensors produce a LINEAR |
| Call 1 right, call 2 numerical result wrong | Combined recorded linears, parameter substitution, memory plan, compilation | Inspect “captured linear”; try simple non-graph route |
| Call 2 right, call 3 rejects | Names or view/Variable/dtype/device contract | Print expected versus current input info; minimize one changed fact |
| Call 2 right, call 3 silently follows old Python branch | Ordinary scalar/closure was never an input | Tensorize, symbolize, specialize, or reset deliberately |
| Call 2 right, retained old output changes after call 3 | Stored return wrapper/storage reuse | Clone and realize at the ownership boundary |
| Read/write state fails only after replay | Buffer roles, aliasing, STORE/AFTER, input substitution | Compare explicit bases and call outs/ins |
| `JIT=2` correct, `JIT=1` wrong | First the graph-splitting rewrite; only then a backend graph if one was actually formed | Inspect post-split calls for omitted `SLICE`s and a graph custom function; inspect graph dependencies/patching only when present |
| Both `JIT=1` and `JIT=2` wrong on replay | Plain captured plan, binding, mutation, output reuse | Inspect PARAM slots and replay inputs before graph code |
| Only a symbolic value fails | Variable bindings, view expression, scalar slot, launch dimensions | Compare unbound view/Variable set and `var_vals` |
| Replay works until `free_intermediates()` | Reallocation, graph-cache recreation, view/base ownership | Inspect pure-write buffers and allocator state |
| Nested decorated function fails on call 2 | Active capture nesting | Use one capture boundary |

### A disciplined five-run matrix

For a new JIT bug, prefer these small runs in fresh processes:

```text
1. undecorated function
2. decorated, JIT=0
3. decorated, JIT=2
4. decorated, JIT=1
5. smallest backend-independent DEV=PYTHON form, when semantics permit
```

This matrix separates wrapper/input-preparation behavior, capture/replay,
device graphing, and backend-specific execution.  It does not automatically
prove the cause, but it makes the first divergent boundary visible.

## Question-led source stops

Do these stops after the labs.  Each link includes enough surrounding code to
answer a concrete question; do not begin at an isolated import or declaration.

### Stop 1: What does the schedule cache reuse?

Read [`lower_sink_to_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L107-L135).

Question: which object supplies the key, what call is avoided on a hit, and
which later work is visibly outside this function?  Answer before proceeding:
the cache can reuse the scheduled form, but this range does not execute the
model's Python or the resulting runtime calls.

### Stop 2: Where does realization become capture?

Read all of [`create_linear_with_vars`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L170-L199).

Trace these local names: `linear_call`, `linear`, `used_vars`, and `var_vals`.
Then compare the two returns.  Why does the capture branch append the fully
resolved `linear` but return an empty `Ops.LINEAR`?  What ordinary memory-plan
step is bypassed at that moment and performed later by JIT lowering?

### Stop 3: Why are virtual and lazy inputs different?

Read [`UOp.device` and `UOp.is_virtual`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/uop/ops.py#L851-L870),
then the complete [`Tensor.realize`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L190-L196),
then the complete [`_prepare_jit_inputs`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L202-L220).

Write the order of checks on paper.  Which line rejects virtual inputs?  Which
later line auto-realizes normal lazy Tensors?  Which line reduces views to base
buffers before duplicate detection?  This stop corrects the inaccurate rule
that every unrealized input fails.

### Stop 4: Which exact call chooses ignore, capture, or replay?

Read `_TinyJit` from [`__init__` through `__call__`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L222-L289).

For each branch, list whether it calls `self.fxn`, realizes returned parameters,
creates `_linears`, validates the return, constructs `CapturedJit`, or compares
the expected input contract.  Notice that input preparation precedes all three
branches and counter increment follows them.

### Stop 5: What transformations make a plan replayable?

Read [`jit_lower`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L67-L77)
with one lab `captured._linear` in mind.

Locate, in order, input substitution, memory planning, compilation, and graph
splitting.  Why is graph splitting conditional on `JIT < 2`?  Where would you
place an artifact dump to distinguish the captured pre-graph plan from the
post-graph plan without changing semantics?

If you plan to use pruning, also read all of
[`prune_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L16-L24).
Which set seeds reachability?  Notice that the function receives buffer UOps,
not `var_vals`; connect that fact to the symbolic-scale observation in Lab 2.

### Stop 6: What is graphable, and when is one call left alone?

Read [`create_graph_call` and `graph_split_rewrite`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L26-L61),
then [`GraphRunner.supports_uop` and `MultiGraphRunner.supports_uop`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L147-L161).

Question: how are external `PARAM`s selected for the graph call?  Which body
operation is skipped?  Which backend property gates graphing?  Why does a
supported single kernel normally remain a `PROGRAM` call?

### Stop 7: What can a graph update?

Read [`GraphRunner.__init__` and its update generators](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L91-L145).

Find the three maps/lists corresponding to input buffer positions, scalar
Variable values, and launch dimensions.  Then, only for a physical CUDA issue,
read the complete [`CUDAGraph.__call__` and destructor](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/runtime/graph/cuda.py#L47-L74)
and identify where each category is patched before launch.

### Stop 8: Why does output reuse happen?

Read the complete [`CapturedJit`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/jit.py#L165-L200).

Question: what object does `__call__` return?  Where is linking cached?  How is
the pure-write set calculated?  Which exact buffers can
`free_intermediates()` deallocate?  Compare your answers with the wrapper and
mutation sections of both labs.

### Stop 9: Why does prepared replay not compile again?

Read [`compile_linear`, `link_linear`, and `run_linear`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L268-L281).

State the behavior of `jit=False` and `jit=True` without using the phrase
“turn JIT on.”  Also locate why `DEBUG>=2` changes `wait`.

### Stop 10: Which tests are contracts and which are documented footguns?

Read the focused ranges, not the whole suite at once:

- [output reuse, virtual inputs, pruning, and nested capture](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_jit.py#L315-L424);
- [stored-output reuse and feedback/multiple-output alias copying](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_jit_footguns.py#L30-L120);
- [input, scalar, closure, side-effect, and host-read footguns](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_jit_footguns.py#L152-L362); and
- [correct lazy returns, keyword ordering, and input mutation](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/test/unit/test_jit_footguns.py#L365-L410).

For each behavior, label it one of: supported contract, protective rejection,
current footgun, or backend-specific requirement.  A test named “footgun” can
deliberately preserve evidence of surprising current behavior; do not turn its
assertion into a general design recommendation.

## Exercises

### 1. Predict the phase, call counter, and result

```python
calls = 0

@TinyJit
def f(x, choose_add):
  global calls
  calls += 1
  if choose_add:
    return (x + 3).realize()
  return (x * 3).realize()

print(f(Tensor([2]), True).item())
print(f(Tensor([2]), False).item())
print(f(Tensor([2]), True).item())
print(calls)
```

Assume `JIT=1` and compatible concrete Tensor inputs.  Predict all four lines.

### 2. Classify five inputs

For a captured `f(x)`, decide whether each is accepted, auto-realized, or
rejected:

1. a new realized `float32` Tensor with the same view and device;
2. a lazy `base + 1` with concrete dtype/device and matching view;
3. `Tensor(UOp.const(2.0).cast(dtypes.float32))`;
4. a same-shape slice with a different base offset; and
5. two argument views sharing one realized base.

Name the check responsible for each answer.

### 3. Repair retained-output semantics

A loop appends every `jitted_step(x)` result to a Python list.  At the end, most
entries show the final step's numbers.  Change the loop so each entry is an
independent snapshot, and explain when the copy must complete.

### 4. Design a branch specialization

A model has `training: bool`.  The True and False branches construct different
captured computations.  Propose two safe designs: one based on separate
captures and one based on `reset`.  Explain why simply passing a different bool
on replay is unsafe.

### 5. Localize a `JIT=1`-only failure

The undecorated function, `JIT=0`, and `JIT=2` are correct on CUDA.  `JIT=1`
returns a wrong value only when two calls share a view of one allocation.  List
the first three artifacts or facts you would inspect before changing code.

### 6. Decide whether pruning is legal

A captured step computes `prepared = weights * scale`, then combines
`prepared` with explicit input `x`.  `weights` is a persistent realized closure
Tensor; `scale` is an ordinary Python float changed every iteration.  Would you
enable `prune=True`?  What must change first?

## Exercise answers

<details>
<summary>1. Phase, counter, and result</summary>

The lines are `5`, `6`, `6`, and `2`.  Ignore executes the add branch, capture
executes and records the multiply branch, replay ignores the new True bool and
runs the captured multiply plan.  Python ran twice.

</details>

<details>
<summary>2. Input classification</summary>

1. Accepted after the names and `(view, Variables, dtype, device)` descriptions
   match.
2. Auto-realized by `_prepare_jit_inputs`, then accepted if its post-realization
   structure matches.
3. Rejected by the virtual check because the UOp has no device-backed storage.
4. Rejected by expected-input-info comparison because the unbound view differs.
5. Rejected before phase selection by duplicate base-buffer detection.

</details>

<details>
<summary>3. Retained outputs</summary>

Use `history.append(jitted_step(x).clone().realize())`.  The clone must be
realized before a later replay overwrites the captured source storage.  On an
asynchronous backend, preserve the dependency or synchronize at the ownership
boundary as appropriate; do not time that snapshot copy as pure replay issue
overhead.

</details>

<details>
<summary>4. Branch specialization</summary>

Keep two `_TinyJit` objects, one whose function always builds the training plan
and one whose function always builds the inference plan; or call `reset()` when
the mode changes and drive ignore/capture again before replay.  A bool is not a
captured input, so changing it on replay cannot re-execute the Python branch.

</details>

<details>
<summary>5. Graph-only failure</summary>

Inspect the no-split `JIT=2` calls and their buffer outs/ins, then the `JIT=1`
post-split plan.  Check first for `SLICE` omission and whether a graph custom
function was formed at all.  If it was, inspect its batch, dependency edges, and
the graph runner's input-address/offset updates for the shared base/view.  The
mode matrix isolates the rewrite path; only the artifact proves that backend
graph execution is involved, and the exact defect could still be bad role
metadata fed into graph construction.

</details>

<details>
<summary>6. Pruning legality</summary>

Not as written.  The Python `scale` is frozen at capture and is not an explicit
input.  Pruning can move its weight-preparation call to one-time work, making
the stale specialization even less visible.  Represent the changing scale as a
compatible **Tensor input whose buffer reaches that preprocessing**, or
deliberately recapture per scale; then prove pruning equivalence and captured
call counts with a test.  A bound symbolic UOp by itself is insufficient because
`prune_linear` seeds reachability from `input_buf_uops`, not `var_vals`.

</details>

## Contribution-shaped workflows

### Change an input contract

1. Write a three-call reproducer which identifies ignore, capture, and replay.
2. Add the smallest focused test under `test/unit/test_jit.py` or the footgun
   suite, depending on whether the desired behavior is a supported contract or
   a documented surprising case.
3. Show the exact old input discovery/description and the proposed new one.
4. Test direct arguments, kwargs ordering, shallow containers, duplicate bases,
   views, dtype/device changes, and symbolic bindings affected by the change.
5. Verify that an apparently more recursive or permissive rule does not capture
   model weights or aliases accidentally.

### Fix a plain replay error

1. Prove call 1 and the undecorated function are correct.
2. Reproduce under `JIT=2` so device graphing is absent.
3. Preserve the no-split `JIT=2` plan or an explicitly instrumented pre-graph
   `LINEAR`, plus `PARAM` slots, current input bases, `var_vals`, and output
   roles.  Do not call `captured._linear` pre-graph under `JIT=1`; it is already
   post-split.
4. Minimize fusion and realization boundaries only as needed to retain the bug.
5. Add a correctness test with at least one replay beyond capture and, when
   relevant, retained-output and mutation assertions.

### Fix or add device-graph support

1. Keep a `JIT=2` oracle using the same compiled calls.
2. Identify the concrete graph class and its `supports_uop` boundary.
3. Record call bodies, devices, buffer roles, base/offset ranges, scalar
   Variables, and symbolic launch dimensions.
4. Test pointer changes across multiple replay inputs, not merely the capture
   buffers.
5. Test read-after-write, write-after-read, output/input aliasing, copies, and
   failure cleanup on the real backend.
6. Measure host issue and device completion separately only after correctness.

### Claim a performance improvement

1. State whether the change targets Python/frontend work, scheduling,
   compilation, ordinary replay submission, graph submission, or device work.
2. Use fresh processes and controlled cache state for `JIT=1/2` comparisons.
3. Exclude ignore/capture from steady-state replay statistics.
4. Record synchronization and debug/profile settings.
5. Preserve numerical equivalence and captured call/graph structure.
6. Report a distribution across enough iterations, not one fastest timing.

These workflows turn “JIT is wrong/slow” into a reviewable claim with a narrow
owner and regression surface.

## Checkpoint

Continue when you can do all of the following without this page open:

- draw the `cnt=0 → 1 → 2+` state machine and say when Python runs;
- explain why capture scheduling records a resolved `LINEAR` but returns an
  empty one to ordinary realization;
- distinguish schedule/program caches from captured replay;
- explain `_linear`, cached `linear`, default `link_linear`, and
  `run_linear(jit=True)`;
- state the exact shallow input walk, direct-kwarg sorting, container insertion
  order, and realized-base duplicate rule;
- distinguish an ordinary lazy input from a virtual UOp and give the correct
  order of rejection versus auto-realization;
- list the recorded view/Variable/dtype/device contract and name matching;
- predict scalar/closure branch freezing and host-read rejection;
- distinguish guaranteed returned-wrapper reuse from the conditional overwriting
  of storage reachable from that wrapper, and make a snapshot safely;
- distinguish read/write mutation from the `outs - ins` pure-write rule and say
  when `_copy_input` is actually selected;
- explain `prune`, `reset`, and `free_intermediates` without conflating them;
- use separate-process `JIT=0/2/1` results to localize graph behavior; and
- design a timing interval that distinguishes host issue from device completion.

If one item is shaky, rerun the corresponding small lab or source stop.  Do not
compensate by reading an entire GPU graph implementation without a question.

## Quick reference

```text
ordinary call
  Python → lazy UOps → schedule → memory plan → compile/cache → execute

TinyJit call 1, cnt=0
  prepare inputs → Python → realize returns normally                    [ignore]

TinyJit call 2, cnt=1
  prepare inputs → Python while realization records LINEARs            [capture]
  flatten → optional prune → PARAM substitution → memory plan
  → compile → optional graph split → CapturedJit → execute once

TinyJit call 3+, cnt>=2
  prepare inputs → check names/view/Variables/dtype/device
  → bind buffers/values → run_linear(prepared, jit=True) → stored ret   [replay]

input discovery
  exact direct Tensor args + exact Tensors one container level deep
  direct Tensor kwargs sorted; container kwargs use insertion order and no names
  virtual check first → normal lazy auto-realization second
  only realized bases become PARAMs; duplicate collected bases rejected

Python values
  bool/int/float/closure/side effects are capture-time specialization
  .item/.tolist-style host reads reject during capture

outputs
  capture and replay return the same stored Tensor/container objects
  dedicated output storage is often overwritten; returned input aliases can be stale
  clone().realize() submits an independent snapshot before the next replay

mutation
  pure-write set = output slots − input slots
  copy only when a current explicit input UOp is in that BUFFER/SLICE set
  read/write buffers are not copied by that rule; prune seeds Tensor buffers, not var_vals

JIT=0  Python every call; no capture; input preparation and cnt still happen
JIT=1  capture/replay; graph-splitting pass runs, graph batching is conditional
JIT=2  capture/replay; graph splitting skipped

PYTHON structural evidence ≠ CUDA graph or asynchronous timing evidence
```

[← Devices and runtimes](12-runtime.md) · [Next: NVIDIA on Ubuntu →](14-nvidia.md)
