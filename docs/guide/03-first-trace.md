# 3. First trace: follow one expression end to end

## The promise of this chapter

Chapter 1 supplied a map. This chapter turns that map into artifacts produced
by one real process. We will keep the mathematical expression fixed while its
representation changes, and we will stop before planning, after planning,
after compilation, after execution, and after value observation.

No compiler or GPU background is assumed. Terms such as *program*, *binary*,
*dispatch*, and *materialization* are defined before the source uses them. The
goal is not to understand every rewrite. It is to learn what kind of evidence
exists at each boundary and how to find the earliest boundary that can already
explain a bug or performance decision.

By the end, you will be able to:

- draw the exact frontend dependency graph for one Tensor expression;
- distinguish a lazy Tensor root from planned output storage;
- read `CALL(SINK)` as a planned compute call and `CALL(PROGRAM)` as its
  compiled form;
- distinguish the two different `LINEAR` containers visible in a compiled
  trace;
- explain why planning and compiling do not compute the requested output;
- read enough generated C to prove that an elementwise chain fused;
- distinguish a program, an invocation, and an accelerator kernel launch;
- change a materialization boundary and explain the resulting call count; and
- use the first divergent artifact to choose where to investigate next.

## Establish the working directory and snapshot

The checked-in trace lab belongs to the guide repository. Run these checks
from the guide root:

```bash
pwd
test -f labs/phase1/first_trace.py
test -x ../tinygrad-study/.venv/bin/python
git rev-parse --show-toplevel
git -C ../tinygrad-study rev-parse HEAD
```

The last command must print:

```text
874d33128b4e4785beea736d97df6716e0321717
```

If the checkouts are not siblings, replace `../tinygrad-study` with its
absolute path. Do not continue with an unexplained import or commit mismatch;
Chapter 2's environment card is the prerequisite for interpreting exact
artifact names and expected output.

You need ordinary Python loops, lists, and object references. You do not need
to know compiler-pass algorithms, assembly language, CUDA, or GPU scheduling.
Those are later chapters.

## Keep one expression fixed

The carried expression remains deliberately small:

```python
x = Tensor([-2.0, -1.0, 0.0, 1.0])
y = (x * 2 + 1).relu()
```

Compute it by hand before inspecting tinygrad:

| Name | Operation | Value |
| --- | --- | --- |
| `x` | input | `[-2, -1, 0, 1]` |
| `twice` | `x * 2` | `[-4, -2, 0, 2]` |
| `shifted` | `twice + 1` | `[-3, -1, 1, 3]` |
| `y` | `max(shifted, 0)` | `[0, 0, 1, 3]` |

An ordinary scalar loop expresses the same semantics:

```python
for i in range(4):
  shifted_i = x[i] * 2 + 1
  y[i] = shifted_i if shifted_i > 0 else 0
```

That loop is not a claim about tinygrad's final implementation. It is a
semantic reference: every later representation must still produce the same
four values, subject to the usual dtype rules.

## What “trace” means here

A Python call-stack trace answers “which Python functions called which other
functions?” That can be useful, but it is not the main trace a compiler
contributor needs. A compiler repeatedly turns one representation into
another. We therefore capture the artifacts on both sides of important
transformations.

For this chapter, the complete ladder is:

```text
frontend UOp DAG rooted at WHERE
       │ planning chooses storage, fusion, arguments, and order
       ▼
outer LINEAR plan containing CALL(SINK, output, input)
       │ lowering + rendering + target compilation
       ▼
outer LINEAR plan containing CALL(PROGRAM, output, input)
       │ runtime dispatch invokes the program
       ▼
backed output BUFFER
       │ tolist() makes values observable as Python objects
       ▼
[0.0, 0.0, 1.0, 3.0]
```

The boxes and arrows are different kinds of things:

| Kind | Examples | Does it compute `y`'s elements? |
| --- | --- | --- |
| Static description | frontend DAG, `SINK`, `SOURCE`, `BINARY` | No; it describes work or a program. |
| Host-side transformation | planning, lowering, rendering, compiling | It constructs another artifact. It does not dispatch the scheduled output call. |
| Dynamic invocation | Python program call, CPU function call, GPU kernel launch | Yes; it performs the requested compute work. |
| Observation | `tolist()`, `item()`, `numpy()` | It makes values host-readable and may force pending work or synchronization. |

“Compilation does not execute `y`” needs one narrow caveat. On some
accelerator targets, local-size selection performed during `compile_linear`
can benchmark candidate programs. It still does not dispatch the scheduled
call whose output is `y`. Our portable `PYTHON` trace does not need that
accelerator search.

## Boundary 1: the frontend DAG describes values

When Python evaluates the expression, overloaded Tensor methods construct UOp
nodes. At the pinned snapshot, ReLU is represented as a less-than comparison
and a conditional selection, so the root is `WHERE`, not an operation named
`RELU`.

A compact local numbering makes the actual edges readable:

```text
N0 CONST(0)
N1 CONST(4)
N2 BUFFER       <- N1
N3 CONST(2)
N4 MUL          <- N2, N3
N5 CONST(1)
N6 ADD          <- N4, N5
N7 CMPLT        <- N0, N6
N8 WHERE        <- N7, N6, N0
```

`N1` is the input buffer's four-element size metadata; it is not another
arithmetic constant. `N6` is used by both the comparison and the true branch
of `WHERE`. `N0` is used by both the comparison and the false branch. Those
shared references are why this is a directed acyclic graph rather than a tree
that duplicates each shared subtree.

At this point tinygrad knows the dependency structure, shape `(4,)`, dtype,
and device. The derived values have not been computed. `y.uop.is_realized` is
therefore false.

Three objects must remain distinct:

- `y` is a Python `Tensor` wrapper;
- `y.uop` is the wrapper's current UOp root; and
- a runtime `Buffer` is storage that can eventually hold elements.

The Tensor's root can change as planning associates the value with storage.
That change does not itself fill the storage with the answer.

## Boundary 2: planning turns value dependencies into work

A value DAG is not yet an execution plan. A planner must answer at least four
questions:

1. Which results require storage?
2. Which operations may share one compute program?
3. In what order must compute, copies, and other calls occur?
4. Which concrete buffers or symbolic values become program arguments?

For the carried expression, multiplication, addition, comparison, and
selection are all elementwise over the same four positions. Their intermediate
full tensors are not required by the user. A single loop can therefore read
`x`, perform all four scalar operations for an index, and write `y`.

The idealized alternatives are:

```text
unfused                          fused
-------                          -----
write tmp_mul                    for each i:
read tmp_mul; write tmp_add        v = x[i] * 2 + 1
read tmp_add; write tmp_cmp         y[i] = v if v > 0 else 0
read tmp_add + tmp_cmp; write y
```

Fusion is not merely “fewer Python lines.” It removes complete intermediate
storage boundaries when correctness and the target allow it.

### Read `CALL(SINK)` from the inside out

The planned artifact in this example is:

```text
LINEAR[
  CALL(SINK, output_buffer, input_buffer)
]
```

Each name has a specific role:

- `SINK` is an artificial dependency root for one complete kernel body. It
  gathers the output-producing work; it is not an array and not executable
  bytes.
- `CALL` says that its first source is a body to invoke. In this controlled
  case, its remaining two sources identify output and input buffers.
- The outer `LINEAR` is an ordered sequence of execution-plan calls. “Linear”
  means the calls have been put into an order the runtime can traverse; it does
  not mean a neural-network linear layer.

The scheduled `SINK` already introduces more explicit vocabulary than the
frontend graph. For this example it contains:

- two `PARAM` placeholders for output and input arguments;
- one `RANGE` representing the four output positions;
- two `INDEX` nodes connecting that position to output and input storage;
- the familiar `MUL`, `ADD`, `CMPLT`, and `WHERE` semantics;
- one `STORE` for the result;
- an `END` that closes the ranged work; and
- a `SINK` that roots the whole body.

At this stage an `INDEX` can express access through a parameter without a
separate `LOAD` node. Later target lowering makes reads explicit where needed.
Do not infer “the program never loads input memory” from the absence of the
word `LOAD` in this particular scheduled form.

### Planning mutates the Tensor root, but does not execute

`linear_with_vars()` applies the planner's buffer mapping to relevant live
Tensor objects. Consequently, after planning this example:

```text
y.uop.op.name       == "BUFFER"
y.uop.is_realized   == False
```

The new `BUFFER` root means “this value now has a planned storage identity.”
The false flag means the storage is not yet backed with the computed result.
This is an important three-state progression:

```text
lazy operation root       WHERE, unrealized
planned storage identity  BUFFER, unrealized
backed result              BUFFER, realized
```

The returned plan should be executed exactly once. After using an internal
planning helper, do not discard its `LINEAR` and casually call `y.realize()`;
the Tensor root has already been updated. Ordinary application code should use
`y.realize()` and let tinygrad own planning plus dispatch. Splitting the steps
is a snapshot-specific study technique.

## Boundary 3: compilation packages one program

`compile_linear()` transforms the planned compute body into a target program.
The outer schedule still has one call and the same two buffer arguments, but
the call body changes:

```text
before: CALL(SINK,    output_buffer, input_buffer)
after:  CALL(PROGRAM, output_buffer, input_buffer)
```

The `PROGRAM` in this snapshot has four child artifacts:

```text
PROGRAM
├── SINK
├── LINEAR
├── SOURCE
└── BINARY
```

They answer different questions:

- The `SINK` retains the lowered dependency form and kernel metadata.
- This **inner** `LINEAR` is the ordered low-level operation sequence for one
  program. It is not the outer `LINEAR` list of runtime calls.
- `SOURCE` is the renderer's target-representation string. On CPU it is
  readable C. The Python backend uses an encoded serialization, so its string
  is not intended as a first code-reading exercise.
- `BINARY` is the compiler's byte payload for the runtime to load or use. It
  is not universally the final native machine instructions; for example, a
  driver may perform another compilation step when loading a virtual ISA.

The program also carries metadata: a function name, target, launch dimensions,
symbolic values, and the positions of global buffer arguments. In our trace,
argument position `0` is an output and position `1` is an input.

The repeated name `LINEAR` is easy to misread. Use its parent to disambiguate:

| Location | Sources contain | Meaning |
| --- | --- | --- |
| Top-level schedule `LINEAR` | `CALL` nodes | Ordered actions for the runtime. |
| `PROGRAM` child `LINEAR` | lowered operation nodes | Ordered instructions/operations for one rendered program. |

Compilation does real host work: graph transforms run, source is rendered,
and a compiler may produce bytes. But the output Tensor remains unrealized
until the outer `CALL(PROGRAM, ...)` is dispatched.

## Boundary 4: dispatch performs an invocation

The execution loop walks the outer `LINEAR` plan. A call whose body is
`PROGRAM` is routed to the kernel-execution path. That path resolves and
allocates buffers, obtains a runtime program, calculates launch dimensions,
and invokes it with concrete buffer handles and scalar values.

Use this vocabulary precisely:

- A **program** is the reusable compiled description/package.
- An **invocation** is one execution of that program with particular arguments
  and dimensions.
- A **kernel launch** is an accelerator invocation that submits indexed work
  to a device. A CPU function call and a Python-backend program invocation are
  not GPU kernel launches.

On the `PYTHON` backend used in the portable lab, the invocation completes the
calculation synchronously. After dispatch, `y.uop.is_realized` is true and
`y.tolist()` returns `[0.0, 0.0, 1.0, 3.0]`.

On an accelerator, dispatch can enqueue work and return before the device has
finished it. A backed or allocated buffer is not by itself proof that the host
waited. Host-readable observation eventually requires the appropriate
completion and copyout behavior. Chapter 12 makes that timeline explicit.

## Paper lab: predict the trace before running it

Do this on paper or in a notebook. Do not inspect the worked answer until you
have made a prediction.

1. Draw one node each for input `x`, constants `2`, `1`, and `0`, `MUL`,
   `ADD`, `CMPLT`, and `WHERE`. Draw source-to-consumer edges.
2. Mark the two nodes that each have two consumers.
3. Give one topological order and compute every four-element value by hand.
4. Under the idealized assumption that full-tensor reads and writes dominate,
   count element transfers for:
   - one fused loop that reads `x` and writes `y`; and
   - two loops separated by a materialized affine result.
5. Put these cards in causal order:
   - backed output `BUFFER`;
   - frontend `WHERE` DAG;
   - outer `LINEAR[CALL(PROGRAM, ...)]`;
   - Python list returned by `tolist()`;
   - outer `LINEAR[CALL(SINK, ...)]`.
6. Which transitions plan or compile, and which transition actually computes
   the output elements?
7. Suppose the frontend DAG and scheduled `SINK` both represent `x * 2 + 1`,
   but generated CPU source contains `x * 3 + 1`. What is the earliest known
   bad boundary?
8. Suppose `.realize()` is called on `x * 2 + 1` before `.relu()` is built.
   Predict whether values and compute-call count change.

??? success "Worked answer"

    **1–3. Graph, sharing, order, and values**

    ```text
    x, 2 -> MUL -> ADD <- 1
                    ├──────────────┐
    0 ────────┐     ▼              ▼
              └-> CMPLT --------> WHERE <- 0
    ```

    More precisely, `CMPLT` has sources `(0, ADD)` and `WHERE` has sources
    `(CMPLT, ADD, 0)`. `ADD` and constant zero therefore each have two
    consumers. One valid order is `x`, `2`, `MUL`, `1`, `ADD`, `0`, `CMPLT`,
    `WHERE`; independent constants may appear earlier.

    ```text
    MUL   [-4, -2, 0, 2]
    ADD   [-3, -1, 1, 3]
    CMPLT [False, False, True, True]
    WHERE [0, 0, 1, 3]
    ```

    **4. Idealized traffic**

    The fused loop reads four elements of `x` and writes four of `y`: eight
    element transfers. The split form reads four and writes four for the
    affine result, then reads four and writes four for ReLU: sixteen. This is
    intuition about an intermediate-buffer cost, not a timing prediction;
    caches, vectorization, target behavior, and launch overhead also matter.

    **5–6. Artifact order and execution**

    ```text
    frontend WHERE DAG
      -> outer LINEAR[CALL(SINK, ...)]
      -> outer LINEAR[CALL(PROGRAM, ...)]
      -> backed output BUFFER
      -> Python list
    ```

    Planning creates the `CALL(SINK)` plan. Lowering, rendering, and target
    compilation create `CALL(PROGRAM)`. Runtime dispatch computes the output.
    `tolist()` observes it.

    **7. Earliest divergence**

    The scheduled semantics are still correct and the generated source is
    already wrong. The fault was introduced between the scheduled `SINK` and
    `SOURCE`, so investigate lowering, optimization, or rendering—not Tensor
    arithmetic or runtime argument submission first.

    **8. Materialization barrier**

    The values remain `[0, 0, 1, 3]`. The affine expression is planned and
    invoked first; ReLU is constructed later and requires a second plan and
    invocation. The observed total is two compute calls across two separate
    realization plans.

## Runnable lab A: capture the whole artifact ladder

Run from the guide root. The lab intentionally uses internal APIs from the
pinned snapshot; they are observation tools, not stable application APIs.

```bash
CACHEDB=/tmp/tinygrad-guide-first-trace.db DEV=PYTHON DEBUG=0 \
  ../tinygrad-study/.venv/bin/python labs/phase1/first_trace.py
```

The stable output is:

```text
device: PYTHON
frontend: WHERE realized= False
frontend op counts: {'ADD': 1, 'BUFFER': 1, 'CMPLT': 1, 'CONST': 4, 'MUL': 1, 'WHERE': 1}
frontend graph:
  N0 CONST <- [] arg=0.0
  N1 CONST <- [] arg=4
  N2 BUFFER <- [N1]
  N3 CONST <- [] arg=2.0
  N4 MUL <- [N2,N3]
  N5 CONST <- [] arg=1.0
  N6 ADD <- [N4,N5]
  N7 CMPLT <- [N0,N6]
  N8 WHERE <- [N7,N6,N0]
shared ADD/zero: True True
planned tensor: BUFFER realized= False
planned calls: [('CALL', 'SINK', 'PYTHON', 2)]
planned body ops: {'ADD': 1, 'CMPLT': 1, 'CONST': 4, 'END': 1, 'INDEX': 2, 'MUL': 1, 'PARAM': 2, 'RANGE': 1, 'SINK': 1, 'STORE': 1, 'WHERE': 1}
compiled calls: [('CALL', 'PROGRAM', 'PYTHON', 2)]
program children: ['SINK', 'LINEAR', 'SOURCE', 'BINARY']
program buffer roles: outs= (0,) ins= (1,)
program payload types: str bytes
before execution: False
after execution: True
execution calls: 1
value: [0.0, 0.0, 1.0, 3.0]
fused calls/value: 1 [0.0, 0.0, 1.0, 3.0]
barrier calls/value: 2 [0.0, 0.0, 1.0, 3.0]
```

Read the output in five groups.

### 1. Frontend evidence

The first block proves the exact edges rather than merely counting operation
names. `N6` and `N0` appear twice as sources, and the explicit identity checks
confirm the references are shared objects. The graph is lazy and rooted at
`WHERE`.

### 2. Planning evidence

The Tensor root has changed to `BUFFER`, yet remains unrealized. The outer plan
contains one `CALL`, its body is `SINK`, and the final tuple field `2` is the
number of call arguments after the body in this lab. The planned-body counts
show that positions, indexed parameters, and one output store now surround the
same four semantic operations.

### 3. Compilation evidence

The outer call remains one call with two arguments, but its body is now a
`PROGRAM`. The child list proves the four-part package. `outs=(0,)` and
`ins=(1,)` identify buffer roles; the payload types establish that rendering
produced a string and compilation produced bytes. Neither fact alone proves
that the runtime invoked them.

### 4. Execution evidence

Immediately before dispatch the output is still unrealized. The lab calls
`run_linear` on the exact already-compiled plan. Its internal `jit=True`
argument means “do not compile this plan again” in this call; it is not the
`@TinyJit` feature taught in Chapter 13. After dispatch, the output is backed,
the isolated tracked-call count is one, and the values match the hand result.

`GlobalCounters.kernel_count` is historical terminology. The counter is
incremented for tracked execution-plan calls, including some copies and views.
It equals compute-program invocations only because this region was warmed,
reset, and constrained to one compute call.

### 5. Boundary evidence

The final two lines keep semantics fixed while moving a materialization
boundary. The fused case creates and invokes one plan. The barrier case first
realizes the affine expression, then constructs and realizes ReLU. Its count of
two is cumulative across **two separate plans**, not evidence that one outer
`LINEAR` held two calls.

The lab contains assertions for all stable facts. It deliberately does not
assert generated names, hashes, timings, source length, binary length, or
optimizer choices that are not part of the lesson.

## Runnable lab B: read one generated CPU program

The Python backend is the clean semantic control, but its `SOURCE` is an
encoded UOp serialization. CPU gives us readable C and a real native
compile/invoke route without GPU concepts.

With the sibling layout established in Chapter 2, change from the guide root
to the pinned tinygrad root, then run. If your checkouts are elsewhere, replace
the first path with the study checkout's absolute path.

```bash
cd ../tinygrad-study

CACHEDB=/tmp/tinygrad-guide-trace-cpu.db DEV=CPU DEBUG=0 \
  .venv/bin/python - <<'PY'
from tinygrad import Context, Device, GlobalCounters, Tensor

x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()
GlobalCounters.reset()
y = (x * 2 + 1).relu()

print("device:", Device.DEFAULT)
with Context(DEBUG=4):
  y.realize()
print("result:", y.tolist())
PY
```

The pinned snapshot currently renders this essential structure:

```c
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_4(float* restrict data0_4, float* restrict data1_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  float alu0 = ((val0[0]*2.0f)+1.0f);
  float alu1 = ((val0[1]*2.0f)+1.0f);
  float alu2 = ((val0[2]*2.0f)+1.0f);
  float alu3 = ((val0[3]*2.0f)+1.0f);
  float alu4 = ((0.0f<alu0)?alu0:0.0f);
  float alu5 = ((0.0f<alu1)?alu1:0.0f);
  float alu6 = ((0.0f<alu2)?alu2:0.0f);
  float alu7 = ((0.0f<alu3)?alu3:0.0f);
  *((float4*)((data0_4+0))) = (float4){alu4,alu5,alu6,alu7};
}
```

You may also see scheduling and optimization diagnostics before the function,
then one line beginning `*** CPU`, followed by:

```text
result: [0.0, 0.0, 1.0, 3.0]
```

Read the generated function with only this much C vocabulary:

- A pointer such as `float* data1_4` identifies memory containing floats.
- `restrict` promises this function that the relevant pointer regions do not
  alias in a way that invalidates its optimizations.
- `float4` is a compiler-supported four-float vector type, aligned to a
  16-byte boundary.
- The first statement loads four adjacent input floats from `data1_4`.
- `val0[0]` through `val0[3]` select the four vector lanes.
- `condition ? true_value : false_value` is C's conditional expression. Here
  it implements ReLU's select.
- The final statement constructs one four-float vector and stores it to
  `data0_4`.

The program metadata from Lab A tells us argument `0` is output and argument
`1` is input, matching `data0_4` and `data1_4`. The multiply, add, comparison,
selection, and one output store are all present in one function. That is direct
fusion evidence; there are no full intermediate arrays for `x*2` or `x*2+1`.

The exact name `E_4`, vectorization choice, variable names, cache hash, timing,
memory rates, and optimization messages are incidental. A different target or
optimizer revision may spell an equivalent program differently.

`DEBUG=4` proves that rendering reached a source artifact. It also includes
the behavior of lower debug levels: `DEBUG>=2` requests timed waiting, so the
`*** CPU` line is dynamic invocation evidence and debug timing can alter an
accelerator's normal asynchronous behavior. Generated source without a
runtime line is not proof that invocation succeeded.

## Reason carefully about materialization and fusion

The lab's barrier experiment is equivalent to:

```python
x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()

fused = (x * 2 + 1).relu().realize()

affine = (x * 2 + 1).realize()  # first plan and invocation
split = affine.relu().realize()  # second plan and invocation
```

Both routes return the same values. The second deliberately ends the lazy
region before ReLU exists. That is why it cannot fuse ReLU into the already
executed affine program.

Materialization is neither universally bad nor merely a performance hint. It
can be required for observation, mutation ordering, device copies, reuse by
independent consumers, or target constraints. Conversely, an unnecessary
materialization can add program-invocation overhead and intermediate traffic.
The contributor question is always narrower:

> For this dependency graph and target, is this boundary required for
> correctness, chosen for profitability, or introduced accidentally?

Call count alone cannot answer that question. Pair it with values, the planned
call kinds, generated code, and—when performance is the claim—sound timing.

## Accelerator branch: replay the same evidence on the RTX 4090

Only take this branch after the portable and CPU routes pass. Run each target
in a fresh process from the tinygrad study root:

```bash
for dev in CUDA NVK+NV; do
  echo "--- $dev ---"
  CACHEDB="/tmp/tinygrad-guide-trace-$dev.db" DEV="$dev" DEBUG=0 \
    .venv/bin/python - <<'PY'
from tinygrad import Context, Device, GlobalCounters, Tensor
from tinygrad.helpers import DEV

x = Tensor([-2.0, -1.0, 0.0, 1.0]).realize()
GlobalCounters.reset()
y = (x * 2 + 1).relu()

print("target:", DEV)
print("device:", Device.DEFAULT)
with Context(DEBUG=4):
  y.realize()
print("tracked calls:", GlobalCounters.kernel_count)
print("result:", y.tolist())
PY
done
```

For a successful route, the stable semantic result is:

```text
tracked calls: 1
result: [0.0, 0.0, 1.0, 3.0]
```

`DEV=NVK+NV` prints `target: NVK+NV` but `device: NV`, because
`Device.DEFAULT` omits the selected interface. Save the generated source and
the one `*** CUDA` or `*** NV` execution line separately. On these accelerator
routes, the compute-program invocation is a kernel launch.

This branch is conditional evidence, not a promise that driver visibility is
enough. `nvidia-smi` can see a 4090 while CUDA initialization, compiler-library
loading, allocation, or submission still fails. Preserve the first exception
and the last successfully captured artifact.

If CUDA and NV render equivalent source but only one runtime route fails, the
evidence points below the shared source boundary. It does not prove the
runtime is at fault: compiler payloads, target metadata, arguments, launch
dimensions, and synchronization remain candidates. “Points below” is a
search reduction, not a verdict.

## Localize by the earliest divergent artifact

Use the first artifact that disagrees with the intended invariant. Later
failures are often consequences.

| Evidence | Earliest useful suspicion | Do not begin by changing |
| --- | --- | --- |
| Hand values and frontend DAG disagree | Tensor semantics, dtype/broadcast behavior, or frontend construction | GPU driver or launch settings |
| Frontend is right; planned boundaries or order are wrong | callification, fusion, dependency scheduling, or materialization policy | renderer spelling |
| Planned `SINK` is right; lowered/source semantics are wrong | optimization, lowering, decomposition, or rendering | Tensor API |
| Source is plausible; target compiler rejects it | renderer/compiler contract, target features, or compiler environment | scheduling count |
| Program builds; runtime invocation errors | allocation, argument roles, dimensions, program loading, submission, or device state | algebraic simplification first |
| Invocation appears successful; host values are stale or wrong | execution semantics, copyout, synchronization, or earlier silent miscompile | timing optimization |
| PYTHON and CPU agree; one GPU route disagrees | first GPU-specific lowering/compiler/runtime boundary | shared frontend without contrary evidence |

“Source is plausible” is weaker than “source is proved correct.” A visual scan
can miss dtype, signedness, indexing, undefined behavior, and ABI errors. Use
the table to choose the next artifact, then construct a test that can falsify
the specific hypothesis.

## Guided source tour: confirm the ladder one question at a time

Only open these links after the artifact model and labs make sense. Each range
targets the pinned commit. Read the question first, follow the named lines,
translate them into the stated answer, and ignore unrelated helpers.

### Stop 1: why can planning change a live Tensor root?

Read [`_apply_map_to_tensors` lines 19–33](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L19-L33).

Question: what happens to in-scope Tensor objects whose UOps are affected by a
mapping?

Translation: tinygrad finds relevant live Tensors, substitutes the mapped
UOps, and assigns each changed Tensor's `uop` to its replacement. This is why
the lab's `y` wrapper survives while `y.uop` changes from `WHERE` to `BUFFER`.
Ignore weak-reference lifetime and profiling details on the first read.

Now read [`linear_with_vars`, `schedule_linear`, and `realize` lines 175–196](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/tensor.py#L175-L196).

Question: which method returns a plan, and which method additionally runs it?

Translation: `linear_with_vars` transforms roots into calls, applies the
buffer map, and creates the ordered plan. `schedule_linear` is a convenience
for plans with no unresolved variables. `realize` selects values that need
work and sends the returned plan to `run_linear`.

### Stop 2: where do `SINK`, `CALL`, and the outer `LINEAR` arise?

Read only [`split_store` lines 520–529](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/rangeify.py#L520-L529).

Question: what wraps the rewritten kernel body, and what is passed to its
call?

Translation: after kernel-local rewriting, `ret.sink(...)` gives the body a
`SINK` root and `.call(...)` supplies mapped parameters and variables. Focus on
lines 526–529; range legality and local-buffer context belong to Chapters 7–10.

Then read [`create_schedule` lines 64–80](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/schedule/__init__.py#L64-L80).

Question: what does the queue emit, and what wraps the final sequence?

Translation: calls whose dependencies are ready are appended in order, and
the result becomes a `LINEAR` whose sources are that ordered tuple. The earlier
part of the function computes dependency edges and degrees; Chapter 7 derives
them from read/write hazards.

### Stop 3: how does `CALL(SINK)` become `CALL(PROGRAM)`?

Read [`pm_compile` and `compile_linear` lines 247–273](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L247-L273).

Question: which source of each call is replaced, and what remains around it?

Translation: a call whose body is `SINK` or an existing `PROGRAM` passes that
body through `to_program`; the call's remaining buffer arguments remain in
place. `compile_linear` applies this conversion across the plan and may also
apply validation, BEAM, HCQ, or local-size behavior when enabled. Pattern
matching syntax is taught in Chapter 6; for now, follow the before/after shape.

### Stop 4: where do `SOURCE` and `BINARY` come from?

Read [`do_render` and `do_compile` lines 433–441](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L433-L441).

Question: what type of payload does each step append?

Translation: the renderer converts a linear operation list into target source
and appends a `SOURCE`; the target compiler compiles/cache-looks-up that string
and appends a `BINARY` payload.

Then read the docstring and main path of [`do_to_program` lines 451–478](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/codegen/__init__.py#L451-L478).

Question: what four children does the returned program promise?

Translation: a `SINK` is fully rewritten for the renderer, given
`ProgramInfo`, optionally instruction-selected, wrapped in `PROGRAM`, and run
through the linearize/render pipeline. The promised result contains
`SINK/LINEAR/SOURCE/BINARY`. Decomposition and instruction selection are later
subjects; do not chase their matchers yet.

### Stop 5: what crosses the runtime boundary?

Read [`run_linear` lines 277–281](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L277-L281).

Question: when does the normal path compile, and what is visited during
execution?

Translation: unless told it already has a compiled/JIT plan, `run_linear`
compiles and links first. It then visits every outer `LINEAR` call through the
execution matcher.

Finish with [`exec_kernel` lines 176–186](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/tinygrad/engine/realize.py#L176-L186).

Question: identify the final operation that actually invokes the runtime
program.

Translation: the function resolves call arguments, ensures program buffers
are allocated, obtains a runtime object, resolves launch dimensions, and calls
that object with buffer handles, dimensions, scalar values, and wait/timeout
settings. On an accelerator, that call is the launch boundary. Multi-device
iteration and statistics context are not required for this example.

## Troubleshooting by the artifact you expected

| Observation | Explanation or next check |
| --- | --- |
| `labs/phase1/first_trace.py` is missing | You are not at the guide root. Run `pwd` and `git rev-parse --show-toplevel`. |
| Import or artifact names differ | Print the interpreter, imported `tinygrad.__file__`, commit, `DEV`, and `DEBUG`; return to Chapter 2. |
| Frontend root is already `BUFFER` | The Tensor was realized or planned earlier. Start a fresh process and construct fresh values. |
| `planned tensor: BUFFER realized=False` looks contradictory | Storage identity has been planned; the program has not filled/allocated the result yet. |
| Calling `.realize()` after manually planning behaves surprisingly | Execute the returned plan exactly once, as the lab does. Use ordinary `.realize()` when not studying internals. |
| Python `SOURCE` is unreadable encoded text | Expected: the Python backend serializes UOps. Use the CPU lab for readable generated C. |
| CPU trace contains extra work | Realize the input before resetting counters and enabling scoped debug output. Classify every call body rather than assuming all calls are compute. |
| Repeated run does not print source | Use a fresh process and controlled `CACHEDB`; in-process program caches may already contain the artifact. |
| Exact C name or vectorization differs | Confirm the snapshot and compare semantics, arguments, stores, and call count rather than incidental spelling. |
| `nvidia-smi` works but CUDA/NV fails | Keep the portable trace. Record the first failing initialization/compiler/runtime layer; driver visibility is only one prerequisite. |
| Timing changes with `DEBUG=2/4` | Expected: timed debug requests waiting/synchronization. Do not use debug timing as evidence of ordinary asynchronous behavior. |

## Checkpoint: produce an artifact ledger

Fill this from your own run:

```text
study commit and backend:
frontend root, realized state, and shared edges:
planned Tensor root and realized state:
outer planned call body and argument count:
scheduled body landmarks:
outer compiled call body:
PROGRAM children and buffer roles:
state immediately before dispatch:
tracked call kind/count:
state immediately after dispatch:
observed value:
fused versus barrier counts:
accelerator result or first failing layer:
```

You pass when you can explain all of the following without relying on the
expected-output block:

1. Why are `ADD` and zero shared in the frontend DAG?
2. Why can `y.uop` be `BUFFER` while `is_realized` is false?
3. What does the body of a `CALL` describe, and what do its other sources do?
4. What is the difference between `SINK` and `PROGRAM`?
5. Why are there two `LINEAR` nodes, and what does each order?
6. What evidence proves compilation happened? What separate evidence proves
   invocation happened?
7. Why is a CPU program invocation not a GPU kernel launch?
8. Why does the materialization experiment count two separate plans?
9. Why can `DEBUG=4` change the synchronization behavior being observed?
10. Given two neighboring artifacts, how would you state the invariant that
    should survive between them?

## Quick reference

| Artifact or event | Meaning in this chapter |
| --- | --- |
| frontend `WHERE` | Lazy value dependency root for ReLU's comparison/select form. |
| planned `BUFFER`, unrealized | Output has storage identity but has not been computed. |
| `SINK` | Root of one planned/lowered kernel body. |
| `CALL(body, args...)` | Description of one invocation with a body and arguments. |
| outer `LINEAR` | Ordered execution-plan calls. |
| `PROGRAM` | Compiled program package plus metadata; not an invocation. |
| inner `LINEAR` | Ordered low-level operations inside one program. |
| `SOURCE` | Renderer-produced target representation string. |
| `BINARY` | Compiler-produced bytes used by the runtime; not always final native ISA. |
| program invocation | One execution with concrete arguments and dimensions. |
| accelerator kernel launch | A program invocation submitted to an accelerator. |
| materialization | Establish and compute a storage-backed value at a boundary. |
| observation | Make values available to Python, potentially forcing work/copy/sync. |

Snapshot-specific observation helpers:

```text
Tensor.uop.toposort()        inspect frontend nodes and edges
Tensor.linear_with_vars()    construct a plan and variable bindings
compile_linear(linear)       replace planned bodies with compiled PROGRAMs
run_linear(..., jit=True)    dispatch an already-compiled internal plan
Context(DEBUG=4)             print generated source in a scoped region
GlobalCounters.reset()       reset tracked ops/memory/time/call totals, not allocated-memory state
```

## Optional reinforcement—not missing prerequisites

- Read only [lines 34–48 of tinygrad's pinned `docs/abstractions3.py`](https://github.com/tinygrad/tinygrad/blob/874d33128b4e4785beea736d97df6716e0321717/docs/abstractions3.py#L34-L48).
  It shows the same “create a `LINEAR`, then run it” boundary for a much larger
  training step. Do not run the whole file yet; dataset loading, autograd, and
  optimizer mutation obscure the small trace.
- Reopen the local [architecture map](../reference/architecture-map.md) and
  point to the exact row for every arrow in the trace ladder. Stop when each
  arrow has one responsibility; implementation depth comes later.
- Change only one constant in the checked-in lab in a temporary copy. Predict
  which frontend nodes, generated C literals, and final values change, and
  which call/program structure remains invariant.

## What is deliberately left for later

- Chapter 4 explains how forward value graphs produce gradient expressions.
- Chapter 5 develops UOp fields, identity, traversal, and operation families.
- Chapter 6 explains the rewrite machinery used by transformations.
- Chapter 7 derives materialization, fusion, dependencies, and plan ordering.
- Chapters 8–10 explain ranges, indices, optimization, and lowering.
- Chapter 11 opens the inner `LINEAR`, `SOURCE`, and `BINARY` construction.
- Chapter 12 explains buffers, allocation, runtime programs, arguments, and
  synchronization.
- Chapter 14 specializes the execution model to NVIDIA hardware and the CUDA
  and NV routes.

[← Development setup](02-setup.md) · [Next: Tensor frontend and autograd →](04-tensor-and-autograd.md)
