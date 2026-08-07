# How to use this guide

This guide is a course on learning how to investigate tinygrad, not a sparse
index of source files. Read it front to back on the first pass. Later, use the
chapter routes, checkpoints, and quick references to revisit a concept without
rereading the whole course.

The intended reader already knows Python and machine learning but may know
little about compilers, GPU programming, operating-system interfaces, or
upstream contribution practice. A chapter should therefore explain a new idea
from first principles before asking you to interpret tinygrad source. When a
topic needs a deeper prerequisite, the chapter names the gap, gives a bounded
resource, and tells you what you should be able to do when you return.

## The chapter contract

The long chapters are not forced into one mechanical template, but they share
the same learning contract:

1. **Promise and route** — what you will be able to explain or investigate,
   and why the sections appear in that order.
2. **First-principles model** — the vocabulary, data flow, invariants, and
   small examples needed before implementation details.
3. **Evidence boundaries** — what an observation, test, generated artifact,
   or timing result establishes and what it does not.
4. **Guided source stops** — bounded, commit-pinned source evidence with a
   prediction or question. These stops confirm a model; declarations in
   isolation are not meant to teach the model.
5. **Exercises and executable labs** — predictions, paper traces, controlled
   runs, negative cases, and explicit claims or non-claims.
6. **Background ladder** — focused resources and a return exercise when the
   next step requires knowledge the guide cannot safely compress.
7. **Checkpoint and quick reference** — evidence that you can proceed, plus a
   compact refresher for later investigations.

Do not optimize for reaching the bottom of a page. Predict before running a
lab, answer source-stop questions in your own words, and pass the checkpoint
without guessing. If a linked range still makes no sense after its preceding
explanation, inspect its named callers and tests or report the missing bridge
as a guide defect; merely staring at more declarations is not progress.

Start with [Chapter 1's first-principles orientation](01-orientation.md), then
use [Chapter 2](02-setup.md) to create the exact working arrangement below.

## Three clocks for claims

The guide distinguishes facts by how quickly they can expire:

- **Durable model** — concepts such as graph rewriting, dependency ordering,
  memory hierarchy, test power, and the compile/execute boundary.
- **Pinned source snapshot** — paths, symbols, pass order, flags, expected lab
  output, and recorded CI behavior at commit `874d331`. Use the detached study
  checkout and commit-pinned links for these claims.
- **Live project state** — current source, contribution policy, issues,
  bounties, pull requests, CI, and maintainer direction. Recheck these in a
  current work checkout immediately before relying on them.

This prevents two opposite mistakes: memorizing names without understanding
the system, and applying a generic compiler model without checking what the
recorded or current tinygrad implementation actually does. The exact snapshot
and translation procedure are recorded in the
[source-snapshot reference](../reference/source-snapshot.md).

## Keep three repository roles separate

Use three sibling directories when practical:

```text
projects/
├── tinygrad_docs/     this guide, its labs, and scripts/run_labs.py
├── tinygrad-study/    tinygrad detached at the guide's exact snapshot
└── tinygrad-work/     current tinygrad fork/branch for real investigations
```

The names may differ; the separation of roles must not.

| Repository | Required state | Use |
| --- | --- | --- |
| Guide | This documentation revision | Read chapters and run bundled guide scripts. |
| `tinygrad-study` | Detached at `874d33128b4e4785beea736d97df6716e0321717` | Follow pinned source, reproduce expected observations, and run labs. Do not turn it into a contribution branch. |
| `tinygrad-work` | Current upstream base plus a named branch in your fork | Recheck live behavior, investigate a real issue, bounty, or self-chosen improvement, and prepare commits. Do not force it back to the study commit. |

The bundled runner is invoked from the **guide root** but receives the detached
study checkout and its interpreter explicitly:

```bash
cd /absolute/path/to/tinygrad_docs
python3 scripts/run_labs.py \
  --tinygrad /absolute/path/to/tinygrad-study \
  --python /absolute/path/to/tinygrad-study/.venv/bin/python
```

Tinygrad source experiments and pinned tests run from `tinygrad-study` with its
own `.venv/bin/python`. Real candidate reproduction, history inspection,
branches, tests, and patches run from `tinygrad-work`. Chapter 18 explains how
to translate a pinned lesson into a current contribution claim; the guide's
runner is deliberately not a validator for a changing work checkout.

## Treat runner results as bounded evidence

A current local runner success proves only that the processes selected by that
runner completed on the supplied checkout, interpreter, and requested devices.
It does not execute every prose exercise, tinygrad's full suite, or a technical
review of the explanations.

There is also a historical hardware observation in this repository. On
2026-08-05, the owner ran the **then-current** runner against the pinned study
commit on a reported Ubuntu/RTX 4090 system with `CUDA` and `NVK+NV`; its 11
selected subprocesses passed. Labs and runner entries were expanded or
rewritten afterward, so that owner smoke test alone is not evidence for the
present runner. Separately, on 2026-08-07, Codex ran the expanded runner with
`CPU`, `CUDA`, and `NVK+NV`; all 41 selected processes passed. That was
model-operated validation, not owner or independent human review, and it still
says nothing about commands outside the selection. The exact coverage and
limits are in [Provenance and validation](../reference/provenance.md). Run the
present runner yourself and record its exact selection.

## Exercise discipline

Keep a study notebook outside both tinygrad checkouts. For each exercise or
lab, record:

- repository root, commit, interpreter, device, and relevant environment;
- your prediction before the run;
- the command and the first relevant artifact or result;
- the oracle or invariant and whether it held;
- what the evidence does **not** establish;
- what surprised you; and
- the source, test, or resource that resolved the surprise.

Compiler output is large enough that passive reading can feel like
understanding. A prediction, an independent check, and a bounded conclusion
make the difference visible.

## Hardware convention

The course uses these labels:

- **Portable** — designed for the readable `PYTHON` route or a controlled CPU
  route; check the chapter's exact command.
- **Structural NVIDIA** — renders or lowers for an Ada target while executing
  through a hardware-free route; this is not physical-GPU evidence.
- **Accelerator** — requires a supported physical GPU but is not necessarily
  vendor-specific.
- **NVIDIA** — written for the primary Ubuntu plus RTX 4090 environment, while
  still naming which claims can transfer elsewhere.

Begin with a smaller Python or CPU reproducer whenever it answers the same
semantic question. Move to a physical GPU when launch behavior, generated
device code, queues, synchronization, memory hierarchy, or performance is
actually part of the claim.

[Next: curriculum map →](curriculum.md)
