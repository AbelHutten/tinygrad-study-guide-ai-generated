# How to use this guide

This guide is a sequence, not a bag of articles. Later chapters assume the
mental models and tools established earlier even when they link back to a quick
reference.

## The chapter contract

Most chapters follow the same shape:

1. **Purpose** — the contribution problem this knowledge unlocks.
2. **Prerequisite gate** — what you must understand before continuing.
3. **Mental model** — the smallest useful explanation of the new idea.
4. **Source tour** — exact tinygrad paths and symbols to inspect.
5. **Lab** — an observation, modification, or test you perform yourself.
6. **Checkpoint** — evidence that you are ready for the next chapter.
7. **Quick reference** — a compact refresher for later use.

Do not optimize for finishing pages. Optimize for passing checkpoints without
guessing. When a checkpoint exposes a gap, follow the named resource, return,
and try it again.

## Three kinds of claim

The guide separates information by how quickly it can go stale:

- **Durable model** — concepts such as graph rewriting, dependency ordering,
  memory hierarchy, and the compile/execute boundary.
- **Source snapshot** — current paths, symbols, pass order, environment
  variables, and test commands. These are tied to the recorded commit.
- **Live project state** — contribution policy, open issues, bounties, and CI.
  Always re-check the linked upstream source before acting.

This distinction prevents two common failures: memorizing current function
names without understanding the system, and learning a generic compiler model
that does not match tinygrad's actual code.

## Working checkout

Keep a separate tinygrad checkout beside this guide. Unless a chapter says
otherwise, commands run from the tinygrad repository root:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m venv .venv
.venv/bin/pip install -e '.[testing_minimal]'
```

The source evolves quickly. For a line-for-line match with this guide, create a
throwaway study branch at the [recorded snapshot](../reference/source-snapshot.md).
For contribution work, return to current `master` and use the guide's update
workflow to translate renamed symbols.

## Exercise discipline

Keep a study notebook or a directory outside your tinygrad checkout. For each
lab, save:

- the command and relevant environment variables;
- the before/after graph, generated source, or benchmark result;
- your prediction before running it;
- what surprised you; and
- the source locations that explain the result.

Predicting first matters. Compiler output is large enough that passive reading
can feel like understanding even when it is not.

## Hardware convention

The path uses these labels:

- **Portable** — should work with the Python, NULL, or CPU backend.
- **Accelerator** — needs a supported GPU but is not vendor-specific.
- **NVIDIA** — written for Ubuntu plus an RTX 4090. The bundled runner's selected
  `CUDA` and `NVK+NV` probes passed once on that setup; chapter exercises outside
  the runner may not have been executed, and backend details may not apply
  elsewhere.

Never begin an investigation on hardware if a smaller Python/CPU reproducer can
answer the same semantic question. Move to the GPU when launch behavior,
generated device code, memory hierarchy, or performance is the subject.
