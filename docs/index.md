# Learn tinygrad well enough to change it

!!! warning "Independent, unofficial, and AI-generated"

    This is an independently published learning resource about tinygrad. It is
    not part of the tinygrad project and is not affiliated with, maintained by,
    sponsored by, approved by, or endorsed by the tinygrad project, tiny corp,
    or tinygrad's maintainers. For authoritative information, use
    [tinygrad's official repository](https://github.com/tinygrad/tinygrad) and
    [official documentation](https://docs.tinygrad.org/). Questions and
    corrections about this guide belong in this repository, not in tinygrad's
    issue tracker or support channels.

    The guide and its supporting files were generated end-to-end by OpenAI
    Codex using GPT-5.6 Sol Ultra, from prompts and scope decisions supplied by
    the repository owner. The owner did not manually author, edit, or
    technically review the generated material. The owner did manually run the
    bundled lab runner on Ubuntu with an RTX 4090, selecting both `CUDA` and
    `NVK+NV`; all selections in that run passed. This was a limited smoke test,
    not verification of every explanation, command, exercise, device, or
    tinygrad revision. See
    [Provenance and validation](reference/provenance.md) for the exact scope.

This is a contribution-readiness course for tinygrad. It starts where many
framework guides stop: you can already write Python and train models, but terms
such as *IR*, *lowering*, *kernel fusion*, *warp*, and *command queue* do not yet
form one coherent picture.

The goal is not merely to run tinygrad. By the end, you should be able to take an
unfamiliar bug, bounty, optimization, backend task, or feature request and:

1. place it in the end-to-end execution pipeline;
2. find the code and tests that define the current behavior;
3. identify any compiler, GPU, or hardware knowledge you still need;
4. design a small, measurable change with a regression test; and
5. evaluate whether it meets tinygrad's unusually high bar for simplicity.

That is what **ready to tackle any contribution** means here. No finite guide
can pre-teach every device manual or compiler technique. It can give you the
map, vocabulary, investigative habits, and working feedback loops needed to
learn the task-specific remainder without getting stuck.

## Who this is for

The main path assumes you can:

- read non-trivial Python, including decorators, generators, dataclasses, and
  type annotations;
- reason about tensor shapes, broadcasting, reductions, autograd, and common
  neural-network operations; and
- use a terminal for ordinary development tasks.

It does **not** assume prior compiler construction, GPU programming, assembly,
driver development, or detailed computer architecture knowledge. Git and
testing workflows are covered where tinygrad's conventions matter.

The bundled lab runner has been manually smoke-tested on Ubuntu and an NVIDIA
RTX 4090 with both `CUDA` and `NVK+NV` selected. Exercises and commands outside
that runner are not thereby verified. The core path begins with
hardware-neutral and CPU/Python backends; sections that require NVIDIA hardware
are marked and alternatives are provided where practical.

## How the path works

Read the chapters in order and do the exercises. Each chapter has a quick
reference summary so it remains useful when revisited later. Large background
subjects are introduced only far enough to make the next tinygrad concept
understandable, then paired with a focused external resource and a clear
stopping point.

Start with [How to use this guide](guide/how-to-use.md).

Code links are pinned to a recorded source snapshot, while links to policies
and project activity point to their live upstream locations.
