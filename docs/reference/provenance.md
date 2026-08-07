# Provenance and validation

## Independence

This guide is an independently published third-party learning resource about
tinygrad. It is not part of the tinygrad project and is not affiliated with,
maintained by, sponsored by, approved by, or endorsed by the tinygrad project,
tiny corp, or tinygrad's maintainers. The name *tinygrad* is used only to
identify the open-source project discussed by the guide.

For authoritative project information, use the
[tinygrad repository](https://github.com/tinygrad/tinygrad) and
[official documentation](https://docs.tinygrad.org/). Questions, corrections,
and support requests concerning this guide belong in this repository, not in
tinygrad's upstream issue tracker or support channels.

## Generation

The initial guide and its supporting files were generated end-to-end by OpenAI
Codex using GPT-5.6 Sol Ultra. The repository owner supplied the prompts, goals,
environment information, and scope decisions. Before initial publication, the
owner did not manually author, edit, or technically review the generated
material. The later reader-directed work described below changes the scope of
owner review, but it does not turn model-written text into manual authorship.

The commits present at initial publication were also prepared and created by
the model using the repository owner's configured Git identity. Their authorship
metadata does not indicate manual technical authorship.

This repository is published as a reusable generated artifact. Access to a
comparable model and workflow could produce a similar artifact; publishing this
version may save readers the time and tokens required to regenerate one.

## Post-publication reader review

On 2026-08-07, the repository owner began reading the published guide and
reported that Chapter 1 assumed too much compiler and GPU background, was too
sparse, and assigned upstream source ranges that were not meaningful in
isolation. In response, the model rewrote that chapter from first principles,
narrowed and annotated its source tour, added paper and runnable exercises, and
made related terminology corrections in the glossary and later chapters.

After seeing that revision, the owner requested the same first-principles,
stepwise improvement for every chapter and asked the model to be meticulous.
The model then rewrote Chapters 2–18 in chapter-scoped commits, expanded the
compiler/GPU background and question-led source stops, added or strengthened
the executable labs, and reconciled the setup, curriculum, navigation, and
reference material. The owner supplied the goals and feedback but did not
manually author or edit the replacement text.

The model checked examples, executable contracts, and source ranges against
the pinned tinygrad checkout and used additional model agents for adversarial
audits. Those are useful model-operated checks, not independent human technical
review. This reader-directed revision means the repository should no longer be
described as having received no owner review at all. The specific owner review
recorded here is the Chapter 1 feedback and the resulting guide-wide direction;
it does not establish comprehensive technical review of any chapter or of the
guide as a whole.

## Manual hardware run

On 2026-08-05, the repository owner manually ran the then-current bundled lab
runner on a user-reported Ubuntu system with an NVIDIA RTX 4090. The run used
the study checkout and its explicit virtual-environment interpreter, and
selected both included NVIDIA paths:

```bash
python3 scripts/run_labs.py \
  --tinygrad <pinned-study-checkout> \
  --python <pinned-study-checkout>/.venv/bin/python \
  --device CUDA --device NVK+NV
```

After the run, that checkout was confirmed at tinygrad commit
`874d33128b4e4785beea736d97df6716e0321717`. All 11 subprocesses selected by
that version of the runner completed without a failure or skip. They covered:

- three portable graph and rewrite labs with `DEV=PYTHON` and JIT enabled;
- the program-inspection and TinyJit labs with `DEV=PYTHON`;
- the TinyJit lab with JIT disabled;
- program inspection under Ada-targeted `DEV=PYTHON::sm_89` emulation; and
- the program-inspection and TinyJit labs on each of the real `CUDA` and
  `NVK+NV` backends.

The CUDA and `NVK+NV` program probes each compiled and executed the test
workload, reported an `sm_89` target, and returned the expected result. The JIT
probe completed its ignore, capture, and replay sequence with the expected
values on both backends.

## Post-rewrite model-operated run

On 2026-08-07, after the numbered-chapter rewrites and Chapter 18 integration,
OpenAI Codex ran the expanded runner against the same pinned source snapshot
with `CPU`, `CUDA`, and `NVK+NV` selected:

```bash
/usr/bin/python3 scripts/run_labs.py \
  --tinygrad /tmp/tinygrad-docs-review \
  --python /usr/bin/python3 \
  --device CPU --device CUDA --device NVK+NV
```

All 41 selected lab subprocesses completed without a failure or skip. This run
included the portable graph/compiler labs, the exact `CPU:CLANG` injected-fault
and fixed-regression modes, the full-tree contribution audit, the CPU runtime
and rendering paths, and the physical CUDA and NVK+NV runtime and
performance-mechanics probes. The physical probes compiled, executed,
synchronized, and returned the expected values through both NVIDIA routes; no
speed threshold was asserted. This was model-operated validation, not an
additional owner run or human review.

## Other automated checks

During generation and revision, the model also ran documentation/link checks,
validated every pinned blob/tree target and line range plus recorded source
symbols against the tinygrad snapshot, built the site in MkDocs strict mode,
and exercised targeted negative/adversarial cases. The repository's validation
workflow repeats the portable checks on GitHub Actions and deploys the rendered
site; it is not a hardware test.

## Limits of validation

The manual and model-operated runs were bounded checks of their respective
runner versions. They were not manual reviews of the guide's technical
explanations. Neither they nor the automated checks executed every command or
reader exercise in the prose, ran tinygrad's complete test suite, or
established performance, stress, concurrency, or exhaustive correctness
claims. The physical runs did not cover the `CUDA:PTX` route, and they say
nothing about hardware other than the reported setup or tinygrad revisions
other than the recorded snapshot.

Readers should verify important claims against the pinned source and re-check
live upstream documentation, policy, issues, and code before acting on them.
