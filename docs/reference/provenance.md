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
environment information, and scope decisions. Other than running the validation
command recorded below, the owner did not manually author, edit, or technically
review the generated material.

The commits present at initial publication were also prepared and created by
the model using the repository owner's configured Git identity. Their authorship
metadata does not indicate manual technical authorship.

This repository is published as a reusable generated artifact. Access to a
comparable model and workflow could produce a similar artifact; publishing this
version may save readers the time and tokens required to regenerate one.

## Manual hardware run

On 2026-08-05, the repository owner manually ran the bundled lab runner on a
user-reported Ubuntu system with an NVIDIA RTX 4090. The run used the study
checkout supplied to the runner and selected both included NVIDIA paths:

```bash
python3 scripts/run_labs.py --tinygrad <pinned-study-checkout> \
  --device CUDA --device NVK+NV
```

After the run, that checkout was confirmed at tinygrad commit
`874d33128b4e4785beea736d97df6716e0321717`. All 11 subprocesses selected by the
runner completed without a failure or skip. They covered:

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

## Other automated checks

During generation, the model also ran documentation/link checks, checked pinned
source paths and symbols against the recorded tinygrad checkout, built the site
in MkDocs strict mode, and exercised the portable, CPU, and Ada-targeted runner
paths. The repository's validation workflow repeats the portable checks on
GitHub Actions; it is not a hardware test.

## Limits of validation

The manual run was a smoke test of the bundled lab runner. It was not a manual
review of the guide's technical explanations. Neither it nor the automated
checks executed every command or reader exercise in the prose, ran tinygrad's
complete test suite, or established performance, stress, concurrency, or
exhaustive correctness claims. The run did not cover the `CUDA:PTX` route, and
it says nothing about hardware other than the reported setup or tinygrad
revisions other than the recorded snapshot.

Readers should verify important claims against the pinned source and re-check
live upstream documentation, policy, issues, and code before acting on them.
