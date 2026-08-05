# Contributing to this guide

This repository is an independent learning resource, not tinygrad's upstream
documentation tree. Changes should improve a reader's ability to investigate
and validate tinygrad itself.

## Content standard

A source-heavy chapter should contain:

- a durable mental model separated from snapshot-sensitive implementation;
- links pinned to the commit in `upstream-snapshot.toml`;
- a small source tour that names responsibilities, not just files;
- a predict-before-running exercise with an observable oracle;
- correctness and failure cases, not only the happy path;
- a checkpoint and compact quick reference; and
- focused external prerequisites with a return condition.

Prefer executable probes and existing upstream tests over copied source. Avoid
asserting a full UOp `repr`, generated variable name, timing, or other brittle
artifact unless that exact representation is the lesson.

## Validate a change

Create the documentation environment as described in `README.md`, then run:

```bash
.venv/bin/mkdocs build --strict
python3 scripts/check_docs.py
```

For source and lab validation, check out the recorded tinygrad commit separately
and run:

```bash
python3 scripts/check_docs.py --tinygrad ../tinygrad-study
```

Run every executable command changed by the patch. Portable labs should be
verified on the backend they name. Performance results must record the device,
backend, renderer, commit, warm-up, synchronization, sample distribution, and
correctness oracle.

## Update the tinygrad snapshot

Do not merely replace the commit hash. To advance the snapshot:

1. change the metadata and symbol expectations in `upstream-snapshot.toml`;
2. update the source table in `docs/reference/source-snapshot.md`;
3. replace pinned GitHub links;
4. run the source checker against the new exact commit;
5. execute all labs;
6. inspect source-heavy explanations for semantic and pass-order changes; and
7. record migration notes where a returning reader could otherwise be misled.

The checker rejects mixed snapshot hashes and source links to moving
`master`/`main` branches.

## Keep changes reviewable

Separate mechanical link/snapshot updates from conceptual rewrites when
possible. Do not commit `site/`, virtual environments, caches, captured traces,
or benchmark output. Explain what learner capability improves and how the
change was validated.
