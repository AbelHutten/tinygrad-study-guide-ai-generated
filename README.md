# tinygrad Contributor Guide

An unofficial, contribution-oriented path through tinygrad for readers who are
already comfortable with Python and machine learning, but are new to compilers
and GPU programming.

The guide is designed to be read from front to back. It combines concise
background explanations, deliberate source-reading, and hands-on exercises.
When a topic is too large to teach responsibly in place, the guide names the
missing prerequisite and links to a focused external resource.

The research baseline is tinygrad commit
[`874d331`](https://github.com/tinygrad/tinygrad/commit/874d33128b4e4785beea736d97df6716e0321717).
Concepts are written to remain useful across revisions; exact paths and symbols
are explicitly treated as snapshot-sensitive.

## Read locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-docs.txt
.venv/bin/mkdocs serve
```

Then open the address printed by MkDocs. A production build uses:

```bash
.venv/bin/mkdocs build --strict
```

The guide is independent of the tinygrad project. For tinygrad's official API
and developer documentation, see <https://docs.tinygrad.org/>.

After creating the pinned study checkout in Chapter 2, validate the executable
portable labs with:

```bash
python3 scripts/check_docs.py --tinygrad ../tinygrad-study
python3 scripts/run_labs.py --tinygrad ../tinygrad-study
```
