# An Independent Study Guide to tinygrad

> [!IMPORTANT]
> **Independent, unofficial, AI-generated material**
>
> This repository is an independently published learning resource about
> tinygrad. It is not part of the tinygrad project and is not affiliated with,
> maintained by, sponsored by, approved by, or endorsed by the tinygrad
> project, tiny corp, or tinygrad's maintainers. For authoritative information,
> use [tinygrad's official repository](https://github.com/tinygrad/tinygrad) and
> [official documentation](https://docs.tinygrad.org/). Questions and
> corrections about this guide belong in this repository, not in tinygrad's
> issue tracker or support channels.
>
> The initial guide and its supporting files were generated end-to-end by
> OpenAI Codex using GPT-5.6 Sol Ultra, from prompts and scope decisions
> supplied by the repository owner. Before initial publication, the owner did
> not manually author, edit, or technically review the generated material. The
> owner is now reading the guide and directing model-written revisions; this
> does not imply comprehensive technical review of the guide. The owner also
> manually ran the bundled lab runner on Ubuntu with an RTX 4090, selecting both
> `CUDA` and `NVK+NV`; all selections in that run passed. This was a limited
> smoke test, not verification of every explanation, command, exercise, device,
> or tinygrad revision. See
> [Provenance and validation](docs/reference/provenance.md) for the exact scope.
>
> This generated artifact is published so others can use it without spending
> the time and tokens required to generate a similar guide.

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

## License

This repository's documentation and code are released under the permissive
[MIT License](LICENSE). Third-party projects and linked materials retain their
own terms; the one adapted tinygrad test fragment is identified in
[Third-party notices](THIRD_PARTY_NOTICES.md).
