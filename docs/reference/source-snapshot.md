# Source snapshot

Source-specific material in this generated guide targets:

| Field | Value |
| --- | --- |
| Repository | [`tinygrad/tinygrad`](https://github.com/tinygrad/tinygrad) |
| Commit | [`874d33128b4e4785beea736d97df6716e0321717`](https://github.com/tinygrad/tinygrad/commit/874d33128b4e4785beea736d97df6716e0321717) |
| Commit date | 2026-08-05 |
| Package version | `0.13.0` |
| Required Python | 3.11 or newer |

## Reproduce the snapshot

Use a dedicated study checkout so you do not detach or overwrite contribution
work:

```bash
git clone https://github.com/tinygrad/tinygrad.git tinygrad-study
cd tinygrad-study
git switch --detach 874d33128b4e4785beea736d97df6716e0321717
```

## Translate the guide to current master

Before making a real change:

1. update your normal tinygrad checkout;
2. find the named symbol with `rg`, rather than assuming its old path;
3. read callers, tests, and recent history around the symbol;
4. rerun the chapter's observation command on current `master`; and
5. record any semantic difference, not just a rename.

Useful commands:

```bash
rg -n 'def create_linear_with_vars|def to_program|class UOp' tinygrad
git log --oneline --follow -- tinygrad/schedule/rangeify.py
git blame -L 555,580 tinygrad/schedule/rangeify.py
```

Line numbers deliberately appear only in snapshot permalinks. Prose refers to
symbols and responsibilities because those survive ordinary refactors better.

## Update policy for this guide

An update should change the recorded snapshot only after all executable labs
and source links have been checked against the new commit. If a symbol moved but
the model stayed valid, update the link. If behavior or pass ordering changed,
update the explanation and add a short migration note where an existing reader
would otherwise be misled.
