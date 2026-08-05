#!/usr/bin/env python3
"""Check local Markdown links and the tinygrad source snapshot contract."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SNAPSHOT_FILE = ROOT / "upstream-snapshot.toml"
LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
PINNED_TINYGRAD_RE = re.compile(
  r"https://github\.com/tinygrad/tinygrad/(?:blob|tree|commit)/([0-9a-f]{40})(?:[/)#?]|$)"
)
MOVING_SOURCE_RE = re.compile(
  r"https://github\.com/tinygrad/tinygrad/(?:blob|tree)/(?:master|main)(?:[/)#?]|$)"
)


def load_snapshot() -> dict:
  with SNAPSHOT_FILE.open("rb") as f:
    return tomllib.load(f)


def markdown_files() -> list[Path]:
  return sorted([ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOCS.rglob("*.md")])


def check_markdown(snapshot: dict) -> list[str]:
  errors: list[str] = []
  expected_commit = snapshot["commit"]
  for md in markdown_files():
    text = md.read_text(encoding="utf-8")
    relative_name = md.relative_to(ROOT)

    for match in PINNED_TINYGRAD_RE.finditer(text):
      if match.group(1) != expected_commit:
        errors.append(f"{relative_name}: tinygrad link pins {match.group(1)}, expected {expected_commit}")
    if MOVING_SOURCE_RE.search(text):
      errors.append(f"{relative_name}: source link uses moving master/main instead of the recorded commit")

    for line_number, line in enumerate(text.splitlines(), 1):
      for match in LINK_RE.finditer(line):
        raw_target = match.group(1).strip()
        # Drop an optional Markdown title after a whitespace separator.
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith(("#", "//")):
          continue
        path_part = unquote(parsed.path)
        if not path_part:
          continue
        resolved = (md.parent / path_part).resolve()
        if not resolved.exists():
          errors.append(f"{relative_name}:{line_number}: missing link target {target!r}")
  return errors


def check_tinygrad_checkout(snapshot: dict, checkout: Path) -> list[str]:
  errors: list[str] = []
  if not checkout.is_dir():
    return [f"tinygrad checkout does not exist: {checkout}"]

  try:
    actual_commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    return [f"cannot read tinygrad checkout commit at {checkout}: {exc}"]

  if actual_commit != snapshot["commit"]:
    errors.append(f"tinygrad checkout is {actual_commit}, expected {snapshot['commit']}")

  for symbol in snapshot["symbols"]:
    path = checkout / symbol["path"]
    if not path.is_file():
      errors.append(f"{symbol['label']}: missing {symbol['path']}")
      continue
    text = path.read_text(encoding="utf-8")
    if re.search(symbol["pattern"], text, re.MULTILINE) is None:
      errors.append(f"{symbol['label']}: pattern {symbol['pattern']!r} not found in {symbol['path']}")
  return errors


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--tinygrad",
    type=Path,
    help="path to a tinygrad checkout; also verify its commit and source-symbol manifest",
  )
  args = parser.parse_args()

  snapshot = load_snapshot()
  errors = check_markdown(snapshot)
  if args.tinygrad is not None:
    errors.extend(check_tinygrad_checkout(snapshot, args.tinygrad.resolve()))

  if errors:
    print("documentation checks failed:", file=sys.stderr)
    for error in errors:
      print(f"- {error}", file=sys.stderr)
    return 1

  source_note = f" and {len(snapshot['symbols'])} source symbols" if args.tinygrad is not None else ""
  print(f"checked {len(markdown_files())} Markdown files{source_note} against {snapshot['commit'][:8]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
