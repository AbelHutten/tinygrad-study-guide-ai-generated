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
SOURCE_LINE_FRAGMENT_RE = re.compile(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?$")
MOVING_SOURCE_RE = re.compile(
  r"https://github\.com/tinygrad/tinygrad/(?:blob|tree)/(?:master|main)(?:[/)#?]|$)"
)
LIVE_SOURCE_MARKER = "<!-- live-upstream -->"


def load_snapshot() -> dict:
  with SNAPSHOT_FILE.open("rb") as f:
    return tomllib.load(f)


def markdown_files() -> list[Path]:
  return sorted([
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    *DOCS.rglob("*.md"),
  ])


def check_markdown(snapshot: dict) -> list[str]:
  errors: list[str] = []
  expected_commit = snapshot["commit"]
  for md in markdown_files():
    text = md.read_text(encoding="utf-8")
    relative_name = md.relative_to(ROOT)

    for match in PINNED_TINYGRAD_RE.finditer(text):
      if match.group(1) != expected_commit:
        errors.append(f"{relative_name}: tinygrad link pins {match.group(1)}, expected {expected_commit}")
    for line_number, line in enumerate(text.splitlines(), 1):
      if MOVING_SOURCE_RE.search(line) and LIVE_SOURCE_MARKER not in line:
        errors.append(
          f"{relative_name}:{line_number}: source link uses moving master/main without {LIVE_SOURCE_MARKER}"
        )
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


def check_tinygrad_source_links(snapshot: dict, checkout: Path) -> tuple[list[str], int]:
  """Verify every pinned GitHub blob/tree target and any explicit line range."""
  errors: list[str] = []
  checked = 0
  expected_commit = snapshot["commit"]
  prefix = ("tinygrad", "tinygrad")

  for md in markdown_files():
    relative_name = md.relative_to(ROOT)
    for line_number, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
      for match in LINK_RE.finditer(line):
        raw_target = match.group(1).strip()
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme != "https" or parsed.netloc != "github.com":
          continue

        parts = tuple(part for part in unquote(parsed.path).split("/") if part)
        if len(parts) < 4 or parts[:2] != prefix or parts[2] not in {"blob", "tree"}:
          continue
        kind, commit = parts[2], parts[3]
        if commit != expected_commit:
          # check_markdown reports the commit mismatch with a less redundant message.
          continue
        checked += 1

        source_parts = parts[4:]
        if not source_parts and kind == "blob":
          errors.append(f"{relative_name}:{line_number}: pinned blob link has no source path: {target!r}")
          continue
        source_path = checkout.joinpath(*source_parts).resolve()
        try:
          source_path.relative_to(checkout)
        except ValueError:
          errors.append(f"{relative_name}:{line_number}: source link escapes checkout: {target!r}")
          continue

        expected_kind = "file" if kind == "blob" else "directory"
        exists = source_path.is_file() if kind == "blob" else source_path.is_dir()
        if not exists:
          errors.append(
            f"{relative_name}:{line_number}: pinned {kind} target is not a {expected_kind}: "
            f"{'/'.join(source_parts)!r}"
          )
          continue

        if not parsed.fragment:
          continue
        if kind != "blob":
          errors.append(f"{relative_name}:{line_number}: line fragment on non-blob source link: {target!r}")
          continue
        range_match = SOURCE_LINE_FRAGMENT_RE.fullmatch(parsed.fragment)
        if range_match is None:
          errors.append(f"{relative_name}:{line_number}: unsupported source line fragment #{parsed.fragment}")
          continue

        start = int(range_match.group(1))
        end = int(range_match.group(2) or range_match.group(1))
        if end < start:
          errors.append(f"{relative_name}:{line_number}: reversed source line range #{parsed.fragment}")
          continue
        with source_path.open("r", encoding="utf-8", errors="replace") as source_file:
          line_count = sum(1 for _ in source_file)
        if end > line_count:
          errors.append(
            f"{relative_name}:{line_number}: source range #{parsed.fragment} exceeds "
            f"{'/'.join(source_parts)} ({line_count} lines)"
          )

  return errors, checked


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
  source_links_checked = 0
  if args.tinygrad is not None:
    checkout = args.tinygrad.resolve()
    errors.extend(check_tinygrad_checkout(snapshot, checkout))
    source_errors, source_links_checked = check_tinygrad_source_links(snapshot, checkout)
    errors.extend(source_errors)

  if errors:
    print("documentation checks failed:", file=sys.stderr)
    for error in errors:
      print(f"- {error}", file=sys.stderr)
    return 1

  source_note = (
    f", {source_links_checked} pinned source links, and {len(snapshot['symbols'])} source symbols"
    if args.tinygrad is not None else ""
  )
  print(f"checked {len(markdown_files())} Markdown files{source_note} against {snapshot['commit'][:8]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
