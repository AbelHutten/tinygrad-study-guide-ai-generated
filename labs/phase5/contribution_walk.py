#!/usr/bin/env python3
"""Audit contribution-readiness gates without changing or contacting tinygrad."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


if not __debug__:
  raise RuntimeError("contribution_walk.py requires assertions; do not run Python with -O")


# These settings constrain the lab's read-only Git subprocesses.  There is no
# tinygrad import, network client, or checkout write.  Fixed gate order and
# explicit predicates make both modes deterministic.
CONTROLLED_ENV = {
  "GIT_NO_REPLACE_OBJECTS": "1",
  "GIT_OPTIONAL_LOCKS": "0",
  "LANG": "C",
  "LC_ALL": "C",
  "NO_COLOR": "1",
}
for key, value in CONTROLLED_ENV.items():
  os.environ[key] = value

# Git normally honors these variables before inspecting cwd.  Inheriting one
# would let a caller redirect an apparent checkout audit to another repository
# or index.  Configuration injection is scrubbed for the same reason.
GIT_REDIRECTION_ENV = {
  "GIT_ALTERNATE_OBJECT_DIRECTORIES",
  "GIT_CEILING_DIRECTORIES",
  "GIT_COMMON_DIR",
  "GIT_DIR",
  "GIT_DISCOVERY_ACROSS_FILESYSTEM",
  "GIT_INDEX_FILE",
  "GIT_NAMESPACE",
  "GIT_OBJECT_DIRECTORY",
  "GIT_PREFIX",
  "GIT_WORK_TREE",
}


PINNED_COMMIT = "874d33128b4e4785beea736d97df6716e0321717"
REQUIRED_FILES = (
  ".github/workflows/test.yml",
  ".github/actions/process-replay/action.yml",
  "LICENSE",
  "README.md",
  "test/external/process_replay/README.md",
  "test/external/process_replay/process_replay.py",
)
PLACEHOLDERS = {"", "?", "n/a", "none", "todo", "tbd", "unknown"}


@dataclass(frozen=True)
class Packet:
  """One reviewable contribution claim; strings are evidence, not promises."""

  candidate: str = ""
  candidate_origin: str = ""
  live_state: str = ""
  contract: str = ""
  success: tuple[str, ...] = ()
  non_goals: tuple[str, ...] = ()
  reproducer: str = ""
  oracle: str = ""
  actual: str = ""
  baseline_red: str = ""
  first_bad_artifact: str = ""
  owning_layer: str = ""
  current_source_and_history: str = ""
  issue_pr_overlap: str = ""
  nearest_test: str = ""
  proposed_change: str = ""
  validation: tuple[str, ...] = ()
  performance_scope: str = ""
  hardware_scope: str = ""
  risk: str = ""
  rollback: str = ""
  commit_plan: tuple[str, ...] = ()
  communication: str = ""
  license_and_provenance: str = ""
  ai_disclosure: str = ""


@dataclass(frozen=True)
class Gate:
  gate_id: str
  stage: str
  question: str
  predicate: Callable[[Packet], bool]


def present(value: str) -> bool:
  return value.strip().lower() not in PLACEHOLDERS


def several(values: tuple[str, ...], minimum: int = 1) -> bool:
  return len(values) >= minimum and all(present(value) for value in values)


GATES = (
  Gate("T1-candidate", "triage", "Is one candidate named and sourced?",
       lambda p: present(p.candidate) and present(p.candidate_origin)),
  Gate("T2-live-state", "triage", "Is live ownership/status evidence recorded or explicitly inapplicable to a teaching case?",
       lambda p: present(p.live_state)),
  Gate("T3-contract", "triage", "Is the required behavior falsifiable?", lambda p: present(p.contract)),
  Gate("T4-boundaries", "triage", "Are success and non-goals explicit?",
       lambda p: several(p.success, 2) and several(p.non_goals, 2)),
  Gate("E1-reproduction", "evidence", "Can a reviewer reproduce expected and actual behavior?",
       lambda p: present(p.reproducer) and present(p.actual)),
  Gate("E2-oracle", "evidence", "Is expectation independent of the suspected implementation?",
       lambda p: present(p.oracle)),
  Gate("E3-baseline-red", "evidence", "Did the focused check fail on the unpatched baseline for the intended reason?",
       lambda p: present(p.baseline_red)),
  Gate("E4-localization", "evidence", "Are first bad artifact and owning layer named?",
       lambda p: present(p.first_bad_artifact) and present(p.owning_layer)),
  Gate("E5-current-context", "evidence", "Were current source/history and overlapping issue/PR work checked?",
       lambda p: present(p.current_source_and_history) and present(p.issue_pr_overlap)),
  Gate("P1-test", "patch", "Is the nearest stable regression location named?", lambda p: present(p.nearest_test)),
  Gate("P2-smallest-change", "patch", "Is the smallest owning-layer change stated?", lambda p: present(p.proposed_change)),
  Gate("P3-validation", "patch", "Are focused and proportional broader checks named?",
       lambda p: several(p.validation, 2)),
  Gate("P4-claim-limits", "patch", "Are performance and hardware claims explicitly bounded?",
       lambda p: present(p.performance_scope) and present(p.hardware_scope)),
  Gate("P5-recovery", "patch", "Are risk and rollback observable?", lambda p: present(p.risk) and present(p.rollback)),
  Gate("R1-atomic-history", "review", "Can each proposed commit be reviewed and validated alone?",
       lambda p: several(p.commit_plan)),
  Gate("R2-communication", "review", "Is upstream communication/stop policy explicit?",
       lambda p: present(p.communication)),
  Gate("R3-provenance", "review", "Are license, copied-source, and AI provenance addressed?",
       lambda p: present(p.license_and_provenance) and present(p.ai_disclosure)),
)


def incomplete_packet() -> Packet:
  """A tempting patch idea with deliberately missing contribution evidence."""
  return Packet(
    candidate="Make generated ADD code shorter",
    candidate_origin="self-chosen idea",
    contract="",
    success=("delete one line",),
    reproducer="",
    actual="the source looks verbose",
    proposed_change="combine the renderer branches",
    performance_scope="no measured performance claim",
    hardware_scope="no hardware claim",
    commit_plan=("renderer cleanup",),
    license_and_provenance="author wrote the proposed edit from the pinned MIT-licensed source",
    ai_disclosure="this teaching packet was generated with the guide and is not an upstream submission",
  )


def complete_packet() -> Packet:
  """A complete packet for the artificial Chapter 15 renderer-fault case."""
  return Packet(
    candidate="Artificial scalar-float32 ADD renderer fault from Chapter 15",
    candidate_origin="local teaching case; it is not an upstream tinygrad bug or bounty",
    live_state="upstream submission is explicitly out of scope; a real candidate requires a fresh UTC issue/policy/PR check",
    contract="For float32 x=[1,2,3], rendering and executing x+4 preserves ADD and returns [5,6,7]",
    success=(
      "standard SOURCE output store uses addition and equals the independent arithmetic oracle",
      "focused regression is red for the local faulty renderer and green for the standard renderer",
    ),
    non_goals=(
      "no claim that pinned or current upstream tinygrad contains this artificial fault",
      "no GPU, speedup, all-dtype, all-shape, or full-CI claim",
    ),
    reproducer="DEV=CPU:CLANG debugging_walk.py --mode injected, then --mode fixed, in fresh controlled processes",
    oracle="hand arithmetic [1+4,2+4,3+4] = [5,6,7], independent of tinygrad rendering",
    actual="fault mode produces exact [-3,-2,-1]; standard mode produces exact [5,6,7]",
    baseline_red="the same semantic assertion rejects the intentionally faulty a-b renderer before the standard route passes",
    first_bad_artifact="SOURCE: shared lowered LINEAR contains ADD, while faulty output store first changes to subtraction",
    owning_layer="the process-local FaultyClangRenderer operation-to-text mapping",
    current_source_and_history=("pinned cstyle renderer mapping and compiler boundary inspected; history is inapplicable "
                                "to a fabricated local subclass"),
    issue_pr_overlap="inapplicable by construction: no upstream issue, bounty, or PR is claimed; real work must search all three live surfaces",
    nearest_test=("upstream placement is inapplicable to the fabricated subclass; the exact stable guide regression is "
                  "labs/phase5/debugging_walk.py --mode fixed"),
    proposed_change="remove the local faulty mapping from the executed route; upstream checkout remains unchanged",
    validation=(
      "exact known-bad injected value and ADD-to-SUB source boundary",
      "exact standard-renderer value plus immutable upstream class mapping",
      "portable Python oracle control and hostile inherited-environment run",
    ),
    performance_scope="not claimed: no timings were collected or interpreted",
    hardware_scope="not claimed: CPU:CLANG and PYTHON do not establish GPU behavior",
    risk="a broad ALU mutation could corrupt index arithmetic, so the teaching fault is restricted to scalar float32 ADD",
    rollback="stop executing the local subclass and run the standard renderer regression; no checkout file needs restoration",
    commit_plan=("one educational lab plus its matching explanation; no unrelated runner, setup, or upstream edits",),
    communication="do not contact upstream about the artificial case; for real work ask only after current evidence reveals ambiguity or conflict",
    license_and_provenance=("this Chapter 18 lab contains no known adapted third-party implementation; the guide's adapted "
                            "phase-2 tinygrad fragment is disclosed with its retained local notice in THIRD_PARTY_NOTICES.md; "
                            "the guide and pinned tinygrad licenses are recorded as MIT"),
    ai_disclosure=("OpenAI Codex using GPT-5.6 Sol Ultra generated the guide and teaching packet from owner prompts; "
                   "real upstream use must follow and disclose under current policy"),
  )


def evaluate(packet: Packet) -> list[Gate]:
  return [gate for gate in GATES if not gate.predicate(packet)]


def file_digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def git_stdout(checkout: Path, *args: str) -> bytes:
  """Run one read-only Git query without inherited repository redirection."""
  git_env = os.environ.copy()
  for key in tuple(git_env):
    if key in GIT_REDIRECTION_ENV or key == "GIT_CONFIG" or key.startswith("GIT_CONFIG_"):
      git_env.pop(key)
  git_env.update(CONTROLLED_ENV)
  return subprocess.run(
    ["git", "--no-optional-locks", *args], cwd=checkout, env=git_env,
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
  ).stdout


def verify_head_tree(checkout: Path) -> int:
  """Compare every pinned HEAD blob and executable bit without trusting index flags."""
  object_format = git_stdout(checkout, "rev-parse", "--show-object-format").decode().strip()
  assert object_format == "sha1", f"pinned checkout unexpectedly uses {object_format} objects"

  tree: dict[str, tuple[str, str]] = {}
  for record in git_stdout(checkout, "ls-tree", "-rz", "HEAD").split(b"\0"):
    if not record: continue
    metadata, raw_path = record.split(b"\t", 1)
    mode, kind, object_id = metadata.decode().split()
    relative = raw_path.decode()
    assert kind == "blob" and mode in ("100644", "100755"), (relative, mode, kind)
    tree[relative] = (mode, object_id)

  index_records = [record for record in git_stdout(checkout, "ls-files", "-vz").split(b"\0") if record]
  unusual_flags = [(record[:1].decode(errors="replace"), record[2:].decode(errors="replace"))
                   for record in index_records if record[:2] != b"H "]
  assert not unusual_flags, f"nonstandard index flags present: {unusual_flags[:10]}"
  index: dict[str, tuple[str, str]] = {}
  for record in git_stdout(checkout, "ls-files", "-sz").split(b"\0"):
    if not record: continue
    metadata, raw_path = record.split(b"\t", 1)
    mode, object_id, stage = metadata.decode().split()
    assert stage == "0", f"unmerged index entry: {raw_path.decode(errors='replace')}"
    index[raw_path.decode()] = (mode, object_id)
  assert index == tree, "index paths, modes, or object IDs differ from pinned HEAD"

  failures: list[str] = []
  for relative, (mode, expected_id) in tree.items():
    path = checkout / relative
    try:
      path_stat = path.lstat()
      if not stat.S_ISREG(path_stat.st_mode):
        failures.append(f"{relative}: expected regular file")
        continue
      expected_executable = mode == "100755"
      if bool(path_stat.st_mode & stat.S_IXUSR) != expected_executable:
        failures.append(f"{relative}: executable bit differs")
        continue
      contents = path.read_bytes()
    except OSError as error:
      failures.append(f"{relative}: {error}")
      continue
    digest = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents, usedforsecurity=False).hexdigest()
    if digest != expected_id: failures.append(f"{relative}: content differs")
  assert not failures, f"worktree differs from pinned HEAD: {failures[:10]}"
  return len(tree)


def audit_checkout(checkout: Path) -> dict[str, str]:
  """Read exact pinned policy/process facts and compare two file observations."""
  checkout = checkout.resolve()
  assert checkout.is_dir(), f"tinygrad checkout does not exist: {checkout}"

  top_level = Path(git_stdout(checkout, "rev-parse", "--show-toplevel").decode().strip()).resolve()
  assert top_level == checkout, f"Git top level is {top_level}, expected {checkout}"
  head = git_stdout(checkout, "rev-parse", "HEAD").decode().strip()
  assert head == PINNED_COMMIT, f"checkout is {head}, expected {PINNED_COMMIT}"
  status_before = git_stdout(checkout, "status", "--porcelain=v1", "--untracked-files=no")
  assert status_before == b"", f"tracked checkout changes present: {status_before.decode(errors='replace').strip()}"
  tracked_count = verify_head_tree(checkout)

  paths = {relative: checkout / relative for relative in REQUIRED_FILES}
  assert all(path.is_file() for path in paths.values()), f"not the expected tinygrad checkout: {checkout}"
  before_bytes = {relative: path.read_bytes() for relative, path in paths.items()}
  head_bytes = {relative: git_stdout(checkout, "show", f"HEAD:{relative}") for relative in REQUIRED_FILES}
  mismatched = [relative for relative in REQUIRED_FILES if before_bytes[relative] != head_bytes[relative]]
  assert not mismatched, f"required worktree files differ from HEAD: {mismatched}"

  readme = before_bytes["README.md"].decode()
  license_text = before_bytes["LICENSE"].decode()
  replay_readme = before_bytes["test/external/process_replay/README.md"].decode()
  replay_code = before_bytes["test/external/process_replay/process_replay.py"].decode()
  workflow = before_bytes[".github/workflows/test.yml"].decode()
  action = before_bytes[".github/actions/process-replay/action.yml"].decode()

  assert "If you are a new contributor with something that looks even close to AI written" in readme
  assert "Bug fixes (with a regression test) are great!" in readme
  assert "Anything you claim is a \"speedup\" must be benchmarked." in readme
  assert "Permission is hereby granted, free of charge" in license_text
  assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in license_text
  assert "By default, process replay doesn't assert kernel diffs." in replay_readme
  assert "[pr]" in replay_readme
  assert 'ASSERT_DIFF = int((flag:="[PR]")' in replay_code
  assert "changed = 0" in replay_code and "if changed > MAX_DIFF_PCT:" in replay_code
  assert "CAPTURE_PROCESS_REPLAY:" in workflow and "contains(github.event.pull_request.title, '[pr]')" in workflow
  assert 'export PR_TITLE=$(jq -r .pull_request.title "$GITHUB_EVENT_PATH")' in action
  assert 'export COMMIT_MESSAGE=$(git show -s --format=%B "$CURRENT_SHA")' in action

  before = {relative: hashlib.sha256(contents).hexdigest() for relative, contents in before_bytes.items()}
  after = {relative: file_digest(path) for relative, path in paths.items()}
  assert before == after, "a required file differed between the lab's two observations"
  status_after = git_stdout(checkout, "status", "--porcelain=v1", "--untracked-files=no")
  assert status_after == b"", f"tracked checkout changes appeared: {status_after.decode(errors='replace').strip()}"
  assert verify_head_tree(checkout) == tracked_count
  return {
    "checkout": str(checkout),
    "head": head,
    "python": platform.python_version(),
    "policy": "AI disclosure + regression tests + benchmarked speed claims",
    "license": "MIT",
    "process_replay": "kernel diff; workflow has a lowercase literal, Python assertion code tests uppercase [PR]",
    "tracked_worktree_clean": "True",
    "head_tracked_files_verified": str(tracked_count),
    "required_files_match_head": "True",
    "required_files_equal_at_two_observations": "True",
  }


def run_case(case: str, packet: Packet) -> None:
  failures = evaluate(packet)
  failure_ids = tuple(gate.gate_id for gate in failures)

  if case == "incomplete":
    expected = (
      "T2-live-state", "T3-contract", "T4-boundaries", "E1-reproduction",
      "E2-oracle", "E3-baseline-red",
      "E4-localization", "E5-current-context", "P1-test", "P3-validation",
      "P5-recovery", "R2-communication",
    )
    assert failure_ids == expected, (failure_ids, expected)
    print("case: incomplete-patch-idea")
    print("decision: RESEARCH")
    print("failed gates:", list(failure_ids))
    for gate in failures: print(f"  {gate.gate_id} [{gate.stage}]: {gate.question}")
    print("proposed edit present:", present(packet.proposed_change))
    print("lesson: a plausible edit is not a contribution-ready claim")
    print("limit: passing fields still require human verification; this checker cannot prove their truth")
    print("status: expected-incompleteness-detected")
  else:
    assert failure_ids == ()
    print("case: complete-artificial-evidence-packet")
    print("decision: READY for the bounded teaching claim")
    print("passed gates:", len(GATES), "of", len(GATES))
    for stage in ("triage", "evidence", "patch", "review"):
      print(f"  {stage}:", sum(gate.stage == stage for gate in GATES), "passed")
    print("contract:", packet.contract)
    print("first bad / owner:", packet.first_bad_artifact, "/", packet.owning_layer)
    print("upstream action: none; artificial case, not an upstream bug")
    print("status: complete-evidence-packet-passed")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tinygrad", required=True, type=Path, help="path to the pinned tinygrad study checkout")
  parser.add_argument("--case", required=True, choices=("incomplete", "complete"))
  args = parser.parse_args()

  facts = audit_checkout(args.tinygrad)
  print("controlled env: GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 LANG=C LC_ALL=C NO_COLOR=1")
  print("read-only audit:", facts)
  run_case(args.case, incomplete_packet() if args.case == "incomplete" else complete_packet())


if __name__ == "__main__":
  main()
