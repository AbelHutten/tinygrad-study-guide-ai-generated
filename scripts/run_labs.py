#!/usr/bin/env python3
"""Run the guide's executable labs against a separate tinygrad checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_LABS = (
  ROOT / "labs/phase1/first_trace.py",
  ROOT / "labs/phase2/frontend_autograd.py",
  ROOT / "labs/phase2/uop_walk.py",
  ROOT / "labs/phase2/rewrite_lab.py",
  ROOT / "labs/phase3/schedule_walk.py",
  ROOT / "labs/phase3/shapes_and_indexing.py",
)
RUNTIME_LABS = (
  ROOT / "labs/phase3/inspect_program.py",
  ROOT / "labs/phase4/jit_three_calls.py",
)


def run_lab(python: Path, checkout: Path, lab: Path, device: str, cache_dir: Path, jit: int = 1) -> None:
  env = os.environ.copy()
  old_pythonpath = env.get("PYTHONPATH")
  env.update({
    "DEV": device,
    "DEBUG": "0",
    "JIT": str(jit),
    "NO_MEMORY_PLANNER": "0",
    "CACHEDB": str(cache_dir / f"{lab.stem}-{device.replace(':', '_')}-jit{jit}.db"),
    "PYTHONPATH": str(checkout) + (os.pathsep + old_pythonpath if old_pythonpath else ""),
  })
  command = [str(python), str(lab)]
  print(f"\n==> DEV={device} JIT={jit} {lab.relative_to(ROOT)}", flush=True)
  subprocess.run(command, cwd=checkout, env=env, check=True)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--tinygrad", required=True, type=Path, help="path to the pinned tinygrad study checkout")
  parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable to use")
  parser.add_argument(
    "--device", action="append", default=[], metavar="DEV",
    help="additional runtime matrix entry, such as CPU, CUDA, CUDA:PTX, or NVK+NV; repeat as needed",
  )
  args = parser.parse_args()

  checkout, python = args.tinygrad.resolve(), args.python.resolve()
  if not (checkout / "tinygrad/__init__.py").is_file():
    parser.error(f"not a tinygrad checkout: {checkout}")
  if not python.is_file(): parser.error(f"Python executable does not exist: {python}")

  with tempfile.TemporaryDirectory(prefix="tinygrad-guide-labs-") as cache_name:
    cache_dir = Path(cache_name)
    for lab in PORTABLE_LABS: run_lab(python, checkout, lab, "PYTHON", cache_dir)
    for lab in RUNTIME_LABS: run_lab(python, checkout, lab, "PYTHON", cache_dir)
    run_lab(python, checkout, ROOT / "labs/phase4/jit_three_calls.py", "PYTHON", cache_dir, jit=0)

    # This route exercises NVIDIA-targeted lowering without claiming to test a
    # physical GPU, driver, concurrent execution, launch behavior, or performance.
    run_lab(python, checkout, ROOT / "labs/phase3/inspect_program.py", "PYTHON::sm_89", cache_dir)

    for device in args.device:
      for lab in RUNTIME_LABS: run_lab(python, checkout, lab, device, cache_dir)

  print("\nAll selected labs passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
