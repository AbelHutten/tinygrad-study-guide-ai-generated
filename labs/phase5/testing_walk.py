#!/usr/bin/env python3
"""Make one test go deliberately red, then run the same contract against tinygrad."""

from __future__ import annotations

import argparse
import os
import unittest
from collections.abc import Callable


if not __debug__:
  raise RuntimeError("testing_walk.py requires assertions; do not run Python with -O")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=("red", "green"))
  return parser.parse_args()


args = parse_args()
requested_dev = os.environ.get("DEV")
if requested_dev != "PYTHON":
  raise RuntimeError(f"testing_walk.py requires the explicit environment DEV='PYTHON'; received {requested_dev!r}")


# These variables are read while tinygrad modules are imported.  Pin them before
# importing tinygrad so the lab does not silently inherit optimizer, JIT, cache,
# visualization, or validation choices from the reader's shell.
for key, value in {
  "BEAM": "0",
  "CACHELEVEL": "0",
  "CAPTURING": "0",
  "CCACHE": "0",
  "CHECK_OOB": "0",
  "DEBUG": "0",
  "DEBUG_RANGEIFY": "0",
  "DEFAULT_FLOAT": "float32",
  "DEFAULT_INT": "int32",
  "DISALLOW_BROADCAST": "0",
  "HCQ2": "0",
  "IMAGE": "0",
  "JIT": "0",
  "NO_COLOR": "1",
  "NOLOCALS": "0",
  "NOOPT": "1",
  "PROFILE": "0",
  "SCACHE": "0",
  "SPEC": "2",
  "TC": "0",
  "THREADS": "1",
  "TRACK_MATCH_STATS": "0",
  "VALIDATE_WITH_CPU": "0",
  "VIZ": "0",
}.items():
  os.environ[key] = value

from tinygrad import Device, Tensor, dtypes  # noqa: E402


Candidate = Callable[[list[float], int, int], list[float]]


def independent_oracle(data: list[float], rows: int, cols: int) -> list[float]:
  """Loop model for (x.permute(1, 0) + 1).sum(axis=1).

  It deliberately shares neither Tensor movement operations nor Tensor
  reduction code with the implementation under test.
  """
  assert len(data) == rows * cols
  return [sum(data[row * cols + col] + 1.0 for row in range(rows)) for col in range(cols)]


def row_sum_mutant(data: list[float], rows: int, cols: int) -> list[float]:
  """Known-bad stand-in: it forgets the transpose and reduces original rows."""
  assert len(data) == rows * cols
  return [sum(data[row * cols + col] + 1.0 for col in range(cols)) for row in range(rows)]


def tinygrad_candidate(data: list[float], rows: int, cols: int) -> list[float]:
  x = Tensor(data, device=Device.DEFAULT, dtype=dtypes.float32).reshape(rows, cols)
  return (x.permute(1, 0) + 1.0).sum(axis=1).tolist()


def deterministic_data(rows: int, cols: int) -> list[float]:
  # Small integers and sums are exactly representable in float32.  Exact
  # equality therefore tests the intended contract rather than roundoff policy.
  return [float((index * 7 + rows * 3 + cols) % 17 - 8) for index in range(rows * cols)]


def make_contract_case(candidate: Candidate) -> type[unittest.TestCase]:
  """Create one reusable test contract for a candidate implementation."""

  class MovementReductionContract(unittest.TestCase):
    maxDiff = None

    def test_00_weak_symmetric_square(self) -> None:
      # This tempting one-example test is intentionally weak: transposing a
      # symmetric square matrix does not change its row sums, so the mutant
      # passes.  A green test can still have almost no power against the bug.
      data = [0.0, 1.0,
              1.0, 0.0]
      self.assertEqual(candidate(data, 2, 2), independent_oracle(data, 2, 2))

    def test_10_focused_rectangular_counterexample(self) -> None:
      # This is the smallest readable counterexample used to explain the
      # historical symptom.  Shape and values are written literally on purpose.
      data = [0.0, 1.0, 2.0,
              3.0, 4.0, 5.0]
      self.assertEqual(independent_oracle(data, 2, 3), [5.0, 7.0, 9.0])
      self.assertEqual(candidate(data, 2, 3), [5.0, 7.0, 9.0])

    def test_20_output_shape_contract(self) -> None:
      # After (rows, cols) -> (cols, rows) -> reduce axis 1, exactly cols
      # values remain.  This is a semantic shape contract, not an incidental
      # assertion about temporary UOp order or generated variable names.
      for rows, cols in ((1, 4), (2, 3), (4, 1)):
        data = deterministic_data(rows, cols)
        self.assertEqual(len(candidate(data, rows, cols)), cols)

    def test_30_bounded_differential_grid(self) -> None:
      # A deterministic grid covers degenerate, square, and rectangular shapes.
      # It is bounded so a failure stays cheap and directly reproducible.
      for rows in range(1, 5):
        for cols in range(1, 6):
          with self.subTest(rows=rows, cols=cols):
            data = deterministic_data(rows, cols)
            self.assertEqual(candidate(data, rows, cols), independent_oracle(data, rows, cols))

    def test_40_add_constant_metamorphic_relation(self) -> None:
      # This relation needs no full numerical oracle: adding delta to every
      # input must add rows*delta to each reduced column.  It complements, but
      # does not replace, the independent loop model above.
      rows, cols, delta = 2, 5, 3.0
      data = deterministic_data(rows, cols)
      base = candidate(data, rows, cols)
      shifted = candidate([value + delta for value in data], rows, cols)
      self.assertEqual(len(base), cols)
      self.assertEqual(shifted, [value + rows * delta for value in base])

  MovementReductionContract.__name__ = "MovementReductionContract"
  MovementReductionContract.__qualname__ = "MovementReductionContract"
  return MovementReductionContract


def run_contract(candidate: Candidate) -> unittest.TestResult:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(make_contract_case(candidate))
  result = unittest.TestResult()
  suite.run(result)
  return result


def failed_test_names(result: unittest.TestResult) -> list[str]:
  # unittest reports every failing subTest separately and appends its parameter
  # description to id().  Collapse those records to the owning test method.
  names = {test.id().rsplit(".", 1)[-1].split(" (", 1)[0] for test, _ in (*result.failures, *result.errors)}
  return sorted(names)


def demonstrate_red() -> None:
  result = run_contract(row_sum_mutant)
  failures = failed_test_names(result)
  expected = [
    "test_10_focused_rectangular_counterexample",
    "test_20_output_shape_contract",
    "test_30_bounded_differential_grid",
    "test_40_add_constant_metamorphic_relation",
  ]

  assert result.testsRun == 5
  assert not result.wasSuccessful()
  assert len(result.failures) == 22, len(result.failures)
  assert result.errors == [], [(test.id(), traceback) for test, traceback in result.errors]
  assert failures == expected, (failures, expected)

  print("mode: deliberate-red")
  print("candidate: known-bad row-sum mutant")
  print("tests run:", result.testsRun)
  print("assertion failures/errors:", len(result.failures), len(result.errors))
  print("weak symmetric example passed:", "test_00_weak_symmetric_square" not in failures)
  print("failed contract tests:", failures)
  print("red reason: transpose was omitted before the reduction")
  print("status: expected-regression-detected")


def run_regression() -> None:
  assert Device.DEFAULT == "PYTHON", f"regression mode requires DEV=PYTHON, got {Device.DEFAULT!r}"
  result = run_contract(tinygrad_candidate)
  failures = failed_test_names(result)

  assert result.testsRun == 5
  assert result.wasSuccessful(), failures

  # This is an artifact observation for localization, not part of the reusable
  # semantic contract.  A future legal canonicalization could change these op
  # names without changing the expression's result.
  probe = (Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=dtypes.float32)
           .permute(1, 0) + 1.0).sum(axis=1)
  frontend_ops = [node.op.name for node in probe.uop.toposort()]
  assert probe.shape == (3,)
  assert "PERMUTE" in frontend_ops and "REDUCE" in frontend_ops
  assert probe.tolist() == [5.0, 7.0, 9.0]

  print("mode: tinygrad-green")
  print("device:", Device.DEFAULT)
  print("tests run:", result.testsRun)
  print("failures/errors:", len(result.failures), len(result.errors))
  print("focused result/oracle:", probe.tolist(), [5.0, 7.0, 9.0])
  print("localization observation: frontend shape", probe.shape)
  print("localization observation: frontend ops", frontend_ops)
  print("claim: semantic contract passed on the portable Python route")
  print("non-claim: no compiled renderer, driver, GPU, timing, or full CI matrix was tested")
  print("status: regression-passed")


def main() -> None:
  print("controlled env: BEAM=0 CACHELEVEL=0 CAPTURING=0 CCACHE=0 CHECK_OOB=0 DEBUG=0 DEBUG_RANGEIFY=0 "
        "DEFAULT_FLOAT=float32 DEFAULT_INT=int32 DISALLOW_BROADCAST=0 HCQ2=0 IMAGE=0 JIT=0 NO_COLOR=1 "
        "NOLOCALS=0 NOOPT=1 PROFILE=0 SCACHE=0 SPEC=2 TC=0 THREADS=1 TRACK_MATCH_STATS=0 "
        "VALIDATE_WITH_CPU=0 VIZ=0")
  if args.mode == "red": demonstrate_red()
  else: run_regression()


if __name__ == "__main__":
  main()
