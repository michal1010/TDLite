import io
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import coverage
from coverage.exceptions import NoDataError
from utils.config import PROGRAM_FILE, TEST_FILE


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _run_tests(test_path: Path, base_dir: Path) -> unittest.TestResult:
    test_dir = test_path.parent
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    try:
        os.chdir(base_dir)
        sys.path.insert(0, str(base_dir))
        if str(test_dir) not in sys.path:
            sys.path.insert(0, str(test_dir))

        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(test_dir), pattern=test_path.name)
        runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
        return runner.run(suite)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path


def _safe_percent(covered: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((covered / total) * 100.0, 2)


def _analyze_coverage(
    cov: coverage.Coverage, patch_path: Path
) -> Tuple[Dict[str, Any], Optional[str]]:
    data = cov.get_data()
    measured_files = {Path(path).resolve() for path in data.measured_files()}
    if patch_path not in measured_files:
        return (
            {
                "line_total": 0,
                "line_covered": 0,
                "line_percent": 0.0,
                "branch_total": 0,
                "branch_covered": 0,
                "branch_percent": 0.0,
            },
            "No coverage data collected for the patch file.",
        )

    analysis_obj = None
    if hasattr(cov, "_analyze"):
        try:
            analysis_obj = cov._analyze(str(patch_path))
        except Exception:
            analysis_obj = None

    line_total = line_missing = 0
    branch_total = branch_missing = 0

    if analysis_obj is not None:
        line_total = getattr(analysis_obj, "n_statements", 0)
        if not line_total and hasattr(analysis_obj, "statements"):
            line_total = len(getattr(analysis_obj, "statements", []))

        line_missing = getattr(analysis_obj, "n_missing", 0)
        if not line_missing and hasattr(analysis_obj, "missing"):
            line_missing = len(getattr(analysis_obj, "missing", []))

        branch_total = getattr(analysis_obj, "n_branches", 0)
        branch_missing = getattr(analysis_obj, "n_missing_branches", 0)
    else:
        try:
            analysis = cov.analysis2(str(patch_path))
            statements = analysis[1]
            missing = analysis[3]
            line_total = len(statements)
            line_missing = len(missing)
            if len(analysis) > 4:
                missing_branches = analysis[4]
                branch_missing = len(missing_branches)
        except NoDataError:
            return (
                {
                    "line_total": 0,
                    "line_covered": 0,
                    "line_percent": 0.0,
                    "branch_total": 0,
                    "branch_covered": 0,
                    "branch_percent": 0.0,
                },
                "No coverage data collected for the patch file.",
            )

    line_covered = max(line_total - line_missing, 0)
    branch_covered = max(branch_total - branch_missing, 0)

    return (
        {
            "line_total": line_total,
            "line_covered": line_covered,
            "line_percent": _safe_percent(line_covered, line_total),
            "branch_total": branch_total,
            "branch_covered": branch_covered,
            "branch_percent": _safe_percent(branch_covered, branch_total),
        },
        None,
    )


def generate_coverage_report(
    test_path: Path = TEST_FILE,
    patch_path: Path = PROGRAM_FILE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate a concise line/branch coverage report for a test run.
    
    Args:
        test_path: The path to the test file to run from the project root.
        patch_path: The path to the patch file to analyze coverage for from the project root.
    """
    base_dir = Path(__file__).resolve().parents[1]

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    cov = coverage.Coverage(branch=True, source=[str(patch_path.parent)])
    cov.start()
    try:
        test_result = _run_tests(test_path, base_dir)
    finally:
        cov.stop()

    coverage_stats, warning = _analyze_coverage(cov, patch_path)

    report = {
        "test_file": str(test_path),
        "patch_file": str(patch_path),
        "tests_run": test_result.testsRun,
        "tests_failed": len(test_result.failures),
        "tests_errors": len(test_result.errors),
        "tests_passed": test_result.wasSuccessful(),
        "line_coverage": {
            "percent": coverage_stats["line_percent"],
            "covered": coverage_stats["line_covered"],
            "total": coverage_stats["line_total"],
        },
        "branch_coverage": {
            "percent": coverage_stats["branch_percent"],
            "covered": coverage_stats["branch_covered"],
            "total": coverage_stats["branch_total"],
        },
    }

    if warning:
        report["warning"] = warning

    if verbose:
      print("Coverage report")
      print(f"Tests: {report['tests_run']} run, {report['tests_failed']} failed, {report['tests_errors']} errors")
      print(
          "Line coverage: "
          f"{report['line_coverage']['percent']:.2f}% "
          f"({report['line_coverage']['covered']}/{report['line_coverage']['total']})"
      )
      print(
          "Branch coverage: "
          f"{report['branch_coverage']['percent']:.2f}% "
          f"({report['branch_coverage']['covered']}/{report['branch_coverage']['total']})"
      )
      if warning:
          print(f"Warning: {warning}")

    return report

# Quick test
if __name__ == "__main__":
    generate_coverage_report()