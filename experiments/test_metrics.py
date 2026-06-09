import io
import os
import sys
import unittest
import ast
import py_compile
import re
import shutil
import subprocess
import tempfile
import coverage

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from coverage.exceptions import NoDataError
from utils.config import PROGRAM_FILE, TEST_FILE

def _empty_coverage_stats() -> Dict[str, Any]:
    return {
        "line_total": 0,
        "line_covered": 0,
        "line_percent": 0.0,
        "branch_total": 0,
        "branch_covered": 0,
        "branch_percent": 0.0,
    }


def _compile_file(path: Path, label: str) -> Optional[str]:
    if not path.exists():
        return f"{label} file not found: {path}"
    try:
        py_compile.compile(str(path), doraise=True)
        return None
    except py_compile.PyCompileError as exc:
        return f"{label} file does not compile; skipping metric. {exc}"

def _run_tests(test_path: Path, base_dir: Path) -> unittest.TestResult:
    test_dir = test_path.parent
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    try:
        os.chdir(base_dir)
        sys.path.insert(0, str(base_dir))
        outputs_dir = base_dir / "outputs"
        if str(outputs_dir) not in sys.path:
            sys.path.insert(0, str(outputs_dir))
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
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate a concise line/branch coverage report for a test run.
    
    Args:
        test_path: The path to the test file to run from the project root.
        patch_path: The path to the patch file to analyze coverage for from the project root.
    """
    base_dir = Path(__file__).resolve().parents[1]

    test_path = (base_dir / TEST_FILE).resolve()
    patch_path = (base_dir / PROGRAM_FILE).resolve()

    cov = coverage.Coverage(branch=True, source=[str(patch_path.parent)])
    cov.start()
    try:
        test_result = _run_tests(test_path, base_dir)
    finally:
        cov.stop()

    coverage_stats, warning = _analyze_coverage(cov, patch_path)

    report = {
        "test_file": str(TEST_FILE),
        "patch_file": str(PROGRAM_FILE),
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

def _parse_last_int(patterns: list[str], text: str) -> int:
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
    return 0


def _parse_mutmut_results(text: str) -> dict[str, int]:
    return {
        "killed": _parse_last_int(
            [r"🎉\s*(\d+)", r"\bkilled\D+(\d+)", r"(\d+)\s+mutants?\s+killed"],
            text,
        ),
        "survived": _parse_last_int(
            [r"🙁\s*(\d+)", r"\bsurvived\D+(\d+)", r"(\d+)\s+mutants?\s+survived"],
            text,
        ),
        "timeout": _parse_last_int(
            [r"⏰\s*(\d+)", r"\btimeout\D+(\d+)", r"\btimed\s*out\D+(\d+)"],
            text,
        ),
        "suspicious": _parse_last_int(
            [r"🤔\s*(\d+)", r"\bsuspicious\D+(\d+)"],
            text,
        ),
    }


def _empty_mutation_result(warning: str = "") -> dict:
    result = {
        "mutation_score":     -1.0,
        "mutants_killed":     -1,
        "mutants_survived":   -1,
        "mutants_timeout":    -1,
        "mutants_suspicious": -1,
        "mutants_total":      -1,
    }
    if warning:
        result["mutation_warning"] = warning
    return result


def _copy_metric_inputs(
    *,
    temp_root: Path,
    code_file: Path,
    test_file: Path,
    code_text: Optional[str],
) -> tuple[Path, Path]:
    temp_code = temp_root / "generated_program.py"
    temp_test = temp_root / "test_generated_program.py"

    source_code = code_text if code_text is not None else code_file.read_text(encoding="utf-8")
    temp_code.write_text(source_code.rstrip() + "\n", encoding="utf-8")

    test_content = test_file.read_text(encoding="utf-8")
    test_content = test_content.replace(
        "from outputs.generated_program import",
        "from generated_program import",
    )
    temp_test.write_text(test_content.rstrip() + "\n", encoding="utf-8")

    (temp_root / "conftest.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent))\n",
        encoding="utf-8",
    )
    return temp_code, temp_test


def _write_mutmut_config(temp_root: Path, temp_code: Path, temp_test: Path) -> None:
    rel_code = temp_code.relative_to(temp_root).as_posix()
    (temp_root / "setup.cfg").write_text(
        "[mutmut]\n"
        f"paths_to_mutate={rel_code}\n"
        f"runner={sys.executable} -m pytest {temp_test} -x -q\n"
        "backup=False\n"
        "\n"
        "[tool:pytest]\n"
        "testpaths = .\n",
        encoding="utf-8",
    )


def _run_pytest_baseline(
    temp_root: Path,
    temp_test: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    rel_test = temp_test.relative_to(temp_root).as_posix()
    return subprocess.run(
        [sys.executable, "-m", "pytest", rel_test, "-q", "--tb=short"],
        cwd=temp_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def compute_mutation_score(
    code_file: Path = PROGRAM_FILE,
    test_file: Path = TEST_FILE,
    timeout: int = 120,
    code_text: Optional[str] = None,
    debug: bool = False,
    keep_temp_dir: bool = False,
) -> dict:
    """
    Mutate code_file and run test_file against each mutant to compute mutation score.

    If code_text is provided it is used as the implementation under test (e.g. the
    canonical HumanEval solution) and code_file is ignored for the source content.
    """
    base_dir = Path(__file__).resolve().parents[1]

    code_path = (base_dir / code_file).resolve() if not Path(code_file).is_absolute() else Path(code_file)
    test_path = (base_dir / test_file).resolve() if not Path(test_file).is_absolute() else Path(test_file)

    mutmut_exec = Path(sys.executable).parent / "mutmut"
    if not mutmut_exec.exists():
        mutmut_exec = shutil.which("mutmut")
    if mutmut_exec is None:
        warning = "mutmut executable not found; install requirements before computing mutation score."
        print(f"Warning: {warning}")
        return _empty_mutation_result(warning)
    mutmut_exec = str(mutmut_exec)

    if code_text is None and not code_path.exists():
        warning = f"code file not found for mutation scoring: {code_path}"
        print(f"Warning: {warning}")
        return _empty_mutation_result(warning)

    if not test_path.exists():
        warning = f"test file not found for mutation scoring: {test_path}"
        print(f"Warning: {warning}")
        return _empty_mutation_result(warning)

    if code_text is not None:
        try:
            ast.parse(code_text)
        except SyntaxError as exc:
            warning = f"code text does not compile; skipping mutation score. {exc}"
            print(f"Warning: {warning}")
            return _empty_mutation_result(warning)
    else:
        compile_warning = _compile_file(code_path, "Code")
        if compile_warning:
            print(f"Warning: {compile_warning}")
            return _empty_mutation_result(compile_warning)

    compile_warning = _compile_file(test_path, "Test")
    if compile_warning:
        print(f"Warning: {compile_warning}")
        return _empty_mutation_result(compile_warning)

    cleanup_context = None
    temp_root = None

    try:
        if keep_temp_dir:
            temp_root = Path(tempfile.mkdtemp(prefix="tdlite-mutmut-"))
        else:
            cleanup_context = tempfile.TemporaryDirectory(prefix="tdlite-mutmut-")
            temp_root = Path(cleanup_context.name)

        temp_code, temp_test = _copy_metric_inputs(
            temp_root=temp_root,
            code_file=code_path,
            test_file=test_path,
            code_text=code_text,
        )
        _write_mutmut_config(temp_root, temp_code, temp_test)

        if debug:
            print(f"\n[Mutation Debug] Temp dir: {temp_root}")
            print(f"\n[Mutation Debug] Code:\n{temp_code.read_text(encoding='utf-8')}")
            print(f"\n[Mutation Debug] Tests:\n{temp_test.read_text(encoding='utf-8')}")

        baseline = _run_pytest_baseline(temp_root, temp_test, timeout)

        if debug:
            print(f"\n[Mutation Debug] Baseline stdout:\n{baseline.stdout}")
            print(f"\n[Mutation Debug] Baseline stderr:\n{baseline.stderr}")

        if baseline.returncode != 0:
            warning = (
                "baseline tests failed before mutation; mutation score is invalid.\n"
                + (baseline.stdout + baseline.stderr)[-2000:]
            )
            print(f"Warning: {warning}")
            return _empty_mutation_result(warning)

        run_proc = subprocess.run(
            [mutmut_exec, "run"],
            cwd=temp_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if debug:
            print(f"\n[Mutation Debug] mutmut run stdout:\n{run_proc.stdout}")
            print(f"\n[Mutation Debug] mutmut run stderr:\n{run_proc.stderr}")

        results_proc = subprocess.run(
            [mutmut_exec, "results"],
            cwd=temp_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if debug:
            print(f"\n[Mutation Debug] mutmut results stdout:\n{results_proc.stdout}")
            print(f"\n[Mutation Debug] mutmut results stderr:\n{results_proc.stderr}")

        combined_output = (
            run_proc.stdout + run_proc.stderr + "\n" + results_proc.stdout + results_proc.stderr
        )
        counts = _parse_mutmut_results(combined_output)

        killed   = counts["killed"]
        survived = counts["survived"]
        timeout_count = counts["timeout"]
        suspicious    = counts["suspicious"]
        total    = killed + survived + timeout_count + suspicious
        detected = killed + timeout_count
        score    = round(detected / total, 4) if total > 0 else 0.0

        warning = ""
        if run_proc.returncode not in (0, 1) and total == 0:
            warning = "mutmut did not produce parseable results. " + combined_output[-2000:]
        elif total == 0:
            warning = "mutmut produced zero mutants for the target file."
        elif suspicious:
            warning = f"{suspicious} suspicious mutants need manual inspection."

        print(
            f"[Mutation Score] killed={killed} survived={survived} "
            f"timeout={timeout_count} suspicious={suspicious} "
            f"total={total} score={score:.2%}"
        )

        result = {
            "mutation_score":     score,
            "mutants_killed":     killed,
            "mutants_survived":   survived,
            "mutants_timeout":    timeout_count,
            "mutants_suspicious": suspicious,
            "mutants_total":      total,
        }
        if warning:
            result["mutation_warning"] = warning
        return result

    except subprocess.TimeoutExpired:
        warning = "mutmut timed out globally."
        print(f"Warning: {warning}")
        return _empty_mutation_result(warning)

    except Exception as exc:
        warning = f"mutation scoring failed: {exc}"
        print(f"Warning: {warning}")
        return _empty_mutation_result(warning)

    finally:
        if keep_temp_dir and temp_root is not None:
            print(f"\n[Mutation Debug] Kept temp directory: {temp_root}")
        elif cleanup_context is not None:
            cleanup_context.cleanup()
