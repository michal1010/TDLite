from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from utils.state import AgentState

DEFAULT_SPECIFICATION_FILE = Path("outputs/specifications.txt")
DEFAULT_PROGRAM_FILE = Path("outputs/generated_program.py")
DEFAULT_TEST_FILE = Path("outputs/test_generated_program.py")
DEFAULT_RESULTS_CSV = "results.csv"
EXPERIMENT_FILES_DIR = "experiment_files"
AGENTS = ("a11", "a12", "a2", "a3", "a4")


def _normalize_folder_name(folder_name: str) -> str:
    name = Path(folder_name).name
    return name or "results"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sum_tokens(tokens: Dict[str, Any]) -> int:
    return sum(_safe_int(value) for value in tokens.values())


def _build_token_summary_from_results(results: list[dict]) -> Dict[str, Any]:
    by_agent = {}
    for agent in AGENTS:
        input_total = sum(_safe_int(r.get("input_tokens", {}).get(agent)) for r in results)
        output_total = sum(_safe_int(r.get("output_tokens", {}).get(agent)) for r in results)
        by_agent[agent] = {
            "input": input_total,
            "output": output_total,
        }

    total_input = sum(tokens["input"] for tokens in by_agent.values())
    total_output = sum(tokens["output"] for tokens in by_agent.values())
    return {
        "input": total_input,
        "output": total_output,
        "by_agent": by_agent,
    }


def _read_existing_header(csv_path: Path) -> Optional[list[str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, None)


def _rewrite_csv_with_header(csv_path: Path, fieldnames: list[str]) -> None:
    """Preserve old result rows while adding newly introduced metric columns."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_snapshot(content: str, output_path: Path) -> None:
    output_path.write_text(content, encoding="utf-8")


def _write_json(content: Dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(content, indent=2),
        encoding="utf-8",
    )


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        return sum(1 for _ in reader)


def _max_instance_id(experiment_files_dir: Path) -> int:
    if not experiment_files_dir.exists():
        return 0

    max_id = 0
    for path in experiment_files_dir.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("instance_"):
            continue
        suffix = name.split("instance_", 1)[-1]
        if suffix.isdigit():
            max_id = max(max_id, int(suffix))
    return max_id


def _next_instance_id(csv_path: Path, experiment_files_dir: Path) -> int:
    csv_rows = _count_csv_rows(csv_path)
    existing = _max_instance_id(experiment_files_dir)
    return max(csv_rows, existing) + 1


def _build_row(final_state: AgentState, coverage_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    coverage_report = coverage_report or {}
    input_tokens = final_state.get("input_tokens", {})
    output_tokens = final_state.get("output_tokens", {})
    line_coverage = coverage_report.get("line_coverage", {})
    branch_coverage = coverage_report.get("branch_coverage", {})

    row: Dict[str, Any] = {}
    row["pipeline_tests_passed"] = final_state.get("tests_passed", False)
    row["pipeline_iterations"] = final_state.get("iteration", 0)

    row["coverage_tests_run"] = coverage_report.get("tests_run", 0)
    row["coverage_tests_failed"] = coverage_report.get("tests_failed", 0)
    row["coverage_tests_errors"] = coverage_report.get("tests_errors", 0)
    row["coverage_tests_passed"] = coverage_report.get("tests_passed", False)
    row["coverage_test_file"] = coverage_report.get("test_file", "")
    row["coverage_patch_file"] = coverage_report.get("patch_file", "")

    row["line_coverage_percent"] = line_coverage.get("percent", 0.0)
    row["line_coverage_covered"] = line_coverage.get("covered", 0)
    row["line_coverage_total"] = line_coverage.get("total", 0)
    row["branch_coverage_percent"] = branch_coverage.get("percent", 0.0)
    row["branch_coverage_covered"] = branch_coverage.get("covered", 0)
    row["branch_coverage_total"] = branch_coverage.get("total", 0)
    row["coverage_warning"] = coverage_report.get("warning", "")

    row["total_input_tokens"] = _sum_tokens(input_tokens)
    row["total_output_tokens"] = _sum_tokens(output_tokens)

    row["mutation_score"]       = -1.0
    row["mutants_killed"]       = -1
    row["mutants_survived"]     = -1
    row["mutants_timeout"]      = -1
    row["mutants_suspicious"]   = -1
    row["mutants_total"]        = -1
    row["mutation_warning"]     = ""

    for agent in AGENTS:
        row[f"{agent}_input_tokens"] = _safe_int(input_tokens.get(agent, 0))
        row[f"{agent}_output_tokens"] = _safe_int(output_tokens.get(agent, 0))

    return row


def _build_agent_outputs(final_state: AgentState) -> Dict[str, Any]:
    """Collect the generated artifacts that only live in the pipeline state."""
    return {
        "program_description": final_state.get("program_description", ""),
        "a11_specifications": final_state.get("specifications", ""),
        "a12_tests": final_state.get("tests", ""),
        "final_code": final_state.get("code", ""),
        "compile_output": final_state.get("compile_output", ""),
        "test_outputs": final_state.get("test_outputs", []),
        "debug_reports": final_state.get("debug_reports", []),
        "ever_passed_tests": final_state.get("ever_passed_tests", []),
        "tests_passed": final_state.get("tests_passed", False),
        "iteration": final_state.get("iteration", 0),
        "input_tokens": final_state.get("input_tokens", {}),
        "output_tokens": final_state.get("output_tokens", {}),
    }


def create_results_dir(results_folder: str) -> Path:
    results_root = Path(__file__).resolve().parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    folder_name = _normalize_folder_name(results_folder)
    results_dir = results_root / folder_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def append_single_run_results(
    final_state: AgentState,
    coverage_report: Optional[Dict[str, Any]],
    results_folder: str,
    mutation_results: Optional[Dict[str, Any]] = None,
    specification_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    hidden_test: Optional[str] = None,
    save_generated_test_artifacts: bool = True,
) -> Path:
    """
    Append a pipeline run summary to a CSV file under experiments/results/<folder>.

    Args:
        final_state: Final pipeline state produced by the graph.
        coverage_report: Coverage report dict from generate_coverage_report(), if run.
        results_folder: Folder name for this experiment's results.
        mutation_results: Results of mutation score testing
        specification_path: Optional path to the specification file.
        hidden_test: Optional HumanEval hidden test code to snapshot.
        save_generated_test_artifacts: Whether to snapshot generated tests/specifications.

    Returns:
        The per-run directory containing snapshots and JSON artifacts.
    """
    results_dir = create_results_dir(results_folder)

    experiment_files_dir = results_dir / EXPERIMENT_FILES_DIR
    experiment_files_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / DEFAULT_RESULTS_CSV
    instance_id = _next_instance_id(csv_path, experiment_files_dir)
    instance_dir = experiment_files_dir / f"instance_{instance_id}"
    print(f"{instance_dir}")
    while instance_dir.exists():
        instance_id += 1
        instance_dir = experiment_files_dir / f"instance_{instance_id}"
    instance_dir.mkdir(parents=True, exist_ok=False)

    base_dir = Path(__file__).resolve().parents[1]
    coverage_report = coverage_report or {}
    patch_file_value = coverage_report.get("patch_file", DEFAULT_PROGRAM_FILE)
    patch_path = _resolve_path(str(patch_file_value), base_dir)

    test_path = None
    if save_generated_test_artifacts:
        test_file_value = coverage_report.get("test_file", DEFAULT_TEST_FILE)
        test_path = _resolve_path(str(test_file_value), base_dir)

    if specification_path is None:
        specification_path = DEFAULT_SPECIFICATION_FILE
    spec_path = _resolve_path(str(specification_path), base_dir) if save_generated_test_artifacts else None

    if test_path is not None and not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    if spec_path is not None and not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    patch_snapshot = instance_dir / patch_path.name
    task_snapshot = instance_dir / "task.txt"
    hidden_test_snapshot = instance_dir / "hidden_test.py"

    if test_path is not None:
        _write_snapshot(_read_text(test_path), instance_dir / test_path.name)
    _write_snapshot(_read_text(patch_path), patch_snapshot)
    if spec_path is not None:
        _write_snapshot(_read_text(spec_path), instance_dir / spec_path.name)
    _write_snapshot(final_state.get("program_description", ""), task_snapshot)
    if hidden_test is not None:
        _write_snapshot(hidden_test, hidden_test_snapshot)

    agent_outputs_path = instance_dir / "agent_outputs.json"
    coverage_report_path = instance_dir / "coverage_report.json"
    run_summary_path = instance_dir / "run_summary.json"
    _write_json(_build_agent_outputs(final_state), agent_outputs_path)
    if coverage_report:
        _write_json(coverage_report, coverage_report_path)

    row = _build_row(final_state, coverage_report)
    if mutation_results:
        row.update({
            "mutation_score":     mutation_results.get("mutation_score", -1.0),
            "mutants_killed":     mutation_results.get("mutants_killed", -1),
            "mutants_survived":   mutation_results.get("mutants_survived", -1),
            "mutants_timeout":    mutation_results.get("mutants_timeout", -1),
            "mutants_suspicious": mutation_results.get("mutants_suspicious", -1),
            "mutants_total":      mutation_results.get("mutants_total", -1),
            "mutation_warning":   mutation_results.get("mutation_warning", ""),
        })
    row["instance_id"] = instance_id

    _write_json(row, run_summary_path)

    header = _read_existing_header(csv_path)
    fieldnames = list(header or row.keys())

    added_columns = False
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
            added_columns = True

    if header and added_columns:
        _rewrite_csv_with_header(csv_path, fieldnames)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return instance_dir

def append_experiments_results(results: list[dict], results_folder: str, timestamp: str) -> None:
    results_dir = create_results_dir(results_folder)
    total = len(results)
    passed = sum(1 for r in results if r["humaneval_pass"])
    avg_line_coverage = sum(r.get("line_coverage_percent", 0) for r in results) / total if total else 0
    avg_branch_coverage = sum(r.get("branch_coverage_percent", 0) for r in results) / total if total else 0
    avg_iters = sum(r["iterations"] for r in results) / total if total else 0
    valid_mutation = [r.get("mutation_score", -1.0) for r in results if r.get("mutation_score", -1.0) >= 0]
    avg_mutation_score = sum(valid_mutation) / len(valid_mutation) if valid_mutation else -1.0
    
    summary = {
        "timestamp": timestamp,
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "avg_iters": avg_iters,
        "tokens": _build_token_summary_from_results(results),
        "avg_line_coverage": avg_line_coverage,
        "avg_branch_coverage": avg_branch_coverage,
        "avg_mutation_score": avg_mutation_score,
    }
    _write_json(summary, results_dir / f"summary_{timestamp}.json")
