from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from utils.state import AgentState

DEFAULT_SPECIFICATION_FILE = Path("outputs/specifications.txt")
DEFAULT_RESULTS_CSV = "results.csv"
EXPERIMENT_FILES_DIR = "experiment_files"


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


def _read_existing_header(csv_path: Path) -> Optional[list[str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, None)


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


def _build_row(final_state: AgentState, coverage_report: Dict[str, Any]) -> Dict[str, Any]:
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

    for agent in ("a11", "a12", "a2", "a3", "a4"):
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
    coverage_report: Dict[str, Any],
    results_folder: str,
    specification_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Append a pipeline run summary to a CSV file under experiments/results/<folder>.

    Args:
        final_state: Final pipeline state produced by the graph.
        coverage_report: Coverage report dict from generate_coverage_report().
        results_folder: Folder name for this experiment's results.
        specification_path: Optional path to the specification file.

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

    test_file_value = coverage_report.get("test_file")
    patch_file_value = coverage_report.get("patch_file")
    if not test_file_value:
        raise ValueError("coverage_report is missing 'test_file'.")
    if not patch_file_value:
        raise ValueError("coverage_report is missing 'patch_file'.")

    base_dir = Path(__file__).resolve().parents[1]
    test_path = _resolve_path(str(test_file_value), base_dir)
    patch_path = _resolve_path(str(patch_file_value), base_dir)
    if specification_path is None:
        specification_path = DEFAULT_SPECIFICATION_FILE
    spec_path = _resolve_path(str(specification_path), base_dir)

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    test_snapshot = instance_dir / test_path.name
    patch_snapshot = instance_dir / patch_path.name
    spec_snapshot = instance_dir / spec_path.name
    task_snapshot = instance_dir / "task.txt"

    _write_snapshot(_read_text(test_path), test_snapshot)
    _write_snapshot(_read_text(patch_path), patch_snapshot)
    _write_snapshot(_read_text(spec_path), spec_snapshot)
    _write_snapshot(final_state.get("program_description", ""), task_snapshot)

    agent_outputs_path = instance_dir / "agent_outputs.json"
    coverage_report_path = instance_dir / "coverage_report.json"
    run_summary_path = instance_dir / "run_summary.json"
    _write_json(_build_agent_outputs(final_state), agent_outputs_path)
    _write_json(coverage_report, coverage_report_path)

    row = _build_row(final_state, coverage_report)
    row["instance_id"] = instance_id
    row["instance_dir"] = str(Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}")
    row["test_file_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / test_path.name
    )
    row["patch_file_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / patch_path.name
    )
    row["specification_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / spec_path.name
    )
    row["task_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / task_snapshot.name
    )
    row["agent_outputs_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / agent_outputs_path.name
    )
    row["coverage_report_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / coverage_report_path.name
    )
    row["run_summary_snapshot"] = str(
        Path(EXPERIMENT_FILES_DIR) / f"instance_{instance_id}" / run_summary_path.name
    )
    _write_json(row, run_summary_path)

    header = _read_existing_header(csv_path)
    fieldnames: Iterable[str] = header or list(row.keys())

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return instance_dir

def append_experiments_results(results: dict, results_folder: str, timestamp: str) -> None:
    results_dir = create_results_dir(results_folder)
    summary = {
        "timestamp": timestamp,
        "total":     len(results),
        "passed":    sum(1 for r in results if r["humaneval_pass"]),
        "pass_rate": sum(1 for r in results if r["humaneval_pass"]) / len(results),
        "avg_iters": sum(r["iterations"] for r in results) / len(results),
        "results":   results,
    }
    _write_json(summary, results_dir / f"summary_{timestamp}.json")
