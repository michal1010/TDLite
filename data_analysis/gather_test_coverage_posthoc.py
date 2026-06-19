#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Scenario prefix (folder) -> list of batch subfolders, exactly as named on disk.
SCENARIOS: dict[str, list[str]] = {
    "default":    ["default_0_54", "default_55_109", "default_110_163"],
    "refine":     ["REF_0_54", "REF_55_109", "REF_110_163"],
    "tdd":        ["tdd_0_50", "tdd_51_105", "tdd_106_163"],
    "tdd_refine": ["TDD_REF_0_54", "TDD_REF_55_109", "TDD_REF_110_163"],
}

# Name of the program file under test (as it appears inside coverage.json,
# regardless of any outputs/ prefix or the path separator).
PROGRAM_FILENAME = "generated_program.py"

# What to report:
#   False -> coverage of the program file under test only (recommended): this
#            answers "how well do the generated tests cover the generated
#            program". The test file is excluded.
#   True  -> coverage.json's combined "totals" (program file + test file).
USE_TOTALS = False

OUTPUT_CSV_NAME = "test_results_posthoc.csv"

# Base directory = wherever this script lives (the data_analysis/ folder).
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def instance_number(instance_dir: Path) -> int:
    """Extract the integer N from a folder named 'instance_N'."""
    m = re.search(r"instance_(\d+)$", instance_dir.name)
    return int(m.group(1)) if m else -1


def pct(covered: int, total: int) -> float:
    """Percentage covered. coverage.py reports 100% when there is nothing to cover."""
    return 100.0 if total == 0 else round(100.0 * covered / total, 2)


def program_summary(data: dict) -> dict | None:
    """Return the summary dict for the file under test, or None if absent."""
    for key, section in data.get("files", {}).items():
        basename = key.replace("\\", "/").rsplit("/", 1)[-1]
        if basename == PROGRAM_FILENAME:
            # Per-file sections nest the numbers under "summary".
            return section.get("summary", section)
    return None


def blank_metrics() -> dict:
    return {
        "line_coverage": 0,
        "branch_coverage": 0,
        "covered_lines": None,
        "num_statements": None,
        "covered_branches": None,
        "num_branches": None,
    }


def extract_metrics(coverage_path: Path) -> dict:
    """
    Read one coverage.json and return line/branch coverage metrics.
    Returns blanks if the file is missing, unreadable, or lacks the target file.
    """
    if not coverage_path.is_file():
        return blank_metrics()

    try:
        with coverage_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    WARNING: could not read {coverage_path}: {exc}")
        return blank_metrics()

    if USE_TOTALS:
        summary = data.get("totals", {})
    else:
        summary = program_summary(data)
        if summary is None:
            print(f"    WARNING: {PROGRAM_FILENAME} not in {coverage_path}; row left blank")
            return blank_metrics()

    covered_lines = summary.get("covered_lines", 0)
    num_statements = summary.get("num_statements", 0)
    covered_branches = summary.get("covered_branches", 0)
    num_branches = summary.get("num_branches", 0)

    # Prefer coverage.py's own percentages; compute as a fallback.
    line_cov = summary.get("percent_statements_covered")
    line_cov = round(line_cov, 2) if line_cov is not None else pct(covered_lines, num_statements)

    branch_cov = summary.get("percent_branches_covered")
    branch_cov = round(branch_cov, 2) if branch_cov is not None else pct(covered_branches, num_branches)

    return {
        "line_coverage": line_cov,
        "branch_coverage": branch_cov,
        "covered_lines": covered_lines,
        "num_statements": num_statements,
        "covered_branches": covered_branches,
        "num_branches": num_branches,
    }


# ---------------------------------------------------------------------------
# Per-batch processing
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "instance",
    "line_coverage",
    "branch_coverage",
    "covered_lines",
    "num_statements",
    "covered_branches",
    "num_branches",
]


def process_batch(scenario: str, subfolder: str) -> None:
    batch_dir = BASE_DIR / scenario / subfolder
    exp_dir = batch_dir / "experiment_files"

    if not exp_dir.is_dir():
        print(f"  WARNING: missing experiment_files dir, skipping: {exp_dir}")
        return

    # Map instance number -> folder.
    instances: dict[int, Path] = {}
    for d in exp_dir.glob("instance_*"):
        if d.is_dir():
            n = instance_number(d)
            if n >= 1:
                instances[n] = d

    if not instances:
        print(f"  WARNING: no instance_* folders in {exp_dir}")
        return

    max_n = max(instances)
    rows = []
    # Iterate 1..max_n so that row position == instance number, even with gaps.
    for n in range(1, max_n + 1):
        instance_dir = instances.get(n)
        if instance_dir is None:
            print(f"    WARNING: instance_{n} folder missing; writing blank row")
            metrics = blank_metrics()
        else:
            metrics = extract_metrics(instance_dir / "coverage.json")
        rows.append({"instance": n, **metrics})

    out_path = batch_dir / OUTPUT_CSV_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  wrote {out_path}  ({len(rows)} rows)")


def main() -> None:
    print(f"Base directory : {BASE_DIR}")
    print(f"Reporting      : {'combined totals' if USE_TOTALS else PROGRAM_FILENAME + ' only'}\n")
    for scenario, subfolders in SCENARIOS.items():
        for subfolder in subfolders:
            print(f"Batch: {scenario}/{subfolder}")
            process_batch(scenario, subfolder)
    print("\nDone.")


if __name__ == "__main__":
    main()