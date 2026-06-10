"""
Post-hoc mutation scoring for a completed experiment run.

Usage:
    python -m experiments.run_mutation_posthoc --results_folder <folder> [--debug]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.test_metrics import compute_mutation_score


MUTATION_FIELDS = [
    "mutation_score", "mutants_killed", "mutants_survived",
    "mutants_timeout", "mutants_suspicious", "mutants_total", "mutation_warning",
]


def _update_run_summary(summary_path: Path, mutation_results: dict) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data.update({
        "mutation_score":     mutation_results.get("mutation_score", -1.0),
        "mutants_killed":     mutation_results.get("mutants_killed", -1),
        "mutants_survived":   mutation_results.get("mutants_survived", -1),
        "mutants_timeout":    mutation_results.get("mutants_timeout", -1),
        "mutants_suspicious": mutation_results.get("mutants_suspicious", -1),
        "mutants_total":      mutation_results.get("mutants_total", -1),
        "mutation_warning":   mutation_results.get("mutation_warning", ""),
    })
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _patch_csv(csv_path: Path, instance_id: int, mutation_results: dict) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []
    for field in MUTATION_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        if str(row.get("instance_id")) == str(instance_id):
            row["mutation_score"]     = mutation_results.get("mutation_score", -1.0)
            row["mutants_killed"]     = mutation_results.get("mutants_killed", -1)
            row["mutants_survived"]   = mutation_results.get("mutants_survived", -1)
            row["mutants_timeout"]    = mutation_results.get("mutants_timeout", -1)
            row["mutants_suspicious"] = mutation_results.get("mutants_suspicious", -1)
            row["mutants_total"]      = mutation_results.get("mutants_total", -1)
            row["mutation_warning"]   = mutation_results.get("mutation_warning", "")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_posthoc(results_folder: str, debug: bool = False) -> None:
    results_root = Path(__file__).resolve().parent / "results"
    results_dir  = results_root / Path(results_folder).name
    exp_dir      = results_dir / "experiment_files"
    csv_path     = results_dir / "results.csv"

    if not exp_dir.exists():
        print(f"No experiment_files directory found in {results_dir}")
        sys.exit(1)

    instance_dirs = sorted(
        (d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("instance_")),
        key=lambda d: int(d.name.split("_")[1]),
    )

    print(f"Found {len(instance_dirs)} instances in {results_dir}")

    for instance_dir in instance_dirs:
        instance_id   = int(instance_dir.name.split("_")[1])
        canonical_path = instance_dir / "canonical.py"
        test_path      = instance_dir / "generated_tests.py"
        summary_path   = instance_dir / "run_summary.json"

        if not canonical_path.exists() or not test_path.exists():
            print(f"  [{instance_id}] Skipping — no canonical/test artifacts saved.")
            continue

        print(f"  [{instance_id}] Running mutation scoring...")
        canonical_text = canonical_path.read_text(encoding="utf-8")
        test_text      = test_path.read_text(encoding="utf-8")

        mutation_results = compute_mutation_score(
            code_text=canonical_text,
            test_text=test_text,
            debug=debug,
        )

        score = mutation_results.get("mutation_score", -1.0)
        print(f"  [{instance_id}] Score: {score:.1f}%  warning: {mutation_results.get('mutation_warning', '')}")

        if summary_path.exists():
            _update_run_summary(summary_path, mutation_results)
        _patch_csv(csv_path, instance_id, mutation_results)

    print("\nDone. CSV and run_summary.json files updated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mutation scoring post-hoc on saved artifacts.")
    parser.add_argument("--results_folder", required=True, help="Results folder name (e.g. my_experiment)")
    parser.add_argument("--debug", action="store_true", help="Verbose mutmut output")
    args = parser.parse_args()
    run_posthoc(args.results_folder, debug=args.debug)


if __name__ == "__main__":
    main()