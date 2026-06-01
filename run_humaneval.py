"""
Run TDLite on HumanEval problems.

Handpicked:   python run_humaneval.py --ids 0 5 12 47
Range:        python run_humaneval.py --range 0 9
All 164:      python run_humaneval.py --range 0 163
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from humaneval_loader import get_problems, get_hidden_tests
from langgraph_agents import run_pipeline


RESULTS_DIR = Path("results")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_solution(code: str, test_code: str, entry_point: str) -> tuple[bool, str]:
    """
    Run generated code against HumanEval's hidden test suite.
    Called ONLY after run_pipeline() has finished — never before.

    Returns (passed: bool, error_message: str).
    """
    namespace: dict = {}
    try:
        exec(code, namespace)
        exec(test_code, namespace)
        namespace["check"](namespace[entry_point])
        return True, ""
    except AssertionError as exc:
        return False, f"AssertionError: {exc}"
    except KeyError:
        return False, f"Function '{entry_point}' not found in generated code"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ── Per-problem runner ────────────────────────────────────────────────────────

def run_single(problem: dict, hidden: dict, run_dir: Path) -> dict:
    """
    Run the pipeline on one problem, then evaluate against hidden tests.

    problem — pipeline-safe dict: {problem_id, task_id, prompt}   (no test)
    hidden  — evaluation dict:    {test, entry_point}              (never touches pipeline)
    """
    pid      = problem["problem_id"]
    work_dir = run_dir / f"problem_{pid}"
    work_dir.mkdir(parents=True, exist_ok=True)
    origin   = Path.cwd()

    try:
        # ── Pipeline sees only the prompt ─────────────────────────────────────
        os.chdir(work_dir)
        final_state = run_pipeline(problem["prompt"])
        os.chdir(origin)

        code          = final_state.get("code", "")
        internal_pass = final_state.get("tests_passed", False)
        iterations    = final_state.get("iteration", 0)

        # ── Hidden tests used only here, after pipeline is done ───────────────
        humaneval_pass, error_msg = evaluate_solution(
            code, hidden["test"], hidden["entry_point"]
        )

    except Exception:
        os.chdir(origin)
        code          = ""
        internal_pass = False
        humaneval_pass = False
        iterations    = 0
        error_msg     = traceback.format_exc()

    result = {
        "problem_id":     pid,
        "task_id":        problem["task_id"],
        "entry_point":    hidden["entry_point"],
        "iterations":     iterations,
        "internal_pass":  internal_pass,
        "humaneval_pass": humaneval_pass,
        "error":          error_msg,
    }

    (work_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_on_problems(
    problems:     list[dict],
    hidden_tests: dict[int, dict],
    run_dir:      Path,
) -> list[dict]:
    results = []

    for i, problem in enumerate(problems):
        pid = problem["problem_id"]
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(problems)}] Problem {pid}  —  {hidden_tests[pid]['entry_point']}")
        print(f"{'#'*60}")

        result = run_single(problem, hidden_tests[pid], run_dir)
        results.append(result)

        status = "PASS ✓" if result["humaneval_pass"] else "FAIL ✗"
        print(f"\n  Result: {status}  |  iterations: {result['iterations']}"
              f"  |  internal_pass: {result['internal_pass']}")
        if result["error"]:
            print(f"  Error:  {result['error'].splitlines()[0]}")

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], run_dir: Path) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["humaneval_pass"])
    avg_it = sum(r["iterations"] for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY  —  {run_dir}")
    print(f"{'='*60}")
    print(f"  Problems:        {total}")
    print(f"  HumanEval pass:  {passed}/{total}  ({100*passed/total:.1f}%)")
    print(f"  Avg iterations:  {avg_it:.1f}")
    print()
    print(f"  {'ID':<6} {'Entry point':<32} {'HumanEval':<11} {'Internal':<10} {'Iters'}")
    print(f"  {'-'*65}")
    for r in results:
        he = "PASS" if r["humaneval_pass"] else "FAIL"
        ip = "pass" if r["internal_pass"]  else "fail"
        print(f"  {r['problem_id']:<6} {r['entry_point']:<32} {he:<11} {ip:<10} {r['iterations']}")
    print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TDLite on HumanEval problems.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_humaneval.py --ids 0 5 12 47\n"
            "  python run_humaneval.py --range 0 9\n"
            "  python run_humaneval.py --range 0 163\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ids", nargs="+", type=int, metavar="ID",
        help="Specific problem IDs  e.g. --ids 0 5 12 47",
    )
    group.add_argument(
        "--range", nargs=2, type=int, metavar=("START", "END"),
        help="Inclusive ID range    e.g. --range 0 9",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parse selection args once, pass same filter to both loaders
    ids         = args.ids if args.ids else None
    range_start = args.range[0] if args.range else None
    range_end   = args.range[1] if args.range else None

    print("Loading HumanEval dataset...")
    problems     = get_problems(ids=ids, range_start=range_start, range_end=range_end)
    hidden_tests = get_hidden_tests(ids=ids, range_start=range_start, range_end=range_end)

    if not problems:
        print("No matching problems found. HumanEval has IDs 0–163.")
        sys.exit(1)

    print(f"Running {len(problems)} problem(s) → results in {run_dir}\n")
    results = run_on_problems(problems, hidden_tests, run_dir)

    summary = {
        "timestamp": timestamp,
        "total":     len(results),
        "passed":    sum(1 for r in results if r["humaneval_pass"]),
        "pass_rate": sum(1 for r in results if r["humaneval_pass"]) / len(results),
        "avg_iters": sum(r["iterations"] for r in results) / len(results),
        "results":   results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print_summary(results, run_dir)


if __name__ == "__main__":
    main()