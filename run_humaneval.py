from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from experiments.test_metrics import generate_coverage_report
from humaneval_loader import get_problems, get_hidden_tests
import langgraph_agents
import langgraph_s1
import langgraph_s2
from utils.state import AgentState
from experiments.result_utils import (
    append_experiments_results,
    append_single_run_results
)


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


def _build_initial_state(description: str) -> AgentState:
    return  {
        "messages": [],
        "program_description": description,
        "tests": "",
        "code": "",
        "compile_output": "",
        "tests_passed": False,
        "test_outputs": [],
        "debug_reports": [],
        "ever_passed_tests": [],
        "next_agent": "a1",
        "iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "compile_iteration": 0,
        "debug_has_issues": True
    }


def run_pipeline(graph: Any, description: str) -> AgentState:
    """
    Run the full TDLite pipeline on a problem description string.
    Used by run_humaneval.py and main() below.
    Returns the final AgentState.
    """
    app = graph
    return app.invoke(_build_initial_state(description))


def run_single(
    problem: dict,
    hidden: dict,
    graph: Any,
) -> tuple[dict, AgentState | None, Path]:
    """
    Run the pipeline on one problem, then evaluate against hidden tests.

    problem — pipeline-safe dict: {problem_id, task_id, prompt}   (no test)
    hidden  — evaluation dict:    {test, entry_point}              (never touches pipeline)
    """
    pid      = problem["problem_id"]
    final_state: AgentState | None = None
    input_tokens: dict = {}
    output_tokens: dict = {}
    pipeline_error = False

    try:
        # ── Pipeline sees only the prompt ─────────────────────────────────────
        print(f"Running pipeline for problem {pid}...")
        final_state = run_pipeline(graph, problem["prompt"])

        code          = final_state.get("code", "")
        internal_pass = final_state.get("tests_passed", False)
        iterations    = final_state.get("iteration", 0)
        input_tokens  = final_state.get("input_tokens", {})
        output_tokens = final_state.get("output_tokens", {})

        # ── Hidden tests used only here, after pipeline is done ───────────────
        humaneval_pass, error_msg = evaluate_solution(
            code, hidden["test"], hidden["entry_point"]
        )

    except Exception:
        code          = ""
        internal_pass = False
        humaneval_pass = False
        iterations    = 0
        error_msg     = traceback.format_exc()
        pipeline_error = True

    result = {
        "problem_id":     pid,
        "task_id":        problem["task_id"],
        "entry_point":    hidden["entry_point"],
        "iterations":     iterations,
        "internal_pass":  internal_pass,
        "humaneval_pass": humaneval_pass,
        "pipeline_error": pipeline_error,
        "error":          error_msg,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
    }

    return result, final_state


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_on_problems(
    problems:     list[dict],
    hidden_tests: dict[int, dict],
    graph:        Any,
    results_folder: str | None,
) -> list[dict]:
    results = []

    for i, problem in enumerate(problems):
        pid = problem["problem_id"]
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(problems)}] Problem {pid}  —  {hidden_tests[pid]['entry_point']}")
        print(f"{'#'*60}")

        result, agent_state = run_single(problem, hidden_tests[pid], graph)

        status = "PASS ✓" if result["humaneval_pass"] else "FAIL ✗"
        print(f"\n  Result: {status}  |  iterations: {result['iterations']}"
              f"  |  internal_pass: {result['internal_pass']}")
        if result["error"]:
            error_lines = result["error"].splitlines()
            print(f"  Error:  {error_lines[-1] if error_lines else result['error']}")

        print(f"Input tokens: {result['input_tokens']}")
        print(f"Output tokens: {result['output_tokens']}")

        report = None
        report = generate_coverage_report(verbose=True)

        result["line_coverage_percent"] = report['line_coverage']['percent']
        result["branch_coverage_percent"] = report['branch_coverage']['percent']
        results.append(result)

        if results_folder:
            append_single_run_results(
                final_state=agent_state,
                coverage_report=report,
                results_folder=results_folder,
                hidden_test=hidden_tests[pid]["test"],
            )

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


def choose_scenario(scenario: str):
    if scenario == "s1":
        return langgraph_s1.build_graph()
    elif scenario == "s2":
        return langgraph_s2.build_graph()
    elif scenario == "s3d":
        return ValueError("Scenario s3 is not implemented yet")
    elif scenario == 's4':
        return langgraph_agents.build_graph()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


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

    parser.add_argument(
        "--scenario",
        choices=["s1", "s2", "s3", "s4"],
        required=True,
        help="Scenario to run: s1, s2, s3, or s4"
    )

    parser.add_argument(
        "--store_results",
        nargs="?",
        const="results",
        default=None,
        help="defines folder to store experiment results in, implicitly enables results storage",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ids         = args.ids if args.ids else None
    range_start = args.range[0] if args.range else None
    range_end   = args.range[1] if args.range else None

    print("Loading HumanEval dataset...")
    problems     = get_problems(ids=ids, range_start=range_start, range_end=range_end)
    hidden_tests = get_hidden_tests(ids=ids, range_start=range_start, range_end=range_end)

    if not problems:
        print("No matching problems found. HumanEval has IDs 0–163.")
        sys.exit(1)

    graph = choose_scenario(args.scenario)

    results = run_on_problems(problems, hidden_tests, graph, args.store_results)

    if args.store_results:
        append_experiments_results(results, args.store_results, timestamp)
        print_summary(results, args.store_results)


if __name__ == "__main__":
    main()
