"""
HumanEval data loading with a hard structural separation between
what the pipeline sees and what the evaluation uses.

Pipeline:   get_problems()      → list of dicts, NO test field, NO entry_point
Evaluation: get_hidden_tests()  → separate dict keyed by problem_id

The two are never in the same structure, making it impossible to
accidentally pass hidden tests to any agent.
"""

from __future__ import annotations

import json
from pathlib import Path



def _load_all(local_path: str | None = None) -> list[dict]:
    """
    Load the full HumanEval dataset internally.
    Not exported — callers use get_problems() and get_hidden_tests() instead.
    """
    if local_path is not None:
        return _load_from_jsonl(Path(local_path))

    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/openai_humaneval", split="test")
    except ImportError:
        raise ImportError(
            "Install the datasets library:  pip install datasets\n"
            "Or use a local file:           set LOCAL_HUMANEVAL_PATH in your env"
        )

    problems = []
    for item in dataset:
        task_id    = item["task_id"]
        problem_id = int(task_id.split("/")[1])
        problems.append({
            "problem_id":  problem_id,
            "task_id":     task_id,
            "prompt":      item["prompt"],
            "_test":       item["test"],         # underscore = internal only
            "_entry_point": item["entry_point"], # underscore = internal only
            "_canonical_solution": item.get("canonical_solution", ""),
        })

    return sorted(problems, key=lambda x: x["problem_id"])


def _apply_filter(
    problems: list[dict],
    ids:         list[int] | None,
    range_start: int        | None,
    range_end:   int        | None,
) -> list[dict]:
    if ids is not None:
        id_set = set(ids)
        return [p for p in problems if p["problem_id"] in id_set]
    if range_start is not None and range_end is not None:
        return [p for p in problems if range_start <= p["problem_id"] <= range_end]
    return problems


def _load_from_jsonl(path: Path) -> list[dict]:
    problems = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item       = json.loads(line)
            task_id    = item["task_id"]
            problem_id = int(task_id.split("/")[1])
            problems.append({
                "problem_id":   problem_id,
                "task_id":      task_id,
                "prompt":       item["prompt"],
                "_test":        item["test"],
                "_entry_point": item["entry_point"],
                "_canonical_solution": item.get("canonical_solution", ""),
            })
    return sorted(problems, key=lambda x: x["problem_id"])


# ── Public API ────────────────────────────────────────────────────────────────

def get_problems(
    ids:          list[int] | None = None,
    range_start:  int        | None = None,
    range_end:    int        | None = None,
    local_path:   str        | None = None,
) -> list[dict]:
    """
    Return pipeline-safe problem descriptions — NO test code, NO entry_point.

    Each dict has only:
        problem_id  (int)
        task_id     (str)   "HumanEval/0"
        prompt      (str)   function signature + docstring

    This is the only data structure passed to the pipeline.
    """
    all_problems = _load_all(local_path)
    selected     = _apply_filter(all_problems, ids, range_start, range_end)
    return [
        {
            "problem_id": p["problem_id"],
            "task_id":    p["task_id"],
            "prompt":     p["prompt"],
        }
        for p in selected
    ]


def get_hidden_tests(
    ids:          list[int] | None = None,
    range_start:  int        | None = None,
    range_end:    int        | None = None,
    local_path:   str        | None = None,
) -> dict[int, dict]:
    """
    Return hidden evaluation data keyed by problem_id.
    Used ONLY in evaluate_solution() after the pipeline has finished.

    Each value has:
        test         (str)   the check(candidate) test function
        entry_point  (str)   name of the function being tested
    """
    all_problems = _load_all(local_path)
    selected     = _apply_filter(all_problems, ids, range_start, range_end)
    return {
        p["problem_id"]: {
            "test":        p["_test"],
            "entry_point": p["_entry_point"],
        }
        for p in selected
    }

def get_canonical_solutions(
    ids:         list[int] | None = None,
    range_start: int        | None = None,
    range_end:   int        | None = None,
    local_path:  str        | None = None,
) -> dict[int, str]:
    """
    Return prompt + canonical_solution keyed by problem_id.
    Used only for post-pipeline mutation scoring — never passed to any agent.
    """
    all_problems = _load_all(local_path)
    selected     = _apply_filter(all_problems, ids, range_start, range_end)
    return {
        p["problem_id"]: p["prompt"] + p["_canonical_solution"]
        for p in selected
    }

    