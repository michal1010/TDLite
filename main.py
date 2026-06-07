from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from utils.io_utils import parse_args
from workflows import ref_workflow, standard_workflow, tdd_ref_workflow, tdd_workflow


@dataclass(frozen=True)
class WorkflowEntry:
    name: str
    runner: Callable[[], object]


WORKFLOW_RUNNERS: dict[str, WorkflowEntry] = {
    "TDD": WorkflowEntry(name="TDD", runner=tdd_workflow.main),
    "TDD_REF": WorkflowEntry(name="TDD_REF", runner=tdd_ref_workflow.main),
    "REF": WorkflowEntry(name="REF", runner=ref_workflow.main),
    "default": WorkflowEntry(name="default", runner=standard_workflow.main),
}


def run_workflow(workflow_name: str):
    try:
        workflow_entry = WORKFLOW_RUNNERS[workflow_name]
    except KeyError as exc:
        supported = ", ".join(WORKFLOW_RUNNERS)
        raise ValueError(f"Unknown workflow {workflow_name!r}. Supported workflows: {supported}") from exc

    print(f"Running workflow: {workflow_entry.name}")
    return workflow_entry.runner()


def main():
    args = parse_args()
    return run_workflow(args.workflow)


if __name__ == "__main__":
    main()