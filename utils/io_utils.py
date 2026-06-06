import argparse
from pathlib import Path

from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE

SUPPORTED_DATASETS = ("HumanEval", "MBPP")


def read_program_description(path: Path) -> str:
    """Read a program description file."""
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse the optional program description file name."""
    parser = argparse.ArgumentParser(description="Run the LangGraph TDD pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file",
        nargs="?",
        default=None,
        help="program description file to use",
    )
    group.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="MBPP",
        help="dataset to use (HumanEval or MBPP)",
    )

    # Implement the ability to specify if we are in debug mode or not
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to run in debug mode with more verbose output and no retries",
    )

    parser.add_argument(
        "--sample_run",
        action="store_true",
        help="whether to run random experiment or a problem from the given description",
    )

    parser.add_argument(
        "--store_results",
        nargs="?",
        const="results",
        default=None,
        help="defines folder to store experiment results in, implicitly enables results storage",
    )

    group = parser.add_mutually_exclusive_group(required=False)

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

    return parser.parse_args()


def save_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")