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
        "--store_results",
        nargs="?",
        const="results",
        default=None,
        help="defines folder to store experiment results in, implicitly enables results storage",
    )

    return parser.parse_args()


def save_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")