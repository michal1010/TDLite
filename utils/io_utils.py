import argparse
from pathlib import Path
from coverage import parser
from coverage import parser
from datasets import load_dataset

from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE

SUPPORTED_DATASETS = ("HumanEval", "MBPP")
SUPPORTED_SCENARIOS = ("TDD", "TDD_REF", "REF", "default")


def _normalize_optional_arg(value: str | None, default: str) -> str:
    return default if value in (None, "") else value


def read_program_description(path: Path) -> str:
    """Read a program description file."""
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse the optional program description file name."""
    parser = argparse.ArgumentParser(description="Run the LangGraph TDD pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--file",
        nargs="?",
        default=None,
        help="program description file to use",
    )
    
    group.add_argument(
        "-d", "--dataset",
        nargs="?",
        default="MBPP",
        help="dataset to use (HumanEval or MBPP)",
    )

    parser.add_argument(
        "-s","--scenario",
        nargs="?",
        default="default",
        help="scenario to run (TDD, TDD_REF, REF, or default)",
    )
    
    parser.add_argument(
        "--sample_run",
        action="store_true",
        default=False,
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

    # Implement the ability to specify if we are in debug mode or not
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to run in debug mode with more verbose output and no retries",
    )

 

    args = parser.parse_args()
    args.dataset = _normalize_optional_arg(args.dataset, "MBPP")
    args.scenario = _normalize_optional_arg(args.scenario, "default")

    if args.dataset not in SUPPORTED_DATASETS:
        parser.error(
            f"argument --dataset: invalid choice: {args.dataset!r} (choose from {', '.join(SUPPORTED_DATASETS)})"
        )
    if args.scenario not in SUPPORTED_SCENARIOS:
        parser.error(
            f"argument --scenario: invalid choice: {args.scenario!r} (choose from {', '.join(SUPPORTED_SCENARIOS)})"
        )

    return args


def save_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")

def get_humaneval_prompts() -> list[str]:
    ds = load_dataset("openai/openai_humaneval")
    samples = ds["test"][:10]
    print(samples)
    return samples["prompt"]


def get_mbpp_prompts() -> list[str]:
    ds = load_dataset("google-research-datasets/mbpp", "full")
    samples = ds["test"][:10]
    return samples["text"]

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")