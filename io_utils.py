import argparse
from pathlib import Path

from config import DEFAULT_PROGRAM_DESCRIPTION_FILE


def read_program_description(path: Path) -> str:
    """Read a program description file."""
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse the optional program description file name."""
    parser = argparse.ArgumentParser(description="Run the LangGraph TDD pipeline.")
    parser.add_argument(
        "--description-file",
        nargs="?",
        default=str(DEFAULT_PROGRAM_DESCRIPTION_FILE),
        help="program description file to use",
    )
    return parser.parse_args()


def save_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")