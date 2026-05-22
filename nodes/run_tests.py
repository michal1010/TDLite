import subprocess
import sys

from config import TEST_FILE
from state import AgentState


def run_tests(state: AgentState) -> AgentState:
    """Run the generated unit tests and record the result."""
    print(f"\n Running tests with the current implementation:")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_FILE.stem],
        capture_output=True,
        text=True,
    )
    test_output = (result.stdout + result.stderr).strip()
    tests_passed = result.returncode == 0
    next_agent = "end" if tests_passed else "a4"

    print(f"\n{'=' * 60}")
    print(f"[Test Run] routing -> {next_agent}")
    print(test_output or "(no test output)")
    print("=" * 60)

    return {
        "messages": [],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": state["code"],
        "compile_output": state["compile_output"],
        "tests_passed": tests_passed,
        "test_output": test_output,
        "debug_report": state["debug_report"],
        "next_agent": next_agent,
        "iteration": state["iteration"],
    }