import subprocess
import sys
import re
from utils.config import TEST_FILE
from utils.state import AgentState

def parse_passed_test_names(output: str) -> list[str]:
    passed = []
    for line in output.splitlines():
        match = re.match(r'^(\w+)\s+\(.*?\)\s+\.\.\.\s+ok\s*$', line.strip())
        if match:
            passed.append(match.group(1))
    return passed

def run_tests(state: AgentState) -> AgentState:
    """Run the generated unit tests and record the result."""
    print(f"\n Running tests with the current implementation:")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(TEST_FILE)],
        capture_output=True,
        text=True,
    )
    test_output = (result.stdout + result.stderr).strip()
    tests_passed = result.returncode == 0
    passed_now = parse_passed_test_names(test_output)
    ever_passed = list(set(state["ever_passed_tests"]) | set(passed_now))
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
        "test_outputs": state["test_outputs"] + [test_output],
        "debug_reports": state["debug_reports"],
        "ever_passed_tests": ever_passed,
        "next_agent": next_agent,
        "iteration": state["iteration"],
        "specifications": state["specifications"],
    }