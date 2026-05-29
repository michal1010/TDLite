import subprocess
import sys

from langchain_core.messages import HumanMessage

from config import TEST_EXAMPLE, TEST_FILE
from llm import get_llm
from schemas import Response
from state import AgentState


def test_syntax_check(state: AgentState) -> AgentState:
    """Check for syntax errors in the generated tests."""

    print(f"Checking whether the generated tests contain critical errors:")

    result = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_FILE.stem],
        capture_output=True,
        text=True,
    )
    test_output = (result.stdout + result.stderr).strip()
    print(test_output)
    next_agent = "a1"

    # Some easy errors, no LLM call needed
    easy_errors = ("NameError: name 'unittest' is not defined" in test_output or # Not importing unittest
                    "Ran 0 tests" in test_output or # Empty test suite, or tests not in unittest class
                    TEST_EXAMPLE in test_output or # No changes from template test suite
                   "if __name__ == '__main__':" not in state["tests"]) # No way to run the tests included

    if not easy_errors:
        print("LLM manually checking tests for syntax...")
        llm = get_llm()
        # Use LLM to deduce whether there are actually tests and whether they contain syntax errors.
        prompt = f"""
            You are a Python test checker.
            Check the following test code and determine if it contains tests.
            Also check if there are critical errors.
            Reply 'False' if any of the following is true:
            - There are no test cases defined.
            - THere are critical syntax errors.
            Otherwise, reply 'True'.
            Test code:
            {state["tests"]}
            """.strip()
        
        llm = llm.with_structured_output(Response)
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"\nLLM response for test check: {response.content}")
        if response.content.strip().lower() == "true":
            next_agent = "a2"
            iterations = state["iteration"]
        else:
            iterations = state["iteration"] + 1
    else:
        iterations = state["iteration"] + 1
        print("Easy error detected, retrying one-shot with A1...")

    print(f"\n{'=' * 60}")
    print(f"[Test Compile Check] routing -> {next_agent}")
    print("=" * 60)

    return {
        "messages": [],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": state["code"],
        "compile_output": state["compile_output"],
        "tests_passed": state["tests_passed"],
        "test_outputs": state["test_outputs"],
        "debug_reports": state["debug_reports"],
        "ever_passed_tests": state["ever_passed_tests"],
        "next_agent": next_agent,
        "iteration": iterations,
    }