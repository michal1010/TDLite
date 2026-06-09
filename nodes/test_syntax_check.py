import py_compile
import sys

from langchain_core.messages import HumanMessage

from utils.config import TEST_EXAMPLE, TEST_FILE
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState, update_token_state


def test_syntax_check(state: AgentState) -> AgentState:
    """Check for syntax errors in the generated tests."""

    print(f"Checking whether the generated tests contain critical errors:")

    tests_content = state.get("tests", "")
    if not tests_content and TEST_FILE.exists():
        tests_content = TEST_FILE.read_text(encoding="utf-8")

    if 'import unittest' not in tests_content and 'from unittest import' not in tests_content:
        tests_content = f"import unittest\n\n{tests_content}"
        TEST_FILE.write_text(tests_content, encoding="utf-8")

    try:
        py_compile.compile(str(TEST_FILE), doraise=True)
        syntax_ok = True
        test_output = "Syntax check passed."

    except py_compile.PyCompileError as exc:
        syntax_ok = False
        test_output = str(exc)
        print("Compilation error in generated tests:")
        print(test_output)
    print(test_output)
    next_agent = "a2"
    compile_iter = state["compile_iteration"] 
    test_iter = state["test_iteration"]
    # If there is a syntax error we can retry with the initial agent
    if not syntax_ok:
        test_iter += 1
        next_agent = "a12"
        print("Syntax error detected in generated tests, retrying one-shot with A12...")

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
        "iteration": state["iteration"],
        "test_iteration": test_iter,
        "specifications": state["specifications"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "compile_iteration": compile_iter,
        "debug_has_issues": state["debug_has_issues"],
        "debug_mode": state["debug_mode"],
    }