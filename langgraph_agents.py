from __future__ import annotations

import argparse
from pathlib import Path
import py_compile
import subprocess
import sys
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI  # works with any OpenAI-compatible endpoint
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROGRAM_DESCRIPTION_FILE = Path("program_description_easy.txt")

PROGRAM_FILE = Path("generated_program.py")
TEST_FILE = Path("test_generated_program.py")
TEST_EXAMPLE = """import unittest
from generated_program import ClassToTest

class Tests(unittest.TestCase):
    # Rename tests, classes and methods based on the program description
    def test_example_1(self):
        class_to_test = ClassToTest()
        self.assertEqual(class_to_test.some_method(), "expected result")

    def test_example_2(self):
        class_to_test = ClassToTest()
        self.assertEqual(class_to_test.some_other_method(), 42)
        
if __name__ == '__main__':
    unittest.main()
"""
MAX_ITERATIONS = 8


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    program_description: str
    tests: str
    code: str
    compile_output: str
    tests_passed: bool
    test_output: str
    debug_report: str
    next_agent: str
    iteration: int

class Response(BaseModel):
    content: str


def get_llm() -> ChatOpenAI:
    '''Changed model to qwen2.5-coder:3b, takes a bit longer usually, but no more 5 min+ thinking.
    Change model back to model="qwen3:0.6b" if your laptop cannot handle.
    '''
    return ChatOpenAI(
        model="qwen2.5-coder:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
    )


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


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def agent_a1_test_generation(state: AgentState) -> AgentState:
    """A1: write tests from the program description before any implementation exists."""
    print(f"\n A1 received program description, preparing to write tests:")
    llm = get_llm()
    program_description = state["program_description"]

    save_text(TEST_FILE, TEST_EXAMPLE)

    prompt = f"""
        You are Agent A1, the test-writing agent in a TDD pipeline.
        Write unit tests in Python based on the program description and example template.
        Do NOT just return the example template.
        Ensure test assertions are correct.
        The import modules (unittest and generated_program) are already correct.

        Program description:
        {program_description}

        Example template:
        {TEST_EXAMPLE}
        """.strip()

    llm = llm.with_structured_output(Response)
    response = llm.invoke([HumanMessage(content=prompt)])
    tests = response.content
    save_text(TEST_FILE, tests)

    print(f"\n{'=' * 60}")
    print("[A1 - Test Generation] wrote tests")
    print(f"Saved to: {TEST_FILE}")
    print("=" * 60)

    return {
        "messages": [tests],
        "program_description": program_description,
        "tests": tests,
        "code": state["code"],
        "compile_output": state["compile_output"],
        "tests_passed": state["tests_passed"],
        "test_output": state["test_output"],
        "debug_report": state["debug_report"],
        "next_agent": "a2",
        "iteration": state["iteration"],
    }


def test_syntax_check(state: AgentState) -> AgentState:
    """Check for syntax errors in the generated tests."""

    print(f"Checking whether the generated tests contain critical errors:")

    result = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_FILE.stem],
        capture_output=True,
        text=True,
    )
    test_output = (result.stdout + result.stderr).strip()

    next_agent = "a1"

    # Some easy errors, no LLM call needed
    easy_errors = ("NameError: name 'unittest' is not defined" in test_output or # Not importing unittest
                    "Ran 0 tests" in test_output or # Empty test suite, or tests not in unittest class
                    TEST_EXAMPLE in test_output or # No changes from template test suite
                    (test_output.count("AssertionError") >= test_output.count("ERROR") and # AssertionErrors are allowed
                        test_output.count("ModuleNotFoundError") >= test_output.count("ERROR")) or # generated_program does not exist yet, this error is allowed
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
        "test_output": state["test_output"],
        "debug_report": state["debug_report"],
        "next_agent": next_agent,
        "iteration": iterations,
    }

def agent_a2_patch_creation(state: AgentState) -> AgentState:
    """A2: write or revise the program so that it satisfies the generated tests."""
    print(f"\n A2 received state with test description, preparing to write implementation:")
    llm = get_llm()

    debug_context = ""
    if state["debug_report"]:
        debug_context = f"""
            The last test run failed. A debug agent reported:
            {state["debug_report"]}

            Current code:
            {state["code"]}
            """.strip()

    prompt = f"""
        You are Agent A2, the patch-writing agent in a TDD pipeline.

        Write the implementation for generated_program.py.

        Rules:
        - Return only Python code.
        - Do not include markdown fences or explanations.
        - Do not include ```python ```fences, just return the code.
        - Make the code satisfy the tests.
        - Keep the implementation simple.

        Program description:
        {state["program_description"]}

        Tests that must pass:
        {state["tests"]}

        {debug_context}
        """.strip()

    llm = llm.with_structured_output(Response)
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    save_text(PROGRAM_FILE, code)

    print(f"\n{'=' * 60}")
    print(f"[A2 - Patch Creation] wrote implementation attempt {state['iteration'] + 1}")
    print(f"Saved to: {PROGRAM_FILE}")
    print("=" * 60)

    return {
        "messages": [code],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": code,
        "compile_output": "",
        "tests_passed": False,
        "test_output": "",
        "debug_report": "",
        "next_agent": "compile_check",
        "iteration": state["iteration"] + 1,
    }


def compile_check(state: AgentState) -> AgentState:
    """Compile the generated program and decide whether it can move to tests."""
    try:
        py_compile.compile(str(PROGRAM_FILE), doraise=True)
        compile_output = "Compilation succeeded."
        next_agent = "run_tests"
    except py_compile.PyCompileError as exc:
        compile_output = str(exc)
        if "```python" in compile_output:
            compile_output = "Code failed, because you included ```python fences. DO NOT include markdown fences, return only code."
        next_agent = "a3"

    print(f"\n{'=' * 60}")
    print(f"[Compile Check] routing -> {next_agent}")
    print(compile_output)
    print("=" * 60)

    return {
        "messages": [],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": state["code"],
        "compile_output": compile_output,
        "tests_passed": state["tests_passed"],
        "test_output": state["test_output"],
        "debug_report": state["debug_report"],
        "next_agent": next_agent,
        "iteration": state["iteration"],
    }


def agent_a3_patch_revision(state: AgentState) -> AgentState:
    """A3: fix syntax or compilation errors before tests are run."""
    print(f"\n A3 received compile error, preparing to fix it:")
    llm = get_llm()

    prompt = f"""
        You are Agent A3, the compilation-fix agent.

        The current code in generated_program.py does not compile.
        Fix only what is needed so that it becomes valid Python.

        Rules:
        - Return only Python code.
        - Do not include markdown fences or explanations.
        - Do not include ```python ```fences, just return the code.
        - Keep the intended behavior from the program description.

        Program description:
        {state["program_description"]}

        Current code:
        {state["code"]}

        Compilation error:
        {state["compile_output"]}
        """.strip()

    llm = llm.with_structured_output(Response)
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    save_text(PROGRAM_FILE, code)

    print(f"\n{'=' * 60}")
    print("[A3 - Patch Revision] revised code after compilation failure")
    print("=" * 60)

    return {
        "messages": [code],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": code,
        "compile_output": "",
        "tests_passed": False,
        "test_output": "",
        "debug_report": state["debug_report"],
        "next_agent": "compile_check",
        "iteration": state["iteration"] + 1,
    }


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


def agent_a4_debug(state: AgentState) -> AgentState:
    """A4: explain why tests failed and send that explanation back to A2."""
    print(f"\n A4 received failed test output, preparing to analyze it:")
    llm = get_llm()

    prompt = f"""
        You are Agent A4, the debugging agent in a TDD pipeline.

        The tests failed. Investigate the failure and explain what is probably wrong
        with the implementation. Do not write code. Assume the test code is fully correct. Give a short, concrete diagnosis
        that the patch-writing agent can act on.

        Tests:
        {state["tests"]}

        Current code:
        {state["code"]}

        Test output:
        {state["test_output"]}
        """.strip()

    llm = llm.with_structured_output(Response)
    response = llm.invoke([HumanMessage(content=prompt)])
    debug_report = response.content

    print(f"\n{'=' * 60}")
    print("[A4 - Debug] sending diagnosis back to A2")
    print(debug_report)
    print("=" * 60)

    return {
        "messages": [debug_report],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": state["code"],
        "compile_output": state["compile_output"],
        "tests_passed": False,
        "test_output": state["test_output"],
        "debug_report": debug_report,
        "next_agent": "a2",
        "iteration": state["iteration"],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_compile(state: AgentState) -> Literal["run_tests", "a3", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS and state["next_agent"] != "run_tests":
        return "__end__"
    if state["next_agent"] == "run_tests":
        return "run_tests"
    return "a3"


def route_after_test_compile(state: AgentState) -> Literal["a2", "a1", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS and state["next_agent"] != "a2":
        return "__end__"
    if state["next_agent"] == "a2":
        return "a2"
    return "a1"


def route_after_tests(state: AgentState) -> Literal["a4", "__end__"]:
    if state["tests_passed"]:
        return "__end__"
    if state["iteration"] >= MAX_ITERATIONS:
        return "__end__"
    return "a4"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a1", agent_a1_test_generation)
    graph.add_node("test_syntax_check", test_syntax_check)
    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("compile_check", compile_check)
    graph.add_node("a3", agent_a3_patch_revision)
    graph.add_node("run_tests", run_tests)
    graph.add_node("a4", agent_a4_debug)

    graph.set_entry_point("a1")
    # graph.add_edge("a1", "a2")
    graph.add_edge("a1", "test_syntax_check")
    graph.add_conditional_edges(
        "test_syntax_check",
        route_after_test_compile,
        {
            "a2": "a2",
            "a1": "a1",
            "__end__": END,
        },
    )
    graph.add_edge("a2", "compile_check")
    graph.add_conditional_edges(
        "compile_check",
        route_after_compile,
        {
            "run_tests": "run_tests",
            "a3": "a3",
            "__end__": END,
        },
    )
    graph.add_edge("a3", "compile_check")
    graph.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {
            "a4": "a4",
            "__end__": END,
        },
    )
    graph.add_edge("a4", "a2")

    return graph.compile()


def main() -> AgentState:
    args = parse_args()
    description_file = Path(args.description_file)
    app = build_graph()
    program_description = read_program_description(description_file)

    initial_state: AgentState = {
        "messages": [],
        "program_description": program_description,
        "tests": "",
        "code": "",
        "compile_output": "",
        "tests_passed": False,
        "test_output": "",
        "debug_report": "",
        "next_agent": "a1",
        "iteration": 0,
    }

    print("\n" + "=" * 60)
    print("  LangGraph TDD Pipeline - Starting")
    print(f"  Description file: {description_file}")
    print("  A1 tests -> test compile -> A2 code -> compile -> tests -> debug loop")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["tests_passed"]:
        print("  Pipeline finished: all tests passed.")
    else:
        print("  Pipeline stopped before all tests passed.")
    print(f"  Iterations used: {final_state['iteration']}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    main()
