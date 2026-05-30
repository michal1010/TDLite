from langchain_core.messages import HumanMessage

from config import TEST_EXAMPLE, TEST_FILE
from io_utils import save_text
from llm import get_llm
from schemas import Response
from state import AgentState


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
        "test_outputs": state["test_outputs"],
        "debug_reports": state["debug_reports"],
        "ever_passed_tests": state["ever_passed_tests"],
        "next_agent": "a2",
        "iteration": state["iteration"],
    }