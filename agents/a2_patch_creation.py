from langchain_core.messages import HumanMessage

from config import PROGRAM_FILE
from io_utils import save_text
from llm import get_llm
from schemas import Response
from state import AgentState


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