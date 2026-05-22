from langchain_core.messages import HumanMessage

from llm import get_llm
from schemas import Response
from state import AgentState


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