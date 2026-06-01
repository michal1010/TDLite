from langchain_core.messages import HumanMessage

from utils.config import PROGRAM_FILE
from utils.io_utils import save_text
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState


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
        "test_outputs": state["test_outputs"],
        "debug_reports": state["debug_reports"],
        "ever_passed_tests": state["ever_passed_tests"],
        "next_agent": "compile_check",
        "iteration": state["iteration"] + 1,
    }