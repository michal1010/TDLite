from langchain_core.messages import HumanMessage

from llm import get_llm
import re
from schemas import DebugReport
from state import AgentState

def parse_failing_test_names(output: str) -> list[str]:
    failing = []
    for line in output.splitlines():
        match = re.match(r'^(\w+)\s+\(.*?\)\s+\.\.\.\s+(?:FAIL|ERROR)\s*$', line.strip())
        if match:
            failing.append(match.group(1))
    return failing

def agent_a4_debug(state: AgentState) -> AgentState:
    """A4: explain why tests failed and send that explanation back to A2."""
    print(f"\n A4 received failed test output, preparing to analyze it:")
    llm = get_llm()

    current_output = state["test_outputs"][-1]

    failing_now = parse_failing_test_names(current_output)
    regressions = [t for t in state["ever_passed_tests"] if t in failing_now]

    regression_section = ""
    if regressions:
        regression_section = f"""
The following tests passed before but are now failing — these are REGRESSIONS the patch must not break:
{', '.join(regressions)}
""".strip()

    history_section = ""
    if state["debug_reports"]:
        previous = "\n\n".join(
            f"Diagnosis {i + 1}:\n{r}"
            for i, r in enumerate(state["debug_reports"])
        )
        history_section = f"""
Previous diagnoses — do not repeat the same diagnosis:
{previous}
""".strip()

    prompt = f"""
You are Agent A4, the debugging agent in a TDD pipeline.

Analyse why the current tests fail. Do not write code. Assume the test code is fully correct.
Give a short, concrete diagnosis that the patch-writing agent can act on.

{regression_section}

{history_section}

Program description:
{state["program_description"]}

Tests:
{state["tests"]}

Current code:
{state["code"]}

Current test output:
{current_output}
""".strip()

    llm = llm.with_structured_output(DebugReport)
    response = llm.invoke([HumanMessage(content=prompt)])

    debug_report = (
        f"Error Analysis: {response.error_analysis}\n"
        f"Root Cause: {response.root_cause}\n"
        f"Suggested Fix: {response.suggested_fix}"
    )

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
        "test_outputs": state["test_outputs"],
        "debug_reports": state["debug_reports"] + [debug_report],
        "ever_passed_tests": state["ever_passed_tests"],
        "next_agent": "a2",
        "iteration": state["iteration"],
    }