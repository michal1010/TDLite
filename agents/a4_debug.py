import re


from langchain_core.messages import HumanMessage

from utils.config import TEMP_DEBUG
from utils.llm import get_llm
from utils.schemas import DebugReport
from utils.state import AgentState
from utils.debug_helpers import trim_test_output

def parse_failing_test_names(output: str) -> list[str]:
    failing = []
    for line in output.splitlines():
        match = re.match(r'^(\w+)\s+\(.*?\)\s+\.\.\.\s+(?:FAIL|ERROR)\s*$', line.strip())
        if match:
            failing.append(match.group(1))
    return failing

def with_line_numbers(text: str) -> str:
    return "\n".join(
        f"{i:>4}: {line}"
        for i, line in enumerate(text.splitlines(), start=1)
    )

def agent_a4_debug(state: AgentState) -> AgentState:
    """A4: explain why tests failed and send that explanation back to A2."""
    print(f"\n{'='*60}")
    print(f"[A4 - Debug] analysing failure at iteration {state['iteration']}")
    print("="*60)

    current_output = trim_test_output(state["test_outputs"][-1])

    failing_now = parse_failing_test_names(current_output)
    regressions = [t for t in state["ever_passed_tests"] if t in failing_now]

    regression_section = ""
    if regressions:
        regression_section = f"""\

The following tests passed before but are now failing — these are REGRESSIONS:
{', '.join(regressions)}

"""



    history_section = ""
    if state["debug_reports"]:
        previous = "\n\n".join(
            f"Diagnosis {i + 1}:\n{r}"
            for i, r in enumerate(state["debug_reports"])
        )
        history_section = f"""\
Previous diagnoses — provide a NEW diagnosis, do not repeat these:
{previous}


"""

    prompt = f"""\
You are a debugging agent in a TDD pipeline.
Analyse why the tests are failing. Do not write code. The test code is correct.

Be specific:
- Reference exact line numbers from the code and test output.
- State the actual value versus the expected value in failed assertions.
- Identify which method or condition is wrong, not just that something is wrong.

{regression_section}{history_section}\
Program description:
{state["program_description"]}

Tests:
{with_line_numbers(state["tests"])}

Current code:
{with_line_numbers(state["code"])}

Test output:
{current_output}"""


    llm = get_llm(temperature=TEMP_DEBUG).with_structured_output(DebugReport)
    response = llm.invoke([HumanMessage(content=prompt)])

    debug_report = (
        f"Error Analysis: {response.error_analysis}\n"
        f"Root Cause: {response.root_cause}\n"
        f"Suggested Fix: {response.suggested_fix}"
    )

    print("[A4 - Debug] diagnosis:")
    print(debug_report)
    print("="*60)


    return {
        "messages":            [],
        "program_description": state["program_description"],
        "tests":               state["tests"],
        "code":                state["code"],
        "compile_output":      state["compile_output"],
        "tests_passed":        False,
        "test_outputs":        state["test_outputs"],
        "debug_reports":       state["debug_reports"] + [debug_report],
        "ever_passed_tests":   state["ever_passed_tests"],
        "next_agent":          "a2",
        "iteration":           state["iteration"],
    }