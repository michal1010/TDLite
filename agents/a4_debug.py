import re


from langchain_core.messages import HumanMessage, SystemMessage
from utils.io_utils import print_section
from utils.config import TEMP_DEBUG
from utils.llm import get_llm
from utils.schemas import DebugReport
from utils.state import AgentState, update_token_state
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

# def agent_a4_debug(state: AgentState) -> AgentState:
#     """A4: Verify if explain why tests failed and send that explanation back to A2."""
   
#     print_section(f"[A4 - Debug] analysing failure at iteration {state['iteration']}")

#     has_tests = bool(state["tests"])

#     current_output = ""
#     regression_section = ""

#     if has_tests:
#         current_output = trim_test_output(state["test_outputs"][-1])
#         failing_now = parse_failing_test_names(current_output)
#         regressions = [t for t in state["ever_passed_tests"] if t in failing_now]
#         if regressions:
#             regression_section = f"""\

#                 The following tests passed before but are now failing — these are REGRESSIONS:
#                 {', '.join(regressions)}

#                 """



#     history_section = ""
#     if state["debug_reports"]:
#         previous = "\n\n".join(
#             f"Diagnosis {i + 1}:\n{r}"
#             for i, r in enumerate(state["debug_reports"])
#         )
#         history_section = f"""\
#             Previous diagnoses — provide a NEW diagnosis, do not repeat these:
#             Before raising the same issue again, verify it still genuinely exists in the CURRENT code shown below.
#             Do not repeat a diagnosis that has already been addressed:
#             {previous}


#             """

#     if has_tests:
#         prompt = f"""\
# You are a debugging agent in a TDD pipeline.
# Analyse why the tests are failing. Do not write code. The test code is correct.

# Be specific:
# - Reference exact line numbers from the code and test output.
# - State the actual value versus the expected value in failed assertions.
# - Identify which method or condition is wrong, not just that something is wrong.

# {regression_section}{history_section}\
# Program description:
# {state["program_description"]}

# Tests:
# {with_line_numbers(state["tests"])}

# Current code:
# {with_line_numbers(state["code"])}

# Test output:
# {current_output}"""
#     else:
#         prompt = f"""\
# You are a code review agent in a pipeline.
# Your job is to find bugs, not to confirm correctness. Assume the code contains at least one issue until you can explicitly prove otherwise.

# Do not write code. Be specific:
# - Trace through the requirements in the description one by one and verify the code handles each correctly.
# - Reference exact line numbers when identifying issues.
# - Check explicitly for: off-by-one errors, unhandled edge cases, incorrect return types, missing conditions, and logic that diverges from the description.
# - State what the code actually does versus what the problem requires.
# - Set has_issues to False ONLY if you have traced through every requirement and every edge case and found no discrepancy.

# {history_section}\
# Problem description:
# {state["program_description"]}

# Current code:
# {with_line_numbers(state["code"])}"""


#     llm = get_llm(temperature=TEMP_DEBUG).with_structured_output(schema=DebugReport, include_raw=True)
#     response = llm.invoke([HumanMessage(content=prompt)])
#     update_token_state(state, response["raw"], "a4")

#     debug_report = (
#         f"Error Analysis: {response['parsed'].error_analysis}\n"
#         f"Root Cause: {response['parsed'].root_cause}\n"
#         f"Suggested Fix: {response['parsed'].suggested_fix}"
#     )

#     print_section(f"[A4 - Debug] diagnosis\n{debug_report}")

#     return {
#         "messages":            [],        
#         "program_description": state["program_description"],
#         "tests":               state["tests"],
#         "specifications":       state["specifications"],
#         "code":                state["code"],
#         "compile_output":      state["compile_output"],
#         "tests_passed":        False,
#         "test_outputs":        state["test_outputs"],
#         "debug_reports":       state["debug_reports"] + [debug_report],
#         "ever_passed_tests":   state["ever_passed_tests"],
#         "next_agent":          "a2",
#         "iteration":           state["iteration"],
#         "input_tokens":        state["input_tokens"],
#         "output_tokens":       state["output_tokens"],
#         "compile_iteration": state["compile_iteration"],
#         "debug_has_issues":    response["parsed"].has_issues,
#     }   


def _build_system_prompt(has_tests: bool) -> str:
    if has_tests:
        return (
            "You are a debugging agent in a TDD pipeline.\n"
            "Task: identify the root cause of the failing tests. Output only a diagnosis, no code.\n"
            "- Reference exact line numbers from the code and test output.\n"
            "- State the actual value versus the expected value in failed assertions.\n"
            "- Identify which method or condition is wrong, not just that something is wrong."
        )
    return (
        "You are a code review agent in a pipeline.\n"
        "Task: find bugs in the code. Output only a diagnosis, no code. IT IS POSSIBLE THAT THERE ARE NO BUGS.\n"
        "- Trace through the requirements one by one and verify the code handles each correctly.\n"
        "- Reference exact line numbers when identifying issues.\n"
        "- If no bug has been found, say explicitly that you found no issues and the code looks correct.\n"
        "- Check for: off-by-one errors, unhandled edge cases, incorrect return types, missing conditions, logic diverging from the description.\n"
        "- State what the code actually does versus what the problem requires."
    )


def _build_user_prompt(state: AgentState, has_tests: bool) -> str:
    regression_section = ""
    history_section = ""

    if has_tests:
        current_output = trim_test_output(state["test_outputs"][-1])
        failing_now = parse_failing_test_names(current_output)
        regressions = [t for t in state["ever_passed_tests"] if t in failing_now]
        if regressions:
            regression_section = (
                "REGRESSIONS (passed before, now failing):\n"
                f"{', '.join(regressions)}\n\n"
            )

    if state["debug_reports"]:
        recent = state["debug_reports"][-2:]
        previous = "\n\n".join(
            f"Diagnosis {i + 1}:\n{r}"
            for i, r in enumerate(recent)
        )
        history_section = (
            "Recent diagnoses — avoid repeating these:\n"
            f"{previous}\n\n"
        )

    json_example = (
        'Respond in this exact JSON format:\n'
        '{"error_analysis": "...", "root_cause": "...", "suggested_fix": "...", "has_issues": true}\n\n'
    )

    if has_tests:
        current_output = trim_test_output(state["test_outputs"][-1])
        relevant_code = extract_relevant_code(state["code"], failing_now)
        return (
            f"{json_example}"
            f"{regression_section}"
            f"{history_section}"
            f"Program description:\n{state['program_description']}\n\n"
            f"Relevant code:\n{with_line_numbers(relevant_code)}\n\n"
            f"Tests:\n{with_line_numbers(state['tests'])}\n\n"
            f"Test output:\n{current_output}"
        )
    return (
        f"{json_example}"
        f"{history_section}"
        f"Problem description:\n{state['program_description']}\n\n"
        f"Current code:\n{with_line_numbers(state['code'])}"
    )


def extract_relevant_code(code: str, failing_tests: list[str]) -> str:
    """Extract only the functions/methods referenced in failing test names."""
    if not failing_tests:
        return code

    # Heuristic: test_foo or test_foo_bar likely tests method 'foo'
    candidates = {
        name.removeprefix("test_").split("_")[0]
        for name in failing_tests
    }

    lines = code.splitlines(keepends=True)
    result = []
    inside = False
    indent_level = None

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            fn_name = stripped.split("(")[0].split()[1]
            inside = any(c in fn_name for c in candidates)
            indent_level = len(line) - len(stripped)

        if inside:
            result.append(line)
        elif result and not stripped and indent_level is not None:
            result.append(line)  # keep blank lines between blocks

    return "".join(result) if result else code  # fall back to full code


def agent_a4_debug(state: AgentState) -> AgentState:
    """A4: explain why tests failed and send that explanation back to A2."""
    print_section(f"[A4 - Debug] analysing failure at iteration {state['iteration']}")

    has_tests = bool(state["tests"])
    system_prompt = _build_system_prompt(has_tests)
    user_prompt = _build_user_prompt(state, has_tests)

    print(f"\n[Prompt for A4]\nSystem:\n{system_prompt}\n\nHuman:\n{user_prompt}\n")
    
    llm = get_llm(temperature=TEMP_DEBUG).with_structured_output(schema=DebugReport, include_raw=True)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    update_token_state(state, response["raw"], "a4")

    parsed = response["parsed"]
    debug_report = (
        f"Error Analysis: {parsed.error_analysis}\n"
        f"Root Cause: {parsed.root_cause}\n"
        f"Suggested Fix: {parsed.suggested_fix}"
    )
    print_section(f"[A4 - Debug] diagnosis\n{debug_report}")

    # Derive has_issues programmatically instead of trusting the model's self-assessment
    has_issues = bool(parsed.error_analysis.strip()) or bool(parsed.root_cause.strip())

    return {
        **state,
        "messages":         [],
        "tests_passed":     False,
        "debug_reports":    state["debug_reports"] + [debug_report],
        "next_agent":       "a2",
        "debug_has_issues": has_issues,
    }