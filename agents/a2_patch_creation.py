from langchain_core.messages import HumanMessage

from utils.config import PROGRAM_FILE, TEMP_GENERATE
from utils.io_utils import save_text
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState, update_token_state
from utils.debug_helpers import trim_test_output, extract_suggested_fix



def _build_prompt(state: AgentState) -> str:
    """
    Builds the prompt using a three-tier context strategy:

    Tier 1 — always present, always full:
        problem description + test suite

    Tier 2 — older debug reports, one line each (compressed):
        extract only 'Suggested Fix' from debug_reports[:-1]
        ~20 tokens per entry regardless of how verbose A4 was

    Tier 3 — latest attempt, shown in full:
        debug_reports[-1]  — full Error Analysis / Root Cause / Suggested Fix
        test_outputs[-1]   — trimmed to tracebacks only
        code               — current implementation to fix

    Token budget at iteration 8 stays well under 2,500 regardless of
    how many retries have happened.
    """
    description   = state["program_description"]
    tests         = state["tests"]
    code          = state["code"]
    debug_reports = state["debug_reports"]
    test_outputs  = state["test_outputs"]

    is_first_attempt = not code

    if is_first_attempt:
        task = "Write a Python implementation that passes all the tests below."
    else:
        task = "Fix the implementation below so it passes the failing tests."

    prompt = f"""\
You are a code-writing agent in a TDD pipeline.
{task}

Rules:
- Put ONLY raw Python code in the `content` field. No markdown fences. No explanations.
- Match the exact class/function names imported in the tests.
- Handle every edge case visible in the tests.
- Do not repeat an approach that has already failed.
- Do NOT add any import from outputs.generated_program. That import is only for the test file.

Problem description:
{description}

Tests to pass:
{tests}"""


    older_reports = debug_reports[:-1] if len(debug_reports) > 1 else []
    if older_reports:
        lines = "\n".join(
            f"  Attempt {i + 1}: {extract_suggested_fix(r)}"
            for i, r in enumerate(older_reports)
        )
        prompt += f"""

Earlier diagnoses and suggested fixes — use as context:
{lines}
If a previous suggestion is still correct, implement it properly this time; do not merely reproduce an unchanged failed implementation."""

    if debug_reports:
        prompt += f"""

Latest diagnosis:
{debug_reports[-1]}"""

    if code:
        prompt += f"""

Current implementation (fix this):
{code}"""

    if test_outputs:
        prompt += f"""

Test failures:
{trim_test_output(test_outputs[-1])}"""

    return prompt.strip()



def agent_a2_patch_creation(state: AgentState) -> dict:
    """
    A2 — Patch Creation

    Input  (from state): program_description, tests, code,
                         debug_reports, test_outputs
    Output (to state):   code, PROGRAM_FILE written to disk,
                         iteration incremented
    """
    print(f"\n{'='*60}")
    print(f"[A2 - Patch Creation] attempt {state['iteration'] + 1}")
    print("="*60)

    llm = get_llm(temperature=TEMP_GENERATE).with_structured_output(schema=Response, include_raw=True)
    prompt = _build_prompt(state)
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response["parsed"].content
    update_token_state(state, response["raw"], "a2")

    save_text(PROGRAM_FILE, code)

    print(f"[A2] Code written ({len(code.splitlines())} lines) → {PROGRAM_FILE}")

    return {
        "code":           code,
        "compile_output": "",
        "tests_passed":   False,
        "next_agent":     "compile_check",
        "iteration":      state["iteration"] + 1,
        "input_tokens":  state["input_tokens"],
        "output_tokens": state["output_tokens"],
    }