from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import PROGRAM_FILE, TEMP_GENERATE
from utils.io_utils import print_section, save_text
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState, update_token_state
from utils.debug_helpers import trim_test_output, extract_suggested_fix

import inspect
from typing import TypedDict, Any

import re

_TYPING_NAMES = {
    "List", "Dict", "Tuple", "Set", "Optional", "Union",
    "Any", "Callable", "Type", "Sequence", "Iterator", "Generator"
}

def fix_typing_imports(code: str) -> str:
    if "from typing import" in code or "import typing" in code:
        return code
    used = _TYPING_NAMES & set(re.findall(r'\b([A-Z][a-zA-Z]+)\b', code))
    if not used:
        return code
    return f"from typing import {', '.join(sorted(used))}\n{code}"

def _build_system_prompt(has_tests: bool, task_instruction: str) -> str:
    pipeline_type = "TDD " if has_tests else ""
    naming_rule = (
        "Match the exact class/function names imported in the tests."
        if has_tests else
        "Use the function or class name specified in the problem description."
    )
    edge_case_rule = (
        "Handle every edge case visible in the tests."
        if has_tests else
        "Handle every edge case described in the problem."
    )

    # Note: Kept original logic but fixed indentation to prevent 
    # inspect.cleandoc from preserving unwanted leading spaces on rule lines.
    prompt_template = f"""
    You are a code-writing agent in a {pipeline_type}pipeline.
    {task_instruction}

    Rules:
    - Put ONLY raw Python code in the `content` field. No markdown fences. No explanations.
    - {naming_rule}
    - {edge_case_rule}
    - Do not repeat an approach that has already failed.
    - Do NOT add any import from outputs.generated_program. That import is only for the test file.
    """
    return inspect.cleandoc(prompt_template)

def _build_human_prompt(state: AgentState) -> str:
    """Builds the human prompt using a three-tier context strategy.
    
    Suitable for direct input into LangChain's HumanMessage.
    """
    description   = state["program_description"]
    tests         = state["tests"]
    code          = state["code"]
    debug_reports = state["debug_reports"]
    test_outputs  = state["test_outputs"]

    has_tests = bool(tests)
    human_sections = []
    
    # Tier 1: Problem Description & Test Suite
    human_sections.append(f"Problem description:\n{description}")
    if has_tests:
        human_sections.append(f"Tests to pass:\n{tests}")

    # Tier 2: Compressed Historical Fixes
    if len(debug_reports) > 1:
        history_log = "\n".join(
            f"  Attempt {i + 1}: {extract_suggested_fix(r)}"
            for i, r in enumerate(debug_reports[:-1])
        )
        human_sections.append(inspect.cleandoc(f"""
            Earlier diagnoses and suggested fixes — use as context:
            {history_log}
            If a previous suggestion is still correct, implement it properly this time; 
            do not merely reproduce an unchanged failed implementation.
        """))

    # Tier 3: Latest Diagnostic Context
    if debug_reports:
        human_sections.append(f"Latest diagnosis:\n{debug_reports[-1]}")
    if code:
        human_sections.append(f"Current implementation (fix this):\n{code}")
    if test_outputs:
        trimmed_failure = trim_test_output(test_outputs[-1])
        human_sections.append(f"Test failures:\n{trimmed_failure}")

    return "\n\n".join(human_sections)


def _build_prompt(state: AgentState) -> tuple[str, str]:
    """Builds the system and human prompts.
    
    Returns:
        tuple[str, str]: (system_prompt, human_prompt)
    """
    code          = state["code"]
    tests         = state["tests"]

    is_first_attempt = not code
    has_tests = bool(tests)

    # 1. Compile System Prompt Components
    task_instruction = _determine_task_instruction(has_tests, is_first_attempt)
    system_prompt = _build_system_prompt(has_tests, task_instruction)

    # 2. Compile Human Prompt Components
    human_prompt = _build_human_prompt(state)
    
    if state.get("debug_mode") == True:
        print(f"\n[Prompt for A2]\n\tSystem:\n\t{system_prompt}\n\n\tHuman:\n\t{human_prompt}\n")
    
    return system_prompt, human_prompt


def _determine_task_instruction(has_tests: bool, is_first_attempt: bool) -> str:
    if has_tests:
        return (
            "Write a Python implementation that passes all the tests below."
            if is_first_attempt else
            "Fix the implementation below so it passes the failing tests."
        )
    return (
        "Write a Python implementation for the problem description below."
        if is_first_attempt else
        "Fix the implementation below based on the diagnostic report below."
    )



def agent_a2_patch_creation(state: AgentState) -> dict:
    """
    A2 — Patch Creation

    Input  (from state): program_description, tests, code,
                         debug_reports, test_outputs
    Output (to state):   code, PROGRAM_FILE written to disk,
                         iteration incremented
    """
    print_section(f"[A2 - Patch Creation] attempt {state['iteration'] + 1}")
    
    llm = get_llm(temperature=TEMP_GENERATE).with_structured_output(schema=Response, include_raw=True)
    system_prompt, human_prompt = _build_prompt(state)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    code = response["parsed"].content
    code = fix_typing_imports(code)
    update_token_state(state, response["raw"], "a2")

    save_text(PROGRAM_FILE, code)
    if state.get("debug_mode") == True:
        print(f"[A2] Code written ({len(code.splitlines())} lines) → {PROGRAM_FILE}")

    return {
        "code":           code,
        "compile_output": "",
        "tests_passed":   False,
        "next_agent":     "compile_check",
        "iteration":      state["iteration"] + 1,
        "input_tokens":  state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "compile_iteration": 0,
    }