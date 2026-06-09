from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import TEST_FILE
from utils.io_utils import save_text, print_section
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState, update_token_state
from utils.prompts import TEST_1_EXAMPLE_1, TEST_1_EXAMPLE_2, TEST_1_SYSTEM_PROMPT, TEST_2_EXAMPLE_1, TEST_2_EXAMPLE_2, TEST_2_EXAMPLE_2, TEST_2_SYSTEM_PROMPT

from pydantic import BaseModel, Field
from typing import List

class TestCase(BaseModel):
    description: str = Field(description="One sentence describing what this test verifies")
    input: str = Field(description="Concrete input values with variable names (e.g., grid = [[1, 1]], m = 2)")
    expected_output: str = Field(description="Exact expected return value as a concrete literal")
    comment: str = Field(description="A short comment grouping related cases (e.g., 'Cases: small grid')")

class TestSpecifications(BaseModel):
    test_cases: List[TestCase]


def agent_a11_structured_nl_specifications(state: AgentState) -> AgentState:
    """A11: write structured natural language specifications from the program description."""
    print(f"\n A11 received program description, preparing to write structured NL specifications:")
    program_description = state["program_description"]

    system_prompt = TEST_1_SYSTEM_PROMPT
    user_prompt = f"""
        Example Behavior: {TEST_1_EXAMPLE_1} \n{TEST_1_EXAMPLE_2}
        Programming Problem: {program_description}
        """

    if state.get("debug_mode"):
        print(f"\n[Prompt for A11]\nSystem:\n{system_prompt}\n\nUser:\n{user_prompt}\n")

    llm = get_llm()
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    update_token_state(state, response, "a11")
    specifications = response.content
    if state.get("debug_mode"):
        print(f"\n[Raw LLM Response for A11]\n{specifications}\n")

    state["specifications"] = specifications
    
    with open("outputs/specifications.txt", "w") as f:
        f.write(state["specifications"])
    
    print(f"\n A11 generated structured NL specifications.")
    return state


def agent_a12_test_generation(state: AgentState) -> AgentState:
    """A12: write Python tests from structured natural language specifications."""
    print(f"\n A12 received structured NL specifications, preparing to write Python tests:")
    program_description = state["program_description"]
    specifications = state["specifications"]
    if state.get("debug_mode"):
        print(f"\n[Specifications for A12]\n{specifications}\n")

    system_prompt = TEST_2_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {program_description}
        Specifications: {specifications}
        Example Behavior: {TEST_2_EXAMPLE_1},{TEST_2_EXAMPLE_2}
    """

    llm = get_llm()

    llm_with_output = llm.with_structured_output(schema=Response, include_raw=True)
    response = llm_with_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    state["tests"] = response["parsed"].content
    update_token_state(state, response["raw"], "a12")
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_text(TEST_FILE, state["tests"])
    return state
