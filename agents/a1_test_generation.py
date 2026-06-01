from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import TEST_FILE
from utils.io_utils import save_text
from utils.llm import get_llm
from utils.schemas import Response
from utils.state import AgentState
from utils.prompts import AGENT1_EXAMPLE_1, AGENT1_SYSTEM_PROMPT, AGENT2_EXAMPLE_1, AGENT2_SYSTEM_PROMPT





def agent_a11_structured_nl_specifications(state: AgentState) -> AgentState:
    """A11: write structured natural language specifications from the program description."""
    print(f"\n A11 received program description, preparing to write structured NL specifications:")
    program_description = state["program_description"]

    system_prompt = AGENT1_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {program_description}
        Example Response: {AGENT1_EXAMPLE_1}
    """

    llm = get_llm()

    llm_with_output = llm.with_structured_output(Response)
    response = llm_with_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    specifications = response.content
    state["specifications"] = specifications

    # Write specifications on a file for easier debugging
    with open("outputs/specifications.txt", "w") as f:
        f.write(specifications)
    
    print(f"\n A11 generated structured NL specifications.")
    return state


def agent_a12_test_generation(state: AgentState) -> AgentState:
    """A12: write Python tests from structured natural language specifications."""
    print(f"\n A12 received structured NL specifications, preparing to write Python tests:")
    program_description = state["program_description"]
    specifications = state["specifications"]

    system_prompt = AGENT2_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {program_description}
        Specifications: {specifications}
        Example Response: {AGENT2_EXAMPLE_1}
    """

    llm = get_llm()

    llm_with_output = llm.with_structured_output(Response)
    response = llm_with_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    state["tests"] = response.content
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_text(TEST_FILE, state["tests"])
    return state