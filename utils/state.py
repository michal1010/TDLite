from __future__ import annotations

from typing import Annotated, Dict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Needs to be updated if we change LLM wrapper to something other than ChatOpenAI()
def update_token_state(state: AgentState, response: AIMessage, agent_name: str) -> None:
    try:
        state["input_tokens"][agent_name] = state["input_tokens"].get(agent_name, 0) + response.usage_metadata.get("input_tokens")
        state["output_tokens"][agent_name] = state["output_tokens"].get(agent_name, 0) + response.usage_metadata.get("output_tokens")
    except AttributeError:
        print(f"Warning: Could not update token counts for {agent_name} due to missing usage metadata.")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    program_description: str
    tests: str
    code: str
    compile_output: str
    tests_passed: bool
    ever_passed_tests: list[str]
    test_outputs: list[str]
    debug_reports: list[str]
    next_agent: str
    iteration: int
    specifications: str
    input_tokens: Dict[str, int]
    output_tokens: Dict[str, int]
    compile_iteration: int
    debug_has_issues: bool
    debug_mode: bool


