from typing import Dict

from state import AgentState

def parse_token_amounts(state: AgentState) -> Dict[str, int]:
    """Extract token counts per agent for csv storage."""
    token_counts = {}
    agent_names = ["a1", "test_syntax_check", "a2", "a3", "a4"]
    for agent in agent_names:
        input_tokens = state["input_tokens"].get(agent, 0)
        output_tokens = state["output_tokens"].get(agent, 0)
        token_counts[agent + "_input_tokens"] = input_tokens
        token_counts[agent + "_output_tokens"] = output_tokens
    return token_counts
