from __future__ import annotations

from typing import Literal
from langgraph.graph import END, StateGraph
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from utils.config import  MAX_COMPILE_ITERATIONS
from nodes.compile_check import compile_check
from utils.state import AgentState


def route_after_compile(state: AgentState) -> Literal["a3", "__end__"]:
    if state["next_agent"] == "run_tests":  # compile succeeded
        return "__end__"
    if state["compile_iteration"] >= MAX_COMPILE_ITERATIONS:
        return "__end__"
    return "a3"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("compile_check", compile_check)
    graph.add_node("a3", agent_a3_patch_revision)

    graph.set_entry_point("a2")
    graph.add_edge("a2", "compile_check")
    graph.add_conditional_edges(
        "compile_check",
        route_after_compile,
        {
            "a3": "a3",
            "__end__": END,
        },
    )
    graph.add_edge("a3", "compile_check")

    return graph.compile()


def create_initial_state(program_description: str) -> AgentState:
    return {
        "messages": [],
        "program_description": program_description,
        "tests": "",
        "code": "",
        "compile_output": "",
        "tests_passed": False,
        "test_outputs": [],
        "debug_reports": [],
        "ever_passed_tests": [],
        "next_agent": "a2",
        "iteration": 0,
        "compile_iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "debug_has_issues": True
    }
