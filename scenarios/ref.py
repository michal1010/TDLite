from __future__ import annotations

from typing import Literal
from langgraph.graph import END, StateGraph
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from agents.a4_debug import agent_a4_debug
from utils.config import MAX_ITERATIONS, MAX_COMPILE_ITERATIONS
from nodes.compile_check import compile_check
from utils.state import AgentState


def route_after_compile(state: AgentState) -> Literal["a4", "a3", "__end__"]:
    if state["next_agent"] == "run_tests":  # compile succeeded
        if state["iteration"] >= MAX_ITERATIONS:
            return "__end__"
        return "a4"
    if state["compile_iteration"] >= MAX_COMPILE_ITERATIONS:
        return "__end__"
    return "a3"


def route_after_debug(state: AgentState) -> Literal["a2", "__end__"]:
    if not state["debug_has_issues"]:
        return "__end__"
    if state["iteration"] >= MAX_ITERATIONS:
        return "__end__"
    return "a2"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("compile_check", compile_check)
    graph.add_node("a3", agent_a3_patch_revision)
    graph.add_node("a4", agent_a4_debug)

    graph.set_entry_point("a2")
    graph.add_edge("a2", "compile_check")
    graph.add_conditional_edges(
        "compile_check",
        route_after_compile,
        {
            "a4": "a4",
            "a3": "a3",
            "__end__": END,
        },
    )
    graph.add_edge("a3", "compile_check")
    graph.add_conditional_edges(
        "a4",
        route_after_debug,
        {
            "a2": "a2",
            "__end__": END,
        },
    )

    return graph.compile()

def create_initial_state() -> AgentState:
    return {
        "messages": [],
        "program_description": "",
        "tests": "",
        "code": "",
        "compile_output": "",
        "tests_passed": False,
        "test_outputs": [],
        "debug_reports": [],
        "debug_has_issues": True,
        "ever_passed_tests": [],
        "next_agent": "a2",
        "iteration": 0,
        "compile_iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "debug_mode": False,
    }
