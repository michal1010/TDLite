from __future__ import annotations

from typing import Literal
from langgraph.graph import END, StateGraph
from agents.a1_test_generation import agent_a11_structured_nl_specifications, agent_a12_test_generation
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from agents.a4_debug import agent_a4_debug
from utils.config import MAX_ITERATIONS
from nodes.compile_check import compile_check
from nodes.test_syntax_check import test_syntax_check
from nodes.run_tests import run_tests
from utils.state import AgentState


def route_after_compile(state: AgentState) -> Literal["run_tests", "a3", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS and state["next_agent"] != "run_tests":
        return "__end__"
    if state["next_agent"] == "run_tests":
        return "run_tests"
    return "a3"


def route_after_test_compile(state: AgentState) -> Literal["a2", "a12", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS and state["next_agent"] != "a2":
        return "__end__"
    if state["next_agent"] == "a2":
        return "a2"
    return "a12"


def route_after_tests(state: AgentState) -> Literal["a4", "__end__"]:
    if state["tests_passed"]:
        return "__end__"
    if state["iteration"] >= MAX_ITERATIONS:
        return "__end__"
    return "a4"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a11", agent_a11_structured_nl_specifications)
    graph.add_node("a12", agent_a12_test_generation)
    graph.add_node("test_syntax_check", test_syntax_check)
    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("compile_check", compile_check)
    graph.add_node("a3", agent_a3_patch_revision)
    graph.add_node("run_tests", run_tests)
    graph.add_node("a4", agent_a4_debug)

    graph.set_entry_point("a11")
    graph.add_edge("a11", "a12")
    graph.add_edge("a12", "test_syntax_check")

    graph.add_conditional_edges(
        "test_syntax_check",
        route_after_test_compile,
        {
            "a2": "a2",
            "a12": "a12",
            "__end__": END,
        },
    )
    graph.add_edge("a2", "compile_check")
    graph.add_conditional_edges(
        "compile_check",
        route_after_compile,
        {
            "run_tests": "run_tests",
            "a3": "a3",
            "__end__": END,
        },
    )
    graph.add_edge("a3", "compile_check")
    graph.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {
            "a4": "a4",
            "__end__": END,
        },
    )
    graph.add_edge("a4", "a2")

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
        "ever_passed_tests": [],
        "next_agent": "a11",
        "iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "compile_iteration": 0,
        "debug_has_issues": True,
        "debug_mode": False,
    }
