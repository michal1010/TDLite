"""
This module defines the TDD workflow using LangGraph. The workflow consists of the following steps:
1. Agent A11 generates structured natural language specifications from the program description.
2. Agent A12 generates tests based on the specifications.
3. The generated tests are checked for syntax errors. If there are syntax errors, the flow goes back to A12 for regeneration.
   If there are no syntax errors, the flow proceeds to A2.
4. Agent A2 generates code based on the specifications and tests.
5. The generated code is compiled. If there are compilation errors, the flow goes to A3 for patch revision and then back to compilation.
   If there are no compilation errors, the flow proceeds to run tests.
6. The generated code is tested against the generated tests. If the tests do not pass, the workflow ends.
   If the tests pass, the workflow ends successfully.

The workflow also includes iteration limits to prevent infinite loops, and it can be run in a debug mode with more verbose output and no retries.
"""

from __future__ import annotations

from pathlib import Path
from random import randint
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.a1_test_generation import agent_a11_structured_nl_specifications, agent_a12_test_generation
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from experiments.test_metrics import generate_coverage_report
from nodes.compile_check import compile_check
from nodes.run_tests import run_tests
from nodes.test_syntax_check import test_syntax_check
from utils.config import MAX_ITERATIONS, MAX_TEST_ITERATIONS, MAX_COMPILE_ITERATIONS
from utils.io_utils import get_humaneval_prompts, get_mbpp_prompts, parse_args, read_program_description
from utils.state import AgentState


def route_after_compile(state: AgentState) -> Literal["run_tests", "a3", "__end__"]:
    if state["next_agent"] == "run_tests":
        return "run_tests"
    if state["compile_iteration"] >= MAX_COMPILE_ITERATIONS:
        return "__end__"
    if state["iteration"] >= MAX_ITERATIONS:
        return "__end__"
    return "a3"


def route_after_test_compile(state: AgentState) -> Literal["a2", "a12", "__end__"]:
    if state["next_agent"] == "a2":
        return "a2"
    if state["test_iteration"] >= MAX_TEST_ITERATIONS:
        return "__end__"
    return "a12"

def route_after_tests(state: AgentState) -> Literal["__end__"]:
    return "__end__"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a11", agent_a11_structured_nl_specifications)
    graph.add_node("a12", agent_a12_test_generation)
    graph.add_node("test_syntax_check", test_syntax_check)
    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("compile_check", compile_check)
    graph.add_node("a3", agent_a3_patch_revision)
    graph.add_node("run_tests", run_tests)

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
            "__end__": END,
        },
    )

    return graph.compile()


def create_initial_state(debug: bool = False) -> AgentState:
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
        "test_iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "compile_iteration": 0,
        "debug_has_issues": True,
        "debug_mode": debug,
    }



