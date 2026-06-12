#!/usr/bin/env python3
"""Executable version of the practice notebook pipeline for the tdlite folder."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "practice"))
sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import END, StateGraph

import utils.prompts as prompts
import utils.io_utils as practice_io_utils
from tdlite.nodes import test_syntax_check
from tdlite.utils.config import TEST_FILE
from tdlite.utils.io_utils import save_text
from tdlite.utils.schemas import Response
from tdlite.utils.state import AgentState


MAX_ITERATIONS = 6


llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.2,
    model_kwargs={"think": "False"},
)


def get_humaneval_prompts() -> list[str]:
    ds = load_dataset("openai/openai_humaneval")
    samples = ds["test"][:10]
    print(samples)
    return samples["prompt"]



def get_mbpp_prompts() -> list[str]:
    ds = load_dataset("google-research-datasets/mbpp", "full")
    samples = ds["test"][:10]
    return samples["text"]


# Agent test 1

def agent_test_1(state: AgentState) -> AgentState:
    print(f"\n A11 received program description, preparing to write structured NL specifications:")
    program_description = state["program_description"]

    system_prompt = prompts.AGENT1_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {program_description}
        Example Response: {prompts.AGENT1_EXAMPLE_1}
    """

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


def agent_test_2(state: AgentState) -> AgentState:
    print(f"\n A12 received structured NL specifications, preparing to write Python tests:")
    program_description = state["program_description"]
    specifications = state["specifications"]

    system_prompt = prompts.AGENT2_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {program_description}
        Specifications: {specifications}
        Example Response: {prompts.AGENT2_EXAMPLE_1}
    """

    llm_with_output = llm.with_structured_output(Response)
    response = llm_with_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    state["tests"] = response.content
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_text(TEST_FILE, state["tests"])
    return state



def route_after_test_compile(state: AgentState) -> Literal["a12", "a11", "__end__"]:
    if state["next_agent"] == "a2":
        return "__end__"
    if state["iteration"] >= MAX_ITERATIONS:
        return "__end__"
    return "a11"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("a11", agent_test_1)
    graph.add_node("a12", agent_test_2)
    graph.add_node("test_syntax_check", test_syntax_check.test_syntax_check)

    graph.set_entry_point("a11")
    graph.add_edge("a11", "a12")
    graph.add_edge("a12", "test_syntax_check")
    graph.add_conditional_edges(
        "test_syntax_check",
        route_after_test_compile,
        {
            "a12": "a12",
            "a11": "a11",
            "__end__": END,
        },
    )
    return graph.compile()


def debug_graph() -> StateGraph:
    # Build a graph that runs only the syntax check against the existing tests file.
    graph = StateGraph(AgentState)
    graph.add_node("test_syntax_check", test_syntax_check.test_syntax_check)
    graph.set_entry_point("test_syntax_check")
    return graph.compile()

def main() -> AgentState:
    
    args = practice_io_utils.parse_args()
    program_description = ""
    tests = ""
    if args.debug:
        print("Debug mode enabled: skipping test generation and using existing tests if available.")
        if TEST_FILE.exists():
            tests = TEST_FILE.read_text(encoding="utf-8")
            print(f"Loaded existing tests from {TEST_FILE} for syntax checking.")
        else:
            print(f"No existing test file found at {TEST_FILE}. Syntax check will run on empty content.")
    else:
        if args.file is not None:
            print(f"Reading program description from file: {args.file}")
            program_description = practice_io_utils.read_program_description(Path(args.file))
        elif args.dataset == "MBPP":
            print("Loading program description from MBPP dataset...")
            program_description = get_mbpp_prompts()[0]
        else:
            print("Reading program description from HumanEval dataset...")
            program_description = get_humaneval_prompts()[0]

    
    
    app = debug_graph() if args.debug else build_graph()


    initial_state: AgentState = {
        "messages": [],
        "program_description": program_description,
        "tests": tests,
        "code": "",
        "specifications": "",
        "compile_output": "",
        "tests_passed": False,
        "test_outputs": [],
        "debug_reports": [],
        "ever_passed_tests": [],
        "next_agent": "a11",
        "iteration": 0,
    }

    print("\n" + "=" * 60)
    print("  LangGraph TDD Pipeline - Starting")
    print(f"  Description file: None, using dataset example")
    print("  A11 structured tests -> A12 python tests -> test_syntax_check")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["next_agent"] == "a2":
        print("  Pipeline finished: generated tests passed the syntax check.")
    else:
        print("  Pipeline stopped before successful syntax validation.")
    print(f"  Iterations used: {final_state['iteration']}")
    print("=" * 60)
    return final_state




if __name__ == "__main__":
    main()
