from __future__ import annotations

from pathlib import Path
from random import randint
from typing import Literal

from langgraph.graph import END, StateGraph

from datasets import load_dataset
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE, MAX_COMPILE_ITERATIONS
from utils.io_utils import parse_args, read_program_description
from nodes.compile_check import compile_check
from utils.state import AgentState


def get_humaneval_prompts() -> list[str]:
    ds = load_dataset("openai/openai_humaneval")
    samples = ds["test"][:10]
    print(samples)
    return samples["prompt"]


def get_mbpp_prompts() -> list[str]:
    ds = load_dataset("google-research-datasets/mbpp", "full")
    samples = ds["test"][:10]
    return samples["text"]


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


def main() -> AgentState:
    args = parse_args()
    program_description = ""
    description_file = None

    if args.dataset == "MBPP":
        print("Loading program description from MBPP dataset...")
        descriptions = get_mbpp_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]
    elif args.dataset == "HumanEval":
        print(f"Reading program description from HumanEval dataset...")
        descriptions = get_humaneval_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]
    else:
        file_name = args.file or str(DEFAULT_PROGRAM_DESCRIPTION_FILE)
        description_file = Path("inputs") / file_name
        print(f"Reading program description from file: {description_file}")
        program_description = read_program_description(description_file)

    app = build_graph()

    initial_state: AgentState = {
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

    print("\n" + "=" * 60)
    print("  LangGraph Non-TDD No-Iteration Pipeline - Starting")
    print(f"  Description file: {description_file if description_file else 'None, using dataset example'}")
    print("  A2 code -> compile -> done")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["compile_output"] == "Compilation succeeded.":
        print("  Pipeline finished: code compiled successfully.")
    else:
        print("  Pipeline stopped: max compile iterations reached without success.")
    print(f"  Compile iterations used: {final_state['compile_iteration']}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    final_state = main()
    print(f"Input tokens: {final_state['input_tokens']}")
    print(f"Output tokens: {final_state['output_tokens']}")
