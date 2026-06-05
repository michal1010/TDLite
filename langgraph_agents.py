from __future__ import annotations

from pathlib import Path
from random import randint
from typing import Literal

from langgraph.graph import END, StateGraph

from datasets import load_dataset
from agents.a1_test_generation import agent_a11_structured_nl_specifications, agent_a12_test_generation
from agents.a2_patch_creation import agent_a2_patch_creation
from agents.a3_patch_revision import agent_a3_patch_revision
from agents.a4_debug import agent_a4_debug
from experiments.result_utils import append_run_results
from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE, MAX_ITERATIONS
from experiments.test_metrics import generate_coverage_report
from utils.io_utils import parse_args, read_program_description
from nodes.compile_check import compile_check
from nodes.test_syntax_check import test_syntax_check
from nodes.run_tests import run_tests
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
    # graph.add_edge("a11", "a12")
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


def main() -> AgentState:
    args = parse_args()
    program_description = ""
    description_file = None
    if args.debug:
        import utils.config as _cfg
        _cfg.MAX_ITERATIONS = 1
        print("[DEBUG MODE] max iterations set to 1")
    if args.file is not None:
        file_name = args.file or str(DEFAULT_PROGRAM_DESCRIPTION_FILE)
        description_file = Path("inputs") / file_name
        print(f"Reading program description from file: {description_file}")
        program_description = read_program_description(description_file)
    elif args.dataset == "MBPP":
        print("Loading program description from MBPP dataset...")
        descriptions = get_mbpp_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]
    elif args.dataset == "HumanEval":
        print(f"Reading program description from HumanEval dataset...")
        descriptions = get_humaneval_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]


    results_folder_name = None

    if args.store_results:
        results_folder_name = args.store_results
        print(f"Results will be stored in folder: {results_folder_name}")

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
        "next_agent": "a1",
        "iteration": 0,
        "specifications": "",
        "input_tokens": {},
        "output_tokens": {},
        "compile_iteration": 0,
        "debug_has_issues": True
    }

    print("\n" + "=" * 60)
    print("  LangGraph TDD Pipeline - Starting")
    print(f"  Description file: {description_file if description_file else 'None, using dataset example'}")
    print("  A1 tests -> test compile -> A2 code -> compile -> tests -> debug loop")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["tests_passed"]:
        print("  Pipeline finished: all tests passed.")
    else:
        print("  Pipeline stopped before all tests passed.")
    print(f"  Iterations used: {final_state['iteration']}")
    print("=" * 60)

    return final_state, results_folder_name


if __name__ == "__main__":
    final_state, results_folder_name = main()
    print(f"Input tokens: {final_state['input_tokens']}")
    print(f"Output tokens: {final_state['output_tokens']}")
    report = generate_coverage_report(verbose=True)
    if results_folder_name:
        append_run_results(final_state=final_state, coverage_report=report, results_folder=results_folder_name)