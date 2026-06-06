from __future__ import annotations

import datetime
from pathlib import Path
from random import randint
from langgraph.graph import StateGraph

from datasets import load_dataset
from experiments.result_utils import append_single_run_results
from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE
from experiments.test_metrics import generate_coverage_report
from utils.io_utils import parse_args, read_program_description
from utils.state import AgentState
import langgraph_s4
import langgraph_s1
import langgraph_s2
from run_humaneval import print_summary, run_experiments_humaneval

def get_humaneval_prompts() -> list[str]:
    ds = load_dataset("openai/openai_humaneval")
    samples = ds["test"][:10]
    return samples["prompt"]



def get_mbpp_prompts() -> list[str]:
    ds = load_dataset("google-research-datasets/mbpp", "full")
    samples = ds["test"][:10]
    return samples["text"]


def choose_scenario(scenario: str, program_description: str) -> tuple[StateGraph, AgentState]:
    if scenario == "s1":
        return langgraph_s1.build_graph(), langgraph_s1.create_initial_state(program_description)
    elif scenario == "s2":
        return langgraph_s2.build_graph(), langgraph_s2.create_initial_state(program_description)
    elif scenario == "s3d":
        return ValueError("Scenario s3 is not implemented yet")
    elif scenario == 's4':
        return langgraph_s4.build_graph(), langgraph_s4.create_initial_state(program_description)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

def run_problem(program_description: str, args, graph: StateGraph, initial_state: AgentState, description_file=None) -> dict:

    results_folder_name = None

    if args.store_results:
        results_folder_name = args.store_results
        print(f"Results will be stored in folder: {results_folder_name}")

    print("\n" + "=" * 60)
    print("  LangGraph TDD Pipeline - Starting")
    print(f"  Description file: {description_file if description_file else 'None, using dataset example'}")
    print(f'  Scenario: {args.scenario}')
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["tests_passed"]:
        print("  Pipeline finished: all tests passed.")
    else:
        print("  Pipeline stopped before all tests passed.")
    print(f"  Iterations used: {final_state['iteration']}")
    print("=" * 60)

    print(f"Input tokens: {final_state['input_tokens']}")
    print(f"Output tokens: {final_state['output_tokens']}")
    report = generate_coverage_report(verbose=True)
    if results_folder_name:
        append_single_run_results(final_state=final_state, coverage_report=report, results_folder=results_folder_name)

def main() -> AgentState:
    args = parse_args()
    program_description = ""
    description_file = None

    if args.debug:
        import utils.config as _cfg
        _cfg.MAX_ITERATIONS = 1
        print("[DEBUG MODE] max iterations set to 1")

    if args.sample_run:
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

        graph, initial_state = choose_scenario(args.scenario, program_description)
        run_problem(program_description, args, graph, initial_state, description_file)
    else:
        run_experiments_humaneval(args)


if __name__ == "__main__":
    main()