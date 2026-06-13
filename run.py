"""
This module defines the TDD workflow using LangGraph. The workflow consists of the following steps:
1. Agent A11 generates structured natural language specifications from the program description.
2. Agent A12 generates tests based on the specifications.
3. The generated tests are checked for syntax errors. If there are syntax errors, the flow goes back to A12 for regeneration. 
    If there are no syntax errors, the flow proceeds to A2.
4. Agent A2 generates code based on the specifications and tests.
5. The generated code is compiled. If there are compilation errors, the flow goes to A3 for patch revision and then back to compilation.
    If there are no compilation errors, the flow proceeds to run tests.
6. The generated code is tested against the generated tests. If the tests do not pass, the flow goes to A4 for debugging and then back to A2 for patch creation.
    If the tests pass, the workflow ends successfully.
    
The workflow also includes iteration limits to prevent infinite loops, and it can be run in a debug mode with more verbose output and no retries.
"""


from __future__ import annotations

from pathlib import Path
from random import randint
import shutil
from langgraph.graph import StateGraph
from experiments.result_utils import append_single_run_results
from utils.config import DEFAULT_PROGRAM_DESCRIPTION_FILE
from experiments.test_metrics import generate_coverage_report, compute_mutation_score
from utils.io_utils import parse_args, read_program_description, get_mbpp_prompts, get_humaneval_prompts, print_section
from nodes.compile_check import compile_check
from nodes.test_syntax_check import test_syntax_check
from nodes.run_tests import run_tests
from utils.state import AgentState
import scenarios.tdd_ref as tdd_ref
import scenarios.base as base
import scenarios.ref as ref
import scenarios.tdd as tdd
from experiments.run_humaneval import run_experiments_humaneval

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"




def choose_scenario(scenario: str) -> tuple[StateGraph, AgentState]:
    match scenario:
        case "default":
            return base.build_graph(), base.create_initial_state()
        case "REF":
            return ref.build_graph(), ref.create_initial_state()
        case "TDD":
            return tdd.build_graph(), tdd.create_initial_state()
        case "TDD_REF":
            return tdd_ref.build_graph(), tdd_ref.create_initial_state()
        case _:
            raise ValueError(f"Unknown scenario: {scenario}")


def _has_generated_test_artifacts(state: AgentState | None) -> bool:
    return bool(
        state
        and state.get("tests")
        and state.get("specifications")
    )


def clear_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUTS_DIR.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_problem(args, graph: StateGraph, initial_state: AgentState, description_file=None, reference_code=None) -> dict:
    results_folder_name = None

    if args.store_results:
        results_folder_name = args.store_results
        print(f"Results will be stored in folder: {results_folder_name}")

    title = (
        f"LangGraph TDD Pipeline - Starting\n"
        f"Description file: {description_file if description_file else 'None, using dataset example'}\n"
        f"Scenario: {args.scenario}"
    )
    print_section(title)


    clear_outputs_dir()
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state["tests_passed"]:
        print_section(f"Pipeline finished: all tests passed.\nIterations used: {final_state['iteration']}")
    elif not final_state.get("tests") and not final_state.get("debug_has_issues", True):
        print_section(f"Pipeline finished: code review found no issues.\nIterations used: {final_state['iteration']}")
    elif not final_state.get("tests") and final_state.get("next_agent") == "run_tests":
        print_section(f"Pipeline finished: code compiled successfully.\nIterations used: {final_state['iteration']}")
    else:
        print_section(f"Pipeline stopped before all tests passed.\nIterations used: {final_state['iteration']}")
    print("=" * 60)

    print(f"Input tokens: {final_state['input_tokens']}")
    print(f"Output tokens: {final_state['output_tokens']}")
    save_generated_test_artifacts = _has_generated_test_artifacts(final_state)
    report = None
    mutation_results = None
    if save_generated_test_artifacts:
        report = generate_coverage_report(verbose=True)
        mutation_results = compute_mutation_score(
            code_text=reference_code,
            debug=args.mutation_debug,
        )
    else:
        print("Coverage report skipped: this scenario did not generate tests/specifications.")
    if results_folder_name:
        append_single_run_results(
            final_state=final_state,
            coverage_report=report,
            results_folder=results_folder_name,
            save_generated_test_artifacts=save_generated_test_artifacts,
            mutation_results=mutation_results,
        )


def choose_sample_run(args) -> tuple[str, str | None]:
    reference_code = None
    if args.file is not None:
        file_name = args.file or str(DEFAULT_PROGRAM_DESCRIPTION_FILE)
        description_file = Path("inputs") / file_name
        print(f"Reading program description from file: {description_file}")
        program_description = read_program_description(description_file)
    elif args.dataset == "MBPP":
        print("Loading program description from MBPP dataset...")
        descriptions, reference_solutions = get_mbpp_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]
        reference_code = reference_solutions[r]
    elif args.dataset == "HumanEval":
        print("Reading program description from HumanEval dataset...")
        descriptions, canonical_solutions = get_humaneval_prompts()
        r = randint(0, len(descriptions) - 1)
        program_description = descriptions[r]
        reference_code = canonical_solutions[r]
    return program_description, reference_code


def main():
    args = parse_args()
    program_description = ""
    description_file = None

    graph, initial_state = choose_scenario(args.scenario)
    print(f"Sample run: {args.sample_run}")
    if args.debug:
        import utils.config as _cfg
        _cfg.MAX_ITERATIONS = 1
        print("[DEBUG MODE] max iterations set to 1")
    if args.sample_run == False:
        program_description, reference_code = choose_sample_run(args)
        initial_state["program_description"] = program_description
        print(f"Debug mode: {args.debug}")
        run_problem(args, graph, initial_state, description_file, reference_code)
    else:
        clear_outputs_dir()
        run_experiments_humaneval(args, graph, initial_state)


if __name__ == "__main__":
    main()
