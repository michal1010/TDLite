import py_compile
from utils.config import PROGRAM_FILE
from utils.state import AgentState


def compile_check(state: AgentState) -> AgentState:
    """Compile the generated program and decide whether it can move to tests."""
    try:
        py_compile.compile(str(PROGRAM_FILE), doraise=True)
        compile_output = "Compilation succeeded."
        next_agent = "run_tests"
    except py_compile.PyCompileError as exc:
        compile_output = str(exc)
        if "```python" in compile_output:
            compile_output = "Code failed, because you included ```python fences. DO NOT include markdown fences, return only code."
        next_agent = "a3"

    print(f"\n{'=' * 60}")
    print(f"[Compile Check] routing -> {next_agent}")
    print(compile_output)
    print("=" * 60)

    return {
        "messages": [],
        "program_description": state["program_description"],
        "tests": state["tests"],
        "code": state["code"],
        "compile_output": compile_output,
        "tests_passed": state["tests_passed"],
        "test_outputs": state["test_outputs"],
        "debug_reports": state["debug_reports"],
        "ever_passed_tests": state["ever_passed_tests"],
        "next_agent": next_agent,
        "iteration": state["iteration"],
        "specifications": state["specifications"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
    }