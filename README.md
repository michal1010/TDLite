# TDLite

TDLite is a small experiment in building a **TDD AI pipeline** with LangGraph and a local Ollama model.

The pipeline starts from a plain-text problem description and lets several agents work together:

1. **A11 - Specification Generation** writes structured natural language specs from the problem description.
2. **A12 - Test Generation** writes unit tests from the specs.
3. **A2 - Patch Creation** writes the implementation from the tests.
4. **Compile Check** verifies the generated program is valid Python.
5. **A3 - Patch Revision** fixes compilation errors when they appear.
6. **Test Run** executes the generated tests.
7. **A4 - Debug** explains failing tests and sends feedback back to A2.

The loop stops when all tests pass or the maximum number of iterations is reached.

## Project Structure

- `scenarios/` — The four pipeline graphs: `tdd.py`, `tdd_ref.py`, `ref.py`, `base.py`.
- `agents/` — Individual agent implementations (A1, A2, A3, A4).
- `nodes/` — Shared pipeline nodes (compile check, test run, syntax check).
- `experiments/` — Evaluation utilities: HumanEval loader, coverage/mutation metrics, result storage.
- `utils/` — Config, LLM wrapper, prompts, state definition.
- `outputs/` — Generated files written during a run (`generated_program.py`, `generated_tests.py`).
- `inputs/` — Optional hand-written problem description files.

## Requirements

- Python 3.10+
- Ollama running locally
- Python dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

## Ollama Setup

The pipeline uses `qwen2.5-coder:7b` by default (change to `3b` in `.env` or `utils/config.py` if needed).

```bash
ollama pull qwen2.5-coder:7b
ollama run qwen2.5-coder:7b
```

Useful Ollama commands:

```bash
ollama list                         # list installed models
ollama ps                           # check what is currently loaded
curl http://localhost:11434/api/tags
```

## Scenarios

| Flag | Graph | Description |
|------|-------|-------------|
| `default` | A2 → compile → A3 | No TDD, no debug loop |
| `REF` | A2 → compile → A3/A4 | No TDD, with A4 debug loop |
| `TDD` | A11 → A12 → A2 → compile → A3 → run tests | TDD, single pass |
| `TDD_REF` | A11 → A12 → A2 → compile → A3 → run tests → A4 | TDD with A4 debug loop |

## How to Run

Set up a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate (or) .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The entry point is `run.py`.

**Run one random problem from a dataset (single run):**

```bash
python run.py -d HumanEval -s TDD
python run.py -d MBPP -s TDD_REF --store_results experiment_x
```

**Run from a file in `inputs/`:**

```bash
python run.py -f input_easy.txt -s TDD --store_results experiment_x
```

**Run a batch of HumanEval problems (full experiment):**

```bash
python run.py -d HumanEval -s TDD --sample_run --range 0 9 --store_results experiment_x
python run.py -d HumanEval -s TDD --sample_run --ids 0 5 12 --store_results experiment_x
```

## Flags

| Flag | Description |
|------|-------------|
| `-s`, `--scenario` | Pipeline to run: `TDD`, `TDD_REF`, `REF`, `default` |
| `-d`, `--dataset` | Dataset for random single run: `HumanEval` or `MBPP` |
| `-f`, `--file` | Read problem from `inputs/<name>` instead of a dataset |
| `--sample_run` | Run a batch HumanEval experiment (with `--range` or `--ids`) instead of a single random problem |
| `--range START END` | Inclusive HumanEval ID range for batch runs |
| `--ids ID ...` | Specific HumanEval IDs for batch runs |
| `--store_results <folder>` | Write artifacts to `experiments/results/<folder>/` |
| `--debug` | Set max iterations to 1 for quick testing |
| `--mutation_debug` | Print verbose mutation scoring output (temp dir, pytest output, mutmut output) |

## Results Storage

When `--store_results <folder>` is passed, each run writes to:

```
experiments/results/<folder>/
├── results.csv                        ← one row per problem; coverage + mutation columns
└── experiment_files/
    └── instance_N/
        ├── run_summary.json           ← all metrics as JSON
        ├── coverage_report.json       ← line/branch coverage detail
        ├── agent_outputs.json         ← generated code, tests, specs from pipeline state
        ├── generated_program.py       ← snapshot of generated implementation
        ├── generated_tests.py         ← snapshot of generated tests
        ├── specifications.txt         ← snapshot of A11 specs
        └── task.txt                   ← the problem description
```

## Metrics

**Coverage** (line and branch) measures how well the generated tests exercise the generated code.

**Mutation score** measures test quality: generated tests are run against mutations of the canonical solution. A score of `1.0` means all mutants were killed. A score of `-1.0` means scoring was skipped or failed (check `mutation_warning` in `run_summary.json`). Only computed for TDD scenarios that produce tests.

## Typical Flow (TDD_REF)

```
problem description
        ↓
A11 writes specs
        ↓
A12 writes tests
        ↓
syntax check
   ↙        ↘
A12         A2 writes code
                  ↓
            compile check
              ↙         ↘
         A3 fix →     run tests
                       ↙      ↘
                  A4 debug   finish
                       ↓
                      A2
```
