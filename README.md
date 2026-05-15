# TDLite

TDLite is a small experiment in building a simple **TDD AI pipeline** with
LangGraph and a local Ollama model.

The pipeline starts from a plain-text problem description and lets several agents
work together:

1. **A1 - Test Generation** writes unit tests from the problem description.
2. **A2 - Patch Creation** writes the program implementation from the tests.
3. **Compile Check** verifies that the generated program is valid Python.
4. **A3 - Patch Revision** fixes compilation errors when they appear.
5. **Test Run** executes the generated tests.
6. **A4 - Debug** explains failing tests and sends feedback back to A2.

The loop stops when all tests pass or when the maximum number of iterations is
reached.

## Project files

- `langgraph_agents.py` - the TDD pipeline
- `program_description_easy.txt` - an easier example problem
- `program_description_hard.txt` - a harder example problem
- `generated_program.py` - created by the pipeline
- `test_generated_program.py` - created by the pipeline

## Requirements

- Python 3.10+
- Ollama running locally
- Python dependencies from `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Ollama setup

In this experiment qwen3:0.6b is used. Start Ollama if it is not already running:

```bash
ollama run qwen3:0.6b
```

Pull the model used by the pipeline:

```bash
ollama pull qwen3:0.6b
```

List installed models:

```bash
ollama list
```

Check whether a model is currently loaded and what it is doing:

```bash
ollama ps
```

```bash
curl http://localhost:11434/api/tags
```

## How to run

Run the pipeline with the default easy description:

```bash
python3 langgraph_agents.py
```

Run it with a specific problem description file:

```bash
python3 langgraph_agents.py program_description_hard.txt
```

The script prints the chosen description file when it starts.

## What the program generates

During a run, the pipeline writes:

```text
test_generated_program.py
generated_program.py
```

The generated test file is written first. The generated implementation is then
revised until:

- it compiles,
- the tests pass,
- or the pipeline reaches the maximum iteration limit.

## Typical flow

```text
problem description
        ↓
A1 writes tests
        ↓
A2 writes code
        ↓
compile check
   ↙         ↘
A3 fix  ->   run tests
               ↙      ↘
          A4 debug    finish
               ↓
              A2
```

## Notes

- The pipeline currently uses the local model `qwen3:0.6b`.
- Smaller models may produce weak tests or incorrect patches more often.
- If the model is CPU-only, responses can be slow, especially when prompts get
  large.
- `ollama ps` is the quickest way to inspect the currently loaded model while a
  run is in progress.
