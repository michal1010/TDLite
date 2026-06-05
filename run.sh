#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# How we found Python executable in other project, seemed to work across OS's.
# Have to name your virtual environment ".venv" for this to work.
# `python -m venv .venv` to create the virtual environment.
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    echo "ERROR: Could not find Python in venv or .venv"
    exit 1
fi

rm -f generated_program.py test_generated_program.py

DESCRIPTION_ARG=()
if [ $# -ge 1 ]; then
    DESCRIPTION_ARG=(--file "$1")
fi

"$PYTHON_EXEC" langgraph_agents.py "${DESCRIPTION_ARG[@]}"
