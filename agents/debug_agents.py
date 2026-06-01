from __future__ import annotations

import argparse
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
from nodes import test_syntax_check
from utils.config import TEST_FILE
from utils.io_utils import save_text
from utils.schemas import Response
from utils.state import AgentState

MAX_ITERATIONS = 6

def debug_agent_12():
    # This agent reads the output folder "specificatinos.txt" and produces the tests.
    with open("specifications.txt", "r") as f:
        specifications = f.read()
    print(f"Read specifications:\n{specifications}")
    system_prompt = prompts.AGENT2_SYSTEM_PROMPT
    user_prompt = f"""
        Programming Problem: {specifications}
        Example Response: {prompts.AGENT2_EXAMPLE_1}
    """ 

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0.2,
        model_kwargs={"think": "False"},
    ).with_structured_output(Response)  

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    print(f"\nLLM response for test generation: {response.content}")    
    tests = response.content.strip()
    with open(TEST_FILE, "w") as f:
        f.write(tests)
    print(f"Written generated tests to {TEST_FILE}")

# Create a function that reads the output file "test syntax check" 
#  and performs the syntax check, then print