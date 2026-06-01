from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    program_description: str
    tests: str
    code: str
    compile_output: str
    tests_passed: bool
    ever_passed_tests: list[str]
    test_outputs: list[str]
    debug_reports: list[str]
    next_agent: str
    iteration: int
    specifications: str

