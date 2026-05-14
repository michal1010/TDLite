from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI  # works with any OpenAI-compatible endpoint
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import random

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int
    next_agent: str
    iteration: int


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen3:4b-instruct",           
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.3,
    )


MAX_ITERATIONS = 5 


def agent_a1_test_generation(state: AgentState) -> AgentState:
    """A1  Test Generation: receives an initial number, increments it, passes on."""
    llm = get_llm()
    counter = state["counter"]
    new_counter = counter + random.randint(0, 10)

    prompt = (
        f"You are Agent A1 (Test Generation). "
        f"You received the number {counter}. "
        f"Increment it by a random number from 0 to 10 to get {new_counter} and introduce yourself briefly, "
        f"then announce the new number you are passing forward."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n{'='*60}")
    print(f"[A1 - Test Generation] counter: {counter} -> {new_counter}")
    print(f"Response: {response.content}")
    print('='*60)

    return {
        "messages": [response],
        "counter": new_counter,
        "next_agent": "a2",
        "iteration": state.get("iteration", 0),
    }


def agent_a2_patch_creation(state: AgentState) -> AgentState:
    """A2  Patch Creation: receives a number, increments it, decides what happens next."""
    llm = get_llm()
    counter = state["counter"]
    new_counter = counter + random.randint(0, 10)
    iteration = state.get("iteration", 0)


    if new_counter % 2 == 0:
        routing_decision = "a3"
        routing_reason = "the patch looks good but needs revision (even counter)"
    elif iteration >= MAX_ITERATIONS:
        routing_decision = "end"
        routing_reason = f"maximum iterations ({MAX_ITERATIONS}) reached – finishing"
    else:
        routing_decision = "a4"
        routing_reason = "the patch has issues and needs debugging (odd counter)"

    prompt = (
        f"You are Agent A2 (Patch Creation). "
        f"You received the number {counter}. "
        f"Increment it by a random number from 0 to 10 to get {new_counter}. "
        f"Introduce yourself briefly, announce the new number, "
        f"and explain your routing decision: {routing_reason}."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n{'='*60}")
    print(f"[A2 - Patch Creation] counter: {counter} -> {new_counter} | routing -> {routing_decision}")
    print(f"Response: {response.content}")
    print('='*60)

    return {
        "messages": [response],
        "counter": new_counter,
        "next_agent": routing_decision,
        "iteration": iteration + 1,
    }


def agent_a3_patch_revision(state: AgentState) -> AgentState:
    """A3  Patch Revision: optional node, increments and finishes."""
    llm = get_llm()
    counter = state["counter"]
    new_counter = counter + random.randint(0, 10)

    prompt = (
        f"You are Agent A3 (Patch Revision). "
        f"You received the number {counter}. "
        f"Increment it by a random number from 0 to 10 to get {new_counter}. "
        f"Introduce yourself briefly and announce the final revised number."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n{'='*60}")
    print(f"[A3 - Patch Revision] counter: {counter} -> {new_counter}")
    print(f"Response: {response.content}")
    print('='*60)

    return {
        "messages": [response],
        "counter": new_counter,
        "next_agent": "end",
        "iteration": state.get("iteration", 0),
    }


def agent_a4_debug(state: AgentState) -> AgentState:
    """A4  Debug: increments the number and loops back to A2."""
    llm = get_llm()
    counter = state["counter"]
    new_counter = counter + random.randint(0, 10)

    prompt = (
        f"You are Agent A4 (Debug). "
        f"You received the number {counter}. "
        f"Increment it by a random number from 0 to 10 to get {new_counter}. "
        f"Introduce yourself briefly, announce the new number, "
        f"and explain that you are sending it back to Agent A2 for another patch attempt."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n{'='*60}")
    print(f"[A4 - Debug] counter: {counter} -> {new_counter} | routing -> a2 (loop)")
    print(f"Response: {response.content}")
    print('='*60)

    return {
        "messages": [response],
        "counter": new_counter,
        "next_agent": "a2",
        "iteration": state.get("iteration", 0),
    }


def router(state: AgentState):
    if state['counter'] % 2 == 0:
        return {"next": "a3"}
    else:
        return {"next": "a4"}


def route_after_router(state: AgentState) -> Literal["a3", "a4", "__end__"]:
    decision = state["next_agent"]
    if decision == "a3":
        return "a3"
    elif decision == "a4":
        return "a4"
    else:
        return "__end__"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("a1", agent_a1_test_generation)
    graph.add_node("a2", agent_a2_patch_creation)
    graph.add_node("router", router)
    graph.add_node("a3", agent_a3_patch_revision)
    graph.add_node("a4", agent_a4_debug)

    graph.set_entry_point("a1")
    graph.add_edge("a1", "a2")
    graph.add_edge("a2", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "a3": "a3",
            "a4": "a4",
            "__end__": END,
        },
    )

    graph.add_edge("a3", "a2")
    graph.add_edge("a4", "a2")

    return graph.compile()




def main():
    app = build_graph()

    initial_state: AgentState = {
        "messages": [],
        "counter": 0,
        "next_agent": "a1",
        "iteration": 0,
    }

    print("\n" + "="*60)
    print("  LangGraph Multi-Agent Pipeline - Starting")
    print("  Architecture: A1 -> A2 -> A3(cond) || A4 -> A2(loop)")
    print("="*60)

    final_state = app.invoke(initial_state)

    print("\n" + "="*60)
    print(f"  Pipeline finished. Final counter value: {final_state['counter']}")
    print("="*60)

    return final_state


if __name__ == "__main__":
    main()