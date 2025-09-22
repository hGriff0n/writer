"""
Copyright (c) 2025 AI Leader X (aileaderx.com). All Rights Reserved.

This software is the property of AI Leader X. Unauthorized copying, distribution,
or modification of this software, via any medium, is strictly prohibited without
prior written permission. For inquiries, visit https://aileaderx.com
"""

# -------------------------
# Imports
# -------------------------
# LangGraph core for graph-based agent flow
from typing_extensions import Literal
from langgraph.graph import StateGraph, END

# LangChain core message types
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangChain tool definition decorator
from langchain_core.tools import tool

# Utility to convert LangChain tools into OpenAI-compatible schema
from langchain_core.utils.function_calling import convert_to_openai_tool

import logging


from agent_base import ModelAdapter, AgentBase
import agent_base
from graph import SystemHelper, GraphState


# -------------------------
# 1. TOOL DEFINITIONS
# -------------------------
@tool
def calculator(expression: str) -> str:
    """
    A basic calculator tool that evaluates mathematical expressions.
    Example input: '2 + 2', '5 * 10'
    """
    print("🧮 Evaluating expression:", expression)
    logging.info(f"Evaluating: {expression}")
    return expression


# Register tools and convert them to OpenAI-compatible schema
tools = [calculator]
openai_tools = [
    convert_to_openai_tool(tool) for tool in tools
]  # Needed for models that support OpenAI-style tool calling
tool_registry = {
    tool.name: tool for tool in tools
}  # Mapping tool name → function reference

# -------------------------
# 2. LLM Setup (LM Studio or other local OpenAI-compatible API)
# -------------------------
llm = AgentBase(ModelAdapter.mistral())
marshall = SystemHelper(llm, tools=tool_registry)


# -------------------------
# 6. Agent Nodes
# TODO: Expand a bit more into defined objects
# -------------------------
def generator(state: GraphState) -> Command[Literal["librarian"]]:
    state = marshall.invoke(state)
    response = state["messages"][-1]
    print("🤖 Generator response:", response.content)
    return Command(
        update={
            "messages": [
                SystemMessage(content=agent_base.WORLD_LIBRARIAN_PROMPT),
                response,
            ]
        },
        goto="librarian",
    )


def librarian(state: GraphState) -> Command[Literal[END]]:
    ask = input("What questions would you ask to test the consistency of the world rules? ")
    print(
        "🤖 Librarian response:",
        marshall.invoke(
            {"messages": state["messages"] + [HumanMessage(content=ask)]}
        )["messages"][-1].content,
    )
    return Command(
        goto=END,
    )


# -------------------------
# 7. LangGraph DAG Setup
# -------------------------
# Initialize LangGraph with shared state definition
builder = StateGraph(GraphState)

# Define graph nodes and their functions
# builder.add_node("agent", lambda x: marshall.invoke(x))
# builder.add_node("tool", lambda x: marshall.invoke_tool(x))
builder.add_node("agent", generator)
builder.add_node("librarian", librarian)

# Set the start of the graph
builder.set_entry_point("agent")
builder.add_edge("librarian", END)

# Compile graph into runnable object
graph = builder.compile()


# -------------------------
# 8. Run the Graph
# -------------------------
if __name__ == "__main__":
    # Define conversation input
    inputs = {
        "messages": [
            SystemMessage(content=agent_base.WORLD_GENERATOR_PROMPT),
            HumanMessage(content="""Wheel of Time, pre-Breaking of the World era."""),
        ]
    }

    # TODO: me - Investigate ASCII throbbers
    print("\n🚀 Running agents...\n")
    result = graph.invoke(inputs)  # Run the compiled LangGraph DAG with input state
