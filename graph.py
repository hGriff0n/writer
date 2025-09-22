from typing import Annotated, TypedDict, List, Union

# LangChain core message types
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from agent_base import AgentBase

# -------------------------
# 3. LangGraph Shared State Definition
# -------------------------
AiMessage = Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]
class GraphState(TypedDict):
    # Shared state for LangGraph. Tracks all messages in the conversation.
    messages: Annotated[List[AiMessage], "messages"]


# Not sure if I want this to be a consistent part of the system or just an initial setup helper
class SystemHelper:
    def __init__(self, llm: AgentBase, tools: dict=None):
        self._llm = llm
        self._tool_registry = tools if tools else {}

    def _call_agent(self, messages: List[AiMessage]) -> None:
        return self._llm.call(messages)

    def _call_tool(self, tool_name: str, tool_args: dict, call_id: str) -> None:
        # Validate tool existence
        if tool_name not in self._tool_registry:
            raise NameError(f"⚠️ Tool {tool_name} not found in registry")

        # Invoke the tool using registered function
        output = self._tool_registry[tool_name].invoke(tool_args)
        return [ToolMessage(tool_call_id=call_id, content=str(output))]

    # TODO: me - this has to be the api because that's what the graph expects
    def invoke(self, state: GraphState) -> GraphState:
        try:
            response = self._call_agent(state["messages"])
            state["messages"] = state["messages"] + [response]
        except Exception as e:
            print(f"❌ Agent invocation failed: {str(e)}")
            state["messages"] = state["messages"] + [AIMessage(content=str(e))]
        return state
    
    def invoke_tool(self, state: GraphState) -> GraphState:
        messages = state["messages"]
        last_msg = messages[-1]  # Get last AI message that might have a tool call

        if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
            print("⚠️ No tool calls in last message")
            return {"messages": messages}

        # Extract first tool call from AIMessage
        tool_call = last_msg.tool_calls[0]
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        call_id = tool_call.get("id")

        try:
            response = self._call_tool(name, args, call_id)
            state["messages"] = state["messages"] + response

        except Exception as e:
            print(f"❌ Tool invocation failed: {str(e)}")
            state["messages"] = state["messages"] + [AIMessage(content=str(e))]

        return state