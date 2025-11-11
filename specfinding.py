from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools
from langchain.agents import create_openapi_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path

# Discover skills
manager = SkillManager(
    Path(r"C:\Users\ghoop\Desktop\writer\prompts\specfinding\aspects"))
manager.discover()
print(manager.list_skills())

# Convert to LangChain tools
tools = create_langchain_tools(manager)

# Create agent
llm = ChatOpenAI(model="gpt-4")
prompt = "You are a helpful assistant. use the available skills tools to answer the user queries."
agent = create_openapi_agent(
    llm,
    tools,
    system_prompt=prompt
)

# Use agent
query = "What are Common Architectural Scenarios in python?"
messages = [HumanMessage(content=query)]
result = agent.invoke({"messages": messages})
