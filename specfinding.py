from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

from lib.ai import init_model
from lib.config import DataConstants

DEFS = DataConstants()
model_config = DEFS.get_config_for_model('gemini-2.5-pro')

# Discover skills
manager = SkillManager(Path(DEFS.aspects_dir))
manager.discover()
print(manager.list_skills())

# Convert to LangChain tools
tools = create_langchain_tools(manager)

# Create agent
llm = init_model(model_config).bind_tools(create_langchain_tools(manager))
prompt = "You are a helpful assistant. use the available skills tools to answer the user queries."

# Use agent
query = "What are Common Architectural Scenarios in python?"
messages = [SystemMessage(prompt), HumanMessage(content=query)]
result = llm.invoke({"messages": messages})
