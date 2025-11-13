from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pathlib import Path

from lib.ai import init_model
from lib.config import DataConstants, load_config

DEFS = DataConstants()
model_config = DEFS.get_config_for_model('gemini-2.5-pro')
config = load_config(DEFS)

# Discover skills
manager = SkillManager(Path(DEFS.aspects_dir))
manager.discover()
print(manager.list_skills())

# Convert to LangChain tools
tools = create_langchain_tools(manager)

# Create agent
llm = init_model(model_config).bind_tools(create_langchain_tools(manager))
base_prompt = config.load_prompt_file('specfinding/aspects/orchestrator_v2')

# Adjust the prompt for the initial input
is_first_turn = True
prompt = f'{base_prompt}\n\n{config.load_prompt_file('specfinding/aspects/initial_handling')}'

messages = [SystemMessage(prompt)]

# Use agent
while True:
    messages.append(HumanMessage(input("> ".strip())))
    result = llm.invoke({"messages": messages}).content
    print(f'AI: {result}')
    if len(messages) == 2:
        messages[0] = SystemMessage(base_prompt)
    messages.append(AIMessage(result))
