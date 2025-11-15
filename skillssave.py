from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.agent_toolkits import FileManagementToolkit
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

from lib.ai import init_model
from lib.config import DataConstants, load_config

DEFS = DataConstants()
model_config = DEFS.get_config_for_model('gemini-2.5-pro')
config = load_config(DEFS)

# Load skill tools for dynamic loading
manager = SkillManager(Path(DEFS.aspects_dir))
manager.discover()
tools = create_langchain_tools(manager)
# Workaround because I defined the skills with spaces
for t in tools:
    t.name = t.name.strip().replace(' ', '_').lower()

# Add read file tools so we can dynamically load in files as needed too
tools.extend(FileManagementToolkit(
    root_dir=r"C:\Users\ghoop\Desktop\writer\prompts\specfinding\aspects",
    selected_tools=["read_file"],
).get_tools())

# Create agent
llm = init_model(model_config).bind_tools(tools)
# base_prompt = PromptTemplate.from_template(config.load_prompt_file('specfinding/aspects/orchestrator_v2')).format(tools=tools)
base_prompt = config.load_prompt_file('experiments/fullsession')

# Adjust the prompt for the initial input
is_first_turn = True
prompt = f'{base_prompt}\n\n{config.load_prompt_file('specfinding/aspects/initial_handling')}'

messages = [SystemMessage(prompt)]

from langchain_core.load.dump import dumpd
import json

# Use agent
# TODO: me - Not seeming to actually use the skills at all
while True:
    messages.append(HumanMessage(input("> ").strip()))
    result = llm.invoke(messages)
    # print(f'AI: {result.content[0]['text']}')
    print(result)
    if len(messages) == 2:
        messages[0] = SystemMessage(base_prompt)
    messages.append(result)

    with open('foo.json', 'w+', encoding='utf-8') as f:
        json.dump({'msgs': [dumpd(m) for m in messages]}, f)

# https://github.com/maxvaega/skillkit/blob/main/examples/langchain_agent.py
