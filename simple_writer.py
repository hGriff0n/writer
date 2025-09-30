
import json
import os
import re
import yaml

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

with open('./data/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_CHOICE = 'gemini'
llm_config = config['ai-providers'][MODEL_CHOICE]
os.environ[llm_config['api-key']['name']] = llm_config['api-key']['value']

model = init_chat_model(
    llm_config['name'], model_provider=llm_config.get('provider'))
# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/

# Helpers for loading data from prompt and story files
# TODO: me - Not sure if this is the best approach for initial development
def load_prompt(config, prompt: str) -> str:
    with open(f'./{config['prompt-dir']}/{prompt}.md', 'r') as f:
        return f.read()
    
# TODO: me - stories should probably come with their own yaml file
def load_story_file(config, story: str, file: str) -> str:
    with open(f'./{config['story-dir']}/{story}/{file}.md', 'r') as f:
        return f.read()

STORY = 'reality'
WRITER_PROMPT = load_prompt(config, 'simple_writer')
INITIAL_INPUT = load_story_file(config, STORY, 'input')

def message(role, content):
    return {'user': role, 'content': content}

chat_log = {
    'template': WRITER_PROMPT,
    'conversation': []
}

#
# Wrapper for sending a message to the llm
#
# Also automatically records a history of all chat communications to a log file
# so that I can easily upload these to an analysis prompt that can identify was
# of improving the initial prompt (based on the corrections I had to make)
# TODO: me - Add typings
#
def send_message(content):
    # Call the llm and record the request in the chat-log
    response = model.invoke(input=[
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=content)
    ])
    chat_log['conversation'].extend([
        {'role': 'ME', 'msg': content},
        {'role': 'AI', 'msg': response.content}
    ])
    return response.content

# Helper to extract xml encoded text sections
# This is useful because llms prefer xml for referential data for some reason
def extract_between_tags(tag, text):
    m = re.match(f"<{tag}>((?:.|[\r\n])*)</{tag}>", text)
    return m and m.group(1) or ""

# 
# Preparing initial story context
# TODO: me - Add error handling when this is abstracted
# 
scene = ""
constraints = load_story_file(config, STORY, 'input')

# 
# Start the writing by sending the initial, context-less, direction
# 
context = []
response = send_message(
    f'Plot Direction: {scene}\n{constraints}', context)
context.append(AIMessage(content=response))
print(response)


#
# Keep writing until you want to stop
# 
# At the moment, there is one "commands":
#   - exit, /finish: stop the loop
# 
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
while True:
    prompt = input("Change> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break

    # Need a better way to continue on from the previous location
    scene = f"Plot Direction: \"{prompt}\""
    response = send_message(f'{scene}\n{constraints}', context)
    context.append(AIMessage(content=response))
    print(response)


# TODO: me - Load these paths from a common config file
# TODO: me - Generate filename based on conversation to simplify loading
with open(f'./{config['output-dir']}/data.json', 'a', encoding='utf-8') as f:
    json.dump(chat_log, f, ensure_ascii=False, indent=4)