
import json
import os
import re
import yaml

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# 
# Load config file
#
with open('./data/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

#
# Initialize AI Model
#
MODEL_CHOICE = 'gemini'
llm_config = config['ai-providers'][MODEL_CHOICE]
os.environ[llm_config['api-key']['name']] = llm_config['api-key']['value']

model = init_chat_model(
    llm_config['name'], model_provider=llm_config.get('provider'))
# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/

### Input Specification
# If length is an issue
# Prose quality and natural paragraphing are ALWAYS more important than hitting a specific paragraph count.

# Helpers for loading data from prompt and story files
# TODO: me - move this to a common library
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
# This automatically truncates the provided context to the last response
# from the model.
# TODO: me - The truncation works for now, because the story concept
# I'm testing is episodic and doesn't need more history
#
def send_message(content, context):
    if context is None:
        raise Exception("Context must be specified")

    # Assemble prompt and context
    # TODO: me - Eventually use more memory channels
    input = [SystemMessage(content=WRITER_PROMPT)]
    if context:
        input.append(SystemMessage(f'<story_so_far>{context[-1].content}</story_so_far'))
    context.clear()
    context.extend(input + [HumanMessage(content=content)])

    # Call the llm and record the request in the chat-log
    context.append(model.invoke(input=context))
    chat_log['conversation'].extend([
        {'role': 'ME', 'msg': content},
        {'role': 'AI', 'msg': context[-1].content}
    ])
    return context[-1].content

# Helper to extract xml encoded text sections
# This is useful because llms prefer xml for referential data for some reason
def extract_between_tags(tag, text):
    m = re.match(f"<{tag}>((?:.|[\r\n])*)</{tag}>", text)
    return m and m.group(1) or ""

# 
# Simple helper for introducing some ai help with next direction
# Eventually, this'll become a full-fledged GM system
# 
BIBLE_TAGS = 'story_bible'
choice_generation_prompt = load_story_file(config, STORY, 'choices')
def ask_for_ideas(context):
    if not context:
        return "Cannot provide ideas with no context"

    bible = f"<{BIBLE_TAGS}>{extract_between_tags(BIBLE_TAGS, context[-1].content)}</{BIBLE_TAGS}>"
    response = model.invoke(input=[
        SystemMessage(content=choice_generation_prompt),
        HumanMessage(content=bible)
    ])
    return response.content

# 
# Preparing initial story context
# TODO: me - Add error handling when this is abstracted
# 
scene = load_story_file(config, STORY, 'start')
constraints = load_story_file(config, STORY, 'constraints')
initial_story_bible = f'<{BIBLE_TAGS}>{load_story_file(config, STORY, 'lorebook')}</{BIBLE_TAGS}>'


# 
# Start the writing by sending the initial, context-less, direction
# 
context = []
response = send_message(
    f'Plot Direction: {scene}\n{constraints}\n{initial_story_bible}', context
)
print(response)


#
# Keep writing until you want to stop
# 
# At the moment, there are two "commands":
#   - exit, /finish: stop the loop
#   - help, /help: request ai help for generating next actions
# 
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
while True:
    prompt = input("Change> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break

    # TODO: me - this might have issues if send_message doesn't mutate context
    if prompt in ["help", "/help"]:
        print(ask_for_ideas(context))
        continue

    # Need a better way to continue on from the previous location
    scene = f"Plot Direction: \"{prompt}\""
    response = send_message(f'{scene}\n{constraints}', context)
    print(response)



# TODO: me - Generate filename based on conversation to simplify loading
with open(f'./{config['output-dir']}/data.json', 'a', encoding='utf-8') as f:
    json.dump(chat_log, f, ensure_ascii=False, indent=4)
