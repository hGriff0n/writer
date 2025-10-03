
import json

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config(DataConstants())
model = LlmEngine(config, 'gemini', 'expander')

with open(f'{config.output_dir}/stories.json', 'r', encoding='utf-8') as f:
    story = json.load(f)['1']

title = 'reality'
text_to_analyze = f'<prose>{story[0]}</prose>'

# Took a long time to re-establish the system "rules" and even then they weren't fully understood
# The transitions between the old and new scenes weren't great, the intention for these was to blend smoothly
# Didn't change the existing story at all, but that might've been necessary in order to transition better
# Maybe could be a ploy for a transition engine or have the expander ensure it uses the space to transition nicely
# Maybe worthwhile to have the initial writer explicitly annotate thinking and system events for later stages? Maybe even have that as the first stage?
context = [
    SystemMessage(content=model.prompt),
    SystemMessage(content=text_to_analyze),
]
prompt = input("How to expand the scene> ")
while prompt != "exit":
    print(model.invoke(prompt, context))
    prompt = input("> ")

model.chat_log.save(config.output_dir)
print(f'Cost of Run: {model.est_cost()}')
