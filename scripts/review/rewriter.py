
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

import json

from lib.ai import LlmEngine
from lib.config import Config

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = Config()
model = LlmEngine(config, 'gemini', 'edit/expander')

with open(f'{config.output_dir}/stories.json', 'r', encoding='utf-8') as f:
    story = json.load(f)['1']

title = 'reality'
text_to_analyze = f'<prose>{story[0]}</prose>'

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
