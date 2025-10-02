
import json

from lib.ai import init_model
from lib.config import load_config

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config()
model = init_model(config, 'gemini')


REWRITER_PROMPT = config.load_prompt_file('expander')
with open(f'{config.output_dir}/stories.json', 'r', encoding='utf-8') as f:
    story = json.load(f)['1']

title = 'reality'
text_to_analyze = f'<prose>{story[0]}</prose>'


prompt = input("How to expand the scene> ")
response = model.invoke(input=[
    SystemMessage(content=REWRITER_PROMPT),
    SystemMessage(content=text_to_analyze),
    HumanMessage(content=prompt)
])

print(response.content)