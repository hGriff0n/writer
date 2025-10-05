
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config(DataConstants())
# model = LlmEngine(config, 'gemini', 'principles/context_generator')
model = LlmEngine(config, 'gemini', 'experiments/story_definer')

with open(f'{config.output_dir}/book.txt', 'r', encoding='utf-8') as f:
    story = f.read()

title = 'reality'
principles = config.load_story_file(title, 'principles')
WRITER = config.load_prompt_file('simple_writer')
# model.prompt.replace('{PRINCIPLES}', principles)
# model.prompt.replace('{STORY}', story)

context = [
    SystemMessage(content=model.prompt),
    # SystemMessage(content=f'PRINCIPLES:\n{principles}')
    SystemMessage(content=WRITER)
]
# print(model.invoke(f'WRITER PROMPT:\n{WRITER}', context))
print(model.invoke(principles, context))
print(f'Cost of Run: {model.est_cost()}')
