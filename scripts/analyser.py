
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config(DataConstants())
model = LlmEngine(config, 'gemini', 'principles/reviewer')

with open(f'{config.output_dir}/book.txt', 'r', encoding='utf-8') as f:
    story = f.read()

title = 'reality'
principles = config.load_story_file(title, 'principles')
# model.prompt.replace('{PRINCIPLES}', principles)
# model.prompt.replace('{STORY}', story)

context = [
    SystemMessage(content=model.prompt),
    HumanMessage(content=f'PRINCIPLES:\n{principles}')
]
print(model.invoke(f'STORY:\n{story}', context))
print(f'Cost of Run: {model.est_cost()}')
