
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config(DataConstants())
model = LlmEngine(config, 'gemini', 'principles/reviewer')
model.prompt = """
You are an expert editor. Your task is to rewrite the following story based on the provided critique.

Your rewrite should:
1.  Address the shortcomings and fix the problems identified in the critique.
2.  Lean into and amplify the positive elements mentioned in the critique.
3.  Maintain the original story's tone, voice, and style as closely as possible."""

with open(f'{config.output_dir}/book.txt', 'r', encoding='utf-8') as f:
    story = f.read()

title = 'reality'
principles = config.load_story_file(title, 'principles')
# model.prompt.replace('{PRINCIPLES}', principles)
# model.prompt.replace('{STORY}', story)

critique=input("Critique>")
context = [
    SystemMessage(content=model.prompt),
    SystemMessage(content=f'ORIGINAL STORY:\n{story}')
]
print(model.invoke(f'CRITIQUE:\n{critique}', context))
print(f'Cost of Run: {model.est_cost()}')

# Initial Thoughts:
# Pretty decent, but greatly shortened the individual chapters (because I passed in the full story). Not sure how to apply a more focused pass
# The critique agent could also use some refinement