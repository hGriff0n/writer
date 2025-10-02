
import json

from lib.ai import init_model
from lib.config import load_config

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config()
model = init_model(config, 'gemini')


REWRITER_PROMPT = config.load_prompt_file('rewriter')
with open(f'{config.output_dir}/stories.json', 'r', encoding='utf-8') as f:
    story = json.load(f)['1']

title = 'reality'
text_to_analyze = f'<SourceText>{story[0]}\n\n{story[1]}</SourceText>'

# not sure how to take things from here
# the rewriter approach isn't working for now
# 
narrative_context = """
<MetaRules>
</MetaRules>
"""


prompt = input("> ")
response = model.invoke(input=[
    SystemMessage(content=REWRITER_PROMPT),
    SystemMessage(content=text_to_analyze),
    SystemMessage(content=narrative_context),
    HumanMessage(content=prompt)
])

print(response.content)