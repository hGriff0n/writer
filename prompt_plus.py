import json
import os
import yaml

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

with open('./data/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# MODEL_CHOICE = 'openai'
MODEL_CHOICE = 'gemini'
llm_config = config['ai-providers'][MODEL_CHOICE]
os.environ[llm_config['api-key']['name']] = llm_config['api-key']['value']

model = init_chat_model(llm_config['name'], model_provider=llm_config.get('provider'))

# Helpers for loading data from prompt and story files
def load_prompt(config, prompt: str) -> str:
    with open(f'./{config['prompt-dir']}/{prompt}.md', 'r') as f:
        return f.read()

PROMPT_ANALYSIS_PROMPT = load_prompt(config, 'feedback_analysis')

# Load the chat file from the last conversation
# We only keep the human messages because the feedback we give to the model
# should already encode enough state about what needs to be improved.
# In my experience, adding the full AI output sometimes causes the AI to
# focus more on identifying with what went wrong in your specific case, then
# improving on the general prompt
with open(f'./{config['output-dir']}/data.json', 'r', encoding='utf-8') as f:
    chat_log = json.load(f)
    template = chat_log['template']
    conversation = '\n'.join(
        map(lambda msg: f'{msg['role']}: {msg['msg']}',
            filter(lambda m: m['role'] == 'ME', chat_log['conversation'])))


# Send the whole data to the ai model and print the response
# TODO: me - this should probably be another chat app
response = model.invoke(input=[
    SystemMessage(content=PROMPT_ANALYSIS_PROMPT),
    HumanMessage(content=template),
    HumanMessage(content=conversation),
])
print(response.content)
