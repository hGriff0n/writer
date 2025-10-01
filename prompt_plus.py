
from lib.ai import init_model, load_chat_log
from lib.config import load_config

from langchain_core.messages import HumanMessage, SystemMessage


# Initialize chat app
config = load_config()
model = init_model(config, 'gemini')

PROMPT_ANALYSIS_PROMPT = config.load_prompt_file('feedback_analysis')

# Load the chat file from the last conversation
# We only keep the human messages because the feedback we give to the model
# should already encode enough state about what needs to be improved.
# In my experience, adding the full AI output sometimes causes the AI to
# focus more on identifying with what went wrong in your specific case, then
# improving on the general prompt
chat_log = load_chat_log(config.output_dir)
conversation = '\n'.join(
    map(lambda msg: f'ME: {msg}', chat_log.having_role('ME')))


# Send the whole data to the ai model and print the response
# TODO: me - this should probably be another chat app
response = model.invoke(input=[
    SystemMessage(content=PROMPT_ANALYSIS_PROMPT),
    HumanMessage(content=chat_log.template),
    HumanMessage(content=conversation),
])
print(response.content)
