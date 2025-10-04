
from lib.ai import LlmEngine, load_chat_log
from lib.config import load_config, DataConstants

from langchain_core.messages import HumanMessage, SystemMessage


# Initialize chat app
config = load_config(DataConstants())
model = LlmEngine(config, 'gemini', 'feedback_analysis')

# Load the chat file from the last conversation
# We only keep the human messages because the feedback we give to the model
# should already encode enough state about what needs to be improved.
# In my experience, adding the full AI output sometimes causes the AI to
# focus more on identifying with what went wrong in your specific case, then
# improving on the general prompt
chat_log = load_chat_log(config.output_dir, 'data')
conversation = '\n'.join(
    map(lambda msg: f'ME: {msg}', chat_log.having_role('ME')))


# Send the whole data to the ai model and print the response
# TODO: me - this should probably be another chat app
response = model.invoke(f'{chat_log.template}\n\n{conversation}', [])
print(response.content)
print(f'Cost of Run: {model.est_cost()}')
