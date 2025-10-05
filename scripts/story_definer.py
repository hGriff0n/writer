
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from lib.ai import init_model, LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

config = load_config(DataConstants())
model = LlmEngine(config, 'gemini', 'experiments/story_definer')

context = [
    SystemMessage(content=model.prompt),
]
prompt = input("Input story idea> ")
while prompt != "exit":
    print(model.invoke(prompt, context))
    prompt = input("> ")

model.chat_log.save(config.output_dir)
print(f'Cost of Run: {model.est_cost()}')
