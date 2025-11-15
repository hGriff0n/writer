
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from argparse import ArgumentParser
import json

from lib.ai import LlmEngine, init_model
from lib.config import load_config, DataConstants, load_markdown
from langchain_core.prompts import PromptTemplate

# For state tracking only
from langchain_core.load.dump import dumpd
import json


DEFS = DataConstants()

# TODO: me - Rearchitect these arguments so that they are fully customizable
parser = ArgumentParser(prog='specfinding', description='story spec discussion')
parser.add_argument('story')
parser.add_argument('-m', '--profile',
                    choices=LlmEngine.supported_models(),
                    default=LlmEngine.DEFAULT_MODEL)
parser.add_argument('-p', '--prompt',
                    choices=['simple_writer'], default='simple_writer')
parser.add_argument('-c', '--resume', action='store_true')
parser.add_argument('-r', '--runs', type=int, default=0)


# Initialize chat app
args = parser.parse_args()
config = load_config(DEFS)
story = config.load_story(args.story)

# Initialize the model
# TODO: me - Update this to the paid tier limits
# flash_limiter = InMemoryRateLimiter(
#     requests_per_second=10 / 60,  # 10 RPM for 2.5flash
#     check_every_n_seconds=0.1,    # Wake up every 100 ms
#     max_bucket_size=5,
# )
llm = LlmEngine(config, args.profile, prompt=story.writer, temperature=0.8)


initial = PromptTemplate.from_template(config.load_prompt_file('specfinding/aspects/spec_extraction')).format(document=load_markdown(story.fullspec))

question = """
Analyze the source document and compare it against your current list of Core Concepts.
Are there any Core Concepts in the original document that are not captured in your current list?

If yes, add those concepts to the list and repeat this process until you can't add any more concepts.
Otherwise, if there are no new concepts that can be added, you MUST output a single word: 'DONE'
"""

messages = [SystemMessage(initial)]
for i in range(0, 5):
    result, usage = llm.invoke(question, messages)
    print(result)
    print('='*5)
    if result == 'DONE':
        break

# result = SystemMessage('')
# if story.fullspec:
#     result, usage = llm.invoke(load_markdown(f'{story._path}/{story.fullspec}'), messages)
# else:
#     result, usage = llm.invoke(input("> ").strip(), messages)
# messages[0] = SystemMessage(base_prompt)
# messages.append(AIMessage(result))
# print(f'AI({usage['total_tokens']}): {result}')

# # TODO: me - Add context pruning functions
# def prune_context(llm, messages):
#     messages.append(HumanMessage('Create snapshot'))
#     with open('tmp.md', 'w') as f:
#         f.write(llm.invoke(messages).content)
#     exit()
#     return messages

# # Use agent
# while True:
#     msg = input(f"{usage['total_tokens']}> ").strip()
#     if msg == 'exit':
#         prune_context(llm, messages)
#     result, usage = llm.invoke(msg, messages)

#     print(f'AI: {result}')
#     messages.append(result)
#     if usage['total_tokens'] >= 30000:
#         messages = prune_context(llm, messages)
