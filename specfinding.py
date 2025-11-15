
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from argparse import ArgumentParser
import regex as re

from lib.ai import LlmEngine, init_model
from lib.config import load_config, DataConstants
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
schema = config.load_schema('specfinding/essay/spec')

# Initialize the model
# TODO: me - Update this to the paid tier limits
# flash_limiter = InMemoryRateLimiter(
#     requests_per_second=10 / 60,  # 10 RPM for 2.5flash
#     check_every_n_seconds=0.1,    # Wake up every 100 ms
#     max_bucket_size=5,
# )
llm = LlmEngine(config, args.profile, prompt=story.writer, temperature=0.8)

# Update with assembled doc after every "step"
system_prompt = PromptTemplate.from_template(config.load_prompt_file('specfinding/essay/ingest'), partial_variables={'document': story.fullspec})
initial = system_prompt.format(living_spec="")

first = "<living_spec/> is a markdown document formatted according to the `Output Format` containing the \"living document\" for the story. Your task is:\n\n"
question = PromptTemplate.from_template(r"""
{header}Analyze <source_essay/> and compare it against the "living document" to identify if there are any {component} specified or implied in the essay that are not currently in the "living document".

If you identify a missing component, you must add those components to the list. Repeat this process until you can't identify any new components and then output the detailed list of added components and then stop.

If you are unable to find any new components, you must return immediately and only output the following string: [[STOP]].
""")

components = [
    'World Codex',
    'Core Concepts',
    'Narrative Engines',
    # 'Beat Generation Rules'
]
living_spec = ""
extract_md = re.compile(r"###.+\n((?s).+)")

for c in components:
    added_content = []
    messages = [SystemMessage(system_prompt.format(living_spec=living_spec))]
    resp, usage = llm.invoke(question.format(header=first, component=c), messages)
    for i in range(0, 5):
        # Handle the break
        if resp == "[[STOP]]": break
        # Don't report engine fluff
        m = re.search(extract_md, resp)
        if not m:
            print(m)
            print(resp)
            exit()
        added_content.append(m.group(1))
        # And repeat until done
        print(f'Making followup query for `{c}: {i} out of 5')
        resp, usage = llm.invoke(question.format(header='', component=c), messages)
    
    # We've done all we can so add the established information to the doc
    print(f"Finishing for {c}")
    living_spec += f'### {c}\n{'\n'.join(added_content)}\n\n'
    
with open('tmp.md', 'w+', encoding='utf-8') as f:
    f.write(living_spec)

# TODO: Move into manual review

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
#     with open('tmp.md', 'w', encoding='utf-8') as f:
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
