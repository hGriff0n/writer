
from langchain_core.messages import SystemMessage
from argparse import ArgumentParser
import subprocess

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants, load_markdown
from langchain_core.prompts import PromptTemplate


DEFS = DataConstants()
parser = ArgumentParser(prog='specfinding', description='story spec discussion')
parser.add_argument('story')
parser.add_argument('-s', '--input_spec', type=str)
parser.add_argument('-r', '--review_all', action='store_true')
parser.add_argument('-e', '--extract_essay', action='store_true')
parser.add_argument('-m', '--profile',
                    choices=LlmEngine.supported_models(),
                    default=LlmEngine.DEFAULT_MODEL)

# Initialize chat app
args = parser.parse_args()
config = load_config(DEFS)
story = config.load_story(args.story)


# Launch script for extracting a proto specsheet from a narrative essay
# before doing anything else, if that is requested by the arguments.
# This just forwards to a separate script which runs the conversation automatically
# This also auto-sets the "review_all" flag to True
def extract_proto_script(args) -> str:
    file = f'./.tmp/{args.story}.md'
    try:
        subprocess.run([
            'python',
            './scripts/spec_from_essay.py',
            args.story,
            '-f',
            args.input_spec,
            '-o',
            file
        ])
        return file
    except Exception as e:
        print(f'[ERROR]: {e}')
        exit()

if args.extract_essay:
    args.input_spec = extract_proto_script(args)
    args.review_all = True


# Assemble the prompt from a generic template
# NOTE: The orchestrator doesn't have any templates **yet**
# TODO: me - Still want to make this use skills, but doesn't seem possible for now
p = PromptTemplate.from_template(config.load_prompt_file('specfinding/aspects/orchestrator'))


# Initialize the conversation agent
llm = LlmEngine(config, args.profile, prompt=p.format(), temperature=0.8)


# Start the initial specfinding conversation
# If the user provided a prior spec file that was created via the essay extraction
# agent, then we auto append a "review all" command to the end of the input.
# This will kick off a review procedure for every item in the spec doc when the
# conversation starts up.
input_spec = load_markdown(f'./{args.input_spec}') if args.input_spec else ""
if input_spec and args.review_all:
    input_spec += "\n/review"


# TODO: me - Need to develop a lot of context pruning strategies
# Helper methods for io/context management
def get_author_response(usage):
    tokens = usage and usage.get('total_tokens', 0) or 0
    return input(f'(tkns: {tokens})> ').strip()

def display_ai_response(resp):
    print(f'AI: {resp}')


# Main specfinding loop
# TODO: me - Need to add in internal tracking of living state doc
# I think I can get away with just tracking the updates and stiching together later
def run_repl_loop(llm):
    messages = [SystemMessage(llm.prompt)]
    if input_spec:
        resp, usage = llm.invoke(input_spec, messages)
        display_ai_response(resp)

    while True:
        msg = input('> ').strip()
        if msg == 'exit':
            return
        resp, usage = llm.invoke(msg, messages)
        display_ai_response(resp)

        if usage['total_tokens'] >= 30000:
            print('!!! [ALERT] You are apporaching Memory Danger Zone [ALERT] !!!')

try:
    run_repl_loop(llm)
except:
    llm.chat_log.save(config.output_dir)
