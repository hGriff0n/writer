import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from langchain_core.messages import SystemMessage, AnyMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from argparse import ArgumentParser, Namespace
import subprocess
from typing import Dict, Set, List, Callable, Optional
import regex as re
from io import StringIO

from lib.ai import LlmEngine
from lib.config import Config
from lib.context import ContextManager
from lib.util import load_markdown

ChatContext = List[AnyMessage]


CONF = Config()
parser = ArgumentParser(
    prog='specfinding', description='story spec discussion')
parser.add_argument('story')
parser.add_argument('-i', '--input', type=str, action='append')
parser.add_argument('-r', '--review_all', action='store_true')
parser.add_argument('-e', '--extract_essay', action='store_true')
parser.add_argument('-m', '--profile',
                    choices=CONF.supported_models,
                    default=LlmEngine.DEFAULT_MODEL)
# Skip the specfinding conversation and start with styler_gen.md
parser.add_argument('-s', '--scene_scripting', action='store_true')
parser.add_argument('-w', '--writer', action='store_true')
parser.add_argument('-o', '--orchestration', action='store_false', default=True)

# Initialize chat app
args = parser.parse_args()
story = CONF.load_story(args.story)


# Launch script for extracting a proto specsheet from a narrative essay
# before doing anything else, if that is requested by the arguments.
# This just forwards to a separate script which runs the conversation automatically
# This also auto-sets the "review_all" flag to True
def extract_proto_script(args: Namespace) -> str:
    file = f'./.tmp/{args.story}.md'
    try:
        subprocess.run([
            'python',
            './scripts/spec_from_essay.py',
            args.story,
            '-f',
            args.input[0],
            '-o',
            file
        ])
        return file
    except Exception as e:
        print(f'[ERROR]: {e}')
        exit()


if args.extract_essay:
    args.input[0] = extract_proto_script(args)
    args.review_all = True


# TODO: generalize this with the style_gen
# Define helper methods/types to structure the local state tracking
PROPOSAL_PARSER = re.compile(
    r"<proposal>(.*)</proposal>.+<impact_report>(.*)</impact_report>", re.DOTALL)
SIDEBAR_PARSER = re.compile(
    r"<architect_sidebar>(.*)</architect_sidebar>", re.DOTALL)
EXTRACT_TAG = re.compile(r"<(?P<tag>\w+)>(.*)</(?P=tag)>", re.DOTALL)
EXTRACT_TAG_SMALL = re.compile(r"<(?P<tag>\w+)>(.*?)</(?P=tag)>", re.DOTALL)
living_document = {
    'core_concept': set(),
    'narrative_engine': set(),
    'narrative_rule': set(),
    'beat_generation_system': set(),
    'world_codex': set(),
    'sketchpad': set(),
    'workshop': set(),
    'codex': set()
}

# TODO: me - This is a very simple differ relying on llm reporting whole aspect
def add_updates_to_living_doc(resp: str):
    m = SIDEBAR_PARSER.search(resp)
    if not m:
        return
    for tag, operation in EXTRACT_TAG.findall(m.group(1).strip()):
        for action, content in EXTRACT_TAG_SMALL.findall(operation.strip()):
            match action:
                case 'add': living_document[tag].add(content)
                case 'remove': living_document[tag].remove(content)

def assemble_snapshot() -> str:
    doc = StringIO()
    for key, aspects in living_document.items():
        doc.write(f'### {' '.join(k.capitalize() for k in key.split('_'))}')
        for component in aspects:
            doc.write('\n\n')
            doc.write(component)
        doc.write('\n\n')
    return doc.getvalue()



# TODO: me - Turn this into a class for parsing the llm response
# This will be useful for filtering what is displayed
class ParsedResponse(object):

    def __init__(self, resp: str):
        self._sidebar = {}
        self._parse_to_doc(resp)

    def _parse_to_doc(self, resp: str):
        m = SIDEBAR_PARSER.search(resp)
        if not m:
            return
        for tag, operation in EXTRACT_TAG.findall(m.group(1).strip()):
            if tag not in self._sidebar:
                self._sidebar[tag] = set()
            for action, content in EXTRACT_TAG.findall(operation.strip()):
                self._sidebar[tag].add(f'{action}::{content}')
    
    @property
    def sidebar(self) -> Dict[str, Set[str]]:
        return self._sidebar


# TODO: me - Need to develop a lot of context pruning strategies
# Helper methods for io/context management
def get_author_msg(usage: Dict) -> str:
    tokens = usage and usage.get('total_tokens', 0) or 0
    return input(f'(tkns: {tokens})> ').strip()

def display_ai_response(resp: str):
    print(f'AI: {resp}')

def get_clean_context(llm) -> ChatContext:
    m = [SystemMessage(llm.prompt)]
    return m

def flush_and_restart(llm: LlmEngine, _: str, messages: ChatContext) -> Dict:
    snapshot = assemble_snapshot()
    messages.clear()
    messages.extend(get_clean_context(llm))
    resp, usage = llm.invoke(snapshot, messages)
    display_ai_response(resp)
    return usage


# Main specfinding loop
# TODO: me - Need to add in internal tracking of living state doc
# I think I can get away with just tracking the updates and stiching together later
REPL: Dict[str, Callable[[LlmEngine, str, ChatContext], Dict]] = {
    'exit': lambda _x, _y, _z: exit(),
    '/reset': flush_and_restart,
}


def send_message_orc(llm: LlmEngine, msg: str, messages: ChatContext) -> Dict:
    resp, usage = llm.invoke(msg, messages)
    # TODO: me - parse resp to extract reported updates
    # This uses a lookahead to split the string on the tag while
    # keeping the tag in the "right" hand string. This reduces what
    # we have to display to the end user without losing usability
    parts = re.split("(?=<architect_sidebar>)", resp, maxsplit=1)
    display_ai_response(parts[0])
    if len(parts) > 1:
        add_updates_to_living_doc(parts[1])

    if usage['total_tokens'] >= 30000:
        print('!!! [ALERT] You are apporaching Memory Danger Zone [ALERT] !!!')
    return usage

def load_input_spec() -> str:
    return '\n\n'.join(
        load_markdown(f'./{input_file}') if input_file else "" for input_file in (args.input or [])
    )


input_spec = ""

# TODO: me - Port over the work from the context manager to here
# Run the orchestration prompt sub-loop
if args.orchestration:
    # If the user provided a prior spec file that was created via the essay extraction
    # agent, then we auto append a "review all" command to the end of the input.
    # This will kick off a review procedure for every item in the spec doc when the
    # conversation starts up.
    input_spec = load_input_spec()
    if input_spec:
        # TODO: me - parse the input spec into a structured representation
        if args.review_all:
            input_spec += "\n/review"

    # Assemble the prompt from a generic template
    # NOTE: The orchestrator doesn't have any templates **yet**
    # TODO: me - Still want to make this use skills, but doesn't seem possible for now
    p = PromptTemplate.from_template(
        CONF.load_prompt('specfinding/orchestrator'))

    # Initialize the conversation agent
    llm = LlmEngine(CONF, args.profile, prompt=p.format())

    print('starting orchestration repl')
    try:
        messages = get_clean_context(llm)
        usage = {'total_tokens': 0}
        if input_spec:
            usage = send_message_orc(llm, input_spec, messages)

        while True:
            msg = get_author_msg(usage)
            usage = REPL.get(msg, send_message_orc)(llm, msg, messages)
            # TODO: me - runs one loop
            break
    except Exception as e:
        print(f'[ERROR]: {e}')

    llm.chat_log.save(CONF.output_dir)
    input_spec += assemble_snapshot()


# TODO: me - ContextManager isn't fully utilized so this is a bit messy
def send_message_scen(llm: LlmEngine, msg: str, c: ContextManager) -> Dict:
    m = get_clean_context(llm).copy()
    m.extend(AIMessage(ms) for ms in c.context)
    resp, usage = llm.invoke(msg, m)
    c.add_to_context(msg)
    c.add_to_context(resp)
    display_ai_response(resp)
    return usage

def flush_and_restart_scen(llm: LlmEngine, _: str, c: ContextManager) -> Dict:
    snapshot = c.assemble_snapshot()
    c.clear()
    usage = send_message_scen(llm, snapshot, c)
    return usage

REPL_SCEN: Dict[str, Callable[[LlmEngine, str, ContextManager], Dict]] = {
    'exit': lambda _x, _y, _c: exit(),
    '/reset': flush_and_restart_scen,
}

def run_repl_loop(config: Config, args: Namespace, input_spec: str, prompt_file: str) -> str:
    # If the input spec isn't set, assume it's the input file
    if not input_spec:
        input_spec = load_input_spec()

    # TODO: Convert the config class to returning filepath
    p = PromptTemplate.from_template(config.load_prompt(prompt_file))

    # Initialize the conversation agent
    llm = LlmEngine(config, args.profile, prompt=p.format())

    # Prepare the living document and bookmark handler
    c = ContextManager()
    
    try:
        # Start by sending a hello message with the input context
        # TODO: me - this doesn't work if no input provided (ie. `load_input_spec`) is empty
        usage = {'total_tokens': 0}
        if input_spec:
            usage = send_message_scen(llm, input_spec, c)

        while True:
            msg = get_author_msg(usage)
            usage = REPL_SCEN.get(msg, send_message_scen)(llm, msg, c)
    except Exception as e:
        print(f'[ERROR]: {e}')

    llm.chat_log.save(config.output_dir)
    return input_spec + c.assemble_snapshot()


if args.scene_scripting:
    # First: https://aistudio.google.com/app/prompts/1rEbum4Q106PAQOufW9LGb0r0eJZq4KzP
    # Later: https://aistudio.google.com/app/prompts/1W6ztSXtfxrPdUQn5S-tWOSiiX6fEhx1w
    print('starting scene assembly repl')
    input_spec = run_repl_loop(CONF, args, input_spec, prompt_file='specfinding/scenegen')

if args.writer:
    print('starting writer styling repl')
    input_spec = run_repl_loop(CONF, args, input_spec, prompt_file='specfinding/writerstyle')
