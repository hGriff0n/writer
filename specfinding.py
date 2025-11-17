
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.prompts import PromptTemplate
from argparse import ArgumentParser, Namespace
import subprocess
from typing import Dict, Set, List, Callable
import regex as re
from io import StringIO

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants, load_markdown

ChatContext = List[AnyMessage]


DEFS = DataConstants()
parser = ArgumentParser(
    prog='specfinding', description='story spec discussion')
parser.add_argument('story')
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-r', '--review_all', action='store_true')
parser.add_argument('-e', '--extract_essay', action='store_true')
parser.add_argument('-m', '--profile',
                    choices=LlmEngine.supported_models(),
                    default=LlmEngine.DEFAULT_MODEL)
# Skip the specfinding conversation and start with styler_gen.md
parser.add_argument('-s', '--start_styling', action='store_true')

# Initialize chat app
args = parser.parse_args()
config = load_config(DEFS)
story = config.load_story(args.story)


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
            args.input,
            '-o',
            file
        ])
        return file
    except Exception as e:
        print(f'[ERROR]: {e}')
        exit()


if args.extract_essay:
    args.input = extract_proto_script(args)
    args.review_all = True


# Assemble the prompt from a generic template
# NOTE: The orchestrator doesn't have any templates **yet**
# TODO: me - Still want to make this use skills, but doesn't seem possible for now
p = PromptTemplate.from_template(
    config.load_prompt_file('specfinding/aspects/orchestrator'))


# Initialize the conversation agent
llm = LlmEngine(config, args.profile, prompt=p.format(), temperature=0.8)


# Define helper methods/types to structure the local state tracking
PROPOSAL_PARSER = re.compile(
    r"<proposal>(.*)</proposal>.+<impact_report>(.*)</impact_report>", re.DOTALL)
SIDEBAR_PARSER = re.compile(
    r"<architect_sidebar>(.*)</architect_sidebar>", re.DOTALL)
EXTRACT_TAG = re.compile(r"<(?P<tag>\w+)>(.*)</(?P=tag)>", re.DOTALL)
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
        for action, content in EXTRACT_TAG.findall(operation.strip()):
            match action:
                case 'add': living_document[tag].add(content)
                case 'remove': living_document[tag].remove(content)

def assemble_snapshot():
    doc = StringIO()
    for key, aspects in living_document:
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


# Start the initial specfinding conversation
# If the user provided a prior spec file that was created via the essay extraction
# agent, then we auto append a "review all" command to the end of the input.
# This will kick off a review procedure for every item in the spec doc when the
# conversation starts up.
input_spec = load_markdown(f'./{args.input}') if args.input else ""
if input_spec:
    # TODO: me - parse the input spec into a structured representation
    if args.review_all:
        input_spec += "\n/review"


# TODO: me - Need to develop a lot of context pruning strategies
# Helper methods for io/context management
def get_author_msg(usage: Dict) -> str:
    tokens = usage and usage.get('total_tokens', 0) or 0
    return input(f'(tkns: {tokens})> ').strip()

def display_ai_response(resp: str):
    print(f'AI: {resp}')

def get_clean_context(llm) -> ChatContext:
    return [SystemMessage(llm.prompt)]

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


def send_message(llm: LlmEngine, msg: str, messages: ChatContext) -> Dict:
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


print('starting repl')
try:
    messages = get_clean_context(llm)
    usage = {'total_tokens': 0}
    if input_spec:
        usage = send_message(llm, input_spec, messages)

    while True:
        msg = get_author_msg(usage)
        usage = REPL.get(msg, send_message)(llm, msg, messages)
        break
except Exception as e:
    print(f'[ERROR]: {e}')

llm.chat_log.save(config.output_dir)
