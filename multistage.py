
from argparse import ArgumentParser
import json
import re
from typing import Dict, List, Optional, Tuple
import yaml

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, AIMessage, SystemMessage


DEFS = DataConstants()

# TODO: me - Rearchitect these arguments so that they are fully customizable
parser = ArgumentParser(prog='simple_writer', description='simple ai writer')
parser.add_argument('story')
parser.add_argument('-m', '--profile',
                    choices=LlmEngine.supported_models(),
                    default=LlmEngine.DEFAULT_MODEL)
parser.add_argument('-p', '--prompt',
                    choices=['simple_writer'], default='simple_writer')
parser.add_argument('-c', '--resume', action='store_true')


# Initialize chat app
args = parser.parse_args()
config = load_config(DEFS)
story = config.load_story(args.story)

#
# The current `LlmEngine` approach "assumes" one prompt per llm
# Which has downsides (such as this) and upsides (cleaner calling, mostly)
# Is it better to have one engine per prompt, or reuse the same engine for
# multiple prompts? The former is required when using different models
#

# The architect is a multi-purpose agent dealing with all things about plot
# direction. There are basically 3 modes: planning, options, and
# translation. Planning pre-creates a sequence of `7` chapters whenever no
# input is provided. If the user inputs a change of their own, the plan is
# dropped and the "translation" mode is used to produce an output that the
# writer can use for generation. The user can also query the ai for some
# options, which will provide `3` different ways for progressing the story
architect = LlmEngine(config, args.profile,
                      prompt_file='architect', temperature=0.4)

# The writer is solely responsible for taking the plot beat provided by the
# architect and expand it into an actual chapter of prose that extends the
# story it is currently writing
writer = LlmEngine(config, args.profile, prompt=story.writer, temperature=1.7)


#
# Wrapper for sending a message to the llm
#
def send_message(llm: LlmEngine, message: str,
                     context: List[AnyMessage],
                     summary: Optional[str] = None) -> str:
    if context is None:
        raise Exception("Context must be specified")
    addl_context = [] if not summary else [summary]
    response = llm.invoke(message, context + addl_context)
    if response.startswith('```'):
        response = response[7:]
    if response.endswith('```'):
        response = response[:-3]
    return response


# TODO: me - Figure out a way to control pacing generically
EXTRA = "\npacing_modifier: 1" if story.title == 'reality' else ''

# Helper method for splitting the next input from the existing plan.
# If there are no planned inputs currently, this requests a new set
REQUEST_PLAN = f"""```yaml
directive:
    mode: Sequential{EXTRA}
    count: 7
```"""
PLOT_PLAN = []


def get_next_input_from_plan(context: List[AnyMessage],
                                 summary: Optional[str]):
    global PLOT_PLAN
    if not PLOT_PLAN:
        plan = send_message(architect, REQUEST_PLAN, context, summary)
        PLOT_PLAN = [yaml.safe_dump(o) for o in next(yaml.safe_load_all(plan))]

    prompt, PLOT_PLAN = PLOT_PLAN[0], PLOT_PLAN[1:]
    return prompt


# Helper method for requesting potential next options from the planner
CHOICE_PROMPT = f"""```yaml
directive:
    mode: Options{EXTRA}
    count: 3
```"""


def summarize_plot_beats(beats: List[str]) -> List[str]:
    return [o['beat_summary' if story.title == 'curse' else 'title'] for o in options]


def ask_for_ideas(context: List[AnyMessage],
                      summary: Optional[str] = None,
                      request: Optional[str] = None
                      ) -> Tuple[List[str], List[str]]:
    addl_focus = '' if request is None else f'\n\tfocus:{request}'
    output = send_message(
        architect, CHOICE_PROMPT + addl_focus, context, summary)
    options = next(yaml.safe_load_all(output))
    return summarize_plot_beats(options), [yaml.safe_dump(o) for o in options]


# Prepare context from assembled data
arch_context = [
    SystemMessage(content=story.narrative_intent),
    SystemMessage(content=story.core_concepts),
    SystemMessage(content=story.engines),
    SystemMessage(content=story.rules),
    SystemMessage(content=story.schemas),
    SystemMessage(content=story.beat_assembly)
]
writer_context: List[AnyMessage] = []


# TODO: me - would this need to parsed into yaml?
# Allow for resuming an in-progress story
if args.resume:
    print("Loading in-progress story...")
    file = f'{config.directories.story}/{story.title}/principles/tmp.json'
    with open(file, 'r', encoding='utf-8') as f:
        story = json.load(f)

    print("Restoring prior context...")
    writer.chat_log.conversation.extend(
        {"role": "AI", "msg": chap} for chap in story['chapters'])
    summary = story['chapters'][-1]

    print(f"Restoring current plan...")
    PLOT_PLAN = story.get('plan', [])

    print(f"Loaded previous story from {file}")
    print(summary)

# Otherwise we're starting a new story, so simply load up the default
# start command and start writing automatically.
else:
    plan = send_message(architect, story.first_turn, arch_context)
    summary = send_message(writer,
                           f'```yaml\n{plan}```', writer_context)
    print(summary)


#
# Keep writing until you want to stop
#
# At the moment, there are two "commands":
#   - whereami, /context: print prior chapter
#   - exit, /finish: stop the loop
#   - help, /help: request ai help for generating next actions
#       TODO: me - Incorporate the new `request` arg
#
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
while True:
    prompt = input("Change> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break
    if prompt in ["whereami", "/context"]:
        print(summary)
        continue
    if prompt in ["help", "/help"]:
        PLOT_PLAN.clear()
        display, options = ask_for_ideas(arch_context, summary=summary)
        print(f'A: {display[0]}')
        print(f'B: {display[1]}')
        print(f'C: {display[2]}')
        choice = input("Select Option (A/B/C)>").lower()
        prompt = {'a': options[0], 'b': options[1], 'c': options[2]}[choice]
    if prompt in ["plan", "/plan"]:
        print('- ' + '\n- '.join(summarize_plot_beats(PLOT_PLAN)))
        continue

    # Allow for planning of plot events
    if not prompt:
        prompt = get_next_input_from_plan(arch_context, summary=summary)
    else:
        PLOT_PLAN.clear()
    summary = send_message(writer, prompt, writer_context, summary)
    print(summary)


# Store the conversation in a per-run file so we can easily send it to
# other prompts for improvements/etc.
print(f'Cost of Run: {writer.est_cost() + architect.est_cost()}')
chat_file = writer.chat_log.save(config.output_dir)
arch_file = architect.chat_log.save(config.output_dir)


# TODO: me - What does this do that's not already in the chat log?
# Aside from saving in the same location as the story files ???
# Save the current state of generation in a temp file in the story directory
# This is to enable continuations through the --resume flag
book = [response for response in writer.chat_log.having_role('AI')]
with open(f'./{config.directories.story}/{story.title}/principles/tmp.json', 'w', encoding='utf-8') as f:
    json.dump({'chapters': book, 'writer_file': chat_file, 'architect': arch_file,
              'plan': PLOT_PLAN}, f, ensure_ascii=False, indent=4)
