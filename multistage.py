
from argparse import ArgumentParser
import json
import re
from typing import Dict, List, Tuple

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
architect = LlmEngine(config, args.profile, args.prompt, temperature=0.4)
architect.prompt = config.load_story_file(args.story, 'principles/architect')

# The writer is solely responsible for taking the plot beat provided by the
# architect and expand it into an actual chapter of prose that extends the
# story it is currently writing
writer = LlmEngine(config, args.profile, args.prompt, temperature=1.7)
writer.prompt = config.load_story_file(args.story, 'principles/scene_write')


#
# Wrapper for sending a message to the llm
#
def send_message(llm: LlmEngine, message: str,
                 context: List[AnyMessage]) -> str:
    if context is None:
        raise Exception("Context must be specified")

    # Assemble prompt and context
    # TODO: me - Eventually use more memory channels
    input = [SystemMessage(content=llm.prompt)]
    if context:
        input.append(SystemMessage(
            f'NARRATIVE CONTEXT: {context[-1].content}'))
    context.clear()
    context.extend(input)

    # Call the llm and record the request in the chat-log
    return llm.invoke(message, context)


# `PACE` is a meta-control for the reality story that adjusts how quickly
# the story progresses
EXTRA = "\nPACING MODIFIER: 1" if args.story == 'reality' else ''
PLOT_PLAN = []

# Helper method for splitting the next input from the existing plan.
# If there are no planned inputs currently, this requests a new set
def get_next_input_from_plan():
    global PLOT_PLAN
    if not PLOT_PLAN:    
        REQUEST_PLAN = f"GENERATION MODE: Sequential{EXTRA}\nNUMBER OF BEATS: 7"
        plan = send_message(architect, REQUEST_PLAN, write_context)
        if args.story == 'reality':
            PLOT_PLAN = re.split("\\*\\*\\*", plan)[1:]
        else:
            PLOT_PLAN = [yaml.safe_dump(o) for o in next(yaml.safe_load_all(plan[8:-4]))]
    
    prompt, PLOT_PLAN = PLOT_PLAN[0], PLOT_PLAN[1:]
    return prompt


# Otherwise we're starting a new story, so simply load up the default
# start command and start writing automatically.
if args.resume:
    print("Loading in-progress story...")
    file = f'{config.directories.story}/{args.story}/principles/tmp.json'
    with open(file, 'r', encoding='utf-8') as f:
        story = json.load(f)
    
    print("Restoring prior context...")
    writer.chat_log.conversation.extend({"role": "AI", "msg": chap} for chap in story['chapters'])
    write_context = [AIMessage(content=story['chapters'][-1])]

    print(f"Restoring current plan...")
    PLOT_PLAN = story.get('plan', [])

    print(f"Loaded previous story from {file}")
    print(write_context[0].content)

else:
    START = config.load_story_file(args.story, 'start')
    arch_context: List[AnyMessage] = []
    plan = send_message(architect, START, arch_context)

    write_context: List[AnyMessage] = []
    text = send_message(writer,
                        f'<beat_data>{plan}</beat_data>', write_context)
    print(text)

#
# This will require some processing to work
#
CHOICE_PROMPT = f"""
GENERATION MODE: Options{EXTRA}
NUMBER OF BEATS: """
def summarize_plot_beats(beats: List[str]) -> List[str]:
    return [
        re.search('\\*\\*Change Command:\\*\\* `(.*)`', o).group(1)
        for o in beats
    ]

# TODO: for curse, this is ``yaml{formatted yaml}```
import yaml
def ask_for_ideas(context: List[AnyMessage]) -> Tuple[List[str], List[str]]:
    output = send_message(architect, CHOICE_PROMPT + "3", context)
    if args.story == 'reality':
        options = re.split('\\*\\*\\*', output)[1:]
        return summarize_plot_beats(options), options
    else:
        options = next(yaml.safe_load_all(output[8:-4]))
        return [o['beat_summary'] for o in options], [yaml.safe_dump(o) for o in options]


#
# Keep writing until you want to stop
#
# At the moment, there are two "commands":
#   - whereami, /context: print prior chapter
#   - exit, /finish: stop the loop
#   - help, /help: request ai help for generating next actions
#
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
while True:
    prompt = input("Change> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break
    if prompt in ["whereami", "/context"]:
        print(write_context[-1].content)
        continue
    if prompt in ["help", "/help"]:
        PLOT_PLAN.clear()
        display, options = ask_for_ideas(write_context)
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
        prompt = get_next_input_from_plan()
    else:
        PLOT_PLAN.clear()
    response = send_message(writer, prompt, write_context)
    print(response)

# Store the conversation in a per-run file so we can easily send it to
# other prompts for improvements/etc.
print(f'Cost of Run: {writer.est_cost() + architect.est_cost()}')
chat_file = writer.chat_log.save(config.output_dir)
arch_file = architect.chat_log.save(config.output_dir)

# TODO: me - What does this do that's not already in the chat log?
# Aside from saving in the same location as the story files ???
# Save the current state of generation in a temp file in the story directory
# This is to enable continuations through the --resume flag
story = [response for response in writer.chat_log.having_role('AI')]
with open(f'./{config.directories.story}/{args.story}/principles/tmp.json', 'w', encoding='utf-8') as f:
    json.dump({ 'chapters': story, 'writer_file': chat_file, 'architect': arch_file, 'plan': PLOT_PLAN }, f, ensure_ascii=False, indent=4)
