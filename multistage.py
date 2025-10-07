
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
if args.story != 'reality':
    raise Exception('Only have multi-stage workflow defined for reality story')

config = load_config(DEFS)


# The current `LlmEngine` approach "assumes" one prompt per llm
# Which has downsides (such as this) and upsides (cleaner calling, mostly)
# Is it better to have one engine per prompt, or reuse the same engine for
# multiple prompts? The former is required when using different models
architect = LlmEngine(config, args.profile, args.prompt, temperature=0.4)
architect.prompt = config.load_story_file(args.story, 'principles/plot_beat_generator')
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


PACE = 1
PLOT_PLAN = []
PLAN_PROMPT = """
GENERATION MODE: Sequential
PACING MODIFIER: {PLAN}
NUMBER OF BEATS: 7
"""

def get_next_input_from_plan():
    global PLOT_PLAN
    if not PLOT_PLAN:    
        REQUEST_PLAN = f"GENERATION MODE: Sequential\nPACING MODIFIER: {PACE}\nNUMBER OF BEATS: 7"
        plan = send_message(architect, REQUEST_PLAN, write_context)
        PLOT_PLAN = re.split("\\*\\*\\*", plan)[1:]
    
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
    START = f"""
    GENERATION MODE: Specified
    CURRENT STATE: Anya, a mildly depressed couch potato stuck in a dead end job as a data clerk at a no-name corporation. Peristent adult acne, poor eyesight requiring thick corrective lenses, 5'4", little money or social engagement. Wears functional unflattering clothes to hide her poor figure
    SPECIFIED CHANGE: I do not need glasses at all
    PACE MODIFIER: {PACE}
    """
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
GENERATION MODE: Options
PACE MODIFIER: {PACE}
NUMBER OF BEATS: """
def summarize_plot_beats(beats: List[str]) -> List[str]:
    return [
        re.search('\\*\\*Change Command:\\*\\* `(.*)`', o).group(1)
        for o in beats
    ]

def ask_for_ideas(context: List[AnyMessage]) -> Tuple[List[str], List[str]]:
    output = send_message(architect, CHOICE_PROMPT + "3", context)
    options = re.split('\\*\\*\\*', output)[1:]
    return summarize_plot_beats(options), options


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
