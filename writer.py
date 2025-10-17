
from argparse import ArgumentParser
import json
import re
from typing import Dict, List, Tuple
import yaml

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter



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
parser.add_argument('-r', '--runs', type=int, default=0)


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

# Assemble the prompt from a generic template
# This uses a mix of `{template}` and xml tags
p = PromptTemplate.from_template(config.load_prompt_file('architect'))
arch_prompt = p.format(
    narrative_intent=story.narrative_intent,
    beat_assembly=story.generation,
    core_concepts=story.core_concepts,
    engines=story.engines,
    rules=story.rules,
    schema=story.schemas
)

# The architect is a multi-purpose agent dealing with all things about plot
# direction. There are basically 3 modes: planning, options, and
# translation. Planning pre-creates a sequence of `7` chapters whenever no
# input is provided. If the user inputs a change of their own, the plan is
# dropped and the "translation" mode is used to produce an output that the
# writer can use for generation. The user can also query the ai for some
# options, which will provide `3` different ways for progressing the story
# Gemma is not as capable at following the instructions
# https://ai.google.dev/gemini-api/docs/rate-limits
pro_limiter = InMemoryRateLimiter(
    requests_per_second=2 / 60,  # 5 RPM for 2.5pro (so the docs say)
    check_every_n_seconds=0.1,  # Wake up every 100 ms
    max_bucket_size=5,
)
architect = LlmEngine(config, args.profile, prompt=arch_prompt, temperature=0.4, rate_limiter=pro_limiter)

# The writer is solely responsible for taking the plot beat provided by the
# architect and expand it into an actual chapter of prose that extends the
# story it is currently writing
# Gemma is not as capable at writing, but flash is
# Finetunes??? https://huggingface.co/ToastyPigeon/Gemma-3-Starshine-12B
# Or other models: https://eqbench.com/creative_writing.html (Kimi)
# Or Gemma 2: https://huggingface.co/lemon07r/Gemma-2-Ataraxy-9B
flash_limiter = InMemoryRateLimiter(
    requests_per_second=10 / 60,  # 10 RPM for 2.5flash
    check_every_n_seconds=0.1,    # Wake up every 100 ms
    max_bucket_size=5,
)
writer = LlmEngine(config, args.profile, prompt=story.writer, temperature=1.7, flash=True)


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


# TODO: me - Figure out a way to control pacing generically
EXTRA = "\npacing_modifier: 1" if story.title == 'reality' else ''

# Helper method for splitting the next input from the existing plan.
# If there are no planned inputs currently, this requests a new set 
REQUEST_PLAN = f"""
directive:
    mode: Sequential{EXTRA}
    count: 7
"""
PLOT_PLAN = []
def request_new_plan():
    global PLOT_PLAN
    plan = send_message(architect, REQUEST_PLAN, write_context)
    PLOT_PLAN = [yaml.safe_dump(o) for o in next(yaml.safe_load_all(plan))]

def get_next_input_from_plan():
    global PLOT_PLAN
    if not PLOT_PLAN:
        request_new_plan()
    
    prompt, PLOT_PLAN = PLOT_PLAN[0], PLOT_PLAN[1:]
    return prompt


# Helper method for requesting potential next options from the planner
CHOICE_PROMPT = f"""
directive:
    mode: Options{EXTRA}
    count: 3
"""
def summarize_plot_beats(beats: List[str]) -> List[str]:
    return [o['beat_summary' if story.title == 'curse' else 'title'] for o in beats]

def ask_for_ideas(context: List[AnyMessage]) -> Tuple[List[str], List[str]]:
    output = send_message(architect, CHOICE_PROMPT, context)
    options = next(yaml.safe_load_all(output))
    return summarize_plot_beats(options), [yaml.safe_dump(o) for o in options]


# TODO: me - would this need to parsed into yaml?
# Allow for resuming an in-progress story
if args.resume:
    print("Loading in-progress story...")
    file = f'{config.directories.story}/{story.title}/principles/tmp.json'
    with open(file, 'r', encoding='utf-8') as f:
        story = json.load(f)
    
    print("Restoring prior context...")
    writer.chat_log.conversation.extend({"role": "AI", "msg": chap} for chap in story['chapters'])
    write_context = [AIMessage(content=story['chapters'][-1])]

    print(f"Restoring current plan...")
    PLOT_PLAN = story.get('plan', [])

    print(f"Loaded previous story from {file}")
    print(write_context[0].content)

# Otherwise we're starting a new story, so simply load up the default
# start command and start writing automatically.
else:
    print("Starting first turn")
    START = story.first_turn
    arch_context: List[AnyMessage] = []
    plan = send_message(architect, START, arch_context)

    write_context: List[AnyMessage] = []
    text = send_message(writer, plan, write_context)
    print(text)


# TODO: me - This is the closest thing I have to the approach I want
# but I don't know yet how to manage the state
# from langgraph.graph import END, MessageGraph
# MessageGraph()

# builder = MessageGraph()
# builder.add_node("generate", generation_node)
# builder.add_node("reflect", reflection_node)
# builder.set_entry_point("generate")
# https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

# If the `runs` parameter was set, automate the process
# Technically, this actually produces args + 1 chapters
if args.runs > 0:
    import time
    for i in range(0, args.runs):
        print(f'Writing chapter {i} out of {args.runs}...')
        prompt = get_next_input_from_plan()
        response = send_message(writer, prompt, write_context)
        print(f'Completed chapter {i} out of {args.runs}...')
        time.sleep(12)
    book = [response for response in writer.chat_log.having_role('AI')]
    with open('./.tmp/book.txt', 'w') as f:
        f.write('\n---\n'.join(book))
    print(f'Finished writing {args.runs} chapters to ./tmp/book.txt')


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
else:
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
            if not PLOT_PLAN:
                request_new_plan()
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
book = [response for response in writer.chat_log.having_role('AI')]
with open(f'./{config.directories.story}/{story.title}/tmp.json', 'w', encoding='utf-8') as f:
    json.dump({ 'chapters': book, 'writer_file': chat_file, 'architect': arch_file, 'plan': PLOT_PLAN }, f, ensure_ascii=False, indent=4)
