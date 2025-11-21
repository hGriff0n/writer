
from argparse import ArgumentParser
import json
from typing import Dict, List, Tuple

from lib.ai import LlmEngine
from lib.config import load_config, DataConstants

from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
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
if args.story != 'late':
    raise Exception("Only 'late' and 'curse' stories are currently supported")

#
# The current `LlmEngine` approach "assumes" one prompt per llm
# Which has downsides (such as this) and upsides (cleaner calling, mostly)
# Is it better to have one engine per prompt, or reuse the same engine for
# multiple prompts? The former is required when using different models
#

# Assemble the prompt from a generic template
# This uses a mix of `{template}` and xml tags
p = PromptTemplate.from_template(
    config.load_prompt_file('architect'))
arch_prompt = p.format(
    story_arch=story.story_arch
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
# pro_limiter = InMemoryRateLimiter(
#     requests_per_second=2 / 60,  # 5 RPM for 2.5pro (so the docs say)
#     check_every_n_seconds=0.1,  # Wake up every 100 ms
#     max_bucket_size=5,
# )

spec = config.load_schema('specfinding/spec')
architect = LlmEngine(
    config,
    args.profile,
    prompt=arch_prompt,
    temperature=1,
    schema=spec
)

# The writer is solely responsible for taking the plot beat provided by the
# architect and expand it into an actual chapter of prose that extends the
# story it is currently writing
# Gemma is not as capable at writing, but flash is
# Finetunes??? https://huggingface.co/ToastyPigeon/Gemma-3-Starshine-12B
# Or other models: https://eqbench.com/creative_writing.html (Kimi)
# Or Gemma 2: https://huggingface.co/lemon07r/Gemma-2-Ataraxy-9B
# flash_limiter = InMemoryRateLimiter(
#     requests_per_second=10 / 60,  # 10 RPM for 2.5flash
#     check_every_n_seconds=0.1,    # Wake up every 100 ms
#     max_bucket_size=5,
# )
writer = LlmEngine(
    config,
    args.profile,
    prompt=story.writer,
    temperature=1
)


# TODO: me - Merge with ContextManager?
class Context(object):
    messages: List[AnyMessage] = []
    tokens: int = 0


architect_context = Context()
architect_context.messages = [SystemMessage(content=architect.prompt)]

write_context = Context()
write_context.messages = [SystemMessage(content=writer.prompt)]

# TODO: me - integrate better world state tracking
global_world_state = {}


#
# Wrapper for sending a message to the llm
#
def send_message(llm: LlmEngine, message: str, context: Context) -> str | Dict:
    if context.messages is None:
        raise Exception("Context must be specified")

    resp, usage = llm.invoke(message, context.messages)
    context.tokens = usage['total_tokens']
    return resp


# Helper method for splitting the next input from the existing plan.
# If there are no planned inputs currently, this requests a new set
REQUEST_PLAN = f"""plan 7 beats"""
PLOT_PLAN = []
def request_new_plan():
    global PLOT_PLAN
    plan = send_message(architect, REQUEST_PLAN, write_context)
    print(plan.keys())
    PLOT_PLAN = plan['plan_or_options']


def get_next_input_from_plan():
    global PLOT_PLAN
    if not PLOT_PLAN:
        request_new_plan()

    scene, PLOT_PLAN = PLOT_PLAN[0], PLOT_PLAN[1:]
    return scene


# Helper method for requesting potential next options from the planner
CHOICE_PROMPT = f"""3 options for next beat"""
def summarize_plot_beats(beats: List[Dict]) -> List[str]:
    return [o['title'] for o in beats]

def ask_for_ideas(context: Context) -> Tuple[List[str], List[str]]:
    resp = send_message(architect, CHOICE_PROMPT, context)
    options = resp['plan_or_options']
    return summarize_plot_beats(options), options

def save_file_path(config, story) -> str:
    return f'./{config.directories.story}/{story.title}/resume.json'

# Allow for resuming an in-progress story
if args.resume:
    print("Loading in-progress story...")
    with open(save_file_path(config, story), 'r', encoding='utf-8') as f:
        story = json.load(f)

    print("Restoring prior context...")
    write_context.messages.extend(
        AIMessage(content=msg) for msg in story['prose']
    )
    architect_context.messages.extend(
        AIMessage(content=msg) for msg in story['plan']
    )
    global_world_state = story.get('state', {})

    # TODO: me - I believe this needs to handle 'parsed'??
    print(f"Restoring current plan...")
    PLOT_PLAN = architect_context.messages[-1].content['plan_or_options'][-story['plan_counter']:]

    print(f"Loaded previous story...")
    print(write_context[-1].content)

# Otherwise we're starting a new story, so simply load up the default
# start command and start writing automatically.
else:
    print("Starting first turn")
    ws = send_message(architect, story.first_turn, architect_context)
    global_world_state = json.loads(ws['full_world_state_json'])
    print(f"Initial World State: {json.dumps(global_world_state, indent=4)}")


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
    print('exiting')
    pass
    # import time
    # for i in range(0, args.runs):
    #     print(f'Writing chapter {i} out of {args.runs}...')
    #     prompt = get_next_input_from_plan()
    #     response = send_message(writer, prompt, write_context)
    #     print(f'Completed chapter {i} out of {args.runs}...')
    #     time.sleep(12)
    # book = [response for response in writer.chat_log.having_role('AI')]
    # with open('./.tmp/book.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n---\n'.join(book))
    # print(f'Finished writing {args.runs} chapters to ./tmp/book.txt')


#
# Keep writing until you want to stop
#
# At the moment, there are two "commands":
#   - /context: print prior chapter
#   - exit, /finish: stop the loop
#   - /help: request ai help for generating next actions
#
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
else:
    def report_tokens():
        return f'A:{architect_context.tokens}|W:{write_context.tokens}>'

    while True:
        prompt = input(report_tokens()).strip()
        if prompt == "exit" or prompt == "/finish":
            break
        if prompt.lower() == "/context":
            print(write_context[-1].messages.content)
            continue
        if prompt.lower() == "/help":
            PLOT_PLAN.clear()
            display, options = ask_for_ideas(architect_context)
            print(f'A: {display[0]}')
            print(f'B: {display[1]}')
            print(f'C: {display[2]}')
            choice = input("Select Option (A/B/C)>").lower()
            prompt = {'a': options[0], 'b': options[1],
                      'c': options[2]}[choice]
            # Remove the request for an option and the ai response
            # And replace it with the selected option to keep the context accurate
            del architect_context.messages[-2:]
            architect_context.messages.append(AIMessage(content=prompt))
        if prompt.lower() == "/plan":
            if not PLOT_PLAN:
                request_new_plan()
            print('- ' + '\n- '.join(summarize_plot_beats(PLOT_PLAN)))
            continue

        # Allow for automated planning of plot events
        if not prompt:
            prompt = get_next_input_from_plan()
        else:
            num_rem = len(PLOT_PLAN)
            # Reject any unpursued plot plans from the context because they are now invalidated
            # TODO: me - probably a sign I should be tracking this someway else
            if num_rem > 0:
                del architect_context.messages[-1].content['parsed']['plan_or_options'][-num_rem:]
            PLOT_PLAN.clear()
            prompt = send_message(
                architect, f'plan 1 beat {prompt}', architect_context)

        # TODO: me - Need to update world state
        scene = send_message(architect, prompt, architect_context)['scene_plan']

        context_start = len(write_context.messages)
        response = []
        # TODO: me - Technically, should aim for splitting by 500/600 words
        for event in scene['key_events']:
            word_budget = event.get('word_target', 150) * 2
            response.append(
                send_message(writer, f'{scene}\nStop after {event['event_description']} {word_budget} words', write_context)
            )
        write_context.messages = write_context.messages[:context_start]
        write_context.messages.append(HumanMessage(content=scene))

        response = '\n'.join(response)
        write_context.messages.append(AIMessage(content='\n'.join(response)))
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
# TODO: me - This should be done based on the context tracking, but that has it's own issues currently
book = [response for response in writer.chat_log.having_role('AI')]
plan = [response for response in architect.chat_log.having_role('AI')]
savefile = {
    'prose': book,
    'plan': plan,
    # Number of planned plot points remaining before requesting a new plan
    'plan_counter': len(PLOT_PLAN),
    'state': global_world_state,
    'files': {
        'writer_file': chat_file,
        'architect_file': arch_file
    }
}

with open(save_file_path(config, story), 'w', encoding='utf-8') as f:
    json.dump(savefile, f, ensure_ascii=False, indent=4)
