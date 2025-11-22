
from argparse import ArgumentParser
import json
from typing import Dict, List, Tuple
from deepmerge import always_merger

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

spec = config.load_schema('spec')
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

class ArchitectContext(Context):
    previous_events: List[Dict] = []
    world_state = {}

class WriterContext(Context):
    previous_scenes: List[str] = []

architect_context = ArchitectContext()
architect_context.messages = [SystemMessage(content=architect.prompt)]

write_context = WriterContext()
write_context.messages = [SystemMessage(content=writer.prompt)]


#
# Wrapper for sending a message to the llm
#
def send_message(llm: LlmEngine, message: str | Dict, context: Context) -> str | Dict:
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
    plan = send_message(architect, REQUEST_PLAN, architect_context)
    print(plan.keys())
    PLOT_PLAN = plan['plan_or_options']


def get_next_input_from_plan() -> Dict:
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


# TODO: me - this might need context compaction
# Allow for resuming an in-progress story
if args.resume:
    print("Loading in-progress story...")
    with open(save_file_path(config, story), 'r', encoding='utf-8') as f:
        state = json.load(f)

    print("Restoring prior context...")
    write_context.messages.extend(
        AIMessage([msg]) for msg in state['prose']
    )
    architect_context.messages.extend(
        AIMessage([msg]) for msg in state['plan']
    )
    architect_context.world_state = state.get('state', {})

    print(f"Restoring current plan...")
    print(architect_context.messages[-1])
    PLOT_PLAN = architect_context.messages[-1].content[0]['plan_or_options'][-state['plan_counter']:]

    print(f"Loaded previous story...")
    if len(write_context.messages) > 1:
        print(write_context.messages[-1].content)
    else:
        print(architect_context.world_state)

# Otherwise we're starting a new story, so simply load up the default
# start command and start writing automatically.
# TODO: me - This can sometimes crash based on how well gemini does structure
else:
    print("Starting first turn")
    ws = send_message(architect, story.first_turn, architect_context)
    architect_context.world_state = json.loads(ws['full_world_state_json'])
    print(f"Initial World State: {json.dumps(architect_context.world_state, indent=4)}")


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
#   - /exit: stop the loop
#   - /help: request ai help for generating next actions
#
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
else:
    def report_tokens():
        return f'A:{architect_context.tokens}|W:{write_context.tokens}>'
    
    # Reject any unpursued plot plans from the context because they are now invalidated
    # TODO: me - probably a sign I should be tracking this someway else
    def unroll_plan_memory():
        num_rem = len(PLOT_PLAN)
        num_planned = len(architect_context.messages[-1].content[0]['plan_or_options'])
        if num_rem == num_planned:
            architect_context.messages.pop()
        elif num_rem > 0:
            del architect_context.messages[-1].content['plan_or_options'][-num_rem:]
        PLOT_PLAN.clear()

    def print_options(display: List[str], options):
        print(f'A: {display[0]}')
        print(f'B: {display[1]}')
        print(f'C: {display[2]}')

    while True:
        prompt = input(report_tokens()).strip()
        if prompt == "/exit":
            break
        if prompt.lower() == "/context":
            print(write_context[-1].messages.content)
            continue
        if prompt.lower() == "/plan":
            if not PLOT_PLAN:
                request_new_plan()
            print('- ' + '\n- '.join(summarize_plot_beats(PLOT_PLAN)))
            continue
        if prompt.lower() == "/help":
            unroll_plan_memory()
            display, options = ask_for_ideas(architect_context)
            print_options(display, options)
            choice = input("Select Option (A/B/C)>").lower()
            prompt = {'a': options[0], 'b': options[1],
                      'c': options[2]}[choice]

            # Remove the request for an option and the ai response
            # And replace it with the selected option to keep the context accurate
            del architect_context.messages[-2:]
            architect_context.messages.append(AIMessage([prompt]))
        elif prompt:
            unroll_plan_memory()
            prompt = send_message(
                architect, f'plan 1 beat {prompt}', architect_context)
        else:
            prompt = get_next_input_from_plan()

        print(prompt)
        architect_context.previous_events.append(prompt)
        scene = send_message(architect, prompt, architect_context)['scene_plan']
        context_start = len(write_context.messages)
        print('Got scene plan. Generating...')

        response = []
        # TODO: me - Technically, should aim for splitting by 500/600 words
        s = json.dumps(scene)
        num_splits = len(scene['key_events'])
        for i, event in enumerate(scene['key_events']):
            print(f'Generating for scene event {i} out of {num_splits}...')
            word_budget = event.get('word_target', 100) * 2
            response.append(
                send_message(writer, f'{s}\nStop after {event['event_description']} {word_budget} words', write_context)
            )
        
        # Merge the updated world state into the global state
        dws = json.loads(scene['world_state_delta_json'])
        architect_context.world_state = always_merger.merge(architect_context.world_state, dws)

        # Reset the context to "pretend" we didn't need to split the scene
        write_context.messages = write_context.messages[:context_start]
        write_context.messages.append(HumanMessage([scene]))

        # And also pretend the "full" response was sent at once
        response = '\n'.join(response)
        write_context.previous_scenes.append(response)
        write_context.messages.append(AIMessage(content=response))
        print(response)


# Store the conversation in a per-run file so we can easily send it to
# other prompts for improvements/etc.
print(f'Cost of Run: {writer.est_cost() + architect.est_cost()}')
chat_file = writer.chat_log.save(config.output_dir)
arch_file = architect.chat_log.save(config.output_dir)


# Save the current state of generation in a temp file in the story directory
# This is to enable continuations through the --resume flag
savefile = {
    'prose': write_context.previous_scenes,
    'plan': PLOT_PLAN,
    'previous': architect_context.previous_events,
    'state': architect_context.world_state,
    'files': {
        'writer_file': chat_file,
        'architect_file': arch_file
    }
}

with open(save_file_path(config, story), 'w', encoding='utf-8') as f:
    json.dump(savefile, f, ensure_ascii=False, indent=4)
