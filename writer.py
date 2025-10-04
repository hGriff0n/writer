
from argparse import ArgumentParser
import json
from typing import List

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
model = LlmEngine(config, args.profile, args.prompt)

# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/

# Input Specification
# If length is an issue
# Prose quality and natural paragraphing are ALWAYS more important than hitting a specific paragraph count.


#
# Wrapper for sending a message to the llm
#
# Also automatically records a history of all chat communications to a log file
# so that I can easily upload these to an analysis prompt that can identify was
# of improving the initial prompt (based on the corrections I had to make)
#
# This automatically truncates the provided context to the last response
# from the model.
# TODO: me - The truncation works for now, because the story concept
# I'm testing is episodic and doesn't need more history
#
def send_message(message: str,
                 context: List[AnyMessage]) -> str:
    if context is None:
        raise Exception("Context must be specified")

    # Assemble prompt and context
    # TODO: me - Eventually use more memory channels
    input = [SystemMessage(content=model.prompt)]
    if context:
        input.append(SystemMessage(
            f'<story_so_far>{context[-1].content}</story_so_far'))
    context.clear()
    context.extend(input)

    # Call the llm and record the request in the chat-log
    return model.invoke(message, context)


#
# Simple helper for introducing some ai help with next direction
# Eventually, this'll become a full-fledged GM system
#
CHOICE_PROMPT = config.load_story_file(args.story, 'choices')
def ask_for_ideas(context: List[AnyMessage]):
    return model.ask_for_help(CHOICE_PROMPT, context)


# The way this will transform to multi-agent up to here is somewhat obvious
# The world generator and librarian agents, plus maybe a story planner, work
# together to develop the input to this prompt stage
#
# But where things go after that is unknown. This approach is good for
# episodic stories, potentially for RPG systems, but not full novels

#
# Preparing initial story context
# TODO: me - Add error handling when file doesn't exist
# I'm not sure what that would be
#
# https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works
# TODO: me - Need to start formalizing this project (1)
#  - With release of velopitt alpha, I can backlog that
#  - Mostly because I'm not sure on next steps
#    a. Formalizing world building procedure into specific prompt
#      - Making the AI shorter to fit into this window
#      <- Somewhat done through the "principles pipeline"
#    b. Developing RAG and context management modules
#    c. Critique/Improver Agent
#      <- Discussed with AI studio on `principles_generator.md`
#      <- This can then be feed into the following prompts:
#        - choices_ii.md (to generate next scene options)
#        - principles_to_context.md (to generate writer context)
#        - principle_reviewer.md (to review the generated document)
#      - This doesn't yet integrate specific plot actions
#      - Not sure how well it'd work in RPG/GM situations
#    d. Expansion/Rewriter Agent
#      - Some notes in rewriter.py
#    e. Improving quality of "next options"
#      <- Might have a potential solution with choices_ii.md
#      - Though without long-term plot guidance, not very good
#         - Can probably tighten up the prompt for reality, focusing on "small" changes this time around
#    f. Dialing out the morbid/profound personality (also for prompts)
#    g. Exploring the temperature/etc. in AI studio
#    h. Can I get decent results with cheaper models (for the writing) DO
#
# CLI Improvements:
#   - cut out the updated bible by default, present if requested
#
context: List[AnyMessage] = []
constraints = config.load_story_file(args.story, 'constraints')
if args.resume:
    print("Loading in-progress story...")
    file = f'{config.directories.story}/{args.story}/tmp.json'
    with open(file, 'r', encoding='utf-8') as f:
        story = json.load(f)
    
    print("Restoring prior context...")
    model.chat_log.conversation.extend({"role": "AI", "msg": chap} for chap in story['chapters'])
    context = [AIMessage(content=model.chat_log.conversation[-1]['msg'])]

    print(f"Loaded previous story from {file}")

else:
    # Otherwise we're starting a new story, so simply load up the default
    # start command and start writing automatically.
    scene = config.load_story_file(args.story, 'start')
    initial_story_bible = config.load_story_file(args.story, 'lorebook')
    response = send_message(
        f'Plot Direction: {scene}\n{constraints}\n<story_bible>{initial_story_bible}</story_bible>', context
    )
    print(response)


#
# Keep writing until you want to stop
#
# At the moment, there are two "commands":
#   - exit, /finish: stop the loop
#   - help, /help: request ai help for generating next actions
#
# All other input is sent directly to the model as the 'Plot Direction'
# along with the story constraints. History is provided through context
while True:
    prompt = input("Change> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break
    if prompt in ["help", "/help"]:
        print(ask_for_ideas(context))
        continue
    if prompt in ["whereami", "/context"]:
        print(context[-1].content)
        continue

    # Need a better way to continue on from the previous location
    response = send_message(
        f'Plot Direction: "{prompt}"\n{constraints}', context)
    print(response)

# Store the conversation in a per-run file so we can easily send it to
# other prompts for improvements/etc.
print(f'Cost of Run: {model.est_cost()}')
chat_file = model.chat_log.save(config.output_dir)

# TODO: me - What does this do that's not already in the chat log?
# Aside from saving in the same location as the story files ???
# Save the current state of generation in a temp file in the story directory
# This is to enable continuations through the --resume flag
story = [response for response in model.chat_log.having_role('AI')]
with open(f'./{config.directories.story}/{args.story}/tmp.json', 'w', encoding='utf-8') as f:
    json.dump({ 'chapters': story, 'chat_log': chat_file }, f, ensure_ascii=False, indent=4)