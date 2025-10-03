
from argparse import ArgumentParser
from typing import List

from lib.ai import LlmEngine
from lib.config import load_config
from lib.util import extract_between_tags

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage


parser = ArgumentParser(prog='simple_writer', description='simple ai writer')
parser.add_argument('story')
parser.add_argument('-m','--model',
                    choices=LlmEngine.supported_models(),
                    default=LlmEngine.DEFAULT_MODEL)
parser.add_argument('-p','--prompt',
                    choices=['simple_writer'], default='simple_writer')
# TODO: me - add an ability to load a story through argparse
parser.add_argument('-c','--continue', action='store_true')


# Initialize chat app
args = parser.parse_args()
config = load_config()
model = LlmEngine(config, args.model, args.prompt)

# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/

### Input Specification
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
        input.append(SystemMessage(f'<story_so_far>{context[-1].content}</story_so_far'))
    context.clear()
    context.extend(input)

    # Call the llm and record the request in the chat-log
    return model.invoke(message, context)

# 
# Simple helper for introducing some ai help with next direction
# Eventually, this'll become a full-fledged GM system
# 
BIBLE_TAGS = 'story_bible'
choice_generation_prompt = config.load_story_file(args.story, 'choices')
def ask_for_ideas(context: List[AnyMessage]):
    if not context:
        return "Cannot provide ideas with no context"

    response = model.llm.invoke(input=[
        SystemMessage(content=choice_generation_prompt),
        HumanMessage(content=context[-1].content)
    ])
    model.usage_stats.append(response.usage_metadata)
    return response.content


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
# TODO: me - Add an option to continue a story
# 
scene = config.load_story_file(args.story, 'start')
constraints = config.load_story_file(args.story, 'constraints')
initial_story_bible = f'<{BIBLE_TAGS}>{config.load_story_file(args.story, 'lorebook')}</{BIBLE_TAGS}>'


# 
# Start the writing by sending the initial, context-less, direction
# 
context: List[AnyMessage] = []
response = send_message(
    f'Plot Direction: {scene}\n{constraints}\n{initial_story_bible}', context
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

    # Need a better way to continue on from the previous location
    scene = f"Plot Direction: \"{prompt}\""
    response = send_message(f'{scene}\n{constraints}', context)
    print(response)

# TODO: me - Generate filename based on conversation to simplify loading
# Bu what information would I record
print(f'Cost of Run: {model.est_cost()}')
model.chat_log.save(config.output_dir)
