
from typing import List, Optional

from lib.ai import init_model, ChatLog
from lib.config import load_config
from lib.util import extract_between_tags

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage


# Initialize chat app
config = load_config()
model = init_model(config, 'gemini')

# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/

### Input Specification
# If length is an issue
# Prose quality and natural paragraphing are ALWAYS more important than hitting a specific paragraph count.

STORY = 'reality'
WRITER_PROMPT = config.load_prompt_file('simple_writer')


#
# Track token usage so I can estimate cost of paid tiers
# TODO: me - Can't estimate until I include costs in llm config
# 
token_stats = {'input': 0, 'output': 0, 'prompts_over_200k': 0}
def update_metadata_stats(response: AnyMessage):
    response = context[-1]
    if not response.response_metadata:
        return
    m = response.response_metadata['usage_metadata']
    token_stats['input'] += m['input_tokens']
    token_stats['output'] += m['output_tokens']
    if m['output_tokens'] >= 200000:
        token_stats['prompts_over_200k'] += 1


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
                 context: List[AnyMessage],
                 chat_log: Optional[ChatLog] = None) -> str:
    if context is None:
        raise Exception("Context must be specified")

    # Assemble prompt and context
    # TODO: me - Eventually use more memory channels
    input = [SystemMessage(content=WRITER_PROMPT)]
    if context:
        input.append(SystemMessage(f'<story_so_far>{context[-1].content}</story_so_far'))
    context.clear()
    context.extend(input + [HumanMessage(content=message)])

    # Call the llm and record the request in the chat-log
    context.append(model.invoke(input=context))
    update_metadata_stats(context[-1])
    if chat_log is not None:
        chat_log.conversation.extend([
            {'role': 'ME', 'msg': message},
            {'role': 'AI', 'msg': context[-1].content}
        ])
    return context[-1].content

# 
# Simple helper for introducing some ai help with next direction
# Eventually, this'll become a full-fledged GM system
# 
BIBLE_TAGS = 'story_bible'
choice_generation_prompt = config.load_story_file(STORY, 'choices')
def ask_for_ideas(context: List[AnyMessage]):
    if not context:
        return "Cannot provide ideas with no context"

    bible = f"<{BIBLE_TAGS}>{extract_between_tags(BIBLE_TAGS, context[-1].content)}</{BIBLE_TAGS}>"
    response = model.invoke(input=[
        SystemMessage(content=choice_generation_prompt),
        HumanMessage(content=bible)
    ])
    update_metadata_stats(response)
    return response.content


# 
# Preparing initial story context
# TODO: me - Add error handling when file doesn't exist
# I'm not sure what that would be
# 
scene = config.load_story_file(STORY, 'start')
constraints = config.load_story_file(STORY, 'constraints')
initial_story_bible = f'<{BIBLE_TAGS}>{config.load_story_file(STORY, 'lorebook')}</{BIBLE_TAGS}>'


# 
# Start the writing by sending the initial, context-less, direction
# 
chat_log = ChatLog()
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

    # TODO: me - this might have issues if send_message doesn't mutate context
    if prompt in ["help", "/help"]:
        print(ask_for_ideas(context))
        continue

    # Need a better way to continue on from the previous location
    scene = f"Plot Direction: \"{prompt}\""
    response = send_message(f'{scene}\n{constraints}', context)
    print(response)



# TODO: me - Generate filename based on conversation to simplify loading
print(f'LLM Usage: {token_stats}')
chat_log.save(config.output_dir)
