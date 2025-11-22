
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from lib.ai import init_model
from lib.config import load_config, DataConstants

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize chat app
config = load_config(DataConstants())
model = init_model(config, 'gemini') # 'openai'
# https://www.philschmid.de/gemini-langchain-cheatsheet#google-gemini-with-langchain-chat-models

# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
# https://python.langchain.com/docs/introduction/


#
# Setup writer sub-agents
# 
# The intended flow with these agents is that a request would start by
# specifying a world where the story takes place. This could include real
# world locations, media universes, or generated worlds. This unified
# handling would allow for AU stories to be easily supported.
# 
# The generator is intended to operate in context with a librarian: the
# generator creates the world and the librarian answers questions about it.
# This split is specifically done to allow for the librarian to be reused
# during plot and story development to ensure information consistency. The
# librarian is specifically instructed to never answer a question unless it
# knows the answer, although there is a separate mode for extrapolations.
# 
WORLD_GENERATOR_PROMPT = config.load_prompt_file('world/world_generator')

# NOTE: There is no way for me to update the world state from this situation
# https://langchain-ai.github.io/langgraph/agents/context/ would be useful here
WORLD_LIBRARIAN_PROMPT = config.load_prompt_file('world/world_librarian')

# 
# Although not currently used in this script, the plot generator was the
# final work I did before switching to a ground-up development. The idea
# with this generator is that it would be the start of a multi-agent
# workflow which would eventually assemble the final novel. The plot
# generator takes the initial story idea and generates a basic scaffolding
# for plot events to follow. This effectively splits the creation of a long
# story into the assembly of multiple shorter stories, potentially
# simplifying the issues of managing the large context necessary.
#
# After a couple iterations of this approach, I shifted to starting with
# the basic writer and then adding on structure from there. The top-down
# approach lacked an understandable way of progressing from these agents
# to the lower agents which blocked further development progress.
# 
# The forcing issue was actually RPG "choice" mechanics, which I am still
# uncertain about how I will manage to accommodate while preserving overall
# plot direction, especially as none of my current testing ideas have the
# tight plotline that we would be trying to preserve.
# 
# Focusing on the writer first allows for that GM role to develop naturally
# when it becomes necessary to add it - basically developing the individual
# agents when the become useful and necessary. This work direction will be
# resumed eventually, as the agent organization I've developed so far is
# well placed in the long run.
# 
PLOT_GENERATOR = config.load_prompt_file('experiments/plot_generator')

# 
# Send the initial generation request
# TODO: me - Get the world from input
#
# Wheel of Time, pre-breaking of the World
# Star Wars, Choices of One
# High fantasy world with fractured political makeup. World has floating islands which most people believe to be unreachable
# 1970s Belgium
#
requested_world = "Wheel of Time"
response = model.invoke(input=[
    SystemMessage(content=WORLD_GENERATOR_PROMPT),
    HumanMessage(content=requested_world),
])

# 
# Now validate the performance of the world generation layer by
# querying the created world state using the librarian
# 
librarian = WORLD_LIBRARIAN_PROMPT.replace("{{WORLD_SUMMARY_HERE}}", response.content)

while True:
    prompt = input("> ").strip()
    if prompt == "exit" or prompt == "/finish":
        break

    print(model.invoke(input=[
        SystemMessage(content=librarian),
        HumanMessage(content=prompt)
    ]).content)

print(f'Cost of Run: {model.est_cost()}')