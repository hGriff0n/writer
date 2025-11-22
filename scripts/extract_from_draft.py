import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from argparse import ArgumentParser
import regex as re

from lib.ai import LlmEngine
from lib.config import Config
from langchain_core.prompts import PromptTemplate


CONF = Config()
parser = ArgumentParser(prog='specfinding', description='story spec discussion')
parser.add_argument('story')
parser.add_argument('-m', '--profile',
                    choices=CONF.supported_models,
                    default=LlmEngine.DEFAULT_MODEL)
parser.add_argument('-o', '--out', default='.tmp/spec.md', type=str)
parser.add_argument('-f', '--file', type=str)


# Initialize chat app
args = parser.parse_args()
story = CONF.load_story(args.story)


# Initialize the model
llm = LlmEngine(CONF, args.profile, prompt=story.writer)


# Update with assembled doc after every "step"
document = story.fullspec if not args.file else load_markdown(args.file)
if not document:
    print('No input in essay file. Exiting immediately')
    exit()

# Load up the ingest script, pre-filling out the essay we are analyzing
# Then format the first iteration with an empty living spec (since we haven't extracted yet)
system_prompt = PromptTemplate.from_template(CONF.load_prompt('specfinding/ingest'), partial_variables={'document': document})
initial = system_prompt.format(living_spec="")

# Prepare the user query that we will repeatedly run to extract all possible components
# The first query needs to provide some extra instructions for tasking but we don't
# need to keep providing it as it's already in the context. We simply swap it out for an
# empty string on subsequent runs, keeping the main directive clear and actionable
first = "<living_spec/> is a markdown document formatted according to the `Output Format` containing the \"living document\" for the story. Your task is:\n\n"
question = PromptTemplate.from_template(r"""
{header}Analyze <source_essay/> and compare it against the "living document" to identify if there are any {component} specified or implied in the essay that are not currently in the "living document".

If you identify a missing component, you must add those components to the list. Repeat this process until you can't identify any new components and then output the detailed list of added components and then stop.

If you are unable to find any new components, you must return immediately and only output the following string: [[STOP]].""")


# Although there are multiple components in the generative framework, in practice, we've
# only gotten good outputs consistently for `World Codex` and `Core Concepts` - anything
# else is hit-or-miss at best. Most things are actually worse and would take far more time
# and effort in the specfinding phase to undo wrong guesses than we save
components = [
    'World Codex',
    'Core Concepts',
    'Narrative Engines',
    # 'Beat Generation Rules'
]
living_spec = ""
extract_md = re.compile(r"###.+\n((?s).+)")

for c in components:
    print(f'Now processing spec initialization queries for {c}')
    added_content = []
    messages = [SystemMessage(system_prompt.format(living_spec=living_spec))]
    resp, usage = llm.invoke(question.format(header=first, component=c), messages)
    for i in range(0, 5):
        # Handle the break
        if resp == "[[STOP]]": break
        # Don't report engine fluff
        m = re.search(extract_md, resp)
        if not m:
            exit()
        added_content.append(m.group(1))
        # And repeat until done
        print(f'Making followup query for `{c}: {i} out of 5')
        resp, usage = llm.invoke(question.format(header='', component=c), messages)
    
    # We've done all we can so add the established information to the doc
    print(f'Finished follow-up queries for {c}')
    living_spec += f'### {c}\n{'\n'.join(added_content)}\n\n'
    
# And then save the document to the output file
with open(f'./{args.out}', 'w+', encoding='utf-8') as f:
    f.write(living_spec)
