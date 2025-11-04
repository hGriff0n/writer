
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works
- Investigate upgrading to paid tier (OR https://vast.ai/): paid

- the cohesive specfinding approach works very well in practice
  - pretty close to the by hand construction
  - has some issues, notably with specifics and schemas
  - also requires very fine-grained context management

# Tracks
- Specfinding Agents
- Improving Specfinding Search
- Longform Storydesign
- Interactive Storydesign
- Enforcing world fidelity and other rules
- RAG/Skills agent
- Create Specsheets for all narratives

# [EXTERNAL] Markdown2pdf
- Integrate with Obsidian to simplify export even more
- Create AI script for running prompt

# [PRIME] Specfinding Agents
- Implement the extrapolation workflow using sub-agents for deep-dives and workshops
  - Figure out context management tricks that can keep the system working
  - Is sub-agents for deep dives really a good idea?
    - We still have to pass in context so the sub-agents can interpret stuff
    - Might be better to focus on summarization/RAG/context stuff

# [SECONDARY] Identifying Specfinding
- Try out architect_v2 (orchestrator_v2?)
- Structured output of produced modules
  - Rework the specfinding process in light of the spec
    - Because the output needs to be usable to produce the spec
  - Integrate it into the orchestrator_v2 so the final output
    - Can't integrate directly with orchestrator_v2 because it ruins the conversational aspect of the prompt
    - I think I'll need to make the orchestrator_v2 export a full report that can be filtered into the 
- Merge engines and generation rules as they are both plot involved
  - Recast generation.md to same format

# Another Look at Long Term Planning
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
- TODO: Requires Paid API tier
- Identify the place of ending and the duration
  - https://aistudio.google.com/app/prompts/18QQT7o5XLqF_ZXpKnRp61ipzNSagsHYQ
  - Solution would seem to require multiple agents

# Refining Principles Approach
- Use the new workflow for the curse story
- Use the new workflow for the mara story
- Use the new workflow for the thousand story
- Use the new workflow for the level story
  - this is actually a game, so may need some adjustments

# Building on the general principles
- Work on adding events to the prompt handling
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)
    1. Attach modules/skills for producing the specific narrative items

# !!! Clean up the Presentation in CLI !!!
- I was thinking about the story bible though that's only in ai studio
- Rewrite LlmEngine in terms of `create_agent`??
  - Allows for dynamic model selection if needed

# Experiment With Basic Prompts
- What would happen if I just use a basic editor prompt
- Migrate the historical progression narrative to engines in curse

# Plot Points to Refinement
- https://python.langchain.com/docs/how_to/output_parser_yaml/
  - migrate schemas to structured output
- Need to migrate story context to yaml
  - Build `story_state` from response
- Investigate making a "plot critique" agent that could parse the output of the ai studio tests and present ways of adjusting the pacing
  - If I'm using a lot of time jumps, then it's probably a sign the steps are too small
  - Would also have to include specific comments about what I didn't like since I don't have any tools to indicate that directly in the conversation
- Would be a good idea to make an agent specifically for understanding this process and all of it's interactions (could be good for adding new features)

# Other Agents:
- Develop reviewer prompt into a very detailed analaysis and editor agent
  - Implement into a Reflection agent
    - https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7
- Implement an iterative agent with critique
  - Plan and Execute (an adaptation on the architect pattern)
- Develop a very basic prompt to test out tool calling
  - Using skills checks in agent.py
- Implement Basic RAG agent w/ vector store and World DB

DSPy

Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

### From Doctors
- https://aistudio.google.com/app/prompts/1PS9dlP3vw7CKXIlfZkNLOSUBoAZzoHIc
  - https://aistudio.google.com/app/prompts/1uhpjcEzT_c8XA_56_IiDYs0xyekQ0d1-
    - Develop sub-agent for refining specific ideas of implementation
      - The main agent would identify when a conversation is going into "refinement mode" and engage the sub-agent
      - This can be done proactively during the final refinement stage
      - The sub-agent can then spend as much context window as it needs on the question of refining the principle in question and return the agreed results for final incorporation
- 4/5 "Systems (for non-RPG)
  - Principles/Concepts - Reference for Intended World/Plot/Emotional Outcomes
    - Basically "Setting the scene" on which all other aspects work
    - Everything else is basically trying to implement these aspects
    - Effectively operate as a review document for identifying compliance
  - Narrative Rules
    - Translation of some qualitative principles into quantitative actions
    - Basically a library of tools for other aspects to utilize
    - TODO: Skill checks and other mechanics and tools would reside here
  - Engines - Responsible for managing a specific aspect of plot advancement
    - Effectively work as "plot advocates", promoting a specific action for the next turn (based on the current/recent plot events)
    - Later resolution stages are responsible for weeding the actions into a cohesive and digestable narrative
    - Internally function as a state machine, so have a large overlap with events
  - Events/Triggers - If This Then That (unimplemented)
    - Way of modifying the prompt data in response to runtime events
    - Can be a useful way for achieving context management
    - Made of a Condition, an Outcome, and a "IsRepeated" flag
  - Beat Generation
    - Takes a proposal document and transforms it into an actionable prose request for the writer
  - Other
    - Data - History+Lore+WorldState (unimplemented)
      - Librarian/RAG
      - World Generator
      - Lore Researcher
      - Summarizer/Context
    - Writer (unspecified)
      - Voice/styles (more writer customization)
      - Long Form
      - Extending/replacing/rewriting (separate agent)
    - And a bunch of uncategorized stuff...
      - Characters
      - Options/Choices/Planning (separate agent)
  - Potential engines
    - Story Promises++
    - "Travel Agent" - Responsible for ensuring group arrives on time
    - "World Agent" - Simulating world stuff, or maybe that's a character
    - "Antagonist"
    - Director

### Engines
- Also the generation of choices and options
- Modelling the progression of narrative events, pushing the story further
    - This can be specific plot arcs, or more emergent "promises"
    - This also includes very mundane concerns like ensuring compliance with historical events and other concerns
    - Unsure about what input would be needed but the output would be a potential plan for what to do next
    - There is then a resolver that analyzes the proposals and decides which ones to accept
    - There would have to be an internal ability to react to plot state, blocking until a certain signal, or upgrading based on a signal
        - though both of these can be accomplished through events
- Engines have a start state, where they are before anything else
- Seems to have a lot of interaction with Events
    - Some engines are dead at the start, ie wait for event
- Blocking situation
- Exit state
- Cleanup phase


https://arxiv.org/pdf/2404.13919