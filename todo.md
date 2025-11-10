
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

# [TERTIARY] Specfinding Agents
- Implement the extrapolation workflow using sub-agents for deep-dives and workshops
  - Figure out context management tricks that can keep the system working
  - Is sub-agents for deep dives really a good idea?
    - We still have to pass in context so the sub-agents can interpret stuff
    - Might be better to focus on summarization/RAG/context stuff

# [PRIME] Rewrite Specfinding From First Principles
- Then work on detailing that understanding into architecture plans

- Creating new version of orchestrator
  - Unifies reporting of resume states and finalized output
  - Implements aspects using skills
    - https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
    - https://github.com/maxvaega/skillkit
    - https://forum.langchain.com/t/we-implemented-anthropics-skills-approach-using-langchain-v1-feedbacks/2126
  - 

- Collaborative loop focusing on synthesis and understanding
  - I think I have a pretty good basis for this process
- Intermediate representation that is processed by follow-up agents
  - The IR can also be feed back into the main loop for updates/revisions
  - This also thereby implements context summarization (though how much?)
- Specific aspects are connected to skills which define the search process
  - What does the aspect mean and what things need to be analyzed/collected
  - How is this used in the broader story generation system
  - What schema is used to represent the stored data
- Follow up agents to develop the specific prompts/input for later stages
  - Re-exploration of plot and scene development and direction
  - I believe I have an okay schema 
- Final architect+writer agent which are run tag-team to make the prose
  - The architect plans long-form plot beats and action
  - The writer then takes those plot beats and develops scenes showing them

# [SECONDARY] Identifying Specfinding
- Adapt the architect prompt to use the json spec
  - Also need to update scripts
  - ai studio doesn't seem to help
- Create schema for writer agent
- Create writer agent (or prompt to produce writer agent from IR)
- Add summarization mode for dealing with context
  - Verify updated prompt using the output of the structured output tests
  - https://aistudio.google.com/app/prompts/1jkTC-zTuwbflL-ZPHL8PqNMbvyIHN5Wt
  - sketchpad needs to be more exploratory
  - intermediate document was very bare in terms of narrative engines
    - said they will be specified later, but that wasn't indicated in brief
- Investigate redoing the orchestration prompt with more fidelity
  - Explicitly cast the individual concepts as "skills"
    - Each run only loads a single "skill" but the AI would be able to identify information that may be relevant for a future skill
  - The main conversation then just becomes an exploration of the entire concept, the intermediate representation is meant to be a report on the world and the narrative 
- Test performance of structured orchestrator_v2 for defining story idea
- Merge engines and generation rules as they are both plot involved
  - Recast generation.md to same format
- Investigate "Writer Explorer" prompt to refine IR for scene/writer
  - Basically specfinding/writer.md
  - Beat Generation/Authorial tone and style/etc.
  - The beat generation would be combined with the extracted plot

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