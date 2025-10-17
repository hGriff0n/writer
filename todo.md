
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works
- Investigate upgrading to paid tier (OR https://vast.ai/): paid

# [PRIME] Experiment with initial agentic system
- I can recast the entire "repl" into a simple agent
  - overkill, but'll help with experience
  - but there is a gain from auto-binding the planner and writer
  - https://docs.langchain.com/oss/python/langgraph/workflows-agents
- Develop reviewer prompt into a very detailed analaysis and editor agent
  - Implement into a Reflection agent
    - https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7
- Implement an iterative agent with critique
  - Plan and Execute (an adaptation on the architect pattern)
- Develop a very basic prompt to test out tool calling
  - Using skills checks in agent.py
- Implement Basic RAG agent w/ vector store and World DB

# [SECONDARY] Identifying Building Blocks
- What is an event/principle/generation/etc.
- Can we make a prompt that would extract these from the premise?
  - https://aistudio.google.com/app/prompts/1DLUcWEu7F5Mb9U8T4PdYKAgLu06CMklC
  - https://aistudio.google.com/app/prompts/1iUYjJW8cjYfuykkTgq2xWEoGkYy-KnE7
  - Doesn't fully work, especially with schemas
- Work on establishing what a principle and rule is

# Agentic Extensions
- Implement world character db (with game state)
  - Building skill check to full model system
- https://www.google.com/search?q=custom+storytelling+world+agent&oq=custom+storytelling+world+agent&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigAdIBCDUyMjBqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8

# Refining Principles Approach
- Try the principles prompt with the mara story
  - Create a better version of "context_generator" for the split workflow
  - Use this story to develop "premise" generation
  - I need to define what each tag means
- Try the principles prompt with the thousand story
- Try the principles prompt with the level story
  - This story is effectively an actual game

# Building on the general principles
- Refining multistage approach
  - Plot planner => scene bullet points
    - How do I do this while maintaining reactions/etc. continuity?
  - scene bullet points => annotated scene script
  - scene script can be passed to focused writers
- Work on adding events to the prompt handling
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)

# Long Form Text Generation
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
- Can maybe explore some of the eqbench framework
- I want detailed novel like prose
- I'm thinking keeping tracking of progression and having a "give next part"
- More planning outlines (read papers?)

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

DSPy


a. Formalizing world building procedure into specific prompt
    - Principles to plot beat scheduler went ok
    - https://aistudio.google.com/app/prompts/1TfQ0X81epFXMMHLU204klSh_v8g_nYp-
    - Current principles workflow will miss a few aspects
        - https://aistudio.google.com/app/prompts/1XZLvOcTQf1ST4TocdfSz57QWPxlWdNBC
    - principles > architect > writer

- Rules may also be useful as a way of "extending schemas" so that I can make the yaml fully general?
  - Would it be possible to using "typed yaml"?
    - Say that this schema applies to `X` fields
    - Then use the `<field>: [X]` to apply the schema
  - This might also imply that schema is a separate category


Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

Unfortunately, I deleted the chats in the history where I asked for the generic writer

### From Doctors
- https://aistudio.google.com/app/prompts/1PS9dlP3vw7CKXIlfZkNLOSUBoAZzoHIc
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