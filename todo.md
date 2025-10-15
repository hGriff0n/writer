
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

# [PRIME] Experiment with initial agentic system
- Implement Rate-Limiting Handlers (or upgrade to paid tier, or start https://vast.ai/)
- Implement an iterative agent with critique
  - ReAct (very similar to what I have now)
    - Because I handle "", I can technically automate this to a point
  - Plan and Execute (an adaptation on the architect pattern)
  - Reflection
    - Basically incorporating my feedback system
    - https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
- Implement a tool calling agent
  - Using skills check in agent.py

# Agentic Extensions
- Implement RAG agent
- Implement world character db (with game state)
  - Building skill check to full model system

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
- Work on establishing what a principle and rule is
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)

# Long Form Text Generation
- I want detailed novel like prose
- I'm thinking keeping tracking of progression and having a "give next part"

# !!! Clean up the Presentation in CLI !!!
- I was thinking about the story bible though that's only in ai studio

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
- 4/5 "Systems (for non-RPG)
  - Principles/Concepts - ???
  - Events/Triggers - If This Then That
    - Condition
    - Outcome
    - Repeated
  - Engines - Responsible for managing a specific aspect of plot advancement
    - Story Promises++
    - "Travel Agent" - Responsible for ensuring group arrives on time
    - "World Agent" - Simulating world stuff, or maybe that's a character
    - "Antagonist"
    - Director
  - Data - History+Lore+WorldState
    - Librarian/RAG
    - World Generator
    - Lore Researcher
    - Summarizer/Context
  - Functions - SkillChecks/Mechanics
    - Style Prohibitions?
  - Principles
  - And a bunch of uncategorized stuff...
    - Characters
    - Voice/styles (more writer customization)
    - Options/Choices/Planning (separate agent)
    - Long Form
    - Extending/replacing/rewriting (separate agent)

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