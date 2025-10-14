
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

# Agentic Extensions
- Implement RAG agent
- Implement skills check agent
- Implement an iterative agent with critique

# Refining Principles Approach
- Try the principles prompt with the mara story
  - Use this story to develop "premise" generation
  - I need to define what each tag means
- Try the principles prompt with the thousand story
- Try the principles prompt with the level story

# Plot Points to Refinement
- Investigate alternative ways of providing context via LangChain
  - https://python.langchain.com/docs/how_to/output_parser_yaml/
    - migrate schemas to structured output
  - Need to migrate story context to yaml

- Fix reality issue: Lost in-world amnesia
  - Not fully following writer prompts
- Fix curse issue: Writer output to json????

- Generalize "writer" of story-specific aspects
  - Experiment with general writer
  - Experiment with optimized writer
    - Might be worth just leaving the writer as "story-specific" for now

# Building on the general principles
- Work on adding events to the prompt handling
- Work on establishing what a principle and rule is
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)

# Long Form Text Generation
- I want detailed novel like prose
- I'm thinking keeping tracking of progression and having a "give next part"

# Cleaning up some minor misses
- Build `story_state` from response

# !!! Clean up the Presentation in CLI !!!
- I was thinking about the story bible though that's only in ai studio

# Experiment With Basic Prompts
- What would happen if I just use a basic editor prompt
- Migrate the historical progression narrative to engines in curse

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