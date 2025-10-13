
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

# Agentic Extensions
- Implement RAG agent
- Implement skills check agent

# Refining Principles Approach
- Try the principles prompt with the mara story
  - Use this story to develop "premise" generation
- Try the principles prompt with the thousand story
- Try the principles prompt with the level story

# Plot Points to Refinement
- Test out "generic" architect

- Generalize "writer" of story-specific aspects
  - Move "story-specific" data into story package and update the yaml and loaders
- Make "generic" architect and writer prompts
  - Writer needs a bit more story specific stuff iir

- Work on adding events to the prompt handling
- Work on establishing what a principle and rule ia
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)

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

## Generalizing architect.md
Intro: Common
Core Principles:
  - Completely story specific, need definition
Narrative Engines:
  - Identified common elements, but mostly story specific
Data:
  - Completely story specific
  - not identified common themes
  - Can be a substate that introduces custom schema types
Input Specification:
  - Mostly common, want to have way of including story specific schema customizations
Generation:
  - Mostly common, minor story specifics in finalization
Output:
  - Almost common, just need story specific schemas

### From Doctors
- 4/5 "Systems (for non-RPG)
  - Events/Triggers - If This Then That
    - Condition
    - Outcome
    - Repeated
  - Engines - Finite Plot Machines
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
