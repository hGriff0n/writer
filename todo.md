
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

TODO: me - Need to start formalizing this project (1)
- With release of velopitt alpha, I can backlog that
- Mostly because I'm not sure on next steps

# Agentic Extensions
- Implement RAG agent
- Implement skills check agent

# Formalising Input/Output Formats
- Update the writer to expect the storybook in a structured format
- Theoretically need to also pass the constraints in json
- Experiment with getting the principles to output in that format
- Plot Beat Generator should definitely ouput json, would make options parsing better

# Refining Principles Approach
- Try the principles prompt with the curse story
- Try the principles prompt with the mara story
- Try the principles prompt with the thousand story

# Plot Points to Refinement
- Generalize the design of the plot and expander
  - Rewrite curse and reality setups to new design
  - https://aistudio.google.com/app/prompts/129vfGT5HIPIjC2a7srmgg6oBzntXxTpR

# Experiment With Basic Prompts
- What would happen if I just use a basic editor prompt

What if I can describe the output and get an author to write like?

DSPy


a. Formalizing world building procedure into specific prompt
    - Principles to plot beat scheduler went ok
    - https://aistudio.google.com/app/prompts/1TfQ0X81epFXMMHLU204klSh_v8g_nYp-
    - Current principles workflow will miss a few aspects
        - https://aistudio.google.com/app/prompts/1XZLvOcTQf1ST4TocdfSz57QWPxlWdNBC
b. Developing RAG and context management modules
c. Critique/Improver/Rewrite/Expand Agent
    - The critique agent works decently well, but the other agents do not
        - With some manual adjustments, I can get passable output by attaching to the existing systems
        - [ ] Need to figure out where this went wrong
        - [ ] Potentially worth investing in structured input/output
    - This doesn't yet integrate specific plot actions
    - Not sure how well it'd work in RPG/GM situations
d. Dialing out the morbid/profound personality (also for prompts)
    - There currently isn't a good way to incorporate drifting psychology
    - It's "foreshadowing" is very on the nose
e. Can I get decent results with cheaper models (for the writing)
    - Would flash give okay results for editing only. YES


CLI Improvements:
- cut out the updated bible by default, present if requested
  - needs a more consistent handling of state especially resume


Can I divide the aspects into general categories
- Shift Pacing and tags can be handled with pre-processing
- Generalize triggers to enable repeated events, don't remove repated events
- Ability for triggered events to directly control generation terms
    - Maybe two-phase, first phase determines triggers/etc.

How to generalize current work:
- **Reality**
    - The mechanistic idea can be a repeatable event
        - The required calculations are specific to encode though
        - Especially the consequential framework, although this can be somewhat integrated into a core component of reality manager
        - Concept of "Triggers" (or "Narrative Engines")
    - Narrative Rules and Principles
    - Descriptive Tone
    - Traits and Attribute Tracking
        - Every character should have a detailed recording of all traits
        - Define quality to number conversion for all options too
    - Making the Beat Generator emotionless is a good idea


I can make a basic story using the help and an adjusted version of the existing narrative constraints. But directly using the principles approach does not seem to work, aside from maybe initially. WHY

Multi-plot lines: Each engine makes a suggestion and then a narrative agent decide which ones to accept/integrate/refuse/delay/etc based on story development


Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

Unfortunately, I deleted the chats in the history where I asked for the generic writer

## Generalizing reality/principles/architect.md

Your role and modes: Good general
Next section: specific story, identifying goals and principles
Core Principles: Story-specific
Domain Rate: Story-specific
Input Parameters: Largely general, story specific interpretations
Output Generation: Mostly story-specific
Output format: Story specific

### Each section use
Prompt role and Modes: Sequential, Specific, Options
Basic overview of the main story trope and intended experience
- Also reinforces following principles and format
  <- Can I explicitly split this into two sections? One for following one for experience. The later could be the intro for section 1 too
List of "Core Principles" for the World/Story
Constant Data Section
Input Parameters
- Should be possible to structure most aspects
- But will also need a lot of story-specific data
Generation Process for Output
- A little bit of modal handling
- Trigger calculation
Output Formatting

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
  - And a bunch of uncategorized stuff...
    - Characters
    - Voice/styles
    - Principles
    - Options/Choices
    - Long Form
    - Extending/replacing/rewriting
- 

### Systematizing Matrix
Domain matrix is then referenced in output generation
The entire output generation is basically one trigger/narrative engine

## Generalizing reality/principles/scene_write.md

Your role: Slightly story-specific
Guiding Principles: Very story specific
Story Beat: Slightly Story Specific
How to Translate: Slightly Story Specific
Directives: Explicitly Story Specific
Execution: General, potentially removable

Scene write is ultimately harder to systematize because it's dealing with the "voice" of the story