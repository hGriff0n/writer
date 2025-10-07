
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
- Make setup work for curse
- Generalize the design of the plot and expander
    - Rewrite curse and reality setups to new design

# Integrating choices into the cli
- Potentially also a matter of integrating into the cli so you can reference one of the options

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
    - Traits and Attribute Tracking
        - Every character should have a detailed recording of all traits
        - Define quality to number conversion for all options too
    - Making the Beat Generator emotionless is a good idea


I can make a basic story using the help and an adjusted version of the existing narrative constraints. But directly using the principles approach does not seem to work, aside from maybe initially. WHY


Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

plot_beat_generator
- still doing direct assignment when it shouldn't
- consequential ripples still need some work, but okay for now
  - like the idea of specifically "attributing" points to each explanation
- lost the aspect of "specific" change, but that can be corrected in editing
  - but that's not an option for options
- consequences try to "follow" the character too long
  - but don't have a good way of systematizing this
- social imperceptibility (some of the options it makes long-term should be banned)
- sexiness (beauty seems to often be enhanced in one go)

expander:
- a bunch of work can be done on touching up the tone/adding hints for later stages