
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

Can I divide the aspects into general categories
- Shift Pacing and tags can be handled with pre-processing
- Generalize triggers to enable repeated events, don't remove repated events
- Ability for triggered events to directly control generation terms
    - Maybe two-phase, first phase determines triggers/etc.

# Refining Principles Approach
- Try the principles prompt with the curse story
- Try the principles prompt with the mara story
- Try the principles prompt with the thousand story

# Plot Points to Refinement
- Experiment with a multi-part story agent
- First stage writes simple stage direction indicating the what that happens
    <- initial implementation experiment in stories/reality/plot_beat_generator and stories/reality/expander
    <- first pass at generalizing in experiments/architect
- These directions could theoretically include a lot of notes and context
- Then have a follow-up writer that takes the input script and outputs story

# Improving Writer and Next Choices Development
- Actually have a decent progress in the choices_ii, but the output isn't usable
- Potentially also a matter of integrating into the cli so you can reference one of the options
- Mostly figuring out how to have a "long-term" plot manager
- Basically trying to find a way to generalise the narrative constraints so that they are more reusable

# Experiment With Basic Prompts
- What would happen if I just use a basic editor prompt

What if I can describe the output and get an author to write like?

DSPy


a. Formalizing world building procedure into specific prompt
    - Principles is a first pass on this but doesn't seem to generate good enough output to ensure the writer stays on track
b. Developing RAG and context management modules
c. Critique/Improver Agent
    - The critique agent works decently well, but the other agents do not
        - With some manual adjustments, I can get passable output by attaching to the existing systems
        - [ ] Need to figure out where this went wrong
        - [ ] Potentially worth investing in structured input/output
    - This doesn't yet integrate specific plot actions
    - Not sure how well it'd work in RPG/GM situations
d. Expansion/Rewriter Agent
    - Some notes in edit/rewriter.md
e. Improving quality of "next options"
    - With some work, choice_ii.md can be used for less "divisive"
    <- Might have a potential solution with choices_ii.md
    - Though without long-term plot guidance, not very good
        - Can probably tighten up the prompt for reality, focusing on "small" changes this time around
f. Dialing out the morbid/profound personality (also for prompts)
    - There currently isn't a good way to incorporate drifting psychology
g. Exploring the temperature/etc. in AI studio
    - Same procedure as with different models, but instead adjusting the settings
h. Can I get decent results with cheaper models (for the writing)
    - Some comparative experiments with Flash Latest
        - Doesn't adhere as well to the prompt constraints as Pro
        - Generally passable so maybe fixable with additional tweaking or by adding a review stage


CLI Improvements:
- cut out the updated bible by default, present if requested
  - needs a more consistent handling of state especially resume


I can make a basic story using the help and an adjusted version of the existing narrative constraints. But directly using the principles approach does not seem to work, aside from maybe initially. WHY

Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

plot_beat_generator
- long term, doesn't handle the incremental updates, still insists on hardcode increments where the examples used them
- overreliance on family history?
- doesn't seem to be outpacing demographics, or escaping
- thematic focus hint, immediately causes all events to reference
- doesnt break out of explanations even after major changes

expander:
- took a little to nail down the constraints to get the text to generate correctly
- the text seems to hew too closely to the scene, like ticking off mentions. the intention is to provide something to build off of and introduce values to make it better