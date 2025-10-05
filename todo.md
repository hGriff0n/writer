
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

TODO: me - Need to start formalizing this project (1)
- With release of velopitt alpha, I can backlog that
- Mostly because I'm not sure on next steps

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