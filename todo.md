
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

TODO: me - Need to start formalizing this project (1)
- With release of velopitt alpha, I can backlog that
- Mostly because I'm not sure on next steps

a. Formalizing world building procedure into specific prompt
    - Making the AI shorter to fit into this window
        <- Somewhat done through the "principles pipeline"
        - No world building may be a follow-on step
b. Developing RAG and context management modules
c. Critique/Improver Agent
    <- Discussed with AI studio on `principles_generator.md`
        - Need to adjust prompts by hand
    <- This can then be feed into the following prompts:
        - choices_ii.md (to generate next scene options)
        - principles_to_context.md (to generate writer context)
        - principle_reviewer.md (to review the generated document)
    - This doesn't yet integrate specific plot actions
    - Not sure how well it'd work in RPG/GM situations
d. Expansion/Rewriter Agent
    - Some notes in scripts/rewriter.py
e. Improving quality of "next options"
    <- Might have a potential solution with choices_ii.md
    - Though without long-term plot guidance, not very good
        - Can probably tighten up the prompt for reality, focusing on "small" changes this time around
f. Dialing out the morbid/profound personality (also for prompts)
g. Exploring the temperature/etc. in AI studio
h. Can I get decent results with cheaper models (for the writing) DO


CLI Improvements:
- cut out the updated bible by default, present if requested
  - needs a more consistent handling of state especially resume
