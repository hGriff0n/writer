
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

- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
# [EXTERNAL] Markdown2pdf
- Integrate with Obsidian to simplify export even more
- Create AI script for running prompt

-- Get specfinding agent
-- Get auto-extraction from spec
-- Update writer
-- Clean up everything

- gemini 3 pretty decent coherence, gen whole story
  - overly sad and morose
  - didn't gen enough characters
    - include questions about population principles
  - have to explicitly say "stop at moment" to prevent full gen
    - how to decide when to split
  - time seems to be a bit fluid

# [PRIME] Rewrite Specfinding From First Principles
- Figure out how to use architect in relation to narrative context
  - Does structured output mean I can give up the output format sections of the prompt?
    - seems to work out ok
  - Tests so far have been in the same chat window (actually, isn't necessarily bad)
    - I'm using two llms anyway so i can keep the window separate
  - Feed architect through token compactor
    - https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03
    - https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm
    - Token Optimization
      - optimised.md works in some respects, a bit behind in others
      - https://aistudio.google.com/app/prompts/1FOvMZwU1xFqo8Kg3xaJui-42pjUoSbzE
      - Cuts tokens in half (vs analyzedPlus which is -1000)
- Work on improving fidelity of analyzer
  - Initial attempts haven't really worked
    - https://aistudio.google.com/app/prompts/1jPlIFKj4y2jiE3LTEjy4A_PUwAzSzNsS (too much compaction?)
    - https://aistudio.google.com/app/prompts/1HFK6j6vfagjmrW88ePBvoiRgB12KdXt8 (too much thinking?)
- Update writer.py to new format
  - have an initialization step, for first turn (should work automatically)
    - streamlines some things in the script
  - also can start integrating the writer.py with specfinding.py
  - Actions are different now
    - Sequential => Plan
    - Options => Options
    - () => Scene
- Integrate ContextManager for orchestrator
  - Migrate internals to langchain message aware
- Validate json schema against updated IR
  - Investigate reporting IR conversations in json, not pseudo-xml
  - switch architect to using json
- Recast all stories to new IR/prewrite model (once voice is not sore)
  - Does include mara, level, thousand
  - `late` needs a prewrite, author update
- Run all stories through style&writer prompts
  - curse
  - reality
  - level/mara/thousand 
- Convert architect+writer into langchain agents (run tag-team to make the prose)
  - The architect plans long-form plot beats and action
  - The writer then takes those plot beats and develops scenes showing them
- Go through all code/prompt and clean up/organize  

- https://arxiv.org/html/2503.04844v1
- https://github.com/narrative-first/narrative-context-protocol
- https://subtxt.app/
- https://dramatica.com/

# [SECONDARY] Identifying Specfinding
- Improve orchestrator for token efficiency: https://aistudio.google.com/app/prompts/1x87pQK4wpb3rNG_S9U9yXkAZbPXOAf9L
- Merge engines and generation rules as they are both plot involved
  - Recast scenescript to be engine based if possible
    - Re-exploration of plot and scene development and direction
  - This and plot beats deserve a lot of investigation
- Investigate adding AI critique agents
- Investigate "Writer Explorer" prompt to refine IR for scene/writer
  - Basically specfinding/writer.md
  - Beat Generation/Authorial tone and style/etc.
  - The beat generation would be combined with the extracted plot
  - Improve writer with style examples or directional hints?
- Investigate using sub-agents for deep-dives dueing specfinding
- Is there anyway to track the constants/config files?
- Investigate compaction layers for merging/reducing prompt bloat
- Experiment with agentic discussion boards/parliaments for long-term plot extrapolation
  - No i have to create my own agent discussion board
- Integrate Bookmarking into Orchestrator (difficult)
  - https://aistudio.google.com/app/prompts/1tUzWztSyO2tq3neM4GOKfEBNvWgvvUtP
- Adding drills to the writer prompt

# Skills and Agents
- Implement agentic layer to dynamically load skill files
  - Work on the detection mechanisms for determining a load is needed
    - I don't think the current logic has enough to do that
  - https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/

# Another Look at Long Term Planning
- Update LlmEngine to auto-create rate limiters
- Investigate new writers that can provide more detailed long-form scenes
- Investigate recasting plot planning to use parliament model
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
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
- Incorporating allegory/commentary: https://aistudio.google.com/app/prompts/1xF53-8YVysnFdbZ88Q_1ueBlm5kyFt8p
- Incorporating research into world building: https://aistudio.google.com/app/prompts/15p_SWfuw-dzpym6oaOb69b9IUIcnY4wX
- Research phases for identifying potential themes, plots, etc.?
  - Also useful for naming/etal

# Artistry
- Doing multiple things with one action
  - The scene doesn't just advance plot A, or relation B, or theme C, but all 3

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
