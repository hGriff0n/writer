
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

# [EXTERNAL] Markdown2pdf
- Integrate with Obsidian to simplify export even more
- Create AI script for running prompt

# [PRIME] Rewrite Specfinding From First Principles
- Impressions of prompt (doing well overall, though prompt is >10k without skills)
  - Don't think it's using the "sketchbook" to it's full potential yet
  - Not sure it has good "cross-reference"/compaction abilities
  - Doesn't seem to be capturing cross-reference ideas
- Implement agentic layer to dynamically load skill files
  - Switch to paid tier due to rate limits
  - Work on the detection mechanisms for determining a load is needed
    - I don't think the current logic has enough to do that
  - Investigate systems for managing context size
    - In practice, I've included everything, which is not accurate
    - Install a very simple flush mechanics around 40k
      - This may cause issues with mid-context loss, but we can maybe save everything since the last confirmed commit
    - Investigate reporting diffs when confirming a new section (architect sidebar is for potential information)
      - The agent would be able to detect this and assemble an internal document
      - If this also detected the sidebar stuff, we could erase the recent context and then "return" with a we discussed this offline?
        - This would fix some of the issue with not using the sketchbook
- Moving back to story development
  - Recast all stories to new IR model
  - Compaction layer?
  - Potentially fill out the other stories
  - Validate json schema against updated IR
- Create new architect that utilises json schema to develop plot plans
  - Starting on trying to implement the recursive parliaments
- Recast scenescript to be engine based if possible
  - This and plot beats deserve a lot of investigation

Not Fully sure what I'm intending here
- Follow up agents to develop the specific prompts/input for later stages
  - Re-exploration of plot and scene development and direction
  - I believe I have an okay schema 
- Final architect+writer agent which are run tag-team to make the prose
  - The architect plans long-form plot beats and action
  - The writer then takes those plot beats and develops scenes showing them

# [SECONDARY] Identifying Specfinding
- TODO: me - Update this after finishing the above task
  - This still has some improvements which aren't captured above
- Create schema for writer agent
- Create writer agent (or prompt to produce writer agent from IR)
- Investigate alternative context assembly
  - Agent maintains an internal image of what the living document is
  - Sends that model in every request, no chat history
    - History would be useful for determining what did/didn't work
  - Updates model when the response indicates the user agrees with the ai's understanding, basically merges them together.
- Merge engines and generation rules as they are both plot involved
- Investigate adding AI critique agents
- Investigate "Writer Explorer" prompt to refine IR for scene/writer
  - Basically specfinding/writer.md
  - Beat Generation/Authorial tone and style/etc.
  - The beat generation would be combined with the extracted plot
- Investigate using sub-agents for deep-dives dueing specfinding

# Another Look at Long Term Planning
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
- Work on adding events to the prompt handling
- Split principles/etc. into explicit events (I don't have an explicit event system yet because everything is incorporated into principles/rules)
    1. Attach modules/skills for producing the specific narrative items
- Research phases for identifying potential themes, plots, etc.?
  - Also useful for naming/etal

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
