
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

-- Clean up everything

- gemini 3 pretty decent coherence, gen whole story
  - overly sad and morose
  - didn't gen enough characters
    - include questions about population principles
  - have to explicitly say "stop at moment" to prevent full gen
    - how to decide when to split
  - time seems to be a bit fluid

# [PRIME] Rewrite Specfinding From First Principles=
- Validate that writer.py works
  - scene generation
  - option selection
  - saving
  - resuming
- Apply world state updates to world state
  - maybe change prompt to report as deltas
- Integrate ContextManager for orchestrator
  - Migrate internals to langchain message aware
- Go through all code/prompt and clean up/organize
  - Incorporate names thing in writer.py initialization to prompt?

# [SECOND] Migrate All Stories to New Approach
- `late`
  - [ ] minor author update
- `curse`
  - [x] _premise
  - [ ] designdoc
  - [ ] generation
  - [ ] author
  - [ ] components
- `reality`
  - [x] _premise
  - [ ] designdoc
  - [ ] generation
  - [ ] author
  - [ ] components
- **`mara`**
  - [ ] _premise
  - [ ] designdoc
  - [ ] generation
  - [ ] author
  - [ ] components
- **`thousand`**
  - [ ] _premise
  - [ ] designdoc
  - [ ] generation
  - [ ] author
  - [ ] components
- `level`
  - [ ] _premise
  - [ ] designdoc
  - [ ] generation
  - [ ] author
  - [ ] components
- maybe one of the other story ideas

# Script Improvements
- Validate json schema against updated IR
  - Investigate reporting IR conversations in json, not pseudo-xml
- Is there anyway to track the constants/config files?

# Skills and Agents
- Implement agentic layer to dynamically load skill files
  - Work on the detection mechanisms for determining a load is needed
    - I don't think the current logic has enough to do that
  - https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/
- Investigate adding AI critique agents
- Investigate using sub-agents for deep-dives dueing specfinding
- Convert architect+writer into langchain agents (run tag-team to make the prose)
  - The architect plans long-form plot beats and action
  - The writer then takes those plot beats and develops scenes showing them
- Experiment with agentic discussion boards/parliaments for long-term plot extrapolation
  - No i have to create my own agent discussion board
- Investigate making a "plot critique" agent that could parse the output of the ai studio tests and present ways of adjusting the pacing
  - If I'm using a lot of time jumps, then it's probably a sign the steps are too small
  - Would also have to include specific comments about what I didn't like since I don't have any tools to indicate that directly in the conversation
- Would be a good idea to make an agent specifically for understanding this process and all of it's interactions (could be good for adding new features)
- Develop reviewer prompt into a very detailed analaysis and editor agent
  - Implement into a Reflection agent
    - https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7
- Implement an iterative agent with critique
  - Plan and Execute (an adaptation on the architect pattern)
- Develop a very basic prompt to test out tool calling
  - Using skills checks in agent.py
- Implement Basic RAG agent w/ vector store and World DB

# Another Look at Long Term Planning
- Update LlmEngine to auto-create rate limiters
  - Update configs to new tier 1
- Investigate new writers that can provide more detailed long-form scenes
- Investigate otherways of exploring plot/story ideas
- Investigate recasting plot planning to use parliament model
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
- Identify the place of ending and the duration
  - https://aistudio.google.com/app/prompts/18QQT7o5XLqF_ZXpKnRp61ipzNSagsHYQ
  - Solution would seem to require multiple agents
- https://arxiv.org/html/2503.04844v1
- https://github.com/narrative-first/narrative-context-protocol
- https://subtxt.app/
- https://dramatica.com/

# Building on the general principles
- Incorporating allegory/commentary: https://aistudio.google.com/app/prompts/1xF53-8YVysnFdbZ88Q_1ueBlm5kyFt8p
- Incorporating research into world building: https://aistudio.google.com/app/prompts/15p_SWfuw-dzpym6oaOb69b9IUIcnY4wX
- Research phases for identifying potential themes, plots, etc.?
  - Also useful for naming/etal
- Doing multiple things with one action
  - The scene doesn't just advance plot A, or relation B, or theme C, but all 3

# Optimizing Tokens
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
- Improve orchestrator for token efficiency: https://aistudio.google.com/app/prompts/1x87pQK4wpb3rNG_S9U9yXkAZbPXOAf9L
  - Integrate Bookmarking into Orchestrator (difficult)
    - https://aistudio.google.com/app/prompts/1tUzWztSyO2tq3neM4GOKfEBNvWgvvUtP
  - Investigate compaction layers for merging/reducing prompt bloat

# !!! Clean up the Presentation in CLI !!!
- I was thinking about the story bible though that's only in ai studio
- Rewrite LlmEngine in terms of `create_agent`??
  - Allows for dynamic model selection if needed

# Experiment With Basic Prompts
- What would happen if I just use a basic editor prompt
- Migrate the historical progression narrative to engines in curse

DSPy

Long Term:
  Local GPU
  API Usage
  Hosted GPU: https://vast.ai/

https://arxiv.org/pdf/2404.13919
