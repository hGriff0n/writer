
https://infiniteworlds.mywikis.wiki/wiki/How_Infinite_Worlds_works

# Tracks
- Specfinding Agents
- Longform Storydesign
- Interactive Storydesign
- Enforcing world fidelity and other rules
- RAG/Skills agent

# [EXTERNAL] Markdown2pdf
- Integrate with Obsidian to simplify export even more
- Create AI script for running prompt

- gemini 3 pretty decent coherence, gen whole story
  - overly sad and morose (should i add a bias towards enjoyment?)
  - didn't gen enough characters
    - include questions about population principles
  - have to explicitly say "stop at moment" to prevent full gen
    - how to decide when to split
  - time seems to be a bit fluid

# [PRIMARY] Migrate All Stories to New Approach
- `curse`
  - [~] designdoc
  - [ ] components
  - [ ] generation
  - [ ] author
- `reality`
  - [~] designdoc
  - [ ] generation
  - [ ] components
  - [ ] author
- **`thousand`**
  - [~] designdoc
  - [ ] generation
  - [ ] components
  - [ ] author
- **`mara`**
  - [~] designdoc
  - [ ] generation
  - [ ] components
  - [ ] author
- `level`
  - [~] designdoc
  - [ ] generation
  - [ ] components
  - [ ] author
- `late`
  - [ ] probably worth refreshing author
  - [ ] add rotating cast
  - [~] fix bloom deck and time scale
  - [ ] allow protagonist to "demonstrate" abilities long term (like football team)
- maybe one of the other story ideas

# [SECONDARY] Agentic Explorations
- Convert architect+writer into langchain agents (run tag-team to make the prose)
  - The architect plans long-form plot beats and action
  - The writer then takes those plot beats and develops scenes showing them
- Experiment with agentic discussion boards/parliaments for long-term plot extrapolation
  - No i have to create my own agent discussion board

# [TERTIARY] Experiment with other models?
- Update LlmEngine to auto-create rate limiters
  - Update configs to new tier 1
  - Update configs to Gemini
  - Add configs for ChatGPT, etc.
- Chat GPT 5.1 has good reviews
- Kimi K2
- Claude Sonnet

# Script Improvements
- Change to pydata structure (gemini support actually week)
- Integrate ContextManager for orchestrator
  - Migrate internals to langchain message aware
  - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Validate json schema against updated IR
  - Investigate reporting IR conversations in json, not pseudo-xml
- Is there anyway to track the constants/config files?

# Integrate Skill Framework to Reduce Context Size (esp in Specfinding)
- Implement agentic layer to dynamically load skill files
  - Work on the detection mechanisms for determining a load is needed
    - I don't think the current logic has enough to do that
  - https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/

# Other Agentic Explorations
- Implementing skill checks for `level`
- Initialization Agents
  - Specialized agents for running story initialization logic, esp characters
    - `late` has difficulties generating appropriate numbers
- Research Agent
  - Specialized agent to collect and structure information for story building
  - `curse` which requires explicit factual information
  - `thousand` which is using the facts as a generative base
  - `mara` which involves media canon
  - Incorporating research into world building: https://aistudio.google.com/app/prompts/15p_SWfuw-dzpym6oaOb69b9IUIcnY4wX
- Critique/Review agents
  - Specialized agents to critique the produced prose from adherence/pacing/emotion/etc
  - Theoretically, this would feedback to produce better output
  - Not sure how to detect issues from nuanced misunderstandings though
  - Implement into a Reflection agent
    - https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7
- RAG/World Lore agent
  - Add an agent to inject complete world state as needed to optimize context
  - Sort of similar issue with skills, how to identify and structure need finding
- Investigate using sub-agents for deep-dives during specfinding
- Would be a good idea to make an agent specifically for understanding this process and all of it's interactions (could be good for adding new features)

# Improving Long Term Planning and Plot Development
- Investigate other ways of exploring plot/story ideas
  - "Long terme what are the for aspects of the story and ask where would they be at this point simulate a couple options choosing various degrees of progression and then critique how those work out with the other goals. Can you integrate them with other intended narratives"
- Use repeated iteration to build longer novel-form scenes
  - https://aistudio.google.com/app/prompts/179Q4SpXMnAfGWaw_M8E_HiqobQcFIQ8g
- Identify the place of ending and the duration
  - https://aistudio.google.com/app/prompts/18QQT7o5XLqF_ZXpKnRp61ipzNSagsHYQ
  - Solution would seem to require multiple agents
- Incorporating allegory/commentary: https://aistudio.google.com/app/prompts/1xF53-8YVysnFdbZ88Q_1ueBlm5kyFt8p
- Research Papers
  - https://arxiv.org/html/2503.04844v1
  - https://github.com/narrative-first/narrative-context-protocol
  - https://subtxt.app/
  - https://dramatica.com/

# Improving Writer Creativity and Prose
- Investigate new writers that can provide more detailed long-form scenes

# General Research Into Story Telling Principles
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

# Migrating Chat Interface to ui/*
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
