# writer

Tested and developed against Gemini 2.5 Pro for the most part. I've done some initial chats with Gemini 3 and it seems to mostly work fine, but it is not fully tested.

lib/... agents are capable of connecting to a local LLM hosted with LM Studio. Currently using mistral 3.1 small but have LFM2, Lyra4b, Mythomax, Dolphin 24b, true abomination, and gemma3 also local.


## Workflow

1. `scripts/specfinding.py`

    This script operates a "spec finding" dialogue where you and the llm try to co-operatively define your story. The script uses several prompts to guide and synthesize the various generative elements that the writer uses to develop the story. This script is also constantly trying to pre-emptively manage the context window, although the current approach may require some snapshot/restarts.

    The process itself is a collection of 4 prompts that can be entered at any stage and resumed at any stage

    - `orchestrator.md` (_premise? => designdoc)
    
        Most of the core work in determining and settling on the idea and narrative theme are done in the orchestrator. This has a review mode that can allow for extracting an initial set from a draft essay, but otherwise this is just trying to nail down the fundamental langauge that defines the idea.

    - `scenegen.md` (designdoc => generation)

        The output of the orchestrator is then fed into the scenegen where we use those generative fundamentals to start defining rules for beat and scene generation. Where the orchestrator defined the forces that underlie the story, this is where we provide the rules and heuristics to marshall them into a multi-layered story.

    - `analyzer.md` (designdoc+generation => components)

        Once we've settled on the fundamental design and pacing rules, they get sent through a quick "analysis" pass to compact and merge them into the final tooling for the generative framework. This is taken to leave the initial "designdoc" as documentation for creative decisions.

    - `writerstyle.md` (generation => author)

        The other output of this process, is the "prose" prompt. Unlike the plot infrastructure, each story get's an individualized writer prompt, as the effort and abstraction to generalize these aspects would likely impair their actual ability to produce the desired effect.

    - `_story.yaml`

        This is a separate file that serves as the "story card", a common configuration place that all of the scripts look in for resolving bindings. This must be in the data folder

TODO: Add mode to run a "test" for additions?

2. `scripts/extract_from_draft`

    An optional script to allow for writing a semi-long freeform narrative describing the intended story. Basically word vomit as much as you can about the intended narrative, lore, and experience into a simple essay. You then pass it into this script and it'll run repeated passes over it to extract an initial version of the "designdoc" that you produce in specfinding. The intention is to then pass this to that with the `-r` flag as a way of kickstarting the spec process.

3. `writer`

    The meat of the generative loop. This script runs a chat conversation that flips between two llms - one for doing plot planning and world state maintenance and the other for writing the final prose. The repl supports several commands to suggest options for next steps and even allows for users to input their own string for this purpose. The architect model is called 1-2 times a beat, once to produce a list of planned beats, if necessary, and a second call to produce a scene plan for the planned beat. This scene includes a high-level breakdown of the desired sequence of events.

    The writer model is specifically directed to not treat it's prose as ending a chapter unless otherwise specified. This allows us to build a long scene by simply repeatedly calling the writer, stopping it after each event. These events also have proposed "word count budgets", which we double to get a usable amount of prose.

    After the writer is finished with the scene, we quickly move to compact all of the used scenery (ie. we toss out the "per-event" context from the writer model and "pretend" we simply asked for the final output). If at any point, the author wants to stop, they can simply enter `/exit` and the ongoing progress will be saved in `<story>/resume.json`