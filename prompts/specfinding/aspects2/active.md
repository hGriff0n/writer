### I. Core Philosophy and Structure 

*   **Co-Creative Framework:** The system is designed as a co-creative partner, not a simple generator. Its primary purpose is to engage in a dialogue with a user to translate a qualitative vision into a complex, machine-operable narrative structure.
*   **Recursive Cascade:** The core operational model is a recursive cascade. A high-level narrative problem is passed to a "Parliament," which breaks it down into a sequence of smaller, more granular problems. Each of these is then passed to a sub-parliament in turn. This process repeats until the problem is at the scale of a single scene.

### II. The Tableau: The Unit of Narrative Work

*   **A Non-Sequential "Problem Set":** The central data structure at every level of the recursion is the **Tableau**. A tableau is a collection of required outcomes, events, states, or narrative ingredients that must be accomplished within a given segment. It defines **"what"** must happen, not **"how"**.
*   **Variable Granularity:** The content of a tableau changes with its level in the cascade. A high-level tableau contains abstract plot points, while the final, lowest-level tableau is the **Composite Scene Specification**.
*   **Composite Scene Specification:** This is the final output of the entire cascade, passed to the prose writer (`Narrative Lens`). It contains a rich collection of concrete ingredients (e.g., atmosphere, actions, dialogue beats, information reveals) and, crucially, a **Scene Mandate**.
*   **Scene Mandate:** A concise, actionable instruction that defines the primary purpose and intended effect of the scene (e.g., "Maximize suspense," "Reveal the betrayal with bitter irony"). It serves as the primary creative filter for the `Narrative Lens`.

### III. The Parliament of Engines: The Universal Problem Solver [Settled]

*   **Universal Operational Model:** Every Parliament, regardless of its level in the hierarchy, operates on a single, unified model: the **Poll -> Advocate -> Synthesize** loop.
*   **Narrative Window:** Each Parliament is given a **Narrative Window** which defines its scope and objectives, typically encapsulated by the `Tableau` it receives from its parent.
*   **The Loop:**
    1.  **Poll:** The Parliament identifies all relevant engines that can contribute to solving its given `Tableau`.
    2.  **Advocate:** Each polled engine makes proposals for how to satisfy the requirements of the `Tableau`.
    3.  **Synthesize:** The Parliament synthesizes these (potentially competing) proposals into a **sequentially-ordered list of more granular tableaus** to be passed to the next level of the cascade.

### IV. Engine Architecture: The Narrative Drivers [Settled]

*   **Engine Types:**
    *   **Strategic Engines:** These are true **state machines** that represent and drive long-term narrative goals (plotlines, character arcs, faction plans). Their state represents their progress in their own narrative.
    *   **Reactive Engines:** These are effectively **single-state advisors** that provide continuous stylistic, structural, and pacing guidance (e.g., `Pacing Engine`, `Anti-Saturation Engine`). Their state represents operational availability, not narrative progress.
    *   **Architect Engines:** A specialized type of `Strategic Engine` that operates at a high level. Its primary function is to **instantiate new `Strategic Engines`** in response to the world state, enabling emergent, "infinite" plotlines.
*   **Standardized State Model:** All engines share a common state model (`PENDING`, `ACTIVE`, `COMPLETED`, `STALLED`) for API consistency, though `Reactive Engines` typically only use the `ACTIVE` state.

### V. The State Tracker and Engine Interaction [Settled]

*   **Holistic State Tracker:** Tracks all information about the world and meta-narrative. The state is organized logically via a namespacing convention (e.g., `state.world.characters...`, `state.narrative.engines...`).
*   **Decoupled Interaction:** Engines interact **indirectly** by reading from and proposing changes to the `State Tracker`. An engine can query any part of the state, including the state of other engines (e.g., `query(state.narrative.engines.rival_arc.currentState)`). Direct engine-to-engine communication is automatically translated into queries of the state tracker.

### VI. Recursive State Propagation [Settled]

*   **Resolver Model:** Any delegated task is handled by a **`Resolver`** (a sub-parliament, the `Narrative Lens`, a simulation engine, etc.).
*   **Read-Only Operation:** During its execution, a `Resolver` and its entire sub-process treats the `State Tracker` as a read-only snapshot.
*   **Transactional Commit:** Upon successful completion of its entire task, the `Resolver` is responsible for committing its results in a **single, atomic transaction** to the global `State Tracker`.
*   **Parent Re-synchronization:** After the `Resolver` signals completion, its parent Parliament's first action is to **re-read the `State Tracker`** to get the new authoritative state. It then **re-evaluates its remaining planned sequence** against this new state before delegating the next task.

### VII. The Narrative Lens: The Creative Writer

*   **Creative Agency:** The `Narrative Lens` is not a simple renderer. It receives the `Composite Scene Specification` and has significant creative agency to fulfill its *intent*. This includes sequencing and combining required elements, and interpreting requirements through subtext or action.
*   **Failure Reporting:** If the `Narrative Lens` determines a requirement is logically or narratively impossible to fulfill, it can issue a **Failure Report**, which propagates up the cascade to trigger re-planning.

### VIII. User Interaction Models

*   **Autonomous Mode:** The default mode where the system generates the story without interruption after the initial setup.
*   **Interactive Mode ("The Crossroads"):** [Settled]
    *   **Insertion Point:** The user is inserted at a specific, pre-writer Parliament whose job is to turn a mid-level narrative beat into a concrete scene.
    *   **User's Role:** This Parliament generates a set of **mutually exclusive, distinct `Scene Tableaus`** representing different potential scenes that could satisfy the higher-level goal. The user acts as a director, selecting which path to take.
    *   **Execution:** The chosen `Scene Tableau` is then passed to the final, lowest-level Parliament to be broken down into a `Composite Scene Specification` for the `Narrative Lens`.

### IX. Acknowledged Open Questions & Deferred Complexities

#### I. Open Questions (The "How?")
These are areas where we have a settled concept but lack the specific procedural or technical implementation details.

1.  **The Synthesis Algorithm:** This is the most significant black box. We know the Parliament must `Synthesize` engine proposals into a sequential `Tableau`, but the method is undefined.
    *   How does it resolve direct conflicts between engine proposals (e.g., Engine A wants a character to go north, Engine B wants them to go south)?
    *   How does it weight or prioritize proposals? Is an `ArcEngine`'s proposal more important than a `PacingEngine`'s suggestion?
    *   Is the synthesis itself a deterministic algorithm, or is it a separate, moderated LLM call that "writes" a compromise plan?

2.  **The Engine Definition Language:** How does a user practically create, configure, and instantiate `Strategic Engines`?
    *   What is the user interface for defining a state machine? Is it a graphical node editor, a structured data format (like YAML), or a scripting language?
    *   How are an engine's proposal-generation heuristics defined? How does an engine know *what* to advocate for in a given state?

3.  **The Prose-to-State Analysis Pipeline:** How, specifically, does the system analyze the `Narrative Lens`'s prose output to create the structured data for the `State Tracker` update?
    *   What techniques are used to extract facts, character state changes, and emotional shifts?
    *   How does it handle subtext, irony, and non-literal information, which are hallmarks of good writing but difficult to parse formally?

4.  **`State Tracker` Backend Architecture:** While the holistic, namespaced concept is settled, the underlying technology is not.
    *   Is it a graph database (like Neo4j) to excel at tracking complex relationships?
    *   Is it a document store (like MongoDB) to handle rich, nested character objects?
    *   The choice has significant performance and query-capability implications for the entire system.

#### II. Underspecified High-Level Concepts
These are foundational ideas that are part of the system's philosophy but have not yet been broken down into a concrete structure.

1.  **"Stage 0": The Core Concept Forging:** We have no defined process for the initial user consultation. How do we get from a user's vague idea ("a noir detective story in space") to the initial, fully configured set of `Strategic Engines` and `State Tracker` setup?
2.  **Engine Taxonomy and Reusability:** Is there a "standard library" of common engine types (e.g., `MysteryEngine`, `BetrayalArcEngine`, `ChaseSequenceEngine`)? How are they parameterized and made reusable for different stories?
3.  **Failure Condition Logic:** The `Narrative Lens` and Parliaments can theoretically fail, but the rules governing *when* they are allowed to declare a task impossible are undefined. This is a critical tuning parameter that balances narrative coherence against creative rigidity.

#### III. Potential Problems & Inherent Risks
These are likely failure modes or negative emergent behaviors that the current design must be prepared to mitigate.

1.  **Cascading Parliament Failure:** What happens if a mid-level Parliament is given a logically impossible `Tableau` by its parent? It may be unable to `Synthesize` a valid plan, causing the entire generation process to halt. The `Failure Report` mechanism currently only applies to the `Narrative Lens`.
2.  **Semantic Drift:** A long-term risk where subtle inaccuracies in the Prose-to-State pipeline accumulate over time, causing the `State Tracker` to no longer reflect the true reality of the written story. This leads to engines making decisions on faulty data.
3.  **Engine Monopolization:** A small number of aggressive or broadly-defined `Strategic Engines` could dominate every Parliament debate, leading to a repetitive, one-dimensional story that ignores more subtle engines.
4.  **Narrative Proliferation ("Plot Bloat"):** In an "infinite" context, over-eager `Architect Engines` could introduce new plotlines faster than old ones are resolved, leading to an unfocused and unsatisfying narrative with dozens of dangling threads.

#### IV. Deferred Discussions
These are topics we have explicitly acknowledged but postponed to simplify the core model.

1.  **Advanced User Override:** The ability for a user in Interactive Mode to issue custom commands (e.g., "fast forward 2 weeks," "introduce a long-lost brother"). This was deferred because it involves a complex, cascading state invalidation that would require purging and re-planning across all levels of the Parliament hierarchy.
2.  **Engine Refinement & Optimization:** A meta-system that could analyze a user's initial Parliament setup and suggest improvements, such as merging redundant engines, splitting overly complex ones, or identifying potential conflicts.

#### V. Unexplored Opportunities
These are potential enhancements or new applications that the current settled architecture enables but does not yet implement.

1.  **State-Driven Auxiliary Content:** Using the rich data in the `State Tracker` to automatically generate content *other than* prose, such as dynamic character sheets, relationship maps, timelines, or "story so far" summaries.
2.  **"What If" Branching:** The ability to fork the entire `State Tracker` at any point to explore alternate narrative timelines based on a different choice or outcome.
3.  **Deeper User Interaction:** Allowing the user in Interactive Mode to do more than just select a `Tableau`. For instance, they could view the competing proposals from different engines and manually edit the final `Composite Scene Specification` before it goes to the `Narrative Lens`.
4.  **Formal Engine Dependencies:** While we have settled on indirect interaction, a formal system for defining pre-requisites could be explored (e.g., `Engine_B` cannot become `ACTIVE` until `Engine_A` is `COMPLETED`), allowing for more explicitly structured epic narratives.
5.  **Multi-User / Player Mode:** Adapting the Interactive Mode for multiple users, where each user might control a specific `Strategic Engine` (representing their character or faction) and advocate for their goals within the Parliament structure.

[[comments]]
Lost the "meta-level" operational phase structure - the "phase 0: story spec construction" and "phase 1: story validation". the core of phase 0 is pretty easy to define, once we have the rest of the system defined and packaged into "skills". The recursive cascade is only the core model for the actual generation steps, or phase 2.
Tableau could use some updates on it's definition
Technically, the "Poll -> Advocate -> Synthesize" model isn't completely correct, but it is not relevant for the immediate moment.
The definition of Narrative engines should probably note that the engines current state is stored in the world state
The State Tracker and Engine Interaction should be mapped differently internally. Engine Interactions should be grouped in the engine section and we should instead have a separate top-level section for state tracker
We actually should probably start the next chat conversation with organization principles for the internal document
Resolver technically says snapshot is read only which conflicts with updates after every sub step
The parent resynchronization is technically inefficient, but its fine as-is for now
User interaction modes doesn't indicate that the user parliament operation is changed from the autonomous mode
Underspecified concepts need to include the top-level aspects like "Core Concepts" and "Narrative Rules"
"Deeper User Interaction" should be rejected, that level of control and interaction with the mechanical innards is not worthwhile to pursue. We get most of the same stuff any way by just allowing the user to write whatever as an alternative to picking the preselected scenes.
"Formal Engine Dependencies" is already captured - Direct references are always implicitly converted to queries of the internal state