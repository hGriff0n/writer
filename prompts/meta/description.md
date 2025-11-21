# Active Document for Writing System

<!-- This document was produced after a laborious chat with gemini, roughly defining my current idea for how to assemble a generative writer -->
<!-- This still needs a lot of work with incorporating foreshadowing, research, and many other good authorial abilities -->
## Narrative Principles/Components

### Authorial Intent Document (AID)

This document is the foundational, purely qualitative specification of the story. It is primarily focused with totally and accurately representing and communicating the user's creative vision in all its nuance and expanse. Its sole purpose is to establish a deep, unambiguous, and shared understanding of the narrative's world, characters, themes, and principles between the user and the system. The AID is a living document during its creation, organized into several distinct sections:

*   **Thematic Core:** The story's logline, central dramatic questions, key thematic pillars, and desired tonal and stylistic directives.
*   **World Bible:** Prose descriptions of the setting, its history, cultures, factions, and key locations.
*   **Dramatis Personae:** Detailed psychological and narrative profiles for all significant characters, outlining their motivations, backstories, core conflicts, and intended arcs.
*   **Principles of Causality:** Natural language explanations of the world's governing logic—its physics, magic systems, rules of society, or technological laws. It defines "how things work" from a narrative perspective.
*   **Narrative Blueprint:** A high-level, prose-based outline of the intended plot structure, major story beats, turning points, and desired ending states.

### World State
A single, unified data store that serves as the dynamic source of truth for the entire narrative. It tracks everything from character stats, emotional states, and relationships to the internal states of narrative engines themselves. There are no hard architectural partitions; logical separation is handled by a namespacing convention (e.g., `state.world...`, `state.narrative...`). All updates to the World State are performed as single, atomic transactions to ensure consistency.

### Narrative Engines
Engines operate as specialized agents that advocate for specific outcomes, goals, or narrative textures. They are the primary drivers and shapers of the story but do not directly produce plot or story aspects by themselves.

#### State Machines
All engines are internally state machines. The simplest engines only have 3 stages - DORMANT, ACTIVE, FINISHED - but more complex engines can have complicated internal graphs which describe a complex plot as a series of sub-goals. Each state informs the advocacy of the engine in the parliament, changing the tone, actions, etc. (DORMANT and FINISHED engines are always silent).

Engine states are stored in the `World State` as part of the meta-narrative namespace. Engines can directly ask questions and reference other engines in their internal logic, but the reference is always silently translated into a query against the `World State`.

##### Fluid Engine Prominence:
This state machine is further complicated (theoretically) by requiring engines to dynamically adjust their advocacy based the current narrative frame. A **Thematic Engine** might be highly prominent in an early, high-level debate, while an **Atmosphere Engine** or **Tension Engine** becomes a dominant voice when the tableau is focused on a specific scene.

#### Engine Categories
Engines are broadly categorized by their function within the narrative generation process.

*   **Strategic Engines:** These are the core narrative drivers of the entire system, responsible for executing long-term, multi-stage plans, character arcs, or plotlines. The aggregate state of all Strategic Engines *is* the plot.
*   **Reactive Engines:** These engines do not have long-term goals but instead react to the current state and the immediate narrative context to propose actions, shape atmosphere, or enforce constraints.
*   **Architect Engines:** A specialized type of Strategic Engine whose primary function is to monitor the high-level narrative state. Under specific conditions, an Architect Engine can instantiate new Strategic Engines. These engines are specifically intended for generating emergent, long-term plotlines in "infinite" or episodic stories.

#### Potential Future Engines
*   **Rules Engine**: Legal council in the parliament, ensuring proposals are consistent with the world's established laws and logic
*   **Character Engines**: In character-driven stories, major characters may get an engine for directly simulating their internal "reasoning"
    *   **Beliefs:** What the character holds to be true about the world (which may be incorrect).
    *   **Desires:** Their short-term and long-term goals.
    *   **Intentions:** Their current plan of action to achieve a desire.
    *   **Emotional State:** A dynamic variable that influences their decision-making logic.
*   **Timeline Engine**: A strategic engine that ensures key mandated events (KMEs) occur at the correct time with the correct audience by subtly influencing the world and character positioning over long, medium, and short-term horizons
    1.  **Long-Range Foreshadowing:** When a KME is far in the future, this mode subtly influences the environment and information presented to characters to align the story's atmosphere with the inevitable future (e.g., generating ambient events, infusing scenes with a sense of dread).
    2.  **Mid-Range Funneling:** As a KME approaches, this mode ensures characters are in the right place at the right time. It achieves this by creating scenarios that logically motivate character movement, making the setup feel authentic rather than forced.
    3.  **Short-Range Execution:** When a KME is imminent, this mode dictates the non-negotiable facts of the environment, ensuring fidelity to historical events or established lore while leaving the character's personal experience as the primary focus.

### Tableau
The central data structure representing a narrative mandate at every level of the recursion. A tableau is a collection of required outcomes, events, states, or narrative ingredients that must be accomplished within a given segment. It defines *what* the higher-level Parliament wants to happen, not *how* (or even *if*) it should be accomplished.

#### Mandate
A `Tableau` combined with a rich `Intent Frame` that is provided as input to a parliament debate. The mandate basically defines the limits and goals of the parliament's debate, constraining it towards a specific goal

##### Intent Frame
Basically a mission briefing from the higher-level context, providing strategic context, positional data, and direction for the tableau planning.
*   **Parental Intent:** Why the parental parliament generated this tableau, wha is the intent of doing these things now?
*   **Positional Data:** The explicit position of the tableau within the parent's planned sequence (ie. beat 2 of 4)
*   **Directives:** Dynamic set of instructions from the parent to guide the process.

### Parliament
A universal, fractal process engine that operates at all levels of the narrative cascade. Its function is formalized as a `Poll -> Advocate -> Synthesize` loop. At any given level, it receives a `Mandate` and synthesizes competing proposals from all relevant engines into a coherent, sequentially-ordered list of more granular Tableaus for the next level down.

#### GM/Conductor
The way the synthesizing process is run is by utilizing a specialized persona that acts as a director for a given parliamentary session, ensuring that the generated plan is a coherent, well-paced, and satisfying narrative segment.

*   **Structural Adherence:** Interpreting the story's current position relative to high-level structural context (ie. the `AID`). It must use this context to weight proposals and assemble tableaus.
*   **Pacing and Rhythm:** The flow of events is just as important as the events themselves. It must ensure that the narrative isn't just a relentless series of high-tension moments, but includes downtime, exposition, etc. as appropriate for the narrative rhythm and the story's intent.
*   **Conflict Resolution:** The final arbiter when engines propose mutually exclusive outcomes. It makes these decisions based on a variety of heuristics, including the "satisfying the `Mandate`", "adherining to structural/thematic goals", "maximizing long-term potential"
*   **Failure Triage:** When a subordinate resolver fails its generation, this is the stage responsible for the analysis - determining the reason for the failure and orchestrating the solution.

#### Parliamentary Deadlock

**A. Synthesis Failure:**
This occurs during the `Synthesize` phase when the `GM/Conductor` cannot form a valid plan.

*   **Condition 1: No Viable Proposals.** After the `Advocate` step, no single engine proposal or combination of proposals can satisfy the core objectives of the `Mandate`. Every potential path violates a hard constraint or fails to make progress.
*   **Condition 2: Mandate Scope Incoherence.** During the initial budgeting step, the `GM/Conductor` performs a preliminary "causal chain estimation" to gauge the inherent complexity of the task it has been given. The required actions are either too complex for the allotted narrative space or too simple (e.g., a budget of 5 for a one-step action).

**B. Iteration Failure:**
This occurs during the iterative generation loop after one or more child `Tableaus` have already been resolved.

*   **Condition: Plan Invalidation.** The outcome of a child `Tableau` (e.g., `Tableau_i`) alters the `World State` so drastically that the parent `Parliament`'s original `Mandate` is now impossible to fulfill with the remaining budget of `Tableaus`. The `GM`'s attempt to frame the debate for the next beat (`Tableau_i+1`) reveals that there are no valid moves left.

### Rules
The hard constraints, mathematical formulas, reusable procedures, and immutable logic of the world. They validate and populate proposed actions from the Parliament.

#### Proto-Rules (in the AID)
A Proto-Rule is a structured, prose-based specification of a logical process. It is created during `Specfinding` by the "Rules Architect" persona in collaboration with the user. These live in the **Principles of Causality** section of the AID. They are designed to be as clear as possible about the intended logic, using formats like pseudo-code, tables, and conditional statements to eliminate ambiguity.

*   **Narrative Intent:** A plain-language description of the rule's purpose in the story. Why does it exist? What feeling or theme does it support?
*   **Trigger:** The specific event or condition that causes this rule to be evaluated.
*   **Logic:** The core of the Proto-Rule, expressed in a structured but non-executable format. This can include:
    *   **Conditional blocks:** `IF condition THEN outcome ELSE other_outcome`
    *   **Formulas:** `new_value = character.attribute * modifier`
    *   **Data Tables:** Data tables used for lookups
    *   **State Changes:** `SET character.state = "Transformed"`
    *   **Schema Snippets:** A description of the data structures the rule operates on.
*   **Outcome:** The narrative or state change that results from the logic.

#### Formal Rules (in the Runtime)
Formal Rules are the machine-executable translation of Proto-Rules. The `Instrumentation Phase` is responsible for this conversion. The dialogue during this phase would be highly targeted

We will now have only **three** top-level rule types: `Data Table`, `Schema`, and `Logic`.

##### 1. Logic Rule (Unified Definition)

A `Logic Rule` can be defined in one of two, mutually exclusive, ways: a "shorthand" form for direct mutations, or a "longhand" form for complex, multi-step processes.

*   **`RuleID`:** A unique identifier.
*   **`SourceProtoRuleID`:** A reference to the `AID`.
*   **`Type`:** Always `Logic`.
*   **`TriggerCondition` (Optional):** The condition for execution.

###### **Direct Mutation or Assignment**
Used when the rule's entire purpose is to set a single value.

*   **`TargetPath`:** The `World State` path to be modified.
*   **`ValueExpression`:** A string containing the value or formula to be assigned. This can be a literal constant or an expression that references other parts of the `World State` or event data.

###### **Multi-Step Processes**
Used for any rule that requires temporary variables, multiple steps, or complex actions.

*   **`ExecutionLogic`:** An ordered array of natural language (or pseudocode) statements
    *   Kinds of Statements:
        *   Variable Assignment
        *   Table queries/lookup
        *   World State mutation

#### 2. Data Table Rule
This rule type stores structured lookup data. The LLM can be instructed to query this data as part of a `Logic Rule`'s execution.

*   **`RuleID`:** `lookups.crafting.potion_recipes`
*   **`SourceProtoRuleID`:** Reference to the prose in the `AID`.
*   **`Type`:** `DataTable`
*   **`Content`:** A string containing the data, typically in a structured, LLM-friendly format like Markdown.
    ```markdown
    | Potion Name      | Ingredient A   | Ingredient B    | Effect              |
    |------------------|----------------|-----------------|---------------------|
    | Healing Draught  | Rivercress     | Sunpetal        | Restore 20 HP       |
    | Mana Potion      | Moondrop       | Ectoplasm       | Restore 15 Mana     |
    ```

#### 3. Schema Rule
This rule defines the structure of a piece of data within the `World State`. This is used by the system to validate state mutations and provides context to the LLM about how data is organized.

*   **`RuleID`:** `schema.character.inventory_item`
*   **`SourceProtoRuleID`:** Reference to the prose in the `AID`.
*   **`Type`:** `Schema`
*   **`TargetPath`:** The `World State` path this schema applies to (e.g., `state.world.characters.*.inventory.*`).
*   **`SchemaDefinition`:** A YAML schema object defining the data structure.
    ```yaml
    type: "object"
    properties:
        name: { "type": "string" },
        quantity: { "type": "integer", "minimum": 1 },
        description: { "type": "string" }
    required: ["name", "quantity"]
    ```

### Narrative Lens
The prose generation module, tuned with the desired POV, voice, and style from the Core Concepts. It receives a final, atomic `Composite Scene Specification` which includes a rich collection of concrete ingredients (e.g., atmosphere, actions, dialogue beats, information reveals) along with specific and actionable scene directions (e.g., "Maximize suspense," "Convey loss through introspection"). This mandate provides a concrete objective, guiding the creative rendition of the scene to fulfill the specific intent of the Parliament.


1.  **Holistic Ingestion & Goal Identification:** The first step is to read the entire CSS to understand the scene's core purpose. It identifies the `SceneObjective` as the primary goal and the `InitialCharacterStates` vs. `TargetCharacterStates` as the required emotional and narrative transformation. This gives the Lens a clear understanding of the scene's beginning, end, and purpose.

2.  **POV Embodiment:** The Lens "becomes" the `PointOfViewCharacterID`. It loads their `InitialCharacterState` (emotions, knowledge, goals) as its primary filter for perception and description. The `Focus` directive from the CSS determines how tightly it must adhere to this perspective—whether it can describe things the character doesn't notice or if it's a strict, limited-third-person view.

3.  **Staging and Atmosphere:** The Lens uses the `LocationID`, `CharactersPresent`, and `Atmosphere` directives to establish the physical and sensory context of the scene. It begins by "painting the set," grounding the reader before any significant action occurs.

4.  **Beat Sequencing and In-filling:** This is the core of the "assembly" process.
    *   **Pillar Construction:** The `MandatoryBeats` are treated as the non-negotiable structural pillars of the scene. The Lens first arranges these beats in the most logical and dramatically effective sequence that respects causality.
    *   **Connective Tissue:** The Lens then writes the "in-between" material. This is where its creative agency is highest. It generates the dialogue, actions, internal thoughts, and descriptions necessary to move the characters from one mandatory beat to the next. For example, if two beats are "A presents evidence" and "B confesses," the Lens must generate the intervening argument, denial, and emotional breakdown.
    *   **Pacing and Rhythm:** The `Pacing` directive governs the density and speed of this connective writing. A "fast" pace means shorter sentences, more action, and less introspection between beats. A "slow" pace allows for more description, internal monologue, and nuanced interaction.

5.  **Continuous Trajectory Alignment:** Throughout the writing process, the Lens constantly self-corrects to ensure the scene is moving from the `InitialCharacterStates` toward the `TargetCharacterStates`. If the POV character is supposed to end the scene feeling "vindicated but troubled," the Lens must sow the seeds of that trouble during the confrontation, not just tack it on at the end.

6.  **Thematic Resonance:** The `Theme` directives act as a final stylistic filter. The Lens will subtly adjust word choice, metaphors, and character introspection to ensure the scene reinforces the specified themes. If a theme is "the cost of secrets," the prose might linger on the physical toll the confession takes on a character.

#### Failure Conditions

##### 1. Logical Contradiction

The check scans the CSS for internal requirements that are mutually exclusive.

*   **Condition:** Two or more fields within the CSS mandate logically incompatible states or events.
*   **Examples:**
    *   `TargetCharacterStates` requires Character A to end the scene "unaware of the betrayal," while a `MandatoryBeat` requires "Character B confesses the betrayal to A."
    *   `CharactersPresent` lists only "char_A" and "char_B," but a `MandatoryBeat` specifies an action for "char_C."
    *   The `SceneObjective` is "A escapes the prison," but a `TargetCharacterState` for A is "imprisoned and hopeless."

##### 2. Narrative Infeasibility (Path-to-Target Failure)

This is a more nuanced check for causal and emotional gaps. It isn't about logical contradiction, but about the lack of a plausible path from the starting state to the target state given the mandated events.

*   **Condition:** The required transformation in a character's emotional state or knowledge is too vast or complex to be plausibly achieved by the provided `MandatoryBeats`.
*   **Methodology:** The Lens calculates a "narrative delta" between the `InitialCharacterStates` and `TargetCharacterStates`. It then assesses whether the `MandatoryBeats` provide sufficient causal and emotional justification to bridge that delta.
*   **Examples:**
    *   **Initial State:** Character A is "consumed by grief over their brother's death."
    *   **Target State:** Character A is "joyful and optimistic."
    *   **Mandatory Beats:** "A buys bread at the market."
    *   **Failure:** The beat is insufficient to justify the massive emotional transformation. The path is infeasible.

##### 3. Character Principle Violation

This check ensures that a character's actions are consistent with their fundamental, established nature. This requires the `Narrative Lens` to have read-only access to the `Dramatis Personae` section of the `AID`.

*   **Condition:** A `MandatoryBeat` or `SceneObjective` requires a character to perform an action that flagrantly violates a core, defining principle of their established personality without any precipitating justification within the scene itself.
*   **Examples:**
    *   A character whose `Dramatis Personae` defines them as a "devout pacifist who has never harmed a soul" is mandated to "brutally murder the guard in cold blood."
    *   A character defined as "illiterate and terrified of magic" is mandated to "decipher the ancient magical tome."
*   **Nuance:** This is not to say characters cannot change or act "out of character." The failure occurs when the CSS mandates such a drastic break *without providing the context or events within the scene* that would precipitate it. The failure report would essentially state, "This action requires more setup than is provided for in this scene."

##### 4. Continuity Violation
A `Composite Scene Specification` (CSS) must represent a single, continuous thread of narrative action or theme. It fails if it forces a break in this continuity *without a specific directorial mandate to do so*.

*   **Heuristics for Failure:**
    *   **Un-unified Action:** The `MandatoryBeats` describe events in multiple locations or with different sets of characters *without* a unifying directive in the CSS (e.g., `Directive: { Type: "Structure", Instruction: "Cross-cut between A's fight and B's escape" }`). Without this directive, it's two scenes masquerading as one.
    *   **Internal Time Jumps:** A significant passage of time is mandated *between* `MandatoryBeats`. A time jump that establishes the scene's starting point is acceptable, but a CSS ordering "A pleads with the king" and then "Two days later, the king gives his answer" is a failure. This constitutes two distinct scenes.
    *   **Montage without Mandate:** The CSS implies a montage (e.g., a series of short, disconnected events over time) without an explicit `Directive: { Type: "Structure", Instruction: "Montage showing the hero's training" }`.


#### Composite Scene Specification (CSS)
The creative brief that is provided to the final writer agents after resolution by the parliaments. This brief contains the basic building blocks for a specific subscene of narrative prose, but it is the writer's responsibility to decide how those aspects are assembled and represented for the final story. The writer is not required to include all items in the brief.

```yaml
# Composite Scene Specification

# --- Core Narrative Mandate ---
# Derived directly from the parent Tableau's primary objective.
# This is the scene's reason for existing.
SceneObjective: "Character A confronts Character B about the stolen artifact, forcing a confession."

# --- Staging & POV ---
# Defines the camera and the stage.
PointOfViewCharacterID: "char_A"
LocationID: "loc_docks_warehouse"
CharactersPresent:
  - "char_A"
  - "char_B"
  - "char_C" # (Observing from the rafters)

# --- Scene Trajectory ---
# Defines the required 'before' and 'after' states.
# This guides the Lens on the scene's emotional/narrative arc.
InitialStates:
  - ...

TargetStates:
  - ...

# --- Key Beats & Information Flow ---
# These are non-negotiable narrative events or reveals.
# They are not dialogue scripts, but instructions on what must happen.
KeyBeats:
  - Type: ...
    Description: ...

# --- Directorial & Thematic Guidance ---
# This is where Reactive Engine advocacy becomes explicit instruction.
# Guides the *how* of the writing.
Directives:
  - Type: ...
    Instruction: ...
```

## Systemic Process

The system consists of three broadly independent processes that operate in sequence to develop the final story. The first two processes, specfinding and preview, are primarily concerned with ensuring fidelity and understanding of the authorial intent, with the human author highly involved with explaining, exploring, and defining their story world and judging initial test outputs.

The final generation process is primarily autonomous, outside of specific interactive stories which allow for a human player to "help" with story direction.

### **Specfinding**
This foundational stage is an interactive, co-creative dialogue between the user and the system, designed to produce the **Authorial Intent Document (AID)**.

*   **Process:**
    1.  **Holistic Dialogue:** The process is an open-ended conversation, with the system adopting various **Specialist Personas** (e.g., "Worldbuilder," "Character Profiler") to guide deep dives into specific aspects of the story.
    2.  **Silent Synthesis:** Throughout the conversation, the system maintains a private **Internal Model** of the narrative. It continuously and silently analyzes and critiques this model to identify patterns, infer consequences, detect inconsistencies, and discover opportunities for narrative depth.
    3.  **Proactive Proposal:** When the system's silent synthesis yields a high-confidence insight, it proactively proposes a new or revised prose entry for the AID. This could be a new character trait, a newly identified world principle, or a structural reorganization of existing entries (e.g., splitting or merging concepts for clarity).
    4.  **Collaborative Refinement:** The user and system then collaboratively refine and optimize the proposed text. Once the user explicitly agrees, the entry is formalized and committed to the official AID.

### **Instrumentation**
This stage serves as a bridge between the qualitative story document and the mechanical generation stages. It uses the same `Exploratory Dialogue` process as the `Specfinding`, just focusing on identifying the mechanical implementations of the AID.

*   The system adopts a "Systems Engineer" persona to parse the AID and then engages in the same process of collaborative dialogue as during specfinding.
*   The system is focused on translating the AID into the mechanical definitions and components that are used by the generation processes. This translation is not guaranteed to be a straightforward process and specific narrative processes may be better implemented by splitting into multiple components (or merging into a single one).

The output of the Instrumentation phase is the complete initial World State for the Generation phase.
    *   **The Initial World State:** Character sheets, world facts, etc.
    *   **Engine Definitions & Initial States:** All Strategic and Reactive engines required to tell the story are instantiated with their starting goals and internal state machines primed.
    *   **The Genesis Mandate:** A single, top-level Tableau that serves as the starting point for the entire Generation process. This is the machine-executable translation of the AID's Narrative Blueprint.

### **Preview/Validation**
Runs an abbreviated, high-level simulation of the entire narrative to provide a "narrative sketch" before committing to detailed generation. This preview is designed to give the user a clear sense of the story's intended emotional and thematic texture, not just its plot.

1.  **High-Level Simulation:** The Conductor and Parliament run a fast, abstracted simulation of the entire story arc.
2.  **Keyframe Tableau Generation:** Instead of a dry plot summary, the output is a **"Keyframe Summary"**—a sequence of high-level tableaus. Each "Keyframe" is a rich description of a pivotal narrative segment, outlining its core events, intended emotional impact, and key thematic relevance, complete with tags (e.g., "Tension," "Romance") to signify its texture.
3.  **User Validation:** The user reviews this emotional blueprint to confirm the story is heading in the desired direction. They can adjust the Core Concepts or engine configurations and re-run the preview until they are satisfied with the proposed narrative arc.

### **Generation**
The system operates as a unified, recursive cycle driven by the Universal Parliament. The process is consistent from the highest level of story structure down to the finest details of a single scene.

#### Procedure
The process is a cascade where each level is given a "problem" (a tableau of required events/states) and its "solution" is a more granular set of sub-tableaus for the next level to solve.

1.  A Parliament at Level `N` (`P_N`) is given a `Mandate`/`Tableau`
    * Upon receiving its `Mandate`, the `GM/Conductor` immediately performs a high-level structural analysis *before* polling any engines for specific actions.
    *   **Determine Narrative Budget:** Based on the `Mandate`'s narrative function and pacing directives, the `GM` determines the number of child `Tableaus` (`N`) required.
    *   **Create the Plan Skeleton:** The `GM` creates an internal "plan skeleton" which is essentially a list of `N` empty slots. This skeleton represents the budgeted narrative windows. It does *not* yet contain any specific content.
2.  **Poll:** `P_N` now iterates through its plan skeleton, from `i=1 to N`, polling all relevant `Strategic` and `Reactive` engines for each slot
    *   **A. Frame the Current Debate:** The `GM` sets the context for the *immediate* next beat. It considers the overall parent `Mandate`, the results of the `i-1` completed child `Tableaus`, and the current `World State`. It formulates a specific, constrained question.
    *   **B. Poll & Advocate (Scoped to the Window):** The `GM` polls the engines. The engines, aware of the constrained context, advocate for proposals that fit *only within this single narrative window*.
    *   **C. Synthesize & Select (For the Current Window):** The `GM` aggregates, scores, and selects the winning proposal for this specific beat.
    *   **D. Intent Abstraction & Delegation:**
        1.  The `GM` synthesizes the winning proposal into a concrete child `Tableau` (e.g., `Tableau_i: "Launch the primary assault on the main gate."`).
        2.  This `Tableau` is delegated to a resolver (a sub-parliament `P_N+1` or the `Narrative Lens`).
    *   **E. Await Resolution & Re-Synchronize:** The `Parliament` pauses its loop and waits for the resolver to complete its task. Upon completion, the resolver commits its changes, and the `World State` is updated. `P_N` is now synchronized with the absolute latest reality of the story.
    *   **The loop then continues to the next iteration (`i+1`), inherently using the newly updated `World State` to frame the next debate.**

3.  **Advocate:** Engines advocate for specific outcomes, events, or character actions that would both satisfy the `Tableau` and advance their own internal goals.
4.  **Synthesize:** `P_N` synthesizes these potentially competing proposals into a sequentially-ordered list of more granular Tableaus.
    1.  **Proposal Aggregation and Option Forming:** The Conductor's first task is to process this raw input into viable, high-level "options."
        - Filter: Discard any proposal that directly violates the Tableau's Constraints.
        - Group: Cluster complementary or synergistic proposals.
        - Conflict Identification: Identify proposal groups that are mutually exclusive
    2.  **Directive-Guided Option Forming:** The Conductor aggregates engine proposals into viable options, actively prioritizing clusters that align with the directives in the `Intent Frame` (e.g., `Prioritize Arc: [X]`).
    3.  **Scoring and Selection:** Each option is scored by computing a total weighted score from the following judges
        - **Mandate Fulfillment Score (Weight: High):** How completely does this option satisfy the Tableau's Objectives? An option that achieves 3 out of 4 objectives will score higher than one that only achieves 1.
        - **Directive Alignment Score (Weight: High)**: This directly measures how well a proposed option adheres to the Intent Frame's Inherited Directives. This is the primary measure of how well the option executes the parent's intent.
        - **Continuity Score (Weight: Medium):** This measures how logically and naturally the proposed Tableau follows from the immediately preceding state of the world. It evaluates factors like character location, emotional state, and immediate causality. A high score means the transition is smooth and believable.
        - **Narrative Momentum Score (Weight: Medium):** A critical counterbalance to the Continuity Score, this is explicitly intended to scene jumps/PoV switches/etc to be introduced to the narrative and ensure the story keeps developing.
        - **Scope Complexity Score (Weight: A Penalty/Bonus Modifier):** The Conductor calculates a complexity value for each option based on the number of major state changes, character actions, and objectives it contains. This value is then compared to a Target Pacing value in the Intent Frame, prioritizing options that best meet the target pace.
        - **Engine Bid Strength Score (Weight: Variable):** How strongly are the contributing engines advocating for this? This is modified by the Prominence Directives. An amplified engine's bid counts for more.
        - **Narrative Efficiency Score (Weight: Low):** Does this option also advance other, non-mandated engine goals in a positive way, or does it create unnecessary future complications?
    4.  **Plan Sequencing:**
        *   In Autonomous Mode:
            - The highest-scoring option is chosen and added to the tableau sequence
            - The GM re-evaluates the Mandate to identify remaining objectives and restarts the debate with the new context
        *   In Interactive Mode:
            - The 2-3 highest scoring, ideally mutually exclusive, options are selected. These options are framed as choices of intent or action for the current point-of-view character, or as proposals to switch to a different character's storyline
            - Brief user facing summaries of the actions are presented to the user, who selects one as the "canon" action
    5.  **Intent Abstraction:** After the a winning "option" is selected, the Constructor then builds the Tableau for appending to it's plan.
        1. Identify the Narrative Verbs: The Conductor scans the core actions and state changes within the selected proposals. It looks for keywords that represent the fundamental action of the scene or sequence.
        2. Identify the Core Subject and Object: It then identifies the primary actors (Subject) and the primary thing being acted upon (Object) in the proposals.
        3. Determine the Narrative Function: The Conductor classifies the purpose of this cluster of actions based on their combined effect, using a predefined set of narrative functions. Is this cluster primarily...
            - An Inciting Event? (Introduces the central conflict)
            - A Reversal? (Subverts expectations or turns the plot in a new direction)
            - A Revelation? (Provides critical information)
            - A Confrontation? (Brings opposing forces into direct conflict)
            - A Rising Action Beat? (Builds tension and moves the plot forward)
            - A Falling Action Beat? (Deals with the aftermath of a climax)
            - A Character Beat? (Focuses on internal development or relationships)
        4. Synthesize the Objective String: After identifying the narrative verbs, subjects/objects, and the overall narrative function, the Conductor performs a Natural Language Synthesis step.
        5. The rest of the Intent frame is built out of
            - Inheritance from parent `Mandate` (plus restrictions from previous tableaus)
            - Synthesis from the selected `Engines`
            - Narrative Logic and position
5.  `P_N` delegates the next task from the `Tableau` plan to a `Resolver` (which could be a sub-parliament `P_N+1`, the `Narrative Lens`, or an off-screen simulation engine).
    1.  If the `Resolver` is a sub-parliament, then we start over this recursive process, passing in the delegated `Tableau` to the sub-parliament as the new `Mandate`.
    2.  If the `Resolver` is a `Narrative Lens`, then we convert the `Tableau` into a **Composite Scene Specification** and pass it off to the **Narrative Lens** for final prose production
        *   The Narrative Lens has complete creative agency to sequence, combine, omit, or interpret these ingredients to write the most compelling prose, fulfilling the *intent* of the tableau.
    3.  If the `Resolver` is an off-screen simulation engine, then the simulation engine runs with the delegated `Tableau` as input
6.  Upon completion, the `Resolver` commits all its resulting changes to the `Holistic World State` in a single, atomic transaction. This specifically includes `Failure Reports`, meta-narrative, and non-prose (ie. "off-screen") updates.
7.  `P_N` must then **re-synchronize** by reading the new, authoritative state from the `Holistic World State`. It then needs to re-evaluate **re-evaluate** its remaining planned sequence of `Tableaus` against this new state before delegating the next task in its sequence.
8.  If there are no remaining tableaus, then `P_N` returns to it's parent parliament
9.  Otherwise, we loop back to step 5

##### Narrative Lens Translation
This defines the process for translating a tableau into a CSS

1.  This final Tableau proposal contains two distinct components:
    *   **Narrative Intent:** A collection of desired beats, character states, and directorial guidance.
    *   **Stateful Outcomes:** A list of explicit, machine-readable `World State` mutations that must occur.
2.  The `GM/Conductor` synthesizes the **Narrative Intent** into the `Composite Scene Specification` and passes it to the `Narrative Lens`.
3.  Simultaneously, the `Parliament` process "holds" the **Stateful Outcomes**.
4.  The `Narrative Lens` generates the prose and returns it.
5.  The `Parliament` receives the prose. In a single atomic transaction, it commits both the returned prose *and* its held `Stateful Outcomes` to the `World State`.

##### Feedback & Adaptation (Backpropagation):
Example questions to ask when trying to re-evaluate the prior Tableau plan against the active world state.
*   **Validation:** Did the generated scene successfully fulfill its mandate? Is it consistent with established lore? Are proposed future events consistent with the events from this scene?
*   **Discovery:** Did any unexpected details or character traits emerge during the writing? Were all of the tableau's events completely satisfied by the lower generation?
*   **Recalibration:** The system must adapt to the active state of the generation and check if the remaining Tableau set is still valid. If the generation failed, didn't satisfy all events, or developed a new discovery that causes a plot hole or opens up a more interesting path, the **GM** and **Parliament** are forced to re-plan. This can take a variety of directions
    *   Redo the first tableau but with additional emphasis to avoid the problematic generation that caused the review
    *   Generate new tableaus that build on the new state, potentially ignoring some events/promises of its **Mandate**
    *   Fail up to the parent parliament (or the human user if at the top-level)

#### User Interaction Model
The system supports two distinct modes of operation, allowing the user to choose their level of creative involvement.

*   **Autonomous Mode:** The system generates the entire narrative without user input, following the core generation cycle from start to finish.
*   **Interactive Mode ("Crossroads"):** This mode inserts the user at specific decision points in the narrative cascade. At a "Crossroads" moment, the Parliament responsible for scene selection, instead of choosing a single path forward, generates a set of mutually exclusive `Tableaus` for the next possible scene. The user is presented with 1-2 sentence summaries of these choices and selects one, which is then passed down the cascade for detailed specification and writing. This provides meaningful plot agency at the scene level without disrupting the system's long-term planning.

TODO: Technically, the user is presented with choices for what action to take from the current state, with the next scene describing the outcomes of that choice. In multi-plot/multi-character stories, this can include switching to a different PoV.

## Et Cetera

### Open Questions

#### Underspecified High-Level Concepts
1.  **The Instrumentation Phase:** While the process is defined, the AI's capability to translate complex prose into formal Rules and Engine logic is the system's most significant technical challenge and remains a conceptual "black box."
2.  **Specialist Personas:** The specific capabilities, knowledge domains, and conversational strategies for each persona (e.g., "Rules Architect," "Worldbuilder") are undefined.

#### Potential Problems and Risks
1.  **Cascading Parliament Failure:** What happens if a mid-level Parliament is given a logically impossible `Tableau` by its parent? It may be unable to `Synthesize` a valid plan, causing the entire generation process to halt. The `Failure Report` mechanism currently only applies to the `Narrative Lens`.
2.  **Semantic Drift:** A long-term risk where subtle inaccuracies in the Prose-to-State pipeline accumulate over time, causing the `World State` to no longer reflect the true reality of the written story. This leads to engines making decisions on faulty data.
3.  **Plot Bloat:** In an "infinite" context, over-eager `Architect Engines` could introduce new plotlines faster than old ones are resolved, leading to an unfocused and unsatisfying narrative with dozens of dangling threads.
4.  **Loss of Thematic Cohesion:** If the application of `Core Concepts` is weak, `Engines` may pursue their goals in a way that is logically sound but tonally or thematically inconsistent with the established genre and story, breaking narrative immersion.
6.  **The Indirect Feedback Loop:** If the Instrumentation Phase misinterprets the AID, the user's only recourse is to modify the qualitative AID and re-run the process, which may be inefficient and frustrating.

#### Deferred Discussions
1.  **Advanced User Override:** The ability for a user in Interactive Mode to issue custom commands (e.g., "fast forward 2 weeks," "introduce a long-lost brother"). This was deferred because it involves a complex, cascading state invalidation that would require purging and re-planning across all levels of the Parliament hierarchy.
2.  **Engine Refinement & Optimization:** A meta-system that could analyze a user's initial Parliament setup and suggest improvements, such as merging redundant engines, splitting overly complex ones, or identifying potential conflicts.
*   **3. Conditional Logic in Rules:** The ability to use `IF/ELSE` blocks within a `Logic Rule`'s `ExecutionLogic`. This was deferred, with the noted risk that its absence forces rule fragmentation and increases trigger complexity, creating a significant point of architectural fragility.
*   **4. Pre-computation ("Complete Map") vs. Just-in-Time Generation:** An architectural choice between generating a full story skeleton upfront versus the current, more flexible just-in-time model.

#### New Unexplored Opportunities
1.  **State-Driven Auxiliary Content:** Using the rich data in the `World State` to automatically generate content *other than* prose, such as dynamic character sheets, relationship maps, timelines, or "story so far" summaries.
2. **"What If" Branching and Merging:** At any point, allow the user to fork the entire `World State` to explore an alternate timeline. A more advanced version could even attempt to "merge" a branch back into the main timeline, reconciling the divergent state changes.
3.  **Multi-User / Player Mode:** Adapting the Interactive Mode for multiple users, where each user might control a specific `Strategic Engine` (representing their character or faction) and advocate for their goals within the Parliament structure.
4.  **Dynamic Narrative Rule Injection:** Allow certain in-story events to dynamically alter the `Narrative Rules`. For example, a magical cataclysm could add a new rule to the physics engine, or a political treaty could introduce new global invariants that all factions must obey.
5.  **Metanarrative Analysis & Feedback:** Since the system has a structural understanding of its own narrative components, it could provide the user with real-time feedback on concepts like pacing (based on `Tableau` density), character screen time (based on `Engine` activity), or plot thread completion status.

#### Rejected Concepts
1.  **Deeper User Interaction:** Allowing the user in Interactive Mode to do more than just select a `Tableau`. For instance, they could view the competing proposals from different engines and manually edit the final `Composite Scene Specification` before it goes to the `Narrative Lens`.

[[comments]]
