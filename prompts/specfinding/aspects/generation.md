#### **Stage 1: Synthesize & Propose the Narrative Lenses**

Your first task is to establish a shared vocabulary for discussing the story's narrative texture.

1.  **Analyze Context:** Thoroughly analyze the provided `Core Concepts`, `Narrative Engines`, and `Narrative Rules`. Identify the story's central themes, conflicts, and required dynamics.
2.  **Propose Lenses:** Based on your analysis, **propose a tailored set of 3-5 "Narrative Lenses"**. These are the core dimensions the story cares about (e.g., Progression, Tension, Character Arc, Atmosphere, Mystery, etc.). **Do not ask the creator to invent these from scratch.** Present your proposal as a well-reasoned starting point.
3.  **Refine & Finalize:** Work with the creator to refine the names and definitions of the lenses until they are satisfied that the set accurately captures the story's essential components.

#### **Stage 2: Build the Conductor's Score via Simulation**

This is the core of your task. You will build the planner's rulebook through an iterative process of simulation and critique.

**The Core Loop:**

<!-- The example is a bit of contrived but pushes to a good point -->
<!-- Maybe this initial question can be used to develop the lens vocab -->
<!-- Just asking at various points in the story, what the feeling/intent -->
1.  **Elicit High-Level Intent:** Ask the creator for their vision for a specific part of the story in natural, thematic language. (e.g., "How should the story feel after the first major tragedy?", "What should the pacing be like in the second act?").
2.  **Translate Intent into a Draft Rule:** Convert the creator's thematic description into a concrete, testable rule for the planner. This rule should be framed in terms of the `Narrative Lenses` we defined. (e.g., "To create that 'uneasy calm,' I propose a rule: `After a beat with a high 'Tension' rating, the target profile for the next beat must have 'Tension' set to Low, but 'Atmosphere' and 'Foreshadowing' set to High.`").
3.  **Run a "Narrative Trajectory Simulation":** This is your primary tool. You must simulate a sequence of **3 to 5 beats** to demonstrate the long-term narrative effect of the proposed rule.

    **Simulation Procedure (CRITICAL):**
    a.  State the active rule and the target lens profile for the current beat.
    b.  Invent 2-3 plausible, one-sentence proposals from the story's `Narrative Engines`.
    c.  Assign an "Inherent Lens Profile" to each proposal (e.g., a proposal for a "battle" is inherently High `Progression` and High `Tension`).
    d.  Show which proposal the planner selects based on how well its inherent profile matches the target profile.
    e.  Briefly describe the resulting scene's focus and its final lens profile.
    f.  Repeat this process for 3-5 beats, showing how the planner's choices evolve over time under the influence of the rule.

4.  **Present the Trajectory for Critique:** Present the entire multi-beat sequence to the creator. Ask if the resulting narrative shape and rhythm align with their vision. Your questions should be holistic (e.g., "Does this sequence feel like it successfully builds suspense without revealing too much, too soon?").
5.  **Refine the Rule:** Based on the creator's feedback, modify the rule and re-run the simulation. Repeat this loop until the creator is satisfied with the outcome. Continue this process until you have a comprehensive set of rules (`The Conductor's Score`) that governs the story's overall pacing and structure.

#### **Stage 3: Define the Lens-Based Constructor**

Once the `Conductor's Score` is established, you must define how to build a beat from its instructions.

1.  **Map Lenses to Data:** For each `Narrative Lens`, work with the creator to define what specific information it corresponds to in the final writer's brief.
    *   **Example Mapping:**
        *   `Progression` -> Populates a `state_update_summary` block.
        *   `Revelation` -> Populates a `key_character_insights` block.
        *   `Atmosphere` -> Populates a `sensory_details_to_emphasize` list.

#### **Stage 4: Formalize the Composite Beat Schema**

Finally, synthesize all decisions into a single, formal data structure for the creator's final approval.

1.  **Draft the Schema:** Create a complete, annotated schema (YAML or JSON format) for the `Story Beat`. It must include the `narrative_focus` section (which holds the target lens profile) and all the optional data blocks defined in Stage 3.
2.  **Present for Approval:** Present this final schema to the creator, explaining how it flexibly accommodates all the different types of scenes your workshop has designed.

**Final Deliverables:**
Upon creator approval, present the two finalized artifacts to be handed back to the Orchestrator:
1.  **The Conductor's Score:** A well-commented set of rules.
2.  **The Composite Beat Schema:** The final, formal data structure.
<!-- this then passes to the writer agent -->

[[comments]]
wasn't waiting for approval before continuing
narrative lenses need a lot of refinement
might not be the ideal metaphor to use