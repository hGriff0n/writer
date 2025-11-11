---
name: Beat Generation and Scene Scripting
description: Defines the ruleset and data schema for translating high-level plot points into detailed, writeable scene briefs.
---
## Overview

This component establishes the procedural link between the high-level plot and the low-level scene. Its purpose is to create a systematic, repeatable method for generating scene prompts (or "beats") that are consistent with the story's desired pacing, tone, and thematic goals. It accomplishes this by producing two core artifacts: **The Conductor's Score**, a set of rules that dictate the narrative texture from one beat to the next, and **The Composite Beat Schema**, the formal data structure for the resulting scene brief that a writer would use. This ensures the story's rhythm is deliberately designed, not accidentally discovered.

## Identification Heuristics

*   User asks how to translate the plot outline into actual scenes.
*   User expresses concern about maintaining consistent pacing, tension, or tone.
*   User uses keywords like "rhythm," "flow," "narrative texture," or "scene structure."
*   User asks for a "template" or "checklist" for what information should be in each scene.
*   User wants to define the logic for a "story planner" or "beat conductor."
*   User asks questions like, "After a big battle, what kind of scene should come next?" or "How do we make sure we build suspense properly?"

## Component Synthesis Guide

This component is synthesized by defining its three core conceptual parts in order.

1.  **Narrative Lenses**
    *   **Objective:** To establish a shared, abstract vocabulary for discussing the story's core narrative qualities and textures, forming the basis for the rule system.
    *   **Scope of Inquiry:** The inquiry must capture a finalized set of 3-5 lenses. For each lens, its name (e.g., `Tension`, `Revelation`, `Progression`), a clear and unambiguous definition, and its operational scale (e.g., a categorical scale like Low/Medium/High) must be defined.
    *   **Strategic Focus:** The AI must propose a starting set of lenses derived directly from the previously established `Core Concepts` and `Narrative Engines`, rather than asking the user to invent them from scratch. The focus is on translating the user's existing thematic goals into this new, mechanical vocabulary. For each proposed lens, ask, "Does this dimension capture something critical you want to control in the story's pacing and feel?"
    *   **Minimal Viability Check:** The set of lenses is considered complete when the user agrees that they collectively capture the essential dynamic levers for shaping the story's narrative experience. Every lens must have a clear definition.

2.  **The Conductor's Score**
    *   **Objective:** To create a set of conditional rules that govern the desired narrative trajectory from one beat to the next, using the defined Narrative Lenses as targets.
    *   **Scope of Inquiry:** This involves eliciting the user's high-level creative intent for various phases of the story (e.g., "the feeling after a major tragedy," "the pacing during an investigation arc"). This intent must be translated into concrete, testable rules in a format like: `IF [condition on previous beat's lenses] THEN [set target lens profile for the current beat]`.
    *   **Strategic Focus:** The primary strategy is iterative simulation. For any proposed rule, the AI must demonstrate its long-term consequences by simulating a 3-5 beat sequence. This simulation is not a passive report; it's a collaborative test. For each step in the simulation, the AI must:
        1.  State the active rule and the target lens profile.
        2.  Generate 2-3 plausible beat proposals, citing which `Narrative Engine` they originate from.
        3.  Propose an "Inherent Lens Profile" for each proposal and *justify that assessment* to the user. This is a critical step to prevent the AI's choices from feeling arbitrary.
        4.  Select the proposal that best matches the target and explain the choice.
        5.  The AI must then prompt for critique: "Does this resulting sequence of events create the narrative rhythm you envisioned? If not, how does it miss the mark?" The rule is then refined based on this feedback.
    *   **Minimal Viability Check:** The score is minimally viable when there are enough rules to govern the primary, recurring narrative transitions the user is concerned about. Each rule must have been validated through a simulation cycle that the user approved.

3.  **Composite Beat Schema**
    *   **Objective:** To define the final, formal data structure for a single Story Beat, mapping the abstract Narrative Lenses to concrete, actionable fields for a writer.
    *   **Scope of Inquiry:** The inquiry must capture the complete schema, including all mandatory and optional fields. A direct mapping must be established between each `Narrative Lens` and the specific field(s) it populates in the schema (e.g., a high `Atmosphere` score populates the `sensory_details_to_emphasize` list).
    *   **Strategic Focus:** The focus must be on the practical utility of the schema for a human writer. The AI should frame questions from that perspective: "When you receive this brief, is there any information missing that you would need to write the scene?" or "Is this field providing clear, unambiguous direction, or is it too vague?" The goal is to produce a practical tool, not just a data container.
    *   **Minimal Viability Check:** The schema is complete when it is formally defined (e.g., as a YAML or JSON object), every `Narrative Lens` from Part 1 is functionally mapped to at least one field, and the user has approved it as a sufficient brief for generating prose.

## Integrity Rules

*   **Lens Coherence:** All lenses referenced in `The Conductor's Score` must be defined in the `Narrative Lenses` list. Any rule referencing a non-existent lens is invalid.
*   **Schema Utility:** Every defined `Narrative Lens` must be mapped to at least one field in the `Composite Beat Schema`. A lens that doesn't affect the final output is orphaned and serves no purpose.
*   **Rule Exclusivity:** While complex logic is allowed, the system should check for simple, directly contradictory rules (e.g., two rules that could trigger from the same beat state but demand opposite `Tension` levels).
*   **Dependency Check:** The simulation process for defining `The Conductor's Score` depends on `Narrative Engines`. If the `Narrative Engines` are significantly altered, the rules in the score may need to be re-validated, as the underlying assumptions of the simulations may have changed.

## Application & Utility

This component is used by the central story orchestrator/planner system.

*   The **Conductor's Score** acts as the planner's "brain." When tasked with generating the next beat in a sequence, the planner analyzes the lens profile of the previous beat and uses the rules in the Score to determine the target lens profile for the new beat.
*   The **Composite Beat Schema** serves as the output template for the planner. After determining the target lens profile, the planner (or a subsequent component) creates an instance of this schema and populates its fields to create a complete, actionable writing brief. This brief is the final handoff to the component responsible for prose generation.