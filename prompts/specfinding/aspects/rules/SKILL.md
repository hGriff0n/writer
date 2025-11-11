---
name: Narrative Rule
description: Defines a specific, computable mechanic, formula, or procedural instruction that governs the story's world.
---
## Overview

A `Narrative Rule` serves as a component of the story's "physics engine" or "legal code." Its purpose is to translate abstract `Core Concepts` or narrative intentions into concrete, executable logic. These components are the home for all hard, computable mechanics, from conditional logic (`if-then` statements) and mathematical formulas to the explicit data structures (schemas) of entities like characters, items, or locations. Each rule provides an unambiguous, procedural instruction that can be used to simulate outcomes, enforce consistency, and govern the behavior of the story's world and its inhabitants.

## Identification Heuristics

*   The user describes the "how" of a system, focusing on mechanics rather than thematic intent.
*   The user employs conditional language, such as "if...then," "when X happens, Y occurs," "unless," or "depends on."
*   The user introduces specific numbers, formulas, calculations, or sequences of events that are meant to be consistently applied.
*   The user defines the data structure or attributes of an object, character, or other entity (e.g., "All characters need to have health, mana, and stamina stats").
*   Use of keywords like: "mechanic," "system," "rule," "formula," "calculate," "triggers," "causes," "schema," "stats," "attributes."

## Component Synthesis Guide

This component is synthesized from two distinct conceptual parts: the justification for the rule's existence and its formal mechanical specification.

1.  **Narrative Justification**
    *   **Objective:** To capture the in-world reason or thematic purpose behind the mechanical specification, answering the question: "Why does this rule exist in the story?"
    *   **Scope of Inquiry:** The inquiry should focus on the rule's diegetic origin and purpose.
        *   What phenomenon, law of nature, societal custom, or magical principle does this rule represent?
        *   Why does it function this way from the perspective of the characters or the world's history?
        *   Does this rule connect to or derive from a broader `Core Concept`?
    *   **Strategic Focus:** Prioritize understanding the "why" before defining the "what." Push past a simple restatement of the mechanic. If the user provides a rule like `fire_damage * 2 against ice_creatures`, probe for the in-world explanation. Is it a matter of thermal shock? A metaphysical opposition between elemental forces? A curse from a forgotten deity? Connecting the rule to the established lore is paramount.
    *   **Minimal Viability Check:** The justification provides a clear, in-world explanation for the rule's existence and is not merely a description of the mechanic itself. It should feel like a snippet of a world bible or design document, explaining the intent.

2.  **Mechanical Specification**
    *   **Objective:** To define the unambiguous, procedural, and computable details of the rule in a structured format.
    *   **Scope of Inquiry:** The inquiry must capture the precise operational details. The specification must be one of the following types:
        *   **Formula:** A mathematical expression (e.g., `final_damage = (base_attack * power_modifier) - target.armor`).
        *   **Conditional Logic:** A pseudo-code block detailing conditions and outcomes (e.g., `if character.status includes "wet" and spell.element == "lightning", then damage_multiplier = 1.5`).
        *   **Data Table:** A Markdown table for lookups (e.g., material hardness vs. damage resistance).
        *   **Data Schema:** An indented list defining the structure of an entity. This is the required format for defining objects, characters, etc., and must utilize base schemas where available.
        *   **Event Listener:** A trigger and effect statement (e.g., `event: on_character_death; effect: trigger_ghost_spawn(character.id)`).
    *   **Strategic Focus:** Emphasize precision and consistency. Ensure all variables and attributes used in a formula or pseudo-code are defined elsewhere, either in a data schema or another rule. When defining a data schema, first check for an applicable base schema to extend. Enforce the use of the specified formats, particularly the indented-list format for schemas, to maintain system-wide consistency and efficiency.
    *   **Minimal Viability Check:** The specification is written in one of the approved formats. The logic is self-contained and unambiguous. All terms, variables, and attributes used are either universally understood (e.g., `+`, `-`) or are defined in another component within the story blueprint.

## Integrity Rules

*   **Completeness:** A `Narrative Rule` must contain both a `Narrative Justification` and a `Mechanical Specification`. A component with only one is incomplete.
*   **Format Adherence:** The `Mechanical Specification` must use one of the five approved formats (Formula, Conditional Logic, Data Table, Data Schema, Event Listener). The use of other formats, especially JSON, is an integrity violation.
*   **Reference Validity:** All attributes, entity names, or other components referenced in a `Mechanical Specification` (e.g., `character.strength`, `item.corrosion_resistance`) must be defined elsewhere in the system's data schemas or components. A rule referencing a non-existent attribute is invalid.
*   **Logical Contradiction:** The system should check for direct contradictions between rules. For example, if one rule defines `character.fire_resistance = 50%` and another defines `character.fire_resistance = -25%` under the same conditions, an integrity conflict must be flagged.
*   **Schema Derivation:** Any `Data Schema` specification that defines an entity for which a base schema exists must properly extend that base schema. Defining a new character schema from scratch when a `base_character` schema is available is an integrity violation.

## Application & Utility

The `Narrative Rule` component is a foundational element for downstream systems that require logical consistency and simulation capabilities.

*   **Simulation Engine:** Can directly execute the `Mechanical Specification` to calculate outcomes of actions, environmental effects, or social interactions. The `Narrative Justification` provides context for describing these outcomes.
*   **Consistency Guardian:** A writing assistance tool can use these rules to validate the narrative. It can flag passages where the author's prose violates an established rule (e.g., "You wrote that the character broke down the iron door, but their `strength` attribute is too low according to the `Material_Strength` rule.").
*   **Data Model Generation:** Rules containing `Data Schema` specifications are used to generate the definitive data models for all story entities, serving as the single source of truth for character sheets, item databases, and more.
*   **Interactive Narrative Engine:** Can use `Event Listener` rules to trigger state changes or branch the story in response to specific in-world occurrences.