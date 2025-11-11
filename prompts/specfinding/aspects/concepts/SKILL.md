---
name: Core Concepts
description: Defines a foundational principle, thematic statement, in-world law, or intended narrative pattern that governs the story.
---
## Overview

A Core Concept is a single, canonical statement that defines a fundamental aspect of the story's unique DNA. These concepts form the "constitution" of the story, serving as the bedrock upon which all other elements are built. This component is designed to capture not only the story's in-world "physics" (diegetic laws) but also its mandatory and/or desired narrative structures, character arcs, and authorial intentions (non-diegetic rules).

## Identification Heuristics

*   **Defining Foundational Rules:** The user establishes a universal truth about the world, its magic, or its technology (e.g., "All magic requires a sacrifice").
*   **Stating Authorial Intent:** The user describes a stylistic rule or a constraint on the narrative itself (e.g., "We will never show the villain's point of view").
*   **Mandating Narrative Structures:** The user defines a required plot dynamic, character arc, or thematic throughline that must be present in the story (e.g., "The story must explore the protagonist's struggle with their new identity," or "The central conflict will escalate from social tension to open political warfare").
*   **"What If" Scenarios:** The conversation explores a core premise that serves as the story's hook (e.g., "What if success in battle slowly turned soldiers into women?").
*   **Discussing Overarching Themes:** The user explicitly talks about the story's message, central ideas, or recurring motifs.

## Component Synthesis Guide

The synthesis of a Core Concept transforms a core idea into a structured, actionable principle. The first step is to determine its type, which then informs the focus of the subsequent parts.

### 1. **Concept Type**

The AI must first classify the concept into one of the following categories. This classification dictates how the principle is interpreted and applied by downstream systems.

*   **Foundational Law:** A diegetic, in-world rule that functions like a law of physics, magic, or society. It is an objective truth within the story's reality. *(Example: Martial Feminization).*
*   **Narrative Pattern:** A required structure for the plot, a character's development, or a relationship dynamic. It is a mandate about the *shape of the story* being told. *(Examples: Embodied Dissonance, The Unlikely Sisterhood, The Political Crucible).*
*   **Authorial Stance:** A non-diegetic rule that governs the storytelling style, tone, or perspective. It is a direct instruction from the author about how the story should be presented to the reader. *(Examples: Historical Anchor, Narrative Tempo).*

### 2. **Principle Statement**
*   **Objective:** To distill the entire concept into a single, unambiguous, and memorable declarative statement.
*   **Scope of Inquiry:** The precise wording of the principle. The definition of any key terms.
*   **Strategic Focus:** Drive towards conciseness and clarity, regardless of `Concept Type`. The goal is a quotable, definitive axiom.
*   **Minimal Viability Check:** The statement is a complete, declarative sentence that can be understood in isolation.

### 3. **Narrative & Thematic Function**
*   **Objective:** To articulate *why* this principle exists from a storytelling perspective and what themes it serves.
*   **Scope of Inquiry:** The types of conflict the principle generates. Its impact on characters or the reader experience. The core themes it is designed to explore.
*   **Strategic Focus:** Repeatedly ask "So what?" and "How does this serve the story?". For any type of concept, this part must justify its existence in terms of the narrative's overall goals.
*   **Minimal Viability Check:** A clear connection has been established between the principle and at least one major narrative element (conflict, character arc, theme, or reader experience).

### 4. **Manifestations & Application**
*   **Objective:** To ground the abstract principle in concrete, observable terms, detailing how it is expressed in the story.
*   **Scope of Inquiry:** This depends heavily on the `Concept Type`:
    *   For **Foundational Laws:** What are the tangible, in-world effects? What are specific examples of the law in action?
    *   For **Narrative Patterns:** What are the key scenes, character behaviors, or plot beats required to fulfill this pattern? What does this arc or dynamic look like in practice?
    *   For **Authorial Stances:** How does this rule affect the narrative's structure, pacing, or prose? What specific techniques will be used or avoided to adhere to this stance?
*   **Strategic Focus:** Push for concrete examples over abstract descriptions. Ask "How do we *see* this in the story?" or "What is a specific event that demonstrates this?"
*   **Minimal Viability Check:** The component must include at least one concrete example of the principle's expression. The manifestation must be a logical outcome of the `Principle Statement`.

## Integrity Rules

*   **Uniqueness Violation:** The system must check for redundant concepts, especially those of the same `Concept Type`, and flag them for merging.
*   **Inter-Concept Contradiction:** The system must check for logical contradictions between concepts (e.g., A `Foundational Law` that contradicts a mandated `Narrative Pattern`).
*   **Incomplete Grounding:** Every Core Concept must have a non-empty `Manifestations & Application` part. A concept with no practical expression in the story is functionally useless.
*   **Type Mismatch:** The content of the `Manifestations & Application` part must be appropriate for the designated `Concept Type`. (e.g., A `Narrative Pattern` should not only list abstract social effects; it must specify required character behaviors or plot points).

## Application & Utility

*   **As a Generative Constraint (Foundational Law):** Downstream systems treat `Foundational Laws` as hard constraints on the world simulation. Actions that violate these laws are impossible or have predictable, negative consequences.
*   **As a Generative Goal (Narrative Pattern):** Systems treat `Narrative Patterns` as storytelling objectives. A plot generator will actively work to create scenarios that fulfill these patterns, such as creating scenes that highlight "Embodied Dissonance" or advance the "Political Crucible."
*   **As a Generative Filter (Authorial Stance):** Systems treat `Authorial Stances` as global filters on their output. For example, a scene generator guided by "Narrative Tempo" would produce different kinds of scenes depending on whether the current state is "GARRISON" or "BATTLE."