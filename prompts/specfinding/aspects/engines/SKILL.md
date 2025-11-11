---
name: narrative-engines
description: Specialized, state-driven agents that advocate for specific narrative outcomes, character arcs, or thematic textures within the parliamentary process.
---
## Overview

Narrative Engines are the primary drivers of the story, acting as internal advocates for specific goals, plotlines, and principles. The story's narrative is the emergent property of several engines indepdently proposing actions/events/etc. in alignment with their internal agenda, relying on separate, external mechanisms to transforming the competing proposals into a coherent narrative. Engines are internally implemented using state machines and are capable of adapting their proposals based on the current story state and narrative scope.

## Identification Heuristics

*   **Plotline Specification:** User defines a specific, long-term sequence of events or a subplot. (e.g., "I want a mystery subplot about the queen's secret parentage that runs through the whole first act.")
*   **Character/Story Arcs:** User outlines the intended developmental path for a character.
*   **Systemic Behavior:** User describes a dynamic process or a long-term causal chain that needs to unfold over the course of the narrative.
*   **Emergent Storytelling:** User expresses a desire for the story to generate new, unexpected plotlines based on world events.
*   **Thematic or Tonal Pressures:** User specifies a recurring mood, theme, or atmospheric element that should be a catalyst for events and reactions.

## Component Synthesis Guide

### 1. Core Identity & Purpose

*   **Objective:** To establish the engine's fundamental narrative role, its unique "agenda," and other general properties.
*   **Scope of Inquiry:**
    *   What is this engine's primary reason for existing? What specific plot, arc, theme, or rule is it responsible for?
    *   Is this engine an `Architect` (ie. its primary purpose to generate new engines over time) or is it focused on direct storytelling.
*   **Strategic Focus:**
    *   Probe for specificity and creative nous. An objective like "make the story interesting" is insufficiently vague and needs to be reduced to a more specific dimension.
    *   Ground the engine in world truths and core concepts. The purpose of the engine is to implement an aspect of the intended narrative.
    *   Actively investigate better implementations. Identify opportunities to split, merge, or refine engine concepts for better fidelity and simplicity.
*   **Minimal Viability:**
    *   The engine has a clearly articulated, unique narrative purpose.
    *   The engine's `DONE` state (if one exists) is understandable.

### 2. Advocacy States
All engines have a `DORMANT` and a `DONE` state in which no proposals are being made. All engines start out in the `DORMANT` state unless they are defined to be active at the story's start.

*   **Objective:** To define the internal states of the engine, the specific advocacy tied to each state, and the logic that governs transitions between them. This is the mechanical core of the engine.
*   **Scope of Inquiry:**
    *   **The State:** What is a descriptive name for this phase of the engine's operation (e.g., Foreshadowing, Investigation, Confrontation)?
    *   **The Advocacy:** While in this state, what is the engine advocating for? What kinds of scenes, events, or directives does it propose? What does the world feel like under this specific influence?
    *   **The Transitions:** What conditions or events (ideally specific and queryable in the `World State`) will cause the engine to transition out of this state and to another? Where does it transition to?
    *   What is the story we are trying to tell with this engine? Do we even need multiple stages or can we accomplish the goal with only one?
    *   Under what narrative contexts should this engine's proposals be considered more important?
*   **Strategic Focus:**
    *   For complex engines, encourage drawing or listing the state graph explicitly to visualize the flow and ensure there are no dead ends.
    *   Constantly link the abstract "state" to the concrete "advocacy." The state is meaningless if it doesn't change what the engine *does* in the `Parliament`.
    *   Help the user define the "texture" of the engine's advocacy. Does it propose subtle nudges or dramatic, plot-altering events?
    *   Explore failure conditions. What does the engine propose if its primary suggestion is rejected or impossible? Does it have a backup plan or a different strategy?
*   **Minimal Viability:**
    *   At least one active state (a state other than `DORMANT` and `DONE`)
    *   Every active state has a defined advocacy behavior.
    *   Every active state has at least one defined transition to another state. A `DONE` state is a valid transition target.

## Integrity Rules

*   **State Reachability:** All defined states (except the initial state) must be reachable from the initial state through a valid sequence of transitions. There can be no orphan or dead-end states that cannot be exited (unless it is a `DONE` state).
*   **Transition Determinism:** The conditions for transitioning out of a given state should be mutually exclusive whenever possible. If multiple exit conditions can be true simultaneously, there must be a clear priority order.
*   **Purpose Conflict:** Check for pairs of Strategic Engines whose core objectives are logically irreconcilable (e.g., Engine A: "Character X must survive," Engine B: "Character X must die to fulfill the prophecy"). This is not an error but should be flagged as a source of core narrative conflict that the `GM/Conductor` will have to resolve.
*   **Silent State:** Any state other than `DORMANT` or `DONE` must have a defined advocacy behavior. An "Active" state with no corresponding proposals is a configuration error.
*   **No Useless States:** Any state other than `DORMANT` or `DONE` must have a defined advocacy behavior. This can include "passive advocacy", where the engine is waiting for some condition to be met before resuming normal activity.

## Application & Utility

*   **Function:** Active engines are the source of potential actions and outcomes. When a Parliament is convened with a Mandate, the GM/Conductor polls all relevant active engines. These engines analyze the World State and the Mandate to generate and advocate for proposals that advance their internal state-driven goals.
*   **Influence:** The weight of an engine's advocacy (its "bid strength") is a key factor used by the GM/Conductor during the Synthesize phase to score and select the winning proposal. This strength can be dynamically modified by Directives within a Mandate's Intent Frame, allowing higher-level narrative structures to amplify or suppress certain engines' voices based on the immediate context.
*   **State Management:** The internal state of every engine is stored within the World State under the state.narrative.engines namespace. Any change to an engine's state is part of the single atomic transaction that occurs after a Resolver completes its task, ensuring the entire system is always working from a consistent, up-to-date view of the narrative.
*   **Meta-Narrative Function (Architects):** Architect Engines have the unique ability to propose World State mutations that create and initialize new Strategic Engines. This is the primary mechanism for introducing major, emergent plotlines in long-running or sandbox narratives if needed.
