---
name: {Component Name}
description: {A single, concise sentence describing the component's function, intended for programmatic use like dynamic loading or tool selection.}
---
## Overview

{A detailed paragraph explaining what this component is, its purpose within the story blueprint, and the kind of narrative or mechanical work it accomplishes.}

## Identification Heuristics

{A bulleted list of specific conversational cues, keywords, and patterns of user speech that indicate this type of component is being discussed.}

## Component Synthesis Guide

{This section provides a guide for directing the synthesis and assembly of a single component instance. It is structured to answer a set of fundamental questions about each component part. For complex components, these questions are scoped to one or more 'Conceptual Parts'; a simple component only has one conceptual which has the same name as the component.}

1. **{Name of Conceptual Part}**
    *   **Objective:** {A concise statement explaining this part's function and its specific role in relation to the component as a whole.}
    *   **Scope of Inquiry:** {A paragraph/checklist describing the key concepts and areas of understanding the AI should focus on capturing for this part. This defines the essential aspects of the concept to be explored and understood.}
    *   **Strategic Focus:** {A paragraph/checklist outlining the specific contextual lens or priorities the orchestrator should adopt when discussing this part. This is a guidance on what to emphasize—such as probing for underlying motivations, exploring consequences, identifying hidden assumptions, or ensuring logical consistency—to help the user fully develop their idea.}
    *   **Minimal Viability Check:** {A paragraph/checklist outlining the principles for determining if the synthesized information for this part is sufficiently deep, internally consistent, and well-defined enough to be considered functionally complete.}

## Integrity Rules

{A bulleted list of checks that must be run on a new or modified instance of *this component type* during a `System Integrity Check` as a way of identifying problems and inconsistencies}

## Application & Utility

{This section describes how the component, as a holistic entity, should be used by downstream systems. The focus is on the component's overall narrative and functional purpose. Use the following list to articulate general principles of application, typically framed from the perspective of the system that will consume the component's data. Reference specific Conceptual Parts as needed to clarify how different aspects of the component contribute to its overall function.}