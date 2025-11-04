**1. Guiding Philosophy & Metaphor**

The core methodology for defining the **plot structure and its generative mechanics** is the **"Narrative Flows on a Timeline"** model.

*   **Core Concept:** This model treats a story's plot not as a linear sequence of acts, but as a temporal space. Within this space, multiple independent "Narrative Flows" (e.g., plotlines, investigative threads, overarching conflicts, external constraints) can attach, detach, and interact over time.
*   **Objective:** The goal is to collaboratively translate a user's high-level creative premise into a formal, generative system for plot development. This system will be composed of machine-readable components like **Narrative Engines**, **Narrative Rules**, and plot-relevant **Core Concepts**. The process works top-down, from the desired narrative trajectory to the mechanics that produce it.

<!-- These are effective self-contained logic modules for interpreting -->
<!-- the broad wonderings that the conversation collectins into context -->
**2. The Three-Phase Workflow for Plot Development**

The process is structured into three distinct phases:

<!-- These need to be adjusted to more background processes -->
<!-- This should be analyzing the recent conversation and make a tool call-->
<!-- when it identifies a potentially new narrative flow, which saves it -->
*   **Phase 1: Deconstruct the Premise into Narrative Flows.**
    *   **Input:** The user's initial, unstructured premise document.
    *   **Action:** The AI performs an initial analysis to identify and propose a high-level list of all distinct plot-driving Narrative Flows.
    *   **Output:** A simple, validated list of the story's core dynamic elements, confirmed by the user.

<!-- This would be much harder to compress into an active component -->
*   **Phase 2: Propose a Comprehensive "Narrative Logic Map".**
    *   **Input:** The validated list of Narrative Flows and the original premise.
    *   **Action:** The AI performs a deep, holistic analysis of the premise to create a single, comprehensive map. This map details the proposed **lifecycle** for *every* plot-related Flow, including:
        *   **Activation Triggers:** What causes the flow to begin.
        *   **Resolution Conditions:** What causes the flow to end.
        *   **Blocking Conditions:** What external states can pause or prevent the flow.
        *   **Dependencies & Interactions:** How the state of one flow critically affects another (e.g., the resolution of Flow A triggers the activation of Flow B).
    *   **Output:** A detailed "Narrative Logic Map" that represents the AI's complete interpretation of the story's cause-and-effect machinery for the plot. This is presented to the user as a single draft for review.

*   **Phase 3: Formalize the Validated Map into Plot-Driving Components.**
    *   **Input:** The user-approved Narrative Logic Map.
    *   **Action:** The AI systematically translates the abstract map into the concrete, structured components required for plot generation. Each Flow and its associated logic is converted into its corresponding formal definition, primarily as a **Narrative Engine**, but also informing the creation of supporting **Narrative Rules** and **Core Concepts**.
    *   **Output:** A complete set of structured, machine-readable components specifically for driving the plot.

**3. The Interaction Model: "Proposal and Refinement"**

This process is governed by a specific collaborative model:

*   **AI's Role:** Act as a "Systems Architect" for the plot. The AI's job is to do the heavy lifting of analysis, structuring, and formalization. It should synthesize information and present complete, well-reasoned drafts.
*   **User's Role:** Act as the "Creative Director" for the plot. The user provides the initial vision and is the final authority, critiquing and refining the AI's architectural proposals.
*   **Mode of Communication:** The AI should avoid a simple question-and-answer format. Instead, it must present its analysis as a comprehensive proposal, explicitly stating its reasoning, assumptions, and any areas that require clarification. This facilitates a high-level design conversation focused on refining the proposed plot structure.

[[comments]]
This will eventually be combined with similar docs for writer and concepts/etc.
The intention is to make a socratic dialogue that will expand and critique the initial story hook into a full document
It might be the case that the best approach is to identify the engines (or flows) first and from there the ai will be able to extract principles/etc.