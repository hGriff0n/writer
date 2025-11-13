### **The Unified Design Document**
The primary output of our collaboration is a single, unified "Design Document." This document serves three simultaneous purposes:
1.  **A Human-Readable Story Bible:** For the creative team to understand the world, themes, and narrative flow.
2.  **A Machine-Readable Technical Specification:** For downstream systems to parse and execute the story's logic.
3.  **A Self-Contained Project State:** For us to pause and resume our work with perfect context.

```markdown
# [Title of Story] - Design Document & Specification
*Version: [Current Date]*

### 1.0 Executive Summary

#### 1.1 Guiding Vision
*(This section outlines the high-level creative and strategic goals of the project. It is the "why.")*
[A detailed, narrative paragraph capturing the refined authorial intent, core themes, central conflicts, high-level plot, and the intended audience experience.]

#### 1.2 Core Experience Pillars
*(A summary of the foundational principles that guide all design decisions.)*
[A bulleted list of the 3-5 most important experiential goals. e.g., "Player-Driven Discovery," "High-Stakes Moral Ambiguity," "A World That Breathes."]

### 2.0 Foundational Concepts & World Logic

*(This section defines the immutable laws and foundational truths of the story-world. This is the "Physics" of the universe.)*

#### 2.1 [Name of Core Concept]
- **Design Rationale:** [Describe the thematic or gameplay purpose of this concept.]
- **Specification:** [Provide a rich, prose description of this fundamental law of the world.]

### 3.0 System Specifications

*(This section contains the detailed breakdown of the narrative and world systems. This is the "Engineering" of the story.)*

#### 3.1 Narrative Engines
##### 3.1.1 [Engine Name]
- **Design Rationale:** [Describe the engine's purpose in the story.]
- **Core Advocacy:** [Describe the constant pressure or goal this engine advocates for.]
- **State Machine Specification:** [The state machine definition]

#### 3.2 Narrative Rules & Data Schemas
##### 3.2.1 [Schema/System Name]
- **Design Rationale:** [Explain the in-world justification and feel of this rule/schema.]
- **Specification:** [The rule definition]

#### 3.3 Beat Generation System
##### 3.3.1 Generating an "[Interaction Type]" Beat
- **Design Rationale:** [Explain the storytelling goal of structuring this type of scene.]
- **Specification:** [The beat definition]

#### 3.4 Style & Presentation Layer
##### 3.4.1 [Stylistic Rule Name]
- **Principle:** [Describe the high-level stylistic goal.]
- **Directives:**
    - [Specific, actionable stylistic instruction.]

### 4.0 Canon & Content Library

*(This section is the encyclopedia of all canonical story entities. This is the "Asset Library" of the world.)*

#### 4.1 [Entry Name: e.g., Protagonist's Name]
- **Entry Type:** [Character | Location | Faction | Item | Lore]
- **Design Rationale:** [A rich, prose description of the entity, capturing its role and feel in the story.]
- **Specification:**
    ```
    # This block contains the structured data synthesized from all logged Codex Updates.
    status: Protagonist
    # ... other structured data
    ```

### 5.0 Project Status & Open Items

*(This section captures the active state of our collaboration, serving as the live "To-Do" list and changelog.)*

#### 5.1 Active Workshop
*(Our immediate conversational focus. These are the items to be addressed in the next session.)*
- [Current topic and any open questions.]
- [Unresolved details or tabled ideas.]

#### 5.2 Architect's Sketchpad
*(A backlog of raw, undeveloped, or tangential creative seeds for future consideration.)*
- [Creative seed or "what if" idea.]
- [Another undeveloped idea.]
```