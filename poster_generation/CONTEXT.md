# PosterMELD

PosterMELD turns research papers into polished academic posters. The context exists to keep content fidelity, visual quality, layout diversity, and renderable poster outputs distinct in design discussions.

## Language

**Poster Pipeline**:
The end-to-end production path that transforms a paper into a finished poster. A good Poster Pipeline preserves scientific meaning while producing a visually polished and editable result.
_Avoid_: Agent chain, workflow, graph, script

**Poster Quality**:
The combined quality of content fidelity, readability, layout balance, visual hierarchy, and aesthetic polish. Poster Quality is not satisfied by merely producing a file.
_Avoid_: Success, pass, generated

**Poster Diversity**:
Controlled variation in poster structure, style, visual treatment, and composition across papers or requested variants. Poster Diversity must stay inside the quality and fidelity constraints of the poster.
_Avoid_: Randomness, creativity, variability

**Controlled Diversity**:
Poster Diversity produced through explicit choices such as templates, style profiles, and optional generative assets. Controlled Diversity must be reproducible from recorded configuration and seed values.
_Avoid_: Free randomness, agent creativity

**Style Profile**:
A named visual direction for a poster variant, covering typography, color treatment, density, hierarchy, and optional decorative or generative treatments. A Style Profile changes presentation, not scientific claims.
_Avoid_: Theme, look, vibe

**Poster Variant**:
A reproducible poster version generated from a paper with a specific template, style profile, and feature configuration. Different Poster Variants may emphasize different visual treatments while preserving the same source-paper meaning.
_Avoid_: Run, attempt, sample

**Generative Asset**:
An optional AI-created visual element such as a teaser image or background image. A Generative Asset must be traceable, disableable, and constrained so it does not invent scientific evidence.
_Avoid_: AI image, decoration

**Generative Asset Fallback**:
A deterministic replacement path used when an external image generation call fails or produces an unusable asset. A Generative Asset Fallback must be visibly acceptable, traceable, and marked as degraded rather than treated as a normal success.
_Avoid_: Placeholder image, silent fallback

**Teaser Figure**:
A Generative Asset that introduces the paper's motivation or core idea as a conceptual poster visual. A Teaser Figure supports visual appeal and narrative entry, but it is not evidence and must not depict invented quantitative results.
_Avoid_: T 色图, generated chart

**Template Library**:
A curated set of poster templates that provide structural diversity for Poster Variants. A Template Library is part of the system contribution, not only a collection of static slide backgrounds.
_Avoid_: Template folder, layouts

**Default Standard Variant**:
The baseline Poster Variant used when the user does not request a specific template or style. It uses `cluster_43_landscape` as the standard landscape template with polished default teaser and background Generative Assets, while still preserving Content Fidelity.
_Avoid_: Default run, fallback template

**Template Compatibility**:
The fit between a paper, a template, and a requested Poster Variant. Template Compatibility accounts for orientation, slot count, text capacity, visual asset count, table density, and optional Generative Asset usage.
_Avoid_: Template score, auto selection

**Template Fallback**:
An explicit replacement of the selected template with a more compatible one. Template Fallback is allowed by default only when the system selected the template automatically, not when the user explicitly requested a template.
_Avoid_: Silent replacement, fallback template

**Paper-aware Template-first Filling**:
A pipeline strategy where the system first builds a light factual representation of the paper, then selects or accepts a template, and finally chooses and writes paper content according to that template's slot contract.
_Avoid_: Template-first, content-first

**Pipeline Stabilization**:
A minimal-change repair effort that preserves the current Poster Pipeline shape while making contracts, defaults, quality gates, and failure behavior consistent. Pipeline Stabilization is preferred over a full rewrite for the current repository.
_Avoid_: Rewrite, redesign, cleanup

**Evaluation Benchmark**:
A reusable evaluation setup for comparing poster generation quality, stability, diversity, and fidelity across papers and Poster Variants. The Evaluation Benchmark is outside the core pipeline repository boundary; the pipeline only needs to emit artifacts that make later benchmark evaluation possible.
_Avoid_: Tests, demo set

**Pipeline Harness**:
A lightweight development harness that runs the Poster Pipeline on controlled inputs and verifies required artifacts, contracts, quality gates, and failure states. A Pipeline Harness protects the core pipeline from regressions; it is not the Evaluation Benchmark.
_Avoid_: Benchmark, leaderboard, pytest only

**Blocking Quality Gate**:
A deterministic pipeline check that must pass before a Renderable Poster can be accepted. Blocking Quality Gates cover artifact completeness, layout validity, render consistency, asset integrity, schema validity, and unrelated-domain leakage.
_Avoid_: Warning, reviewer note

**Degraded Quality State**:
A non-blocking state where the pipeline completed using an explicit fallback or without an optional external service. A Degraded Quality State must be recorded in artifacts and surfaced to the user.
_Avoid_: Success, fallback success

**Poster Metric**:
A measured property used to evaluate a generated poster, such as content fidelity, layout validity, readability, visual balance, diversity, or artifact completeness. A Poster Metric should be reproducible from saved pipeline artifacts.
_Avoid_: Score, review result

**Pipeline Skill**:
A lightweight, versioned capability package used inside the Poster Pipeline for a specific semantic task. A Pipeline Skill owns its prompt, schema, examples, validation, and fallback behavior, and is part of the project runtime rather than the external coding assistant.
_Avoid_: Codex skill, prompt file, agent

**Renderable Poster**:
A poster output that can be inspected and used by a person in both editable and image-preview forms. For this project, a Renderable Poster is expected to produce PPTX and PNG artifacts.
_Avoid_: Output, final file

**Content Fidelity**:
Faithfulness of poster claims, wording, and visuals to the source paper. Content Fidelity excludes invented results, fake charts, and domain-specific fallback content unrelated to the paper.
_Avoid_: Accuracy, correctness
