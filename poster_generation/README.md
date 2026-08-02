<div align="center">
  <h1>PosterMELD</h1>

  <p><strong>Multi-Agent Paper-to-Poster Generation for Controllable Design Diversity<br />with Editable Print-Ready Outputs</strong></p>

  <p>
    <strong>M</strong>ulti-Agent ·
    <strong>E</strong>ditable ·
    <strong>L</strong>ayouts ·
    <strong>D</strong>esign diversity
  </p>

  <p>
    <a href="https://jackey0903.github.io/PosterMELD/"><strong>Project Page</strong></a>
    · <a href="#quick-start"><strong>Quick Start</strong></a>
    · <a href="#method"><strong>Method</strong></a>
    · <a href="#benchmark"><strong>Benchmark</strong></a>
    · <a href="#configuration"><strong>Configuration</strong></a>
    · <a href="#documentation"><strong>Documentation</strong></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11" />
    <img src="https://img.shields.io/badge/Output-Editable_PPTX-0B7A75?style=flat-square&logo=microsoftpowerpoint&logoColor=white" alt="Editable PPTX output" />
    <img src="https://img.shields.io/badge/Templates-24-7256B8?style=flat-square" alt="24 templates" />
    <img src="https://img.shields.io/badge/Benchmark-621_papers-D99A27?style=flat-square" alt="621-paper benchmark" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-17233A?style=flat-square" alt="MIT License" /></a>
  </p>
</div>

<p align="center">
  <sub>
    One paper, several generation routes. PosterMELD produces compact, readable, editable posters
    under an explicit request budget.
  </sub>
</p>

## Overview

PosterMELD turns a scientific paper into a **native, editable PowerPoint poster** and a matching PNG render. It fixes the poster structure before writing: template slots expose their geometry and capacity, then specialized agents select grounded keypoints, write to area budgets, place figures and tables, and render the result. Deterministic gates and a vision-language model (VLM) reviewer send only failed aspects through bounded repair.

The system is designed around four practical requirements:

| | Capability | What it means |
|---|---|---|
| **M** | **Multi-Agent** | Content, template, layout, visual, and review agents operate on one traceable poster state. |
| **E** | **Editable** | Text, figures, tables, shapes, and section structure remain native PowerPoint elements. |
| **L** | **Layouts** | Capacity-aware slot contracts guide writing before rendering, reducing overflow and unused space. |
| **D** | **Design diversity** | Template, style, density, background, header, and seed controls produce reproducible variants. |

### Headline results

Evaluated end-to-end on **621 papers**, PosterMELD achieves:

- **81.3% Print-Ready Rate (PRR)** - 3.4x P2P and 5.2x PosterGen.
- **3.247 conditional CHE** - the highest aesthetic score among generated methods with multiple print-ready outputs.
- **Native editability** - each accepted request returns an editable PPTX and its consistent PNG render.
- **$0.38 per request** - approximately 3.5% of the cost of the Codex+Skill workflow evaluated in the paper.

## Method

PosterMELD follows a structure-first, quality-guided pipeline:

1. **Parse and ground.** MinerU extracts text, figures, tables, formulas, and document structure; Marker remains available as a fallback backend.
2. **Plan capacity.** The selected template exposes slot geometry, reading order, prominence, character budgets, and compatible visual types.
3. **Compose with agents and skills.** Content, template, layout, and visual agents distill keypoints, write within area budgets, and place paper-grounded evidence.
4. **Render editable artifacts.** The renderer creates native PowerPoint elements and a matching PNG preview.
5. **Review and repair.** Deterministic checks and VLM review inspect grounding, bounds, overlap, readability, assets, hierarchy, and occupancy. Failures receive bounded rewrite, reflow, resize, or rerender actions.

Every run records its controls, provenance, latency, token usage, repair history, and final acceptance state.

### Template library

The repository includes **16 landscape** and **8 portrait** templates mined from real poster topologies. A template describes structure rather than content. Runtime geometry can absorb small gaps while preserving page bounds, reading order, and non-overlap constraints.

PosterMELD also provides `adaptive_auto`, a template-independent adaptive layout mode for papers that do not fit a standard topology.

## Benchmark

The benchmark contains **621 papers**, covering **14 publication-source groups** and **10 research domains**. Each method is evaluated at the request level, so failed or missing generations remain in the denominator.

| Method | PRR (%) ↑ | CHE ↑ | Universal ↑ | Editable | Cost / request ↓ |
|---|---:|---:|---:|:---:|---:|
| GPT-Image-2 | **85.2** | 2.698 | **4.948** | No | **$0.18** |
| Codex+Skill | 82.8 | 2.716 | 4.876 | Yes | $10.78 |
| Paper2Poster | 0.2 | - | 2.772 | Yes | $0.34 |
| P2P | 24.2 | 3.071 | 4.033 | No | $0.35 |
| PosterGen | 15.8 | 3.163 | 3.901 | Yes | $0.28 |
| **PosterMELD** | **81.3** | **3.247** | **4.456** | **Yes** | **$0.38** |
| Human reference | 98.7 | 3.287 | 4.995 | Yes | - |

> **Metric scope.** PRR measures request-level artifact validity. CHE averages Craftsmanship, Harmony, and Expressiveness only over print-ready outputs. Universal scores missing generations as zero. Cost includes failed requests and all generation-internal model calls.

### Qualitative comparison

Across five research areas, PosterMELD varies template topology, palette, and text-figure allocation while preserving native editability. The full-resolution comparison is also available on the [project page](https://jackey0903.github.io/PosterMELD/).

## Quick Start

### 1. Install

PosterMELD requires Python `3.11`. [LibreOffice](https://www.libreoffice.org/) is recommended for stable PPTX-to-PNG rendering.

```bash
git clone https://github.com/Shannon4Science/PosterMELD.git
cd PosterMELD/poster_generation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Or install with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

### 2. Configure

Create `.env` from [`.env.example`](.env.example). Never commit real API keys.

```bash
cp .env.example .env
```

At minimum, configure a text-model endpoint:

```bash
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-endpoint/v1
PAPER2POSTER_TEXT_MODEL=gpt-5.4
```

For the full pipeline, also configure VLM review, image generation, and MinerU parsing in `.env`.

### 3. Generate a poster

Minimal generation:

```bash
postermeld /path/to/paper.pdf \
  --layout-template auto \
  --disable-generated-teaser \
  --disable-generated-background
```

Full generation with visual assets and review:

```bash
postermeld /path/to/paper.pdf \
  --layout-template auto \
  --conference AAAI \
  --poster-style navy_serif \
  --visual-density rich \
  --enable-generated-teaser \
  --enable-generated-background \
  --enable-vlm-layout-review \
  --enable-visual-legibility-review
```

Outputs are written to `output/<paper_name>/`:

```text
output/<paper_name>/
├── <paper_name>.pptx          editable poster
├── <paper_name>.png           final render
├── <paper_name>_draft.*       pre-review artifact
├── timing_cost_log.json       latency, calls, and token usage
├── assets/                    extracted and generated visual assets
└── content/                   structured planning and quality reports
```

## Configuration

The following controls are connected to the runtime pipeline and affect planning or rendering:

| Control | Representative values | Effect |
|---|---|---|
| Template | `auto`, `adaptive_auto`, registered template ID | Block topology, reading order, and capacity |
| Poster style | `navy_serif`, `teal_modern`, `burgundy_classic` | Typography, section bars, accents, and panel treatment |
| Visual density | `lean`, `balanced`, `rich` | Figure, table, and result-asset retention |
| Background style | `auto`, `minimal_solid`, `tech_grid`, `academic_paper`, `cartographic`, `blueprint`, `geometric_soft` | Background visual language |
| Background palette | `auto`, `light_blue`, `light_gray`, `warm_ivory`, `mint`, `lavender`, `rose`, `amber` | Background color family |
| Header route | `auto`, `classic_left`, `centered`, `right_title`, `split_logos` | Title, author, and logo composition |
| Title wrap | `auto`, `single_line`, `two_line` | Header line count and title sizing |
| Seed | integer | Reproducible request variation |

List all available templates:

```bash
postermeld --list-layout-templates
```

Inspect all command-line options:

```bash
postermeld --help
```

## Quality and provenance

PosterMELD treats print-readiness as a hard outcome rather than an assumed property.

- **Paper-grounded:** claims, numbers, captions, and visual summaries must be supported by parsed paper content.
- **Editable first:** native PPTX text, shapes, images, and tables are preferred over flattened page images.
- **Readability first:** small figures are enlarged, moved, or summarized before font sizes are reduced.
- **Bounded repair:** only the failed aspect is revised, with a fixed iteration budget.
- **Explicit failures:** an enabled VLM or image service cannot silently pass with a placeholder artifact.
- **Auditable runs:** story boards, slot contracts, occupancy reports, quality-gate results, timing, and token usage are persisted.

Key reports include:

| Report | Purpose |
|---|---|
| `poster_keypoint_selection.json` | Selected paper-grounded poster keypoints |
| `story_board.json` | Section grouping and source-keypoint traceability |
| `styled_layout.json` | Geometry, typography, and visual references |
| `block_occupancy_report.json` | Per-block utilization and whitespace status |
| `final_quality_gate.json` | Acceptance state and blocking defects |
| `timing_cost_log.json` | Runtime, model calls, and input/output tokens |

## Repository layout

```text
poster_generation/
├── assets/                  conference and institution assets
├── config/                  pipeline configuration and prompts
├── scripts/                 benchmark and maintenance utilities
├── src/
│   ├── agents/              content, layout, visual, and review agents
│   ├── layout/              template selection and layout helpers
│   ├── state/               shared PosterState contract
│   ├── template_extraction/ template registry and geometry extraction
│   ├── tools/               model, MinerU, PPTX, and image adapters
│   └── workflow/            pipeline graph and CLI
├── tests/                   contract and regression tests
└── templates/
    ├── landscape/           16 landscape templates
    └── portrait/            8 portrait templates
```

## Testing

Run the full test suite:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

The regression suite covers template registration, capacity planning, keypoint grouping, visual references, MinerU mapping, generation retries, and final quality-gate behavior.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) defines the project vocabulary, boundaries, and system invariants.
- [`src/workflow/pipeline.py`](src/workflow/pipeline.py) is the authoritative CLI and workflow entry point.
- [`config/poster_config.yaml`](config/poster_config.yaml) contains the default style, layout, capacity, and review policies.

## Acknowledgements

PosterMELD builds on [LangGraph](https://github.com/langchain-ai/langgraph), [MinerU](https://github.com/opendatalab/MinerU), [python-pptx](https://github.com/scanny/python-pptx), and [LibreOffice](https://www.libreoffice.org/). The project page follows the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

## License

PosterMELD is released under the [MIT License](../LICENSE).

<div align="center">
  <br />
  <strong>One paper, many valid posters.</strong>
</div>
