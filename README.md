<div align="center">
  <h1>PosterMELD</h1>

  <p><strong>Multi-Agent Paper-to-Poster Generation for Controllable Design Diversity<br />with Editable Print-Ready Outputs</strong></p>

  <p>
    <strong>M</strong>ulti-Agent &middot;
    <strong>E</strong>ditable &middot;
    <strong>L</strong>ayouts &middot;
    <strong>D</strong>esign diversity
  </p>

  <p>
    <a href="https://jackey0903.github.io/PosterMELD/"><strong>Project Page</strong></a>
    &middot; <a href="poster_generation/"><strong>Generation Guide</strong></a>
    &middot; <a href="benchmark_eval/"><strong>Evaluation Guide</strong></a>
    &middot; <a href="#quick-start"><strong>Quick Start</strong></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11" />
    <img src="https://img.shields.io/badge/Output-Editable_PPTX-0B7A75?style=flat-square&logo=microsoftpowerpoint&logoColor=white" alt="Editable PPTX output" />
    <img src="https://img.shields.io/badge/Templates-24-7256B8?style=flat-square" alt="24 templates" />
    <img src="https://img.shields.io/badge/Benchmark-621_papers-D99A27?style=flat-square" alt="621-paper benchmark" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-17233A?style=flat-square" alt="MIT License" /></a>
  </p>
</div>

PosterMELD converts scientific papers into **native, editable PowerPoint posters** and matching PNG renders. The system plans template capacity before writing, grounds poster content in the source paper, places figures and tables, and applies bounded deterministic and vision-language-model quality checks.

## Repository Modules

This public release is organized as two independently runnable modules:

| Module | Purpose | Main entry point |
|---|---|---|
| [`poster_generation/`](poster_generation/) | Complete PDF-to-PPTX/PNG pipeline, including agents, prompts, templates, assets, configuration, scripts, examples, and tests | `postermeld` or `python -m src.workflow.pipeline` |
| [`benchmark_eval/`](benchmark_eval/) | Standalone PRR/CHE, Universal Score, and Keypoint BERTScore evaluation code | `python -m prr_che.evaluate`, `python -m universal_score.evaluate`, and `python -m keypoint_bertscore.*` |

The two modules do not import from each other. You can install and use either one independently.

## Highlights

- **Structure-first generation:** template geometry and block capacity guide content planning before rendering.
- **Multi-agent composition:** specialized agents coordinate paper understanding, keypoint selection, writing, layout, visual placement, and review.
- **Editable artifacts:** text, figures, tables, section bars, and logos remain native PowerPoint elements.
- **Controllable diversity:** 16 landscape and 8 portrait templates support style, density, background, header, and seed variations.
- **Paper-grounded content:** MinerU extracts structured text and visual assets, with Marker retained as an automatic fallback.
- **Bounded quality repair:** geometry checks and VLM review revise only failed aspects instead of repeatedly rebuilding the poster.
- **Reproducible evaluation:** the benchmark package preserves missing outputs in request-level metrics and records model responses and retry metadata.

## Quick Start

### Generate an editable poster

Poster generation requires Python 3.11. LibreOffice is recommended for stable PPTX-to-PNG rendering.

```bash
git clone https://github.com/Shannon4Science/PosterMELD.git
cd PosterMELD/poster_generation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
```

Run the bundled example without optional generated visuals:

```bash
postermeld data/0409_demo/paper.pdf \
  --layout-template auto \
  --disable-generated-teaser \
  --disable-generated-background
```

After configuring the VLM, image-generation, and MinerU endpoints in `.env`, enable the complete visual pipeline:

```bash
postermeld /path/to/paper.pdf \
  --layout-template auto \
  --poster-style navy_serif \
  --visual-density rich \
  --enable-generated-teaser \
  --enable-generated-background \
  --enable-vlm-layout-review \
  --enable-visual-legibility-review
```

See the [generation guide](poster_generation/README.md) for configuration, template controls, output artifacts, and backend setup.

### Evaluate generated posters

```bash
cd ../benchmark_eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m prr_che.evaluate \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/prr_che \
  --model gpt-5.5
```

See the [evaluation guide](benchmark_eval/README.md) for manifest schemas, metric definitions, aggregation rules, and GPU evaluation.

## Project Layout

```text
PosterMELD/
├── poster_generation/        complete poster-generation subproject
│   ├── assets/               conference and institution assets
│   ├── config/               prompts and pipeline configuration
│   ├── data/                 runnable example papers
│   ├── scripts/              batch-generation utilities
│   ├── src/                  agents, layout, tools, state, and workflow
│   ├── templates/            16 landscape and 8 portrait templates
│   └── tests/                generation regression tests
├── benchmark_eval/           standalone benchmark-evaluation subproject
│   ├── common/               manifest, API, image, and JSON utilities
│   ├── prr_che/              print-ready and conditional aesthetic metrics
│   ├── universal_score/      reference-aware universal evaluation
│   ├── keypoint_bertscore/   OCR and content-coverage evaluation
│   ├── examples/             manifest and annotation examples
│   └── tests/                offline evaluation tests
├── Makefile                  repository-level validation commands
└── LICENSE
```

## Validation

Run both offline suites from the repository root:

```bash
make test
```

Or validate the modules independently:

```bash
make test-generation
make test-evaluation
make smoke-generation
```

## Credentials and Data

Real API credentials, generated outputs, benchmark corpora, and local machine paths are intentionally excluded. Copy the provided `.env.example` files and supply credentials at runtime. Never commit `.env` files or service tokens.

## License

PosterMELD is released under the [MIT License](LICENSE).

<div align="center">
  <br />
  <strong>One paper, many valid posters.</strong>
</div>
