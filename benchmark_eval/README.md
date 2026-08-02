# PosterMELD Benchmark Evaluation

This directory contains the standalone evaluation code used for PosterMELD. It provides four poster-evaluation pipelines:

1. **Print-Ready Rate (PRR)**: a VLM-based binary test of whether a poster can be displayed directly at an academic conference.
2. **Conditional Holistic Evaluation (CHE)**: three 1--5 aesthetic scores, evaluated only when PRR is true.
3. **Universal Score**: ten VLM-scored poster criteria on a 0--5 scale, compared with a reference poster. Their arithmetic mean is the reported Universal Score.
4. **Keypoint BERTScore**: MinerU OCR on the poster image followed by BERTScore against ordered paper keypoints.

The module does not import from [`../poster_generation/`](../poster_generation/). API credentials, generated results, benchmark corpora, and local machine paths are intentionally excluded.

## Directory Layout

```text
benchmark_eval/
|-- common/                  Shared manifest, image, API, and JSON utilities
|-- prr_che/                 Cascaded PRR and CHE evaluation
|   `-- prompts/             Final evaluation prompts used in the paper
|-- universal_score/         Universal checklist evaluator and XGBoost model
|-- keypoint_bertscore/      MinerU OCR, BERTScore, and aggregation
|-- examples/                Manifest and annotation format examples
|-- scripts/                 Reproducible command templates
|-- tests/                   Offline unit tests
|-- requirements.txt         CPU/API evaluation dependencies
`-- requirements-gpu.txt     Additional GPU OCR/BERTScore dependencies
```

## Input Manifest

All pipelines use the same JSONL manifest. Each line describes one method/paper pair:

```json
{
  "id": "paper_0001__method_a",
  "method": "Method A",
  "subset": "test",
  "paper_name": "paper_0001",
  "poster_path": "posters/method_a/paper_0001.png",
  "reference_poster_path": "posters/human/paper_0001.png",
  "annotation_path": "annotations/paper_0001.json"
}
```

Paths may be absolute or relative to the manifest file. Required fields are:

| Pipeline | Required fields |
|---|---|
| PRR/CHE | `id`, `method`, `poster_path` |
| Universal Score | `id`, `method`, `poster_path`, `reference_poster_path` |
| Keypoint BERTScore | `id`, `method`, `poster_path`, `annotation_path` |

If `poster_path` is null or the file is absent, the sample is retained as a missing generation. It receives PRR=false and zero in full-benchmark Universal/BERTScore summaries. CHE is not applicable.

See `examples/manifest.jsonl` and `examples/annotation.json` for complete examples.

## Installation

Create a Python 3.10 environment for API-based evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For keypoint BERTScore, install the GPU dependencies in a CUDA-compatible environment:

```bash
pip install -r requirements-gpu.txt
```

The pinned GPU versions match CUDA 12.6-era PyTorch wheels used in our experiments. For a different driver/CUDA stack, install compatible PyTorch and vLLM builds first, then install the remaining packages.

## API Configuration

PRR/CHE and Universal Score use an OpenAI-compatible multimodal chat-completions endpoint. Credentials are read only at runtime:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.example.org/v1"
```

Copy `.env.example` to `.env`, or export the variables in your shell. For multiple keys, place one key per line in a file and pass `--api-key-file`. Keys are never written to result files. `--workers-per-key` is an in-flight request limit for each key.

## PRR and CHE

PRR is evaluated first. CHE is requested only for samples with `print_ready=true`. Each result JSON stores the original VLM response, parsed fields, response metadata, retry history, prompt hashes, and image-encoding metadata.

```bash
python -m prr_che.evaluate \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/prr_che \
  --model gpt-5.5 \
  --workers-per-key 4 \
  --max-attempts 6 \
  --retry-delay 5

python -m prr_che.summarize \
  --manifest examples/manifest.jsonl \
  --result-dir outputs/prr_che \
  --report-dir reports/prr_che
```

The prompts in `prr_che/prompts/print_ready_prompt.txt` and `prr_che/prompts/che_prompt.txt` are the final prompts used for the reported experiments. CHE is the arithmetic mean of Visual Craftsmanship, Stylistic Harmony, and Expressive Distinctiveness.

## Universal Score

The evaluator submits the reference poster and poster under evaluation in one request and obtains all ten criterion scores. The reported Universal Score is the arithmetic mean of the ten values. The optional `xgboost_score` reproduces the learned P2P auxiliary predictor and uses the bundled `xgboost_model.joblib` in the checklist order.

```bash
python -m universal_score.evaluate \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/universal \
  --model gpt-4o \
  --workers-per-key 4 \
  --max-attempts 6 \
  --retry-delay 5

python -m universal_score.summarize \
  --manifest examples/manifest.jsonl \
  --result-dir outputs/universal \
  --report-dir reports/universal
```

`universal_score/checklist.yaml` fixes both criterion wording and feature order. Missing posters are assigned zero to all ten dimensions in the full-benchmark aggregate.

## Keypoint BERTScore

### Models

Model weights are intentionally excluded from this archive. Download or pre-stage:

- `opendatalab/MinerU2.5-Pro-2605-1.2B`
- `roberta-large`

Pass local model directories to the commands. The BERTScore configuration is `roberta-large`, layer 17, IDF disabled, baseline rescaling disabled, and the slow tokenizer, matching the reported evaluation.

### Annotation schema

`paper_poster_keypoints` is a list of objects containing `id`, `key_point`, and `section`. When present, `reading_order` determines concatenation order. Otherwise list order is used.

### Run

On two GPUs, start one independent MinerU vLLM engine per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m keypoint_bertscore.ocr_vllm \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/keypoint_bertscore \
  --model-path /path/to/MinerU2.5-Pro-2605-1.2B \
  --rank 0 --world-size 2 &

CUDA_VISIBLE_DEVICES=1 python -m keypoint_bertscore.ocr_vllm \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/keypoint_bertscore \
  --model-path /path/to/MinerU2.5-Pro-2605-1.2B \
  --rank 1 --world-size 2 &
wait
```

After OCR engines exit, score the extracted text. `torchrun` shards samples across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m keypoint_bertscore.score \
  --manifest examples/manifest.jsonl \
  --output-dir outputs/keypoint_bertscore \
  --model-path /path/to/roberta-large \
  --batch-size 256

python -m keypoint_bertscore.summarize \
  --manifest examples/manifest.jsonl \
  --result-dir outputs/keypoint_bertscore \
  --report-dir reports/keypoint_bertscore
```

The OCR and scoring stages are resumable. Existing successful per-sample JSON files are skipped unless `--force` is supplied.

## Outputs and Aggregation

Every evaluator writes one JSON file per manifest item under a method-neutral safe identifier. Summarizers produce:

- `summary.json`: machine-readable aggregate statistics;
- `method_summary.csv`: one row per method;
- `per_sample.csv`: one row per manifest item;
- a Markdown report for direct inspection.

All reported full-benchmark means retain missing/failed generations as zeros where specified. Available-only statistics are also emitted for keypoint BERTScore.

## Offline Validation

No API call or model weight is needed for the test suite:

```bash
python -m unittest discover -s tests -v
python -m compileall -q common prr_che universal_score keypoint_bertscore
```
