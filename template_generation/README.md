# PosterMELD Demo Generation Utilities

This folder contains the runnable code used to construct the template library described in PosterMELD. It includes two lightweight pipelines:

- `poster_block_extractor/`: converts academic poster images into semantic block JSON files.
- `poster_template_clusterer/`: clusters block layouts and selects center-nearest representative templates.

Generated datasets, extracted outputs, cached files, and private credentials are intentionally excluded from this folder.

## Directory Layout

```text
demo_generation/
|-- poster_block_extractor/
|   |-- pipeline.py
|   |-- batch_process.py
|   |-- config.yaml
|   |-- requirements.txt
|   `-- src/
`-- poster_template_clusterer/
    |-- run.py
    |-- config.yaml
    |-- requirements.txt
    `-- src/
```

## Installation

Create a Python environment from the repository root or from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies for both pipelines:

```bash
pip install -r poster_block_extractor/requirements.txt
pip install -r poster_template_clusterer/requirements.txt
```

If you use the optional local PaddleOCR extractor, install PaddleOCR separately according to your platform and hardware.

## Configuration

The demo configuration files contain placeholders only. Set credentials through environment variables or replace the placeholders in a local, untracked config file.

Required for MinerU extraction:

```bash
export MINERU_TOKEN="your-mineru-token"
```

Required for VLM merging:

```bash
export POSTERMELD_VLM_BASE_URL="https://api.openai.com/v1"
export POSTERMELD_VLM_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:MINERU_TOKEN="your-mineru-token"
$env:POSTERMELD_VLM_BASE_URL="https://api.openai.com/v1"
$env:POSTERMELD_VLM_API_KEY="your-api-key"
```

Do not commit real API keys, MinerU tokens, private endpoints, poster datasets, or generated outputs.

## Step 1: Extract Poster Blocks

Run the extractor for a single poster:

```bash
cd poster_block_extractor
python pipeline.py --image ./data/example_poster.png --output ./output --visualize
```

Run a folder in batch mode:

```bash
python batch_process.py --input ./data/posters --output ./output --limit 5 --visualize
```

The expected output is:

```text
poster_block_extractor/output/
|-- blocks_json/
|   `-- poster_name.json
`-- visualized/
    |-- poster_name_raw.png
    `-- poster_name_merged.png
```

Each block JSON stores normalized coordinates in `[0, 1000]`, semantic labels, member slice IDs, bounding boxes, polygons, and concatenated OCR text.

## Step 2: Cluster Layouts and Select Templates

After block JSON files are available, run:

```bash
cd ../poster_template_clusterer
python run.py --k 8
```

Or sweep multiple cluster counts:

```bash
python run.py --k-sweep 6 8 10 12
```

The clusterer uses spatial layout descriptors only. It does not use semantic labels or text content for clustering. Each poster is represented by global geometry, block count, mean block area, a 12 by 12 occupancy grid, block shape statistics, row/column estimates, and largest-block ratios. Features are z-score standardized, the occupancy segment is weighted, and Ward hierarchical clustering groups posters into layout families.

For each cluster, the sample closest to the cluster center is selected as the representative template for downstream poster generation.

Expected outputs include:

```text
poster_template_clusterer/output/k8/
|-- clusters.json
|-- cluster_summary.json
|-- scatter_pca2.png
|-- cluster_grids/
|-- templates/
|   |-- cluster_0_template.json
|   `-- cluster_0_layout.png
`-- templates.md
```

## Notes

- Coordinates are normalized to `[0, 1000]` with `(0, 0)` at the top-left corner.
- `MinerU` is the default slice extractor. `ppocr` is available as an optional local fallback.
- The block extraction pipeline may call external APIs and therefore requires valid credentials.
- The template clustering pipeline can run offline once block JSON files are available.
- Runtime outputs are ignored by `.gitignore` and should not be committed.
