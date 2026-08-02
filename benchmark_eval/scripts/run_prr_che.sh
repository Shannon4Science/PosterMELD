#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:?Usage: $0 MANIFEST [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-outputs/prr_che}"

python -m prr_che.evaluate \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model "${PRR_CHE_MODEL:-gpt-5.5}" \
  --workers-per-key "${WORKERS_PER_KEY:-4}" \
  --max-attempts "${MAX_ATTEMPTS:-6}" \
  --retry-delay "${RETRY_DELAY:-5}"

python -m prr_che.summarize \
  --manifest "$MANIFEST" \
  --result-dir "$OUTPUT_DIR" \
  --report-dir "${REPORT_DIR:-reports/prr_che}"
