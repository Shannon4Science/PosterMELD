#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:?Usage: $0 MANIFEST [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-outputs/universal}"

python -m universal_score.evaluate \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model "${UNIVERSAL_MODEL:-gpt-4o}" \
  --workers-per-key "${WORKERS_PER_KEY:-4}" \
  --max-attempts "${MAX_ATTEMPTS:-6}" \
  --retry-delay "${RETRY_DELAY:-5}"

python -m universal_score.summarize \
  --manifest "$MANIFEST" \
  --result-dir "$OUTPUT_DIR" \
  --report-dir "${REPORT_DIR:-reports/universal}"
