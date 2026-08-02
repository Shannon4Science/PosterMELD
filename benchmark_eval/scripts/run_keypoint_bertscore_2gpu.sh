#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:?Usage: $0 MANIFEST MINERU_MODEL_DIR BERT_MODEL_DIR [OUTPUT_DIR]}"
MINERU_MODEL="${2:?MinerU model directory is required}"
BERT_MODEL="${3:?BERT model directory is required}"
OUTPUT_DIR="${4:-outputs/keypoint_bertscore}"
LOG_DIR="${LOG_DIR:-logs/keypoint_bertscore}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

CUDA_VISIBLE_DEVICES="$GPU0" python -m keypoint_bertscore.ocr_vllm \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MINERU_MODEL" \
  --rank 0 --world-size 2 \
  --log-dir "$LOG_DIR" &
PID0=$!

CUDA_VISIBLE_DEVICES="$GPU1" python -m keypoint_bertscore.ocr_vllm \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MINERU_MODEL" \
  --rank 1 --world-size 2 \
  --log-dir "$LOG_DIR" &
PID1=$!

wait "$PID0"
wait "$PID1"

CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" torchrun --standalone --nproc_per_node=2 \
  -m keypoint_bertscore.score \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$BERT_MODEL" \
  --batch-size "${BERTSCORE_BATCH_SIZE:-256}" \
  --log-dir "$LOG_DIR"

python -m keypoint_bertscore.summarize \
  --manifest "$MANIFEST" \
  --result-dir "$OUTPUT_DIR" \
  --report-dir "${REPORT_DIR:-reports/keypoint_bertscore}"
