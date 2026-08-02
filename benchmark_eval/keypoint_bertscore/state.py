from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import ManifestItem, read_json, result_path


OCR_MODEL_NAME = "opendatalab/MinerU2.5-Pro-2605-1.2B"
BERTSCORE_MODEL_NAME = "roberta-large"
BERTSCORE_NUM_LAYERS = 17


def ocr_json_path(output_dir: str | Path, item: ManifestItem) -> Path:
    return result_path(output_dir, item, "ocr.json")


def ocr_text_path(output_dir: str | Path, item: ManifestItem) -> Path:
    return result_path(output_dir, item, "ocr.md")


def score_path(output_dir: str | Path, item: ManifestItem) -> Path:
    return result_path(output_dir, item, "bertscore.json")


def successful(path: str | Path, model: str | None = None) -> bool:
    value: Any = read_json(path, {})
    return bool(
        isinstance(value, dict)
        and value.get("status") == "success"
        and (model is None or value.get("model") == model)
    )
