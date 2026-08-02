"""Conservative OCR guard for generated decorative images."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def detect_readable_text(
    image_path: str | Path,
    *,
    timeout_seconds: float = 15.0,
    min_confidence: float = 45.0,
) -> Dict[str, Any]:
    """Return OCR tokens that make an allegedly text-free image unsafe.

    The guard is intentionally conservative: decorative backgrounds and teaser
    images must not contain words, labels, numbers, or fake chart annotations.
    """
    executable = shutil.which("tesseract")
    path = Path(image_path)
    if not executable or not path.exists():
        return {
            "available": False,
            "rejected": False,
            "tokens": [],
            "reason": "tesseract_unavailable" if not executable else "image_unavailable",
        }

    try:
        result = subprocess.run(
            [executable, str(path), "stdout", "--psm", "11", "tsv"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 1.0),
        )
    except Exception as exc:  # noqa: BLE001 - OCR is a non-fatal safety aid
        return {
            "available": False,
            "rejected": False,
            "tokens": [],
            "reason": f"ocr_failed:{type(exc).__name__}",
        }

    tokens = []
    try:
        rows = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
        for row in rows:
            text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
            if not text:
                continue
            try:
                confidence = float(row.get("conf") or -1)
            except (TypeError, ValueError):
                confidence = -1
            if confidence < min_confidence or not _looks_readable(text):
                continue
            tokens.append({"text": text[:80], "confidence": round(confidence, 1)})
    except Exception:
        tokens = []

    return {
        "available": True,
        "rejected": bool(tokens),
        "tokens": tokens[:24],
        "reason": "readable_text_detected" if tokens else "",
    }


def _looks_readable(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if len(compact) < 3:
        return False
    letters = re.sub(r"[^A-Za-z]", "", compact)
    digits = re.sub(r"[^0-9]", "", compact)
    return len(letters) >= 3 or len(digits) >= 3
