from __future__ import annotations

import re
from typing import Any

from common.io import extract_json_object


def parse_prr(raw: str) -> dict[str, Any]:
    full = extract_json_object(raw)
    assessability = str(full.get("assessability", "")).strip().lower()
    if assessability not in {"sufficient", "insufficient"}:
        raise ValueError(f"Invalid PRR assessability: {assessability!r}")
    value = full.get("print_ready")
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        value = value.strip().lower() == "true"
    if not isinstance(value, bool):
        raise ValueError(f"Invalid print_ready value: {value!r}")
    reason = str(full.get("reason", "")).strip()
    if not reason:
        raise ValueError("PRR reason is empty")
    warnings: list[str] = []
    if assessability == "insufficient" and value:
        value = False
        warnings.append("print_ready normalized to false because assessability was insufficient")
    result: dict[str, Any] = {
        "assessability": assessability,
        "print_ready": value,
        "reason": reason,
        "warnings": warnings,
    }
    if isinstance(full.get("checks"), dict):
        result["checks"] = full["checks"]
    return result


def parse_che(raw: str) -> dict[str, Any]:
    full = extract_json_object(raw)
    assessability = str(full.get("assessability", "")).strip().lower()
    if assessability not in {"sufficient", "insufficient"}:
        raise ValueError(f"Invalid CHE assessability: {assessability!r}")
    dimensions: dict[str, dict[str, Any]] = {}
    for key in ("craftsmanship", "harmony", "expressiveness"):
        item = full.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Missing CHE object: {key}")
        score = item.get("score")
        if isinstance(score, str):
            match = re.fullmatch(r"\s*([1-5])(?:\.0+)?\s*", score)
            score = int(match.group(1)) if match else None
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Invalid {key} score: {item.get('score')!r}")
        dimensions[key] = {
            "score": score,
            "reason": str(item.get("reason", "")).strip(),
        }
    result: dict[str, Any] = {
        "assessability": assessability,
        **dimensions,
        "che_score": sum(item["score"] for item in dimensions.values()) / 3.0,
    }
    if isinstance(full.get("evidence"), dict):
        result["evidence"] = full["evidence"]
    return result
