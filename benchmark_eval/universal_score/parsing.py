from __future__ import annotations

import re
from typing import Any

from common.io import extract_json_object


def parse_universal(raw: str, checklist: list[str]) -> tuple[list[dict[str, Any]], str]:
    parsed = extract_json_object(raw)
    raw_items = parsed.get("criteria") or parsed.get("scores") or parsed.get("results") or parsed.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Response JSON must contain a criteria list")

    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        index = item.get("criterion_index", item.get("index", item.get("id")))
        if isinstance(index, str):
            match = re.search(r"\d+", index)
            index = int(match.group(0)) if match else None
        if isinstance(index, int) and 1 <= index <= len(checklist):
            by_index[index] = item
            continue
        description = str(item.get("description") or item.get("criterion") or "").strip()
        if description in checklist:
            by_index[checklist.index(description) + 1] = item

    scores: list[dict[str, Any]] = []
    for index, description in enumerate(checklist, start=1):
        if index not in by_index:
            raise ValueError(f"Missing score for criterion {index}")
        item = by_index[index]
        score = item.get("score")
        if isinstance(score, str):
            match = re.fullmatch(r"\s*([0-5])(?:\.0+)?\s*", score)
            score = int(match.group(1)) if match else None
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError(f"Invalid score for criterion {index}: {item.get('score')!r}")
        scores.append(
            {
                "criterion_index": index,
                "description": description,
                "score": score,
                "max_score": 5,
                "reason": str(item.get("reason") or item.get("rationale") or "").strip(),
            }
        )
    return scores, str(parsed.get("summary", "")).strip()
