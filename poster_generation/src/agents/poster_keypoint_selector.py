"""
Poster keypoint selection for keypoint-first template planning.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from src.state.poster_state import PosterState
from src.utils.text_cleanup import normalize_text_for_poster
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class PosterKeypointSelector:
    """Selects poster-worthy keypoints before curator/layout planning."""

    def __init__(self):
        self.name = "poster_keypoint_selector"
        self.prompt = load_prompt("config/prompts/poster_keypoint_annotator.txt")

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "selecting poster-worthy keypoints")

        report: Dict[str, Any] = {
            "source": "llm",
            "target_keypoints": 10,
            "warnings": [],
        }

        try:
            raw_text = str(state.get("raw_text") or "").strip()
            if not raw_text:
                raise ValueError("missing raw_text from parser")

            agent = LangGraphAgent("expert research communication annotator", state["text_model"], state, self.name)
            response = agent.step(self._build_prompt(raw_text, state.get("fast_block_contract")))
            state["tokens"].add_text(response.input_tokens, response.output_tokens)
            payload = extract_json(response.content)
            keypoints, reading_order, normalize_report = self._normalize_payload(payload)
            report.update(normalize_report)

            if len(keypoints) < 8:
                report["warnings"].append(
                    f"LLM returned {len(keypoints)} usable keypoints; falling back to structured sections"
                )
                keypoints, reading_order = self._fallback_from_structured_sections(state)
                if not keypoints:
                    report["warnings"].append(
                        "Structured fallback returned 0 keypoints; falling back to raw text"
                    )
                    keypoints, reading_order = self._fallback_from_raw_text(state)
                    report["source"] = "raw_text_fallback"
                else:
                    report["source"] = "structured_sections_fallback"

        except Exception as exc:
            log_agent_warning(self.name, f"LLM keypoint selection unavailable: {exc}")
            report["warnings"].append(str(exc))
            try:
                keypoints, reading_order = self._fallback_from_structured_sections(state)
                if not keypoints:
                    report["warnings"].append(
                        "Structured fallback returned 0 keypoints; falling back to raw text"
                    )
                    keypoints, reading_order = self._fallback_from_raw_text(state)
                    report["source"] = "raw_text_fallback"
                else:
                    report["source"] = "structured_sections_fallback"
            except Exception as fallback_exc:
                report["warnings"].append(f"structured fallback failed: {fallback_exc}")
                try:
                    keypoints, reading_order = self._fallback_from_raw_text(state)
                    report["source"] = "raw_text_fallback"
                except Exception as raw_exc:
                    log_agent_error(self.name, f"fallback failed: {raw_exc}")
                    state["errors"].append(f"{self.name}: {raw_exc}")
                    return state

        keypoints, reading_order = self._apply_runtime_keypoint_limit(keypoints, reading_order, report)
        state["paper_poster_keypoints"] = keypoints
        state["poster_reading_order"] = reading_order
        report["keypoint_count"] = len(keypoints)
        report["reading_order"] = reading_order
        state["poster_keypoint_selection_report"] = report
        state["current_agent"] = self.name
        self._save_output(state)

        log_agent_success(self.name, f"selected {len(keypoints)} poster keypoints")
        return state

    def _apply_runtime_keypoint_limit(
        self,
        keypoints: List[Dict[str, Any]],
        reading_order: List[int],
        report: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[int]]:
        try:
            limit = int(os.getenv("PAPER2POSTER_KEYPOINT_LIMIT", "0") or 0)
        except ValueError:
            limit = 0
        if limit <= 0 or len(keypoints) <= limit:
            return keypoints, reading_order

        by_id = {int(item["id"]): item for item in keypoints}
        ordered = [by_id[keypoint_id] for keypoint_id in reading_order if keypoint_id in by_id]
        ordered.extend(item for item in keypoints if item not in ordered)
        base_size, larger_groups = divmod(len(ordered), limit)
        condensed: List[Dict[str, Any]] = []
        cursor = 0
        for index in range(limit):
            group_size = base_size + (1 if index < larger_groups else 0)
            group = ordered[cursor : cursor + group_size]
            cursor += group_size
            sections = list(dict.fromkeys(str(item.get("section") or "Paper") for item in group))
            condensed.append(
                {
                    "id": index + 1,
                    "key_point": " ".join(str(item.get("key_point") or "").strip() for item in group).strip(),
                    "section": " / ".join(sections[:2]) or "Paper",
                    "source_keypoint_ids": [int(item["id"]) for item in group],
                }
            )

        report["target_keypoints"] = limit
        report.setdefault("warnings", []).append(
            f"Runtime recovery condensed {len(keypoints)} keypoints into {limit} sections without dropping facts"
        )
        return condensed, [item["id"] for item in condensed]

    def _build_prompt(self, raw_text: str, fast_block_contract: Dict[str, Any] | None = None) -> str:
        fast_guidance = ""
        if fast_block_contract:
            block_count = len(fast_block_contract.get("blocks") or [])
            blocks = []
            for block in fast_block_contract.get("blocks") or []:
                blocks.append({
                    "slot_id": block.get("slot_id"),
                    "role": block.get("slot_role"),
                    "visual_policy": block.get("visual_policy"),
                    "target_chars": block.get("target_chars"),
                    "source_keypoint_ids": block.get("source_keypoint_ids"),
                })
            fast_guidance = f"""

Fast template-first capacity context:
- The selected poster template is {fast_block_contract.get("template_id")} with {block_count} visual blocks.
- Downstream curator will group the 10 keypoints into these {block_count} blocks, usually 1-3 keypoints per block depending on slot count.
- Ensure the 10 keypoints cover motivation, method/architecture, retrieval or system flow when present, main results, and robustness/evaluation.
- Prefer facts that can support 2 figure blocks, 1-2 table/results blocks, and 2-3 text-heavy blocks.
- Slot capacity summary:
{json.dumps(blocks, ensure_ascii=False, indent=2)}
""".rstrip()
        return f"""
{self.prompt}

Additional runtime constraint for this PosterMELD pipeline:
- Prefer exactly 10 key points as a content pool for poster planning.
- If the paper is sparse, use 8-10 key points.
- If you return 11-12 key points, the pipeline will keep only the first 10 in reading_order.
- Downstream template planning may group related key points into fewer visual blocks when the selected template has fewer high-quality content panels.
- Do not invent facts, result numbers, datasets, or claims.
{fast_guidance}

Paper text:
```text
{raw_text}
```
""".strip()

    def _normalize_payload(self, payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
        if isinstance(payload, list):
            raw_keypoints = payload
            raw_order = []
        elif isinstance(payload, dict):
            raw_keypoints = payload.get("paper_poster_keypoints") or payload.get("keypoints") or []
            raw_order = payload.get("reading_order") or []
        else:
            raw_keypoints = []
            raw_order = []
        if isinstance(raw_keypoints, dict):
            raw_keypoints = list(raw_keypoints.values())
        if not isinstance(raw_keypoints, list):
            raw_keypoints = []

        normalized_by_id: Dict[int, Dict[str, Any]] = {}
        next_id = 1
        for item in raw_keypoints:
            if isinstance(item, str):
                text = normalize_text_for_poster(item.strip())
                section = "Paper"
                raw_id = next_id
            elif isinstance(item, dict):
                text = normalize_text_for_poster(str(item.get("key_point") or item.get("keypoint") or item.get("text") or "").strip())
                section = normalize_text_for_poster(str(item.get("section") or item.get("source_section") or "").strip())
                raw_id = item.get("id")
            else:
                continue
            if not text:
                continue
            try:
                keypoint_id = int(raw_id)
            except (TypeError, ValueError):
                keypoint_id = next_id
            while keypoint_id in normalized_by_id:
                keypoint_id += 1
            normalized_by_id[keypoint_id] = {
                "id": keypoint_id,
                "key_point": text,
                "section": section or "Unknown",
            }
            next_id = max(next_id, keypoint_id + 1)

        order: List[int] = []
        if isinstance(raw_order, list):
            for value in raw_order:
                try:
                    keypoint_id = int(value)
                except (TypeError, ValueError):
                    continue
                if keypoint_id in normalized_by_id and keypoint_id not in order:
                    order.append(keypoint_id)
        for keypoint_id in sorted(normalized_by_id):
            if keypoint_id not in order:
                order.append(keypoint_id)

        kept_ids = order
        warnings = []
        dropped_ids: List[int] = []
        if len(kept_ids) > 10:
            dropped_ids = kept_ids[10:]
            kept_ids = kept_ids[:10]
            warnings.append(f"Returned {len(order)} keypoints; kept first 10 by reading_order")

        renumbered: List[Dict[str, Any]] = []
        for new_id, old_id in enumerate(kept_ids, start=1):
            item = dict(normalized_by_id[old_id])
            item["original_id"] = old_id
            item["id"] = new_id
            renumbered.append(item)

        return renumbered, [item["id"] for item in renumbered], {
            "original_keypoint_count": len(normalized_by_id),
            "dropped_original_ids": dropped_ids,
            "warnings": warnings,
        }

    def _fallback_from_structured_sections(self, state: PosterState) -> tuple[List[Dict[str, Any]], List[int]]:
        structured_sections = state.get("structured_sections") or {}
        paper_sections = structured_sections.get("paper_sections") or []
        keypoints: List[Dict[str, Any]] = []
        seen = set()

        for paper_section in paper_sections:
            section_name = str(paper_section.get("section_name") or paper_section.get("section_type") or "Paper").strip()
            candidates = paper_section.get("key_points") or []
            if not candidates and paper_section.get("content"):
                candidates = [str(paper_section.get("content"))[:240]]
            for candidate in candidates:
                text = normalize_text_for_poster(str(candidate or "").strip())
                if len(text) < 12:
                    continue
                key = text.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                keypoints.append({
                    "id": len(keypoints) + 1,
                    "key_point": text,
                    "section": section_name,
                    "original_id": len(keypoints) + 1,
                })
                if len(keypoints) >= 10:
                    break
            if len(keypoints) >= 10:
                break

        if len(keypoints) < 1:
            raise ValueError("no fallback keypoints available from structured_sections")
        return keypoints, [item["id"] for item in keypoints]

    def _fallback_from_raw_text(self, state: PosterState) -> tuple[List[Dict[str, Any]], List[int]]:
        raw_text = str(state.get("raw_text") or "")
        sentences = self._split_sentences(raw_text)
        if not sentences:
            raise ValueError("no raw_text sentences available for keypoint fallback")

        keywords = {
            "problem", "challenge", "propose", "framework", "method", "model", "arena",
            "elo", "evaluation", "stable", "stability", "uncertainty", "experiment",
            "result", "benchmark", "outperform", "improve", "ranking", "annotator",
        }
        scored = []
        for index, sentence in enumerate(sentences[:240]):
            text = normalize_text_for_poster(sentence)
            if len(text) < 45 or len(text) > 260:
                continue
            lowered = text.lower()
            if "references" in lowered or "arxiv" in lowered:
                continue
            score = sum(1 for keyword in keywords if keyword in lowered)
            score += max(0, 4 - index / 60)
            if score <= 0:
                continue
            scored.append((score, index, text))

        if not scored:
            raise ValueError("no poster-worthy raw_text sentences found")
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(scored[:10], key=lambda item: item[1])
        keypoints = [
            {
                "id": idx,
                "key_point": text,
                "section": "Paper",
                "original_id": idx,
            }
            for idx, (_, _, text) in enumerate(selected, start=1)
        ]
        return keypoints, [item["id"] for item in keypoints]

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        return [
            sentence.strip(" -\t\n")
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
            if sentence.strip()
        ]

    def _save_output(self, state: PosterState) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_poster_keypoints": state.get("paper_poster_keypoints") or [],
            "reading_order": state.get("poster_reading_order") or [],
            "report": state.get("poster_keypoint_selection_report") or {},
        }
        with open(output_dir / "poster_keypoint_selection.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def poster_keypoint_selector_node(state: PosterState) -> Dict[str, Any]:
    result = PosterKeypointSelector()(state)
    return {
        **state,
        "paper_poster_keypoints": result.get("paper_poster_keypoints"),
        "poster_reading_order": result.get("poster_reading_order"),
        "poster_keypoint_selection_report": result.get("poster_keypoint_selection_report"),
        "tokens": result.get("tokens"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
