"""
Block-level content refinement loop.

This agent is the only block-refinement step that edits content. It modifies
only story_board.spatial_content_plan.sections[*].text_content and leaves slot,
section, and visual references intact.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.utils.text_cleanup import fit_complete_sentence_prefix, normalize_text_for_poster
from utils.langgraph_utils import LangGraphAgent, extract_json
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class BlockContentRefiner:
    def __init__(self):
        self.name = "block_content_refiner"
        self.config = load_config()
        self.block_config = self.config.get("block_refinement", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_block_vlm_review", False):
            return state

        log_agent_info(self.name, "refining section text_content for block utilization")

        try:
            patch_report = self.refine(state)
            state["block_content_patch"] = patch_report
            state["block_refinement_required"] = bool(patch_report.get("applied"))
            state["current_agent"] = self.name
            self._save_outputs(state, patch_report)
            log_agent_success(
                self.name,
                f"content refinement complete: applied={patch_report.get('applied')}, patches={len(patch_report.get('patches', []))}",
            )
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def refine(self, state: PosterState) -> Dict[str, Any]:
        story_board = deepcopy(state.get("story_board") or {})
        sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
        if not sections:
            return self._empty_patch("missing story_board sections")

        actions = self._decide_actions(state)
        if not actions:
            return self._empty_patch("no blocks require content changes")

        section_by_id = {
            str(section.get("section_id")): section
            for section in sections
            if section.get("section_id")
        }
        expand_actions = [action for action in actions if action["action"] == "expand"]
        rewrite_patches = self._generate_expansion_patches(state, expand_actions, section_by_id)

        applied_patches: List[Dict[str, Any]] = []
        history = deepcopy(state.get("block_refinement_history") or {})
        for action in actions:
            section = section_by_id.get(action["section_id"])
            if not section:
                continue
            before = list(section.get("text_content") or [])
            after = list(before)

            if action["action"] == "expand":
                patch = rewrite_patches.get(action["section_id"]) or {}
                rewritten = patch.get("rewritten_bullets")
                if rewritten is None and patch.get("new_bullets") is not None:
                    rewritten = self._fallback_rewritten_bullets(before, patch.get("new_bullets") or [], action)
                after = self._apply_rewrite(before, rewritten or [], action)
            elif action["action"] == "reduce":
                if state.get("template_fast_mode"):
                    after = self._reduce_bullets_fast(before, action, section)
                else:
                    after = self._reduce_bullets(before, action)

            if after == before:
                continue

            section["text_content"] = after
            history[action["slot_id"]] = {
                "last_action": action["action"],
                "section_id": action["section_id"],
                "last_utilization": action.get("utilization"),
            }
            applied_patches.append({
                "slot_id": action["slot_id"],
                "section_id": action["section_id"],
                "action": action["action"],
                "target_extra_chars": action.get("target_extra_chars", 0),
                "before_bullets": len(before),
                "after_bullets": len(after),
                "before_chars": sum(len(str(item)) for item in before),
                "after_chars": sum(len(str(item)) for item in after),
                "reason": action.get("reason", ""),
            })

        patch_report = {
            "source": self.name,
            "applied": bool(applied_patches),
            "iteration": int(state.get("block_refinement_count", 0)) + (1 if applied_patches else 0),
            "actions_considered": actions,
            "patches": applied_patches,
            "warnings": [],
        }

        if applied_patches:
            state["story_board"] = story_board
            state["block_refinement_history"] = history
            state["block_refinement_count"] = int(state.get("block_refinement_count", 0)) + 1
            self._reset_downstream_state(state)
        else:
            state["block_refinement_required"] = False

        return patch_report

    def _decide_actions(self, state: PosterState) -> List[Dict[str, Any]]:
        occupancy = state.get("block_occupancy_report") or {}
        vlm_review = state.get("block_vlm_review") or {}
        vlm_by_slot = {
            str(item.get("slot_id")): item
            for item in vlm_review.get("blocks", [])
            if isinstance(item, dict) and item.get("slot_id")
        }
        history = state.get("block_refinement_history") or {}
        settings = occupancy.get("settings") or {}
        acceptable_min = float(settings.get("acceptable_min", self.block_config.get("acceptable_min", 0.90)))
        hard_max = float(settings.get("hard_max", self.block_config.get("hard_max", 0.98)))
        loop_index = int(state.get("block_refinement_count", 0))
        fast_mode = bool(state.get("template_fast_mode"))
        fast_config = self.config.get("template_fast_mode", {})
        fast_underfill_threshold = float(fast_config.get("emergency_underfill_threshold", 0.85))
        fast_hard_min = float(fast_config.get("hard_min_utilization", self.block_config.get("final_min_utilization", 0.88)))
        allow_text_fill_repair = bool(fast_config.get("allow_text_fill_repair", True))
        fast_text_fill_cap = int(fast_config.get("fast_text_fill_max_extra_chars", 280))
        protected_teaser_sections = self._protected_teaser_sections(state)
        teaser_max_extra_chars = int(self.block_config.get("teaser_max_extra_chars", 90))
        section_by_id = {
            str(section.get("section_id") or ""): section
            for section in (((state.get("story_board") or {}).get("spatial_content_plan") or {}).get("sections") or [])
            if section.get("section_id")
        }

        actions = []
        for block in occupancy.get("blocks", []):
            slot_id = str(block.get("slot_id") or "")
            section_id = str(block.get("section_id") or "")
            if not slot_id or not section_id:
                continue
            target_extra_chars = int(block.get("target_extra_chars") or 0)
            teaser_protected = section_id in protected_teaser_sections
            if teaser_protected and not (
                block.get("action") == "expand"
                and 0 < target_extra_chars <= teaser_max_extra_chars
            ):
                continue

            vlm = vlm_by_slot.get(slot_id, {})
            status = str(vlm.get("status") or "").lower()
            severity = str(vlm.get("severity") or "low").lower()
            utilization = float(block.get("utilization") or 0.0)
            action = "keep"
            reason = block.get("reason", "")

            vlm_crowded = (
                status == "overflow"
                or (status == "crowded" and severity == "high")
                or (status == "crowded" and utilization > hard_max)
            )
            vlm_underfilled = (
                status in {"underfilled", "empty"}
                and severity in {"medium", "high"}
                and utilization < acceptable_min
            )
            geometry_expand_requested = block.get("action") == "expand" and target_extra_chars > 0
            bottom_gap_forces_rewrite = self._bottom_whitespace_exceeds_final_limit(block)
            final_gate_repair = bool(block.get("final_gate_repair"))
            visual_too_small = (
                status == "visual_too_small"
                and severity in {"medium", "high"}
                and int(block.get("visual_count") or 0) > 0
            )
            if visual_too_small and utilization < acceptable_min:
                action = "expand"
                if target_extra_chars <= 0:
                    target_extra_chars = int(self.block_config.get("vlm_underfilled_min_extra_chars", 120))
                reason = (
                    vlm.get("description")
                    or reason
                    or "visual is small but the block is underfilled; add concise caption/result context"
                )
            elif visual_too_small:
                action = "reduce"
                reason = "compress text to prioritize visual scale for unreadable figure/table labels"
            elif (
                block.get("action") == "reduce"
                or utilization > hard_max
                or (vlm_crowded and not (geometry_expand_requested and severity != "high"))
            ):
                action = "reduce"
                reason = vlm.get("description") or reason or "block is crowded or overflowing"
            elif (
                (
                    geometry_expand_requested
                    or vlm_underfilled
                )
                and status not in {"overflow", "visual_too_small"}
                and not (status == "crowded" and severity == "high")
            ):
                action = "expand"
                if vlm_underfilled:
                    target_extra_chars = max(
                        target_extra_chars,
                        int(self.block_config.get("underfilled_min_extra_chars", 70)),
                    )
                if bottom_gap_forces_rewrite and not teaser_protected:
                    target_extra_chars = max(
                        target_extra_chars,
                        int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
                    )
                reason = vlm.get("description") or reason or "block is underfilled"

            if fast_mode and action == "expand":
                if visual_too_small and utilization >= fast_hard_min and block.get("action") != "expand":
                    action = "keep"
                    reason = "fast mode keeps visual_too_small block unchanged because utilization is above hard minimum"
                elif not allow_text_fill_repair and not visual_too_small and utilization >= fast_underfill_threshold:
                    action = "keep"
                    reason = "fast mode skips non-emergency underfill repair"
                else:
                    target_extra_chars = min(max(target_extra_chars, int(self.block_config.get("min_extra_chars", 40))), fast_text_fill_cap)
            if teaser_protected and action == "expand":
                target_extra_chars = min(target_extra_chars, teaser_max_extra_chars)
            if (
                fast_mode
                and action == "reduce"
                and status == "crowded"
                and severity != "high"
                and utilization < 1.02
            ):
                action = "keep"
                reason = "fast mode keeps medium crowded block because there is no overflow"

            previous = history.get(slot_id, {})
            if previous.get("last_action") == "expand" and action == "reduce" and not (vlm_crowded or utilization > hard_max):
                action = "keep"
                reason = "suppressed expand/reduce oscillation"
            elif previous.get("last_action") == "reduce" and action == "expand" and utilization >= acceptable_min - 0.03:
                action = "keep"
                reason = "suppressed reduce/expand oscillation"

            if action == "expand" and loop_index >= 1:
                second_round_cap = int(self.block_config.get("second_round_max_extra_chars", 220))
                if previous.get("last_action") != "expand":
                    target_extra_chars = min(
                        max(target_extra_chars, int(self.block_config.get("min_extra_chars", 40))),
                        second_round_cap,
                    )
                elif utilization < float(self.block_config.get("final_min_utilization", 0.88)):
                    target_extra_chars = min(
                        max(target_extra_chars, int(self.block_config.get("vlm_underfilled_min_extra_chars", 120))),
                        second_round_cap,
                    )
                else:
                    scale = float(self.block_config.get("second_round_expand_scale", 0.45))
                    target_extra_chars = min(max(int(target_extra_chars * scale), 0), second_round_cap)
                if target_extra_chars < int(self.block_config.get("min_extra_chars", 40)):
                    action = "keep"
                    reason = "second-round safe extra budget is below minimum"

            if action == "expand":
                safe_extra_chars = self._safe_extra_chars_for_block(block)
                if final_gate_repair and bottom_gap_forces_rewrite:
                    target_extra_chars = max(
                        target_extra_chars,
                        int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
                    )
                elif safe_extra_chars is not None:
                    if safe_extra_chars > 0:
                        target_extra_chars = min(target_extra_chars, safe_extra_chars)
                    elif bottom_gap_forces_rewrite:
                        target_extra_chars = min(
                            max(
                                target_extra_chars,
                                int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
                            ),
                            int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
                        )
                    if target_extra_chars < int(self.block_config.get("min_extra_chars", 10)):
                        action = "keep"
                        reason = "geometry-safe extra budget is below minimum"

            if action != "keep":
                section = section_by_id.get(section_id, {})
                budget = section.get("capacity_budget") or {}
                current_items = section.get("text_content") or []
                target_bullets = int(section.get("target_bullets") or budget.get("target_bullets") or 0)
                max_final_bullets = (
                    max(len(current_items), target_bullets)
                    if target_bullets > 0
                    else len(current_items) + self._max_new_bullets(target_extra_chars)
                )
                capacity_max_chars = int(section.get("max_chars") or budget.get("max_chars") or 0)
                max_final_chars = (
                    max(self._bullet_chars(current_items), capacity_max_chars)
                    if capacity_max_chars > 0
                    else 0
                )
                if (
                    action == "expand"
                    and len(current_items) == 1
                    and (final_gate_repair or bottom_gap_forces_rewrite)
                    and target_extra_chars >= int(self.block_config.get("min_extra_chars", 10))
                ):
                    max_final_bullets = max(max_final_bullets, 2)
                actions.append({
                    "slot_id": slot_id,
                    "section_id": section_id,
                    "action": action,
                    "target_extra_chars": target_extra_chars,
                    "utilization": utilization,
                    "vlm_status": status or None,
                    "vlm_severity": severity,
                    "max_final_bullets": max_final_bullets,
                    "max_final_chars": max_final_chars,
                    "reason": reason,
                })

        return actions

    def _bottom_whitespace_exceeds_final_limit(self, block: Dict[str, Any]) -> bool:
        bottom_whitespace = float(block.get("bottom_whitespace") or 0.0)
        available_height = float(block.get("available_height") or 0.0)
        if bottom_whitespace <= 0 or available_height <= 0:
            return False
        max_inches = float(self.block_config.get("final_max_bottom_whitespace_inches", 0.18) or 0.18)
        max_fraction = float(self.block_config.get("final_max_bottom_whitespace_fraction", 0.012) or 0.012)
        line_fraction = float(self.block_config.get("final_max_bottom_whitespace_line_fraction", 0.0) or 0.0)
        allowed = min(
            value
            for value in (
                max_inches,
                available_height * max_fraction if max_fraction > 0 else max_inches,
            )
            if value > 0
        )
        line_height = float(block.get("line_height") or 0.0)
        if line_height > 0 and line_fraction > 0:
            allowed = max(allowed, line_height * line_fraction)
        return bottom_whitespace > allowed

    def _safe_extra_chars_for_block(self, block: Dict[str, Any]) -> Optional[int]:
        available_height = float(block.get("available_height") or 0.0)
        used_height = float(block.get("visible_content_height") or block.get("used_height") or 0.0)
        line_height = float(block.get("line_height") or 0.0)
        chars_per_line = int(block.get("chars_per_line") or 0)
        if available_height <= 0 or used_height <= 0 or line_height <= 0 or chars_per_line <= 0:
            return None
        safety = float(self.block_config.get("expand_height_safety_inches", 0.14))
        remaining_height = max(available_height - used_height - safety, 0.0)
        safe_lines = int(remaining_height / line_height)
        bottom_whitespace = float(block.get("bottom_whitespace") or 0.0)
        if bottom_whitespace > 0:
            near_line_tolerance = line_height * 0.04
            safe_lines = max(safe_lines, int((bottom_whitespace + near_line_tolerance) / line_height))
        return max(safe_lines, 0) * chars_per_line

    def _protected_teaser_sections(self, state: PosterState) -> set[str]:
        sections = ((state.get("story_board") or {}).get("spatial_content_plan") or {}).get("sections") or []
        protected: set[str] = set()
        for section in sections:
            if section.get("generated_teaser_summary"):
                section_id = str(section.get("section_id") or "")
                if section_id:
                    protected.add(section_id)
                continue
            for visual in section.get("visual_assets") or []:
                if str(visual.get("visual_id") or "").startswith("generated_teaser"):
                    section_id = str(section.get("section_id") or "")
                    if section_id:
                        protected.add(section_id)
                    break
        return protected

    def _reduce_bullets_fast(
        self,
        bullets: List[Any],
        action: Dict[str, Any],
        section: Dict[str, Any],
    ) -> List[str]:
        cleaned = [normalize_text_for_poster(str(item or "").strip()) for item in bullets if str(item or "").strip()]
        if not cleaned:
            return []

        budget = section.get("capacity_budget") or {}
        min_chars = int(section.get("min_chars") or budget.get("min_chars") or 0)
        max_chars = int(section.get("max_chars") or budget.get("max_chars") or 0)
        current_chars = self._bullet_chars(cleaned)
        status = str(action.get("vlm_status") or "").lower()

        if current_chars <= max(min_chars, 1):
            return cleaned

        if status == "visual_too_small":
            scale = 0.80
        elif status == "overflow":
            scale = 0.86
        else:
            scale = 0.92
        target_chars = max(min_chars, int(current_chars * scale))
        if max_chars > 0:
            target_chars = min(target_chars, max_chars)
        if current_chars - target_chars < int(self.block_config.get("min_extra_chars", 20)):
            return cleaned

        result = list(cleaned)
        min_item_len = 72 if len(result) > 1 else max(72, min_chars)
        guard = 0
        while self._bullet_chars(result) > target_chars and guard < 20:
            guard += 1
            idx = max(range(len(result)), key=lambda index: len(result[index]))
            current_len = len(result[idx])
            if current_len <= min_item_len:
                break
            overage = self._bullet_chars(result) - target_chars
            shrink_by = min(current_len - min_item_len, max(18, overage))
            candidate = self._truncate_on_word_boundary(result[idx], current_len - shrink_by)
            if candidate == result[idx]:
                break
            result[idx] = candidate

        if self._bullet_chars(result) < min_chars and current_chars >= min_chars:
            return self._restore_until_min_chars(cleaned, result, min_chars)
        return result

    def _restore_until_min_chars(self, original: List[str], reduced: List[str], min_chars: int) -> List[str]:
        result = list(reduced)
        for index, original_item in sorted(enumerate(original), key=lambda item: len(item[1]), reverse=True):
            if self._bullet_chars(result) >= min_chars:
                break
            if index < len(result):
                result[index] = original_item
        return result

    def _bullet_chars(self, bullets: List[Any]) -> int:
        return sum(len(str(item or "")) for item in bullets or [])

    def _generate_expansion_patches(
        self,
        state: PosterState,
        actions: List[Dict[str, Any]],
        section_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not actions:
            return {}

        prompt_payload = []
        for action in actions:
            section = section_by_id.get(action["section_id"], {})
            prompt_payload.append({
                "slot_id": action["slot_id"],
                "section_id": action["section_id"],
                "section_title": section.get("section_title"),
                "current_bullets": section.get("text_content") or [],
                "target_extra_chars": action.get("target_extra_chars", 0),
                "max_new_bullets": self._max_new_bullets(action.get("target_extra_chars", 0)),
                "max_final_bullets": action.get("max_final_bullets"),
                "max_final_chars": action.get("max_final_chars"),
                "source_context": self._source_context_for_section(state, section, action),
            })

        try:
            agent = LangGraphAgent(
                "expert academic poster editor who expands bullets only from supplied paper facts",
                state["text_model"],
                state,
                self.name,
            )
            response = agent.step(self._build_expansion_prompt(prompt_payload))
            state["tokens"].add_text(response.input_tokens, response.output_tokens)
            payload = extract_json(response.content)
            patches = payload.get("patches") or []
            normalized = {}
            for patch in patches:
                if not isinstance(patch, dict):
                    continue
                section_id = str(patch.get("section_id") or "")
                if section_id in section_by_id:
                    rewritten = patch.get("rewritten_bullets")
                    if rewritten is None and patch.get("new_bullets") is not None:
                        rewritten = self._fallback_rewritten_bullets(
                            section_by_id.get(section_id, {}).get("text_content") or [],
                            self._clean_bullets(patch.get("new_bullets") or []),
                            next((action for action in actions if action["section_id"] == section_id), {}),
                        )
                    normalized[section_id] = {
                        "rewritten_bullets": self._clean_bullets(rewritten or []),
                    }
            for action in actions:
                section_id = action["section_id"]
                if normalized.get(section_id, {}).get("rewritten_bullets"):
                    continue
                normalized[section_id] = {
                    "rewritten_bullets": self._fallback_rewritten_bullets(
                        section_by_id.get(section_id, {}).get("text_content") or [],
                        self._fallback_new_bullets(
                            self._source_context_for_section(state, section_by_id.get(section_id, {}), action),
                            section_by_id.get(section_id, {}).get("text_content") or [],
                            action,
                        ),
                        action,
                    )
                }
            return normalized
        except Exception as exc:
            log_agent_warning(self.name, f"LLM expansion unavailable; using source sentence fallback: {exc}")
            return {
                action["section_id"]: {
                    "rewritten_bullets": self._fallback_rewritten_bullets(
                        section_by_id.get(action["section_id"], {}).get("text_content") or [],
                        self._fallback_new_bullets(
                            self._source_context_for_section(state, section_by_id.get(action["section_id"], {}), action),
                            section_by_id.get(action["section_id"], {}).get("text_content") or [],
                            action,
                        ),
                        action,
                    )
                }
                for action in actions
            }

    def _build_expansion_prompt(self, blocks: List[Dict[str, Any]]) -> str:
        return f"""
Rewrite selected academic poster blocks with concise bullets.

Rules:
- Use only facts present in source_context.
- Do not invent experimental results, numbers, datasets, claims, or citations.
- Do not modify section_id, slot_id, titles, or visuals.
- Return the complete rewritten_bullets list for the block, not only added text.
- Rewrite and rebalance the existing bullets; do not simply append a visibly separate final note.
- Keep all important facts from current_bullets unless they are redundant or low-value.
- Target final length should be close to current length plus target_extra_chars, but never exceed that by more than 15%.
- Never return more than max_final_bullets items; expand existing items instead of adding another paragraph when that limit is reached.
- When max_final_chars is a positive number, never exceed it in total; shorten semantically or use fewer complete items.
- Prefer compact poster text items. Each item should be self-contained, complete, and 8-22 words.
- Do not include literal bullet symbols, nested bullets, ordered-list prefixes, empty strings, or multiline items.
- Keep new items parallel with existing block style; use bold lead-ins only when they improve scanability.
- Do not mention table or figure numbers such as "Table 2" or "Figure 3"; summarize the finding directly.
- Do not output file paths, page ids, markdown image syntax, raw captions, bibliography text, citation lists, or metadata fields such as section_name, section_type, contains_figures, contains_tables, or importance.
- If a table or figure is not displayed in the target block, do not refer to it as an external visual.

Return strict JSON only:
{{
  "patches": [
    {{
      "section_id": "same id",
      "slot_id": "same slot",
      "rewritten_bullets": ["complete rewritten poster text item", "..."]
    }}
  ]
}}

Blocks:
{json.dumps(blocks, ensure_ascii=False, indent=2)}
"""

    def _apply_rewrite(self, current: List[Any], rewritten_bullets: List[str], action: Dict[str, Any]) -> List[str]:
        cleaned_current = self._clean_bullets(current)
        cleaned_rewrite = self._clean_bullets(rewritten_bullets)
        if not cleaned_rewrite:
            return cleaned_current

        target_extra = int(action.get("target_extra_chars") or 0)
        current_chars = self._bullet_chars(cleaned_current)
        complete_sentence_budget = int(self.block_config.get("near_line_rewrite_extra_chars", 80))
        max_final_chars = current_chars + max(int(target_extra * 1.15), target_extra, complete_sentence_budget)
        capacity_max_chars = int(action.get("max_final_chars") or 0)
        if capacity_max_chars > 0:
            max_final_chars = min(max_final_chars, max(current_chars, capacity_max_chars))
        if max_final_chars <= current_chars:
            max_final_chars = max(current_chars, self._bullet_chars(cleaned_rewrite))

        deduped: List[str] = []
        seen: set[str] = set()
        for bullet in cleaned_rewrite:
            key = self._dedupe_key(bullet)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(bullet)

        if not deduped:
            return cleaned_current

        max_final_bullets = int(action.get("max_final_bullets") or 0)
        if max_final_bullets > 0:
            deduped = deduped[:max_final_bullets]

        while self._bullet_chars(deduped) > max_final_chars and deduped:
            idx = max(range(len(deduped)), key=lambda index: len(deduped[index]))
            overage = self._bullet_chars(deduped) - max_final_chars
            target_len = max(55, len(deduped[idx]) - max(18, overage))
            shortened = self._truncate_on_word_boundary(deduped[idx], target_len)
            if shortened == deduped[idx]:
                if len(deduped) > 1:
                    deduped.pop(idx)
                    continue
                break
            deduped[idx] = shortened

        if self._bullet_chars(deduped) <= self._bullet_chars(cleaned_current) and target_extra > 0:
            return cleaned_current
        return deduped

    def _fallback_rewritten_bullets(
        self,
        current: List[Any],
        extra_bullets: List[str],
        action: Dict[str, Any],
    ) -> List[str]:
        cleaned_current = self._clean_bullets(current)
        cleaned_extra = self._clean_bullets(extra_bullets)
        if not cleaned_extra:
            return cleaned_current

        target_extra = int(action.get("target_extra_chars") or 0)
        max_added_chars = max(
            target_extra,
            int(target_extra * 1.05),
            int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
        )
        if max_added_chars <= 0:
            max_added_chars = int(self.block_config.get("near_line_rewrite_extra_chars", 80))

        result = list(cleaned_current)
        seen = {self._dedupe_key(item) for item in result}
        added_chars = 0
        final_char_limit = int(action.get("max_final_chars") or 0)
        for bullet in cleaned_extra:
            key = self._dedupe_key(bullet)
            if not key or key in seen:
                continue
            remaining = max_added_chars - added_chars
            if remaining <= 0:
                break
            candidate = bullet
            if len(candidate) > remaining:
                if remaining < 30:
                    break
                candidate = self._truncate_on_word_boundary(candidate, remaining)
                if len(candidate) > remaining:
                    continue
            if final_char_limit > 0 and self._bullet_chars(result) + len(candidate) > final_char_limit:
                continue
            result.append(candidate)
            seen.add(key)
            added_chars += len(candidate)
        return result

    def _reduce_bullets(self, current: List[Any], action: Dict[str, Any]) -> List[str]:
        bullets = self._clean_bullets(current)
        if not bullets:
            return bullets

        reduced = list(bullets)
        status = str(action.get("vlm_status") or "").lower()
        utilization = float(action.get("utilization") or 1.0)
        if status == "visual_too_small":
            return [self._truncate_on_word_boundary(reduced[0], 115)] if reduced else reduced

        if len(reduced) > 3:
            reduced = reduced[:-2] if status == "overflow" or utilization > 1.08 else reduced[:-1]
        elif len(reduced) > 1:
            reduced = reduced[:-1]

        max_len = 150 if status == "overflow" or utilization > 1.0 else 180
        shortened = [
            self._truncate_on_word_boundary(item, max_len)
            if len(item) > max_len
            else item
            for item in reduced
        ]

        if shortened == bullets and shortened:
            shortened[-1] = self._truncate_on_word_boundary(shortened[-1], max(60, int(len(shortened[-1]) * 0.75)))
        return shortened

    def _reset_downstream_state(self, state: PosterState) -> None:
        state["optimized_story_board"] = None
        state["optimized_column_assignment"] = None
        state["balancer_decisions"] = None
        state["initial_layout_data"] = None
        state["column_analysis"] = None
        state["design_layout"] = None
        state["styled_layout"] = None
        state["header_block_review"] = None
        state["header_block_patch_applied"] = False
        state["visual_legibility_review"] = None
        state["vlm_layout_review"] = None
        state["vlm_layout_patch"] = None
        state["block_occupancy_report"] = None
        state["block_vlm_review"] = None
        state["vlm_reflow_required"] = False
        state["vlm_patch_applied"] = False
        state["template_repair_required"] = False
        state["template_repair_decision"] = None
        state["adaptive_relayout_required"] = False
        state["adaptive_layout_decision"] = None
        state["render_stage"] = "draft"
        state["draft_status"] = "pending"
        state["final_poster_accepted"] = False

    def _source_context_for_section(
        self,
        state: PosterState,
        section: Dict[str, Any],
        action: Dict[str, Any],
    ) -> str:
        title = str(section.get("section_title") or action.get("section_id") or "")
        source_text = "\n".join(
            [
                self._stringify_source(state.get("structured_sections")),
                self._stringify_source(state.get("narrative_content")),
                str(state.get("raw_text") or ""),
            ]
        )
        sentences = self._split_sentences(source_text)
        query_terms = self._terms(f"{title} {action.get('section_id', '')}")
        scored = []
        for sentence in sentences:
            terms = self._terms(sentence)
            overlap = len(query_terms & terms)
            if overlap or len(scored) < 12:
                scored.append((overlap, len(sentence), sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in scored[:18]]
        context = "\n".join(selected).strip()
        return context[: int(self.block_config.get("source_context_chars", 7000))]

    def _fallback_new_bullets(
        self,
        source_context: str,
        current: List[Any],
        action: Dict[str, Any],
    ) -> List[str]:
        target = int(action.get("target_extra_chars") or 0)
        max_added_chars = max(
            target,
            int(target * 1.05),
            int(self.block_config.get("near_line_rewrite_extra_chars", 80)),
        )
        final_char_limit = int(action.get("max_final_chars") or 0)
        if final_char_limit > 0:
            current_chars = self._bullet_chars(self._clean_bullets(current))
            max_added_chars = min(max_added_chars, max(final_char_limit - current_chars, 0))
        existing_keys = {self._dedupe_key(item) for item in self._clean_bullets(current)}
        bullets = []
        used_chars = 0
        for sentence in self._split_sentences(source_context):
            cleaned = self._clean_bullets([sentence.strip()])
            if not cleaned:
                continue
            candidate = cleaned[0]
            if self._is_weak_expansion_bullet(candidate):
                continue
            if len(candidate) < 40:
                continue
            if len(candidate) > 190:
                candidate = self._truncate_on_word_boundary(candidate, 190)
                cleaned = self._clean_bullets([candidate])
                if not cleaned:
                    continue
                candidate = cleaned[0]
                if self._is_weak_expansion_bullet(candidate):
                    continue
            key = self._dedupe_key(candidate)
            if key in existing_keys:
                continue
            if used_chars + len(candidate) > max_added_chars:
                remaining = max_added_chars - used_chars
                if remaining >= int(self.block_config.get("fallback_min_truncated_extra_chars", 42)):
                    candidate = self._truncate_on_word_boundary(candidate, remaining)
                    cleaned = self._clean_bullets([candidate])
                    if not cleaned:
                        continue
                    candidate = cleaned[0]
                    if self._is_weak_expansion_bullet(candidate):
                        continue
                else:
                    break
            if used_chars + len(candidate) > max_added_chars:
                continue
            bullets.append(candidate)
            existing_keys.add(key)
            used_chars += len(candidate)
            if len(bullets) >= self._max_new_bullets(target):
                break
        return bullets

    def _is_weak_expansion_bullet(self, text: str) -> bool:
        plain = re.sub(r"<color:[^>]+>|</color>|\*\*", "", str(text or "")).strip()
        lowered = plain.lower()
        if re.search(r"\b(?:the|this|that)\s+(?:method|approach|model|policy|framework|paper)\.$", lowered):
            return True
        if re.search(r"\b(?:main|key|central)\s+reason\.$", lowered):
            return True
        return len(self._terms(plain)) < 5

    def _stringify_source(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            pieces = []
            for key, item in value.items():
                pieces.append(str(key))
                pieces.append(self._stringify_source(item))
            return "\n".join(pieces)
        if isinstance(value, list):
            return "\n".join(self._stringify_source(item) for item in value)
        return str(value)

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [sentence.strip(" -\t\n") for sentence in sentences if sentence.strip()]

    def _terms(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")
            if token.lower() not in {"the", "and", "for", "with", "from", "this", "that", "section"}
        }

    def _clean_bullets(self, bullets: List[Any]) -> List[str]:
        cleaned = []
        for item in bullets:
            text = normalize_text_for_poster(str(item or "").strip())
            text = re.sub(r"^\s*[•●◦▪▫*\-]\s*", "", text).strip()
            text = re.sub(r"^\s*(?:\d+[\.)]|step\s+\d+[\.:]?)\s*", "", text, flags=re.IGNORECASE).strip()
            without_index = re.sub(r"^\s*\d+\s*,\s*", "", text).strip()
            if without_index != text and without_index:
                text = without_index[:1].upper() + without_index[1:]
            if text:
                cleaned.append(text)
        return cleaned

    def _dedupe_key(self, text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()[:120]

    def _truncate_on_word_boundary(self, text: str, char_limit: int) -> str:
        return fit_complete_sentence_prefix(text, char_limit)

    def _remove_dangling_truncation(self, text: str) -> str:
        text = re.sub(
            r"\s+(and|or|but|with|in|of|to|for|by|as|at|from|than|while|where|when|that|which|through|into|over|under|via)$",
            "",
            str(text or "").strip(),
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+and\s+[A-Za-z][A-Za-z-]{0,10}$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+by\s+first\s+[A-Za-z-]+ing(?:\s+[A-Za-z-]+){0,2}$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(
            r"\s+(?:creates?|created|creating|causes?|caused|causing|forms?|formed|forming|poses?|posed|posing|injects?|injected|injecting|includes?|included|including)\s+(?:a|an|the)?\s*[A-Za-z-]{0,28}$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"\s+under\s+(?:tight|limited|strict)$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+with\s+(?:a|an|the)\s+[A-Za-z-]{0,16}$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+and\s+a\s+share$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+with\s+(?:either|any|the|a|an)\s+[A-Za-z-]*(?:unif|uniform|vi)$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+(?:and|or|for|with|under|to|by|of|over|via)\s+[A-Za-z-]*(?:cos|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|unif|vi)$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+(?:[A-Za-z]|fo|fou|ou|ar|cos|evic|prob|unif|unifor|withi|cha|dis|se|lo|ri|mo|res|vis|analys|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|vi)$", "", text, flags=re.IGNORECASE).strip()
        return text

    def _max_new_bullets(self, target_extra_chars: int) -> int:
        chars_per_bullet = int(self.block_config.get("chars_per_bullet", 120))
        max_bullets = int(self.block_config.get("max_new_bullets_per_block", 5))
        return max(1, min(max_bullets, int((max(target_extra_chars, 1) + chars_per_bullet - 1) / chars_per_bullet)))

    def _empty_patch(self, reason: str) -> Dict[str, Any]:
        return {
            "source": self.name,
            "applied": False,
            "iteration": 0,
            "actions_considered": [],
            "patches": [],
            "warnings": [reason],
        }

    def _save_outputs(self, state: PosterState, patch_report: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "block_content_patch.json", "w", encoding="utf-8") as f:
            json.dump(patch_report, f, indent=2)
        with open(output_dir / "story_board.json", "w", encoding="utf-8") as f:
            json.dump(state.get("story_board", {}), f, indent=2)


def block_content_refiner_node(state: PosterState) -> Dict[str, Any]:
    result = BlockContentRefiner()(state)
    return {
        **state,
        "story_board": result.get("story_board"),
        "block_content_patch": result.get("block_content_patch"),
        "block_refinement_history": result.get("block_refinement_history"),
        "block_refinement_required": result.get("block_refinement_required", False),
        "block_refinement_count": result.get("block_refinement_count", 0),
        "block_occupancy_report": result.get("block_occupancy_report"),
        "block_vlm_review": result.get("block_vlm_review"),
        "optimized_story_board": result.get("optimized_story_board"),
        "optimized_column_assignment": result.get("optimized_column_assignment"),
        "balancer_decisions": result.get("balancer_decisions"),
        "initial_layout_data": result.get("initial_layout_data"),
        "column_analysis": result.get("column_analysis"),
        "design_layout": result.get("design_layout"),
        "styled_layout": result.get("styled_layout"),
        "header_block_review": result.get("header_block_review"),
        "header_block_patch_applied": result.get("header_block_patch_applied", False),
        "visual_legibility_review": result.get("visual_legibility_review"),
        "vlm_layout_review": result.get("vlm_layout_review"),
        "vlm_layout_patch": result.get("vlm_layout_patch"),
        "vlm_reflow_required": result.get("vlm_reflow_required", False),
        "vlm_patch_applied": result.get("vlm_patch_applied", False),
        "template_repair_required": result.get("template_repair_required", False),
        "adaptive_relayout_required": result.get("adaptive_relayout_required", False),
        "render_stage": result.get("render_stage", state.get("render_stage", "draft")),
        "draft_status": result.get("draft_status", state.get("draft_status", "pending")),
        "final_poster_accepted": result.get("final_poster_accepted", False),
        "tokens": result.get("tokens"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
