"""
Soft template-prior planner for cluster_* poster templates.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.tools.layout_api import LayoutTemplates
from src.template_extraction.block_template_registry import is_block_template_id
from src.utils.text_cleanup import fit_complete_sentence_prefix, normalize_text_for_poster
from src.utils.visual_footprint import visual_requirements
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class TemplatePriorPlanner:
    def __init__(self):
        self.name = "template_prior_planner"
        self.prompt = load_prompt("config/prompts/template_block_planner.txt")
        self.config = load_config()
        self.block_config = self.config.get("block_refinement", {})

    def __call__(self, state: PosterState) -> PosterState:
        template_name = state.get("resolved_layout_template") or state.get("layout_template")
        if not is_block_template_id(template_name):
            return state

        log_agent_info(self.name, f"building soft template prior for {template_name}")

        try:
            template_layout = self._resolve_template_layout(state, template_name)
            story_board = deepcopy(state.get("story_board") or {})
            sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
            if not sections:
                raise ValueError("missing story_board sections for template prior planning")

            normalized_sections = self._normalize_sections(sections)
            layout_intent = self._build_layout_intent(normalized_sections, template_layout, state)
            rewritten_story_board = self._rewrite_story_board(story_board, layout_intent, template_layout)

            state["layout_intent"] = layout_intent
            state["template_prior_source_story_board"] = deepcopy(story_board)
            state["block_capacity_contract"] = layout_intent.get("block_capacity_contract")
            state["capacity_aware_story_board"] = deepcopy(rewritten_story_board)
            state["capacity_planning_report"] = layout_intent.get("capacity_planning_report")
            state["template_block_plan"] = {
                "template_id": template_name,
                "active_region_ids": [item["region_id"] for item in layout_intent["region_plan"]],
                "hero_section": layout_intent["hero_section"],
                "blocks": [
                    {
                        "block_id": section["section_id"],
                        "slot_id": section["region_id"],
                        "content_role": section["content_role"],
                        "target_title": section["section_title"],
                        "target_bullets": section["text_content"],
                        "visual_assets": section.get("visual_assets") or [],
                        "keypoint_id": section.get("keypoint_id"),
                        "source_keypoint_ids": section.get("source_keypoint_ids"),
                        "source_section": section.get("source_section"),
                        "preferred_slot_id": section.get("preferred_slot_id"),
                        "capacity_budget": section.get("capacity_budget"),
                    }
                    for section in layout_intent["active_sections"]
                ],
            }
            state["story_board"] = rewritten_story_board
            state["layout_template_metadata"] = template_layout
            state["template_layout_mode"] = "template_prior"
            state["resolved_layout_template"] = template_name
            state["render_stage"] = "draft"
            state["final_poster_accepted"] = False
            state["current_agent"] = self.name
            self._save_outputs(state)

            log_agent_success(
                self.name,
                f"planned {len(layout_intent['active_sections'])} active sections across {len(layout_intent['region_plan'])} regions",
            )
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _resolve_template_layout(self, state: PosterState, template_name: str) -> Dict[str, Any]:
        from src.config.poster_config import load_config

        config = load_config()
        return LayoutTemplates(
            state["poster_width"],
            state["poster_height"],
            margin=config["layout"]["poster_margin"],
            col_gap=config["layout"]["column_spacing"],
        ).get_template(template_name)

    def _normalize_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for index, section in enumerate(sections):
            title = str(section.get("section_title") or f"Section {index + 1}").strip()
            content_role = str(section.get("content_type") or "").strip() or self._infer_role(title)
            normalized.append({
                "section_id": section.get("section_id", f"section_{index + 1}"),
                "section_title": title,
                "text_content": [str(item).strip() for item in section.get("text_content") or [] if str(item).strip()],
                "visual_assets": list(section.get("visual_assets") or []),
                "column_assignment": section.get("column_assignment", "middle"),
                "vertical_priority": section.get("vertical_priority", "middle"),
                "content_role": content_role,
                "source_sections": list(section.get("source_sections") or [section.get("section_id", f"section_{index + 1}")]),
                "keypoint_id": section.get("keypoint_id"),
                "source_keypoint_ids": list(section.get("source_keypoint_ids") or []),
                "source_section": section.get("source_section"),
                "source_keypoint": section.get("source_keypoint"),
                "preferred_slot_id": section.get("preferred_slot_id"),
                "capacity_budget": section.get("capacity_budget"),
                "target_chars": section.get("target_chars"),
                "min_chars": section.get("min_chars"),
                "max_chars": section.get("max_chars"),
                "target_bullets": section.get("target_bullets"),
                "generated_teaser_summary": section.get("generated_teaser_summary"),
                "generated_teaser_original_text_count": section.get("generated_teaser_original_text_count"),
            })
        if any(item.get("keypoint_id") for item in normalized):
            normalized.sort(key=lambda item: self._keypoint_sort_value(item))
        else:
            normalized.sort(
                key=lambda item: (
                    self._role_priority(item["content_role"]),
                    self._priority_rank(item.get("vertical_priority")),
                    self._lane_rank(item.get("column_assignment")),
                )
            )
        return normalized

    def _build_layout_intent(
        self,
        sections: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        state: PosterState,
    ) -> Dict[str, Any]:
        regions = list(template_layout.get("regions") or [])
        if not regions:
            raise ValueError("template prior has no regions")

        active_regions = self._select_active_regions(regions, sections, template_layout, state)
        hero_section = self._choose_hero_section(sections, state)
        hero_region_id = template_layout.get("hero_region_id") or active_regions[0]["region_id"]

        ordered_sections = self._select_active_sections(sections, active_regions, hero_section, state)
        assigned_sections = self._assign_sections_to_regions(
            ordered_sections,
            active_regions,
            hero_section,
            hero_region_id,
            preserve_order=self._has_keypoints(state),
            template_id=str(template_layout.get("template_name") or ""),
        )
        capacity_contract = self._build_block_capacity_contract(assigned_sections, template_layout, state)
        if state.get("template_fast_mode"):
            semantic_sections = assigned_sections
            fast_config = self.config.get("template_fast_mode", {})
            if (
                bool(fast_config.get("semantic_capacity_rewrite", True))
                and self._needs_semantic_capacity_refinement(assigned_sections, capacity_contract)
            ):
                semantic_sections = (
                    self._refine_with_llm(assigned_sections, template_layout, state, capacity_contract)
                    or assigned_sections
                )
            capacity_sections = self._apply_capacity_contract(semantic_sections, capacity_contract, state)
            refined_sections = capacity_sections
        else:
            capacity_sections = self._apply_capacity_contract(assigned_sections, capacity_contract, state)
            refined_sections = self._refine_with_llm(capacity_sections, template_layout, state, capacity_contract) or capacity_sections
        refined_sections = self._restore_generated_teaser_summaries(refined_sections, capacity_sections)
        refined_sections = self._apply_capacity_contract(refined_sections, capacity_contract, state)
        capacity_report = self._build_capacity_planning_report(refined_sections, capacity_contract)

        active_section_ids = [section["section_id"] for section in refined_sections]
        drop_candidates = [
            section["section_id"]
            for section in sections
            if section["section_id"] not in active_section_ids
        ]
        compressible_sections = [
            section["section_id"]
            for section in refined_sections
            if section.get("region_meta", {}).get("text_density_limit") != "high"
        ]

        return {
            "template_id": template_layout.get("template_name"),
            "hero_section": hero_section["section_id"],
            "hero_region_id": hero_region_id,
            "visual_priority": [section["section_id"] for section in refined_sections if section.get("visual_assets")],
            "active_sections": refined_sections,
            "supporting_sections": [section["section_id"] for section in refined_sections if section["section_id"] != hero_section["section_id"]],
            "suggested_region_assignment": {
                section["section_id"]: section["region_id"]
                for section in refined_sections
            },
            "drop_candidates": drop_candidates,
            "compressible_sections": compressible_sections,
            "region_plan": [
                {
                    "region_id": region["region_id"],
                    "region_rank": region["region_rank"],
                    "region_tier": region["region_tier"],
                    "text_density_limit": region["text_density_limit"],
                    "is_hero_region": region.get("is_hero_region", False),
                }
                for region in active_regions
            ],
            "block_capacity_contract": capacity_contract,
            "capacity_planning_report": capacity_report,
        }

    def _select_active_regions(
        self,
        regions: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        state: PosterState,
    ) -> List[Dict[str, Any]]:
        density = template_layout.get("template_density_profile") or "balanced"
        section_count = len(sections)
        if self._has_keypoints(state):
            target_regions = min(len(regions), section_count, len(state.get("paper_poster_keypoints") or sections), 10)
            ordered_regions = sorted(regions, key=lambda region: (float(region.get("y", 0.0)), float(region.get("x", 0.0))))
            return ordered_regions[:target_regions]

        large_visual = any(self._visual_is_large(section, state) for section in sections)
        if template_layout.get("orientation") == "portrait":
            max_regions = min(len(regions), max(3, min(section_count, len(regions))))
        elif density == "hero_wide" and large_visual:
            max_regions = 3
        elif density in {"hero_wide", "dual_primary"}:
            max_regions = 4
        else:
            max_regions = min(4, len(regions))
        max_regions = min(max_regions, len(regions), max(3, min(section_count, len(regions))))
        ranked = sorted(
            regions,
            key=lambda region: (
                0 if region.get("is_hero_region") else 1,
                region.get("region_rank", 999),
                -float(region.get("area_ratio", 0.0)),
            ),
        )
        return sorted(ranked[:max_regions], key=lambda region: (float(region.get("y", 0.0)), float(region.get("x", 0.0))))

    def _choose_hero_section(self, sections: List[Dict[str, Any]], state: PosterState) -> Dict[str, Any]:
        key_visual = ((state.get("classified_visuals") or {}).get("key_visual"))
        if key_visual:
            for section in sections:
                if any(visual.get("visual_id") == key_visual for visual in section.get("visual_assets", [])):
                    return section
        scored = []
        for section in sections:
            role_bonus = {
                "method": 4.0,
                "results": 3.2,
                "overview": 2.0,
                "setup": 1.6,
                "takeaway": 1.0,
            }.get(section["content_role"], 1.0)
            has_visual = 1.5 if section.get("visual_assets") else 0.0
            key_bonus = 2.0 if key_visual and any(v.get("visual_id") == key_visual for v in section.get("visual_assets", [])) else 0.0
            text_bonus = min(len(section.get("text_content") or []), 4) * 0.15
            scored.append((role_bonus + has_visual + key_bonus + text_bonus, section))
        return max(scored, key=lambda item: item[0])[1]

    def _select_active_sections(
        self,
        sections: List[Dict[str, Any]],
        active_regions: List[Dict[str, Any]],
        hero_section: Dict[str, Any],
        state: PosterState,
    ) -> List[Dict[str, Any]]:
        max_sections = len(active_regions)
        if self._has_keypoints(state):
            order_map = self._keypoint_order_map(state)
            ordered = sorted(
                sections,
                key=lambda section: (
                    order_map.get(self._safe_int(section.get("keypoint_id")), 999),
                    self._keypoint_sort_value(section),
                ),
            )
            return [deepcopy(section) for section in ordered[:max_sections]]

        if (state.get("layout_template_metadata") or {}).get("template_density_profile") == "hero_wide":
            max_sections = min(max_sections, 3 if self._visual_is_large(hero_section, state) else 4)

        picked: List[Dict[str, Any]] = [deepcopy(hero_section)]

        for desired_role in ["overview", "results", "takeaway", "setup"]:
            if len(picked) >= max_sections:
                break
            candidate = next(
                (
                    deepcopy(section)
                    for section in sections
                    if section["section_id"] not in {item["section_id"] for item in picked}
                    and section["content_role"] == desired_role
                ),
                None,
            )
            if candidate is not None:
                picked.append(candidate)

        for section in sections:
            if len(picked) >= max_sections:
                break
            if section["section_id"] in {item["section_id"] for item in picked}:
                continue
            picked.append(deepcopy(section))

        return picked[:max_sections]

    def _assign_sections_to_regions(
        self,
        sections: List[Dict[str, Any]],
        regions: List[Dict[str, Any]],
        hero_section: Dict[str, Any],
        hero_region_id: str,
        preserve_order: bool = False,
        template_id: str = "",
    ) -> List[Dict[str, Any]]:
        region_map = {region["region_id"]: deepcopy(region) for region in regions}
        assigned: List[Dict[str, Any]] = []
        used_region_ids: set[str] = set()

        remaining_regions = [region for region in regions if region["region_id"] != hero_region_id]
        hero_region = deepcopy(region_map[hero_region_id])

        for index, section in enumerate(sections):
            item = deepcopy(section)
            preferred_slot_id = str(item.get("preferred_slot_id") or "")
            if preserve_order and preferred_slot_id:
                region = deepcopy(region_map[preferred_slot_id]) if preferred_slot_id in region_map and preferred_slot_id not in used_region_ids else None
                if region is None:
                    later_visual_count = sum(1 for later in sections[index + 1:] if later.get("visual_assets"))
                    region = self._region_for_ordered_section(item, regions, used_region_ids, later_visual_count)
            elif preserve_order:
                later_visual_count = sum(1 for later in sections[index + 1:] if later.get("visual_assets"))
                region = self._region_for_ordered_section(item, regions, used_region_ids, later_visual_count)
            elif section["section_id"] == hero_section["section_id"]:
                region = hero_region
            else:
                region = self._best_region_for_section(item, remaining_regions) or remaining_regions[0]
                remaining_regions = [candidate for candidate in remaining_regions if candidate["region_id"] != region["region_id"]]
            used_region_ids.add(str(region["region_id"]))
            item["region_id"] = region["region_id"]
            item["column_assignment"] = region["region_id"]
            item["semantic_lane"] = region.get("semantic_lane", item.get("column_assignment", "middle"))
            item["slot_id"] = region["region_id"]
            item["preferred_slot_id"] = item.get("preferred_slot_id")
            item["region_meta"] = region
            item["visual_assets"] = self._limit_visuals_for_region(item, region)
            item["text_content"] = self._clean_bullets(item["text_content"])
            assigned.append(item)
        return assigned

    def _region_for_ordered_section(
        self,
        section: Dict[str, Any],
        regions: List[Dict[str, Any]],
        used_region_ids: set[str],
        later_visual_count: int = 0,
    ) -> Dict[str, Any]:
        available = [region for region in regions if str(region.get("region_id")) not in used_region_ids]
        if not available:
            return deepcopy(regions[-1])

        if section.get("visual_assets"):
            visual_regions = [region for region in available if region.get("can_host_visual")]
            if visual_regions:
                role = str(section.get("content_role") or "").lower()
                if role == "results":
                    canvas_right = max(float(r.get("x", 0.0)) + float(r.get("w", 0.0)) for r in regions)
                    ranked = sorted(
                        visual_regions,
                        key=lambda region: (
                            0 if region.get("text_density_limit") != "low" else 1,
                            0 if region.get("region_tier") == "primary" else 1,
                            0 if float(region.get("x", 0.0)) >= 0.45 * canvas_right else 1,
                            -float(region.get("area_ratio", 0.0)),
                            float(region.get("y", 0.0)),
                        ),
                    )
                else:
                    ranked = sorted(
                        visual_regions,
                        key=lambda region: (
                            float(region.get("y", 0.0)),
                            0 if region.get("region_tier") == "primary" else 1,
                            -float(region.get("area_ratio", 0.0)),
                        ),
                )
                return deepcopy(ranked[0])

        if later_visual_count > 0:
            remaining_medium_visual_regions = [
                region
                for region in available
                if region.get("can_host_visual") and region.get("text_density_limit") != "low"
            ]
            if len(remaining_medium_visual_regions) <= later_visual_count:
                text_safe_regions = [
                    region
                    for region in available
                    if not (region.get("can_host_visual") and region.get("text_density_limit") != "low")
                ]
                if text_safe_regions:
                    ranked = sorted(
                        text_safe_regions,
                        key=lambda region: (
                            0 if not region.get("can_host_visual") else 1,
                            float(region.get("y", 0.0)),
                            float(region.get("x", 0.0)),
                            float(region.get("area_ratio", 0.0)),
                        ),
                    )
                    return deepcopy(ranked[0])

        if self._is_main_result_section(section):
            canvas_right = max(float(r.get("x", 0.0)) + float(r.get("w", 0.0)) for r in regions)
            ranked = sorted(
                available,
                key=lambda region: (
                    0 if region.get("text_density_limit") != "low" else 1,
                    0 if float(region.get("x", 0.0)) >= 0.45 * canvas_right else 1,
                    0 if region.get("region_tier") == "primary" else 1,
                    -float(region.get("area_ratio", 0.0)),
                    float(region.get("y", 0.0)),
                ),
            )
            return deepcopy(ranked[0])

        return deepcopy(available[0])

    def _is_main_result_section(self, section: Dict[str, Any]) -> bool:
        role = str(section.get("content_role") or "").lower()
        title = str(section.get("section_title") or "").lower()
        section_id = str(section.get("section_id") or "").lower()
        return role == "results" or "main result" in title or "main_result" in section_id

    def _best_region_for_section(self, section: Dict[str, Any], regions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not regions:
            return None
        role = section["content_role"]
        if role == "overview":
            ranked = sorted(
                regions,
                key=lambda region: (
                    0 if region.get("region_tier") == "primary" else 1,
                    -float(region.get("area_ratio", 0.0)),
                    float(region.get("y", 0.0)),
                ),
            )
            return deepcopy(ranked[0])
        if role == "results":
            ranked = sorted(
                regions,
                key=lambda region: (
                    0 if region.get("region_tier") == "primary" else 1,
                    -float(region.get("w", 0.0)),
                    -float(region.get("h", 0.0)),
                ),
            )
            return deepcopy(ranked[0])
        if role == "takeaway":
            ranked = sorted(regions, key=lambda region: (float(region.get("area_ratio", 0.0)), float(region.get("y", 0.0))))
            return deepcopy(ranked[0])
        ranked = sorted(
            regions,
            key=lambda region: (
                0 if region.get("can_host_visual") else 1,
                -float(region.get("area_ratio", 0.0)),
                region.get("region_rank", 999),
            ),
        )
        return deepcopy(ranked[0])

    def _limit_visuals_for_region(self, section: Dict[str, Any], region: Dict[str, Any]) -> List[Dict[str, Any]]:
        visuals = list(section.get("visual_assets") or [])
        if not visuals:
            return []
        if not region.get("can_host_visual", False):
            return []
        if region.get("text_density_limit") == "low":
            if self._allows_low_density_visual(section, region):
                return visuals[:1]
            return visuals[:0]
        visual_policy = str((section.get("capacity_budget") or {}).get("visual_policy") or "")
        if "table" in visual_policy:
            preferred = [visual for visual in visuals if str(visual.get("visual_id", "")).startswith("table_")]
            if preferred:
                visuals = preferred + [visual for visual in visuals if visual not in preferred]
        elif section["content_role"] == "results":
            preferred = [visual for visual in visuals if str(visual.get("visual_id", "")).startswith("figure_")]
            if preferred:
                visuals = preferred + [visual for visual in visuals if visual not in preferred]
        return visuals[:self._visual_limit_for_region(section, region)]

    def _visual_limit_for_region(self, section: Dict[str, Any], region: Dict[str, Any]) -> int:
        if not section.get("capacity_budget"):
            return 1
        density = str(region.get("text_density_limit") or "medium")
        area = float(region.get("area_ratio", 0.0) or 0.0)
        if density == "high" or area >= 0.12:
            return 2
        return 1

    def _allows_low_density_visual(self, section: Dict[str, Any], region: Dict[str, Any]) -> bool:
        region_id = str(region.get("region_id") or "")
        if region_id == "slot_3" and self._allows_slot3_method_low_density_visual(section):
            return True
        preferred_slot = str(section.get("preferred_slot_id") or section.get("slot_id") or "")
        visual_policy = str((section.get("capacity_budget") or {}).get("visual_policy") or "")
        if preferred_slot != region_id:
            return False
        if "table" in visual_policy:
            return any(str(visual.get("visual_id") or "").startswith("table_") for visual in section.get("visual_assets") or [])
        if "figure" in visual_policy:
            return any(str(visual.get("visual_id") or "").startswith("figure_") for visual in section.get("visual_assets") or [])
        return False

    def _allows_slot3_method_low_density_visual(self, section: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(section.get("section_title") or ""),
                str(section.get("section_id") or ""),
                str(section.get("content_role") or ""),
            ]
        ).lower()
        return any(token in text for token in ["arena", "setting", "diagram", "method", "framework"])

    def _build_block_capacity_contract(
        self,
        sections: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        state: PosterState,
    ) -> Dict[str, Any]:
        settings = self._capacity_settings()
        section_by_slot = {section.get("region_id") or section.get("slot_id"): section for section in sections}
        blocks = []
        for region in template_layout.get("regions") or []:
            slot_id = region.get("region_id") or region.get("slot_id") or region.get("id")
            section = section_by_slot.get(slot_id)
            if not slot_id or not section:
                continue
            blocks.append(self._capacity_budget_for_section(section, region, state, settings))

        blocks.sort(key=lambda item: (float(item["slot_bbox"]["y"]), float(item["slot_bbox"]["x"])))
        return {
            "source": "template_first_capacity",
            "settings": settings,
            "blocks": blocks,
            "by_slot": {block["slot_id"]: block for block in blocks},
            "by_section": {block["section_id"]: block for block in blocks},
        }

    def _capacity_budget_for_section(
        self,
        section: Dict[str, Any],
        region: Dict[str, Any],
        state: PosterState,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        slot_w = max(float(region.get("w", 0.0) or 0.0), 0.1)
        slot_h = max(float(region.get("h", 0.0) or 0.0), 0.1)
        text_padding = float(self.config["layout"]["text_padding"]["left_right"])
        usable_text_width = max(slot_w - 2 * text_padding, 0.5)
        visual_count = len(section.get("visual_assets") or [])
        visual_height = self._reserved_visual_height(section, usable_text_width, slot_h, state)
        visual_policy = self._visual_policy(section, usable_text_width, state)

        reserved_visual_height = visual_height + (float(settings["visual_gap_inches"]) if visual_count else 0.0)
        target_used_height = slot_h * float(settings["target_utilization"])
        available_text_height = max(
            target_used_height
            - float(settings["title_height_inches"])
            - float(settings["title_content_gap_inches"])
            - reserved_visual_height
            - float(settings["section_padding_inches"]),
            0.0,
        )
        line_height = self._line_height(int(settings["body_font_size"]), float(settings["line_spacing"]))
        target_lines = max(int(math.floor(available_text_height / max(line_height, 0.01))), 0)
        chars_per_line = self._chars_per_line(usable_text_width, int(settings["body_font_size"]), state)
        raw_target_chars = int(target_lines * chars_per_line * float(settings["safety_factor"]))
        min_floor = int(settings["visual_min_text_chars"] if visual_count else settings["min_capacity_chars"])
        target_chars = max(min(raw_target_chars, int(settings["max_capacity_chars"])), min_floor)
        if visual_policy == "prioritize_visual_scale":
            target_chars = min(target_chars, max(min_floor, int(settings["max_capacity_chars"] * 0.45)))
        min_chars = max(int(target_chars * float(settings["capacity_min_factor"])), min_floor)
        max_chars = max(
            target_chars,
            min(int(settings["max_capacity_chars"]), int(target_chars * float(settings["capacity_max_factor"]))),
        )
        target_bullets = max(1, min(8, int(math.ceil(target_chars / max(int(settings["chars_per_bullet"]), 1)))))

        warning = None
        if raw_target_chars < min_floor:
            warning = "low_text_capacity"
        if visual_policy == "prioritize_visual_scale":
            warning = "visual_too_small_risk"

        budget = {
            "slot_id": str(region.get("region_id") or region.get("slot_id") or region.get("id") or ""),
            "section_id": str(section.get("section_id") or ""),
            "content_role": section.get("content_role"),
            "slot_bbox": {
                "x": round(float(region.get("x", 0.0) or 0.0), 4),
                "y": round(float(region.get("y", 0.0) or 0.0), 4),
                "w": round(slot_w, 4),
                "h": round(slot_h, 4),
            },
            "usable_text_width": round(usable_text_width, 4),
            "reserved_title_height": round(float(settings["title_height_inches"]), 4),
            "reserved_visual_height": round(reserved_visual_height, 4),
            "available_text_height": round(available_text_height, 4),
            "line_height": round(line_height, 4),
            "target_lines": target_lines,
            "chars_per_line": chars_per_line,
            "raw_target_chars": raw_target_chars,
            "target_chars": target_chars,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "target_bullets": target_bullets,
            "visual_policy": visual_policy,
            "capacity_warning": warning,
        }
        fast_budget = self._fast_budget_for_slot(state, budget["slot_id"])
        if fast_budget:
            fast_target_chars = int(fast_budget.get("target_chars") or budget["target_chars"])
            use_actual_visual_capacity = bool(visual_count) and target_chars > fast_target_chars
            budget.update({
                "source": (
                    "fast_template_actual_visual_contract"
                    if use_actual_visual_capacity
                    else "fast_template_first_fixed_contract"
                ),
                "slot_role": fast_budget.get("slot_role"),
                "content_role": fast_budget.get("content_role") or budget.get("content_role"),
                "visual_policy": fast_budget.get("visual_policy") or budget.get("visual_policy"),
                "visual_footprint": fast_budget.get("visual_footprint"),
                "hard_min_utilization": fast_budget.get("hard_min_utilization"),
                "source_keypoint_ids": fast_budget.get("source_keypoint_ids") or [],
            })
            if not use_actual_visual_capacity:
                budget.update({
                    "target_chars": fast_target_chars,
                    "min_chars": int(fast_budget.get("min_chars") or budget["min_chars"]),
                    "max_chars": int(fast_budget.get("max_chars") or budget["max_chars"]),
                    "target_bullets": int(fast_budget.get("target_bullets") or budget["target_bullets"]),
                    "capacity_warning": fast_budget.get("capacity_warning"),
                })
        return budget

    def _capacity_settings(self) -> Dict[str, Any]:
        micro_config = self.config.get("micro_layout_refinement", {})
        return {
            "target_utilization": float(self.block_config.get("target_utilization", 0.95)),
            "acceptable_min": float(self.block_config.get("acceptable_min", 0.90)),
            "acceptable_max": float(self.block_config.get("acceptable_max", 0.97)),
            "hard_max": float(self.block_config.get("hard_max", 0.98)),
            "safety_factor": float(self.block_config.get("safety_factor", 0.82)),
            "capacity_min_factor": float(self.block_config.get("capacity_min_factor", 0.88)),
            "capacity_max_factor": float(self.block_config.get("capacity_max_factor", 1.08)),
            "title_height_inches": float(self.block_config.get("title_height_inches", 1.0)),
            "title_content_gap_inches": float(self.block_config.get("title_content_gap_inches", 0.4)),
            "visual_gap_inches": float(self.block_config.get("visual_gap_inches", 0.3)),
            "section_padding_inches": float(self.block_config.get("section_padding_inches", 0.4)),
            "min_capacity_chars": int(self.block_config.get("min_capacity_chars", 90)),
            "max_capacity_chars": int(self.block_config.get("max_capacity_chars", 900)),
            "visual_min_text_chars": int(self.block_config.get("visual_min_text_chars", 90)),
            "visual_policy_too_small_width_inches": float(self.block_config.get("visual_policy_too_small_width_inches", 13.0)),
            "chars_per_bullet": int(self.block_config.get("chars_per_bullet", 120)),
            "body_font_size": int(self.config["typography"]["sizes"]["body_text"]),
            "line_spacing": float(self.config["typography"].get("line_spacing", 1.0)),
            "ppt_chars_per_inch_at_44pt": float(
                self.block_config.get(
                    "ppt_chars_per_inch_at_44pt",
                    micro_config.get("ppt_chars_per_inch_at_44pt", 4.2),
                )
            ),
            "portrait_ppt_chars_per_inch_at_44pt": float(
                self.block_config.get(
                    "portrait_ppt_chars_per_inch_at_44pt",
                    micro_config.get(
                        "portrait_ppt_chars_per_inch_at_44pt",
                        self.block_config.get(
                            "ppt_chars_per_inch_at_44pt",
                            micro_config.get("ppt_chars_per_inch_at_44pt", 4.2),
                        ),
                    ),
                )
            ),
        }

    def _fast_budget_for_slot(self, state: PosterState, slot_id: Any) -> Dict[str, Any]:
        if not state.get("template_fast_mode"):
            return {}
        by_slot = ((state.get("fast_block_contract") or {}).get("by_slot") or {})
        budget = by_slot.get(str(slot_id or "")) or {}
        return dict(budget) if isinstance(budget, dict) else {}

    def _reserved_visual_height(self, section: Dict[str, Any], visual_width: float, slot_height: float, state: PosterState) -> float:
        total = 0.0
        visual_assets = state.get("visual_assets") or {}
        for visual in section.get("visual_assets") or []:
            visual_id = str(visual.get("visual_id") or "")
            if not visual_id:
                continue
            asset = visual_assets.get(visual_id) or {}
            if visual_id.startswith("table_"):
                aspect = float(asset.get("aspect") or 1.5)
            elif visual_id.startswith("figure_"):
                aspect = float(asset.get("aspect") or 1.2)
            else:
                aspect = float(asset.get("aspect") or 1.0)
            original_height = visual_width / max(aspect, 0.2)
            scale = 0.8 if original_height > slot_height * 0.4 else 1.0
            footprint = visual_requirements(
                visual_id,
                asset,
                {
                    "w": visual_width,
                    "h": slot_height,
                    "poster_orientation": "portrait"
                    if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
                    else "landscape",
                },
                self.config,
            )
            total += max(original_height * scale, float(footprint.get("min_height") or 0.0))
        return total

    def _visual_policy(self, section: Dict[str, Any], visual_width: float, state: PosterState) -> str:
        visuals = section.get("visual_assets") or []
        if not visuals:
            return "text_only"
        threshold = float(self.block_config.get("visual_policy_too_small_width_inches", 13.0))
        visual_assets = state.get("visual_assets") or {}
        risky = visual_width < threshold
        for visual in visuals:
            asset = visual_assets.get(visual.get("visual_id")) or {}
            aspect = float(asset.get("aspect") or 1.0)
            if aspect >= 2.2 or asset.get("asset_type") == "table":
                risky = True
        return "prioritize_visual_scale" if risky else "reserve_visual_space"

    def _line_height(self, font_size: int, line_spacing: float) -> float:
        return (font_size / 72) * max(line_spacing, 0.9) * 1.15

    def _chars_per_line(self, width_inches: float, font_size: int, state: Optional[PosterState] = None) -> int:
        chars_per_inch = self._chars_per_inch(state)
        return max(int(width_inches * chars_per_inch * (44 / max(font_size, 1))), 18)

    def _chars_per_inch(self, state: Optional[PosterState] = None) -> float:
        micro_config = self.config.get("micro_layout_refinement", {})
        default = float(
            self.block_config.get(
                "ppt_chars_per_inch_at_44pt",
                micro_config.get("ppt_chars_per_inch_at_44pt", 4.2),
            )
        )
        if state and float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0):
            return float(
                self.block_config.get(
                    "portrait_ppt_chars_per_inch_at_44pt",
                    micro_config.get("portrait_ppt_chars_per_inch_at_44pt", default),
                )
            )
        return default

    def _apply_capacity_contract(
        self,
        sections: List[Dict[str, Any]],
        capacity_contract: Dict[str, Any],
        state: PosterState,
        *,
        allow_expand: bool = True,
    ) -> List[Dict[str, Any]]:
        by_section = capacity_contract.get("by_section") or {}
        refined = []
        for section in sections:
            item = deepcopy(section)
            budget = by_section.get(str(item.get("section_id"))) or {}
            if not budget:
                refined.append(item)
                continue
            if self._is_generated_teaser_section(item):
                item["capacity_budget"] = budget
                item["target_chars"] = budget.get("target_chars")
                item["min_chars"] = budget.get("min_chars")
                item["max_chars"] = budget.get("max_chars")
                item["target_bullets"] = budget.get("target_bullets")
                item["capacity_warning"] = "generated_teaser_summary_preserved"
                refined.append(item)
                continue
            if len(item.get("visual_assets") or []) >= 2:
                budget = self._multi_visual_text_budget(budget)
            bullets, warning = self._fit_bullets_to_budget(
                item.get("text_content") or [],
                budget,
                item,
                state,
                allow_expand=allow_expand,
            )
            item["text_content"] = bullets
            item["capacity_budget"] = budget
            item["target_chars"] = budget.get("target_chars")
            item["min_chars"] = budget.get("min_chars")
            item["max_chars"] = budget.get("max_chars")
            item["target_bullets"] = budget.get("target_bullets")
            item["capacity_warning"] = warning or budget.get("capacity_warning")
            refined.append(item)
        return refined

    def _needs_semantic_capacity_refinement(
        self,
        sections: List[Dict[str, Any]],
        capacity_contract: Dict[str, Any],
    ) -> bool:
        by_section = capacity_contract.get("by_section") or {}
        for section in sections:
            if self._is_generated_teaser_section(section):
                continue
            budget = by_section.get(str(section.get("section_id"))) or {}
            min_chars = int(budget.get("min_chars") or 0)
            target_chars = int(budget.get("target_chars") or 0)
            max_chars = int(budget.get("max_chars") or 0)
            target_bullets = int(budget.get("target_bullets") or 0)
            bullets = section.get("text_content") or []
            actual_chars = self._bullet_chars(bullets)
            if (min_chars and actual_chars < min_chars) or (max_chars and actual_chars > max_chars):
                return True
            if target_bullets and len(bullets) < target_bullets and actual_chars < target_chars:
                return True
        return False

    def _multi_visual_text_budget(self, budget: Dict[str, Any]) -> Dict[str, Any]:
        adjusted = dict(budget)
        scale = float(self.block_config.get("multi_visual_text_budget_scale", 0.58))
        for key in ("min_chars", "target_chars", "max_chars"):
            if adjusted.get(key) is not None:
                adjusted[key] = max(90, int(float(adjusted.get(key) or 0) * scale))
        if adjusted.get("target_bullets") is not None:
            adjusted["target_bullets"] = max(1, min(3, int(adjusted.get("target_bullets") or 1)))
        adjusted["capacity_warning"] = adjusted.get("capacity_warning") or "multi_visual_text_budget_reduced"
        return adjusted

    def _fit_bullets_to_budget(
        self,
        bullets: List[Any],
        budget: Dict[str, Any],
        section: Dict[str, Any],
        state: PosterState,
        *,
        allow_expand: bool,
    ) -> tuple[List[str], Optional[str]]:
        cleaned = self._clean_bullets(bullets)
        if not cleaned:
            cleaned = ["Key takeaway."]

        max_chars = int(budget.get("max_chars") or 0)
        min_chars = int(budget.get("min_chars") or 0)
        target_chars = int(budget.get("target_chars") or max_chars or min_chars)
        target_bullets = int(budget.get("target_bullets") or 1)
        warning = budget.get("capacity_warning")

        fitted = self._trim_bullets_to_budget(cleaned, max_chars, max(target_bullets, 1))
        if allow_expand and self._bullet_chars(fitted) < min_chars:
            fitted = self._expand_bullets_from_source(fitted, budget, section, state)

        if self._bullet_chars(fitted) < min_chars:
            warning = warning or "insufficient_source_facts_for_capacity"
        if self._bullet_chars(fitted) > max_chars:
            fitted = self._trim_bullets_to_budget(fitted, max_chars, max(target_bullets, 1))
        if max_chars > 0 and self._bullet_chars(fitted) > max_chars:
            warning = warning or "complete_sentence_exceeds_capacity"
        if self._bullet_chars(fitted) < max(20, int(target_chars * 0.5)) and not allow_expand:
            warning = warning or "capacity_refinement_below_target"
        return fitted, warning

    def _is_generated_teaser_section(self, section: Dict[str, Any]) -> bool:
        if section.get("generated_teaser_summary"):
            return True
        return any(
            str(visual.get("visual_id") or "").startswith("generated_teaser")
            for visual in section.get("visual_assets") or []
            if isinstance(visual, dict)
        )

    def _restore_generated_teaser_summaries(
        self,
        refined_sections: List[Dict[str, Any]],
        source_sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        source_by_id = {str(section.get("section_id")): section for section in source_sections}
        restored = []
        for section in refined_sections:
            item = deepcopy(section)
            source = source_by_id.get(str(item.get("section_id"))) or {}
            if self._is_generated_teaser_section(source):
                item["text_content"] = list(source.get("text_content") or item.get("text_content") or [])
                item["generated_teaser_summary"] = source.get("generated_teaser_summary", True)
                item["generated_teaser_original_text_count"] = source.get("generated_teaser_original_text_count")
            restored.append(item)
        return restored

    def _trim_bullets_to_budget(self, bullets: List[str], max_chars: int, target_bullets: int) -> List[str]:
        if max_chars <= 0:
            return bullets[:target_bullets]
        result = []
        used = 0
        max_bullets = max(1, target_bullets + 1)
        for bullet in bullets:
            if len(result) >= max_bullets:
                break
            remaining = max_chars - used
            if remaining <= 0:
                break
            candidate = bullet
            if len(candidate) > remaining:
                candidate = fit_complete_sentence_prefix(candidate, remaining)
                if len(candidate) > remaining:
                    continue
            if not self._is_clean_poster_bullet(candidate):
                continue
            result.append(candidate)
            used += len(candidate)
        if result:
            return result

        # If no complete item fits, preserve one complete factual item and let
        # layout/QA report the capacity mismatch instead of fabricating a fragment.
        fallback = next((bullet for bullet in bullets if self._is_clean_poster_bullet(bullet)), "")
        return [fallback] if fallback else ["Key takeaway."]

    def _expand_bullets_from_source(
        self,
        bullets: List[str],
        budget: Dict[str, Any],
        section: Dict[str, Any],
        state: PosterState,
    ) -> List[str]:
        max_chars = int(budget.get("max_chars") or 0)
        min_chars = int(budget.get("min_chars") or 0)
        target_bullets = int(budget.get("target_bullets") or 1)
        result = list(bullets)
        existing = {self._dedupe_key(item) for item in result}
        for sentence in self._source_sentences_for_section(section, state):
            if self._bullet_chars(result) >= min_chars or len(result) >= max(target_bullets + 1, 2):
                break
            candidate = normalize_text_for_poster(sentence)
            if len(candidate) < 35 or not self._is_clean_poster_bullet(candidate):
                continue
            key = self._dedupe_key(candidate)
            if not key or key in existing:
                continue
            remaining = max_chars - self._bullet_chars(result)
            if remaining <= 0:
                break
            if len(candidate) > remaining:
                continue
            if not self._is_clean_poster_bullet(candidate):
                continue
            result.append(candidate)
            existing.add(key)
        return result

    def _source_sentences_for_section(self, section: Dict[str, Any], state: PosterState) -> List[str]:
        source_text = "\n".join(
            [
                self._stringify_source(state.get("structured_sections")),
                self._stringify_source(state.get("narrative_content")),
                str(state.get("raw_text") or ""),
            ]
        )
        source_text = self._strip_reference_sections(source_text)
        sentences = self._split_sentences(source_text)
        if not sentences:
            return []
        query_terms = self._terms(
            " ".join(
                [
                    str(section.get("section_title") or ""),
                    str(section.get("section_id") or ""),
                    str(section.get("content_role") or ""),
                    str(section.get("source_section") or ""),
                    " ".join(str(item or "") for item in section.get("source_sections") or []),
                    " ".join(str(item or "") for item in section.get("source_keypoints") or []),
                    " ".join(str(item or "") for item in section.get("text_content") or []),
                ]
            )
        )
        is_results_context = self._is_results_context(section)
        scored = []
        for sentence in sentences:
            sentence = normalize_text_for_poster(sentence)
            if not self._is_clean_poster_bullet(sentence):
                continue
            if self._looks_like_result_summary_text(sentence) and not is_results_context:
                continue
            overlap = len(query_terms & self._terms(sentence))
            if overlap:
                scored.append((overlap, len(sentence), sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:24]]

    def _is_results_context(self, section: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(section.get("section_title") or ""),
                str(section.get("section_id") or ""),
                str(section.get("content_role") or ""),
                str(section.get("source_section") or ""),
                " ".join(str(item or "") for item in section.get("source_sections") or []),
            ]
        ).lower()
        return any(token in text for token in ("result", "evaluation", "experiment", "takeaway", "benchmark"))

    def _looks_like_result_summary_text(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(overall empirical conclusion|strongest method|outperforms?|best performing|"
                r"performance|target rates?|cost models?|budgets?|main result|key result)\b",
                str(text or "").lower(),
            )
        )

    def _build_capacity_planning_report(
        self,
        sections: List[Dict[str, Any]],
        capacity_contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        reports = []
        for section in sections:
            budget = (capacity_contract.get("by_section") or {}).get(str(section.get("section_id"))) or {}
            chars = self._bullet_chars(section.get("text_content") or [])
            target = int(budget.get("target_chars") or 0)
            reports.append({
                "slot_id": section.get("slot_id") or section.get("region_id"),
                "section_id": section.get("section_id"),
                "target_chars": target,
                "min_chars": budget.get("min_chars"),
                "max_chars": budget.get("max_chars"),
                "actual_chars": chars,
                "char_target_ratio": round(chars / max(target, 1), 3),
                "target_bullets": budget.get("target_bullets"),
                "actual_bullets": len(section.get("text_content") or []),
                "visual_policy": budget.get("visual_policy"),
                "capacity_warning": section.get("capacity_warning"),
            })
        return {
            "source": "template_first_capacity",
            "block_count": len(reports),
            "within_budget_count": sum(
                1
                for item in reports
                if int(item.get("min_chars") or 0) <= int(item.get("actual_chars") or 0) <= int(item.get("max_chars") or 0)
            ),
            "blocks": reports,
        }

    def _compress_bullets_for_region(self, bullets: List[str], region: Dict[str, Any], has_visual: bool) -> List[str]:
        density = region.get("text_density_limit", "medium")
        max_bullets = {"high": 4, "medium": 3, "low": 2}.get(density, 3)
        if has_visual:
            max_bullets = max(1, max_bullets - 1)
        char_limit = {"high": 150, "medium": 115, "low": 90}.get(density, 115)
        trimmed = []
        for bullet in bullets:
            text = str(bullet).strip()
            if not text:
                continue
            if len(text) > char_limit:
                text = self._truncate_on_word_boundary(text, char_limit)
            if not self._is_clean_poster_bullet(text):
                continue
            trimmed.append(text)
            if len(trimmed) >= max_bullets:
                break
        return trimmed or ["Key takeaway."]

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
        text = re.sub(r"\s+(?:while|where|when|because|although|whereas)\s+[A-Za-z]{1,12}$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+under\s+(?:tight|limited|strict)$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+with\s+(?:a|an|the)\s+[A-Za-z-]{0,16}$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+and\s+a\s+share$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+with\s+(?:either|any|the|a|an)\s+[A-Za-z-]*(?:unif|uniform|vi)$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+(?:and|or|for|with|under|to|by|of|over|via)\s+[A-Za-z-]*(?:cos|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|unif|vi)$", "", text, flags=re.IGNORECASE).strip()
        return re.sub(r"\s+(?:fo|fou|ou|ar|cos|evic|prob|unif|unifor|withi|cha|dis|se|lo|ri|mo|res|vis|analys|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|vi)$", "", text, flags=re.IGNORECASE).strip()

    def _clean_bullets(self, bullets: List[Any]) -> List[str]:
        cleaned = []
        for item in bullets or []:
            text = normalize_text_for_poster(str(item or "").strip())
            text = re.sub(r"^\s*[•\-]\s*", "", text).strip()
            without_index = re.sub(r"^\s*\d+\s*,\s*", "", text).strip()
            if without_index != text and without_index:
                text = without_index[:1].upper() + without_index[1:]
            if text and self._is_clean_poster_bullet(text):
                cleaned.append(text)
        return cleaned

    def _strip_reference_sections(self, text: str) -> str:
        if not text:
            return ""
        return re.split(
            r"(?im)^\s*(references|bibliography|works\s+cited|literature\s+cited)\s*$",
            text,
            maxsplit=1,
        )[0]

    def _is_clean_poster_bullet(self, text: Any) -> bool:
        text = str(text or "").strip()
        if len(text) < 8:
            return False
        lowered = text.lower().strip(" .")
        if lowered in {"references", "bibliography", "works cited"}:
            return False
        if re.search(r"\b(?:doi|isbn|arxiv)\s*[:/]", text, flags=re.IGNORECASE):
            return False
        if re.search(r"\b(?:proceedings|conference|journal|transactions|press)\b", text, flags=re.IGNORECASE) and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", text):
            return False
        if len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.", text)) >= 2 and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", text):
            return False
        if re.search(r"\b(?:fo|fou|ou|ar|cos|evic|prob|unif|unifor|withi|cha|dis|se|lo|ri|mo|res|vis|analys|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|vi)\.$", text, flags=re.IGNORECASE):
            return False
        return True

    def _bullet_chars(self, bullets: List[Any]) -> int:
        return sum(len(str(item or "")) for item in bullets or [])

    def _dedupe_key(self, text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()[:120]

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
        stop_words = {"the", "and", "for", "with", "from", "this", "that", "section", "using", "into"}
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")
            if token.lower() not in stop_words
        }

    def _refine_with_llm(
        self,
        sections: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        state: PosterState,
        capacity_contract: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            agent = LangGraphAgent("expert academic poster editor", state["text_model"], state, self.name)
            prompt = self._build_capacity_refinement_prompt(sections, template_layout, capacity_contract, state)
            response = agent.step(prompt)
            state["tokens"].add_text(response.input_tokens, response.output_tokens)
            payload = extract_json(response.content)
            blocks = payload.get("blocks")
            if not isinstance(blocks, list) or len(blocks) != len(sections):
                return None
            refined = []
            for original, candidate in zip(sections, blocks):
                item = deepcopy(original)
                if self._is_generated_teaser_section(original):
                    refined.append(item)
                    continue
                item["section_title"] = str(candidate.get("target_title") or original["section_title"]).strip()
                item["text_content"] = self._clean_bullets(
                    candidate.get("text_content") or candidate.get("target_bullets") or original["text_content"]
                )
                if candidate.get("capacity_warning"):
                    item["capacity_warning"] = str(candidate.get("capacity_warning"))
                refined.append(item)
            return refined
        except Exception as exc:
            log_agent_warning(self.name, f"LLM region refinement unavailable: {exc}")
            return None

    def _build_capacity_refinement_prompt(
        self,
        sections: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        capacity_contract: Dict[str, Any],
        state: PosterState,
    ) -> str:
        by_section = capacity_contract.get("by_section") or {}
        blocks = []
        for section in sections:
            budget = by_section.get(str(section.get("section_id"))) or {}
            blocks.append({
                "section_id": section.get("section_id"),
                "slot_id": section.get("region_id"),
                "content_role": section.get("content_role"),
                "target_title": section.get("section_title"),
                "current_bullets": section.get("text_content") or [],
                "visual_assets": section.get("visual_assets") or [],
                "keypoint_id": section.get("keypoint_id"),
                "source_keypoint_ids": section.get("source_keypoint_ids") or [],
                "source_section": section.get("source_section"),
                "capacity_budget": {
                    "target_chars": budget.get("target_chars"),
                    "min_chars": budget.get("min_chars"),
                    "max_chars": budget.get("max_chars"),
                    "target_bullets": budget.get("target_bullets"),
                    "visual_policy": budget.get("visual_policy"),
                    "capacity_warning": budget.get("capacity_warning"),
                },
                "generated_teaser_summary": bool(section.get("generated_teaser_summary")),
                "source_context": "\n".join(self._source_sentences_for_section(section, state))[:2400],
            })
        return f"""
Rewrite academic poster block content to match each template block's text capacity.

Rules:
- Use only facts present in source_context.
- Do not invent experimental results, numbers, datasets, citations, or claims.
- Preserve section_id, slot_id, visual_assets, source_sections, and source_keypoint_ids.
- Keep each block's total text_content character count near target_chars.
- Do not go below min_chars unless source_context lacks enough facts; then set capacity_warning.
- Never exceed max_chars.
- For visual_policy="prioritize_visual_scale", keep text concise and do not compensate for small visuals with extra text.
- For generated_teaser_summary=true, preserve the existing short text_content exactly.
- Return clean, self-contained poster text items that can be rendered directly.
- Do not include literal bullet symbols, nested bullets, ordered-list prefixes, empty strings, or multiline items.
- Do not mention table or figure numbers such as "Table 2" or "Figure 3"; summarize the finding directly.
- Match target_bullets exactly when the source contains enough facts.
- Figure/table blocks normally use 1-2 short interpretation lines, but use target_bullets when it is 3 or more; this means the selected wide visual leaves real text capacity.
- Each item should be 8-22 words, complete, and not end with dangling connectors.
- Use bold lead-ins sparingly and consistently, e.g. "**Core idea:** ...", "**Result:** ...".

Return strict JSON only:
{{
  "blocks": [
    {{
      "section_id": "same id",
      "slot_id": "same slot",
      "target_title": "section title",
      "text_content": ["poster text item", "..."],
      "capacity_warning": null
    }}
  ]
}}

Template: {template_layout.get("template_name")}
Blocks:
{json.dumps(blocks, ensure_ascii=False, indent=2)}
""".strip()

    def _rewrite_story_board(
        self,
        base_story_board: Dict[str, Any],
        layout_intent: Dict[str, Any],
        template_layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        rewritten = deepcopy(base_story_board)
        sections = []
        for order, section in enumerate(layout_intent["active_sections"]):
            sections.append({
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "column_assignment": section["region_id"],
                "semantic_lane": section["region_meta"].get("semantic_lane", section.get("column_assignment", "middle")),
                "vertical_priority": section["region_meta"].get("vertical_band", "middle"),
                "text_content": section["text_content"],
                "visual_assets": section.get("visual_assets") or [],
                "content_role": section.get("content_role", "body"),
                "slot_id": section["region_id"],
                "template_prior": True,
                "source_sections": section.get("source_sections") or [],
                "keypoint_id": section.get("keypoint_id"),
                "source_keypoint_ids": section.get("source_keypoint_ids") or [],
                "source_section": section.get("source_section"),
                "preferred_slot_id": section.get("preferred_slot_id"),
                "order_index": order,
                "region_tier": section["region_meta"].get("region_tier"),
                "capacity_budget": section.get("capacity_budget"),
                "target_chars": section.get("target_chars"),
                "min_chars": section.get("min_chars"),
                "max_chars": section.get("max_chars"),
                "target_bullets": section.get("target_bullets"),
                "capacity_warning": section.get("capacity_warning"),
                "generated_teaser_summary": section.get("generated_teaser_summary"),
                "generated_teaser_original_text_count": section.get("generated_teaser_original_text_count"),
            })
        rewritten.setdefault("spatial_content_plan", {})
        rewritten["spatial_content_plan"]["sections"] = sections
        rewritten["layout_intent"] = layout_intent
        rewritten["template_layout"] = {
            "template_id": template_layout.get("template_name"),
            "hero_region_id": template_layout.get("hero_region_id"),
            "regions": template_layout.get("regions"),
        }
        return rewritten

    def _has_keypoints(self, state: PosterState) -> bool:
        return bool(state.get("paper_poster_keypoints"))

    def _keypoint_order_map(self, state: PosterState) -> Dict[int, int]:
        result = {}
        for index, value in enumerate(state.get("poster_reading_order") or []):
            keypoint_id = self._safe_int(value)
            if keypoint_id is not None:
                result[keypoint_id] = index
        return result

    def _keypoint_sort_value(self, section: Dict[str, Any]) -> int:
        value = self._safe_int(section.get("keypoint_id"))
        return value if value is not None else 999

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _infer_role(self, title: str) -> str:
        lowered = str(title or "").lower()
        if any(token in lowered for token in ["result", "experiment", "ablation", "evaluation", "performance"]):
            return "results"
        if any(token in lowered for token in ["method", "framework", "approach", "model", "pipeline", "search", "hierarchical"]):
            return "method"
        if any(token in lowered for token in ["conclusion", "discussion", "future", "limitation", "takeaway"]):
            return "takeaway"
        if any(token in lowered for token in ["setup", "data", "task", "objective"]):
            return "setup"
        return "overview"

    def _role_priority(self, role: str) -> int:
        return {
            "method": 0,
            "overview": 1,
            "results": 2,
            "takeaway": 3,
            "setup": 4,
        }.get(role, 5)

    def _priority_rank(self, priority: Optional[str]) -> int:
        return {"top": 0, "middle": 1, "bottom": 2}.get(str(priority or "middle"), 1)

    def _lane_rank(self, lane: Optional[str]) -> int:
        return {"left": 0, "middle": 1, "right": 2}.get(str(lane or "middle"), 1)

    def _visual_is_large(self, section: Dict[str, Any], state: PosterState) -> bool:
        visual_assets = state.get("visual_assets") or {}
        for visual in section.get("visual_assets") or []:
            asset = visual_assets.get(visual.get("visual_id"))
            if not asset:
                continue
            aspect = float(asset.get("aspect") or 1.0)
            if aspect >= 2.2 or asset.get("asset_type") == "table":
                return True
        return False

    def _save_outputs(self, state: PosterState) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "layout_intent.json", "w", encoding="utf-8") as f:
            json.dump(state.get("layout_intent", {}), f, indent=2)
        with open(output_dir / "template_block_plan.json", "w", encoding="utf-8") as f:
            json.dump(state.get("template_block_plan", {}), f, indent=2)
        with open(output_dir / "block_capacity_contract.json", "w", encoding="utf-8") as f:
            json.dump(state.get("block_capacity_contract", {}), f, indent=2)
        with open(output_dir / "capacity_planning_report.json", "w", encoding="utf-8") as f:
            json.dump(state.get("capacity_planning_report", {}), f, indent=2)
        with open(output_dir / "story_board.json", "w", encoding="utf-8") as f:
            json.dump(state.get("story_board", {}), f, indent=2)


TemplateBlockPlanner = TemplatePriorPlanner


def template_block_planner_node(state: PosterState) -> Dict[str, Any]:
    result = TemplatePriorPlanner()(state)
    return {
        **state,
        "story_board": result.get("story_board"),
        "template_block_plan": result.get("template_block_plan"),
        "layout_intent": result.get("layout_intent"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "template_layout_mode": result.get("template_layout_mode"),
        "resolved_layout_template": result.get("resolved_layout_template"),
        "block_capacity_contract": result.get("block_capacity_contract"),
        "capacity_aware_story_board": result.get("capacity_aware_story_board"),
        "capacity_planning_report": result.get("capacity_planning_report"),
        "tokens": result.get("tokens"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
