"""
Fast template-first capacity planner.

This deterministic node runs before keypoint selection so downstream agents
know the selected template's block capacity before writing content.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.template_extraction.block_template_registry import is_block_template_id
from src.tools.layout_api import LayoutTemplates
from src.utils.style_options import normalize_visual_density, resolve_visual_density_settings
from src.utils.visual_footprint import visual_footprint_config, visual_requirements, visual_slot_is_feasible
from utils.src.logging_utils import log_agent_info, log_agent_success


class TemplateCapacityPlanner:
    def __init__(self):
        self.name = "template_capacity_planner"
        self.config = load_config()
        self.fast_config = self.config.get("template_fast_mode", {})

    def __call__(self, state: PosterState) -> PosterState:
        template_name = str(state.get("resolved_layout_template") or state.get("layout_template") or "")
        enabled_templates = set(self.fast_config.get("templates") or ["cluster_43_landscape"])
        enabled = bool(self.fast_config.get("enabled", True))

        if not enabled or template_name not in enabled_templates or not is_block_template_id(template_name):
            state["template_fast_mode"] = False
            state["fast_pipeline_report"] = {
                "enabled": False,
                "reason": "template not configured for fast template-first mode",
                "template_id": template_name,
            }
            state["current_agent"] = self.name
            return state

        log_agent_info(self.name, f"building fast block contract for {template_name}")

        template_layout = self._resolve_template_layout(state, template_name)
        contract = self._build_contract(template_layout, template_name)
        visual_policy = self._build_visual_policy(template_name, contract, template_layout, state)
        report = {
            "enabled": True,
            "source": self.name,
            "template_id": template_name,
            "visual_density": visual_policy.get("visual_density"),
            "block_count": len(contract.get("blocks") or []),
            "slot_order": contract.get("slot_order") or [],
            "strategy": "template bbox and visual policy are fixed before keypoint/content generation",
            "emergency_repair_max_iterations": int(self.fast_config.get("emergency_repair_max_iterations", 1)),
            "hard_min_utilization": float(self.fast_config.get("hard_min_utilization", 0.88)),
            "gap_absorption_report": template_layout.get("gap_absorption_report"),
        }

        state["template_fast_mode"] = True
        state["resolved_layout_template"] = template_name
        state["layout_template_metadata"] = template_layout
        state["fast_block_contract"] = contract
        state["fast_visual_policy"] = visual_policy
        state["fast_pipeline_report"] = report
        state["current_agent"] = self.name
        self._save_outputs(state)

        log_agent_success(self.name, f"fast contract ready with {len(contract.get('blocks') or [])} blocks")
        return state

    def _resolve_template_layout(self, state: PosterState, template_name: str) -> Dict[str, Any]:
        return LayoutTemplates(
            state["poster_width"],
            state["poster_height"],
            margin=self.config["layout"]["poster_margin"],
            col_gap=self.config["layout"]["column_spacing"],
        ).get_template(template_name)

    def _build_contract(self, template_layout: Dict[str, Any], template_name: str) -> Dict[str, Any]:
        regions = {
            str(region.get("region_id") or region.get("slot_id") or region.get("id") or ""): region
            for region in template_layout.get("regions") or []
        }
        slot_specs = self._slot_specs_for_template(template_layout, template_name, regions)
        slot_order = [str(slot_id) for slot_id in template_layout.get("slot_order") or [] if str(slot_id) in regions]
        slot_order.extend(
            slot_id
            for slot_id in sorted(regions, key=lambda sid: (float(regions[sid].get("y", 0.0)), float(regions[sid].get("x", 0.0))))
            if slot_id not in slot_order
        )

        blocks: List[Dict[str, Any]] = []
        for slot_id in slot_order:
            if slot_id not in slot_specs:
                continue
            region = dict(regions[slot_id])
            region.setdefault("poster_orientation", template_layout.get("orientation"))
            spec = slot_specs[slot_id]
            min_chars = int(spec.get("min_chars", 90))
            target_chars = int(spec.get("target_chars", max(min_chars, 120)))
            max_chars = int(spec.get("max_chars", max(target_chars, min_chars)))
            blocks.append({
                "slot_id": slot_id,
                "section_id": None,
                "slot_role": spec.get("role") or slot_id,
                "content_role": spec.get("content_role") or "body",
                "source_keypoint_ids": list(spec.get("keypoint_ids") or []),
                "slot_bbox": {
                    "x": round(float(region.get("x", 0.0) or 0.0), 4),
                    "y": round(float(region.get("y", 0.0) or 0.0), 4),
                    "w": round(float(region.get("w", 0.0) or 0.0), 4),
                    "h": round(float(region.get("h", 0.0) or 0.0), 4),
                },
                "target_chars": target_chars,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "target_bullets": int(spec.get("target_bullets", max(1, round(target_chars / 120)))),
                "visual_policy": spec.get("visual_policy") or "text_only",
                "target_utilization": float(self.config.get("block_refinement", {}).get("target_utilization", 0.95)),
                "acceptable_min": float(self.config.get("block_refinement", {}).get("acceptable_min", 0.90)),
                "acceptable_max": float(self.config.get("block_refinement", {}).get("acceptable_max", 0.97)),
                "hard_min_utilization": float(self.fast_config.get("hard_min_utilization", 0.88)),
                "hard_max": float(self.config.get("block_refinement", {}).get("hard_max", 0.98)),
                "capacity_warning": None,
                "visual_footprint": self._visual_footprint_for_policy(spec.get("visual_policy") or "text_only", region),
                "source": "fast_template_first_fixed_contract",
            })

        return {
            "source": "fast_template_first_fixed_contract",
            "template_id": template_name,
            "settings": {
                "target_utilization": float(self.config.get("block_refinement", {}).get("target_utilization", 0.95)),
                "acceptable_min": float(self.config.get("block_refinement", {}).get("acceptable_min", 0.90)),
                "acceptable_max": float(self.config.get("block_refinement", {}).get("acceptable_max", 0.97)),
                "hard_min_utilization": float(self.fast_config.get("hard_min_utilization", 0.88)),
                "hard_max": float(self.config.get("block_refinement", {}).get("hard_max", 0.98)),
            },
            "slot_order": [block["slot_id"] for block in blocks],
            "blocks": blocks,
            "by_slot": {block["slot_id"]: block for block in blocks},
        }

    def _build_visual_policy(
        self,
        template_name: str,
        contract: Dict[str, Any],
        template_layout: Dict[str, Any],
        state: PosterState,
    ) -> Dict[str, Any]:
        policy = self.fast_config.get("visual_policy") or {}
        blocks = contract.get("blocks") or []
        density_settings = resolve_visual_density_settings(state, self.config)
        visual_density = normalize_visual_density(state.get("visual_density"), self.config)
        is_portrait = template_layout.get("orientation") == "portrait"
        figure_count = int(
            density_settings.get(
                "portrait_figure_count" if is_portrait else "figure_count",
                policy.get("figure_count", 2),
            )
        )
        table_count = int(
            density_settings.get(
                "portrait_table_count" if is_portrait else "table_count",
                policy.get("table_count", 1),
            )
        )
        max_visuals_total = int(
            density_settings.get(
                "portrait_max_visuals_total" if is_portrait else "max_visuals_total",
                figure_count + table_count,
            )
        )
        figure_slots = [block["slot_id"] for block in blocks if "figure" in str(block.get("visual_policy") or "")]
        table_slots = [block["slot_id"] for block in blocks if "table" in str(block.get("visual_policy") or "")]
        figure_slots = self._sort_slots_by_area(
            self._merge_slot_order(figure_slots, list(policy.get("figure_slots") or ["slot_2", "slot_3"])),
            contract,
        )
        table_slots = self._sort_slots_by_area(
            self._merge_slot_order(table_slots, list(policy.get("table_slots") or ["slot_4"])),
            contract,
        )
        rejected_visual_slots: Dict[str, List[Dict[str, Any]]] = {"figure": [], "table": []}
        figure_slots, rejected_visual_slots["figure"] = self._filter_visual_slots(
            figure_slots,
            contract,
            template_layout,
            "figure",
        )
        table_slots, rejected_visual_slots["table"] = self._filter_visual_slots(
            table_slots,
            contract,
            template_layout,
            "table",
        )
        if is_portrait:
            figure_slots, portrait_rejected_figures = self._filter_portrait_visual_slots(
                figure_slots,
                contract,
                "figure",
            )
            table_slots, portrait_rejected_tables = self._filter_portrait_visual_slots(
                table_slots,
                contract,
                "table",
            )
            rejected_visual_slots["figure"].extend(portrait_rejected_figures)
            rejected_visual_slots["table"].extend(portrait_rejected_tables)
            self._retarget_rejected_portrait_visual_slots(
                contract,
                rejected_visual_slots,
                {*figure_slots, *table_slots},
            )
            figure_count = min(figure_count, len(figure_slots))
            table_count = min(table_count, len(table_slots))
            max_visuals_total = min(max_visuals_total, figure_count + table_count)
        else:
            table_count = min(table_count, len(table_slots))
            max_visuals_total = min(max_visuals_total, figure_count + table_count)
        return {
            "source": "fast_template_first_visual_policy",
            "template_id": template_name,
            "visual_density": visual_density,
            "figure_count": figure_count,
            "table_count": table_count,
            "figure_slots": figure_slots[:figure_count],
            "table_slots": table_slots[:table_count],
            "max_visuals_total": max(0, min(max_visuals_total, figure_count + table_count)),
            "figure_max_height_fraction": policy.get("figure_max_height_fraction", 0.55),
            "table_max_height_fraction": policy.get("table_max_height_fraction", 0.62),
            "default_max_height_fraction": policy.get("default_max_height_fraction", 0.42),
            "table_unreadable_strategy": policy.get("table_unreadable_strategy") or "summarize_as_text",
            "visual_footprint": visual_footprint_config(self.config),
            "rejected_visual_slots": rejected_visual_slots,
        }

    def _visual_footprint_for_policy(self, visual_policy: str, region: Dict[str, Any]) -> Dict[str, Any]:
        visual_policy = str(visual_policy or "")
        if "figure" in visual_policy:
            return visual_requirements("figure_contract", {"asset_type": "figure"}, region, self.config)
        if "table" in visual_policy:
            return visual_requirements("table_contract", {"asset_type": "table"}, region, self.config)
        return {"enabled": bool(visual_footprint_config(self.config).get("enabled", True)), "visual_type": "none"}

    def _merge_slot_order(self, primary: List[str], fallback: List[str]) -> List[str]:
        ordered: List[str] = []
        for slot_id in [*primary, *fallback]:
            slot_id = str(slot_id or "").strip()
            if slot_id and slot_id not in ordered:
                ordered.append(slot_id)
        return ordered

    def _sort_slots_by_area(self, slot_ids: List[str], contract: Dict[str, Any]) -> List[str]:
        by_slot = contract.get("by_slot") or {}

        def area(slot_id: str) -> float:
            bbox = (by_slot.get(slot_id) or {}).get("slot_bbox") or {}
            return float(bbox.get("w", 0.0) or 0.0) * float(bbox.get("h", 0.0) or 0.0)

        original_index = {slot_id: index for index, slot_id in enumerate(slot_ids)}
        return sorted(slot_ids, key=lambda slot_id: (-area(slot_id), original_index.get(slot_id, 999)))

    def _filter_visual_slots(
        self,
        slot_ids: List[str],
        contract: Dict[str, Any],
        template_layout: Dict[str, Any],
        visual_kind: str,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        by_slot = contract.get("by_slot") or {}
        region_by_slot = {
            str(region.get("region_id") or region.get("slot_id") or region.get("id") or ""): region
            for region in template_layout.get("regions") or []
        }
        visual_id = f"{visual_kind}_contract"
        visual_assets = {
            visual_id: {
                "asset_type": visual_kind,
                "aspect": 1.5 if visual_kind == "table" else 2.0,
            }
        }
        text_padding = 2 * float(self.config["layout"]["text_padding"]["left_right"])
        kept: List[str] = []
        rejected: List[Dict[str, Any]] = []
        for slot_id in slot_ids:
            block = by_slot.get(slot_id)
            if not block:
                continue
            bbox = block.get("slot_bbox") or {}
            region = dict(region_by_slot.get(slot_id) or {})
            region.setdefault("id", slot_id)
            region.setdefault("slot_id", slot_id)
            region.setdefault("region_id", slot_id)
            region.setdefault("x", bbox.get("x", 0.0))
            region.setdefault("y", bbox.get("y", 0.0))
            region.setdefault("w", bbox.get("w", 0.0))
            region.setdefault("h", bbox.get("h", 0.0))
            region.setdefault("poster_orientation", template_layout.get("orientation"))
            max_width = max(float(region.get("w", 0.0) or 0.0) - text_padding, 0.1)
            failed = []
            if not bool(region.get("can_host_visual", True)):
                failed.append("visual_host")
            if not visual_slot_is_feasible(
                visual_id,
                region,
                visual_assets,
                self.config,
                max_width=max_width,
            ):
                failed.append("footprint")
            if failed:
                rejected.append({
                    "slot_id": slot_id,
                    "visual_type": visual_kind,
                    "width": round(float(region.get("w", 0.0) or 0.0), 4),
                    "height": round(float(region.get("h", 0.0) or 0.0), 4),
                    "failed": failed,
                })
                continue
            kept.append(slot_id)
        return kept, rejected

    def _filter_portrait_visual_slots(
        self,
        slot_ids: List[str],
        contract: Dict[str, Any],
        visual_kind: str,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Keep portrait visuals out of narrow slots so figures land in larger blocks."""
        cfg = visual_footprint_config(self.config)
        by_slot = contract.get("by_slot") or {}
        prefix = f"portrait_{visual_kind}"
        min_width = float(cfg.get(f"{prefix}_min_slot_width_inches", 0.0) or 0.0)
        min_height = float(cfg.get(f"{prefix}_min_slot_height_inches", 0.0) or 0.0)
        min_area = float(cfg.get(f"{prefix}_min_slot_area_inches", 0.0) or 0.0)

        kept: List[str] = []
        rejected: List[Dict[str, Any]] = []
        for slot_id in slot_ids:
            if slot_id not in by_slot:
                continue
            bbox = ((by_slot.get(slot_id) or {}).get("slot_bbox") or {})
            width = float(bbox.get("w", 0.0) or 0.0)
            height = float(bbox.get("h", 0.0) or 0.0)
            area = width * height
            failed = []
            if min_width and width < min_width:
                failed.append("width")
            if min_height and height < min_height:
                failed.append("height")
            if min_area and area < min_area:
                failed.append("area")
            if failed:
                rejected.append({
                    "slot_id": slot_id,
                    "visual_type": visual_kind,
                    "width": round(width, 4),
                    "height": round(height, 4),
                    "area": round(area, 4),
                    "failed": failed,
                })
                continue
            kept.append(slot_id)

        return kept, rejected

    def _retarget_rejected_portrait_visual_slots(
        self,
        contract: Dict[str, Any],
        rejected_visual_slots: Dict[str, List[Dict[str, Any]]],
        selected_visual_slots: set[str],
    ) -> None:
        rejected_slot_ids = {
            str(item.get("slot_id") or "")
            for items in rejected_visual_slots.values()
            for item in items
        }
        rejected_slot_ids = {slot_id for slot_id in rejected_slot_ids if slot_id and slot_id not in selected_visual_slots}
        if not rejected_slot_ids:
            return

        for block in contract.get("blocks") or []:
            slot_id = str(block.get("slot_id") or "")
            if slot_id not in rejected_slot_ids:
                continue
            bbox = block.get("slot_bbox") or {}
            region = {
                "x": bbox.get("x", 0.0),
                "y": bbox.get("y", 0.0),
                "w": bbox.get("w", 0.0),
                "h": bbox.get("h", 0.0),
                "poster_orientation": "portrait",
            }
            capacity = self._estimate_capacity(region, "text_summary")
            block.update({
                "visual_policy": "text_summary",
                "visual_footprint": {
                    "enabled": bool(visual_footprint_config(self.config).get("enabled", True)),
                    "visual_type": "none",
                },
                "capacity_warning": "visual_slot_too_narrow_text_fallback",
                "target_chars": capacity["target_chars"],
                "min_chars": capacity["min_chars"],
                "max_chars": capacity["max_chars"],
                "target_bullets": capacity["target_bullets"],
            })

    def _slot_specs_for_template(
        self,
        template_layout: Dict[str, Any],
        template_name: str,
        regions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        by_template = (self.fast_config.get("template_slot_contracts") or {}).get(template_name)
        if by_template:
            return by_template

        slot_order = [str(slot_id) for slot_id in template_layout.get("slot_order") or [] if str(slot_id) in regions]
        if not slot_order:
            slot_order = sorted(regions, key=lambda sid: (float(regions[sid].get("y", 0.0)), float(regions[sid].get("x", 0.0))))
        keypoint_groups = self._keypoint_groups_for_slots(len(slot_order))
        roles = self._roles_for_slots(len(slot_order))
        visual_policy_config = self.fast_config.get("visual_policy") or {}
        configured_figure_slots = {str(slot_id) for slot_id in visual_policy_config.get("figure_slots") or []}
        configured_table_slots = {str(slot_id) for slot_id in visual_policy_config.get("table_slots") or []}

        specs: Dict[str, Dict[str, Any]] = {}
        for index, slot_id in enumerate(slot_order):
            role_spec = roles[min(index, len(roles) - 1)]
            visual_policy = self._capacity_visual_policy_for_slot(
                slot_id,
                role_spec["visual_policy"],
                configured_figure_slots,
                configured_table_slots,
            )
            region = dict(regions[slot_id])
            region.setdefault("poster_orientation", template_layout.get("orientation"))
            capacity = self._estimate_capacity(region, visual_policy)
            specs[slot_id] = {
                "role": role_spec["role"],
                "content_role": role_spec["content_role"],
                "visual_policy": visual_policy,
                "min_chars": capacity["min_chars"],
                "target_chars": capacity["target_chars"],
                "max_chars": capacity["max_chars"],
                "target_bullets": capacity["target_bullets"],
                "keypoint_ids": keypoint_groups[index] if index < len(keypoint_groups) else [],
            }
        return specs

    def _capacity_visual_policy_for_slot(
        self,
        slot_id: str,
        role_policy: str,
        configured_figure_slots: set[str],
        configured_table_slots: set[str],
    ) -> str:
        if slot_id in configured_table_slots and "figure" not in role_policy:
            return "table_with_callouts"
        if slot_id in configured_figure_slots and "table" not in role_policy:
            return "figure_caption"
        return role_policy

    def _keypoint_groups_for_slots(self, slot_count: int) -> List[List[int]]:
        if slot_count <= 0:
            return []
        if slot_count == 4:
            return [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10]]
        if slot_count == 5:
            return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        if slot_count == 6:
            return [[1, 2], [3], [4], [5, 6], [7, 8], [9, 10]]
        if slot_count == 7:
            return [[1, 2], [3], [4], [5, 6], [7], [8, 9], [10]]
        groups = []
        for index in range(slot_count):
            start = math.floor(index * 10 / slot_count) + 1
            end = math.floor((index + 1) * 10 / slot_count)
            if end < start:
                end = start
            groups.append(list(range(start, min(end, 10) + 1)))
        return groups

    def _roles_for_slots(self, slot_count: int) -> List[Dict[str, str]]:
        if slot_count <= 4:
            return [
                {"role": "Motivation", "content_role": "foundation", "visual_policy": "text_only"},
                {"role": "Method and system visual", "content_role": "method", "visual_policy": "figure_caption"},
                {"role": "Method details", "content_role": "method", "visual_policy": "text_summary"},
                {"role": "Main results with table", "content_role": "results", "visual_policy": "table_with_callouts"},
            ]
        if slot_count == 5:
            return [
                {"role": "Motivation", "content_role": "foundation", "visual_policy": "text_only"},
                {"role": "Method visual", "content_role": "method", "visual_policy": "figure_caption"},
                {"role": "System details", "content_role": "method", "visual_policy": "figure_caption"},
                {"role": "Evaluation results", "content_role": "results", "visual_policy": "table_with_callouts"},
                {"role": "Takeaway", "content_role": "results", "visual_policy": "text_summary"},
            ]
        return [
            {"role": "Motivation", "content_role": "foundation", "visual_policy": "text_only"},
            {"role": "Method visual", "content_role": "method", "visual_policy": "figure_caption"},
            {"role": "System flow", "content_role": "method", "visual_policy": "figure_caption"},
            {"role": "Method details", "content_role": "method", "visual_policy": "text_summary"},
            {"role": "Evaluation setup", "content_role": "results", "visual_policy": "text_summary"},
            {"role": "Main results with table", "content_role": "results", "visual_policy": "table_with_callouts"},
            {"role": "Takeaway", "content_role": "results", "visual_policy": "text_summary"},
        ]

    def _estimate_capacity(self, region: Dict[str, Any], visual_policy: str) -> Dict[str, int]:
        block_cfg = self.config.get("block_refinement", {})
        typography = self.config.get("typography", {})
        width = max(float(region.get("w", 0.0) or 0.0), 0.5)
        height = max(float(region.get("h", 0.0) or 0.0), 0.5)
        target_used_height = height * float(block_cfg.get("target_utilization", 0.95))
        title_height = float(block_cfg.get("title_height_inches", 1.0))
        gap = float(block_cfg.get("title_content_gap_inches", 0.4))
        padding = float(block_cfg.get("section_padding_inches", 0.4))
        orientation = str(region.get("poster_orientation") or "").lower()
        if not orientation:
            orientation = "portrait" if height > width else "landscape"
        footprint_cfg = visual_footprint_config(self.config)
        usable_width = max(width - padding * 2, 0.5)
        split_text_width = usable_width
        portrait_side_by_side_figure = False
        if orientation == "portrait" and "figure" in str(visual_policy or ""):
            split_min_width = float(footprint_cfg.get("portrait_split_min_width_inches", 18.0) or 18.0)
            split_min_height = float(footprint_cfg.get("portrait_split_min_height_inches", 4.8) or 4.8)
            split_min_aspect = float(footprint_cfg.get("portrait_split_min_aspect", 2.35) or 2.35)
            if width >= split_min_width and height >= split_min_height and width / max(height, 0.01) >= split_min_aspect:
                split_gap = float(footprint_cfg.get("portrait_split_gap_inches", 0.45) or 0.45)
                visual_fraction = float(footprint_cfg.get("portrait_split_visual_width_fraction", 0.48) or 0.48)
                min_text_width = float(footprint_cfg.get("portrait_split_min_text_width_inches", 8.0) or 8.0)
                split_visual_width = min(
                    usable_width * visual_fraction,
                    usable_width - split_gap - min_text_width,
                )
                if split_visual_width > 0:
                    split_text_width = max(usable_width - split_visual_width - split_gap, min_text_width)
                    portrait_side_by_side_figure = True
        visual_reserved = 0.0
        if "figure" in visual_policy and not portrait_side_by_side_figure:
            footprint = visual_requirements("figure_contract", {"asset_type": "figure"}, region, self.config)
            visual_fraction = float((self.fast_config.get("visual_policy") or {}).get("figure_max_height_fraction", 0.70))
            visual_reserved = max(height * visual_fraction, float(footprint.get("min_height") or 0.0))
        elif "table" in visual_policy:
            footprint = visual_requirements("table_contract", {"asset_type": "table"}, region, self.config)
            visual_fraction = float((self.fast_config.get("visual_policy") or {}).get("table_max_height_fraction", 0.74))
            visual_reserved = max(height * visual_fraction, float(footprint.get("min_height") or 0.0))
        line_height = max((float((typography.get("sizes") or {}).get("body_text", 44)) / 72.0) * 1.05, 0.32)
        chars_per_inch = self._chars_per_inch(region)
        chars_per_line = max(20, int(max(split_text_width, 0.5) * chars_per_inch))
        text_height_budget = max(target_used_height - title_height - gap - padding - visual_reserved, line_height)
        target_lines = max(1, int(math.floor(text_height_budget / line_height)))
        target_chars = int(target_lines * chars_per_line * float(block_cfg.get("safety_factor", 0.82)))
        min_capacity = int(block_cfg.get("visual_min_text_chars" if visual_reserved else "min_capacity_chars", 90))
        max_capacity = int(block_cfg.get("max_capacity_chars", 900))
        target_chars = max(min_capacity, min(target_chars, max_capacity))
        min_chars = max(min_capacity, int(target_chars * float(block_cfg.get("capacity_min_factor", 0.88))))
        max_chars = min(max_capacity, max(target_chars, int(target_chars * float(block_cfg.get("capacity_max_factor", 1.08)))))
        return {
            "min_chars": min_chars,
            "target_chars": target_chars,
            "max_chars": max_chars,
            "target_bullets": max(1, min(6, round(target_chars / int(block_cfg.get("chars_per_bullet", 120))))),
        }

    def _chars_per_inch(self, region: Dict[str, Any]) -> float:
        block_cfg = self.config.get("block_refinement", {})
        micro_cfg = self.config.get("micro_layout_refinement", {})
        default = float(block_cfg.get("ppt_chars_per_inch_at_44pt", micro_cfg.get("ppt_chars_per_inch_at_44pt", 4.2)))
        orientation = str(region.get("poster_orientation") or "").lower()
        if not orientation:
            width = float(region.get("w", 0.0) or 0.0)
            height = float(region.get("h", 0.0) or 0.0)
            orientation = "portrait" if height > width else "landscape"
        if orientation == "portrait":
            return float(
                block_cfg.get(
                    "portrait_ppt_chars_per_inch_at_44pt",
                    micro_cfg.get("portrait_ppt_chars_per_inch_at_44pt", default),
                )
            )
        return default

    def _default_slot_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "slot_1": {"role": "Motivation", "content_role": "foundation", "visual_policy": "text_only", "min_chars": 450, "target_chars": 500, "max_chars": 550, "target_bullets": 4, "keypoint_ids": [1, 2]},
            "slot_2": {"role": "Stable estimation figure", "content_role": "method", "visual_policy": "figure_caption", "min_chars": 80, "target_chars": 100, "max_chars": 120, "target_bullets": 1, "keypoint_ids": [3]},
            "slot_3": {"role": "Method or architecture figure", "content_role": "method", "visual_policy": "figure_caption", "min_chars": 80, "target_chars": 100, "max_chars": 120, "target_bullets": 1, "keypoint_ids": [4]},
            "slot_4": {"role": "Main results with table", "content_role": "results", "visual_policy": "table_with_callouts", "min_chars": 280, "target_chars": 320, "max_chars": 360, "target_bullets": 3, "keypoint_ids": [9, 10]},
            "slot_5": {"role": "Robustness and results summary", "content_role": "results", "visual_policy": "text_summary", "min_chars": 520, "target_chars": 590, "max_chars": 650, "target_bullets": 5, "keypoint_ids": [7, 8]},
            "slot_6": {"role": "Method details", "content_role": "method", "visual_policy": "text_only", "min_chars": 430, "target_chars": 500, "max_chars": 560, "target_bullets": 4, "keypoint_ids": [5, 6]},
        }

    def _save_outputs(self, state: PosterState) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, payload in {
            "fast_block_contract.json": state.get("fast_block_contract") or {},
            "fast_visual_policy.json": state.get("fast_visual_policy") or {},
            "fast_pipeline_report.json": state.get("fast_pipeline_report") or {},
        }.items():
            with open(output_dir / filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)


def template_capacity_planner_node(state: PosterState) -> Dict[str, Any]:
    result = TemplateCapacityPlanner()(state)
    return {
        **state,
        "template_fast_mode": result.get("template_fast_mode", False),
        "fast_block_contract": result.get("fast_block_contract"),
        "fast_visual_policy": result.get("fast_visual_policy"),
        "fast_pipeline_report": result.get("fast_pipeline_report"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "resolved_layout_template": result.get("resolved_layout_template"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
