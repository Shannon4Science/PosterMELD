"""
Deterministic post-font layout refinement.

This stage keeps the semantic three-lane reading flow intact while tightening
section geometry, spacing, and font sizes to avoid overlap and overflow before
visual assets are resolved and rendered.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.poster_config import load_config
from src.layout.text_height_measurement import measure_text_height
from src.state.poster_state import PosterState
from src.tools.layout_api import LayoutTemplates
from src.utils.text_cleanup import fit_complete_sentence_prefix, repair_truncated_sentence_end
from src.utils.visual_footprint import enforce_visual_footprint, visual_footprint_config
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class MicroLayoutRefiner:
    def __init__(self):
        self.name = "micro_layout_refiner"
        self.config = load_config()
        self.refine_config = self.config["micro_layout_refinement"]
        self.layout_config = self.config["layout"]
        self.typography_config = self.config["typography"]

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "refining styled layout for final fit")
        state["draft_status"] = "pending"
        state["final_poster_accepted"] = False

        try:
            styled_layout = state.get("styled_layout") or []
            if not styled_layout:
                raise ValueError("missing styled_layout from font agent")

            template_layout = state.get("layout_template_metadata") or self._resolve_template_layout(state)
            refined_layout, report = self._refine_layout(styled_layout, template_layout, state)

            state["styled_layout"] = refined_layout
            state["current_agent"] = self.name
            self._save_outputs(state, report)

            if report["validation"]["issues"]:
                state["draft_status"] = "rejected"
                state["draft_rejection_reason"] = (
                    "layout refinement failed validation: "
                    + "; ".join(report["validation"]["issues"])
                )
                log_agent_warning(self.name, state["draft_rejection_reason"])
                return state
            state["draft_status"] = "accepted"
            if report["force_fit_used"]:
                log_agent_warning(self.name, "force-fit fallback used on at least one lane")
            log_agent_success(self.name, f"refined layout with {report['validated_elements']} positioned elements")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["draft_status"] = "rejected"
            state["draft_rejection_reason"] = str(e)
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _resolve_template_layout(self, state: PosterState) -> Dict[str, Any]:
        poster_width = state["poster_width"]
        poster_height = state["poster_height"]
        poster_margin = self.layout_config["poster_margin"]
        column_spacing = self.layout_config["column_spacing"]
        title_height_fraction = self.layout_config["title_height_fraction"]
        effective_height = poster_height - 2 * poster_margin
        title_region_height = effective_height * title_height_fraction

        requested_template = state.get("resolved_layout_template") or state.get("layout_template", "three_column_postergen")
        return LayoutTemplates(
            poster_width,
            poster_height,
            margin=poster_margin,
            col_gap=column_spacing,
        ).get_template(requested_template, header_height=title_region_height)

    def _refine_layout(self, styled_layout: List[Dict[str, Any]], template_layout: Dict[str, Any], state: PosterState) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        lane_map = {lane["id"]: lane for lane in template_layout["lanes"]}
        title_and_global = []
        section_containers = []
        section_groups: Dict[str, Dict[str, Any]] = {}

        for element in styled_layout:
            if element.get("type") == "section_container":
                lane_id = element.get("lane_id") or self._match_lane_for_element(element, lane_map)
                group = {
                    "section_id": element["section_id"],
                    "lane_id": lane_id,
                    "container": deepcopy(element),
                    "children": [],
                }
                section_containers.append(group)
                section_groups[element["section_id"]] = group
            elif element.get("type") in {
                "title",
                "conf_logo",
                "aff_logo",
                "institution_logo",
                "logo_divider",
                "qr_code",
                "template_background",
                "template_header_background",
                "template_footer_background",
            }:
                title_and_global.append(deepcopy(element))

        for element in styled_layout:
            if element.get("type") == "section_container" or element.get("type") in {
                "title",
                "conf_logo",
                "aff_logo",
                "institution_logo",
                "logo_divider",
                "qr_code",
                "template_background",
                "template_header_background",
                "template_footer_background",
            }:
                continue
            section_id = self._assign_section_id(element, section_containers, lane_map)
            if section_id and section_id in section_groups:
                section_groups[section_id]["children"].append(deepcopy(element))
            else:
                title_and_global.append(deepcopy(element))

        for group in section_containers:
            group["children"].sort(key=lambda item: (item.get("y", 0), item.get("priority", 0.5)))

        if template_layout.get("layout_mode") == "template_prior":
            lane_map = self._rebalance_template_block_slots(template_layout, section_containers, lane_map, state)
        lane_map = self._rebalance_soft_template_lanes(template_layout, section_containers, lane_map, state)

        lane_reports = []
        refined_elements = list(title_and_global)
        force_fit_used = False

        ordered_lane_ids = [lane["id"] for lane in template_layout["lanes"]]
        for lane_id in ordered_lane_ids:
            groups = [group for group in section_containers if group["lane_id"] == lane_id]
            groups.sort(key=lambda group: group["container"].get("y", 0))
            lane_result = self._refine_lane(groups, lane_map[lane_id], state, template_layout)
            refined_elements.extend(lane_result["elements"])
            lane_reports.append(lane_result["report"])
            force_fit_used = force_fit_used or lane_result["report"]["force_fit_used"]

        refined_elements.sort(key=lambda item: (item.get("priority", 0.5), item.get("y", 0), item.get("x", 0)))

        validation = self._validate_refined_layout(refined_elements, lane_map, state)
        report = {
            "template_name": template_layout["template_name"],
            "force_fit_used": force_fit_used,
            "lanes": lane_reports,
            "validation": validation,
            "validated_elements": len(refined_elements),
        }
        return refined_elements, report

    def _rebalance_template_block_slots(
        self,
        template_layout: Dict[str, Any],
        section_containers: List[Dict[str, Any]],
        lane_map: Dict[str, Dict[str, Any]],
        state: PosterState,
    ) -> Dict[str, Dict[str, Any]]:
        if template_layout.get("layout_mode") != "template_prior":
            return lane_map

        adjacency_graph = template_layout.get("adjacency_graph") or {}
        if not adjacency_graph:
            return lane_map

        params = {
            "section_gap": self.layout_config["section_spacing"],
            "title_to_content_gap": self.layout_config["title_to_content_spacing"],
            "visual_gap": self.layout_config["visual_spacing"]["below_visual"],
            "text_padding": self.layout_config["text_padding"]["left_right"],
            "body_font_reduction": 0,
            "title_font_reduction": 0,
            "body_font_boost": 0,
            "title_font_boost": 0,
            "visual_scale": 1.0,
        }
        demand_by_lane: Dict[str, float] = {}
        report_slots: Dict[str, Any] = {}
        for group in section_containers:
            lane_id = group["lane_id"]
            lane = dict(lane_map[lane_id])
            lane["y"] = 0.0
            lane["h"] = 1000.0
            _, section_bottom = self._layout_section(group, lane, 0.0, state, params, template_layout)
            demand = max(section_bottom, group["container"].get("height", 0.25))
            demand_by_lane[lane_id] = demand
        for lane_id, lane in lane_map.items():
            available = max(lane.get("h", 0.1), 0.1)
            pressure = demand_by_lane.get(lane_id, 0.0) / available
            report_slots[lane_id] = {
                "slot_id": lane_id,
                "demanded_height": round(demand_by_lane.get(lane_id, 0.0), 4),
                "available_height": round(available, 4),
                "pressure": round(pressure, 4),
            }

        updated = {lane_id: dict(lane) for lane_id, lane in lane_map.items()}
        max_shift_ratio = 0.10
        gutter = max(self.layout_config.get("column_spacing", 1.0) * 0.2, 0.08)
        transferred = False

        receivers = sorted(
            (lane_id for lane_id, slot in report_slots.items() if slot["pressure"] > 1.0),
            key=lambda lane_id: report_slots[lane_id]["pressure"],
            reverse=True,
        )
        for receiver_id in receivers:
            neighbors = adjacency_graph.get(receiver_id) or []
            donors = [
                neighbor for neighbor in neighbors
                if report_slots.get(neighbor["slot_id"], {}).get("pressure", 1.0) < 0.8
            ]
            donors.sort(key=lambda item: report_slots[item["slot_id"]]["pressure"])
            for donor_edge in donors:
                donor_id = donor_edge["slot_id"]
                receiver = updated[receiver_id]
                donor = updated[donor_id]
                if donor_edge.get("orientation") == "vertical":
                    transferred = self._transfer_slot_width(receiver, donor, max_shift_ratio, gutter) or transferred
                else:
                    transferred = self._transfer_slot_height(receiver, donor, max_shift_ratio, gutter) or transferred
                if transferred:
                    break
            if transferred:
                break

        if transferred:
            overlap_pair = self._first_overlapping_template_lane_pair(updated)
            if overlap_pair:
                state["slot_pressure_report"] = {
                    "slots": report_slots,
                    "slot_resize_applied": False,
                    "slot_resize_skipped": "resize_would_overlap_template_slots",
                    "overlap_pair": overlap_pair,
                }
                return lane_map
            ordered_ids = template_layout.get("slot_order") or [lane["id"] for lane in template_layout.get("lanes", [])]
            template_layout["lanes"] = [updated[lane_id] for lane_id in ordered_ids if lane_id in updated]
            template_layout["columns"] = template_layout["lanes"]
            state["slot_pressure_report"] = {
                "slots": report_slots,
                "slot_resize_applied": True,
            }
            return updated

        state["slot_pressure_report"] = {
            "slots": report_slots,
            "slot_resize_applied": False,
        }
        return updated

    def _first_overlapping_template_lane_pair(self, lane_map: Dict[str, Dict[str, Any]]) -> Optional[List[str]]:
        lanes = list(lane_map.values())
        for index, left in enumerate(lanes):
            for right in lanes[index + 1:]:
                if self._lane_boxes_overlap(left, right):
                    return [str(left.get("id") or ""), str(right.get("id") or "")]
        return None

    def _lane_boxes_overlap(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        tolerance = 0.02
        left_x = float(left.get("x", 0.0) or 0.0)
        left_y = float(left.get("y", 0.0) or 0.0)
        left_right = left_x + float(left.get("w", left.get("width", 0.0)) or 0.0)
        left_bottom = left_y + float(left.get("h", left.get("height", 0.0)) or 0.0)
        right_x = float(right.get("x", 0.0) or 0.0)
        right_y = float(right.get("y", 0.0) or 0.0)
        right_right = right_x + float(right.get("w", right.get("width", 0.0)) or 0.0)
        right_bottom = right_y + float(right.get("h", right.get("height", 0.0)) or 0.0)
        return not (
            left_right <= right_x + tolerance
            or right_right <= left_x + tolerance
            or left_bottom <= right_y + tolerance
            or right_bottom <= left_y + tolerance
        )

    def _transfer_slot_width(self, receiver: Dict[str, Any], donor: Dict[str, Any], max_shift_ratio: float, gutter: float) -> bool:
        shift = min(donor["w"] * max_shift_ratio, donor["w"] - 1.6)
        if shift <= 0.05:
            return False
        receiver_right = receiver["x"] + receiver["w"]
        donor_right = donor["x"] + donor["w"]
        if receiver["x"] < donor["x"] and abs(receiver_right - donor["x"]) < 0.4:
            new_receiver_w = receiver["w"] + shift
            new_donor_x = donor["x"] + shift
            new_donor_w = donor["w"] - shift
            if new_donor_w <= 1.2:
                return False
            receiver["w"] = new_receiver_w
            donor["x"] = new_donor_x
            donor["w"] = new_donor_w
            return True
        if donor["x"] < receiver["x"] and abs(donor_right - receiver["x"]) < 0.4:
            new_receiver_x = receiver["x"] - shift
            new_receiver_w = receiver["w"] + shift
            new_donor_w = donor["w"] - shift
            if new_donor_w <= 1.2:
                return False
            receiver["x"] = new_receiver_x
            receiver["w"] = new_receiver_w
            donor["w"] = new_donor_w
            return True
        return False

    def _transfer_slot_height(self, receiver: Dict[str, Any], donor: Dict[str, Any], max_shift_ratio: float, gutter: float) -> bool:
        shift = min(donor["h"] * max_shift_ratio, donor["h"] - 1.4)
        if shift <= 0.05:
            return False
        receiver_bottom = receiver["y"] + receiver["h"]
        donor_bottom = donor["y"] + donor["h"]
        if receiver["y"] < donor["y"] and abs(receiver_bottom - donor["y"]) < 0.4:
            new_receiver_h = receiver["h"] + shift
            new_donor_y = donor["y"] + shift
            new_donor_h = donor["h"] - shift
            if new_donor_h <= 1.0:
                return False
            receiver["h"] = new_receiver_h
            donor["y"] = new_donor_y
            donor["h"] = new_donor_h
            return True
        if donor["y"] < receiver["y"] and abs(donor_bottom - receiver["y"]) < 0.4:
            new_receiver_y = receiver["y"] - shift
            new_receiver_h = receiver["h"] + shift
            new_donor_h = donor["h"] - shift
            if new_donor_h <= 1.0:
                return False
            receiver["y"] = new_receiver_y
            receiver["h"] = new_receiver_h
            donor["h"] = new_donor_h
            return True
        return False

    def _rebalance_soft_template_lanes(
        self,
        template_layout: Dict[str, Any],
        section_containers: List[Dict[str, Any]],
        lane_map: Dict[str, Dict[str, Any]],
        state: PosterState,
    ) -> Dict[str, Dict[str, Any]]:
        if not (
            template_layout.get("extracted_template")
            and template_layout.get("geometry_policy") == "soft"
        ):
            return lane_map

        lanes = template_layout.get("lanes") or []
        is_vertical_stack = bool(lanes) and len({round(lane.get("x", 0), 3) for lane in lanes}) == 1
        if template_layout.get("orientation") != "portrait" and not is_vertical_stack:
            return lane_map

        ordered_lanes = sorted(lanes, key=lambda lane: lane.get("y", 0))
        if len(ordered_lanes) != 3:
            return lane_map

        body_top = min(lane["y"] for lane in ordered_lanes)
        body_bottom = max(lane["y"] + lane["h"] for lane in ordered_lanes)
        existing_gaps = []
        for index in range(len(ordered_lanes) - 1):
            gap = ordered_lanes[index + 1]["y"] - (ordered_lanes[index]["y"] + ordered_lanes[index]["h"])
            if gap > 0:
                existing_gaps.append(gap)
        lane_gap = min(existing_gaps) if existing_gaps else min(self.layout_config["column_spacing"], 0.8)
        available_h = max(body_bottom - body_top - lane_gap * (len(ordered_lanes) - 1), 0.1)

        pressure_params = {
            "section_gap": self.layout_config["section_spacing"],
            "title_to_content_gap": self.layout_config["title_to_content_spacing"],
            "visual_gap": self.layout_config["visual_spacing"]["below_visual"],
            "text_padding": self.layout_config["text_padding"]["left_right"],
            "body_font_reduction": 0,
            "title_font_reduction": 0,
            "body_font_boost": 0,
            "title_font_boost": 0,
            "visual_scale": 1.0,
        }
        pressure_by_lane = {}
        for lane in ordered_lanes:
            groups = [
                group
                for group in section_containers
                if group.get("lane_id") == lane["id"]
            ]
            measured_pressure = 0.0
            probe_lane = dict(lane)
            probe_lane["y"] = 0.0
            probe_lane["h"] = 1000.0
            for index, group in enumerate(groups):
                _, section_bottom = self._layout_section(group, probe_lane, measured_pressure, state, pressure_params, template_layout)
                measured_pressure = section_bottom
                if index < len(groups) - 1:
                    measured_pressure += self.layout_config["section_spacing"]
            fallback_pressure = sum(max(group["container"].get("height", 0.0), 0.25) for group in groups)
            pressure_by_lane[lane["id"]] = max(measured_pressure, fallback_pressure, lane["h"] * 0.25)

        total_pressure = sum(pressure_by_lane.values()) or 1.0
        raw_ratios = {lane_id: pressure / total_pressure for lane_id, pressure in pressure_by_lane.items()}
        min_ratio = 0.18
        remaining = max(1.0 - min_ratio * len(ordered_lanes), 0.01)
        excess_total = sum(max(raw_ratios[lane["id"]] - min_ratio, 0.0) for lane in ordered_lanes)
        if excess_total <= 0:
            ratios = {lane["id"]: 1.0 / len(ordered_lanes) for lane in ordered_lanes}
        else:
            ratios = {
                lane["id"]: min_ratio + remaining * max(raw_ratios[lane["id"]] - min_ratio, 0.0) / excess_total
                for lane in ordered_lanes
            }

        current_y = body_top
        updated_map = dict(lane_map)
        for lane in ordered_lanes:
            updated = dict(lane)
            updated["y"] = current_y
            updated["h"] = available_h * ratios[lane["id"]]
            updated["soft_rebalanced"] = True
            current_y += updated["h"] + lane_gap
            updated_map[lane["id"]] = updated

        template_layout["lanes"] = [updated_map[lane["id"]] for lane in lanes]
        template_layout["columns"] = template_layout["lanes"]
        return updated_map

    def _assign_section_id(self, element: Dict[str, Any], section_containers: List[Dict[str, Any]], lane_map: Dict[str, Dict[str, Any]]) -> Optional[str]:
        explicit_section_id = element.get("section_id")
        if explicit_section_id and any(group["section_id"] == explicit_section_id for group in section_containers):
            return explicit_section_id

        element_id = element.get("id", "")
        if element_id:
            matches = [
                group["section_id"]
                for group in section_containers
                if element_id.startswith(f"{group['section_id']}_")
            ]
            if matches:
                return max(matches, key=len)

        lane_id = element.get("lane_id") or self._match_lane_for_element(element, lane_map)
        lane_groups = [group for group in section_containers if group["lane_id"] == lane_id]
        lane_groups.sort(key=lambda group: group["container"]["y"])
        element_y = element.get("y", 0)

        for idx, group in enumerate(lane_groups):
            start_y = group["container"]["y"]
            next_start = lane_groups[idx + 1]["container"]["y"] if idx + 1 < len(lane_groups) else lane_map[lane_id]["y"] + lane_map[lane_id]["h"] + 10
            if start_y - 0.1 <= element_y < next_start:
                return group["section_id"]
        return lane_groups[-1]["section_id"] if lane_groups else None

    def _match_lane_for_element(self, element: Dict[str, Any], lane_map: Dict[str, Dict[str, Any]]) -> str:
        element_x = element.get("x", 0)
        element_y = element.get("y", 0)
        for lane_id, lane in lane_map.items():
            within_x = lane["x"] - 0.05 <= element_x <= lane["x"] + lane["w"] + 0.05
            within_y = lane["y"] - 0.05 <= element_y <= lane["y"] + lane["h"] + 20
            if within_x and within_y:
                return lane_id
        return next(iter(lane_map))

    def _refine_lane(self, groups: List[Dict[str, Any]], lane: Dict[str, Any], state: PosterState, template_layout: Dict[str, Any]) -> Dict[str, Any]:
        if not groups:
            return {
                "elements": [],
                "report": {
                    "lane_id": lane["id"],
                    "force_fit_used": False,
                    "iterations": 0,
                    "final_overflow": 0.0,
                },
            }

        params = {
            "section_gap": self.layout_config["section_spacing"],
            "title_to_content_gap": self.layout_config["title_to_content_spacing"],
            "visual_gap": self.layout_config["visual_spacing"]["below_visual"],
            "text_padding": self.layout_config["text_padding"]["left_right"],
            "body_font_reduction": 0,
            "title_font_reduction": 0,
            "body_font_boost": 0,
            "title_font_boost": 0,
            "visual_scale": 1.0,
        }
        if self._is_soft_portrait_template(template_layout):
            params.update({
                "section_gap": 0.45,
                "title_to_content_gap": 0.2,
                "visual_gap": 0.18,
                "text_padding": 0.24,
                "body_font_reduction": 8,
                "title_font_reduction": 10,
                "visual_scale": 0.82,
            })
        params["visual_scale"] = max(
            params["visual_scale"],
            self._visual_scale_floor(groups, state, template_layout),
        )

        best_layout = None
        best_overflow = float("inf")
        best_params = deepcopy(params)

        max_iterations = self.refine_config["max_iterations"]
        if state.get("template_fast_mode"):
            max_iterations = min(int(max_iterations), 4)
        iteration_count = 0

        for iteration in range(max_iterations):
            lane_layout, overflow = self._layout_lane(groups, lane, state, params, template_layout)
            iteration_count = iteration + 1

            if overflow < best_overflow:
                best_overflow = overflow
                best_layout = lane_layout
                best_params = deepcopy(params)

            if overflow <= 0.0:
                expanded_layout, expanded_report = self._expand_underfilled_lane(
                    groups,
                    lane,
                    state,
                    template_layout,
                    params,
                    lane_layout,
                    overflow,
                )
                return {
                    "elements": expanded_layout,
                    "report": {
                        "lane_id": lane["id"],
                        "force_fit_used": False,
                        "iterations": iteration_count,
                        "final_overflow": expanded_report["overflow"],
                        "final_utilization": expanded_report["utilization"],
                        "underflow_expanded": expanded_report["expanded"],
                        "params": expanded_report["params"],
                    },
                }

            params = self._tighten_params(params, groups, state, template_layout)

        if self._is_soft_portrait_template(template_layout) and best_overflow <= 0.5:
            return {
                "elements": best_layout or [],
                "report": {
                    "lane_id": lane["id"],
                    "force_fit_used": False,
                    "iterations": iteration_count,
                    "final_overflow": best_overflow,
                    "soft_overflow_tolerated": True,
                    "params": best_params,
                },
            }

        force_fit_layout = self._force_fit_lane(best_layout or [], lane, state, template_layout)
        final_overflow = self._lane_overflow(force_fit_layout, lane)
        lane_height = max(float(lane.get("h", 0.0) or 0.0), 0.01)
        return {
            "elements": force_fit_layout,
            "report": {
                "lane_id": lane["id"],
                "force_fit_used": True,
                "iterations": iteration_count,
                "pre_force_fit_overflow": best_overflow,
                "final_overflow": final_overflow,
                "final_utilization": (lane_height + final_overflow) / lane_height,
                "params": best_params,
            },
        }

    def _tighten_params(
        self,
        params: Dict[str, Any],
        groups: Optional[List[Dict[str, Any]]] = None,
        state: Optional[PosterState] = None,
        template_layout: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tightened = deepcopy(params)
        tightened["section_gap"] = max(
            self.refine_config["min_section_gap"],
            tightened["section_gap"] - self.refine_config["section_gap_step"],
        )
        tightened["title_to_content_gap"] = max(
            self.refine_config["min_title_to_content_gap"],
            tightened["title_to_content_gap"] - 0.05,
        )
        tightened["visual_gap"] = max(
            self.refine_config["min_visual_gap"],
            tightened["visual_gap"] - 0.04,
        )
        tightened["text_padding"] = max(
            self.refine_config["min_text_padding"],
            tightened["text_padding"] - 0.02,
        )
        tightened["body_font_reduction"] += self.refine_config["body_font_shrink_step"]
        tightened["title_font_reduction"] += self.refine_config["title_font_shrink_step"]
        min_visual_scale = self.refine_config["min_visual_scale"]
        if groups is not None and state is not None and template_layout is not None:
            min_visual_scale = max(
                min_visual_scale,
                self._visual_scale_floor(groups, state, template_layout),
            )
        tightened["visual_scale"] = max(
            min_visual_scale,
            tightened["visual_scale"] - self.refine_config["visual_scale_step"],
        )
        return tightened

    def _loosen_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        loosened = deepcopy(params)
        loosened["section_gap"] = min(
            self.refine_config.get("max_section_gap", self.layout_config["section_spacing"]),
            loosened["section_gap"] + self.refine_config.get("section_gap_expand_step", 0.12),
        )
        loosened["body_font_boost"] = min(
            self.refine_config.get("max_body_font_boost", 0),
            loosened.get("body_font_boost", 0) + self.refine_config.get("body_font_boost_step", 2),
        )
        loosened["title_font_boost"] = min(
            self.refine_config.get("max_section_title_font_boost", 0),
            loosened.get("title_font_boost", 0) + self.refine_config.get("title_font_boost_step", 1),
        )
        loosened["visual_scale"] = min(
            self.refine_config.get("max_visual_scale", 1.0),
            loosened["visual_scale"] + self.refine_config.get("visual_scale_step", 0.05),
        )
        return loosened

    def _expand_underfilled_lane(
        self,
        groups: List[Dict[str, Any]],
        lane: Dict[str, Any],
        state: PosterState,
        template_layout: Dict[str, Any],
        params: Dict[str, Any],
        lane_layout: List[Dict[str, Any]],
        overflow: float,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        lane_height = lane["h"]
        used_height = lane_height + overflow
        utilization = used_height / max(lane_height, 0.01)
        target_utilization = self.refine_config.get("target_lane_utilization", 0.9)
        free_space = -overflow

        best_layout = lane_layout
        best_params = deepcopy(params)
        best_overflow = overflow
        best_utilization = utilization

        if free_space < self.refine_config.get("underflow_expand_threshold", 1.0) or utilization >= target_utilization:
            return best_layout, {
                "expanded": False,
                "overflow": best_overflow,
                "utilization": best_utilization,
                "params": best_params,
            }

        trial_params = deepcopy(params)
        for _ in range(self.refine_config.get("max_underflow_iterations", 8)):
            trial_params = self._loosen_params(trial_params)
            candidate_layout, candidate_overflow = self._layout_lane(groups, lane, state, trial_params, template_layout)
            if candidate_overflow > 0.0:
                break

            candidate_used = lane_height + candidate_overflow
            candidate_utilization = candidate_used / max(lane_height, 0.01)
            if candidate_utilization > best_utilization:
                best_layout = candidate_layout
                best_params = deepcopy(trial_params)
                best_overflow = candidate_overflow
                best_utilization = candidate_utilization

            if candidate_utilization >= target_utilization:
                break

            if trial_params == self._loosen_params(trial_params):
                break

        return best_layout, {
            "expanded": best_layout is not lane_layout,
            "overflow": best_overflow,
            "utilization": best_utilization,
            "params": best_params,
        }

    def _layout_lane(self, groups: List[Dict[str, Any]], lane: Dict[str, Any], state: PosterState, params: Dict[str, Any], template_layout: Dict[str, Any]) -> tuple[List[Dict[str, Any]], float]:
        elements: List[Dict[str, Any]] = []
        current_y = lane["y"]

        for index, group in enumerate(groups):
            section_elements, section_bottom = self._layout_section(
                group,
                lane,
                current_y,
                state,
                params,
                template_layout,
                is_last_group=index == len(groups) - 1,
            )
            elements.extend(section_elements)
            current_y = section_bottom
            if index < len(groups) - 1:
                current_y += params["section_gap"]

        elements = self._stretch_table_visual_after_force_fit(
            self._sync_container_bounds(elements),
            lane,
        )
        elements = self._sync_container_bounds(elements)
        current_y = max(
            (float(element.get("y", 0.0) or 0.0) + float(element.get("height", 0.0) or 0.0))
            for element in elements
        )
        lane_bottom = lane["y"] + lane["h"]
        overflow = current_y - lane_bottom
        return elements, overflow

    def _layout_section(
        self,
        group: Dict[str, Any],
        lane: Dict[str, Any],
        section_y: float,
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        *,
        is_last_group: bool = True,
    ) -> tuple[List[Dict[str, Any]], float]:
        container = deepcopy(group["container"])
        children = [
            deepcopy(child)
            for child in group["children"]
            if not self._is_generated_bottom_fill(child)
        ]
        section_id = group["section_id"]

        title_elements = [child for child in children if child.get("type") in {"section_title", "title_accent_block", "title_accent_line"}]
        visual_elements = [child for child in children if child.get("type") == "visual"]
        text_elements = [child for child in children if child.get("type") == "text"]
        other_elements = [child for child in children if child.get("type") not in {"section_title", "title_accent_block", "title_accent_line", "visual", "text"}]

        original_section_y = container.get("y", section_y)
        current_y = section_y
        rebuilt_children: List[Dict[str, Any]] = []
        content_bottom = current_y

        section_title_element = next((child for child in title_elements if child.get("type") == "section_title"), None)
        if section_title_element:
            original_font_size = int(section_title_element.get("font_size", self.typography_config["sizes"]["section_title"]))
            title_font_size = max(
                self._min_section_title_font_size(template_layout),
                original_font_size - params["title_font_reduction"] + params.get("title_font_boost", 0),
            )
            title_font_size = min(
                self.refine_config.get("max_section_title_font_size", title_font_size),
                title_font_size,
            )
            # One uniform dark title-bar height for every section so the bars line up neatly.
            uniform_bar_height = float(
                (self.config.get("poster_visual_style", {}) or {})
                .get("section_title", {})
                .get("bar_height_inches", 0.78)
                or 0.78
            )
            title_scale = title_font_size / max(original_font_size, 1)
            title_x_offset = section_title_element.get("x", lane["x"]) - container.get("x", lane["x"])
            # Keep the title text inside the uniform bar so it never spills out of the dark block,
            # which also stops per-section title-font differences from changing the bar height.
            max_title_font_for_bar = int((uniform_bar_height - 0.06) * 72)
            if max_title_font_for_bar >= self._min_section_title_font_size(template_layout):
                title_font_size = min(title_font_size, max_title_font_for_bar)
            title_height = max((title_font_size / 72) + 0.05, section_title_element.get("height", 0.8) * title_scale)

            for child in title_elements:
                child_type = child.get("type")
                child_x_offset = child.get("x", lane["x"]) - container.get("x", lane["x"])
                child_y_offset = max(child.get("y", original_section_y) - original_section_y, 0.0) * title_scale

                if child_type == "section_title":
                    child["x"] = lane["x"] + child_x_offset
                    child["y"] = section_y + child_y_offset
                    child["height"] = title_height
                    child["width"] = max(lane["w"] - (child["x"] - lane["x"]) - params["text_padding"], 0.5)
                    child["font_size"] = title_font_size
                    child["alignment"] = "center"
                else:
                    child["x"] = lane["x"] + child_x_offset
                    child["y"] = section_y + child_y_offset
                    if child_type == "title_accent_block":
                        child["x"] = lane["x"]
                        child["width"] = lane["w"]
                        # Force one uniform dark-bar height for every section (instead of the
                        # per-section layout-assigned or font-scaled height) so all section
                        # title bars line up at the same height across the poster.
                        child["height"] = uniform_bar_height
                    elif child_type == "title_accent_line":
                        child["x"] = lane["x"]
                        child["width"] = lane["w"]
                        child["height"] = max(child.get("height", 0.3), 0.08)
                    else:
                        child["height"] = max(child.get("height", 0.3) * title_scale, 0.08)
                        child["width"] = max(child.get("width", 0.3) * title_scale, 0.08)
                rebuilt_children.append(child)
                content_bottom = max(content_bottom, child["y"] + child["height"])

            current_y = content_bottom + params["title_to_content_gap"]

        split_tail = self._layout_portrait_split_visual_text(
            visual_elements,
            text_elements,
            lane,
            current_y,
            state,
            params,
            template_layout,
            section_id,
        )
        if split_tail:
            tail_elements, current_y, content_bottom = split_tail
            rebuilt_children.extend(tail_elements)
            visual_elements = []
            text_elements = []
        else:
            visual_available_width = self._get_visual_width_for_lane(lane, state, template_layout, params)
            for visual in visual_elements:
                lane_for_footprint = self._lane_with_poster_orientation(lane, state, template_layout)
                aspect_ratio = visual.get("width", 1.0) / max(visual.get("height", 0.01), 0.01)
                visual_scale = params["visual_scale"]
                if str(visual.get("visual_id") or visual.get("id") or "").startswith("generated_teaser"):
                    visual_scale = max(visual_scale, 1.0)
                scaled_width = min(visual_available_width, visual.get("width", visual_available_width) * visual_scale)
                scaled_height = scaled_width / max(aspect_ratio, 0.01)
                scaled_width, scaled_height, footprint_report = enforce_visual_footprint(
                    visual.get("visual_id") or visual.get("id"),
                    scaled_width,
                    scaled_height,
                    visual_available_width,
                    lane_for_footprint,
                    state,
                    self.config,
                )

                visual["width"] = scaled_width
                visual["height"] = scaled_height
                visual["x"] = lane["x"] + (lane["w"] - scaled_width) / 2
                visual["y"] = current_y
                visual["visual_footprint"] = footprint_report

                rebuilt_children.append(visual)
                current_y = visual["y"] + visual["height"] + params["visual_gap"]
                content_bottom = max(content_bottom, visual["y"] + visual["height"])

        wide_text_columns = self._layout_wide_text_columns_for_fill(
            text_elements,
            lane,
            current_y,
            state,
            params,
            template_layout,
            section_id,
            is_last_group,
        )
        if wide_text_columns:
            column_elements, current_y, content_bottom = wide_text_columns
            rebuilt_children.extend(column_elements)
            text_elements = []

        for index, text_element in enumerate(text_elements):
            original_font_size = int(text_element.get("font_size", self.typography_config["sizes"]["body_text"]))
            font_size = max(
                self._min_body_font_size(template_layout),
                original_font_size - params["body_font_reduction"] + params.get("body_font_boost", 0),
            )
            font_size = min(self.refine_config.get("max_body_font_size", font_size), font_size)
            text_width = max(lane["w"] - 2 * params["text_padding"], 0.5)
            line_spacing = float(text_element.get("line_spacing", 1.0) or 1.0)
            if is_last_group and index == len(text_elements) - 1:
                target_bottom = (
                    float(lane["y"])
                    + float(lane["h"])
                    - self._real_content_fill_bottom_padding(template_layout)
                )
                max_text_height = max(target_bottom - current_y, 0.2)
                text_element, font_size, line_spacing = self._expand_text_content_to_fill(
                    text_element,
                    section_id,
                    state,
                    text_width,
                    font_size,
                    line_spacing,
                    max_text_height,
                    template_layout,
                )
            plain_text = self._strip_markup_for_measurement(text_element.get("content", ""))
            measured = self._measure_text_height_for_refinement(
                text_content=plain_text,
                width_inches=text_width,
                font_name=text_element.get("font_family", self.typography_config["fonts"]["body_text"]),
                font_size=font_size,
                line_spacing=line_spacing,
                template_layout=template_layout,
            )

            text_element["x"] = lane["x"] + params["text_padding"]
            text_element["y"] = current_y
            text_element["width"] = text_width
            text_element["height"] = (
                measured["optimal_height"] * self.refine_config.get("text_height_safety_factor", 1.0)
                + self.refine_config.get("text_height_safety_padding", 0.05)
                + self.refine_config.get("text_box_overflow_safety_inches", 0.0)
            )
            text_element["font_size"] = font_size
            text_element["line_spacing"] = line_spacing

            rebuilt_children.append(text_element)
            current_y = text_element["y"] + text_element["height"]
            content_bottom = max(content_bottom, text_element["y"] + text_element["height"])

        for other in other_elements:
            other["x"] = lane["x"] + (other.get("x", lane["x"]) - container.get("x", lane["x"]))
            other["y"] = section_y + max(other.get("y", original_section_y) - original_section_y, 0.0)
            rebuilt_children.append(other)
            content_bottom = max(content_bottom, other["y"] + other.get("height", 0))

        if is_last_group:
            rebuilt_children, content_bottom = self._portrait_expand_visual_to_absorb_bottom_gap(
                rebuilt_children,
                lane,
                params,
                template_layout,
                content_bottom,
            )
            bottom_fill = self._bottom_fill_elements(
                group,
                lane,
                state,
                params,
                template_layout,
                content_bottom,
            )
            for fill_element in bottom_fill:
                rebuilt_children.append(fill_element)
                content_bottom = max(content_bottom, fill_element["y"] + fill_element["height"])

        container["x"] = lane["x"]
        container["y"] = section_y
        container["width"] = lane["w"]
        container["height"] = max(
            content_bottom - section_y + self.refine_config.get("container_bottom_padding", 0.0),
            0.25,
        )

        return [container] + rebuilt_children, container["y"] + container["height"]

    def _is_generated_bottom_fill(self, element: Dict[str, Any]) -> bool:
        element_id = str(element.get("id") or "")
        return (
            element_id.endswith("_bottom_takeaway")
            or "_bottom_fill_" in element_id
            or element_id.endswith("_bottom_edge_fill")
        )

    def _layout_wide_text_columns_for_fill(
        self,
        text_elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        current_y: float,
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        section_id: str,
        is_last_group: bool,
    ) -> Optional[tuple[List[Dict[str, Any]], float, float]]:
        if not is_last_group or not self._real_content_fill_enabled(template_layout):
            return None
        if template_layout.get("layout_mode") != "template_prior":
            return None
        if not text_elements:
            return None

        lane_width = float(lane.get("w", 0.0) or 0.0)
        min_width = float(self.refine_config.get("wide_text_columns_min_width_inches", 24.0) or 24.0)
        if lane_width < min_width:
            return None

        lane_bottom = float(lane["y"]) + float(lane["h"])
        bottom_padding = self._real_content_fill_bottom_padding(template_layout)
        available_height = lane_bottom - bottom_padding - current_y
        if available_height < float(self.refine_config.get("wide_text_columns_min_height_inches", 2.2) or 2.2):
            return None

        original = deepcopy(text_elements[0])
        merged_content = "\n".join(
            str(text_element.get("content") or "").strip()
            for text_element in text_elements
            if str(text_element.get("content") or "").strip()
        )
        font_size = min(
            self.refine_config.get("max_body_font_size", int(original.get("font_size", self.typography_config["sizes"]["body_text"]))),
            max(
                self._min_body_font_size(template_layout),
                int(original.get("font_size", self.typography_config["sizes"]["body_text"])) - params["body_font_reduction"] + params.get("body_font_boost", 0),
            ),
        )
        line_spacing = float(original.get("line_spacing", 1.0) or 1.0)
        padding = float(params.get("text_padding", 0.3) or 0.3)
        gap = float(self.refine_config.get("wide_text_columns_gap_inches", 0.42) or 0.42)
        col_width = max((lane_width - 2 * padding - gap) / 2, 0.5)

        lines = self._content_lines_for_fill(
            section_id,
            state,
            merged_content,
            int(self.refine_config.get("real_content_fill_max_sentences", 16) or 16),
        )
        if len(lines) < 2:
            return None

        lines = self._pack_wide_column_lines_for_fill(
            lines,
            col_width,
            original,
            font_size,
            line_spacing,
            available_height,
            template_layout,
        )
        if len(lines) < 2:
            return None

        target_height = max(available_height - self._real_content_fill_target_slack(template_layout), 0.4)
        max_line_spacing = float(self.refine_config.get("real_content_fill_max_line_spacing", 1.08) or 1.08)
        max_font_boost = int(self.refine_config.get("real_content_fill_max_font_boost", 0) or 0)
        max_font_size = min(
            int(self.refine_config.get("max_body_font_size", font_size) or font_size),
            font_size + max(max_font_boost, 0),
        )
        height_tolerance = self._real_content_fill_height_tolerance(template_layout)

        spacing_values = [line_spacing]
        trial_spacing = line_spacing
        spacing_step = self._real_content_fill_spacing_step(template_layout)
        while trial_spacing + spacing_step <= max_line_spacing + 1e-9:
            trial_spacing = round(trial_spacing + spacing_step, 3)
            spacing_values.append(trial_spacing)

        best_payload: Optional[tuple[float, int, int, float, List[str], int, float, float]] = None
        fill_threshold = self._real_content_fill_threshold(template_layout)
        best_score: Optional[tuple[int, int, float, float, int, float]] = None
        for trial_font_size in range(font_size, max_font_size + 1):
            for trial_line_spacing in spacing_values:
                for end in range(2, len(lines) + 1):
                    trial_lines = lines[:end]
                    for split in range(1, len(trial_lines)):
                        left = "\n".join(trial_lines[:split])
                        right = "\n".join(trial_lines[split:])
                        left_h = self._text_height_for_width(left, col_width, original, trial_font_size, trial_line_spacing, template_layout)
                        right_h = self._text_height_for_width(right, col_width, original, trial_font_size, trial_line_spacing, template_layout)
                        max_height_used = max(left_h, right_h)
                        if max_height_used > available_height + height_tolerance:
                            continue
                        height_gap = abs(max_height_used - target_height)
                        balance = abs(left_h - right_h)
                        fills_target = max_height_used >= target_height - fill_threshold
                        score = (
                            0 if fills_target else 1,
                            -len(trial_lines),
                            height_gap,
                            balance,
                            -trial_font_size,
                            -trial_line_spacing,
                        )
                        if best_score is None or score < best_score:
                            best_score = score
                            best_payload = (
                                max_height_used,
                                trial_font_size,
                                split,
                                trial_line_spacing,
                                trial_lines,
                                len(trial_lines),
                                left_h,
                                right_h,
                            )
        if best_payload is None:
            return None
        _, font_size, best_split, line_spacing, best_lines, _, _, _ = best_payload
        left_lines = best_lines[:best_split]
        right_lines = best_lines[best_split:]

        elements: List[Dict[str, Any]] = []
        bottoms = []
        for index, column_lines in enumerate([left_lines, right_lines]):
            if not column_lines:
                continue
            item = deepcopy(original)
            content = "\n".join(column_lines)
            measured_height = self._text_height_for_width(content, col_width, item, font_size, line_spacing, template_layout)
            item.update(
                {
                    "id": f"{section_id}_text_col_{index + 1}",
                    "x": float(lane["x"]) + padding + index * (col_width + gap),
                    "y": current_y,
                    "width": col_width,
                    "height": min(measured_height, available_height),
                    "content": content,
                    "font_size": font_size,
                    "line_spacing": line_spacing,
                }
            )
            elements.append(item)
            bottoms.append(item["y"] + item["height"])
        if not elements:
            return None
        content_bottom = max(bottoms)
        return elements, content_bottom, content_bottom

    def _portrait_expand_visual_to_absorb_bottom_gap(
        self,
        elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        content_bottom: float,
    ) -> tuple[List[Dict[str, Any]], float]:
        if not self._real_content_fill_enabled(template_layout):
            return elements, content_bottom

        lane_bottom = float(lane["y"]) + float(lane["h"])
        bottom_gap = lane_bottom - content_bottom
        allowed_gap = self._final_bottom_whitespace_limit(lane)
        target_gap = min(allowed_gap, 0.08)
        if bottom_gap <= allowed_gap:
            return elements, content_bottom

        visuals = [
            element
            for element in elements
            if element.get("type") == "visual"
            and not str(element.get("visual_id") or element.get("id") or "").startswith("generated_teaser")
        ]
        if not visuals:
            return elements, content_bottom
        visual = max(
            visuals,
            key=lambda item: float(item.get("width", 0.0) or 0.0) * float(item.get("height", 0.0) or 0.0),
        )
        split_layout = str(visual.get("portrait_split_layout") or "")
        if split_layout in {"image_left_text_right", "text_left_image_right"}:
            return elements, content_bottom
        visual_width = float(visual.get("width", 0.0) or 0.0)
        visual_height = float(visual.get("height", 0.0) or 0.0)
        if visual_width <= 0 or visual_height <= 0:
            return elements, content_bottom

        aspect_ratio = visual_width / max(visual_height, 0.01)
        padding = max(float(params.get("text_padding", 0.24) or 0.24), 0.18)
        max_width = max(float(lane["w"]) - 2 * padding, visual_width)
        max_height = max_width / max(aspect_ratio, 0.01)
        grow_by = min(bottom_gap - target_gap, max_height - visual_height)
        stretch_height_only = False
        if grow_by <= 0.03 and self._can_stretch_visual_height_to_absorb_gap(visual):
            grow_by = bottom_gap - target_gap
            stretch_height_only = grow_by > 0.03
        if grow_by <= 0.03:
            return elements, content_bottom

        old_visual_bottom = float(visual.get("y", 0.0) or 0.0) + visual_height
        new_height = visual_height + grow_by
        new_width = visual_width if stretch_height_only else min(new_height * aspect_ratio, max_width)
        visual["width"] = new_width
        visual["height"] = new_height
        if not stretch_height_only:
            visual["x"] = float(lane["x"]) + (float(lane["w"]) - new_width) / 2

        for element in elements:
            if element is visual:
                continue
            element_y = float(element.get("y", 0.0) or 0.0)
            if element_y >= old_visual_bottom - 0.02:
                element["y"] = element_y + grow_by

        return elements, content_bottom + grow_by

    def _can_stretch_visual_height_to_absorb_gap(self, visual: Dict[str, Any]) -> bool:
        visual_id = str(visual.get("visual_id") or visual.get("id") or "")
        return visual_id.startswith("table_") or "_table_" in visual_id

    def _pack_wide_column_lines_for_fill(
        self,
        lines: List[str],
        col_width: float,
        text_element: Dict[str, Any],
        font_size: int,
        line_spacing: float,
        available_height: float,
        template_layout: Dict[str, Any],
    ) -> List[str]:
        """Keep dense portrait columns from getting blocked by one overlong candidate."""
        packed: List[str] = []
        for line in lines:
            trial = [*packed, line]
            if len(trial) < 2 or self._wide_column_lines_fit(
                trial,
                col_width,
                text_element,
                font_size,
                line_spacing,
                available_height,
                template_layout,
            ):
                packed = trial
        return packed if len(packed) >= 2 else lines

    def _wide_column_lines_fit(
        self,
        lines: List[str],
        col_width: float,
        text_element: Dict[str, Any],
        font_size: int,
        line_spacing: float,
        available_height: float,
        template_layout: Dict[str, Any],
    ) -> bool:
        height_tolerance = self._real_content_fill_height_tolerance(template_layout)
        for split in range(1, len(lines)):
            left = "\n".join(lines[:split])
            right = "\n".join(lines[split:])
            left_h = self._text_height_for_width(left, col_width, text_element, font_size, line_spacing, template_layout)
            right_h = self._text_height_for_width(right, col_width, text_element, font_size, line_spacing, template_layout)
            if max(left_h, right_h) <= available_height + height_tolerance:
                return True
        return False

    def _expand_text_content_to_fill(
        self,
        text_element: Dict[str, Any],
        section_id: str,
        state: PosterState,
        text_width: float,
        font_size: int,
        line_spacing: float,
        max_height: float,
        template_layout: Dict[str, Any],
    ) -> tuple[Dict[str, Any], int, float]:
        if not self._real_content_fill_enabled(template_layout):
            return text_element, font_size, line_spacing
        if max_height <= 0.2:
            return text_element, font_size, line_spacing

        target_slack = self._real_content_fill_target_slack(template_layout)
        target_height = max(max_height - target_slack, 0.2)
        threshold = self._real_content_fill_threshold(template_layout)
        height_tolerance = self._real_content_fill_height_tolerance(template_layout)
        content = self._clean_existing_fill_content(str(text_element.get("content") or ""))
        if content != str(text_element.get("content") or ""):
            text_element = deepcopy(text_element)
            text_element["content"] = content
        measured = self._text_height_for_width(content, text_width, text_element, font_size, line_spacing, template_layout)
        if target_height - measured < threshold:
            return text_element, font_size, line_spacing

        candidates = self._content_lines_for_fill(
            section_id,
            state,
            content,
            int(self.refine_config.get("real_content_fill_max_sentences", 16) or 16),
        )
        existing = {self._dedupe_text_key(line) for line in content.splitlines() if line.strip()}
        best_content = content
        best_measured = measured
        max_candidate_chars = int(self.refine_config.get("real_content_fill_max_chars", 190) or 190)
        min_candidate_chars = int(
            self.refine_config.get(
                "real_content_fill_min_candidate_chars",
                self.refine_config.get("portrait_split_text_min_candidate_chars", 42),
            )
            or 42
        )
        for candidate in candidates:
            key = self._dedupe_text_key(candidate)
            if not key or key in existing:
                continue

            best_trial_content = None
            best_trial_measured = best_measured
            for limit in self._split_candidate_char_limits(max_candidate_chars):
                trimmed = self._truncate_takeaway(str(candidate or "").strip(), limit)
                plain = self._strip_markup_for_measurement(trimmed).strip()
                trimmed_key = self._dedupe_text_key(trimmed)
                if len(plain) < min_candidate_chars or not trimmed_key or trimmed_key in existing:
                    continue
                if self._is_bad_fill_sentence(plain):
                    continue
                trial = (best_content.rstrip() + "\n" + trimmed).strip()
                trial_measured = self._text_height_for_width(
                    trial,
                    text_width,
                    text_element,
                    font_size,
                    line_spacing,
                    template_layout,
                )
                if trial_measured <= max_height + height_tolerance and trial_measured > best_trial_measured + 0.01:
                    best_trial_content = trial
                    best_trial_measured = trial_measured
                if target_height - best_trial_measured < threshold:
                    break

            if best_trial_content is not None:
                best_content = best_trial_content
                best_measured = best_trial_measured
                existing.add(self._dedupe_text_key(candidate))
                existing.add(self._dedupe_text_key(best_content.splitlines()[-1]))
            if target_height - best_measured < threshold:
                break

        max_line_spacing = float(self.refine_config.get("real_content_fill_max_line_spacing", 1.08) or 1.08)
        best_line_spacing = line_spacing
        if target_height - best_measured >= threshold:
            trial_spacing = line_spacing
            spacing_step = self._real_content_fill_spacing_step(template_layout)
            while trial_spacing + spacing_step <= max_line_spacing + 1e-9:
                trial_spacing = round(trial_spacing + spacing_step, 3)
                trial_measured = self._text_height_for_width(best_content, text_width, text_element, font_size, trial_spacing, template_layout)
                if trial_measured <= max_height + height_tolerance:
                    best_line_spacing = trial_spacing
                    best_measured = trial_measured
                if target_height - best_measured < threshold:
                    break

        best_font_size = font_size
        if target_height - best_measured >= threshold:
            max_font_boost = int(self.refine_config.get("real_content_fill_max_font_boost", 0) or 0)
            max_font_size = min(
                int(self.refine_config.get("max_body_font_size", font_size) or font_size),
                font_size + max(max_font_boost, 0),
            )
            for trial_font_size in range(font_size + 1, max_font_size + 1):
                trial_measured = self._text_height_for_width(
                    best_content,
                    text_width,
                    text_element,
                    trial_font_size,
                    best_line_spacing,
                    template_layout,
                )
                if trial_measured <= max_height + height_tolerance:
                    best_font_size = trial_font_size
                    best_measured = trial_measured
                if target_height - best_measured < threshold:
                    break

        updated = deepcopy(text_element)
        updated["content"] = best_content
        return updated, best_font_size, best_line_spacing

    def _content_lines_for_fill(
        self,
        section_id: str,
        state: PosterState,
        current_content: str,
        max_items: int,
    ) -> List[str]:
        current_lines = [
            line.strip()
            for line in self._clean_existing_fill_content(str(current_content or "")).splitlines()
            if line.strip()
        ]
        section = self._story_section_by_id(state, section_id)
        allow_new_lines = bool(self.refine_config.get("real_content_fill_allow_new_lines", False))
        teaser_fill = False
        if (
            not allow_new_lines
            and bool(self.refine_config.get("real_content_fill_allow_teaser_new_lines", True))
            and self._section_has_generated_teaser(section)
        ):
            allow_new_lines = True
            teaser_fill = True
        if not allow_new_lines:
            return current_lines

        max_chars = int(self.refine_config.get("real_content_fill_max_chars", 190) or 190)
        max_new_lines = max_items
        if teaser_fill:
            max_new_lines = int(self.refine_config.get("real_content_fill_teaser_max_new_lines", 1) or 1)
        candidates: List[str] = []
        candidates.extend(self._section_sentence_candidates(section))
        candidates.extend(self._source_sentence_candidates(section_id, state))

        existing = {self._dedupe_text_key(line) for line in current_lines}
        result = []
        seen = set(existing)
        for candidate in candidates:
            text = self._truncate_takeaway(str(candidate or "").strip(), max_chars)
            text = self._apply_fill_keyword_highlighting(text, section_id, state)
            key = self._dedupe_text_key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= max_new_lines:
                break
        return current_lines + result

    def _apply_fill_keyword_highlighting(self, text: str, section_id: str, state: PosterState) -> str:
        if not text or any(marker in text for marker in ("<color:", "**", "*")):
            return text
        section_keywords = ((state.get("keywords") or {}).get("section_keywords") or {})
        keywords = section_keywords.get(section_id) or section_keywords.get(section_id.removeprefix("sec_"))
        if not keywords:
            return text
        colors = state.get("color_scheme") or {}
        highlight_color = str(colors.get("contrast") or colors.get("theme") or "#1E3A8A")
        formatted = text
        for style_name, wrapper in (
            ("bold_contrast", lambda value: f"<color:{highlight_color}>{value}</color>"),
            ("bold", lambda value: f"**{value}**"),
            ("italic", lambda value: f"*{value}*"),
        ):
            for keyword in keywords.get(style_name) or []:
                formatted = self._highlight_fill_keyword(formatted, str(keyword or ""), wrapper)
        return formatted

    def _highlight_fill_keyword(self, text: str, keyword: str, wrapper) -> str:
        keyword = keyword.strip()
        if not keyword:
            return text
        pattern = rf"\b{re.escape(keyword)}\b"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if self._inside_existing_inline_markup(text, match.start()):
                continue
            matched = match.group(0)
            return text[: match.start()] + wrapper(matched) + text[match.end() :]
        return text

    def _inside_existing_inline_markup(self, text: str, index: int) -> bool:
        for match in re.finditer(r"<color:[^>]+>.*?</color>|\*\*.*?\*\*|\*[^*]+\*", text):
            if match.start() <= index < match.end():
                return True
        return False

    def _section_has_generated_teaser(self, section: Dict[str, Any]) -> bool:
        if bool(section.get("generated_teaser_summary")):
            return True
        for visual in section.get("visual_assets") or []:
            if str((visual or {}).get("visual_id") or "").startswith("generated_teaser"):
                return True
        return False

    def _source_sentence_candidates(self, section_id: str, state: PosterState) -> List[str]:
        section = self._story_section_by_id(state, section_id)
        title = str(section.get("section_title") or section_id)
        role = str(section.get("content_role") or "")
        query_terms = self._terms(f"{title} {role} {section_id}")
        scored = []
        for sentence in self._iter_source_sentences_for_fill(section, state):
            clean = re.sub(r"\s+", " ", sentence).strip(" -[]{}\"'")
            if len(clean) < 40 or self._is_bad_fill_sentence(clean):
                continue
            overlap = len(query_terms & self._terms(clean))
            source_match = self._source_section_match_bonus(section, clean)
            if overlap + source_match <= 0:
                continue
            scored.append((overlap + source_match + 1, len(clean), clean))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:24]]

    def _iter_source_sentences_for_fill(self, section: Dict[str, Any], state: PosterState) -> List[str]:
        texts: List[str] = []
        narrative = state.get("narrative_content") or {}
        if isinstance(narrative, dict):
            for key in ("poster_hook", "and", "but", "therefore", "key_impact"):
                value = narrative.get(key)
                if isinstance(value, str):
                    texts.append(value)

        structured = state.get("structured_sections") or {}
        paper_sections = structured.get("paper_sections") if isinstance(structured, dict) else []
        source_names = {
            str(name or "").strip().lower()
            for name in (section.get("source_sections") or [])
            if str(name or "").strip()
        }
        role = str(section.get("content_role") or "").lower()
        title_terms = self._terms(
            " ".join(
                [
                    str(section.get("section_title") or ""),
                    str(section.get("section_id") or ""),
                    role,
                ]
            )
        )
        for paper_section in paper_sections or []:
            if not isinstance(paper_section, dict):
                continue
            section_name = str(paper_section.get("section_name") or "").strip()
            section_type = str(paper_section.get("section_type") or "").strip().lower()
            section_terms = self._terms(f"{section_name} {section_type}")
            include = False
            if section_name.lower() in source_names:
                include = True
            elif role and role == section_type:
                include = True
            elif title_terms & section_terms:
                include = True
            elif role == "results" and section_type in {"results", "evaluation", "experiments"}:
                include = True
            elif role in {"method", "methods"} and section_type in {"method", "methods", "approach"}:
                include = True
            elif role in {"foundation", "motivation"} and section_type in {"foundation", "introduction"}:
                include = True
            if not include:
                continue
            content = paper_section.get("content")
            if isinstance(content, str):
                texts.append(content)
            for key_point in paper_section.get("key_points") or []:
                if isinstance(key_point, str):
                    texts.append(key_point)

        sentences: List[str] = []
        for text in texts:
            cleaned = re.sub(r"\s+", " ", self._strip_markup_for_measurement(str(text or ""))).strip()
            if not cleaned:
                continue
            for piece in re.split(r"(?<=[.!?])\s+", cleaned):
                piece = piece.strip(" -\t\n")
                if piece:
                    sentences.append(piece)
        return sentences

    def _source_section_match_bonus(self, section: Dict[str, Any], sentence: str) -> int:
        source_terms = self._terms(" ".join(str(name or "") for name in section.get("source_sections") or []))
        if not source_terms:
            return 0
        return 1 if source_terms & self._terms(sentence) else 0

    def _is_bad_fill_sentence(self, text: str) -> bool:
        if self._is_bad_existing_fill_sentence(text):
            return True
        cleaned = str(text or "").strip()
        if re.match(r"^[A-Z][A-Za-z ,.'’:-]{10,90}\.\s*$", cleaned) and len(self._terms(cleaned)) <= 8:
            return True
        return False

    def _is_bad_existing_fill_sentence(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        lowered = cleaned.lower()
        if re.match(r"^\d{4}\.\s+", cleaned):
            return True
        if re.search(r"\b(?:st|fig|eq|sec)\.\s*$", lowered):
            return True
        if re.search(r"\b(evicted:\s+poverty|using machine learning to help vulnerable tenants|legal representation on tenant outcomes)\b", lowered):
            return True
        if re.search(r"\b(perfo|handle tens|multimodal parcel)\.?$", lowered):
            return True
        if any(token in lowered for token in ("http://", "https://", "www.", "doi:", "arxiv", "isbn")):
            return True
        if re.search(r"\b(references|bibliography|proceedings|journal|journal of|conference on|transactions on)\b", lowered):
            return True
        if re.search(r"\b(in:|eds?\.|vol\.|pp\.)\b", lowered):
            return True
        return False

    def _clean_existing_fill_content(self, content: str) -> str:
        cleaned_lines: List[str] = []
        seen: set[str] = set()
        for raw_line in str(content or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            plain = self._strip_markup_for_measurement(line).strip()
            repaired_plain = repair_truncated_sentence_end(plain)
            if repaired_plain != plain:
                plain = repaired_plain.strip()
                line = repaired_plain.strip()
            if self._is_bad_existing_fill_sentence(plain):
                continue
            key = self._dedupe_text_key(plain)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _text_height_for_width(
        self,
        content: str,
        width: float,
        text_element: Dict[str, Any],
        font_size: int,
        line_spacing: float,
        template_layout: Dict[str, Any],
    ) -> float:
        measured = self._measure_text_height_for_refinement(
            text_content=self._strip_markup_for_measurement(content),
            width_inches=max(width, 0.5),
            font_name=text_element.get("font_family", self.typography_config["fonts"]["body_text"]),
            font_size=font_size,
            line_spacing=line_spacing,
            template_layout=template_layout,
        )
        return (
            float(measured["optimal_height"]) * self.refine_config.get("text_height_safety_factor", 1.0)
            + self.refine_config.get("text_height_safety_padding", 0.05)
            + self.refine_config.get("text_box_overflow_safety_inches", 0.0)
        )

    def _dedupe_text_key(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", self._strip_markup_for_measurement(str(text or "")).lower()).strip()

    def _terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text or "").lower())
            if term not in {"section", "result", "method", "using", "with", "from", "this", "that", "paper"}
        }

    def _real_content_fill_enabled(self, template_layout: Dict[str, Any]) -> bool:
        if not bool(self.refine_config.get("real_content_fill_enabled", True)):
            return False
        if not bool(self.refine_config.get("real_content_fill_portrait_only", False)):
            return True
        return str(template_layout.get("orientation") or "").lower() == "portrait"

    def _real_content_fill_target_slack(self, template_layout: Dict[str, Any]) -> float:
        default = float(self.refine_config.get("real_content_fill_target_slack_inches", 0.12) or 0.12)
        if str(template_layout.get("orientation") or "").lower() != "portrait":
            return default
        return float(
            self.refine_config.get(
                "portrait_real_content_fill_target_slack_inches",
                min(default, 0.04),
            )
            or min(default, 0.04)
        )

    def _real_content_fill_threshold(self, template_layout: Dict[str, Any]) -> float:
        default = float(self.refine_config.get("real_content_fill_threshold_inches", 0.16) or 0.16)
        if str(template_layout.get("orientation") or "").lower() != "portrait":
            return default
        return float(
            self.refine_config.get(
                "portrait_real_content_fill_threshold_inches",
                min(default, 0.04),
            )
            or min(default, 0.04)
        )

    def _real_content_fill_height_tolerance(self, template_layout: Dict[str, Any]) -> float:
        if str(template_layout.get("orientation") or "").lower() == "portrait":
            return 0.02
        return 0.04

    def _real_content_fill_spacing_step(self, template_layout: Dict[str, Any]) -> float:
        configured = self.refine_config.get("real_content_fill_spacing_step")
        if configured is not None:
            return float(configured or 0.01)
        if str(template_layout.get("orientation") or "").lower() == "portrait":
            return 0.01
        return 0.01

    def _real_content_fill_bottom_padding(self, template_layout: Dict[str, Any]) -> float:
        default = float(
            self.refine_config.get(
                "real_content_fill_bottom_padding_inches",
                self.refine_config.get("bottom_takeaway_bottom_padding_inches", 0.06),
            )
            or 0.06
        )
        if str(template_layout.get("orientation") or "").lower() != "portrait":
            return default
        return float(
            self.refine_config.get("portrait_real_content_fill_bottom_padding_inches", default)
            or default
        )

    def _final_bottom_whitespace_limit(self, lane: Dict[str, Any]) -> float:
        block_settings = self.config.get("block_refinement", {})
        max_inches = float(block_settings.get("final_max_bottom_whitespace_inches", 0.0) or 0.0)
        max_fraction = float(block_settings.get("final_max_bottom_whitespace_fraction", 0.0) or 0.0)
        lane_height = float(lane.get("h", 0.0) or 0.0)
        allowed_values = [
            value
            for value in (
                max_inches,
                lane_height * max_fraction if max_fraction > 0 else 0.0,
            )
            if value > 0
        ]
        if not allowed_values:
            return 0.12
        return min(allowed_values)

    def _bottom_fill_elements(
        self,
        group: Dict[str, Any],
        lane: Dict[str, Any],
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        content_bottom: float,
    ) -> List[Dict[str, Any]]:
        cfg = self.refine_config
        if not bool(cfg.get("bottom_takeaway_enabled", False)):
            return []
        if template_layout.get("layout_mode") != "template_prior":
            return []
        if (
            self._real_content_fill_enabled(template_layout)
            and str(template_layout.get("orientation") or "").lower() == "portrait"
            and bool(cfg.get("real_content_fill_disable_portrait_bottom_fill", True))
        ):
            return []

        section_id = str(group.get("section_id") or "")
        if not section_id:
            return []

        lane_bottom = float(lane["y"]) + float(lane["h"])
        threshold = float(cfg.get("bottom_takeaway_threshold_inches", 0.42) or 0.42)
        free_height = lane_bottom - content_bottom
        if free_height < threshold:
            return []

        large_gap_threshold = float(cfg.get("bottom_fill_large_gap_threshold_inches", 1.15) or 1.15)
        if free_height < large_gap_threshold:
            single = self._bottom_takeaway_element(group, lane, state, params, template_layout, content_bottom)
            if single:
                return [single]
            return []

        bottom_padding = float(cfg.get("bottom_takeaway_bottom_padding_inches", 0.06) or 0.06)
        gap = float(cfg.get("bottom_takeaway_gap_inches", 0.12) or 0.12)
        top = content_bottom + gap
        bottom = lane_bottom - bottom_padding
        available_height = bottom - top
        if available_height < 0.45:
            single = self._bottom_takeaway_element(group, lane, state, params, template_layout, content_bottom)
            return [single] if single else []

        padding = max(float(params.get("text_padding", 0.24) or 0.24), 0.18)
        usable_width = max(float(lane["w"]) - 2 * padding, 0.5)
        max_items = int(cfg.get("bottom_fill_max_items", 5) or 5)
        texts = self._bottom_fill_texts(section_id, state, max_items)
        if not texts:
            single = self._bottom_takeaway_element(group, lane, state, params, template_layout, content_bottom)
            return [single] if single else []

        font_size = int(cfg.get("bottom_fill_font_size", cfg.get("bottom_takeaway_font_size", 36)) or 36)
        line_spacing = float(cfg.get("bottom_fill_line_spacing", 0.9) or 0.9)
        fill_color = str(cfg.get("bottom_takeaway_font_color", "#4A1020"))
        font_family = self.typography_config["fonts"].get("body_text", "Arial")

        wide_min = float(cfg.get("bottom_fill_wide_min_width_inches", 24) or 24)
        if usable_width >= wide_min and len(texts) >= 2:
            col_gap = max(padding * 1.35, 0.35)
            col_width = max((usable_width - col_gap) / 2, 0.5)
            split_index = (len(texts) + 1) // 2
            columns = [texts[:split_index], texts[split_index:]]
            elements = []
            for index, column_texts in enumerate(columns):
                if not column_texts:
                    continue
                elements.extend(
                    self._distributed_bottom_text_elements(
                        section_id=section_id,
                        slot_id=str(lane.get("id") or group.get("lane_id") or ""),
                        id_prefix=f"{section_id}_bottom_fill_{index + 1}",
                        x=float(lane["x"]) + padding + index * (col_width + col_gap),
                        width=col_width,
                        top=top,
                        bottom=bottom,
                        texts=column_texts,
                        font_family=font_family,
                        font_size=font_size,
                        font_color=fill_color,
                        line_spacing=line_spacing,
                        template_layout=template_layout,
                    )
                )
            return elements

        item_gap = float(cfg.get("bottom_fill_item_gap_inches", 0.18) or 0.18)
        max_rows = max(1, min(max_items, len(texts)))
        item_height = float(cfg.get("bottom_fill_item_height_inches", 1.05) or 1.05)
        rows_that_fit = max(1, int((available_height + item_gap) // max(item_height + item_gap, 0.01)))
        row_count = max(1, min(max_rows, rows_that_fit))
        if row_count == 1:
            single = self._bottom_takeaway_element(group, lane, state, params, template_layout, content_bottom)
            return [single] if single else []

        elements = []
        elements.extend(
            self._distributed_bottom_text_elements(
                section_id=section_id,
                slot_id=str(lane.get("id") or group.get("lane_id") or ""),
                id_prefix=f"{section_id}_bottom_fill",
                x=float(lane["x"]) + padding,
                width=usable_width,
                top=top,
                bottom=bottom,
                texts=texts[:row_count],
                font_family=font_family,
                font_size=font_size,
                font_color=fill_color,
                line_spacing=line_spacing,
                template_layout=template_layout,
            )
        )
        return elements

    def _distributed_bottom_text_elements(
        self,
        *,
        section_id: str,
        slot_id: str,
        id_prefix: str,
        x: float,
        width: float,
        top: float,
        bottom: float,
        texts: List[str],
        font_family: str,
        font_size: int,
        font_color: str,
        line_spacing: float,
        template_layout: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        measured_heights = [
            min(
                max(
                    self._bottom_fill_text_height(text, width, font_family, font_size, line_spacing, template_layout),
                    0.42,
                ),
                1.45,
            )
            for text in texts
        ]
        elements = []
        count = len(texts)
        for index, (text, height) in enumerate(zip(texts, measured_heights)):
            if count == 1:
                y = bottom - height
            else:
                y = top + (bottom - top - height) * index / max(count - 1, 1)
            elements.append(
                {
                    "type": "text",
                    "id": f"{id_prefix}_{index + 1}",
                    "section_id": section_id,
                    "lane_id": slot_id,
                    "slot_id": slot_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "content": text,
                    "font_family": font_family,
                    "font_size": font_size,
                    "font_color": font_color,
                    "line_spacing": line_spacing,
                    "priority": 0.5,
                }
            )
        return elements

    def _bottom_fill_text_height(
        self,
        text: str,
        width: float,
        font_family: str,
        font_size: int,
        line_spacing: float,
        template_layout: Dict[str, Any],
    ) -> float:
        measured = self._measure_text_height_for_refinement(
            text_content=self._strip_markup_for_measurement(text),
            width_inches=max(width, 0.5),
            font_name=font_family,
            font_size=font_size,
            line_spacing=line_spacing,
            template_layout=template_layout,
        )
        return (
            float(measured["optimal_height"]) * self.refine_config.get("text_height_safety_factor", 1.0)
            + self.refine_config.get("text_height_safety_padding", 0.05)
            + self.refine_config.get("text_box_overflow_safety_inches", 0.0)
        )

    def _bottom_takeaway_element(
        self,
        group: Dict[str, Any],
        lane: Dict[str, Any],
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        content_bottom: float,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.refine_config
        if not bool(cfg.get("bottom_takeaway_enabled", False)):
            return None
        if template_layout.get("layout_mode") != "template_prior":
            return None

        section_id = str(group.get("section_id") or "")
        if not section_id:
            return None

        lane_bottom = float(lane["y"]) + float(lane["h"])
        threshold = float(cfg.get("bottom_takeaway_threshold_inches", 0.42) or 0.42)
        if lane_bottom - content_bottom < threshold:
            return None

        height = float(cfg.get("bottom_takeaway_height_inches", 0.52) or 0.52)
        bottom_padding = float(cfg.get("bottom_takeaway_bottom_padding_inches", 0.06) or 0.06)
        gap = float(cfg.get("bottom_takeaway_gap_inches", 0.12) or 0.12)
        y = lane_bottom - bottom_padding - height
        if y < content_bottom + gap:
            return None

        text = self._bottom_takeaway_text(section_id, state)
        if not text:
            return None

        padding = max(float(params.get("text_padding", 0.24) or 0.24), 0.18)
        slot_id = str(lane.get("id") or group.get("lane_id") or "")
        return {
            "type": "text",
            "id": f"{section_id}_bottom_takeaway",
            "section_id": section_id,
            "lane_id": slot_id,
            "slot_id": slot_id,
            "x": float(lane["x"]) + padding,
            "y": y,
            "width": max(float(lane["w"]) - 2 * padding, 0.5),
            "height": height,
            "content": text,
            "font_family": self.typography_config["fonts"].get("body_text", "Arial"),
            "font_size": int(cfg.get("bottom_takeaway_font_size", 36) or 36),
            "font_color": str(cfg.get("bottom_takeaway_font_color", "#4A1020")),
            "line_spacing": 0.85,
            "priority": 0.52,
        }

    def _bottom_takeaway_text(self, section_id: str, state: PosterState) -> str:
        max_chars = int(self.refine_config.get("bottom_takeaway_max_chars", 92) or 92)
        section = self._story_section_by_id(state, section_id)
        text = self._last_text_sentence(section)
        return self._truncate_takeaway(text, max_chars)

    def _bottom_fill_texts(self, section_id: str, state: PosterState, max_items: int) -> List[str]:
        section = self._story_section_by_id(state, section_id)
        candidates: List[str] = self._section_sentence_candidates(section)
        deduped = []
        seen = set()
        for candidate in candidates:
            text = self._truncate_takeaway(candidate, int(self.refine_config.get("bottom_takeaway_max_chars", 92) or 92))
            key = self._strip_markup_for_measurement(text).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
            if len(deduped) >= max_items:
                break
        return deduped

    def _section_sentence_candidates(self, section: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        for item in section.get("text_content") or []:
            text = self._strip_markup_for_measurement(str(item or "")).strip()
            if not text:
                continue
            for piece in re.split(r"(?<=[.!?])\s+", text):
                piece = piece.strip(" -\t\n")
                if len(piece) >= 28 and not self._is_bad_fill_sentence(piece):
                    candidates.append(piece)
        return candidates

    def _story_section_by_id(self, state: PosterState, section_id: str) -> Dict[str, Any]:
        sections = ((state.get("story_board") or {}).get("spatial_content_plan") or {}).get("sections") or []
        for section in sections:
            if str(section.get("section_id") or "") == section_id:
                return section
        return {}

    def _last_text_sentence(self, section: Dict[str, Any]) -> str:
        for item in reversed(section.get("text_content") or []):
            text = self._strip_markup_for_measurement(str(item or "")).strip()
            if not text:
                continue
            pieces = re.split(r"(?<=[.!?])\s+", text)
            for piece in reversed(pieces):
                piece = piece.strip(" -\t\n")
                if len(piece) >= 18:
                    return piece
        return "Takeaway: focus attention on the strongest poster claim."

    def _truncate_takeaway(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return fit_complete_sentence_prefix(text, max_chars)

    def _layout_portrait_split_visual_text(
        self,
        visual_elements: List[Dict[str, Any]],
        text_elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        current_y: float,
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        section_id: str,
    ) -> Optional[tuple[List[Dict[str, Any]], float, float]]:
        if not self._should_use_portrait_split_layout(visual_elements, text_elements, lane, state, template_layout):
            return None

        cfg = visual_footprint_config(self.config)
        padding = max(float(params.get("text_padding", 0.24)), 0.18)
        gap = float(cfg.get("portrait_split_gap_inches", 0.45) or 0.45)
        bottom_padding = float(cfg.get("portrait_split_bottom_padding_inches", 0.10) or 0.10)
        lane_bottom = float(lane["y"]) + float(lane["h"])
        available_height = max(lane_bottom - current_y - bottom_padding, 0.0)
        if available_height < float(cfg.get("portrait_split_min_height_inches", 4.8) or 4.8):
            return None

        usable_width = max(float(lane["w"]) - 2 * padding, 0.1)
        max_visual_width = min(
            usable_width * float(cfg.get("portrait_split_visual_width_fraction", 0.48) or 0.48),
            usable_width - gap - float(cfg.get("portrait_split_min_text_width_inches", 8.0) or 8.0),
        )
        if max_visual_width <= 0:
            return None

        visual = deepcopy(visual_elements[0])
        aspect_ratio = max(float(visual.get("width", 1.0)) / max(float(visual.get("height", 0.01)), 0.01), 0.2)
        target_width = min(max_visual_width, available_height * aspect_ratio)
        target_height = target_width / aspect_ratio
        lane_for_footprint = self._lane_with_poster_orientation(lane, state, template_layout)
        scaled_width, scaled_height, footprint_report = enforce_visual_footprint(
            visual.get("visual_id") or visual.get("id"),
            target_width,
            target_height,
            max_visual_width,
            lane_for_footprint,
            state,
            self.config,
        )
        if not footprint_report.get("ok"):
            return None

        text_width = usable_width - scaled_width - gap
        if text_width < float(cfg.get("portrait_split_min_text_width_inches", 8.0) or 8.0):
            return None

        allowed_bottom_gap = self._final_bottom_whitespace_limit(lane)
        target_margin = float(self.refine_config.get("portrait_split_bottom_target_margin_inches", 0.06) or 0.06)
        target_text_height = max(
            available_height - max(allowed_bottom_gap - bottom_padding - target_margin, 0.0),
            0.2,
        )
        expanded_text_elements = self._expand_portrait_split_text_to_fill(
            text_elements,
            section_id,
            state,
            text_width,
            available_height,
            params,
            template_layout,
            min(target_text_height, available_height),
        )

        laid_out_text = self._measure_split_text_elements(
            expanded_text_elements,
            text_width,
            available_height,
            params,
            template_layout,
            min(target_text_height, available_height),
        )
        if laid_out_text is None:
            return None

        measured_text, total_text_height = laid_out_text
        visual_on_left = self._split_visual_on_left(visual, lane)
        content_left = float(lane["x"]) + padding
        if visual_on_left:
            visual_x = content_left
            text_x = visual_x + scaled_width + gap
        else:
            text_x = content_left
            visual_x = text_x + text_width + gap

        vertical_alignment = str(cfg.get("portrait_split_vertical_alignment", "bottom") or "bottom").lower()
        if self._real_content_fill_enabled(template_layout):
            vertical_alignment = "top"
        if vertical_alignment == "center":
            visual_y = current_y + max((available_height - scaled_height) / 2, 0.0)
            text_y = current_y + max((available_height - total_text_height) / 2, 0.0)
        elif vertical_alignment == "top":
            visual_y = current_y + max((available_height - scaled_height) / 2, 0.0)
            text_y = current_y
        else:
            content_bottom = lane_bottom - bottom_padding
            visual_y = max(current_y, content_bottom - scaled_height)
            text_y = max(current_y, content_bottom - total_text_height)

        visual["x"] = visual_x
        visual["y"] = visual_y
        visual["width"] = scaled_width
        visual["height"] = scaled_height
        visual["visual_footprint"] = footprint_report
        visual["portrait_split_layout"] = "image_left_text_right" if visual_on_left else "text_left_image_right"

        y = text_y
        tail: List[Dict[str, Any]] = [visual]
        content_bottom = visual["y"] + visual["height"]
        for text_element, text_height, font_size, line_spacing in measured_text:
            item = deepcopy(text_element)
            item["x"] = text_x
            item["y"] = y
            item["width"] = text_width
            item["height"] = text_height
            item["font_size"] = font_size
            item["line_spacing"] = line_spacing
            item["portrait_split_layout"] = visual["portrait_split_layout"]
            tail.append(item)
            y += text_height
            content_bottom = max(content_bottom, item["y"] + item["height"])

        fill_element = self._portrait_split_bottom_fill_element(
            section_id,
            lane,
            state,
            params,
            template_layout,
            content_bottom,
        )
        if fill_element:
            tail.append(fill_element)
            content_bottom = max(content_bottom, fill_element["y"] + fill_element["height"])

        return tail, content_bottom, content_bottom

    def _expand_portrait_split_text_to_fill(
        self,
        text_elements: List[Dict[str, Any]],
        section_id: str,
        state: PosterState,
        text_width: float,
        available_height: float,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        target_height_override: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        expanded_text_elements = [deepcopy(text_element) for text_element in text_elements]
        if not expanded_text_elements or not self._real_content_fill_enabled(template_layout):
            return expanded_text_elements

        original = deepcopy(expanded_text_elements[0])
        merged_content = self._clean_existing_fill_content(
            "\n".join(
                str(text_element.get("content") or "").strip()
                for text_element in expanded_text_elements
                if str(text_element.get("content") or "").strip()
            )
        )
        if merged_content:
            original["content"] = merged_content

        base_size = int(original.get("font_size", self.typography_config["sizes"]["body_text"]))
        preferred_font_size = min(
            int(self.refine_config.get("max_body_font_size", base_size) or base_size),
            max(
                self._min_body_font_size(template_layout),
                base_size - int(params.get("body_font_reduction", 0)) + int(params.get("body_font_boost", 0)),
            ),
        )
        line_spacing = float(original.get("line_spacing", 1.0) or 1.0)
        working, font_size, line_spacing = self._expand_text_content_to_fill(
            original,
            section_id,
            state,
            text_width,
            preferred_font_size,
            line_spacing,
            max(available_height, 0.2),
            template_layout,
        )

        working = deepcopy(working)
        working["font_size"] = font_size
        working["line_spacing"] = line_spacing
        target_fraction = float(self.refine_config.get("portrait_split_text_min_fill_fraction", 0.92) or 0.92)
        target_fraction = min(max(target_fraction, 0.50), 0.98)
        target_height = max(available_height * target_fraction, 0.2)
        if target_height_override is not None:
            target_height = max(target_height, min(max(float(target_height_override), 0.2), available_height))
        threshold = self._real_content_fill_threshold(template_layout)
        height_tolerance = self._real_content_fill_height_tolerance(template_layout)
        measured = self._text_height_for_width(
            str(working.get("content") or ""),
            text_width,
            working,
            font_size,
            line_spacing,
            template_layout,
        )

        if target_height - measured >= threshold:
            working, measured = self._add_short_split_fill_lines(
                working,
                section_id,
                state,
                text_width,
                font_size,
                line_spacing,
                available_height,
                target_height,
                measured,
                height_tolerance,
                template_layout,
            )

        font_size, line_spacing, measured = self._inflate_split_text_style_to_target(
            working,
            text_width,
            font_size,
            line_spacing,
            available_height,
            target_height,
            measured,
            height_tolerance,
            template_layout,
        )
        working["font_size"] = font_size
        working["line_spacing"] = line_spacing
        working["portrait_split_text_fill_ratio"] = round(measured / max(available_height, 0.01), 4)
        return [working]

    def _portrait_split_bottom_fill_element(
        self,
        section_id: str,
        lane: Dict[str, Any],
        state: PosterState,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        content_bottom: float,
    ) -> Optional[Dict[str, Any]]:
        if not self._real_content_fill_enabled(template_layout):
            return None
        if not bool(self.refine_config.get("bottom_takeaway_enabled", False)):
            return None
        lane_bottom = float(lane["y"]) + float(lane["h"])
        allowed_gap = self._final_bottom_whitespace_limit(lane)
        if lane_bottom - content_bottom <= allowed_gap:
            return None

        cfg = self.refine_config
        bottom_padding = float(cfg.get("bottom_takeaway_bottom_padding_inches", 0.06) or 0.06)
        gap = min(float(cfg.get("bottom_takeaway_gap_inches", 0.04) or 0.04), 0.04)
        y = content_bottom + gap
        bottom = lane_bottom - bottom_padding
        available_height = bottom - y
        if available_height < 0.50:
            return None

        text = self._bottom_takeaway_text(section_id, state)
        if not text:
            return None

        padding = max(float(params.get("text_padding", 0.24) or 0.24), 0.18)
        width = max(float(lane["w"]) - 2 * padding, 0.5)
        font_family = self.typography_config["fonts"].get("body_text", "Arial")
        line_spacing = 0.85
        max_font_size = min(int(cfg.get("bottom_takeaway_font_size", 36) or 36), 32)
        min_font_size = max(18, self._min_body_font_size(template_layout) - 12)
        chosen_font_size = min_font_size
        measured_height = available_height
        for font_size in range(max_font_size, min_font_size - 1, -2):
            trial_height = self._bottom_fill_text_height(
                text,
                width,
                font_family,
                font_size,
                line_spacing,
                template_layout,
            )
            if trial_height <= available_height + 0.02:
                chosen_font_size = font_size
                measured_height = min(trial_height, available_height)
                break

        slot_id = str(lane.get("id") or "")
        return {
            "type": "text",
            "id": f"{section_id}_portrait_split_bottom_takeaway",
            "section_id": section_id,
            "lane_id": slot_id,
            "slot_id": slot_id,
            "x": float(lane["x"]) + padding,
            "y": bottom - measured_height,
            "width": width,
            "height": measured_height,
            "content": text,
            "font_family": font_family,
            "font_size": chosen_font_size,
            "font_color": str(cfg.get("bottom_takeaway_font_color", "#4A1020")),
            "line_spacing": line_spacing,
            "priority": 0.5,
            "portrait_split_layout": "bottom_takeaway",
        }

    def _add_short_split_fill_lines(
        self,
        text_element: Dict[str, Any],
        section_id: str,
        state: PosterState,
        text_width: float,
        font_size: int,
        line_spacing: float,
        available_height: float,
        target_height: float,
        measured: float,
        height_tolerance: float,
        template_layout: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float]:
        max_items = int(
            self.refine_config.get(
                "portrait_split_text_max_fill_sentences",
                self.refine_config.get("real_content_fill_max_sentences", 16),
            )
            or 16
        )
        max_chars = int(
            self.refine_config.get(
                "portrait_split_text_max_candidate_chars",
                self.refine_config.get("real_content_fill_max_chars", 190),
            )
            or 190
        )
        min_chars = int(self.refine_config.get("portrait_split_text_min_candidate_chars", 42) or 42)
        threshold = self._real_content_fill_threshold(template_layout)
        existing = {
            self._dedupe_text_key(line)
            for line in str(text_element.get("content") or "").splitlines()
            if line.strip()
        }
        content = str(text_element.get("content") or "").strip()
        added = 0

        candidates = self._content_lines_for_fill(section_id, state, content, max_items)
        for candidate in candidates:
            if target_height - measured < threshold:
                break
            key = self._dedupe_text_key(candidate)
            if not key or key in existing:
                continue

            best_content = None
            best_height = measured
            for limit in self._split_candidate_char_limits(max_chars):
                trimmed = self._truncate_takeaway(candidate, limit)
                plain = self._strip_markup_for_measurement(trimmed).strip()
                if len(plain) < min_chars or self._is_bad_fill_sentence(plain):
                    continue
                trimmed_key = self._dedupe_text_key(trimmed)
                if not trimmed_key or trimmed_key in existing:
                    continue
                trial = (content.rstrip() + "\n" + trimmed).strip() if content else trimmed
                trial_height = self._text_height_for_width(
                    trial,
                    text_width,
                    text_element,
                    font_size,
                    line_spacing,
                    template_layout,
                )
                if trial_height <= available_height + height_tolerance and trial_height > best_height + 0.01:
                    best_content = trial
                    best_height = trial_height

            if best_content is None:
                continue
            content = best_content
            measured = best_height
            existing.add(self._dedupe_text_key(candidate))
            added += 1
            if added >= max_items:
                break

        updated = deepcopy(text_element)
        updated["content"] = content
        return updated, measured

    def _split_candidate_char_limits(self, max_chars: int) -> List[int]:
        limits = [
            max_chars,
            170,
            145,
            125,
            105,
            92,
            78,
            64,
            52,
        ]
        result: List[int] = []
        for value in limits:
            limit = max(1, min(int(value), max_chars))
            if limit not in result:
                result.append(limit)
        return result

    def _inflate_split_text_style_to_target(
        self,
        text_element: Dict[str, Any],
        text_width: float,
        font_size: int,
        line_spacing: float,
        available_height: float,
        target_height: float,
        measured: float,
        height_tolerance: float,
        template_layout: Dict[str, Any],
    ) -> tuple[int, float, float]:
        threshold = self._real_content_fill_threshold(template_layout)
        max_line_spacing = float(self.refine_config.get("real_content_fill_max_line_spacing", 1.08) or 1.08)
        spacing_step = self._real_content_fill_spacing_step(template_layout)
        best_spacing = line_spacing
        trial_spacing = line_spacing
        while target_height - measured >= threshold and trial_spacing + spacing_step <= max_line_spacing + 1e-9:
            trial_spacing = round(trial_spacing + spacing_step, 3)
            trial_measured = self._text_height_for_width(
                str(text_element.get("content") or ""),
                text_width,
                text_element,
                font_size,
                trial_spacing,
                template_layout,
            )
            if trial_measured <= available_height + height_tolerance:
                best_spacing = trial_spacing
                measured = trial_measured

        best_font_size = font_size
        max_font_boost = int(self.refine_config.get("real_content_fill_max_font_boost", 0) or 0)
        max_font_size = min(
            int(self.refine_config.get("max_body_font_size", font_size) or font_size),
            font_size + max(max_font_boost, 0),
        )
        for trial_font_size in range(font_size + 1, max_font_size + 1):
            if target_height - measured < threshold:
                break
            trial_measured = self._text_height_for_width(
                str(text_element.get("content") or ""),
                text_width,
                text_element,
                trial_font_size,
                best_spacing,
                template_layout,
            )
            if trial_measured <= available_height + height_tolerance:
                best_font_size = trial_font_size
                measured = trial_measured
        return best_font_size, best_spacing, measured

    def _should_use_portrait_split_layout(
        self,
        visual_elements: List[Dict[str, Any]],
        text_elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        state: PosterState,
        template_layout: Dict[str, Any],
    ) -> bool:
        if not visual_elements or not text_elements or len(visual_elements) != 1:
            return False
        if self._poster_orientation(state, template_layout) != "portrait":
            return False
        visual_id = str(visual_elements[0].get("visual_id") or visual_elements[0].get("id") or "")
        if visual_id.startswith("table_"):
            return False
        if visual_id.startswith("generated_teaser"):
            return False

        cfg = visual_footprint_config(self.config)
        width = float(lane.get("w", 0.0) or 0.0)
        height = float(lane.get("h", 0.0) or 0.0)
        if width < float(cfg.get("portrait_split_min_width_inches", 18.0) or 18.0):
            return False
        if height < float(cfg.get("portrait_split_min_height_inches", 4.8) or 4.8):
            return False
        return width / max(height, 0.01) >= float(cfg.get("portrait_split_min_aspect", 2.35) or 2.35)

    def _measure_split_text_elements(
        self,
        text_elements: List[Dict[str, Any]],
        text_width: float,
        available_height: float,
        params: Dict[str, Any],
        template_layout: Dict[str, Any],
        target_total_height: Optional[float] = None,
    ) -> Optional[tuple[List[tuple[Dict[str, Any], float, int, float]], float]]:
        base_size = int(text_elements[0].get("font_size", self.typography_config["sizes"]["body_text"]))
        preferred = max(
            self._min_body_font_size(template_layout),
            base_size - int(params.get("body_font_reduction", 0)) + int(params.get("body_font_boost", 0)),
        )
        min_size = self._min_body_font_size(template_layout)
        base_line_spacing = max(float(text_element.get("line_spacing", 1.0) or 1.0) for text_element in text_elements)
        max_line_spacing = float(self.refine_config.get("real_content_fill_max_line_spacing", 1.08) or 1.08)
        spacing_step = self._real_content_fill_spacing_step(template_layout)
        spacing_values = [base_line_spacing]
        trial_spacing = base_line_spacing
        while trial_spacing + spacing_step <= max_line_spacing + 1e-9:
            trial_spacing = round(trial_spacing + spacing_step, 3)
            spacing_values.append(trial_spacing)

        height_tolerance = self._real_content_fill_height_tolerance(template_layout)
        target_height = min(float(target_total_height or available_height), available_height)
        best: Optional[tuple[tuple[float, float, int], List[tuple[Dict[str, Any], float, int, float]], float]] = None

        for font_size in range(preferred, min_size - 1, -2):
            for line_spacing in spacing_values:
                measured: List[tuple[Dict[str, Any], float, int, float]] = []
                total_height = 0.0
                for text_element in text_elements:
                    plain_text = self._strip_markup_for_measurement(text_element.get("content", ""))
                    result = self._measure_text_height_for_refinement(
                        text_content=plain_text,
                        width_inches=text_width,
                        font_name=text_element.get("font_family", self.typography_config["fonts"]["body_text"]),
                        font_size=font_size,
                        line_spacing=line_spacing,
                        template_layout=template_layout,
                    )
                    text_height = (
                        result["optimal_height"] * self.refine_config.get("text_height_safety_factor", 1.0)
                        + self.refine_config.get("text_height_safety_padding", 0.05)
                    )
                    measured.append((text_element, text_height, font_size, line_spacing))
                    total_height += text_height
                if total_height > available_height + height_tolerance:
                    continue
                score = (abs(target_height - total_height), -font_size, -line_spacing)
                if best is None or score < best[0]:
                    best = (score, measured, total_height)

        if best is not None:
            return best[1], best[2]

        for font_size in range(preferred, min_size - 1, -2):
            measured = []
            total_height = 0.0
            for text_element in text_elements:
                plain_text = self._strip_markup_for_measurement(text_element.get("content", ""))
                result = self._measure_text_height_for_refinement(
                    text_content=plain_text,
                    width_inches=text_width,
                    font_name=text_element.get("font_family", self.typography_config["fonts"]["body_text"]),
                    font_size=font_size,
                    line_spacing=text_element.get("line_spacing", 1.0),
                    template_layout=template_layout,
                )
                text_height = (
                    result["optimal_height"] * self.refine_config.get("text_height_safety_factor", 1.0)
                    + self.refine_config.get("text_height_safety_padding", 0.05)
                )
                measured.append((text_element, text_height, font_size, base_line_spacing))
                total_height += text_height
            if total_height <= available_height + 0.05:
                return measured, total_height

        return None

    def _split_visual_on_left(self, visual: Dict[str, Any], lane: Dict[str, Any]) -> bool:
        visual_center = float(visual.get("x", lane.get("x", 0.0))) + float(visual.get("width", 0.0)) / 2
        lane_center = float(lane.get("x", 0.0)) + float(lane.get("w", 0.0)) / 2
        if abs(visual_center - lane_center) < 0.2:
            return True
        return visual_center <= lane_center

    def _lane_with_poster_orientation(
        self,
        lane: Dict[str, Any],
        state: PosterState,
        template_layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        lane_for_footprint = dict(lane)
        lane_for_footprint.setdefault("poster_orientation", self._poster_orientation(state, template_layout))
        return lane_for_footprint

    def _poster_orientation(self, state: PosterState, template_layout: Dict[str, Any]) -> str:
        orientation = str(template_layout.get("orientation") or "").lower()
        if orientation:
            return orientation
        return (
            "portrait"
            if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
            else "landscape"
        )

    def _measure_text_height_for_refinement(
        self,
        text_content: str,
        width_inches: float,
        font_name: str,
        font_size: int,
        line_spacing: float,
        template_layout: Dict[str, Any],
    ) -> Dict[str, float]:
        if (
            bool(self.refine_config.get("use_fast_text_height_measurement", True))
            or template_layout.get("extracted_template")
            or template_layout.get("orientation") == "portrait"
        ):
            return {
                "optimal_height": self._estimate_text_height_fast(
                    text_content,
                    width_inches,
                    font_size,
                    line_spacing,
                    template_layout,
                )
            }
        return measure_text_height(
            text_content=text_content,
            width_inches=width_inches,
            font_name=font_name,
            font_size=font_size,
            line_spacing=line_spacing,
        )

    def _estimate_text_height_fast(
        self,
        text_content: str,
        width_inches: float,
        font_size: int,
        line_spacing: float,
        template_layout: Optional[Dict[str, Any]] = None,
    ) -> float:
        chars_per_inch = self._chars_per_inch_for_template(template_layout)
        chars_per_line = max(int(width_inches * chars_per_inch * (44 / max(font_size, 1))), 18)
        line_count = 0
        for raw_line in text_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_count += max(1, (len(line) + chars_per_line - 1) // chars_per_line)
        if line_count == 0:
            return 0.2
        line_height = (font_size / 72) * max(line_spacing, 0.9) * 1.15
        return line_count * line_height + max(line_count - 1, 0) * 0.04

    def _chars_per_inch_for_template(self, template_layout: Optional[Dict[str, Any]] = None) -> float:
        default = float(self.refine_config.get("ppt_chars_per_inch_at_44pt", 4.2))
        if template_layout and str(template_layout.get("orientation") or "").lower() == "portrait":
            return float(self.refine_config.get("portrait_ppt_chars_per_inch_at_44pt", default))
        return default

    def _get_visual_width_for_lane(self, lane: Dict[str, Any], state: PosterState, template_layout: Dict[str, Any], params: Dict[str, Any]) -> float:
        visual_width = lane["w"] - 2 * params["text_padding"]
        visual_width_cap = template_layout.get("visual_width_cap")
        if visual_width_cap:
            visual_width = min(visual_width, visual_width_cap * params["visual_scale"])
        return max(visual_width, 0.4)

    def _visual_scale_floor(
        self,
        groups: List[Dict[str, Any]],
        state: PosterState,
        template_layout: Dict[str, Any],
    ) -> float:
        if not state.get("template_fast_mode"):
            return float(self.refine_config.get("min_visual_scale", 0.72))
        if not any(
            child.get("type") == "visual"
            for group in groups
            for child in group.get("children", [])
        ):
            return float(self.refine_config.get("min_visual_scale", 0.72))
        cfg = visual_footprint_config(self.config)
        floor = float(cfg.get("min_visual_scale_in_visual_blocks", 0.95) or 0.95)
        has_key_visual = any(
            int((group.get("container") or {}).get("importance_level") or 2) <= 1
            for group in groups
            if any(child.get("type") == "visual" for child in group.get("children", []))
        )
        if has_key_visual:
            floor = max(floor, float(cfg.get("key_visual_min_scale", 1.0) or 1.0))
        if self._is_soft_portrait_template(template_layout):
            floor = min(floor, 0.95)
        return max(float(self.refine_config.get("min_visual_scale", 0.72)), floor)

    def _is_soft_portrait_template(self, template_layout: Dict[str, Any]) -> bool:
        lanes = template_layout.get("lanes") or []
        is_vertical_stack = bool(lanes) and len({round(lane.get("x", 0), 3) for lane in lanes}) == 1
        return (
            template_layout.get("extracted_template")
            and template_layout.get("geometry_policy") == "soft"
            and (template_layout.get("orientation") == "portrait" or is_vertical_stack)
        )

    def _min_body_font_size(self, template_layout: Dict[str, Any]) -> int:
        if self._is_soft_portrait_template(template_layout):
            return 24
        return self.refine_config["min_body_font_size"]

    def _min_section_title_font_size(self, template_layout: Dict[str, Any]) -> int:
        if self._is_soft_portrait_template(template_layout):
            return 32
        return self.refine_config["min_section_title_font_size"]

    def _strip_markup_for_measurement(self, content: str) -> str:
        text = re.sub(r"<color:[^>]+>", "", content)
        text = text.replace("</color>", "")
        text = text.replace("**", "")
        text = text.replace("*", "")
        return text

    def _force_fit_lane(self, lane_layout: List[Dict[str, Any]], lane: Dict[str, Any], state: PosterState, template_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not lane_layout:
            return lane_layout

        max_bottom = max(element.get("y", 0) + element.get("height", 0) for element in lane_layout)
        lane_bottom = lane["y"] + lane["h"]
        used_height = max_bottom - lane["y"]
        if used_height <= 0:
            return lane_layout

        compression_ratio = min(1.0, lane["h"] / used_height)
        if compression_ratio >= 0.999:
            return lane_layout

        compressed = []
        for element in lane_layout:
            item = deepcopy(element)
            relative_y = item.get("y", lane["y"]) - lane["y"]
            item["y"] = lane["y"] + relative_y * compression_ratio
            # Section title bars must keep one uniform height across every lane, so they
            # are never compressed here. The uncompressed excess is reclaimed by pushing
            # content below the bar and shrinking visuals in _settle_force_fit_lane; only a
            # genuinely un-fittable lane compresses its bars (uniformly) as a last resort.
            if item.get("type") not in {"title_accent_block", "title_accent_line"}:
                item["height"] = max(item.get("height", 0.0) * compression_ratio, 0.05)

            if item.get("type") == "visual":
                original_width = item.get("width", 0.5)
                new_width = max(original_width * compression_ratio, 0.25)
                center_x = lane["x"] + lane["w"] / 2
                item["width"] = new_width
                item["x"] = center_x - new_width / 2
                lane_for_footprint = dict(lane)
                lane_for_footprint.setdefault(
                    "poster_orientation",
                    template_layout.get("orientation")
                    or (
                        "portrait"
                        if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
                        else "landscape"
                    ),
                )
                max_visual_width = max(lane["w"] - 2 * self.refine_config.get("min_text_padding", 0.18), 0.4)
                protected_width, protected_height, footprint_report = enforce_visual_footprint(
                    item.get("visual_id") or item.get("id"),
                    item["width"],
                    item["height"],
                    max_visual_width,
                    lane_for_footprint,
                    state,
                    self.config,
                )
                item["width"] = protected_width
                item["height"] = protected_height
                item["x"] = center_x - protected_width / 2
                item["visual_footprint"] = footprint_report
            elif item.get("type") in {"title_accent_block", "title_accent_line"}:
                item["x"] = lane["x"]
                item["width"] = lane["w"]
            elif item.get("type") == "text":
                item["font_size"] = max(
                    self._min_body_font_size(template_layout),
                    int(round(item.get("font_size", 44) * compression_ratio)),
                )
                item["height"] = max(
                    item.get("height", 0.0),
                    self._measured_text_box_height(item, template_layout),
                )
            elif item.get("type") == "section_title":
                item["font_size"] = max(
                    self._min_section_title_font_size(template_layout),
                    int(round(item.get("font_size", 64) * compression_ratio)),
                )
                item["height"] = max(
                    item.get("height", 0.0),
                    (item["font_size"] / 72) + 0.05,
                )

            compressed.append(item)
        compressed = self._push_content_below_title_bars(compressed)
        compressed = self._sync_container_bounds(compressed)
        compressed = self._stretch_table_visual_after_force_fit(compressed, lane)
        compressed = self._sync_container_bounds(compressed)
        return self._settle_force_fit_lane(compressed, lane, state, template_layout)

    def _push_content_below_title_bars(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the uniform-height title bar clear: shift a section's content down so it
        starts below the bar, and overlay the title text on the bar. Any resulting overflow
        is absorbed later by shrinking visuals (and, as a last resort, uniformly compressing
        the bars) in _settle_force_fit_lane."""
        gap = max(float(self.refine_config.get("min_title_to_content_gap", 0.15) or 0.15), 0.06)
        for bar in [e for e in elements if e.get("type") == "title_accent_block"]:
            section_id = str(bar.get("section_id") or "")
            bar_y = float(bar.get("y", 0.0) or 0.0)
            bar_height = float(bar.get("height", 0.0) or 0.0)
            threshold = bar_y + bar_height + gap
            content = [
                element
                for element in elements
                if element.get("type") in {"visual", "text"}
                and str(element.get("section_id") or "") == section_id
            ]
            if content:
                topmost = min(float(element.get("y", 0.0) or 0.0) for element in content)
                if topmost < threshold - 1e-6:
                    shift = threshold - topmost
                    for element in content:
                        element["y"] = float(element.get("y", 0.0) or 0.0) + shift
            # keep the section title text overlaid on (and centered in) its uniform bar
            for element in elements:
                if element.get("type") == "section_title" and str(element.get("section_id") or "") == section_id:
                    element["y"] = bar_y
                    element["height"] = bar_height
        return elements

    def _settle_force_fit_lane(
        self,
        elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        state: PosterState,
        template_layout: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        settled = self._sync_container_bounds(elements, allow_shrink=True)
        overflow_tolerance = 0.02
        for _ in range(4):
            overflow = self._lane_overflow(settled, lane)
            if overflow <= overflow_tolerance:
                return settled
            if not self._shrink_force_fit_visuals_to_absorb_overflow(
                settled,
                lane,
                state,
                template_layout,
                overflow + overflow_tolerance,
            ):
                break
            settled = self._sync_container_bounds(settled, allow_shrink=True)
        # Last resort: a lane so over-full that shrinking visuals still cannot make room.
        # Rather than overflow (which the deterministic quality gate hard-rejects), compress
        # the title bars — but by a single shared ratio so every bar in this lane stays the
        # same height as the others. This only triggers on pathologically dense lanes.
        if self._lane_overflow(settled, lane) > overflow_tolerance:
            settled = self._compress_bars_uniformly_to_fit(settled, lane, overflow_tolerance)
            settled = self._sync_container_bounds(settled, allow_shrink=True)
        return settled

    def _compress_bars_uniformly_to_fit(
        self,
        elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        tolerance: float,
    ) -> List[Dict[str, Any]]:
        overflow = self._lane_overflow(elements, lane)
        if overflow <= tolerance:
            return elements
        accent_blocks = [e for e in elements if e.get("type") == "title_accent_block"]
        total_bar_height = sum(float(e.get("height", 0.0) or 0.0) for e in accent_blocks)
        if not accent_blocks or total_bar_height <= 0.0:
            return elements
        # Shared multiplicative ratio so every bar shrinks by the same proportion (keeping
        # them equal within the lane); never shrink below 50% of the uniform height.
        remove = min(overflow + tolerance, total_bar_height * 0.5)
        ratio = max((total_bar_height - remove) / total_bar_height, 0.5)
        for bar in sorted(accent_blocks, key=lambda e: float(e.get("y", 0.0) or 0.0)):
            old_height = float(bar.get("height", 0.0) or 0.0)
            new_height = max(old_height * ratio, 0.05)
            delta = old_height - new_height
            if delta <= 0.0:
                continue
            old_bottom = float(bar.get("y", 0.0) or 0.0) + old_height
            bar["height"] = new_height
            section_id = str(bar.get("section_id") or "")
            for element in elements:
                if element is bar:
                    continue
                # keep the overlaid title text inside the (now shorter) bar
                if element.get("type") == "section_title" and str(element.get("section_id") or "") == section_id:
                    element["height"] = min(float(element.get("height", 0.0) or 0.0), new_height)
                element_y = float(element.get("y", 0.0) or 0.0)
                if element_y >= old_bottom - 0.02:
                    element["y"] = element_y - delta
        return elements

    def _lane_overflow(self, elements: List[Dict[str, Any]], lane: Dict[str, Any]) -> float:
        if not elements:
            return 0.0
        lane_bottom = float(lane["y"]) + float(lane["h"])
        max_bottom = max(
            float(element.get("y", 0.0) or 0.0) + float(element.get("height", 0.0) or 0.0)
            for element in elements
        )
        return max_bottom - lane_bottom

    def _shrink_force_fit_visuals_to_absorb_overflow(
        self,
        elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
        state: PosterState,
        template_layout: Dict[str, Any],
        needed: float,
    ) -> bool:
        remaining = max(float(needed), 0.0)
        if remaining <= 0.0:
            return False

        candidates = [
            element
            for element in elements
            if element.get("type") == "visual"
            and self._force_fit_visual_shrink_headroom(element) > 0.02
        ]
        candidates.sort(
            key=lambda item: (
                float(item.get("y", 0.0) or 0.0),
                -self._force_fit_visual_shrink_headroom(item),
            )
        )

        changed = False
        for visual in candidates:
            if remaining <= 0.0:
                break
            headroom = self._force_fit_visual_shrink_headroom(visual)
            shrink_by = min(headroom, remaining)
            if shrink_by <= 0.0:
                continue

            old_height = float(visual.get("height", 0.0) or 0.0)
            old_width = float(visual.get("width", 0.0) or 0.0)
            if old_height <= 0.0 or old_width <= 0.0:
                continue

            aspect = max(old_width / old_height, 0.2)
            old_bottom = float(visual.get("y", 0.0) or 0.0) + old_height
            target_height = max(old_height - shrink_by, 0.05)
            target_width = target_height * aspect
            lane_for_footprint = dict(lane)
            lane_for_footprint.setdefault(
                "poster_orientation",
                template_layout.get("orientation")
                or (
                    "portrait"
                    if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
                    else "landscape"
                ),
            )
            max_visual_width = max(lane["w"] - 2 * self.refine_config.get("min_text_padding", 0.18), 0.4)
            protected_width, protected_height, footprint_report = enforce_visual_footprint(
                visual.get("visual_id") or visual.get("id"),
                target_width,
                target_height,
                max_visual_width,
                lane_for_footprint,
                state,
                self.config,
            )
            actual_shrink = old_height - protected_height
            if actual_shrink <= 0.01:
                continue

            center_x = float(lane.get("x", 0.0) or 0.0) + float(lane.get("w", 0.0) or 0.0) / 2
            visual["width"] = protected_width
            visual["height"] = protected_height
            visual["x"] = center_x - protected_width / 2
            visual["visual_footprint"] = footprint_report

            for element in elements:
                if element is visual:
                    continue
                element_y = float(element.get("y", 0.0) or 0.0)
                if element_y >= old_bottom - 0.02:
                    element["y"] = element_y - actual_shrink

            remaining -= actual_shrink
            changed = True

        return changed

    def _force_fit_visual_shrink_headroom(self, visual: Dict[str, Any]) -> float:
        width = float(visual.get("width", 0.0) or 0.0)
        height = float(visual.get("height", 0.0) or 0.0)
        if width <= 0.0 or height <= 0.0:
            return 0.0
        aspect = max(width / height, 0.2)
        footprint = visual.get("visual_footprint") or {}
        min_width = float(footprint.get("min_width", 0.0) or 0.0)
        min_height = float(footprint.get("min_height", 0.0) or 0.0)
        min_area = float(footprint.get("min_area", 0.0) or 0.0)
        required_width = max(
            min_width,
            min_height * aspect,
            (min_area * aspect) ** 0.5 if min_area > 0 else 0.0,
        )
        required_height = max(min_height, required_width / aspect if required_width > 0 else 0.0)
        return max(height - required_height, 0.0)

    def _stretch_table_visual_after_force_fit(
        self,
        elements: List[Dict[str, Any]],
        lane: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        lane_bottom = float(lane["y"]) + float(lane["h"])
        content_elements = [
            element
            for element in elements
            if element.get("type") in {"section_title", "title_accent_block", "text", "visual"}
        ]
        if not content_elements:
            return elements
        content_bottom = max(
            float(element.get("y", 0.0) or 0.0) + float(element.get("height", 0.0) or 0.0)
            for element in content_elements
        )
        allowed_gap = self._final_bottom_whitespace_limit(lane)
        target_gap = min(allowed_gap, 0.08)
        bottom_gap = lane_bottom - content_bottom
        if bottom_gap <= allowed_gap:
            return elements
        candidates = [
            element
            for element in elements
            if element.get("type") == "visual" and self._can_stretch_visual_height_to_absorb_gap(element)
        ]
        if not candidates:
            return elements

        visual = max(
            candidates,
            key=lambda item: float(item.get("width", 0.0) or 0.0) * float(item.get("height", 0.0) or 0.0),
        )
        grow_by = max(bottom_gap - target_gap, 0.0)
        if grow_by <= 0.03:
            return elements

        old_visual_bottom = float(visual.get("y", 0.0) or 0.0) + float(visual.get("height", 0.0) or 0.0)
        visual["height"] = float(visual.get("height", 0.0) or 0.0) + grow_by
        for element in elements:
            if element is visual:
                continue
            element_y = float(element.get("y", 0.0) or 0.0)
            if element_y >= old_visual_bottom - 0.02:
                element["y"] = element_y + grow_by
        return elements

    def _measured_text_box_height(self, item: Dict[str, Any], template_layout: Dict[str, Any]) -> float:
        plain_text = self._strip_markup_for_measurement(str(item.get("content") or ""))
        measured = self._measure_text_height_for_refinement(
            text_content=plain_text,
            width_inches=max(float(item.get("width", 0.0) or 0.0), 0.5),
            font_name=item.get("font_family", self.typography_config["fonts"]["body_text"]),
            font_size=int(item.get("font_size") or self.typography_config["sizes"]["body_text"]),
            line_spacing=float(item.get("line_spacing", 1.0) or 1.0),
            template_layout=template_layout,
        )
        return (
            float(measured["optimal_height"]) * self.refine_config.get("text_height_safety_factor", 1.0)
            + self.refine_config.get("text_height_safety_padding", 0.05)
        )

    def _sync_container_bounds(self, elements: List[Dict[str, Any]], *, allow_shrink: bool = False) -> List[Dict[str, Any]]:
        containers = {
            str(element.get("section_id")): element
            for element in elements
            if element.get("type") == "section_container" and element.get("section_id")
        }
        if not containers:
            return elements

        required_by_section = {section_id: 0.0 for section_id in containers}
        for element in elements:
            if element.get("type") == "section_container":
                continue
            section_id = str(element.get("section_id") or "")
            parent = containers.get(section_id)
            if not parent:
                element_id = str(element.get("id") or element.get("slot_id") or "")
                matches = [candidate for candidate in containers if element_id.startswith(f"{candidate}_")]
                parent = containers.get(max(matches, key=len)) if matches else None
            if not parent:
                continue
            child_bottom = float(element.get("y", 0.0) or 0.0) + float(element.get("height", 0.0) or 0.0)
            required = child_bottom - float(parent.get("y", 0.0) or 0.0) + self.refine_config.get("container_bottom_padding", 0.0)
            if allow_shrink:
                required_by_section[str(parent.get("section_id"))] = max(
                    required_by_section.get(str(parent.get("section_id")), 0.0),
                    required,
                )
            else:
                parent["height"] = max(float(parent.get("height", 0.0) or 0.0), required)

        if allow_shrink:
            for section_id, parent in containers.items():
                if required_by_section.get(section_id, 0.0) > 0.0:
                    parent["height"] = max(required_by_section[section_id], 0.05)
        return elements

    def _validate_refined_layout(self, elements: List[Dict[str, Any]], lane_map: Dict[str, Dict[str, Any]], state: PosterState) -> Dict[str, Any]:
        issues = []
        slide_width = state["poster_width"]
        slide_height = state["poster_height"]

        section_containers = [element for element in elements if element.get("type") == "section_container"]
        container_by_section = {
            section.get("section_id"): section
            for section in section_containers
            if section.get("section_id")
        }
        for element in elements:
            x = element.get("x", 0)
            y = element.get("y", 0)
            width = element.get("width", 0)
            height = element.get("height", 0)
            if x < 0 or y < 0 or x + width > slide_width + 1e-6 or y + height > slide_height + 1e-6:
                issues.append(f"element overflow: {element.get('type')} {element.get('id', element.get('section_id', 'unknown'))}")

            parent = self._find_parent_container(element, container_by_section)
            if parent and element.get("type") != "section_container":
                tolerance = 0.03
                parent_right = parent.get("x", 0) + parent.get("width", 0)
                parent_bottom = parent.get("y", 0) + parent.get("height", 0)
                if x < parent.get("x", 0) - tolerance or x + width > parent_right + tolerance:
                    issues.append(f"child horizontal overflow in section {parent.get('section_id')}: {element.get('id', element.get('type'))}")
                if y < parent.get("y", 0) - tolerance or y + height > parent_bottom + tolerance:
                    issues.append(f"child vertical overflow in section {parent.get('section_id')}: {element.get('id', element.get('type'))}")
                if element.get("type") == "text":
                    min_bottom_padding = float(
                        self.refine_config.get("min_text_container_bottom_padding_inches", 0.0) or 0.0
                    )
                    bottom_gap = parent_bottom - (y + height)
                    padding_tolerance = min(tolerance, max(min_bottom_padding * 0.5, 0.005))
                    if min_bottom_padding > 0 and bottom_gap < min_bottom_padding - padding_tolerance:
                        issues.append(
                            "text bottom padding too small in section "
                            f"{parent.get('section_id')}: {element.get('id', element.get('type'))}"
                        )

                    required_height = self._measured_text_box_height(
                        element,
                        state.get("layout_template_metadata") or {},
                    )
                    if required_height > height + tolerance:
                        issues.append(
                            "text box overflow risk in section "
                            f"{parent.get('section_id')}: {element.get('id', element.get('type'))}"
                        )

        for lane_id, lane in lane_map.items():
            lane_sections = [section for section in section_containers if section.get("lane_id") == lane_id]
            lane_sections.sort(key=lambda item: item.get("y", 0))
            previous_bottom = lane["y"]
            lane_tolerance = 0.5 if self._is_soft_portrait_template(state.get("layout_template_metadata") or {}) else 0.02
            for section in lane_sections:
                if section["y"] < previous_bottom - 0.02:
                    issues.append(f"section overlap in lane {lane_id}: {section.get('section_id')}")
                if section["y"] + section["height"] > lane["y"] + lane["h"] + lane_tolerance:
                    issues.append(f"lane overflow in {lane_id}: {section.get('section_id')}")
                previous_bottom = max(previous_bottom, section["y"] + section["height"])

        fixed_template = (state.get("layout_template_metadata") or {}).get("layout_mode") == "template_prior"
        if fixed_template:
            for index, left in enumerate(section_containers):
                for right in section_containers[index + 1:]:
                    if self._section_boxes_overlap(left, right):
                        issues.append(
                            "section container overlap: "
                            f"{left.get('section_id')} and {right.get('section_id')}"
                        )

        return {"issues": issues}

    def _section_boxes_overlap(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        tolerance = 0.02
        left_x = float(left.get("x", 0.0) or 0.0)
        left_y = float(left.get("y", 0.0) or 0.0)
        left_right = left_x + float(left.get("width", 0.0) or 0.0)
        left_bottom = left_y + float(left.get("height", 0.0) or 0.0)
        right_x = float(right.get("x", 0.0) or 0.0)
        right_y = float(right.get("y", 0.0) or 0.0)
        right_right = right_x + float(right.get("width", 0.0) or 0.0)
        right_bottom = right_y + float(right.get("height", 0.0) or 0.0)
        return not (
            left_right <= right_x + tolerance
            or right_right <= left_x + tolerance
            or left_bottom <= right_y + tolerance
            or right_bottom <= left_y + tolerance
        )

    def _find_parent_container(self, element: Dict[str, Any], container_by_section: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        section_id = element.get("section_id")
        if section_id in container_by_section:
            return container_by_section[section_id]

        element_id = str(element.get("id") or element.get("slot_id") or "")
        matches = [
            section_id
            for section_id in container_by_section
            if element_id.startswith(f"{section_id}_")
        ]
        if matches:
            return container_by_section[max(matches, key=len)]
        return None

    def _save_outputs(self, state: PosterState, report: Dict[str, Any]):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "styled_layout.json", "w", encoding="utf-8") as f:
            json.dump(state.get("styled_layout", []), f, indent=2)

        with open(output_dir / "micro_layout_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


def micro_layout_refiner_node(state: PosterState) -> Dict[str, Any]:
    result = MicroLayoutRefiner()(state)
    return {
        **state,
        "styled_layout": result.get("styled_layout"),
        "slot_pressure_report": result.get("slot_pressure_report"),
        "draft_status": result.get("draft_status", state.get("draft_status", "pending")),
        "final_poster_accepted": result.get("final_poster_accepted", False),
        "draft_rejection_reason": result.get("draft_rejection_reason"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
