"""
Block-level occupancy analysis for template-prior posters.

This agent is deterministic: it measures how much of each template block is
occupied after draft rendering and estimates how many characters are needed to
move underfilled blocks toward the target utilization.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success


class BlockOccupancyAnalyzer:
    def __init__(self):
        self.name = "block_occupancy_analyzer"
        self.config = load_config()
        self.block_config = self.config.get("block_refinement", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_block_vlm_review", False):
            return state

        log_agent_info(self.name, "measuring block occupancy toward configured utilization target")

        try:
            report = self.analyze(state)
            state["block_occupancy_report"] = report
            state["current_agent"] = self.name
            self._save_outputs(state, report)
            expand_count = sum(1 for block in report["blocks"] if block["action"] == "expand")
            reduce_count = sum(1 for block in report["blocks"] if block["action"] == "reduce")
            log_agent_success(
                self.name,
                f"analyzed {len(report['blocks'])} block(s): expand={expand_count}, reduce={reduce_count}",
            )
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def analyze(self, state: PosterState) -> Dict[str, Any]:
        layout = state.get("styled_layout") or []
        template = state.get("layout_template_metadata") or {}
        lanes = template.get("lanes") or []
        if not lanes:
            raise ValueError("layout_template_metadata.lanes is required for block occupancy analysis")
        self._analysis_orientation = str(template.get("orientation") or "").lower()

        lane_map = {str(lane.get("id")): lane for lane in lanes if lane.get("id")}
        sections = self._story_sections_by_id(state.get("story_board") or {})
        containers = self._section_containers(layout)
        children_by_section = self._children_by_section(layout, containers)

        blocks = []
        for container in containers:
            section_id = str(container.get("section_id") or "")
            if not section_id:
                continue
            lane_id = str(container.get("lane_id") or container.get("slot_id") or "")
            if lane_id not in lane_map and container.get("slot_id") in lane_map:
                lane_id = str(container["slot_id"])
            lane = lane_map.get(lane_id) or self._match_lane(container, lane_map)
            if not lane:
                continue

            section = sections.get(section_id, {})
            block = self._analyze_block(container, lane, children_by_section.get(section_id, []), section)
            blocks.append(block)

        blocks.sort(key=lambda block: (float(block["bbox"]["y"]), float(block["bbox"]["x"])))
        settings = self._settings()
        summary = {
            "block_count": len(blocks),
            "target_utilization": settings["target_utilization"],
            "acceptable_range": [settings["acceptable_min"], settings["acceptable_max"]],
            "hard_max": settings["hard_max"],
            "expand_count": sum(1 for block in blocks if block["action"] == "expand"),
            "reduce_count": sum(1 for block in blocks if block["action"] == "reduce"),
            "mean_utilization": round(
                sum(float(block["utilization"]) for block in blocks) / max(len(blocks), 1),
                4,
            ),
        }
        return {
            "source": "deterministic",
            "poster_preview_path": state.get("poster_preview_path"),
            "settings": settings,
            "summary": summary,
            "blocks": blocks,
        }

    def _analyze_block(
        self,
        container: Dict[str, Any],
        lane: Dict[str, Any],
        children: List[Dict[str, Any]],
        section: Dict[str, Any],
    ) -> Dict[str, Any]:
        settings = self._settings()
        available_height = max(float(lane.get("h", container.get("height", 0.0)) or 0.0), 0.1)
        available_width = max(float(lane.get("w", container.get("width", 0.0)) or 0.0), 0.1)
        text_items = [child for child in children if child.get("type") == "text"]
        visual_items = [child for child in children if child.get("type") == "visual"]
        title_items = [child for child in children if child.get("type") == "section_title"]
        lane_top = float(lane.get("y", container.get("y", 0.0)) or 0.0)
        lane_bottom = lane_top + available_height
        content_bounds = self._content_bounds(children, lane, container)
        if content_bounds:
            used_height = max(float(content_bounds["bottom"]) - lane_top, 0.0)
            visible_content_height = max(float(content_bounds["bottom"]) - float(content_bounds["top"]), 0.0)
            bottom_whitespace = max(lane_bottom - float(content_bounds["bottom"]), 0.0)
        else:
            used_height = max(float(container.get("height", 0.0) or 0.0), 0.0)
            visible_content_height = used_height
            bottom_whitespace = max(lane_bottom - (float(container.get("y", lane_top) or lane_top) + used_height), 0.0)
        utilization = used_height / available_height

        representative_text = self._representative_text_item(text_items, lane)
        font_size = int(representative_text.get("font_size") or self.config["typography"]["sizes"]["body_text"])
        line_spacing = float(representative_text.get("line_spacing") or self.config["typography"].get("line_spacing", 1.0))
        text_width = max(float(representative_text.get("width") or (available_width - 0.6)), 0.5)
        line_height = self._line_height(font_size, line_spacing)
        chars_per_line = self._chars_per_line(text_width, font_size)

        target_used_height = available_height * settings["target_utilization"]
        missing_height = max(target_used_height - used_height, 0.0)
        extra_lines = max(int(math.floor(missing_height / max(line_height, 0.01))), 0)
        if extra_lines == 0 and missing_height >= settings["min_missing_height_for_expand"]:
            extra_lines = 1
        target_extra_chars = int(extra_lines * chars_per_line * settings["safety_factor"])
        target_extra_chars = min(target_extra_chars, settings["max_extra_chars_per_block"])
        if target_extra_chars < settings["min_extra_chars"]:
            target_extra_chars = 0
        text_box_slack = self._text_box_slack(text_items)

        if utilization > settings["hard_max"]:
            action = "reduce"
            reason = f"utilization {utilization:.2f} exceeds hard max {settings['hard_max']:.2f}"
        elif utilization < settings["target_utilization"] and target_extra_chars > 0:
            action = "expand"
            reason = f"utilization {utilization:.2f} below target {settings['target_utilization']:.2f}"
        else:
            action = "keep"
            reason = "within target tolerance or no safe extra line budget"

        current_text = "\n".join(str(item.get("content") or "") for item in text_items)
        return {
            "slot_id": str(container.get("slot_id") or lane.get("id") or container.get("lane_id") or ""),
            "lane_id": str(lane.get("id") or container.get("lane_id") or ""),
            "section_id": str(container.get("section_id") or ""),
            "section_title": section.get("section_title") or self._title_from_children(title_items),
            "available_height": round(available_height, 4),
            "available_width": round(available_width, 4),
            "used_height": round(used_height, 4),
            "visible_content_height": round(visible_content_height, 4),
            "bottom_whitespace": round(bottom_whitespace, 4),
            "utilization": round(utilization, 4),
            "target_used_height": round(target_used_height, 4),
            "missing_height": round(missing_height, 4),
            "line_height": round(line_height, 4),
            "chars_per_line": chars_per_line,
            "target_extra_lines": extra_lines,
            "target_extra_chars": target_extra_chars,
            "current_text_chars": len(current_text),
            "text_item_count": len(text_items),
            "max_text_box_slack_ratio": text_box_slack["max_ratio"],
            "max_text_box_slack_inches": text_box_slack["max_inches"],
            "visual_count": len(visual_items),
            "action": action,
            "reason": reason,
            "bbox": {
                "x": round(float(lane.get("x", container.get("x", 0.0)) or 0.0), 4),
                "y": round(float(lane.get("y", container.get("y", 0.0)) or 0.0), 4),
                "w": round(float(lane.get("w", container.get("width", 0.0)) or 0.0), 4),
                "h": round(float(lane.get("h", container.get("height", 0.0)) or 0.0), 4),
            },
            "container_bbox": {
                "x": round(float(container.get("x", 0.0) or 0.0), 4),
                "y": round(float(container.get("y", 0.0) or 0.0), 4),
                "w": round(float(container.get("width", 0.0) or 0.0), 4),
                "h": round(float(container.get("height", 0.0) or 0.0), 4),
            },
        }

    def _content_bounds(
        self,
        children: List[Dict[str, Any]],
        lane: Dict[str, Any],
        container: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        lane_x = float(lane.get("x", container.get("x", 0.0)) or 0.0)
        lane_y = float(lane.get("y", container.get("y", 0.0)) or 0.0)
        lane_w = float(lane.get("w", container.get("width", 0.0)) or 0.0)
        lane_h = float(lane.get("h", container.get("height", 0.0)) or 0.0)
        lane_right = lane_x + lane_w
        lane_bottom = lane_y + lane_h

        boxes = []
        for child in children:
            child_type = child.get("type")
            if child_type not in {"section_title", "title_accent_block", "text", "visual"}:
                continue
            x = float(child.get("x", lane_x) or lane_x)
            y = float(child.get("y", lane_y) or lane_y)
            w = float(child.get("width", child.get("w", 0.0)) or 0.0)
            h = float(child.get("height", child.get("h", 0.0)) or 0.0)
            if w <= 0 or h <= 0:
                continue
            if child_type == "text":
                h = self._visible_text_height(child, h)
            right = x + w
            bottom = y + h
            if right < lane_x - 0.2 or x > lane_right + 0.2 or bottom < lane_y - 0.2 or y > lane_bottom + 0.2:
                continue
            boxes.append((x, y, right, bottom))

        if not boxes:
            return None
        return {
            "left": min(box[0] for box in boxes),
            "top": min(box[1] for box in boxes),
            "right": max(box[2] for box in boxes),
            "bottom": max(box[3] for box in boxes),
        }

    def _settings(self) -> Dict[str, Any]:
        return {
            "target_utilization": float(self.block_config.get("target_utilization", 0.95)),
            "acceptable_min": float(self.block_config.get("acceptable_min", 0.90)),
            "acceptable_max": float(self.block_config.get("acceptable_max", 0.97)),
            "hard_max": float(self.block_config.get("hard_max", 0.98)),
            "safety_factor": float(self.block_config.get("safety_factor", 0.82)),
            "min_extra_chars": int(self.block_config.get("min_extra_chars", 40)),
            "min_missing_height_for_expand": float(self.block_config.get("min_missing_height_for_expand", 0.25)),
            "max_extra_chars_per_block": int(self.block_config.get("max_extra_chars_per_block", 700)),
            "max_text_box_slack_ratio": float(self.block_config.get("max_text_box_slack_ratio", 2.2)),
            "max_text_box_slack_inches": float(self.block_config.get("max_text_box_slack_inches", 2.0)),
        }

    def _visible_text_height(self, item: Dict[str, Any], box_height: float) -> float:
        measured = self._estimated_text_height(item)
        if measured <= 0:
            return min(box_height, 0.1)
        return min(box_height, measured)

    def _text_box_slack(self, text_items: List[Dict[str, Any]]) -> Dict[str, float]:
        max_ratio = 1.0
        max_inches = 0.0
        for item in text_items:
            box_height = float(item.get("height", 0.0) or 0.0)
            measured = self._estimated_text_height(item)
            if box_height <= 0 or measured <= 0:
                continue
            max_ratio = max(max_ratio, box_height / measured)
            max_inches = max(max_inches, box_height - measured)
        return {"max_ratio": round(max_ratio, 3), "max_inches": round(max_inches, 4)}

    def _estimated_text_height(self, item: Dict[str, Any]) -> float:
        text = self._strip_markup(str(item.get("content") or ""))
        width = max(float(item.get("width", 0.0) or 0.0), 0.5)
        font_size = int(item.get("font_size") or self.config["typography"]["sizes"]["body_text"])
        line_spacing = float(item.get("line_spacing") or self.config["typography"].get("line_spacing", 1.0))
        chars_per_line = self._chars_per_line(width, font_size)
        line_count = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_count += max(1, math.ceil(len(line) / max(chars_per_line, 1)))
        if line_count <= 0:
            return 0.0
        return line_count * self._line_height(font_size, line_spacing) + max(line_count - 1, 0) * 0.04 + 0.12

    def _strip_markup(self, content: str) -> str:
        text = re.sub(r"<color:[^>]+>", "", content)
        text = text.replace("</color>", "")
        text = text.replace("**", "")
        text = text.replace("*", "")
        return text

    def _section_containers(self, layout: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            element
            for element in layout
            if element.get("type") == "section_container" and element.get("section_id")
        ]

    def _children_by_section(
        self,
        layout: List[Dict[str, Any]],
        containers: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        section_ids = {str(container.get("section_id")) for container in containers}
        children = {section_id: [] for section_id in section_ids}
        for element in layout:
            if element.get("type") == "section_container":
                continue
            section_id = str(element.get("section_id") or "")
            if section_id in children:
                children[section_id].append(element)
                continue
            element_id = str(element.get("id") or element.get("slot_id") or "")
            for candidate in section_ids:
                if element_id.startswith(f"{candidate}_"):
                    children[candidate].append(element)
                    break
        return children

    def _story_sections_by_id(self, story_board: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            str(section.get("section_id")): section
            for section in (story_board.get("spatial_content_plan") or {}).get("sections", [])
            if section.get("section_id")
        }

    def _representative_text_item(self, text_items: List[Dict[str, Any]], lane: Dict[str, Any]) -> Dict[str, Any]:
        if text_items:
            return max(text_items, key=lambda item: float(item.get("width", 0.0) or 0.0))
        return {
            "width": max(float(lane.get("w", 0.0) or 0.0) - 0.6, 0.5),
            "font_size": self.config["typography"]["sizes"]["body_text"],
            "line_spacing": self.config["typography"].get("line_spacing", 1.0),
        }

    def _match_lane(self, container: Dict[str, Any], lane_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cx = float(container.get("x", 0.0) or 0.0) + float(container.get("width", 0.0) or 0.0) / 2
        cy = float(container.get("y", 0.0) or 0.0) + float(container.get("height", 0.0) or 0.0) / 2
        for lane in lane_map.values():
            if (
                float(lane.get("x", 0.0)) <= cx <= float(lane.get("x", 0.0)) + float(lane.get("w", 0.0))
                and float(lane.get("y", 0.0)) <= cy <= float(lane.get("y", 0.0)) + float(lane.get("h", 0.0))
            ):
                return lane
        return None

    def _line_height(self, font_size: int, line_spacing: float) -> float:
        return (font_size / 72) * max(line_spacing, 0.9) * 1.15

    def _chars_per_line(self, width_inches: float, font_size: int) -> int:
        micro_config = self.config.get("micro_layout_refinement", {})
        if getattr(self, "_analysis_orientation", "") == "portrait":
            chars_per_inch = float(
                self.block_config.get(
                    "portrait_ppt_chars_per_inch_at_44pt",
                    micro_config.get("portrait_ppt_chars_per_inch_at_44pt", 3.25),
                )
            )
        else:
            chars_per_inch = float(
                self.block_config.get(
                    "ppt_chars_per_inch_at_44pt",
                    micro_config.get("ppt_chars_per_inch_at_44pt", 4.2),
                )
            )
        return max(int(width_inches * chars_per_inch * (44 / max(font_size, 1))), 18)

    def _title_from_children(self, title_items: List[Dict[str, Any]]) -> str:
        if not title_items:
            return ""
        return str(title_items[0].get("content") or "")

    def _save_outputs(self, state: PosterState, report: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "block_occupancy_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


def block_occupancy_analyzer_node(state: PosterState) -> Dict[str, Any]:
    result = BlockOccupancyAnalyzer()(state)
    return {
        **state,
        "block_occupancy_report": result.get("block_occupancy_report"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
