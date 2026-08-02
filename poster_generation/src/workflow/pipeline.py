"""
Main workflow pipeline for paper-to-poster generation
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.state.poster_state import create_state, PosterState
from src.tools.layout_api import LayoutTemplates
from src.template_extraction.block_template_registry import get_block_template_info, is_block_template_id
from src.config.poster_config import load_config
from src.utils.style_options import (
    available_background_palettes,
    available_background_styles,
    available_poster_styles,
    available_visual_densities,
    normalize_background_palette,
    normalize_background_style,
    normalize_poster_style,
    normalize_visual_density,
)
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if not env_path.exists():
    env_path = project_root.parent / '.env'
load_dotenv(env_path, override=False)

DEFAULT_STANDARD_TEMPLATE = "cluster_43_landscape"


def resolve_poster_dimensions(layout_template: str, width: float | None, height: float | None) -> tuple[float, float]:
    """Resolve canvas size from the selected template.

    Built-in templates keep the historical 54 x 36 landscape default. Block
    templates use their own aspect ratio and orientation, so current cluster_*
    templates default to a portrait canvas.
    """
    template_for_size = DEFAULT_STANDARD_TEMPLATE if layout_template == "auto" else layout_template
    if is_block_template_id(template_for_size):
        info = get_block_template_info(template_for_size)
        if not info:
            raise ValueError(f"Unknown block template: {template_for_size}")
        aspect_ratio = float(info["aspect_ratio"])
        orientation = info["orientation"]
        recommended = info["recommended_canvas_size"]

        if width is None and height is None:
            width = float(recommended["width"])
            height = float(recommended["height"])
        elif width is None:
            height = float(height)
            width = height * aspect_ratio
        elif height is None:
            width = float(width)
            height = width / max(aspect_ratio, 1e-6)
        else:
            width = float(width)
            height = float(height)
            requested_orientation = "portrait" if width < height else "landscape"
            if requested_orientation != orientation:
                raise ValueError(
                    f"Template {template_for_size} is {orientation}, but requested canvas is {requested_orientation} "
                    f"({width:g} x {height:g})."
                )
        return float(width), float(height)

    return float(width if width is not None else 54.0), float(height if height is not None else 36.0)


def create_timing_wrapper(node_func: Callable, component_name: str) -> Callable:
    """Wrap agent node with timing tracking"""
    def wrapper(state: PosterState) -> PosterState:
        start_time = time.time()
        result = node_func(state)
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)

        if component_name == "parser":
            result["timing_metrics"].parser_time = elapsed
        elif component_name == "standard_template_preselector":
            result["timing_metrics"].standard_template_preselector_time = elapsed
        elif component_name == "template_capacity_planner":
            result["timing_metrics"].template_capacity_planner_time = elapsed
        elif component_name == "poster_keypoint_selector":
            result["timing_metrics"].poster_keypoint_selector_time = elapsed
        elif component_name == "curator":
            result["timing_metrics"].curator_time = elapsed
        elif component_name == "template_block_planner":
            result["timing_metrics"].template_block_planner_time = elapsed
        elif component_name == "layout_optimizer":
            result["timing_metrics"].layout_optimizer_time = elapsed
        elif component_name == "color_agent":
            result["timing_metrics"].color_agent_time = elapsed
        elif component_name == "header_planner":
            result["timing_metrics"].header_planner_time = elapsed
        elif component_name == "header_block_reviewer":
            result["timing_metrics"].header_block_reviewer_time = elapsed
        elif component_name == "font_agent":
            result["timing_metrics"].font_agent_time = elapsed
        elif component_name == "micro_layout_refiner":
            result["timing_metrics"].micro_layout_refiner_time = elapsed
        elif component_name == "section_title_designer":
            result["timing_metrics"].title_designer_time = elapsed
        elif component_name == "visual_asset_agent":
            result["timing_metrics"].visual_asset_agent_time = elapsed
        elif component_name == "generated_teaser_agent":
            result["timing_metrics"].generated_teaser_agent_time = elapsed
        elif component_name == "background_image_agent":
            result["timing_metrics"].background_image_agent_time = elapsed
        elif component_name == "affiliation_logo_agent":
            result["timing_metrics"].affiliation_logo_agent_time = elapsed
        elif component_name == "renderer":
            result["timing_metrics"].renderer_time = elapsed
        elif component_name == "vlm_layout_reviewer":
            result["timing_metrics"].vlm_layout_reviewer_time = elapsed
        elif component_name == "visual_legibility_reviewer":
            result["timing_metrics"].visual_legibility_reviewer_time = elapsed
        elif component_name == "adaptive_column_relayout":
            result["timing_metrics"].adaptive_column_relayout_time = elapsed
        elif component_name == "template_region_relayout":
            result["timing_metrics"].template_region_relayout_time = elapsed
        elif component_name == "block_occupancy_analyzer":
            result["timing_metrics"].block_occupancy_analyzer_time = elapsed
        elif component_name == "block_vlm_reviewer":
            result["timing_metrics"].block_vlm_reviewer_time = elapsed
        elif component_name == "block_content_refiner":
            result["timing_metrics"].block_content_refiner_time = elapsed

        return result
    return wrapper


def _block_refinement_max_iterations(state: PosterState | None = None) -> int:
    config = load_config()
    if state and state.get("template_fast_mode"):
        return int(config.get("template_fast_mode", {}).get("emergency_repair_max_iterations", 1))
    return int(config.get("block_refinement", {}).get("max_iterations", 2))


def _template_region_repair_max_iterations() -> int:
    return int(load_config().get("vlm_layout_review", {}).get("template_prior_max_repairs", 1))


def _route_after_visual_asset_agent(state: PosterState) -> str:
    if state.get("visual_reflow_required") and state.get("visual_reflow_count", 0) <= 1:
        return "layout_optimizer"
    return "renderer"


def _route_after_micro_layout_refiner(state: PosterState) -> str:
    if state.get("draft_status") == "rejected":
        if (
            state.get("template_layout_mode") == "template_prior"
            and state.get("template_repair_count", 0) < _template_region_repair_max_iterations()
        ):
            return "template_region_relayout"
        return "end"
    return "visual_asset_agent"


def _route_after_renderer(state: PosterState) -> str:
    if state.get("render_stage") == "final" or state.get("final_poster_accepted", False):
        return "end"
    if (
        load_config().get("header_block_review", {}).get("enabled", True)
        and not state.get("header_block_review")
    ):
        return "header_block_reviewer"
    if (
        state.get("enable_block_vlm_review", False)
        and state.get("block_refinement_count", 0) < _block_refinement_max_iterations(state)
        and not state.get("block_vlm_review")
    ):
        return "block_occupancy_analyzer"
    if (
        state.get("enable_visual_legibility_review", False)
        and not state.get("visual_legibility_review")
    ):
        return "visual_legibility_reviewer"
    if state.get("enable_vlm_layout_review", False):
        return "vlm_layout_reviewer"
    return "prepare_final_render"


def _route_after_header_block_reviewer(state: PosterState) -> str:
    if state.get("header_block_patch_applied", False):
        return "renderer"
    return _route_after_renderer(state)


def _route_after_visual_legibility_reviewer(state: PosterState) -> str:
    if state.get("template_fast_mode"):
        if state.get("enable_vlm_layout_review", False):
            return "vlm_layout_reviewer"
        return "prepare_final_render"
    if state.get("template_repair_required", False):
        return "template_region_relayout"
    if state.get("adaptive_relayout_required", False):
        return "adaptive_column_relayout"
    if state.get("enable_vlm_layout_review", False):
        return "vlm_layout_reviewer"
    return "prepare_final_render"


def _route_after_vlm_layout_reviewer(state: PosterState) -> str:
    if state.get("template_fast_mode") and not state.get("template_repair_required", False):
        return "prepare_final_render"
    if state.get("vlm_reflow_required", False):
        return "visual_asset_agent"
    if state.get("template_repair_required", False):
        return "template_region_relayout"
    return "prepare_final_render"


def _route_after_block_content_refiner(state: PosterState) -> str:
    if state.get("block_refinement_required", False):
        return "layout_optimizer"
    if (
        state.get("enable_visual_legibility_review", False)
        and not state.get("visual_legibility_review")
    ):
        return "visual_legibility_reviewer"
    if state.get("enable_vlm_layout_review", False):
        return "vlm_layout_reviewer"
    return "prepare_final_render"


def _route_after_template_region_relayout(state: PosterState) -> str:
    if state.get("draft_status") == "rejected":
        return "end"
    return "layout_optimizer"


def _prepare_final_render_node(state: PosterState) -> PosterState:
    state["render_stage"] = "final"
    state["vlm_reflow_required"] = False
    state["template_repair_required"] = False
    state["block_refinement_required"] = False
    state["current_agent"] = "prepare_final_render"
    return state


def _load_content_json(state: PosterState, filename: str) -> Dict[str, Any]:
    path = Path(state.get("output_dir", "")) / "content" / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_placeholder_image(path: str | Path) -> bool:
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            stat = ImageStat.Stat(image.resize((32, 32)).convert("RGB"))
        mean = sum(stat.mean) / 3
        variance = sum(stat.var) / 3
        return variance < 18 and 175 <= mean <= 230
    except Exception:
        return False


def _final_artifact_failures(state: PosterState) -> list[Dict[str, Any]]:
    failures: list[Dict[str, Any]] = []
    if not state.get("pptx_output_path") and not state.get("poster_preview_path"):
        return failures
    for field, label in (("pptx_output_path", "pptx"), ("poster_preview_path", "png")):
        value = state.get(field)
        if not value or not Path(str(value)).exists():
            failures.append({"category": "artifact", "artifact": label, "path": value or "", "reason": "missing"})

    if state.get("enable_generated_background", False):
        report = state.get("background_image_report") or {}
        background_path = report.get("background_image_path") or state.get("background_image_path")
        if not background_path or not Path(str(background_path)).exists():
            failures.append({"category": "generated_asset", "asset": "background", "reason": "missing"})
        elif _is_placeholder_image(background_path):
            failures.append({"category": "generated_asset_placeholder", "asset": "background", "path": str(background_path)})

    if state.get("enable_generated_teaser", False):
        report = state.get("generated_teaser_report") or {}
        if report.get("applied", True) is False:
            failures.append(
                {
                    "category": "generated_asset",
                    "asset": "teaser",
                    "reason": report.get("fallback_reason") or report.get("reason") or "needs_regeneration",
                    "needs_regeneration": bool(report.get("needs_regeneration", True)),
                }
            )
            return failures
        teaser_path = report.get("teaser_path")
        if not teaser_path or not Path(str(teaser_path)).exists():
            failures.append({"category": "generated_asset", "asset": "teaser", "reason": "missing"})
        elif _is_placeholder_image(teaser_path):
            failures.append({"category": "generated_asset_placeholder", "asset": "teaser", "path": str(teaser_path)})

    return failures


def _section_geometry_issues(state: PosterState) -> list[str]:
    layout = state.get("styled_layout") or []
    template = state.get("layout_template_metadata") or {}
    lane_map = {str(lane.get("id")): lane for lane in template.get("lanes") or [] if lane.get("id")}
    containers = [
        element
        for element in layout
        if element.get("type") == "section_container" and element.get("section_id")
    ]
    issues: list[str] = []

    for container in containers:
        lane_id = str(container.get("lane_id") or container.get("slot_id") or "")
        lane = lane_map.get(lane_id)
        if not lane:
            continue
        tolerance = 0.03
        bottom = float(container.get("y", 0.0) or 0.0) + float(container.get("height", 0.0) or 0.0)
        lane_bottom = float(lane.get("y", 0.0) or 0.0) + float(lane.get("h", 0.0) or 0.0)
        if bottom > lane_bottom + tolerance:
            issues.append(f"section exceeds slot {lane_id}: {container.get('section_id')}")

    for index, left in enumerate(containers):
        for right in containers[index + 1:]:
            if _boxes_overlap(left, right):
                issues.append(f"section containers overlap: {left.get('section_id')} and {right.get('section_id')}")
    return issues


def _boxes_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
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


def _micro_layout_lane_overflow_failures(micro_report: Dict[str, Any], config: Dict[str, Any]) -> list[Dict[str, Any]]:
    micro_config = config.get("micro_layout_refinement", {}) or {}
    tolerance = float(micro_config.get("final_lane_overflow_tolerance_inches", 0.02) or 0.02)
    overflows = []
    for lane in micro_report.get("lanes") or []:
        try:
            final_overflow = float(lane.get("final_overflow") or 0.0)
        except (TypeError, ValueError):
            continue
        if final_overflow <= tolerance:
            continue
        overflows.append(
            {
                "lane_id": lane.get("lane_id"),
                "final_overflow": round(final_overflow, 4),
                "tolerance": round(tolerance, 4),
                "force_fit_used": bool(lane.get("force_fit_used")),
            }
        )
    return overflows


def _oversized_body_font_failures(state: PosterState, max_body_font_size: float) -> list[Dict[str, Any]]:
    if max_body_font_size <= 0:
        return []
    oversized = []
    for element in state.get("styled_layout") or []:
        if element.get("type") != "text":
            continue
        try:
            font_size = float(element.get("font_size") or 0.0)
        except (TypeError, ValueError):
            continue
        if font_size <= max_body_font_size:
            continue
        oversized.append(
            {
                "slot_id": element.get("slot_id") or element.get("lane_id"),
                "section_id": element.get("section_id"),
                "element_id": element.get("id"),
                "font_size": round(font_size, 2),
                "max_font_size": round(max_body_font_size, 2),
            }
        )
    return oversized


def _dedupe_degraded_quality_states(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("component") or ""),
            str(item.get("category") or ""),
            str(item.get("fallback") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _logo_degraded_quality_states(state: PosterState) -> list[Dict[str, Any]]:
    degraded: list[Dict[str, Any]] = []
    if state.get("enable_affiliation_logos"):
        affiliations = state.get("affiliations") or []
        logos = [
            logo for logo in (state.get("affiliation_logos") or [])
            if logo.get("logo_path") and Path(logo["logo_path"]).exists()
        ]
        placeholder_logos = [
            logo for logo in logos
            if str(logo.get("status") or "").lower() == "placeholder"
            or str(logo.get("source") or "").lower() == "placeholder"
        ]
        if affiliations and not logos:
            degraded.append(
                {
                    "component": "affiliation_logo_agent",
                    "category": "affiliation_logo_resolution",
                    "reason": "affiliation logo generation was enabled but no institution logo asset was resolved",
                    "fallback": "no_affiliation_logo",
                }
            )
        elif placeholder_logos:
            degraded.append(
                {
                    "component": "affiliation_logo_agent",
                    "category": "affiliation_logo_resolution",
                    "reason": f"{len(placeholder_logos)} institution logo(s) used generated placeholder artwork",
                    "fallback": "placeholder_affiliation_logo",
                }
            )

    if state.get("conference_name") and not (state.get("logo_path") and Path(state["logo_path"]).exists()):
        degraded.append(
            {
                "component": "conference_logo_resolver",
                "category": "conference_logo_resolution",
                "reason": f"conference logo was requested for {state.get('conference_name')} but no local asset was resolved",
                "fallback": "no_conference_logo",
            }
        )
    return degraded


def _run_final_quality_gate(state: PosterState) -> PosterState:
    from src.agents.block_occupancy_analyzer import BlockOccupancyAnalyzer
    from src.utils.visual_footprint import evaluate_visual_footprints

    config = load_config()
    is_template_prior = state.get("template_layout_mode") == "template_prior"
    block_settings = config.get("block_refinement", {})
    min_utilization = float(block_settings.get("final_min_utilization", 0.88))
    min_mean_utilization = float(block_settings.get("final_mean_utilization", min_utilization))
    max_bottom_whitespace_inches = float(block_settings.get("final_max_bottom_whitespace_inches", 0.0) or 0.0)
    max_bottom_whitespace_fraction = float(block_settings.get("final_max_bottom_whitespace_fraction", 0.0) or 0.0)
    max_bottom_whitespace_line_fraction = float(
        block_settings.get("final_max_bottom_whitespace_line_fraction", 0.0) or 0.0
    )
    max_body_font_size = float(
        block_settings.get(
            "final_max_body_font_size",
            (config.get("micro_layout_refinement", {}) or {}).get("max_body_font_size", 0.0),
        )
        or 0.0
    )
    gate: Dict[str, Any] = {
        "source": "deterministic_final_gate",
        "accepted": True,
        "min_utilization": min_utilization,
        "min_mean_utilization": min_mean_utilization,
        "max_body_font_size": max_body_font_size,
        "max_bottom_whitespace_inches": max_bottom_whitespace_inches,
        "max_bottom_whitespace_fraction": max_bottom_whitespace_fraction,
        "template_layout_mode": state.get("template_layout_mode"),
        "degraded_quality_states": _dedupe_degraded_quality_states(
            list(state.get("degraded_quality_states") or []) + _logo_degraded_quality_states(state)
        ),
        "failures": [],
    }
    gate["failures"].extend(_final_artifact_failures(state))

    if is_template_prior:
        try:
            occupancy_report = BlockOccupancyAnalyzer().analyze(state)
            state["final_block_occupancy_report"] = occupancy_report
            gate["occupancy_summary"] = occupancy_report.get("summary")
            gate["blocks"] = [
                {
                    "slot_id": block.get("slot_id"),
                    "section_id": block.get("section_id"),
                    "section_title": block.get("section_title"),
                    "utilization": block.get("utilization"),
                    "bottom_whitespace": block.get("bottom_whitespace"),
                    "action": block.get("action"),
                    "visual_count": block.get("visual_count"),
                }
                for block in occupancy_report.get("blocks", [])
            ]
            whitespace_blocks = []
            allowed_whitespace_by_block: Dict[tuple[str, str], float] = {}
            for block in occupancy_report.get("blocks", []):
                available_height = float(block.get("available_height") or 0.0)
                bottom_whitespace = float(block.get("bottom_whitespace") or 0.0)
                allowed_values = [
                    value
                    for value in (
                        max_bottom_whitespace_inches,
                        available_height * max_bottom_whitespace_fraction if max_bottom_whitespace_fraction > 0 else 0.0,
                    )
                    if value > 0
                ]
                if not allowed_values:
                    continue
                allowed_whitespace = min(allowed_values)
                line_height = float(block.get("line_height") or 0.0)
                if line_height > 0 and max_bottom_whitespace_line_fraction > 0:
                    allowed_whitespace = max(
                        allowed_whitespace,
                        line_height * max_bottom_whitespace_line_fraction,
                    )
                allowed_whitespace_by_block[
                    (str(block.get("slot_id") or ""), str(block.get("section_id") or ""))
                ] = allowed_whitespace
                gap_tolerance = max(min(line_height * 0.04, 0.03), 0.005) if line_height > 0 else 0.005
                if bottom_whitespace > allowed_whitespace + gap_tolerance:
                    whitespace_blocks.append(
                        {
                            "slot_id": block.get("slot_id"),
                            "section_id": block.get("section_id"),
                            "section_title": block.get("section_title"),
                            "bottom_whitespace": block.get("bottom_whitespace"),
                            "allowed": round(allowed_whitespace, 4),
                        }
                    )
            low_blocks = []
            for block in occupancy_report.get("blocks", []):
                utilization = float(block.get("utilization") or 0.0)
                if utilization >= min_utilization:
                    continue
                key = (str(block.get("slot_id") or ""), str(block.get("section_id") or ""))
                allowed_whitespace = allowed_whitespace_by_block.get(key, 0.0)
                bottom_whitespace = float(block.get("bottom_whitespace") or 0.0)
                if allowed_whitespace > 0 and bottom_whitespace <= allowed_whitespace + 1e-6:
                    continue
                low_blocks.append(
                    {
                        "slot_id": block.get("slot_id"),
                        "section_id": block.get("section_id"),
                        "section_title": block.get("section_title"),
                        "utilization": block.get("utilization"),
                    }
                )
            if not occupancy_report.get("blocks"):
                gate["failures"].append({"category": "occupancy", "reason": "no content blocks measured"})
            if low_blocks:
                gate["failures"].append({"category": "occupancy", "low_blocks": low_blocks})
            if whitespace_blocks:
                gate["failures"].append({"category": "bottom_whitespace", "blocks": whitespace_blocks})
            mean_utilization = float((occupancy_report.get("summary") or {}).get("mean_utilization") or 0.0)
            if occupancy_report.get("blocks") and mean_utilization < min_mean_utilization and (low_blocks or whitespace_blocks):
                gate["failures"].append(
                    {
                        "category": "occupancy_mean",
                        "mean_utilization": mean_utilization,
                        "required": min_mean_utilization,
                    }
                )
        except Exception as exc:
            gate["failures"].append({"category": "occupancy", "reason": str(exc)})

        try:
            visual_footprint = evaluate_visual_footprints(
                state.get("styled_layout") or [],
                state.get("layout_template_metadata") or {},
                state,
                config,
            )
            gate["visual_footprint"] = visual_footprint
            if visual_footprint.get("violations"):
                gate["failures"].append(
                    {
                        "category": "visual_footprint",
                        "violations": visual_footprint["violations"],
                    }
                )
        except Exception as exc:
            gate["failures"].append({"category": "visual_footprint", "reason": str(exc)})

    micro_report = _load_content_json(state, "micro_layout_report.json")
    micro_issues = ((micro_report.get("validation") or {}).get("issues") or [])
    if micro_issues:
        gate["failures"].append({"category": "micro_layout", "issues": micro_issues})
    micro_lane_overflows = _micro_layout_lane_overflow_failures(micro_report, config)
    if micro_lane_overflows:
        gate["failures"].append({"category": "micro_layout_lane_overflow", "lanes": micro_lane_overflows})

    if is_template_prior:
        geometry_issues = _section_geometry_issues(state)
        if geometry_issues:
            gate["failures"].append({"category": "section_geometry", "issues": geometry_issues})

        oversized_body_fonts = _oversized_body_font_failures(state, max_body_font_size)
        if oversized_body_fonts:
            gate["failures"].append({"category": "body_font_scale", "blocks": oversized_body_fonts})

    vlm_review = state.get("vlm_layout_review") or {}
    required_reviews = (
        ("vlm_layout_reviewer", "enable_vlm_layout_review", vlm_review),
        ("visual_legibility_reviewer", "enable_visual_legibility_review", state.get("visual_legibility_review") or {}),
        ("block_vlm_reviewer", "enable_block_vlm_review", state.get("block_vlm_review") or {}),
    )
    for component, enabled_key, review in required_reviews:
        if not state.get(enabled_key) or not review:
            continue
        if review.get("degraded") or review.get("review_available") is False:
            gate["failures"].append(
                {
                    "category": "quality_review_unavailable",
                    "component": component,
                    "warnings": review.get("warnings") or [],
                }
            )

    visual_review = state.get("visual_legibility_review") or {}
    high_visual_issues = [
        issue
        for issue in (visual_review.get("issues") or [])
        if str(issue.get("severity", "")).lower() == "high"
    ]
    if high_visual_issues:
        gate["failures"].append({"category": "visual_legibility", "issues": high_visual_issues})

    block_review = state.get("block_vlm_review") or {}
    high_block_visual_issues = [
        block
        for block in (block_review.get("blocks") or [])
        if str(block.get("severity", "")).lower() == "high"
        and str(block.get("status", "")).lower() in {"overflow", "visual_too_small"}
    ]
    if high_block_visual_issues:
        gate["failures"].append({"category": "block_visual_legibility", "blocks": high_block_visual_issues})

    high_issues = [
        issue
        for issue in (vlm_review.get("issues") or [])
        if str(issue.get("severity", "")).lower() == "high"
        and str(issue.get("category", "")).lower() in {"whitespace", "overflow"}
    ]
    global_assessment = vlm_review.get("global_assessment") or {}
    high_whitespace_regions = [
        region
        for region in (global_assessment.get("major_whitespace_regions") or [])
        if str(region.get("severity", "")).lower() == "high"
    ]
    title_readability = str(global_assessment.get("title_readability") or "ok").lower()
    if high_issues:
        gate["failures"].append({"category": "vlm_high_issue", "issues": high_issues})
    if high_whitespace_regions:
        gate["failures"].append({"category": "vlm_major_whitespace", "regions": high_whitespace_regions})
    if title_readability == "too_small":
        override = _single_line_title_readability_override(state, config)
        if override:
            gate.setdefault("overrides", []).append(override)
        else:
            gate["failures"].append({"category": "title_readability", "status": title_readability})
    elif title_readability in {"crowded", "unclear"}:
        gate["failures"].append({"category": "title_readability", "status": title_readability})

    gate["accepted"] = not gate["failures"]
    state["final_quality_gate"] = gate

    output_dir = Path(state["output_dir"]) / "content"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_quality_gate.json", "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2)
    if state.get("final_block_occupancy_report"):
        with open(output_dir / "final_block_occupancy_report.json", "w", encoding="utf-8") as f:
            json.dump(state["final_block_occupancy_report"], f, indent=2)
    if gate["accepted"] and int(state.get("final_quality_repair_count", 0) or 0) <= 0:
        stale_repair_report = output_dir / "final_quality_repair_report.json"
        if stale_repair_report.exists():
            stale_repair_report.unlink()

    if not gate["accepted"]:
        state["final_poster_accepted"] = False
        state.setdefault("errors", []).append(f"final_quality_gate: {gate['failures']}")
        log_agent_error("final_quality_gate", f"rejected final poster: {gate['failures']}")
    else:
        log_agent_success("final_quality_gate", "accepted final poster")
    return state


def _clear_final_quality_gate_errors(state: PosterState) -> None:
    state["errors"] = [
        error
        for error in state.get("errors", [])
        if not str(error).startswith("final_quality_gate:")
    ]


def _final_gate_refinable_block_ids(gate: Dict[str, Any]) -> set[tuple[str, str]]:
    refinable: set[tuple[str, str]] = set()
    for failure in gate.get("failures") or []:
        category = str(failure.get("category") or "")
        if category == "occupancy":
            candidates = failure.get("low_blocks") or []
        elif category == "bottom_whitespace":
            candidates = failure.get("blocks") or []
        elif category == "body_font_scale":
            candidates = failure.get("blocks") or []
        else:
            continue
        for block in candidates:
            slot_id = str(block.get("slot_id") or "")
            section_id = str(block.get("section_id") or "")
            if slot_id and section_id:
                refinable.add((slot_id, section_id))
    return refinable


def _build_final_gate_refinement_occupancy(state: PosterState) -> Dict[str, Any]:
    gate = state.get("final_quality_gate") or {}
    occupancy = state.get("final_block_occupancy_report") or {}
    refinable_ids = _final_gate_refinable_block_ids(gate)
    if not refinable_ids:
        return {}

    config = load_config()
    block_settings = config.get("block_refinement", {})
    near_line_extra = int(block_settings.get("near_line_rewrite_extra_chars", 80))
    min_extra = int(block_settings.get("min_extra_chars", 10))
    repaired_blocks = []
    for block in occupancy.get("blocks") or []:
        slot_id = str(block.get("slot_id") or "")
        section_id = str(block.get("section_id") or "")
        if (slot_id, section_id) not in refinable_ids:
            continue
        repaired = dict(block)
        repaired["action"] = "expand"
        repaired["target_extra_chars"] = max(
            int(repaired.get("target_extra_chars") or 0),
            near_line_extra,
            min_extra,
        )
        repaired["final_gate_repair"] = True
        repaired["reason"] = "final quality gate requested full block text rewrite for bottom whitespace"
        repaired_blocks.append(repaired)

    if not repaired_blocks:
        return {}

    report = dict(occupancy)
    report["source"] = "final_quality_gate_repair"
    report["blocks"] = repaired_blocks
    report["summary"] = {
        **(occupancy.get("summary") or {}),
        "repair_block_count": len(repaired_blocks),
    }
    return report


def _write_final_gate_repair_report(state: PosterState, report: Dict[str, Any]) -> None:
    output_dir = Path(state["output_dir"]) / "content"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_quality_repair_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _attempt_final_gate_block_content_repair(state: PosterState) -> PosterState:
    """Repair final-gate block whitespace by rewriting source text, then rerendering.

    The repair deliberately goes through story_board -> layout -> font -> micro
    layout again. It never creates styled text elements directly, so rewritten
    content inherits the same fonts, colors, bars, spacing, and block styling as
    the rest of the poster.
    """
    if state.get("template_layout_mode") != "template_prior":
        return state
    if not state.get("enable_block_vlm_review", False):
        return state
    if (state.get("final_quality_gate") or {}).get("accepted", True):
        return state
    config = load_config()
    max_final_repairs = int(
        (config.get("block_refinement", {}) or {}).get("final_gate_repair_max_iterations", 1)
    )
    if int(state.get("final_quality_repair_count", 0)) >= max_final_repairs:
        return state

    repair_occupancy = _build_final_gate_refinement_occupancy(state)
    if not repair_occupancy.get("blocks"):
        return state

    from src.agents.background_image_agent import background_image_agent_node
    from src.agents.block_content_refiner import BlockContentRefiner
    from src.agents.font_agent import font_agent_node
    from src.agents.layout_with_balancer import layout_with_balancer_node
    from src.agents.micro_layout_refiner import micro_layout_refiner_node
    from src.agents.renderer import renderer_node
    from src.agents.visual_asset_agent import visual_asset_agent_node

    attempt = int(state.get("final_quality_repair_count", 0)) + 1
    state["final_quality_repair_count"] = attempt
    state["block_occupancy_report"] = repair_occupancy
    state["block_vlm_review"] = {"source": "final_quality_gate_repair", "blocks": []}
    repair_report: Dict[str, Any] = {
        "source": "final_quality_gate_repair",
        "attempt": attempt,
        "attempted": True,
        "blocks": [
            {
                "slot_id": block.get("slot_id"),
                "section_id": block.get("section_id"),
                "section_title": block.get("section_title"),
                "bottom_whitespace": block.get("bottom_whitespace"),
                "target_extra_chars": block.get("target_extra_chars"),
            }
            for block in repair_occupancy.get("blocks", [])
        ],
        "applied": False,
        "rerendered": False,
    }

    log_agent_info(
        "final_quality_gate",
        f"attempting block content repair for {len(repair_occupancy.get('blocks', []))} final-gate whitespace block(s)",
    )
    before_refinement = int(state.get("block_refinement_count", 0))
    state = BlockContentRefiner()(state)
    patch = state.get("block_content_patch") or {}
    repair_report["block_content_patch"] = patch
    repair_report["applied"] = bool(patch.get("applied")) and int(state.get("block_refinement_count", 0)) > before_refinement
    if not repair_report["applied"]:
        repair_report["reason"] = "block content refiner did not apply a rewrite"
        _write_final_gate_repair_report(state, repair_report)
        return state

    _clear_final_quality_gate_errors(state)
    for node in (
        layout_with_balancer_node,
        font_agent_node,
        micro_layout_refiner_node,
        visual_asset_agent_node,
    ):
        state = node(state)
        if state.get("errors"):
            repair_report["reason"] = f"downstream node failed after content rewrite: {state.get('current_agent')}"
            _write_final_gate_repair_report(state, repair_report)
            return state
        if state.get("draft_status") == "rejected":
            repair_report["reason"] = state.get("draft_rejection_reason") or "draft rejected after content rewrite"
            _write_final_gate_repair_report(state, repair_report)
            return state

    state = _prepare_final_render_node(state)
    state = background_image_agent_node(state)
    if not state.get("errors"):
        state = renderer_node(state)
    repair_report["rerendered"] = bool(state.get("final_poster_accepted", False))
    _write_final_gate_repair_report(state, repair_report)
    return state


def _single_line_title_readability_override(state: PosterState, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    header_plan = state.get("header_plan") or {}
    title = header_plan.get("title") or {}
    if not title.get("single_line"):
        return None
    try:
        font_size = float(title.get("font_size") or 0.0)
    except (TypeError, ValueError):
        return None
    header_config = config.get("header_planner", {})
    orientation = str((state.get("layout_template_metadata") or {}).get("orientation") or "").lower()
    min_font_size = float(
        header_config.get(
            "portrait_single_line_title_gate_min_font_size"
            if orientation == "portrait"
            else "single_line_title_gate_min_font_size",
            42,
        )
    )
    if font_size < min_font_size:
        return None
    return {
        "category": "title_readability",
        "status": "too_small",
        "reason": "single_line_title_policy",
        "font_size": font_size,
        "min_font_size": min_font_size,
    }


def create_workflow_graph():
    """create the langgraph workflow"""
    from langgraph.graph import StateGraph, START, END
    from src.agents.adaptive_column_relayout import adaptive_column_relayout_node
    from src.agents.background_image_agent import background_image_agent_node
    from src.agents.block_content_refiner import block_content_refiner_node
    from src.agents.block_occupancy_analyzer import block_occupancy_analyzer_node
    from src.agents.block_vlm_reviewer import block_vlm_reviewer_node
    from src.agents.color_agent import color_agent_node
    from src.agents.affiliation_logo_agent import affiliation_logo_agent_node
    from src.agents.curator import curator_node
    from src.agents.font_agent import font_agent_node
    from src.agents.generated_teaser_agent import generated_teaser_agent_node
    from src.agents.header_block_reviewer import header_block_reviewer_node
    from src.agents.header_planner import header_planner_node
    from src.agents.layout_with_balancer import layout_with_balancer_node as layout_optimizer_node
    from src.agents.micro_layout_refiner import micro_layout_refiner_node
    from src.agents.parser import parser_node
    from src.agents.poster_keypoint_selector import poster_keypoint_selector_node
    from src.agents.renderer import renderer_node
    from src.agents.section_title_designer import section_title_designer_node
    from src.agents.standard_template_preselector import standard_template_preselector_node
    from src.agents.template_capacity_planner import template_capacity_planner_node
    from src.agents.template_block_planner import template_block_planner_node
    from src.agents.template_region_relayout import template_region_relayout_node
    from src.agents.vlm_layout_reviewer import vlm_layout_reviewer_node
    from src.agents.visual_asset_agent import visual_asset_agent_node
    from src.agents.visual_legibility_reviewer import visual_legibility_reviewer_node

    graph = StateGraph(PosterState)

    graph.add_node("parser", create_timing_wrapper(parser_node, "parser"))
    graph.add_node("affiliation_logo_agent", create_timing_wrapper(affiliation_logo_agent_node, "affiliation_logo_agent"))
    graph.add_node("standard_template_preselector", create_timing_wrapper(standard_template_preselector_node, "standard_template_preselector"))
    graph.add_node("template_capacity_planner", create_timing_wrapper(template_capacity_planner_node, "template_capacity_planner"))
    graph.add_node("poster_keypoint_selector", create_timing_wrapper(poster_keypoint_selector_node, "poster_keypoint_selector"))
    graph.add_node("curator", create_timing_wrapper(curator_node, "curator"))
    graph.add_node("generated_teaser_agent", create_timing_wrapper(generated_teaser_agent_node, "generated_teaser_agent"))
    graph.add_node("template_block_planner", create_timing_wrapper(template_block_planner_node, "template_block_planner"))
    graph.add_node("color_agent", create_timing_wrapper(color_agent_node, "color_agent"))
    graph.add_node("header_planner", create_timing_wrapper(header_planner_node, "header_planner"))
    graph.add_node("section_title_designer", create_timing_wrapper(section_title_designer_node, "section_title_designer"))
    graph.add_node("layout_optimizer", create_timing_wrapper(layout_optimizer_node, "layout_optimizer"))
    graph.add_node("font_agent", create_timing_wrapper(font_agent_node, "font_agent"))
    graph.add_node("micro_layout_refiner", create_timing_wrapper(micro_layout_refiner_node, "micro_layout_refiner"))
    graph.add_node("visual_asset_agent", create_timing_wrapper(visual_asset_agent_node, "visual_asset_agent"))
    graph.add_node("renderer", create_timing_wrapper(renderer_node, "renderer"))
    graph.add_node("header_block_reviewer", create_timing_wrapper(header_block_reviewer_node, "header_block_reviewer"))
    graph.add_node("block_occupancy_analyzer", create_timing_wrapper(block_occupancy_analyzer_node, "block_occupancy_analyzer"))
    graph.add_node("block_vlm_reviewer", create_timing_wrapper(block_vlm_reviewer_node, "block_vlm_reviewer"))
    graph.add_node("block_content_refiner", create_timing_wrapper(block_content_refiner_node, "block_content_refiner"))
    graph.add_node("visual_legibility_reviewer", create_timing_wrapper(visual_legibility_reviewer_node, "visual_legibility_reviewer"))
    graph.add_node("adaptive_column_relayout", create_timing_wrapper(adaptive_column_relayout_node, "adaptive_column_relayout"))
    graph.add_node("template_region_relayout", create_timing_wrapper(template_region_relayout_node, "template_region_relayout"))
    graph.add_node("vlm_layout_reviewer", create_timing_wrapper(vlm_layout_reviewer_node, "vlm_layout_reviewer"))
    graph.add_node("prepare_final_render", _prepare_final_render_node)
    graph.add_node("background_image_agent", create_timing_wrapper(background_image_agent_node, "background_image_agent"))

    graph.add_edge(START, "parser")
    graph.add_edge("parser", "affiliation_logo_agent")
    graph.add_edge("affiliation_logo_agent", "standard_template_preselector")
    graph.add_edge("standard_template_preselector", "template_capacity_planner")
    graph.add_edge("template_capacity_planner", "poster_keypoint_selector")
    graph.add_edge("poster_keypoint_selector", "curator")
    graph.add_edge("curator", "color_agent")
    graph.add_edge("color_agent", "header_planner")
    graph.add_edge("header_planner", "generated_teaser_agent")
    graph.add_edge("generated_teaser_agent", "template_block_planner")
    graph.add_edge("template_block_planner", "section_title_designer")
    graph.add_edge("section_title_designer", "layout_optimizer")
    graph.add_edge("layout_optimizer", "font_agent")
    graph.add_edge("font_agent", "micro_layout_refiner")
    graph.add_conditional_edges(
        "micro_layout_refiner",
        _route_after_micro_layout_refiner,
        {
            "visual_asset_agent": "visual_asset_agent",
            "template_region_relayout": "template_region_relayout",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "visual_asset_agent",
        _route_after_visual_asset_agent,
        {
            "layout_optimizer": "layout_optimizer",
            "renderer": "renderer",
        },
    )
    graph.add_conditional_edges(
        "renderer",
        _route_after_renderer,
        {
            "block_occupancy_analyzer": "block_occupancy_analyzer",
            "header_block_reviewer": "header_block_reviewer",
            "visual_legibility_reviewer": "visual_legibility_reviewer",
            "vlm_layout_reviewer": "vlm_layout_reviewer",
            "prepare_final_render": "prepare_final_render",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "header_block_reviewer",
        _route_after_header_block_reviewer,
        {
            "renderer": "renderer",
            "block_occupancy_analyzer": "block_occupancy_analyzer",
            "visual_legibility_reviewer": "visual_legibility_reviewer",
            "vlm_layout_reviewer": "vlm_layout_reviewer",
            "prepare_final_render": "prepare_final_render",
            "end": END,
        },
    )
    graph.add_edge("block_occupancy_analyzer", "block_vlm_reviewer")
    graph.add_edge("block_vlm_reviewer", "block_content_refiner")
    graph.add_conditional_edges(
        "block_content_refiner",
        _route_after_block_content_refiner,
        {
            "layout_optimizer": "layout_optimizer",
            "visual_legibility_reviewer": "visual_legibility_reviewer",
            "vlm_layout_reviewer": "vlm_layout_reviewer",
            "prepare_final_render": "prepare_final_render",
        },
    )
    graph.add_conditional_edges(
        "visual_legibility_reviewer",
        _route_after_visual_legibility_reviewer,
        {
            "template_region_relayout": "template_region_relayout",
            "adaptive_column_relayout": "adaptive_column_relayout",
            "vlm_layout_reviewer": "vlm_layout_reviewer",
            "prepare_final_render": "prepare_final_render",
        },
    )
    graph.add_edge("adaptive_column_relayout", "layout_optimizer")
    graph.add_conditional_edges(
        "template_region_relayout",
        _route_after_template_region_relayout,
        {
            "layout_optimizer": "layout_optimizer",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "vlm_layout_reviewer",
        _route_after_vlm_layout_reviewer,
        {
            "visual_asset_agent": "visual_asset_agent",
            "template_region_relayout": "template_region_relayout",
            "prepare_final_render": "prepare_final_render",
        },
    )
    graph.add_edge("prepare_final_render", "background_image_agent")
    graph.add_edge("background_image_agent", "renderer")

    return graph


def save_timing_log(state: PosterState):
    """Save timing and cost metrics to log file"""
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "timing_cost_log.json"

    metrics = state["timing_metrics"]
    total_time = metrics.get_total_time()

    block_settings = (
        (state.get("block_occupancy_report") or {}).get("settings")
        or load_config().get("block_refinement", {})
    )

    api_calls_by_agent = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for call in metrics.api_calls:
        if call.agent not in api_calls_by_agent:
            api_calls_by_agent[call.agent] = {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": []
            }
        api_calls_by_agent[call.agent]["count"] += 1
        api_calls_by_agent[call.agent]["input_tokens"] += call.input_tokens
        api_calls_by_agent[call.agent]["output_tokens"] += call.output_tokens
        api_calls_by_agent[call.agent]["calls"].append({
            "type": call.call_type,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "timestamp": call.timestamp
        })
        total_input_tokens += call.input_tokens
        total_output_tokens += call.output_tokens

    log_data = {
        "overall": {
            "total_runtime_seconds": total_time,
            "total_runtime_minutes": round(total_time / 60, 2),
            "total_api_calls": metrics.get_api_call_count(),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "component_timing": {
            "parser": {
                "time_seconds": round(metrics.parser_time, 2),
                "percentage": metrics.get_component_percentage(metrics.parser_time)
            },
            "standard_template_preselector": {
                "time_seconds": round(metrics.standard_template_preselector_time, 2),
                "percentage": metrics.get_component_percentage(metrics.standard_template_preselector_time)
            },
            "template_capacity_planner": {
                "time_seconds": round(metrics.template_capacity_planner_time, 2),
                "percentage": metrics.get_component_percentage(metrics.template_capacity_planner_time)
            },
            "poster_keypoint_selector": {
                "time_seconds": round(metrics.poster_keypoint_selector_time, 2),
                "percentage": metrics.get_component_percentage(metrics.poster_keypoint_selector_time)
            },
            "curator": {
                "time_seconds": round(metrics.curator_time, 2),
                "percentage": metrics.get_component_percentage(metrics.curator_time)
            },
            "template_block_planner": {
                "time_seconds": round(metrics.template_block_planner_time, 2),
                "percentage": metrics.get_component_percentage(metrics.template_block_planner_time)
            },
            "layout_optimizer": {
                "time_seconds": round(metrics.layout_optimizer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.layout_optimizer_time)
            },
            "color_agent": {
                "time_seconds": round(metrics.color_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.color_agent_time)
            },
            "header_planner": {
                "time_seconds": round(metrics.header_planner_time, 2),
                "percentage": metrics.get_component_percentage(metrics.header_planner_time)
            },
            "header_block_reviewer": {
                "time_seconds": round(metrics.header_block_reviewer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.header_block_reviewer_time)
            },
            "font_agent": {
                "time_seconds": round(metrics.font_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.font_agent_time)
            },
            "micro_layout_refiner": {
                "time_seconds": round(metrics.micro_layout_refiner_time, 2),
                "percentage": metrics.get_component_percentage(metrics.micro_layout_refiner_time)
            },
            "title_designer": {
                "time_seconds": round(metrics.title_designer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.title_designer_time)
            },
            "visual_asset_agent": {
                "time_seconds": round(metrics.visual_asset_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.visual_asset_agent_time)
            },
            "generated_teaser_agent": {
                "time_seconds": round(metrics.generated_teaser_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.generated_teaser_agent_time)
            },
            "background_image_agent": {
                "time_seconds": round(metrics.background_image_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.background_image_agent_time)
            },
            "affiliation_logo_agent": {
                "time_seconds": round(metrics.affiliation_logo_agent_time, 2),
                "percentage": metrics.get_component_percentage(metrics.affiliation_logo_agent_time)
            },
            "renderer": {
                "time_seconds": round(metrics.renderer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.renderer_time)
            },
            "vlm_layout_reviewer": {
                "time_seconds": round(metrics.vlm_layout_reviewer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.vlm_layout_reviewer_time)
            },
            "visual_legibility_reviewer": {
                "time_seconds": round(metrics.visual_legibility_reviewer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.visual_legibility_reviewer_time)
            },
            "adaptive_column_relayout": {
                "time_seconds": round(metrics.adaptive_column_relayout_time, 2),
                "percentage": metrics.get_component_percentage(metrics.adaptive_column_relayout_time)
            },
            "template_region_relayout": {
                "time_seconds": round(metrics.template_region_relayout_time, 2),
                "percentage": metrics.get_component_percentage(metrics.template_region_relayout_time)
            },
            "block_occupancy_analyzer": {
                "time_seconds": round(metrics.block_occupancy_analyzer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.block_occupancy_analyzer_time)
            },
            "block_vlm_reviewer": {
                "time_seconds": round(metrics.block_vlm_reviewer_time, 2),
                "percentage": metrics.get_component_percentage(metrics.block_vlm_reviewer_time)
            },
            "block_content_refiner": {
                "time_seconds": round(metrics.block_content_refiner_time, 2),
                "percentage": metrics.get_component_percentage(metrics.block_content_refiner_time)
            }
        },
        "api_calls_by_agent": api_calls_by_agent,
        "model_info": {
            "poster_variant": state.get("poster_variant"),
            "text_model": f"{state['text_model'].provider}/{state['text_model'].model_name}",
            "vision_model": f"{state['vision_model'].provider}/{state['vision_model'].model_name}",
            "vlm_layout_review": {
                "enabled": state.get("enable_vlm_layout_review", False),
                "model": state.get("vlm_model") or os.getenv("VLM_MODEL"),
            },
            "visual_legibility_review": {
                "enabled": state.get("enable_visual_legibility_review", False),
                "adaptive_column_width": state.get("enable_adaptive_column_width", False),
                "adaptive_relayout_count": state.get("adaptive_relayout_count", 0),
                "adaptive_lane_widths": state.get("adaptive_lane_widths"),
            },
            "block_refinement": {
                "enabled": state.get("enable_block_vlm_review", False),
                "block_refinement_count": state.get("block_refinement_count", 0),
                "target_utilization": block_settings.get("target_utilization", 0.95),
            },
            "header_plan": {
                "route": (state.get("header_plan") or {}).get("route"),
                "subtitle": bool(((state.get("header_plan") or {}).get("subtitle") or {}).get("text")),
                "fallback": (state.get("header_plan") or {}).get("fallback", False),
                "validation": (state.get("header_plan") or {}).get("validation"),
            },
            "generated_teaser": {
                "enabled": state.get("enable_generated_teaser", False),
                "target_section_id": (state.get("generated_teaser_report") or {}).get("target_section_id"),
                "asset_id": (state.get("generated_teaser_report") or {}).get("asset_id"),
            },
            "generated_background": {
                "enabled": state.get("enable_generated_background", False),
                "requested_style": state.get("background_style"),
                "resolved_style": (state.get("background_image_report") or {}).get("resolved_style"),
                "requested_palette": state.get("background_palette"),
                "resolved_palette": (state.get("background_image_report") or {}).get("resolved_palette"),
            },
        }
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    log_agent_success("pipeline", f"Timing log saved to: {log_path}")
    return log_data


def main():
    config = load_config()
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return value.strip().lower() not in {"0", "false", "no", "off"}

    default_text_model = os.getenv("PAPER2POSTER_TEXT_MODEL") or os.getenv("PAPER2POSTER_MODEL") or "gpt-5.4"
    default_vision_model = (
        os.getenv("PAPER2POSTER_VISION_MODEL")
        or os.getenv("PAPER2POSTER_MODEL")
        or os.getenv("VLM_MODEL")
        or default_text_model
    )
    default_vlm_model = os.getenv("PAPER2POSTER_VLM_MODEL") or os.getenv("VLM_MODEL")
    poster_style_choices = available_poster_styles(config)
    visual_density_choices = available_visual_densities(config)
    background_style_choices = available_background_styles(config)
    background_palette_choices = available_background_palettes(config)
    section_title_numbering_choices = ["off", "small", "inline"]
    default_poster_style = normalize_poster_style(os.getenv("PAPER2POSTER_STYLE"), config)
    default_visual_density = normalize_visual_density(os.getenv("PAPER2POSTER_VISUAL_DENSITY"), config)
    default_background_style = normalize_background_style(os.getenv("PAPER2POSTER_BACKGROUND_STYLE"), config)
    default_background_palette = normalize_background_palette(os.getenv("PAPER2POSTER_BACKGROUND_PALETTE"), config)
    default_affiliation_logos = _env_bool(
        "PAPER2POSTER_AFFILIATION_LOGOS",
        bool((config.get("affiliation_logos") or {}).get("enabled", False)),
    )
    default_generated_teaser = os.getenv("PAPER2POSTER_GENERATED_TEASER", "1").strip().lower() not in {"0", "false", "no", "off"}
    default_generated_background = os.getenv("PAPER2POSTER_GENERATED_BACKGROUND", "1").strip().lower() not in {"0", "false", "no", "off"}
    default_section_title_numbering = str(
        os.getenv("PAPER2POSTER_SECTION_TITLE_NUMBERING")
        or config.get("section_title_numbering")
        or "off"
    ).strip().lower()
    if default_section_title_numbering not in section_title_numbering_choices:
        default_section_title_numbering = "off"
    header_route_choices = ["auto", "classic_left", "centered", "right_title", "split_logos"]
    header_subtitle_choices = ["auto", "off", "always"]
    header_title_wrap_choices = ["auto", "single_line", "two_line"]
    default_header_route = os.getenv("PAPER2POSTER_HEADER_ROUTE", "auto")
    if default_header_route not in header_route_choices:
        default_header_route = "auto"
    default_header_subtitle = os.getenv("PAPER2POSTER_HEADER_SUBTITLE", "auto")
    if default_header_subtitle not in header_subtitle_choices:
        default_header_subtitle = "auto"
    default_header_title_wrap = os.getenv(
        "PAPER2POSTER_HEADER_TITLE_WRAP",
        str((config.get("header_planner") or {}).get("title_wrap_policy", "auto")),
    )
    if default_header_title_wrap not in header_title_wrap_choices:
        default_header_title_wrap = "auto"
    header_seed_env = os.getenv("PAPER2POSTER_HEADER_SEED")
    try:
        default_header_seed = int(header_seed_env) if header_seed_env else None
    except ValueError:
        default_header_seed = None

    parser = argparse.ArgumentParser(
        description=(
            "PosterMELD: multi-agent paper-to-poster generation for controllable "
            "design diversity with editable print-ready outputs"
        )
    )
    parser.add_argument("paper_path_positional", nargs="?", help="Path to the PDF paper")
    parser.add_argument("--paper_path", type=str, required=False, help="Path to the PDF paper")
    parser.add_argument("--text_model", type=str, default=default_text_model,
                       help="Text model for content processing (overrides PAPER2POSTER_TEXT_MODEL/PAPER2POSTER_MODEL env). "
                            "Any model resolvable by _get_model_config is accepted: a known alias "
                            "(e.g. gpt-5, gpt-5.4, gpt-4o, gpt-4.1-2025-04-14, claude-opus-4.5, gemini-2.5-pro, glm-4.6), "
                            "an explicit provider/model (e.g. openai/gpt-4o, anthropic/claude-opus-4.5), or any OpenAI-family name.")
    parser.add_argument("--vision_model", type=str, default=default_vision_model,
                       help="Vision model for image analysis (overrides env). Same resolution rules as --text_model "
                            "(e.g. gpt-5, gpt-4o, glm-4.6v, qwen3-vl-plus, or provider/model).")
    parser.add_argument("--poster_width", type=float, default=None, help="Poster width in inches")
    parser.add_argument("--poster_height", type=float, default=None, help="Poster height in inches")
    parser.add_argument(
        "--layout-template",
        type=str,
        default="auto",
        choices=LayoutTemplates.all_cli_template_choices(),
        help=f"Layout template family to use for poster composition. Defaults to auto ({DEFAULT_STANDARD_TEMPLATE}).",
    )
    parser.add_argument(
        "--list-layout-templates",
        action="store_true",
        help="List built-in and extracted layout templates, then exit.",
    )
    parser.add_argument("--url", type=str, help="URL for QR code on poster") # TODO
    parser.add_argument("--logo", type=str, default="", help="Path to conference/journal logo (overrides --conference)")
    parser.add_argument("--conference", type=str, default="", help="Conference name, e.g. 'CVPR', 'NeurIPS 2025'. Auto-resolved to local logo.")
    parser.add_argument("--aff_logo", "--aff-logo", dest="aff_logo", type=str, default="", help="Path to affiliation logo")
    parser.add_argument(
        "--enable-visual-refinement",
        action="store_true",
        help="Enable the second-phase visual asset agent for edit/generate planning.",
    )
    affiliation_group = parser.add_mutually_exclusive_group()
    affiliation_group.add_argument(
        "--enable-affiliation-logos",
        dest="enable_affiliation_logos",
        action="store_true",
        help="Enable automatic institution logo search/download and place up to the configured maximum in the title area.",
    )
    affiliation_group.add_argument(
        "--disable-affiliation-logos",
        dest="enable_affiliation_logos",
        action="store_false",
        help="Disable automatic institution logo search/download.",
    )
    parser.set_defaults(enable_affiliation_logos=default_affiliation_logos)
    parser.add_argument(
        "--affiliation-logo-mode",
        choices=["single", "multi"],
        default=os.getenv("PAPER2POSTER_AFFILIATION_LOGO_MODE", "single"),
        help="How many institution logos to place: 'single' (one, default) or 'multi' (1-3, up to the configured max, based on what resolves).",
    )
    parser.add_argument(
        "--enable-vlm-layout-review",
        action="store_true",
        help="Enable VLM screenshot review and one-pass safe layout correction.",
    )
    parser.add_argument(
        "--enable-visual-legibility-review",
        action="store_true",
        help="Enable VLM/heuristic review for tiny text inside figures and tables.",
    )
    parser.add_argument(
        "--enable-block-vlm-review",
        action="store_true",
        help="Enable block-level VLM review plus 95%% utilization content refinement.",
    )
    parser.add_argument(
        "--enable-adaptive-column-width",
        action="store_true",
        help="Allow one adaptive three-column width relayout when visual text is too small.",
    )
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument(
        "--enable-generated-background",
        dest="enable_generated_background",
        action="store_true",
        help="Generate a low-contrast academic background image and place it behind the poster.",
    )
    background_group.add_argument(
        "--disable-generated-background",
        dest="enable_generated_background",
        action="store_false",
        help="Disable the default generated poster background.",
    )
    parser.set_defaults(enable_generated_background=default_generated_background)
    teaser_group = parser.add_mutually_exclusive_group()
    teaser_group.add_argument(
        "--enable-generated-teaser",
        dest="enable_generated_teaser",
        action="store_true",
        help="Generate a paper-specific conceptual teaser visual for the motivation/introduction block.",
    )
    teaser_group.add_argument(
        "--disable-generated-teaser",
        dest="enable_generated_teaser",
        action="store_false",
        help="Disable the default generated teaser visual.",
    )
    parser.set_defaults(enable_generated_teaser=default_generated_teaser)
    parser.add_argument(
        "--background-palette",
        choices=background_palette_choices,
        default=default_background_palette,
        help="Palette for the generated poster background when --enable-generated-background is set.",
    )
    parser.add_argument(
        "--background-style",
        choices=background_style_choices,
        default=default_background_style,
        help="Visual style for the generated poster background; auto chooses from paper and layout context.",
    )
    parser.add_argument(
        "--poster-style",
        choices=poster_style_choices,
        default=default_poster_style,
        help="Typography and color preset for poster title bars, panels, and highlights.",
    )
    parser.add_argument(
        "--visual-density",
        choices=visual_density_choices,
        default=default_visual_density,
        help="How aggressively the planner should preserve figures and result tables.",
    )
    parser.add_argument(
        "--section-title-numbering",
        choices=section_title_numbering_choices,
        default=default_section_title_numbering,
        help="Section heading numbering style: off by default, small for compact numeric prefixes, inline for legacy '1. Title' labels.",
    )
    parser.add_argument(
        "--header-route",
        choices=header_route_choices,
        default=default_header_route,
        help="Header composition route for title, authors, and logos.",
    )
    parser.add_argument(
        "--header-subtitle",
        choices=header_subtitle_choices,
        default=default_header_subtitle,
        help="Whether to add a short generated subtitle when the paper title is short enough.",
    )
    parser.add_argument(
        "--header-title-wrap",
        choices=header_title_wrap_choices,
        default=default_header_title_wrap,
        help="Title wrapping policy for the header: auto, single_line, or two_line.",
    )
    parser.add_argument(
        "--header-seed",
        type=int,
        default=default_header_seed,
        help="Optional seed for reproducible auto header route and subtitle choices.",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=default_vlm_model,
        help="VLM model name for layout review. Falls back to VLM_MODEL env var.",
    )
    
    args = parser.parse_args()

    if args.list_layout_templates:
        print("Available layout templates:")
        for template_name in LayoutTemplates.all_cli_template_choices():
            if is_block_template_id(template_name):
                info = get_block_template_info(template_name) or {}
                size = info.get("recommended_canvas_size") or {}
                print(
                    f"- {template_name} "
                    f"({info.get('orientation', 'unknown')}, "
                    f"{info.get('slot_count', '?')} slots, "
                    f"{size.get('width', '?')}x{size.get('height', '?')} in)"
                )
            else:
                print(f"- {template_name}")
        return 0
    
    try:
        final_width, final_height = resolve_poster_dimensions(
            args.layout_template,
            args.poster_width,
            args.poster_height,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    # poster dimensions: use the requested/inferred canvas directly so portrait templates work.
    input_ratio = final_width / final_height
    normalized_ratio = max(input_ratio, 1 / input_ratio)
    if normalized_ratio > 2.1:
        print(f"❌ Poster ratio is out of range: {input_ratio:.3f}. Please use a landscape or portrait ratio up to 2.1:1.")
        return 1
    
    is_cluster_template = is_block_template_id(args.layout_template)
    if is_cluster_template:
        args.enable_visual_legibility_review = True
        args.enable_vlm_layout_review = True
        args.enable_block_vlm_review = True

    # check .env file
    if env_path.exists():
        print(f"✅ .env file found at: {env_path}")
    else:
        print(f"❌ .env file NOT found")
    
    # check api keys
    required_keys = {"openai": "OPENAI_API_KEY", "openai_responses": "VLM_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY", "zhipu": "ZHIPU_API_KEY", "moonshot": "MOONSHOT_API_KEY", "Minimax": "MINIMAX_API_KEY", "Alibaba": "ALIBABA_API_KEY"}
    model_providers = {"claude-sonnet-4-20250514": "anthropic", "claude-opus-4.5": "anthropic", "claude-opus-4-5-20251101": "anthropic", "gemini": "google", "gemini-2.5-pro": "google",
                      "gpt-5.1": "openai", "gpt-5.4-xhigh": "openai", "gpt-5.5-xhigh": "openai", "gpt-4o-2024-08-06": "openai", "gpt-4.1-2025-04-14": "openai", "gpt-4.1-mini-2025-04-14": "openai",
                      "gpt-5.4": "openai",
                      "glm-4.6": "zhipu", "glm-4.6v": "zhipu", "glm-4.5": "zhipu", "glm-4.5-air": "zhipu", "glm-4.5v": "zhipu", "glm-4": "zhipu", "glm-4v": "zhipu",
                      "kimi-k2-turbo-preview": "moonshot", "moonshot-v1-8k-vision-preview": "moonshot",
                      "qwen3-max": "Alibaba", "qwen3-vl-plus": "Alibaba",
                      "MiniMax-M2":"Minimax",}
    
    needed_keys = set()
    if args.text_model in model_providers:
        needed_keys.add(required_keys[model_providers[args.text_model]])
    if args.vision_model in model_providers:
        needed_keys.add(required_keys[model_providers[args.vision_model]])
    if (
        args.enable_vlm_layout_review
        or args.enable_visual_legibility_review
        or args.enable_block_vlm_review
    ) and not os.getenv("VLM_API_KEY"):
        needed_keys.add("VLM_API_KEY")
    
    missing = [k for k in needed_keys if not os.getenv(k)]
    if missing:
        print(f"❌ Missing API keys: {missing}")
        return 1
    
    # get pdf path
    pdf_path = args.paper_path or args.paper_path_positional
    if not pdf_path or not Path(pdf_path).exists():
        print("❌ PDF not found")
        return 1
    
    print("🚀 PosterMELD Pipeline")
    print(f"📄 PDF: {pdf_path}")
    print(f"🤖 Models: {args.text_model}/{args.vision_model}")
    print(f"📏 Size: {final_width}\" × {final_height:.2f}\"")
    print(f"🧩 Layout Template: {args.layout_template}")

    # resolve conference logo: explicit --logo beats --conference auto-resolve
    resolved_logo_path = args.logo if args.logo and Path(args.logo).exists() else ""
    conference_name = args.conference or ""
    if not resolved_logo_path and conference_name:
        from src.utils.conference_logos import resolve_conference_logo
        resolved_logo_path = resolve_conference_logo(conference_name) or ""
        if resolved_logo_path:
            print(f"🏛️  Conference Logo: {resolved_logo_path} (auto: {conference_name})")
        else:
            print(f"⚠️  Conference '{conference_name}' not in local library — no logo")
    elif resolved_logo_path:
        print(f"🏛️  Conference Logo: {resolved_logo_path}")

    print(f"🏫 Affiliation Logo: {args.aff_logo}")
    print(f"🎓 Auto Affiliation Logos: {'enabled' if args.enable_affiliation_logos else 'disabled'}")
    print(f"👁️ VLM Layout Review: {'enabled' if args.enable_vlm_layout_review else 'disabled'}")
    print(f"🔎 Visual Legibility Review: {'enabled' if args.enable_visual_legibility_review else 'disabled'}")
    print(f"🧱 Block 95% Refinement: {'enabled' if args.enable_block_vlm_review else 'disabled'}")
    print(f"📐 Adaptive Column Width: {'enabled' if args.enable_adaptive_column_width else 'disabled'}")
    print(f"🎭 Poster Style: {args.poster_style}")
    print(f"📊 Visual Density: {args.visual_density}")
    print(f"🔢 Section Title Numbering: {args.section_title_numbering}")
    print(f"🧾 Header Route: {args.header_route}")
    print(f"🧾 Header Subtitle: {args.header_subtitle}")
    print(f"🧾 Header Title Wrap: {args.header_title_wrap}")
    if args.header_seed is not None:
        print(f"🧾 Header Seed: {args.header_seed}")
    print(f"🖼️ Generated Teaser: {'enabled' if args.enable_generated_teaser else 'disabled'}")
    print(f"🎨 Generated Background: {'enabled' if args.enable_generated_background else 'disabled'}")
    if args.enable_generated_background:
        print(f"🎨 Background Style: {args.background_style}")
        print(f"🎨 Background Palette: {args.background_palette}")
    
    state = None
    try:
        state = create_state(
            pdf_path, args.text_model, args.vision_model,
            final_width, final_height,
            args.layout_template,
            args.url, resolved_logo_path, args.aff_logo,
            enable_visual_refinement=args.enable_visual_refinement,
            enable_affiliation_logos=args.enable_affiliation_logos,
            affiliation_logo_mode=args.affiliation_logo_mode,
            enable_vlm_layout_review=args.enable_vlm_layout_review,
            enable_visual_legibility_review=args.enable_visual_legibility_review,
            enable_block_vlm_review=args.enable_block_vlm_review,
            enable_adaptive_column_width=args.enable_adaptive_column_width,
            enable_generated_background=args.enable_generated_background,
            enable_generated_teaser=args.enable_generated_teaser,
            background_palette=args.background_palette,
            background_style=args.background_style,
            poster_style_preset=args.poster_style,
            visual_density=args.visual_density,
            section_title_numbering=args.section_title_numbering,
            header_route=args.header_route,
            header_subtitle_policy=args.header_subtitle,
            header_title_wrap_policy=args.header_title_wrap,
            header_seed=args.header_seed,
            vlm_model=args.vlm_model,
            conference_name=conference_name,
        )

        state["timing_metrics"].pipeline_start = time.time()

        log_agent_info("pipeline", "creating workflow graph")
        graph = create_workflow_graph()
        workflow = graph.compile()

        log_agent_info("pipeline", "executing workflow")
        final_state = workflow.invoke(state, config={"recursion_limit": 64})

        final_state["timing_metrics"].pipeline_end = time.time()

        if not final_state.get("errors") and final_state.get("final_poster_accepted", False):
            final_state = _run_final_quality_gate(final_state)
            while not (final_state.get("final_quality_gate") or {}).get("accepted", True):
                before_repair_count = int(final_state.get("final_quality_repair_count", 0))
                final_state = _attempt_final_gate_block_content_repair(final_state)
                after_repair_count = int(final_state.get("final_quality_repair_count", 0))
                if (
                    after_repair_count <= before_repair_count
                    or final_state.get("errors")
                    or not final_state.get("final_poster_accepted", False)
                ):
                    break
                final_state = _run_final_quality_gate(final_state)

        if final_state.get("errors"):
            log_agent_error("pipeline", f"Pipeline errors: {final_state['errors']}")
            save_timing_log(final_state)
            return 1
        if not final_state.get("final_poster_accepted", False):
            log_agent_error("pipeline", f"Poster rejected before final acceptance. Draft status: {final_state.get('draft_status')}")
            save_timing_log(final_state)
            return 1
        required_outputs = ["story_board", "design_layout", "color_scheme", "styled_layout", "resolved_visual_assets"]

        def is_missing_required_output(output_name: str) -> bool:
            value = final_state.get(output_name)
            if output_name == "resolved_visual_assets":
                return value is None
            return not value

        missing = [out for out in required_outputs if is_missing_required_output(out)]
        if missing:
            log_agent_error("pipeline", f"Missing outputs: {missing}")
            save_timing_log(final_state)
            return 1
        
        log_agent_success("pipeline", "Pipeline completed successfully")

        # full pipeline summary
        log_agent_success("pipeline", "Full pipeline complete")

        timing_log = save_timing_log(final_state)
        total_time = timing_log["overall"]["total_runtime_seconds"]
        total_calls = timing_log["overall"]["total_api_calls"]

        log_agent_info("pipeline", f"Total runtime: {total_time}s ({total_time/60:.2f} minutes)")
        log_agent_info("pipeline", f"Total API calls: {total_calls}")
        log_agent_info("pipeline", f"Total tokens: {final_state['tokens'].input_text} → {final_state['tokens'].output_text}")

        output_path = Path(final_state["output_dir"]) / f"{final_state['poster_name']}.pptx"
        log_agent_info("pipeline", f"Final poster saved to: {output_path}")

        return 0
        
    except Exception as e:
        log_agent_error("pipeline", f"Unexpected error: {e}")
        if state is not None:
            state["timing_metrics"].pipeline_end = time.time()
            try:
                save_timing_log(state)
            except Exception as timing_error:
                log_agent_error("pipeline", f"Failed to save partial timing log: {timing_error}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
