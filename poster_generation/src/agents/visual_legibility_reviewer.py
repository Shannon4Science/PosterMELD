"""
VLM-assisted visual legibility review.

This reviewer focuses on whether figures/tables in the rendered poster are
large enough to read. It can request one adaptive column-width relayout, but it
does not directly edit layout coordinates.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.agents.vlm_layout_reviewer import VLMLayoutReviewer
from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class VisualLegibilityReviewer:
    def __init__(self):
        self.name = "visual_legibility_reviewer"
        self.config = load_config()
        self.review_config = self.config.get("visual_legibility_review", {})
        self.vlm_client = VLMLayoutReviewer()

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_visual_legibility_review", False):
            return state

        log_agent_info(self.name, "checking visual readability and column-width needs")
        state["adaptive_relayout_required"] = False
        state["template_repair_required"] = False

        try:
            review = self._review_or_fallback(state)
            review = self._merge_heuristic_review(state, review)
            if review.get("degraded"):
                state.setdefault("degraded_quality_states", []).append(
                    {
                        "component": self.name,
                        "category": "visual_legibility_review",
                        "reason": "; ".join(str(item) for item in review.get("warnings", [])),
                        "fallback": review.get("fallback") or "deterministic_visual_heuristic",
                    }
                )
            state["visual_legibility_review"] = review
            is_template_prior = state.get("template_layout_mode") == "template_prior"
            fast_mode = bool(state.get("template_fast_mode"))

            max_relayout_count = int(self.review_config.get("max_relayout_count", 1))
            if fast_mode:
                if review.get("needs_relayout", False):
                    review.setdefault("warnings", []).append(
                        "Fast template-first mode records visual legibility concerns but does not trigger automatic relayout."
                    )
            elif is_template_prior:
                if review.get("needs_relayout", False):
                    if state.get("template_repair_count", 0) < max_relayout_count:
                        state["template_repair_required"] = True
                        state["template_repair_decision"] = {
                            "source": "visual_legibility_reviewer",
                            "reason": (review.get("layout_recommendation") or {}).get("reason") or "Template prior review requested region-level relayout.",
                            "review": review,
                        }
                    else:
                        review.setdefault("warnings", []).append(
                            "Template-prior visual legibility still has concerns after the available repair pass; escalating to VLM final review."
                        )
            elif (
                state.get("enable_adaptive_column_width", False)
                and review.get("needs_relayout", False)
                and state.get("adaptive_relayout_count", 0) < max_relayout_count
            ):
                state["adaptive_relayout_required"] = True

            self._save_outputs(state)
            state["current_agent"] = self.name
            log_agent_success(self.name, "visual legibility review completed")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _review_or_fallback(self, state: PosterState) -> Dict[str, Any]:
        preview_path = state.get("poster_preview_path")
        if not preview_path or not Path(preview_path).exists():
            return self._fallback_review("poster preview PNG is unavailable; using deterministic visual heuristic")

        base_url = os.getenv("VLM_BASE_URL")
        api_key = os.getenv("VLM_API_KEY")
        model = state.get("vlm_model") or os.getenv("VLM_MODEL")
        if not base_url or not api_key or not model:
            return self._fallback_review("VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL are required; using deterministic visual heuristic")

        prompt = self._build_prompt(state)
        image_data = self.vlm_client._encode_image(preview_path)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            content = self.vlm_client._request_vlm_text(base_url, headers, model, prompt, image_data)
            self.vlm_client._record_usage(state, self.name)
            review = self.vlm_client._parse_json(content)
            return self._normalize_review(review, source="vlm")
        except Exception as exc:
            return self._fallback_review(f"VLM visual legibility request failed ({exc}); using deterministic visual heuristic")

    def _build_prompt(self, state: PosterState) -> str:
        visual_slots = self._visual_slot_summary(state)
        template = state.get("layout_template_metadata") or {}
        is_template_prior = state.get("template_layout_mode") == "template_prior"
        target_schema = (
            '"target": "region id / slot id / visual id", "lane_id": null'
            if is_template_prior
            else '"target": "slot id or visual id", "lane_id": "left|middle|right"'
        )
        recommendation_schema = (
            '{ "target_region": "region id|null", "action": "none|promote_region|reduce_text_density", "preferred_width_ratio": 1.2, "reason": "short reason" }'
            if is_template_prior
            else '{ "target_lane": "left|middle|right|null", "action": "none|widen_lane", "preferred_width_ratio": 1.3, "reason": "short reason" }'
        )
        return f"""
You are checking an academic poster screenshot for figure/table readability.

Task:
- Focus on figures, tables, pipeline diagrams, and method diagrams.
- Decide whether any important visual has text that is too small for a viewer.
- If a central wide method/pipeline diagram is too small, recommend widening exactly one lane.
- Do not propose content rewrites or arbitrary coordinate patches.

Return strict JSON only:
{{
  "needs_relayout": true/false,
  "issues": [
    {{
      "severity": "low|medium|high",
      {target_schema},
      "description": "why the visual is hard to read"
    }}
  ],
  "layout_recommendation": {recommendation_schema},
  "warnings": []
}}

Guidelines:
- Mark needs_relayout=true only when a key visual would benefit from a wider column.
- Prefer target_lane=middle for central wide pipeline/method diagrams.
- If all visuals are readable, return needs_relayout=false and action=none.

Current template metadata:
{json.dumps(template, indent=2)}

Visual slots:
{json.dumps(visual_slots, indent=2)}
""".strip()

    def _visual_slot_summary(self, state: PosterState) -> List[Dict[str, Any]]:
        lane_map = self._lane_map(state)
        visual_assets = state.get("visual_assets") or {}
        resolved_assets = state.get("resolved_visual_assets") or {}
        slots = []
        for element in state.get("styled_layout") or []:
            if element.get("type") != "visual":
                continue
            visual_id = element.get("visual_id")
            slot_id = element.get("slot_id") or element.get("id")
            source = visual_assets.get(visual_id, {})
            resolved = resolved_assets.get(slot_id, {})
            slots.append({
                "id": element.get("id"),
                "slot_id": slot_id,
                "visual_id": visual_id,
                "lane_id": element.get("lane_id") or self._infer_lane(element, lane_map),
                "section_id": element.get("section_id"),
                "width": element.get("width"),
                "height": element.get("height"),
                "aspect": self._safe_aspect(element),
                "caption": source.get("caption") or resolved.get("caption"),
                "asset_type": source.get("asset_type") or resolved.get("asset_type"),
                "provenance": resolved.get("provenance") or source.get("provenance"),
            })
        return slots

    def _merge_heuristic_review(self, state: PosterState, review: Dict[str, Any]) -> Dict[str, Any]:
        heuristic = self._heuristic_review(state)
        if review.get("needs_relayout") or not heuristic.get("needs_relayout"):
            return review

        merged = dict(review)
        merged["needs_relayout"] = True
        merged["source"] = "vlm+heuristic" if review.get("source") == "vlm" else "heuristic"
        merged.setdefault("issues", [])
        merged["issues"].extend(heuristic.get("issues", []))
        merged["layout_recommendation"] = heuristic.get("layout_recommendation", {})
        merged.setdefault("warnings", [])
        merged["warnings"].append("Deterministic heuristic detected a wide key visual that should get a wider lane.")
        return merged

    def _heuristic_review(self, state: PosterState) -> Dict[str, Any]:
        lane_map = self._lane_map(state)
        min_wide_aspect = float(self.review_config.get("min_wide_visual_aspect", 2.2))
        max_width_for_wide_visual = float(self.review_config.get("max_width_for_wide_visual", 18.0))
        issues = []
        is_template_prior = state.get("template_layout_mode") == "template_prior"

        for element in state.get("styled_layout") or []:
            if element.get("type") != "visual":
                continue
            lane_id = element.get("lane_id") or self._infer_lane(element, lane_map)
            aspect = self._safe_aspect(element)
            width = float(element.get("width", 0.0) or 0.0)
            height = float(element.get("height", 0.0) or 0.0)
            caption = self._caption_for_visual(state, element).lower()
            asset = self._asset_for_visual(state, element)
            asset_type = str(asset.get("asset_type") or "").lower()
            is_table = asset_type == "table" or "table" in caption
            is_pipeline_like = any(key in caption for key in ["pipeline", "framework", "architecture", "hierarchical", "method", "overview"])
            if lane_id == "middle" and aspect >= min_wide_aspect and (width <= max_width_for_wide_visual or is_pipeline_like):
                issues.append({
                    "severity": "high",
                    "target": element.get("slot_id") or element.get("id") or element.get("visual_id"),
                    "lane_id": None if is_template_prior else lane_id,
                    "description": "Wide method/pipeline visual is likely too small in an equal-width middle column.",
                    "measured_width": width,
                    "measured_aspect": aspect,
                })
            if is_template_prior and is_table and width < 12.0:
                issues.append({
                    "severity": "medium",
                    "target": element.get("slot_id") or element.get("id") or element.get("visual_id"),
                    "lane_id": None,
                    "description": "Dense table visual is occupying a compact region and should be summarized or resized.",
                    "measured_width": width,
                })
            source_ppi = self._source_pixel_density(asset, width, height)
            hard_min_ppi = float(self.review_config.get("hard_min_table_source_ppi", 48.0))
            min_ppi = float(self.review_config.get("min_table_source_ppi", 72.0))
            if is_table and source_ppi is not None and source_ppi < min_ppi:
                issues.append({
                    "severity": "high" if source_ppi < hard_min_ppi else "medium",
                    "target": element.get("slot_id") or element.get("id") or element.get("visual_id"),
                    "lane_id": None if is_template_prior else lane_id,
                    "description": "Table source resolution is too low for its rendered size; enlarging it would only magnify blur.",
                    "measured_width": width,
                    "source_pixels_per_inch": round(source_ppi, 1),
                })

        if not issues:
            return self._normalize_review({}, source="heuristic")

        if is_template_prior:
            return self._normalize_review({
                "needs_relayout": True,
                "issues": issues,
                "layout_recommendation": {
                    "target_region": issues[0]["target"],
                    "action": "promote_region",
                    "preferred_width_ratio": 1.2,
                    "reason": "Template-prior poster has a key visual or dense table in a region that is too compact to read comfortably.",
                },
            }, source="heuristic")

        return self._normalize_review({
            "needs_relayout": True,
            "issues": issues,
            "layout_recommendation": {
                "target_lane": "middle",
                "action": "widen_lane",
                "preferred_width_ratio": 1.3,
                "reason": "Middle lane contains a wide key visual whose internal labels need more horizontal space.",
            },
        }, source="heuristic")

    def _normalize_review(self, review: Dict[str, Any], source: str) -> Dict[str, Any]:
        if not isinstance(review, dict):
            review = {}
        recommendation = review.get("layout_recommendation") or {}
        if not isinstance(recommendation, dict):
            recommendation = {}
        recommendation.setdefault("target_lane", None)
        recommendation.setdefault("target_region", None)
        recommendation.setdefault("action", "none")
        recommendation.setdefault("preferred_width_ratio", None)
        recommendation.setdefault("reason", "")
        return {
            "source": review.get("source") or source,
            "review_available": bool(review.get("review_available", source == "vlm")),
            "degraded": bool(review.get("degraded", False)),
            "fallback": review.get("fallback"),
            "needs_relayout": bool(review.get("needs_relayout", False)),
            "issues": review.get("issues") if isinstance(review.get("issues"), list) else [],
            "layout_recommendation": recommendation,
            "warnings": review.get("warnings") if isinstance(review.get("warnings"), list) else [],
        }

    def _fallback_review(self, warning: str) -> Dict[str, Any]:
        log_agent_warning(self.name, warning)
        review = self._normalize_review({}, source="fallback")
        review["review_available"] = False
        review["degraded"] = True
        review["fallback"] = "deterministic_visual_heuristic"
        review["warnings"].append(warning)
        return review

    def _lane_map(self, state: PosterState) -> Dict[str, Dict[str, float]]:
        template = state.get("layout_template_metadata") or {}
        return {lane["id"]: lane for lane in template.get("lanes", [])}

    def _infer_lane(self, element: Dict[str, Any], lane_map: Dict[str, Dict[str, float]]) -> Optional[str]:
        center_x = float(element.get("x", 0.0) or 0.0) + float(element.get("width", 0.0) or 0.0) / 2
        for lane_id, lane in lane_map.items():
            if float(lane["x"]) <= center_x <= float(lane["x"]) + float(lane["w"]):
                return lane_id
        return None

    def _safe_aspect(self, element: Dict[str, Any]) -> float:
        height = float(element.get("height", 0.0) or 0.0)
        if height <= 0:
            return 0.0
        return float(element.get("width", 0.0) or 0.0) / height

    def _caption_for_visual(self, state: PosterState, element: Dict[str, Any]) -> str:
        visual_id = element.get("visual_id")
        visual_assets = state.get("visual_assets") or {}
        if visual_id in visual_assets:
            return str(visual_assets[visual_id].get("caption") or "")
        return ""

    def _asset_for_visual(self, state: PosterState, element: Dict[str, Any]) -> Dict[str, Any]:
        visual_id = element.get("visual_id")
        source = dict((state.get("visual_assets") or {}).get(visual_id, {}) or {})
        resolved = (state.get("resolved_visual_assets") or {}).get(element.get("slot_id") or element.get("id"), {}) or {}
        for key in ("asset_type", "source_path", "resolved_path", "caption"):
            if not source.get(key) and resolved.get(key):
                source[key] = resolved[key]
        return source

    def _source_pixel_density(
        self,
        asset: Dict[str, Any],
        width_inches: float,
        height_inches: float,
    ) -> Optional[float]:
        source_path = asset.get("source_path") or asset.get("resolved_path")
        if not source_path or width_inches <= 0 or height_inches <= 0 or not Path(str(source_path)).exists():
            return None
        try:
            with Image.open(str(source_path)) as image:
                horizontal_ppi = image.width / width_inches
                vertical_ppi = image.height / height_inches
            return min(horizontal_ppi, vertical_ppi)
        except Exception:
            return None

    def _save_outputs(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "visual_legibility_review.json", "w", encoding="utf-8") as f:
            json.dump(state.get("visual_legibility_review", {}), f, indent=2)


def visual_legibility_reviewer_node(state: PosterState) -> Dict[str, Any]:
    result = VisualLegibilityReviewer()(state)
    return {
        **state,
        "visual_legibility_review": result.get("visual_legibility_review"),
        "adaptive_relayout_required": result.get("adaptive_relayout_required", False),
        "template_repair_required": result.get("template_repair_required", False),
        "template_repair_decision": result.get("template_repair_decision"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
