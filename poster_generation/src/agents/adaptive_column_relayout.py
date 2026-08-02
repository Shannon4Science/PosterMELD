"""
One-pass adaptive three-column width decision.

The node converts a visual legibility diagnosis into a constrained template
change. It does not move sections across lanes; it only changes lane widths and
lets the existing layout pipeline regenerate geometry.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_warning


class AdaptiveColumnRelayoutAgent:
    def __init__(self):
        self.name = "adaptive_column_relayout"
        self.config = load_config()
        self.adaptive_config = self.config.get("adaptive_column_width", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_adaptive_column_width", False):
            return state
        if not state.get("adaptive_relayout_required", False):
            return state

        max_relayout_count = int(self.adaptive_config.get("max_relayout_count", 1))
        if state.get("adaptive_relayout_count", 0) >= max_relayout_count:
            log_agent_warning(self.name, "adaptive relayout skipped because max relayout count was reached")
            state["adaptive_relayout_required"] = False
            return state

        log_agent_info(self.name, "applying constrained adaptive column-width decision")
        target_lane = self._target_lane(state)
        ratios = self._ratios_for_lane(target_lane)
        decision = {
            "action": "adaptive_three_column_relayout",
            "target_lane": target_lane,
            "lane_width_ratios": ratios,
            "previous_template": state.get("resolved_layout_template") or state.get("layout_template"),
            "new_template": "adaptive_three_column",
            "reason": self._reason(state),
            "max_relayout_count": max_relayout_count,
            "relayout_index": state.get("adaptive_relayout_count", 0) + 1,
        }

        state["layout_template"] = "adaptive_three_column"
        state["resolved_layout_template"] = "adaptive_three_column"
        state["adaptive_lane_widths"] = ratios
        state["adaptive_layout_decision"] = decision
        state["adaptive_relayout_count"] = state.get("adaptive_relayout_count", 0) + 1
        state["adaptive_relayout_required"] = False
        self._clear_downstream_outputs(state)
        self._save_decision(state)
        state["current_agent"] = self.name
        log_agent_success(self.name, f"adaptive lane ratios selected: {ratios}")
        return state

    def _target_lane(self, state: PosterState) -> str:
        review = state.get("visual_legibility_review") or {}
        recommendation = review.get("layout_recommendation") or {}
        target_lane = recommendation.get("target_lane")
        if target_lane in {"left", "middle", "right"}:
            return target_lane

        for issue in review.get("issues") or []:
            lane_id = issue.get("lane_id")
            if lane_id in {"left", "middle", "right"}:
                return lane_id
        return "middle"

    def _ratios_for_lane(self, target_lane: str) -> Dict[str, float]:
        defaults = self.adaptive_config.get("default_ratios", {})
        ratios = defaults.get(target_lane) or defaults.get("middle") or {
            "left": 0.85,
            "middle": 1.30,
            "right": 0.85,
        }
        min_ratio = float(self.adaptive_config.get("min_ratio", 0.75))
        max_ratio = float(self.adaptive_config.get("max_ratio", 1.40))
        return {
            lane_id: max(min_ratio, min(max_ratio, float(ratios.get(lane_id, 1.0))))
            for lane_id in ["left", "middle", "right"]
        }

    def _reason(self, state: PosterState) -> str:
        review = state.get("visual_legibility_review") or {}
        recommendation = review.get("layout_recommendation") or {}
        if recommendation.get("reason"):
            return str(recommendation["reason"])
        issues = review.get("issues") or []
        if issues:
            return str(issues[0].get("description") or "A key visual needs more readable width.")
        return "A key visual needs more readable width."

    def _clear_downstream_outputs(self, state: PosterState):
        for key in [
            "initial_layout_data",
            "column_analysis",
            "optimized_story_board",
            "optimized_column_assignment",
            "balancer_decisions",
            "final_column_analysis",
            "design_layout",
            "styled_layout",
            "visual_plan",
            "resolved_visual_assets",
            "poster_preview_path",
            "pptx_output_path",
            "vlm_reflow_required",
            "vlm_patch_applied",
        ]:
            if key in {"vlm_reflow_required", "vlm_patch_applied"}:
                state[key] = False
            else:
                state[key] = None

    def _save_decision(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "adaptive_layout_decision.json", "w", encoding="utf-8") as f:
            json.dump(state.get("adaptive_layout_decision", {}), f, indent=2)


def adaptive_column_relayout_node(state: PosterState) -> Dict[str, Any]:
    result = AdaptiveColumnRelayoutAgent()(state)
    return {
        **state,
        "layout_template": result.get("layout_template"),
        "resolved_layout_template": result.get("resolved_layout_template"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "adaptive_lane_widths": result.get("adaptive_lane_widths"),
        "adaptive_layout_decision": result.get("adaptive_layout_decision"),
        "adaptive_relayout_required": result.get("adaptive_relayout_required", False),
        "adaptive_relayout_count": result.get("adaptive_relayout_count", 0),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
