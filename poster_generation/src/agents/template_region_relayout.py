"""
One-pass template-prior relayout.

The cluster_* templates are portrait structure priors. A repair must stay inside
the selected template family instead of falling back to horizontal three-column
layouts, otherwise the template selection becomes meaningless.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_warning


class TemplateRegionRelayoutAgent:
    def __init__(self):
        self.name = "template_region_relayout"
        self.max_repairs = int(load_config().get("vlm_layout_review", {}).get("template_prior_max_repairs", 1))

    def __call__(self, state: PosterState) -> PosterState:
        if state.get("template_layout_mode") != "template_prior":
            return state
        if not state.get("template_repair_required", False):
            return state

        if state.get("template_repair_count", 0) >= self.max_repairs:
            log_agent_warning(self.name, "template relayout skipped because max repair count was reached")
            state["draft_status"] = "rejected"
            return state

        template_id = state.get("layout_template")
        current_resolved = state.get("resolved_layout_template") or template_id
        target_template = template_id if str(template_id).startswith("cluster_") else current_resolved
        decision = {
            "source_template": template_id,
            "previous_resolved_template": current_resolved,
            "new_resolved_template": target_template,
            "reason": self._reason(state),
            "repair_index": state.get("template_repair_count", 0) + 1,
            "repair_scope": "portrait_template_internal",
        }

        log_agent_info(self.name, f"re-rendering inside template-prior geometry '{target_template}'")
        state["resolved_layout_template"] = target_template
        state["template_layout_mode"] = "template_prior"
        state["adaptive_lane_widths"] = None
        state["template_repair_decision"] = decision
        state["template_repair_count"] = state.get("template_repair_count", 0) + 1
        state["template_repair_required"] = False
        state["draft_status"] = "pending"
        state["render_stage"] = "draft"
        state["visual_legibility_review"] = None
        state["vlm_layout_review"] = None
        state["vlm_layout_patch"] = None
        state["vlm_reflow_required"] = False
        state["vlm_patch_applied"] = False
        self._clear_downstream_outputs(state)
        self._save_decision(state)
        state["current_agent"] = self.name
        log_agent_success(self.name, f"template prior repair prepared with {target_template}")
        return state

    def _reason(self, state: PosterState) -> str:
        decision = state.get("template_repair_decision") or {}
        if decision.get("reason"):
            return str(decision["reason"])
        review = state.get("vlm_layout_review") or state.get("visual_legibility_review") or {}
        issues = review.get("issues") or []
        if issues:
            return str(issues[0].get("description") or "Template-prior draft failed layout quality gate.")
        return str(state.get("draft_rejection_reason") or "Template-prior draft failed deterministic/VLM quality gate.")

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
            "slot_pressure_report",
        ]:
            state[key] = None

    def _save_decision(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "template_repair_decision.json", "w", encoding="utf-8") as f:
            json.dump(state.get("template_repair_decision", {}), f, indent=2)


def template_region_relayout_node(state: PosterState) -> Dict[str, Any]:
    result = TemplateRegionRelayoutAgent()(state)
    return {
        **state,
        "story_board": result.get("story_board"),
        "resolved_layout_template": result.get("resolved_layout_template"),
        "template_layout_mode": result.get("template_layout_mode"),
        "adaptive_lane_widths": result.get("adaptive_lane_widths"),
        "template_repair_required": result.get("template_repair_required", False),
        "template_repair_count": result.get("template_repair_count", 0),
        "template_repair_decision": result.get("template_repair_decision"),
        "draft_status": result.get("draft_status", state.get("draft_status", "pending")),
        "render_stage": result.get("render_stage", state.get("render_stage", "draft")),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
