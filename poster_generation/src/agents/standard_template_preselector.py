"""Early standard-template selection.

This node runs before keypoint selection so template-first capacity planning can
use the selected standard template instead of waiting for the curator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config.poster_config import load_config
from src.layout.template_selector import TemplateSelector
from src.state.poster_state import PosterState
from src.template_extraction.block_template_registry import get_block_template_info, is_block_template_id
from src.tools.layout_api import LayoutTemplates
from utils.src.logging_utils import log_agent_info, log_agent_success


class StandardTemplatePreselector:
    def __init__(self):
        self.name = "standard_template_preselector"
        self.config = load_config()
        self.policy = self.config.get("standard_template_policy", {})

    def __call__(self, state: PosterState) -> PosterState:
        requested = str(state.get("layout_template") or "auto")
        if requested not in {"auto", "adaptive_auto"} and not self._is_standard_manual(requested):
            state["current_agent"] = self.name
            return state

        selector = TemplateSelector(self.config)
        selection_report = selector.select(
            state=state,
            structured_sections=state.get("structured_sections") or {},
            classified_visuals=state.get("classified_visuals") or {},
            visual_assets=state.get("visual_assets") or {},
        )
        selected = str(selection_report.get("selected_template") or requested)
        log_agent_info(self.name, f"preselected template ({selection_report.get('selection_mode')}): {selected}")

        if is_block_template_id(selected):
            self._apply_block_template_dimensions(state, selected, selection_report)
            state["enable_visual_legibility_review"] = True
            state["enable_vlm_layout_review"] = True
            state["enable_block_vlm_review"] = True

        state["resolved_layout_template"] = selected
        state["layout_template_metadata"] = self._resolve_layout(state, selected)
        state["template_selection_report"] = selection_report
        state["standard_template_selection_report"] = selection_report
        self._update_poster_variant(state, selected, selection_report)
        state["current_agent"] = self.name
        self._save_report(state, selection_report)
        log_agent_success(self.name, f"selected {selected}")
        return state

    def _is_standard_manual(self, requested: str) -> bool:
        return requested in set(self.policy.get("auto_templates") or [])

    def _apply_block_template_dimensions(
        self,
        state: PosterState,
        selected: str,
        selection_report: Dict[str, Any],
    ) -> None:
        info = get_block_template_info(selected) or {}
        size = info.get("recommended_canvas_size") or {}
        try:
            width = float(size.get("width"))
            height = float(size.get("height"))
        except (TypeError, ValueError):
            return
        if width <= 0 or height <= 0:
            return
        if state.get("layout_template") == "auto":
            state["poster_width"] = width
            state["poster_height"] = height
            selection_report["auto_canvas_size"] = {"width": width, "height": height}

    def _resolve_layout(self, state: PosterState, selected: str) -> Dict[str, Any]:
        header_height = (
            (float(state.get("poster_height") or 0.0) - 2 * self.config["layout"]["poster_margin"])
            * self.config["layout"]["title_height_fraction"]
        )
        return LayoutTemplates(
            state["poster_width"],
            state["poster_height"],
            margin=self.config["layout"]["poster_margin"],
            col_gap=self.config["layout"]["column_spacing"],
        ).get_template(selected, header_height=header_height)

    def _update_poster_variant(self, state: PosterState, selected: str, selection_report: Dict[str, Any]) -> None:
        variant = dict(state.get("poster_variant") or {})
        variant.update(
            {
                "requested_template": state.get("layout_template"),
                "resolved_template": selected,
                "selection_mode": selection_report.get("selection_mode"),
                "style_profile": state.get("poster_style_preset"),
                "visual_density": state.get("visual_density"),
                "generated_teaser": {"enabled": bool(state.get("enable_generated_teaser", False))},
                "generated_background": {
                    "enabled": bool(state.get("enable_generated_background", False)),
                    "palette": state.get("background_palette"),
                    "style": state.get("background_style"),
                },
            }
        )
        if state.get("layout_template") == "auto" and selected == "cluster_43_landscape":
            variant["variant_id"] = "default_standard"
        state["poster_variant"] = variant

    def _save_report(self, state: PosterState, report: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "standard_template_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(output_dir / "poster_variant.json", "w", encoding="utf-8") as f:
            json.dump(state.get("poster_variant") or {}, f, indent=2)


def standard_template_preselector_node(state: PosterState) -> Dict[str, Any]:
    result = StandardTemplatePreselector()(state)
    return {
        **state,
        "resolved_layout_template": result.get("resolved_layout_template"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "template_selection_report": result.get("template_selection_report"),
        "standard_template_selection_report": result.get("standard_template_selection_report"),
        "poster_variant": result.get("poster_variant"),
        "poster_width": result.get("poster_width", state.get("poster_width")),
        "poster_height": result.get("poster_height", state.get("poster_height")),
        "enable_visual_legibility_review": result.get("enable_visual_legibility_review", False),
        "enable_vlm_layout_review": result.get("enable_vlm_layout_review", False),
        "enable_block_vlm_review": result.get("enable_block_vlm_review", False),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
