"""
Visual asset planning and preprocessing.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.tools.image_api import ImageTools
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success


class VisualAssetAgent:
    """Plan and resolve per-slot visual assets before rendering."""

    def __init__(self):
        self.name = "visual_asset_agent"
        self.config = load_config()
        self.refinement_config = self.config.get("visual_refinement", {})

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "planning and resolving visual assets")

        try:
            slots = self._collect_visual_slots(state)
            plan = self._build_visual_plan(slots, state)
            state["visual_plan"] = plan
            state["visual_reflow_required"] = False

            if self._requires_reflow(plan) and state.get("visual_reflow_count", 0) < self.refinement_config.get("max_reflow_count", 1):
                state["visual_reflow_required"] = True
                state["visual_reflow_count"] = state.get("visual_reflow_count", 0) + 1
                self._save_outputs(state)
                return state

            state["resolved_visual_assets"] = self._execute_plan(plan, state)
            state["current_agent"] = self.name
            self._save_outputs(state)

            log_agent_success(self.name, f"resolved {len(state['resolved_visual_assets'])} visual slots")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _collect_visual_slots(self, state: PosterState) -> List[Dict[str, Any]]:
        layout = state.get("styled_layout") or state.get("design_layout") or []
        return [element for element in layout if element.get("type") == "visual"]

    def _build_visual_plan(self, slots: List[Dict[str, Any]], state: PosterState) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        refinement_enabled = state.get("enable_visual_refinement", False)

        for element in slots:
            slot_id = element.get("slot_id") or element.get("id") or element["visual_id"]
            section_id = self._extract_section_id(element)
            target_width = float(element.get("width", 0))
            target_height = float(element.get("height", 0))
            target_aspect = (target_width / target_height) if target_width and target_height else 1.0

            action, reason, prompt = self._choose_action(element, state, target_aspect, refinement_enabled)
            plan.append({
                "slot_id": slot_id,
                "section_id": section_id,
                "current_asset_id": element.get("visual_id"),
                "action": action,
                "reason": reason,
                "prompt": prompt,
                "target_aspect": target_aspect,
                "target_size_hint": {
                    "width": target_width,
                    "height": target_height,
                },
            })

        return plan

    def _choose_action(
        self,
        element: Dict[str, Any],
        state: PosterState,
        target_aspect: float,
        refinement_enabled: bool,
    ) -> tuple[str, str, Optional[str]]:
        asset_id = element.get("visual_id")
        source_asset = self._get_source_asset(asset_id, state)
        section_id = self._extract_section_id(element)
        section_context = self._get_section_context(section_id, state)
        color_context = json.dumps(state.get("color_scheme") or {}, ensure_ascii=False)

        if not source_asset:
            prompt = (
                "Create a clean academic poster visual for this section. "
                "Do not invent numeric results or fake charts. Prefer a conceptual diagram, workflow illustration, or abstract schematic. "
                f"Section: {section_context}. Color palette: {color_context}. "
                f"Target aspect ratio: {target_aspect:.2f}."
            )
            return "generate_new", "This visual slot has no usable source asset; generate a safe conceptual fallback.", prompt

        if not refinement_enabled:
            return "crop_only", "Default pipeline resolves each visual slot into a cropped render-ready asset.", None

        asset_type = source_asset.get("asset_type", "figure")
        source_aspect = float(source_asset.get("aspect") or target_aspect or 1.0)
        aspect_delta = abs(source_aspect - target_aspect) / max(target_aspect, 0.1)
        if asset_type != "table" and aspect_delta > self.refinement_config.get("edit_aspect_mismatch_threshold", 0.35):
            caption = source_asset.get("caption", "")
            prompt = (
                "Adapt this academic figure for a poster slot without changing the scientific meaning, labels, data, or claims. "
                "Use only low-risk edits: crop empty borders, enlarge the main visual structure, improve legibility, and harmonize the background. "
                f"Caption: {caption}. Section context: {section_context}. Color palette: {color_context}. "
                f"Target aspect ratio: {target_aspect:.2f}."
            )
            return "edit", "Figure aspect differs from the poster slot; use low-risk poster adaptation.", prompt

        return "crop_only", "Visual refinement is enabled, but crop-only is the safest valid operation for this slot.", None

    def _execute_plan(self, plan: List[Dict[str, Any]], state: PosterState) -> Dict[str, Dict[str, Any]]:
        resolved_visual_assets: Dict[str, Dict[str, Any]] = {}
        image_tools = ImageTools()
        output_assets_dir = Path(state["output_dir"]) / "assets" / "resolved"
        output_assets_dir.mkdir(parents=True, exist_ok=True)
        dpi = self.refinement_config.get("dpi", 300)

        for step in plan:
            asset_id = step.get("current_asset_id")
            slot_id = step["slot_id"]
            source_asset = self._get_source_asset(asset_id, state)
            source_path = source_asset.get("source_path") if source_asset else None
            output_path = str(output_assets_dir / f"{slot_id}.png")
            target_width = max(1, int(step["target_size_hint"]["width"] * dpi))
            target_height = max(1, int(step["target_size_hint"]["height"] * dpi))

            resolved_path = source_path or output_path
            provenance = source_asset.get("provenance", "paper_extracted") if source_asset else "generated"
            action = step["action"]

            if action == "crop_only":
                if not source_path:
                    raise ValueError(f"missing source path for crop-only visual slot '{slot_id}'")
                if source_asset.get("asset_type") == "table":
                    resolved_path = image_tools.fit_and_resize(
                        source_path,
                        target_width,
                        target_height,
                        output_path,
                    )
                    provenance = "fit_resized"
                else:
                    resolved_path = image_tools.crop_and_resize(
                        source_path,
                        target_width,
                        target_height,
                        output_path,
                    )
                    provenance = "cropped"
            elif action == "edit":
                if not source_path:
                    raise ValueError(f"missing source path for edit visual slot '{slot_id}'")
                resolved_path = image_tools.edit_image(
                    source_path,
                    step.get("prompt") or "",
                    output_path,
                )
                provenance = "edited"
            elif action in {"generate_new", "add_new"}:
                resolved_path = image_tools.generate_image(
                    step.get("prompt") or "",
                    width=target_width,
                    height=target_height,
                    output_path=output_path,
                )
                provenance = "generated"
            elif action == "drop":
                continue

            resolved_visual_assets[slot_id] = {
                "slot_id": slot_id,
                "asset_id": asset_id or f"generated_{slot_id}",
                "asset_type": source_asset.get("asset_type", "generated") if source_asset else "generated",
                "source_path": source_path,
                "resolved_path": resolved_path,
                "caption": source_asset.get("caption", "") if source_asset else step.get("reason", ""),
                "aspect": source_asset.get("aspect", step.get("target_aspect", 1.0)) if source_asset else step.get("target_aspect", 1.0),
                "provenance": provenance,
            }

        return resolved_visual_assets

    def _get_source_asset(self, asset_id: Optional[str], state: PosterState) -> Optional[Dict[str, Any]]:
        if not asset_id:
            return None
        visual_assets = state.get("visual_assets") or {}
        return visual_assets.get(asset_id)

    def _requires_reflow(self, plan: List[Dict[str, Any]]) -> bool:
        return any(step["action"] in {"add_new", "drop"} for step in plan)

    def _extract_section_id(self, element: Dict[str, Any]) -> str:
        slot_id = element.get("slot_id") or element.get("id", "")
        visual_id = element.get("visual_id", "")
        suffix = f"_{visual_id}"
        if slot_id.endswith(suffix):
            return slot_id[: -len(suffix)]
        return slot_id

    def _get_section_context(self, section_id: str, state: PosterState) -> str:
        story_board = state.get("story_board") or {}
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        for section in sections:
            if section.get("section_id") == section_id or section.get("id") == section_id:
                title = section.get("section_title") or section.get("title") or section_id
                text_content = section.get("text_content") or []
                if isinstance(text_content, list):
                    text_content = " ".join(str(item) for item in text_content if str(item).strip())
                content = section.get("content") or text_content or section.get("key_points") or section.get("summary") or ""
                return f"{title}: {content}"
        return section_id

    def _save_outputs(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "visual_plan.json", "w", encoding="utf-8") as f:
            json.dump(state.get("visual_plan", []), f, indent=2)

        with open(output_dir / "resolved_visual_assets.json", "w", encoding="utf-8") as f:
            json.dump(state.get("resolved_visual_assets", {}), f, indent=2)


def visual_asset_agent_node(state: PosterState) -> Dict[str, Any]:
    result = VisualAssetAgent()(state)
    return {
        **state,
        "visual_plan": result.get("visual_plan"),
        "resolved_visual_assets": result.get("resolved_visual_assets"),
        "visual_reflow_required": result.get("visual_reflow_required", False),
        "visual_reflow_count": result.get("visual_reflow_count", 0),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
