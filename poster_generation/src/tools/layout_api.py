from typing import Dict, Any, List, Optional

from src.template_extraction.registry import (
    list_extracted_template_ids,
    load_extracted_template,
    scale_template_to_canvas,
)
from src.template_extraction.block_template_registry import (
    is_block_template_id,
    list_block_template_ids,
    load_block_template_layout,
)


SEMANTIC_LANES = ["left", "middle", "right"]


class LayoutTemplates:
    """
    Template generator for PosterMELD layouts.

    Each template returns three semantic lanes (`left`, `middle`, `right`) so the
    curator and balancer can continue reasoning in the same vocabulary while the
    physical geometry changes per template.
    """

    SUPPORTED_TEMPLATES = {
        "adaptive_auto",
        "three_column_postergen",
        "single_column_vertical",
        "two_plus_one_mixed",
        "one_plus_two_mixed",
        "adaptive_three_column",
    }

    @classmethod
    def available_template_names(cls) -> List[str]:
        return sorted(cls.SUPPORTED_TEMPLATES | set(list_extracted_template_ids()) | set(list_block_template_ids()))

    @classmethod
    def all_cli_template_choices(cls) -> List[str]:
        return ["auto"] + cls.available_template_names()

    def __init__(self, page_width: float, page_height: float, margin: float = 1.0, col_gap: float = 1.0):
        self.W = page_width
        self.H = page_height
        self.margin = margin
        self.col_gap = col_gap
        self.usable_w = self.W - 2 * self.margin
        self.usable_h = self.H - 2 * self.margin

    def generate_three_column_postergen(self, header_height: float = 6.0) -> Dict[str, Any]:
        """Classic equal-width three-column layout."""
        col_w = (self.usable_w - 2 * self.col_gap) / 3
        col_y = self.margin + header_height
        col_h = self.usable_h - header_height

        lanes = []
        for i, lane_id in enumerate(SEMANTIC_LANES):
            lanes.append({
                "id": lane_id,
                "x": self.margin + i * (col_w + self.col_gap),
                "y": col_y,
                "w": col_w,
                "h": col_h,
            })

        return self._package_layout("three_column_postergen", header_height, lanes)

    def generate_adaptive_three_column(
        self,
        header_height: float = 6.0,
        width_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Three semantic lanes with data-driven width ratios."""
        ratios = self._normalize_width_ratios(width_ratios)
        lane_y = self.margin + header_height
        lane_h = self.usable_h - header_height
        available_w = self.usable_w - 2 * self.col_gap
        total_ratio = sum(ratios[lane_id] for lane_id in SEMANTIC_LANES)

        lanes = []
        current_x = self.margin
        for lane_id in SEMANTIC_LANES:
            width = available_w * ratios[lane_id] / total_ratio
            lanes.append({
                "id": lane_id,
                "x": current_x,
                "y": lane_y,
                "w": width,
                "h": lane_h,
                "width_ratio": ratios[lane_id],
            })
            current_x += width + self.col_gap

        layout = self._package_layout("adaptive_three_column", header_height, lanes)
        layout["lane_width_ratios"] = ratios
        return layout

    def generate_single_column_vertical(self, header_height: float = 6.0) -> Dict[str, Any]:
        """
        Single-column vertical layout.

        The entire poster body becomes one full-width column, split into three
        vertically stacked semantic lanes to preserve the existing logical flow.
        """
        lane_h = (self.usable_h - header_height - 2 * self.col_gap) / 3
        lane_y = self.margin + header_height

        lanes = []
        for i, lane_id in enumerate(SEMANTIC_LANES):
            lanes.append({
                "id": lane_id,
                "x": self.margin,
                "y": lane_y + i * (lane_h + self.col_gap),
                "w": self.usable_w,
                "h": lane_h,
            })

        return self._package_layout("single_column_vertical", header_height, lanes)

    def generate_two_plus_one_mixed(self, header_height: float = 6.0) -> Dict[str, Any]:
        """
        Mixed layout with two narrow lanes on the left and one wide lane on the right.

        Geometry ratio: 1 : 1 : 2
        """
        lane_y = self.margin + header_height
        lane_h = self.usable_h - header_height
        unit_w = (self.usable_w - 2 * self.col_gap) / 4
        widths = [unit_w, unit_w, unit_w * 2]

        lanes = []
        current_x = self.margin
        for lane_id, width in zip(SEMANTIC_LANES, widths):
            lanes.append({
                "id": lane_id,
                "x": current_x,
                "y": lane_y,
                "w": width,
                "h": lane_h,
            })
            current_x += width + self.col_gap

        return self._package_layout("two_plus_one_mixed", header_height, lanes)

    def generate_one_plus_two_mixed(self, header_height: float = 6.0) -> Dict[str, Any]:
        """
        Mixed layout with one wide lane on the left and two narrow lanes on the right.

        Geometry ratio: 2 : 1 : 1
        """
        lane_y = self.margin + header_height
        lane_h = self.usable_h - header_height
        unit_w = (self.usable_w - 2 * self.col_gap) / 4
        widths = [unit_w * 2, unit_w, unit_w]

        lanes = []
        current_x = self.margin
        for lane_id, width in zip(SEMANTIC_LANES, widths):
            lanes.append({
                "id": lane_id,
                "x": current_x,
                "y": lane_y,
                "w": width,
                "h": lane_h,
            })
            current_x += width + self.col_gap

        return self._package_layout("one_plus_two_mixed", header_height, lanes)

    def generate_two_column_horizontal(self, header_height: float = 6.0) -> Dict[str, Any]:
        """
        Backward-compatible tool helper. The pipeline does not use this template
        family directly because the workflow still reasons over three semantic lanes.
        """
        return self.generate_one_plus_two_mixed(header_height=header_height)

    def generate_two_column_vertical(self, header_height: float = 6.0) -> Dict[str, Any]:
        """Backward-compatible alias for a vertically stacked body layout."""
        return self.generate_single_column_vertical(header_height=header_height)

    def get_template(
        self,
        template_name: str,
        header_height: float = 6.0,
        width_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if is_block_template_id(template_name):
            prior_layout = load_block_template_layout(
                template_name,
                self.W,
                self.H,
                margin=self.margin,
            )
            if prior_layout:
                return prior_layout

        extracted_template = load_extracted_template(template_name)
        if extracted_template:
            return scale_template_to_canvas(
                extracted_template,
                self.W,
                self.H,
                margin=self.margin,
                col_gap=self.col_gap,
                header_height=header_height,
            )

        if template_name == "single_column_vertical":
            return self.generate_single_column_vertical(header_height=header_height)
        if template_name == "two_plus_one_mixed":
            return self.generate_two_plus_one_mixed(header_height=header_height)
        if template_name == "one_plus_two_mixed":
            return self.generate_one_plus_two_mixed(header_height=header_height)
        if template_name == "adaptive_three_column":
            return self.generate_adaptive_three_column(header_height=header_height, width_ratios=width_ratios)
        if template_name == "adaptive_auto":
            return self.generate_three_column_postergen(header_height=header_height)
        return self.generate_three_column_postergen(header_height=header_height)

    def _merge_block_template_prior(self, prior_layout: Dict[str, Any], header_height: float) -> Dict[str, Any]:
        base_template_name = self._base_layout_for_block_template(prior_layout.get("template_id") or prior_layout.get("template_name"))
        if base_template_name == "two_plus_one_mixed":
            base_layout = self.generate_two_plus_one_mixed(header_height=header_height)
        elif base_template_name == "one_plus_two_mixed":
            base_layout = self.generate_one_plus_two_mixed(header_height=header_height)
        else:
            base_layout = self.generate_three_column_postergen(header_height=header_height)

        merged = dict(base_layout)
        merged.update({
            "template_name": prior_layout.get("template_name"),
            "template_id": prior_layout.get("template_id"),
            "layout_mode": "template_prior",
            "template_prior": True,
            "base_layout_template": base_template_name,
            "header_region": prior_layout.get("header_region"),
            "regions": prior_layout.get("regions", []),
            "hero_region_id": prior_layout.get("hero_region_id"),
            "primary_regions": prior_layout.get("primary_regions", []),
            "secondary_regions": prior_layout.get("secondary_regions", []),
            "recommended_visual_anchor": prior_layout.get("recommended_visual_anchor"),
            "template_density_profile": prior_layout.get("template_density_profile"),
            "normalized_slots": prior_layout.get("normalized_slots", []),
            "content_slots": prior_layout.get("content_slots", []),
            "slot_count": prior_layout.get("slot_count", 0),
            "slot_order": prior_layout.get("slot_order", []),
            "adjacency_graph": prior_layout.get("adjacency_graph", {}),
            "slot_prominence_score": prior_layout.get("slot_prominence_score", {}),
            "style_tokens": prior_layout.get("style_tokens", {}),
            "panel_style_tokens": prior_layout.get("panel_style_tokens", {}),
            "logo_regions": prior_layout.get("logo_regions", []),
            "footer": prior_layout.get("footer"),
            "visual_width_cap": prior_layout.get("visual_width_cap"),
            "orientation": prior_layout.get("orientation"),
            "raw_num_posters": prior_layout.get("raw_num_posters"),
            "occupancy_heatmap": prior_layout.get("occupancy_heatmap"),
            "source_template_path": prior_layout.get("source_template_path"),
        })
        merged["header"]["h"] = max(base_layout["header"]["h"], prior_layout.get("header", {}).get("h", 0.0))
        return merged

    def _base_layout_for_block_template(self, template_id: Optional[str]) -> str:
        mapping = {
            "cluster_0": "three_column_postergen",
            "cluster_1": "three_column_postergen",
            "cluster_2": "three_column_postergen",
            "cluster_3": "three_column_postergen",
        }
        return mapping.get(template_id or "", "three_column_postergen")

    def _normalize_width_ratios(self, width_ratios: Optional[Dict[str, float]]) -> Dict[str, float]:
        ratios = {lane_id: 1.0 for lane_id in SEMANTIC_LANES}
        if isinstance(width_ratios, dict):
            for lane_id in SEMANTIC_LANES:
                try:
                    value = float(width_ratios.get(lane_id, ratios[lane_id]))
                except (TypeError, ValueError):
                    value = ratios[lane_id]
                ratios[lane_id] = max(0.1, value)
        return ratios

    def _package_layout(self, template_name: str, header_height: float, lanes: List[Dict[str, float]]) -> Dict[str, Any]:
        visual_width_cap = None
        if template_name == "single_column_vertical":
            visual_width_cap = min(self.usable_w * 0.20, 11.0)

        return {
            "template_name": template_name,
            "header": {
                "x": self.margin,
                "y": self.margin,
                "w": self.usable_w,
                "h": header_height,
            },
            "lanes": lanes,
            "visual_width_cap": visual_width_cap,
            # Backward-compatible alias for older call sites.
            "columns": lanes,
        }
