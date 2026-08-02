"""
precise layout generation using css box model
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.layout.text_height_measurement import measure_text_height
from src.config.poster_config import load_config
from src.tools.layout_api import LayoutTemplates, SEMANTIC_LANES
from src.utils.text_cleanup import normalize_text_for_poster, normalize_title_for_poster, repair_possessive_title_apostrophe
from src.utils.style_options import resolve_poster_visual_style, resolve_typography_config
from src.utils.visual_footprint import enforce_visual_footprint

class LayoutAgent:
    """creates optimized layouts using css box model"""
    
    def __init__(self):
        self.name = "layout_agent"
        self.config = load_config()
        self.poster_margin = self.config["layout"]["poster_margin"]
        self.column_spacing = self.config["layout"]["column_spacing"]
        self.title_height_fraction = self.config["layout"]["title_height_fraction"]
        self.title_font_family = self.config["typography"]["fonts"]["title"]
        self.authors_font_family = self.config["typography"]["fonts"]["authors"]
        self.section_title_font_family = self.config["typography"]["fonts"]["section_title"]
        self.body_text_font_family = self.config["typography"]["fonts"]["body_text"]
        self.visual_style_config = self.config.get("poster_visual_style", {})
        # layout constants
        self.layout_constants = self.config["layout_constants"]
        self.column_balancing = self.config["column_balancing"]
        
        # debug configuration
        self.show_debug_borders = self.config["rendering"]["debug_borders"]  ## enable to see section boundaries for debugging

    def _apply_state_style(self, state: PosterState) -> None:
        self.visual_style_config = resolve_poster_visual_style(state, self.config)
        typography = resolve_typography_config(state, self.config)
        fonts = typography.get("fonts", {})
        self.title_font_family = fonts.get("title", self.title_font_family)
        self.authors_font_family = fonts.get("authors", self.authors_font_family)
        self.section_title_font_family = fonts.get("section_title", self.section_title_font_family)
        self.body_text_font_family = fonts.get("body_text", self.body_text_font_family)
    
    def _resolve_template_layout(self, state: PosterState) -> Dict[str, Any]:
        adaptive_widths = state.get("adaptive_lane_widths")
        if adaptive_widths:
            requested_template = "adaptive_three_column"
        else:
            requested_template = state.get("resolved_layout_template") or state.get("layout_template", "auto")
        if requested_template == "auto":
            requested_template = "three_column_postergen"
        poster_margin = self.poster_margin
        column_spacing = self.column_spacing
        if self._use_adaptive_dense_layout(state, requested_template):
            dense = self.config.get("adaptive_auto_dense_layout", {})
            poster_margin = float(dense.get("poster_margin", poster_margin))
            column_spacing = float(dense.get("column_spacing", column_spacing))
        effective_height = state["poster_height"] - 2 * poster_margin
        title_region_height = effective_height * self.title_height_fraction

        template_layout = LayoutTemplates(
            state["poster_width"],
            state["poster_height"],
            margin=poster_margin,
            col_gap=column_spacing,
        ).get_template(requested_template, header_height=title_region_height, width_ratios=adaptive_widths)

        state["resolved_layout_template"] = template_layout["template_name"]
        state["layout_template_metadata"] = template_layout
        return template_layout

    def _use_adaptive_dense_layout(self, state: PosterState, requested_template: str) -> bool:
        dense = self.config.get("adaptive_auto_dense_layout", {})
        if not dense.get("enabled", True):
            return False
        if state.get("template_layout_mode") == "template_prior":
            return False
        template_name = str(requested_template or "")
        return template_name in {
            "adaptive_auto",
            "three_column_postergen",
            "two_plus_one_mixed",
            "one_plus_two_mixed",
            "adaptive_three_column",
        }

    def _lane_order(self, template_layout: Dict[str, Any]) -> List[str]:
        return [lane["id"] for lane in template_layout.get("lanes", [])]

    def _lane_map(self, template_layout: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        return {lane["id"]: lane for lane in template_layout.get("lanes", [])}

    def _vertical_priority_rank(self, priority: str) -> int:
        order = {"top": 0, "middle": 1, "bottom": 2}
        return order.get(priority, 1)

    def _get_visual_width_for_lane(self, lane_width: float, state: PosterState) -> float:
        text_padding = self.config["layout"]["text_padding"]["left_right"]
        visual_width = lane_width - (2 * text_padding)
        template_layout = state.get("layout_template_metadata") or {}
        visual_width_cap = template_layout.get("visual_width_cap")
        if visual_width_cap:
            visual_width = min(visual_width, visual_width_cap)
        return max(visual_width, 0.1)

    def __call__(self, state: PosterState, mode: str = "initial") -> PosterState:
        if mode == "initial":
            return self._generate_initial_layout(state)
        else:
            return self._generate_final_layout(state)
    
    def _generate_initial_layout(self, state: PosterState) -> PosterState:
        """generate initial layout without optimization - direct curator mapping"""
        log_agent_info(self.name, "generating initial layout from story board")
        self._apply_state_style(state)
        
        try:
            story_board = state.get("story_board")
            if not story_board:
                raise ValueError("missing story_board from curator")
            template_layout = self._resolve_template_layout(state)
            
            # organize sections from story board for layout creation
            sections = story_board["spatial_content_plan"]["sections"]
            optimized_layout = self._organize_sections_by_column(sections, template_layout)
            
            # create layout directly from curator output - no optimization
            layout_data = self._create_precise_layout(
                story_board=story_board,
                optimized_layout=optimized_layout,
                state=state
            )
            
            # generate column analysis for balancer
            column_analysis = self._generate_column_analysis(layout_data, state)
            
            state["initial_layout_data"] = layout_data
            state["column_analysis"] = column_analysis
            state["current_agent"] = self.name
            
            self._save_initial_layout(state)
            
            log_agent_success(self.name, "initial layout generated")
            return state
            
        except Exception as e:
            log_agent_error(self.name, f"initial layout error: {e}")
            state["errors"].append(f"{self.name}: {e}")
            return state
    
    def _generate_final_layout(self, state: PosterState) -> PosterState:
        """generate final layout from optimized story board"""
        log_agent_info(self.name, "generating final layout from optimized story board")
        self._apply_state_style(state)
        
        try:
            optimized_story_board = state.get("optimized_story_board")
            if not optimized_story_board:
                raise ValueError("missing optimized_story_board from balancer")
            template_layout = self._resolve_template_layout(state)
            
            # organize sections from optimized story board
            sections = optimized_story_board["spatial_content_plan"]["sections"]
            organized_layout = self._organize_sections_by_column(sections, template_layout)
            
            # create final layout from optimized story board
            layout_data = self._create_precise_layout(
                story_board=optimized_story_board,
                optimized_layout=organized_layout,
                state=state
            )
            
            # generate final column analysis to verify optimization success
            final_column_analysis = self._generate_column_analysis(layout_data, state)
            
            # validate final layout
            layout_validation = self._validate_precise_layout(layout_data, state["poster_width"], state["poster_height"])
            state["layout_validation"] = layout_validation
            if not layout_validation.get("valid", True):
                if template_layout.get("layout_mode") == "template_prior":
                    log_agent_warning(
                        self.name,
                        "pre-micro template-prior layout validation reported issues; "
                        "micro-layout remains the blocking geometry gate",
                    )
                else:
                    state["draft_status"] = "rejected"
                    state.setdefault("errors", []).append(f"{self.name}: final layout validation failed: {layout_validation.get('issues', [])}")
            
            state["design_layout"] = layout_data
            state["final_column_analysis"] = final_column_analysis
            state["optimized_column_assignment"] = organized_layout["optimized_layout"]["column_assignments"]
            state["current_agent"] = self.name
            
            self._save_final_layout(state)
            
            log_agent_success(self.name, "final layout complete")
            return state
            
        except Exception as e:
            log_agent_error(self.name, f"final layout error: {e}")
            state["errors"].append(f"{self.name}: {e}")
            return state
    
    def _optimize_column_distribution(self, story_board: Dict, poster_width: int, poster_height: int, config, state) -> Dict:
        """rule-based column distribution for optimal space utilization"""
        log_agent_info(self.name, "optimizing column distribution")
        
        # calculate available space
        effective_height = poster_height - 2 * self.poster_margin  # total height minus margins
        title_region_height = effective_height * self.title_height_fraction  # 18% of effective height
        template_layout = self._resolve_template_layout(state)
        lane_map = self._lane_map(template_layout)
        available_height = min(lane["h"] for lane in lane_map.values())
        
        # handle new spatial content plan format
        if "spatial_content_plan" in story_board:
            sections = story_board["spatial_content_plan"]["sections"]
            column_distribution = story_board.get("column_distribution", {})
        else:
            # fallback to old format
            sections = story_board.get("story_board", {}).get("sections", [])
            column_distribution = {}
        
        # create precise spatial layout using css-like calculations
        optimized_layout = self._create_spatial_layout(
            sections, column_distribution, lane_map, state
        )
        
        log_agent_success(self.name, f"created rule-based optimized layout")
        
        return {
            "optimized_layout": {
                "column_assignments": optimized_layout,
                "strategy": "rule_based_intelligent",
                "space_utilization_target": 0.90,
                "column_dimensions": {
                    "width": min(lane["w"] for lane in lane_map.values()),
                    "height": available_height
                }
            }
        }
    
    def _apply_adjustments(self, adjustments: Dict):
        """apply critic-requested adjustments to layout parameters"""
        if adjustments.get("increase_spacing"):
            log_agent_info(self.name, "increased spacing: adjusting layout constants")
        
        if adjustments.get("reduce_sizes"):
            log_agent_info(self.name, "reduced spacing: adjusting layout constants")
        
        if adjustments.get("poster_margin"):
            self.poster_margin = adjustments["poster_margin"]
        
        if adjustments.get("column_spacing"):
            self.column_spacing = adjustments["column_spacing"]

    def _save_initial_layout(self, state: PosterState):
        """save initial layout data"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "initial_layout_data.json", "w", encoding='utf-8') as f:
            json.dump(state.get("initial_layout_data", []), f, indent=2)
        
        with open(output_dir / "column_analysis.json", "w", encoding='utf-8') as f:
            json.dump(state.get("column_analysis", {}), f, indent=2)
    
    def _save_final_layout(self, state: PosterState):
        """save final layout data"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "final_design_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("design_layout", []), f, indent=2)
        
        with open(output_dir / "optimized_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("optimized_column_assignment", {}), f, indent=2)
        
        # save final column analysis to show optimization success
        if state.get("final_column_analysis"):
            with open(output_dir / "final_column_analysis.json", "w", encoding='utf-8') as f:
                json.dump(state.get("final_column_analysis", {}), f, indent=2)
        if state.get("layout_validation"):
            with open(output_dir / "layout_validation.json", "w", encoding='utf-8') as f:
                json.dump(state.get("layout_validation", {}), f, indent=2)
    
    def _generate_column_analysis(self, layout_data: List[Dict], state: PosterState) -> Dict:
        """generate detailed column utilization analysis using exact column calculation method"""
        template_layout = self._resolve_template_layout(state)
        lane_map = self._lane_map(template_layout)
        columns = {lane_id: [] for lane_id in self._lane_order(template_layout)}

        for element in layout_data:
            if element.get("type") == "section_container":
                section_id = element.get("section_id", "")
                lane_id = section_id.split("::", 1)[0] if "::" in section_id else element.get("lane_id")
                if not lane_id or lane_id not in columns:
                    lane_id = self._match_section_container_to_lane(element, lane_map)
                columns[lane_id].append(element)
        
        # calculate utilization for each column
        column_analysis = {
            "available_height": min(lane["h"] for lane in lane_map.values()),
            "template_name": template_layout["template_name"],
            "columns": {}
        }
        
        for col_name, elements in columns.items():
            available_height = lane_map[col_name]["h"]
            if elements:
                max_bottom = max(elem["y"] + elem["height"] for elem in elements)
                min_top = min(elem["y"] for elem in elements) 
                used_height = max_bottom - min_top
            else:
                used_height = 0
            
            utilization_rate = used_height / available_height if available_height > 0 else 0
            
            status = "overflow" if utilization_rate > 1.0 else "underutilized" if utilization_rate < 0.7 else "balanced"
            
            column_analysis["columns"][col_name] = {
                "utilization_rate": utilization_rate,
                "total_height": used_height,
                "status": status,
                "available_space": max(0, available_height - used_height),
                "excess_height": max(0, used_height - available_height),
                "available_height": available_height,
                "width": lane_map[col_name]["w"],
            }
        
        return column_analysis

    def _match_section_container_to_lane(self, element: Dict, lane_map: Dict[str, Dict[str, float]]) -> str:
        element_x = element.get("x", 0)
        element_y = element.get("y", 0)
        for lane_id, lane in lane_map.items():
            within_x = lane["x"] - 0.05 <= element_x <= lane["x"] + lane["w"] + 0.05
            within_y = lane["y"] - 0.05 <= element_y <= lane["y"] + lane["h"] + 0.05
            if within_x and within_y:
                return lane_id
        return next(iter(lane_map))
    
    def _organize_sections_by_column(self, sections: List[Dict], template_layout: Dict[str, Any]) -> Dict:
        """organize sections by column assignment for layout creation"""
        columns = {lane_id: [] for lane_id in self._lane_order(template_layout)}
        
        for section in sections:
            column = section.get("column_assignment", "left")
            if column not in columns and section.get("slot_id") in columns:
                column = section["slot_id"]
            if column in columns:
                columns[column].append(section)

        for lane_sections in columns.values():
            lane_sections.sort(key=lambda section: self._vertical_priority_rank(section.get("vertical_priority", "middle")))
        
        column_assignments = []
        for idx, lane_id in enumerate(self._lane_order(template_layout)):
            column_assignments.append({
                "column_id": idx,
                "column_name": lane_id,
                "sections": columns[lane_id],
            })
        
        return {
            "optimized_layout": {
                "column_assignments": column_assignments,
                "template_name": template_layout["template_name"],
            }
        }
    
    def _create_precise_layout(self, story_board: Dict, optimized_layout: Dict, state: PosterState) -> List[Dict]:
        """create precise layout with exact positioning using measurements"""
        layout_elements = []
        is_template_prior = state.get("template_layout_mode") == "template_prior"
        
        # poster dimensions
        poster_width = state["poster_width"]
        poster_height = state["poster_height"]
        
        # calculate layout dimensions
        effective_height = poster_height - 2 * self.poster_margin
        title_region_height = effective_height * self.title_height_fraction  # 18% fixed region
        base_layout = self._resolve_template_layout(state)
        lane_map = self._lane_map(base_layout)

        layout_elements.extend(self._create_template_style_elements(base_layout, poster_width, poster_height))
        
        # add title element (still uses actual measured height, not fixed region height)
        actual_title_height = base_layout["header"]["h"] if is_template_prior else title_region_height
        title_element = self._create_title_element(state, poster_width, actual_title_height, base_layout)
        if title_element:
            layout_elements.append(title_element)
        
        # add logo elements
        logo_elements = self._create_logo_elements(state, poster_width, base_layout)
        layout_elements.extend(logo_elements)
        
        # process each column
        column_assignments = optimized_layout.get("optimized_layout", {}).get("column_assignments", [])
        highlight_section_ids = self._select_highlight_section_ids(column_assignments, state, base_layout)
        
        for column in column_assignments:
            lane_id = column.get("column_name", "left")
            lane = lane_map[lane_id]
            column_x = lane["x"]
            column_y = lane["y"]
            column_width = lane["w"]
            available_height = lane["h"]
            
            current_y = column_y
            
            # process each section in this column
            for section in column.get("sections", []):
                section_start_y = current_y
                section_elements = self._create_section_elements(
                    section, column_x, current_y, column_width, state, available_height
                )
                
                # calculate section height from actual element positions
                section_height = 0
                if section_elements:
                    # find the bottommost element
                    max_bottom = 0
                    for element in section_elements:
                        element_bottom = element["y"] + element["height"]
                        max_bottom = max(max_bottom, element_bottom)
                    section_height = max_bottom - section_start_y
                
                # create section container for layout structure
                section_container = {
                    "type": "section_container",
                    "x": column_x,
                    "y": section_start_y,
                    "width": column_width,
                    "height": section_height,
                    "section_id": section.get("section_id", "unknown"),
                    "lane_id": lane_id,
                    "slot_id": section.get("slot_id", lane_id),
                    "template_prior": is_template_prior,
                    "importance_level": section.get("importance_level", 2),  # importance level for background styling
                    "priority": 0.1
                }
                self._apply_template_panel_style(section_container, base_layout)
                self._apply_selective_block_frame_style(section_container, section, state, highlight_section_ids)
                self._apply_visual_block_panel_style(section_container, state)
                
                # add debug border only if enabled
                if self.show_debug_borders:
                    section_container["debug_border"] = True
                
                layout_elements.append(section_container)
                
                layout_elements.extend(section_elements)
                current_y += section_height + 1.0  # 1" section spacing for stability
                
                log_agent_info(self.name, f"placed section '{section.get('section_id')}' in lane={lane_id} at y={section_start_y:.2f}, height={section_height:.2f}")
        
        return layout_elements

    def _select_highlight_section_ids(
        self,
        column_assignments: List[Dict[str, Any]],
        state: PosterState,
        template_layout: Dict[str, Any],
    ) -> Dict[str, int]:
        """Pick the 1-2 most important sections for subtle panel backgrounds."""
        highlight_config = self.config.get("selective_block_backgrounds", {})
        if not highlight_config.get("enabled", False):
            return {}
        if template_layout.get("layout_mode") != "template_prior":
            return {}

        max_blocks = int(highlight_config.get("max_highlight_blocks", 2))
        if max_blocks <= 0:
            return {}

        lane_area = {
            lane["id"]: float(lane.get("w", 0.0)) * float(lane.get("h", 0.0))
            for lane in template_layout.get("lanes", [])
        }
        largest_area = max(lane_area.values(), default=0.0)
        role_priority = highlight_config.get("role_priority", {})
        candidates: List[Tuple[float, str]] = []

        for column in column_assignments:
            lane_id = str(column.get("column_name", ""))
            for section in column.get("sections", []):
                section_id = str(section.get("section_id", ""))
                if not section_id:
                    continue
                role = self._section_role(section)
                score = float(role_priority.get(role, role_priority.get(section_id, 50)))
                score += max(0, 3 - int(section.get("importance_level", 2))) * 18
                score += min(len(section.get("visual_assets") or []), 2) * float(highlight_config.get("visual_bonus", 8))
                if largest_area and lane_area.get(lane_id, 0.0) >= largest_area * 0.85:
                    score += float(highlight_config.get("large_slot_bonus", 8))
                candidates.append((score, section_id))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected: Dict[str, int] = {}
        for _, section_id in candidates:
            if section_id in selected:
                continue
            selected[section_id] = len(selected)
            if len(selected) >= max_blocks:
                break
        return selected

    def _apply_selective_highlight_panel(
        self,
        section_container: Dict[str, Any],
        section: Dict[str, Any],
        state: PosterState,
        selected: Dict[str, int],
    ) -> None:
        rank = selected.get(section_container["section_id"], 0)
        highlight_config = self.config.get("selective_block_backgrounds", {})

        section_container["highlight_panel"] = True
        section_container["highlight_rank"] = rank
        section_container["highlight_role"] = self._section_role(section)
        section_container["fill_color"] = highlight_config.get("primary_fill_color", "#EEF0F2")
        border_color = str(highlight_config.get("primary_border_color") or "").strip()
        border_width = float(highlight_config.get("primary_border_width", 0) or 0)
        if border_color and border_width > 0:
            section_container["border_color"] = border_color
            section_container["border_width"] = border_width
            section_container["border_style"] = highlight_config.get("primary_border_style", "solid")
        else:
            section_container.pop("border_color", None)
            section_container.pop("border_width", None)
            section_container.pop("border_style", None)
        section_container["priority"] = min(float(section_container.get("priority", 0.1)), 0.08)

    def _apply_selective_block_frame_style(
        self,
        section_container: Dict[str, Any],
        section: Dict[str, Any],
        state: PosterState,
        selected: Dict[str, int],
    ) -> None:
        highlight_config = self.config.get("selective_block_backgrounds", {})
        if not highlight_config.get("enabled", False):
            return
        if not section_container.get("template_prior"):
            return

        section_id = str(section_container.get("section_id", ""))
        if section_id in selected:
            self._apply_selective_highlight_panel(section_container, section, state, selected)
            return

        if not highlight_config.get("frame_all_blocks", False):
            return

        support = self._is_supporting_block(section)
        section_container["highlight_panel"] = False
        section_container["highlight_role"] = self._section_role(section)
        section_container.pop("fill_color", None)
        if support:
            section_container["border_color"] = highlight_config.get("support_border_color", "#D8DDE3")
            section_container["border_width"] = float(highlight_config.get("support_border_width", 0.7))
            section_container["border_style"] = highlight_config.get("support_border_style", "dashed")
        else:
            section_container["border_color"] = highlight_config.get("normal_border_color", "#D2D6DC")
            section_container["border_width"] = float(highlight_config.get("normal_border_width", 0.7))
            section_container["border_style"] = highlight_config.get("normal_border_style", "solid")
        section_container["priority"] = min(float(section_container.get("priority", 0.1)), 0.09)

    def _apply_visual_block_panel_style(self, section_container: Dict[str, Any], state: PosterState | None = None) -> None:
        panel_style = (self.visual_style_config.get("block_panel") or {})
        if not self.visual_style_config.get("enabled", False):
            return
        if not panel_style.get("enabled", False):
            return
        if not section_container.get("template_prior"):
            return

        if (state or {}).get("enable_generated_background", False):
            overlay_style = panel_style.get("generated_background_overlay") or {}
            if overlay_style.get("enabled", True):
                section_container["fill_color"] = overlay_style.get("fill_color", "#FFFFFF")
                section_container["fill_opacity"] = float(overlay_style.get("fill_opacity", 0.84))
                border_color = str(overlay_style.get("border_color") or "").strip()
                border_width = float(overlay_style.get("border_width", 0) or 0)
                if border_color and border_width > 0:
                    section_container["border_color"] = border_color
                    section_container["border_opacity"] = float(overlay_style.get("border_opacity", 0.58))
                    section_container["border_width"] = border_width
                    section_container["border_style"] = overlay_style.get("border_style", "solid")
                shadow = overlay_style.get("shadow")
                if shadow and shadow.get("enabled", True):
                    section_container["shadow"] = shadow
                else:
                    section_container.pop("shadow", None)
                section_container["background_aware_panel"] = True
                section_container["priority"] = min(float(section_container.get("priority", 0.1)), 0.07)
                return

        section_container["fill_color"] = section_container.get("fill_color") or panel_style.get(
            "fill_color",
            "#F1F2F4",
        )
        border_color = str(panel_style.get("border_color") or "").strip()
        border_width = float(panel_style.get("border_width", 0) or 0)
        if border_color and border_width > 0:
            section_container["border_color"] = border_color
            section_container["border_width"] = border_width
            section_container["border_style"] = panel_style.get("border_style", "solid")
        shadow = panel_style.get("shadow")
        if shadow and shadow.get("enabled", True):
            section_container["shadow"] = shadow
        section_container["priority"] = min(float(section_container.get("priority", 0.1)), 0.07)

    def _is_supporting_block(self, section: Dict[str, Any]) -> bool:
        highlight_config = self.config.get("selective_block_backgrounds", {})
        text = " ".join(
            str(section.get(key, ""))
            for key in ("section_id", "section_title", "content_role")
        ).lower()
        return any(
            str(keyword).lower() in text
            for keyword in highlight_config.get("support_section_keywords", [])
        )

    def _section_role(self, section: Dict[str, Any]) -> str:
        role = str(section.get("content_role") or "").strip().lower()
        if role:
            return role
        text = " ".join(
            str(section.get(key, ""))
            for key in ("section_id", "section_title", "column_assignment", "slot_id")
        ).lower()
        if "result" in text or "performance" in text or "evaluation" in text:
            return "results"
        if "method" in text or "framework" in text or "model" in text:
            return "method"
        if "problem" in text or "motivation" in text or "fail" in text:
            return "problem"
        return "overview"

    def _blend_hex(self, foreground: str, background: str = "#FFFFFF", amount: float = 0.08) -> str:
        def parse(color: str) -> Tuple[int, int, int]:
            value = str(color or "").strip().lstrip("#")
            if len(value) != 6:
                value = "FFFFFF"
            try:
                return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return (255, 255, 255)

        amount = min(max(amount, 0.0), 1.0)
        fg = parse(foreground)
        bg = parse(background)
        blended = [round(bg[i] * (1 - amount) + fg[i] * amount) for i in range(3)]
        return "#" + "".join(f"{channel:02X}" for channel in blended)

    def _create_template_style_elements(self, template_layout: Dict[str, Any], poster_width: float, poster_height: float) -> List[Dict[str, Any]]:
        style = template_layout.get("style_tokens") or {}
        if not style:
            return []

        elements: List[Dict[str, Any]] = []
        background = style.get("background")
        if background and background.upper() != "#FFFFFF":
            elements.append({
                "type": "template_background",
                "x": 0,
                "y": 0,
                "width": poster_width,
                "height": poster_height,
                "fill_color": background,
                "priority": 0.0,
            })

        header_bg = style.get("header_background")
        if header_bg and header_bg.upper() != "#FFFFFF":
            header = template_layout["header"]
            elements.append({
                "type": "template_header_background",
                "x": header["x"],
                "y": header["y"],
                "width": header["w"],
                "height": header["h"],
                "fill_color": header_bg,
                "priority": 0.01,
            })

        footer = template_layout.get("footer")
        footer_bg = style.get("footer_background")
        if footer and footer_bg and footer_bg.upper() != "#FFFFFF":
            elements.append({
                "type": "template_footer_background",
                "x": footer["x"],
                "y": footer["y"],
                "width": footer["w"],
                "height": footer["h"],
                "fill_color": footer_bg,
                "priority": 0.01,
            })

        return elements

    def _apply_template_panel_style(self, section_container: Dict[str, Any], template_layout: Dict[str, Any]) -> None:
        style = template_layout.get("style_tokens") or {}
        if not style:
            return
        if style.get("panel_fill_color"):
            section_container["fill_color"] = style["panel_fill_color"]
        if style.get("panel_border_color"):
            section_container["border_color"] = style["panel_border_color"]
            section_container["border_width"] = 1.2
    
    def _create_title_element(self, state: PosterState, poster_width: float, title_height: float, template_layout: Dict[str, Any]) -> Dict:
        """create title element with exact positioning"""
        header_plan = state.get("header_plan") or {}
        if header_plan.get("validation", {}).get("passed") and header_plan.get("title_box"):
            title_box = header_plan["title_box"]
            title = header_plan.get("title") or {}
            subtitle = header_plan.get("subtitle") or {}
            authors = header_plan.get("authors") or {}
            title_source_text = str(title.get("display_text") or title.get("text") or "")
            title_text = "\n".join(
                normalize_title_for_poster(line) or line.strip()
                for line in title_source_text.splitlines()
                if line.strip()
            ) or normalize_title_for_poster(title.get("text", "")) or "Title"
            subtitle_text = normalize_text_for_poster(subtitle.get("text", "")) if subtitle.get("text") else ""
            authors_text = normalize_text_for_poster(authors.get("text", "")) or "Authors"
            content_lines = [title_text]
            if subtitle_text:
                content_lines.append(subtitle_text)
            content_lines.append(authors_text)
            return {
                "type": "title",
                "x": title_box["x"],
                "y": title_box["y"],
                "width": title_box["w"],
                "height": title_box["h"],
                "content": "\n".join(content_lines),
                "title_text": title_text,
                "title_original_text": title.get("text", ""),
                "subtitle_text": subtitle_text,
                "authors_text": authors_text,
                "alignment": title.get("alignment", "left"),
                "font_family": title.get("font_family", self.title_font_family),
                "font_size": title.get("font_size", 100),
                "title_single_line": title.get("single_line", True),
                "title_wrap_policy": title.get("wrap_policy", "single_line"),
                "subtitle_font_size": subtitle.get("font_size", 54),
                "subtitle_single_line": subtitle.get("single_line", True),
                "subtitle_box_height": subtitle.get("box_height", 0.0),
                "title_box_height": title.get("box_height"),
                "title_to_subtitle_gap_inches": subtitle.get("top_gap_inches", 0.0),
                "author_font_size": authors.get("font_size", 72),
                "author_x": authors.get("x"),
                "author_width": authors.get("w"),
                "author_word_wrap": authors.get("word_wrap", False),
                "author_box_height": authors.get("box_height"),
                "author_top_gap_inches": authors.get(
                    "top_gap_inches",
                    self.config["typography"].get("title_author_gap_points", 16) / 72,
                ),
                "lock_header_typography": True,
                "header_route": header_plan.get("route"),
                "priority": 1.0,
            }

        title_box, _ = self._header_title_logo_boxes(state, template_layout)
        
        # extract title and authors from narrative content
        narrative = state.get("narrative_content") or {}
        meta = narrative.get("meta", {})
        poster_title = normalize_title_for_poster(meta.get("poster_title", state.get('poster_name', 'Title'))) or "Title"
        authors = normalize_text_for_poster(meta.get("authors", "Authors")) or "Authors"
        if template_layout.get("layout_mode") == "template_prior" and template_layout.get("orientation") == "portrait":
            title_font_size = 58
            author_font_size = 34
            author_top_gap_inches = self.config["typography"].get("title_author_gap_points", 16) / 72
        else:
            title_font_size = 100
            author_font_size = 72
            author_top_gap_inches = self.config["typography"].get("title_author_gap_points", 16) / 72
        
        return {
            "type": "title",
            "x": title_box["x"],
            "y": title_box["y"],
            "width": title_box["w"],
            "height": title_box["h"],
            "content": f"{poster_title}\n{authors}",
            "font_family": self.title_font_family,
            "font_size": title_font_size,
            "title_single_line": True,
            "author_font_size": author_font_size,
            "author_top_gap_inches": author_top_gap_inches,
            "priority": 1.0
        }
    
    def _create_logo_elements(self, state: PosterState, poster_width: float, template_layout: Dict[str, Any]) -> List[Dict]:
        """Build logo elements in the title-right reserved area.

        Layout logic (all cases vertically centred in the region):

          conf only          →  conf logo centred in full region
          aff only (1-2)     →  logos in a single row, centred
          aff only (3-4)     →  2×2 grid, centred
          conf + aff (1-2)   →  [aff row | divider | conf] left-to-right
          conf + aff (3-4)   →  [aff 2×2 | divider | conf] left-to-right
        """
        header_plan = state.get("header_plan") or {}
        if header_plan.get("validation", {}).get("passed") and header_plan.get("logo_elements") is not None:
            elements = []
            for element in header_plan.get("logo_elements") or []:
                element_copy = deepcopy(element)
                if element_copy.get("type") == "institution_logo":
                    logo_path = element_copy.get("image_path")
                    if not logo_path or not Path(str(logo_path)).exists():
                        continue
                elif element_copy.get("type") == "conf_logo":
                    if not (state.get("logo_path") and Path(str(state["logo_path"])).exists()):
                        continue
                elements.append(element_copy)
            return elements

        aff_logos = [
            logo for logo in (state.get("affiliation_logos") or [])
            if logo.get("logo_path") and Path(logo["logo_path"]).exists()
        ]
        manual_aff_logo = self._manual_affiliation_logo_entry(state)
        if manual_aff_logo and not any(
            Path(logo["logo_path"]).resolve() == Path(manual_aff_logo["logo_path"]).resolve()
            for logo in aff_logos
        ):
            aff_logos.insert(0, manual_aff_logo)
        logo_config = self.config.get("affiliation_logos", {})
        aff_logos = aff_logos[: logo_config.get("max_logos", 4)]

        has_conf = bool(state.get("logo_path") and Path(state["logo_path"]).exists())
        has_aff  = bool(aff_logos)

        if not has_conf and not has_aff:
            return []

        _, region = self._header_title_logo_boxes(state, template_layout)

        if has_conf and not has_aff:
            return self._layout_conf_only(state["logo_path"], region)

        if has_aff and not has_conf:
            return self._layout_aff_only(aff_logos, region, logo_config)

        # both conf + aff: split region
        return self._layout_combined(state["logo_path"], aff_logos, region, logo_config)

    def _manual_affiliation_logo_entry(self, state: PosterState) -> Dict[str, Any] | None:
        logo_path = state.get("aff_logo_path")
        if not logo_path or not Path(logo_path).exists():
            return None

        return {
            "institution": state.get("affiliation_logo_label") or "Affiliation",
            "logo_path": logo_path,
            "domain": None,
            "source": "manual",
            "aspect": self._get_image_aspect_ratio(logo_path),
        }

    # ------------------------------------------------------------------ #
    #  Sub-layouts                                                         #
    # ------------------------------------------------------------------ #

    def _layout_conf_only(self, conf_path: str, region: Dict[str, float]) -> List[Dict]:
        aspect = self._get_image_aspect_ratio(conf_path)
        logo_h = min(region["h"] * 0.88, region["w"] / max(aspect, 0.1))
        logo_w = logo_h * aspect
        return [{
            "type": "conf_logo",
            "x": region["x"] + (region["w"] - logo_w) / 2,
            "y": region["y"] + (region["h"] - logo_h) / 2,
            "width": logo_w,
            "height": logo_h,
            "priority": 0.9,
        }]

    def _layout_aff_only(
        self,
        aff_logos: List[Dict[str, Any]],
        region: Dict[str, float],
        logo_config: Dict[str, Any],
    ) -> List[Dict]:
        return self._aff_grid_elements(aff_logos, region, logo_config)

    def _layout_combined(
        self,
        conf_path: str,
        aff_logos: List[Dict[str, Any]],
        region: Dict[str, float],
        logo_config: Dict[str, Any],
    ) -> List[Dict]:
        conf_cfg = self.config.get("conference_logos", {})
        divider_w = conf_cfg.get("divider_width", 0.04)
        gap       = conf_cfg.get("divider_gap", 0.30)
        conf_frac = conf_cfg.get("conf_zone_fraction", 0.38)

        conf_zone_w = region["w"] * conf_frac
        aff_zone_w  = region["w"] - conf_zone_w - divider_w - 2 * gap

        aff_region = {
            "x": region["x"],
            "y": region["y"],
            "w": aff_zone_w,
            "h": region["h"],
        }
        conf_region = {
            "x": region["x"] + aff_zone_w + divider_w + 2 * gap,
            "y": region["y"],
            "w": conf_zone_w,
            "h": region["h"],
        }
        divider_x = region["x"] + aff_zone_w + gap

        elements: List[Dict] = []

        # affiliation logos
        elements.extend(self._aff_grid_elements(aff_logos, aff_region, logo_config))

        # divider line (rendered by logo_divider handler)
        elements.append({
            "type": "logo_divider",
            "x": divider_x,
            "y": region["y"] + region["h"] * 0.05,
            "width": divider_w,
            "height": region["h"] * 0.90,
            "priority": 0.85,
        })

        # conf logo
        conf_aspect = self._get_image_aspect_ratio(conf_path)
        logo_h = min(conf_region["h"] * 0.88, conf_region["w"] / max(conf_aspect, 0.1))
        logo_w = logo_h * conf_aspect
        elements.append({
            "type": "conf_logo",
            "x": conf_region["x"] + (conf_region["w"] - logo_w) / 2,
            "y": conf_region["y"] + (conf_region["h"] - logo_h) / 2,
            "width": logo_w,
            "height": logo_h,
            "priority": 0.9,
        })

        return elements

    def _aff_grid_elements(
        self,
        aff_logos: List[Dict[str, Any]],
        region: Dict[str, float],
        logo_config: Dict[str, Any],
    ) -> List[Dict]:
        count = len(aff_logos)
        if count == 0:
            return []

        if count == 1:
            cols, rows = 1, 1
        elif count == 2:
            cols, rows = 2, 1
        elif count == 3:
            cols, rows = 3, 1
        elif count == 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2

        gap = logo_config.get("logo_box_gap", 0.24)
        cell_w = max((region["w"] - (cols - 1) * gap) / cols, 0.2)
        cell_h = max((region["h"] - (rows - 1) * gap) / rows, 0.2)
        max_h  = logo_config.get("max_logo_height", 1.55)
        cell_h = min(cell_h, max_h)

        grid_h = rows * cell_h + (rows - 1) * gap
        grid_w = cols * cell_w + (cols - 1) * gap
        start_y = region["y"] + max((region["h"] - grid_h) / 2, 0)
        start_x = region["x"] + max((region["w"] - grid_w) / 2, 0)

        elements: List[Dict] = []
        for idx, logo in enumerate(aff_logos):
            row, col = divmod(idx, cols)
            aspect = float(logo.get("aspect", self._get_image_aspect_ratio(logo.get("logo_path"))) or 1.0)
            aspect = max(aspect, 0.1)
            logo_h = min(cell_h, cell_w / aspect)
            logo_w = logo_h * aspect
            cell_x = start_x + col * (cell_w + gap)
            cell_y = start_y + row * (cell_h + gap)
            elements.append({
                "type": "institution_logo",
                "x": cell_x + (cell_w - logo_w) / 2,
                "y": cell_y + (cell_h - logo_h) / 2,
                "width": logo_w,
                "height": logo_h,
                "image_path": logo["logo_path"],
                "institution": logo.get("institution", ""),
                "domain": logo.get("domain"),
                "source": logo.get("source"),
                "aspect": aspect,
                "priority": 0.9,
            })
        return elements

    def _create_affiliation_logo_grid(
        self,
        affiliation_logos: List[Dict[str, Any]],
        template_layout: Dict[str, Any],
        is_vertical_stack: bool,
    ) -> List[Dict[str, Any]]:
        """Kept for backward-compat; delegates to new helper."""
        logo_config = self.config.get("affiliation_logos", {})
        _, region = self._header_title_logo_boxes(None, template_layout, is_vertical_stack)
        return self._aff_grid_elements(affiliation_logos, region, logo_config)

    def _title_logo_region(self, template_layout: Dict[str, Any], is_vertical_stack: bool) -> Dict[str, float]:
        _, logo_box = self._header_title_logo_boxes(None, template_layout, is_vertical_stack)
        return logo_box

    def _header_title_logo_boxes(
        self,
        state: PosterState | None,
        template_layout: Dict[str, Any],
        is_vertical_stack: bool | None = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return non-overlapping title and logo boxes inside the header.

        Title and logo placement must be computed together. Previously the
        title used a fixed template-prior width while logos used a lane-based
        heuristic, which could make the conference logo overlap the title.
        """
        header = template_layout["header"]
        lanes = sorted(template_layout.get("lanes", []), key=lambda lane: lane["x"])
        if is_vertical_stack is None:
            is_vertical_stack = bool(lanes) and len({round(lane["x"], 3) for lane in lanes}) == 1

        has_conf = bool(state and state.get("logo_path") and Path(state["logo_path"]).exists())
        aff_logos = [
            logo for logo in ((state or {}).get("affiliation_logos") or [])
            if logo.get("logo_path") and Path(logo["logo_path"]).exists()
        ]
        manual_aff_logo = self._manual_affiliation_logo_entry(state) if state else None
        if manual_aff_logo and not any(
            Path(logo["logo_path"]).resolve() == Path(manual_aff_logo["logo_path"]).resolve()
            for logo in aff_logos
        ):
            aff_logos.append(manual_aff_logo)
        aff_count = len(aff_logos)
        has_logo = has_conf or aff_count > 0 or state is None

        x0 = header["x"]
        y0 = header["y"]
        w = header["w"]
        h = header["h"]
        gap = 0.42 if template_layout.get("orientation") == "portrait" else 0.55
        vertical_pad = min(max(h * 0.10, 0.12), 0.35)

        if not has_logo:
            return (
                {"x": x0, "y": y0, "w": w, "h": max(h - 0.15, 0.8)},
                {"x": x0 + w, "y": y0 + vertical_pad, "w": 0.0, "h": max(h - 2 * vertical_pad, 0.1)},
            )

        explicit_logo_region = self._rightmost_logo_region(template_layout)
        if explicit_logo_region:
            logo_box = {
                "x": explicit_logo_region["x"],
                "y": explicit_logo_region["y"],
                "w": explicit_logo_region["w"],
                "h": explicit_logo_region["h"],
            }
            title_w = max(logo_box["x"] - x0 - gap, w * 0.45)
            title_box = {"x": x0, "y": y0, "w": min(title_w, w), "h": max(h - 0.15, 0.8)}
            return title_box, logo_box

        if template_layout.get("layout_mode") == "template_prior":
            if has_conf and aff_count:
                reserve_frac = 0.36
            elif aff_count >= 3:
                reserve_frac = 0.30
            else:
                reserve_frac = 0.23
        elif is_vertical_stack:
            reserve_frac = 0.28
        else:
            lane_reserve = max((lanes[-1]["w"] if lanes else 0.0), w * 0.22)
            reserve_frac = min(max(lane_reserve / max(w, 0.1), 0.22), 0.36)

        min_logo_w = 2.8 if template_layout.get("orientation") == "portrait" else 4.0
        logo_w = min(max(w * reserve_frac, min_logo_w), w * 0.38)
        min_title_w = w * (0.58 if template_layout.get("orientation") == "portrait" else 0.55)
        if w - logo_w - gap < min_title_w:
            logo_w = max(w - min_title_w - gap, min_logo_w)

        logo_x = x0 + w - logo_w
        title_w = max(logo_x - x0 - gap, min_title_w)
        logo_box = {
            "x": logo_x,
            "y": y0 + vertical_pad,
            "w": logo_w,
            "h": max(h - 2 * vertical_pad, 0.65),
        }
        title_box = {
            "x": x0,
            "y": y0,
            "w": min(title_w, max(logo_x - x0 - gap, 0.1)),
            "h": max(h - 0.15, 0.8),
        }
        return title_box, logo_box

    def _rightmost_logo_region(self, template_layout: Dict[str, Any]) -> Dict[str, float] | None:
        logo_regions = template_layout.get("logo_regions") or []
        if not logo_regions:
            return None
        region = max(logo_regions, key=lambda item: item.get("x", 0))
        return {
            "x": region["x"],
            "y": region["y"],
            "w": region["w"],
            "h": region["h"],
        }

    def _get_image_aspect_ratio(self, image_path: str | None) -> float:
        if not image_path or not Path(image_path).exists():
            return self.layout_constants["default_logo_aspect_ratio"]
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size[0] / max(img.size[1], 1)
    
    def _create_section_elements(self, section: Dict, column_x: float, start_y: float, 
                               column_width: float, state: PosterState, available_height: float = None) -> List[Dict]:
        """create all elements for a section with precise positioning"""
        elements = []
        current_y = start_y
        
        # enhanced section title with design styling
        section_title = section.get("section_title", "")
        if section_title:
            title_elements = self._create_section_title_design(
                section, column_x, current_y, column_width, state
            )
            elements.extend(title_elements)
            
            # calculate total height used by title and accent elements
            title_total_height = max(elem["y"] + elem["height"] - current_y for elem in title_elements)
            current_y += title_total_height + self.config["layout"]["title_to_content_spacing"]
        
        # visual assets first (after title, before text)
        visual_assets = section.get("visual_assets", [])
        for visual_asset in visual_assets:
            visual_id = visual_asset.get("visual_id", "")
            text_padding = self.config["layout"]["text_padding"]["left_right"]
            visual_width = self._get_visual_width_for_lane(column_width, state)
            final_visual_width, final_visual_height, scale_factor = self._calculate_visual_height(visual_id, visual_width, state, available_height)
            
            # center the visual within the section (important for scaled visuals)
            section_content_width = column_width - (2 * text_padding)
            if final_visual_width < section_content_width:
                # center horizontally within the section
                visual_x = column_x + text_padding + (section_content_width - final_visual_width) / 2
            else:
                # use left alignment if visual fills the section
                visual_x = column_x + text_padding
            
            elements.append({
                "type": "visual",
                "x": visual_x,
                "y": current_y,
                "width": final_visual_width,
                "height": final_visual_height,
                "section_id": section.get("section_id"),
                "lane_id": section.get("column_assignment"),
                "visual_id": visual_id,
                "slot_id": f"{section.get('section_id')}_{visual_id}",
                "scale_factor": scale_factor,  # for renderer to apply proper scaling
                "priority": 0.6,
                "id": f"{section.get('section_id')}_{visual_id}",
                "font_family": self.body_text_font_family,
                "font_color": "#000000",
                "font_size": 44,
                "line_spacing": 1.0
            })
            # use the already-scaled height for positioning (no double scaling)
            current_y += final_visual_height + self.config["layout"]["visual_spacing"]["below_visual"]
        
        # text content (after visuals)
        text_content = section.get("text_content", [])
        if text_content:
            combined_text = "\n".join(text_content)
            text_padding = self.config["layout"]["text_padding"]["left_right"]  # consistent with layout positioning
            text_measurement = measure_text_height(
                text_content=combined_text,
                width_inches=column_width - (2 * text_padding),
                font_name=self.body_text_font_family,
                font_size=44,
                line_spacing=1.0
            )
            text_height = text_measurement["optimal_height"] + 0.1
            
            # apply text padding to match measurement calculation
            elements.append({
                "type": "text",
                "x": column_x + text_padding,
                "y": current_y,
                "width": column_width - (2 * text_padding),
                "height": text_height,
                "section_id": section.get("section_id"),
                "lane_id": section.get("column_assignment"),
                "slot_id": section.get("slot_id"),
                "content": combined_text,
                "font_family": self.body_text_font_family,
                "font_size": 44,
                "font_color": "#000000",
                "priority": 0.5,
                "id": f"{section.get('section_id')}_text",
                "line_spacing": 1.0
            })
            current_y += text_height + 0.3
            
        return elements
    
    def _create_section_title_design(self, section: Dict, column_x: float, start_y: float, column_width: float, state: PosterState) -> List[Dict]:
        """create section title with a full-width word-art band."""
        elements = []
        section_title = section.get("section_title", "")
        section_id = section.get("section_id", "")
        
        # get section title design from state
        title_design_state = state.get("section_title_design") or {}
        title_design = title_design_state.get("section_title_design", {})
        
        # find specific section application
        section_app = {}
        for app in title_design.get("section_applications", []):
            if app.get("section_id") == section_id:
                section_app = app
                break
        
        # extract styling information
        title_styling = section_app.get("title_styling", {})
        accent_styling = section_app.get("accent_styling", {})
        section_style = self.visual_style_config.get("section_title", {}) if self.visual_style_config.get("enabled", False) else {}
        portrait_section_title_style = self._portrait_section_title_style_override(state)
        if portrait_section_title_style:
            section_style = {**section_style, **portrait_section_title_style}

        section_title_font_size = title_styling.get(
            "font_size",
            section_style.get("font_size", self.config["typography"]["sizes"]["section_title"]),
        )
        horizontal_padding = float(section_style.get("horizontal_padding_inches", 0.28))
        vertical_padding = float(section_style.get("vertical_padding_inches", 0.04))
        min_bar_height = float(section_title_font_size) / 72 + (2 * vertical_padding)
        bar_height = max(float(section_style.get("bar_height_inches", min_bar_height)), min_bar_height)
        bar_color = accent_styling.get("color") or section_style.get("bar_fill_color", "#06134A")
        title_label = self._section_title_label(section, section_title, state)
        display_title = title_label["title"]

        bar_element = {
            "type": "title_accent_block",
            "x": column_x,
            "y": start_y,
            "width": column_width,
            "height": bar_height,
            "section_id": section_id,
            "lane_id": section.get("column_assignment"),
            "slot_id": section.get("slot_id"),
            "color": bar_color,
            "priority": 0.7
        }
        elements.append(bar_element)

        title_element = {
            "type": "section_title",
            "x": column_x + horizontal_padding,
            "y": start_y + vertical_padding,
            "width": max(column_width - (2 * horizontal_padding), 0.1),
            "height": max(bar_height - (2 * vertical_padding), 0.1),
            "section_id": section_id,
            "lane_id": section.get("column_assignment"),
            "slot_id": section.get("slot_id"),
            "section_title": display_title,
            "font_family": portrait_section_title_style.get(
                "font_family",
                title_styling.get("font_family", section_style.get("font_family", self.section_title_font_family)),
            ),
            "font_size": section_title_font_size,
            "font_weight": portrait_section_title_style.get(
                "font_weight",
                title_styling.get("font_weight", section_style.get("font_weight", "bold")),
            ),
            "font_color": portrait_section_title_style.get(
                "font_color",
                title_styling.get("color", section_style.get("font_color", "#FFFFFF")),
            ),
            "alignment": "center",
            "section_number": title_label.get("number"),
            "section_numbering_mode": title_label.get("numbering_mode"),
            "section_number_font_scale": float(section_style.get("small_number_font_scale", 0.62)),
            "section_number_width": float(section_style.get("small_number_width_inches", 0.46)),
            "section_number_gap": float(section_style.get("small_number_gap_inches", 0.12)),
            "wordart_style": {
                "name": "navy_band_serif"
                if self.visual_style_config.get("selected_preset", "navy_serif") == "navy_serif"
                else f"{self.visual_style_config.get('selected_preset')}_band",
                "shadow": section_style.get("shadow", {}),
            },
            "priority": 0.8
        }
        elements.append(title_element)
        
        return elements

    def _portrait_section_title_style_override(self, state: PosterState) -> Dict[str, Any]:
        if int(state.get("poster_height") or 0) <= int(state.get("poster_width") or 0):
            return {}
        style = (self.config.get("poster_visual_style") or {}).get("portrait_header_section_title") or {}
        if not style.get("enabled", False):
            return {}
        return {key: value for key, value in style.items() if key != "enabled"}

    def _section_title_label(self, section: Dict[str, Any], title: str, state: PosterState) -> Dict[str, Any]:
        raw_title = repair_possessive_title_apostrophe(str(title or "").strip())
        explicit_number, clean_title = self._strip_section_number(raw_title)
        mode = self._section_title_numbering_mode(state)
        number = explicit_number or self._section_number_from_slot(section)

        if mode == "inline" and number:
            return {
                "title": f"{number}. {clean_title}",
                "number": number,
                "numbering_mode": "inline",
            }
        if mode == "small" and number:
            return {
                "title": clean_title,
                "number": number,
                "numbering_mode": "small",
            }
        return {
            "title": clean_title,
            "number": None,
            "numbering_mode": "off",
        }

    def _section_title_numbering_mode(self, state: PosterState) -> str:
        raw_mode = str(state.get("section_title_numbering") or self.config.get("section_title_numbering", "off")).strip().lower()
        aliases = {
            "": "off",
            "none": "off",
            "false": "off",
            "no": "off",
            "0": "off",
            "on": "small",
            "true": "small",
            "yes": "small",
            "1": "small",
        }
        mode = aliases.get(raw_mode, raw_mode)
        if mode not in {"off", "small", "inline"}:
            return "off"
        return mode

    def _strip_section_number(self, raw_title: str) -> Tuple[str, str]:
        match = re.match(r"^\s*(\d+)[\.)]\s*(.+?)\s*$", raw_title)
        if match:
            return match.group(1), match.group(2).strip()
        return "", raw_title

    def _section_number_from_slot(self, section: Dict[str, Any]) -> str:
        for key in ("slot_id", "column_assignment", "lane_id"):
            match = re.search(r"(\d+)", str(section.get(key, "")))
            if match:
                return str(int(match.group(1)))
        return ""
    
    def _validate_precise_layout(self, layout_data: List[Dict], poster_width: float, 
                               poster_height: float) -> Dict[str, Any]:
        """validate layout for overlaps and overflow"""
        issues = []
        valid = True
        
        # check for overflow
        for element in layout_data:
            right_edge = element["x"] + element["width"]
            bottom_edge = element["y"] + element["height"]
            
            if right_edge > poster_width:
                issues.append(f"Element {element.get('id', 'unknown')} overflows right edge")
                valid = False
            
            if bottom_edge > poster_height:
                issues.append(f"Element {element.get('id', 'unknown')} overflows bottom edge")
                valid = False
        
        # check for overlaps (simplified check)
        for i, elem1 in enumerate(layout_data):
            for j, elem2 in enumerate(layout_data[i+1:], i+1):
                if self._elements_overlap(elem1, elem2):
                    issues.append(f"Elements {elem1.get('id', 'unknown')} and {elem2.get('id', 'unknown')} overlap")
                    valid = False
        
        # calculate space utilization
        total_used_area = sum(elem["width"] * elem["height"] for elem in layout_data)
        total_poster_area = poster_width * poster_height
        space_utilization = total_used_area / total_poster_area if total_poster_area > 0 else 0
        
        return {
            "valid": valid,
            "issues": issues,
            "space_utilization": space_utilization,
            "total_elements": len(layout_data)
        }
    
    def _elements_overlap(self, elem1: Dict, elem2: Dict) -> bool:
        """check if two elements overlap"""
        return not (
            elem1["x"] + elem1["width"] <= elem2["x"] or
            elem2["x"] + elem2["width"] <= elem1["x"] or
            elem1["y"] + elem1["height"] <= elem2["y"] or
            elem2["y"] + elem2["height"] <= elem1["y"]
        )
    
    def _create_spatial_layout(self, sections: List[Dict], column_distribution: Dict, 
                             lane_map: Dict[str, Dict[str, float]], state: PosterState) -> List[Dict]:
        """create precise spatial layout using css-like calculations"""
        
        log_agent_info(self.name, "creating spatial layout with css-like precision")
        
        # organize sections by spatial assignment
        columns = {lane_id: {"sections": [], "total_height": 0.0} for lane_id in lane_map}
        
        for section in sections:
            column = section.get("column_assignment", "left")
            if column not in columns and section.get("slot_id") in columns:
                column = section["slot_id"]
            if column in columns:
                columns[column]["sections"].append(section)
        
        log_agent_info(
            self.name,
            ", ".join(
                f"{lane_id}={len(columns[lane_id]['sections'])}"
                for lane_id in lane_map
            ),
        )
        
        # calculate precise heights for each section
        for column_name, column_data in columns.items():
            for section in column_data["sections"]:
                section_height = self._calculate_precise_section_height(
                    section,
                    lane_map[column_name]["w"],
                    state,
                    lane_map[column_name]["h"],
                )
                section["calculated_height"] = section_height
                column_data["total_height"] += section_height
        
        
        # return layout in expected format
        ordered_columns = []
        for index, lane_id in enumerate(lane_map):
            ordered_columns.append({
                "column_id": index,
                "column_name": lane_id,
                "sections": [
                    section for section in sections
                    if section.get("column_assignment") == lane_id or section.get("slot_id") == lane_id
                ],
                "estimated_height": columns[lane_id]["total_height"],
            })
        return ordered_columns
    
    def _calculate_precise_section_height(self, section: Dict, column_width: float, state: PosterState, available_height: float = None) -> float:
        """calculate precise section height using css box model"""
        
        total_height = 0.0
        
        # section title height (if exists)
        title = section.get("section_title", "")
        if title:
            title_padding = self.layout_constants["title_padding"]  # consistent with layout positioning
            title_measurement = measure_text_height(
                text_content=title,
                width_inches=column_width - (2 * title_padding),  # account for padding
                font_name="Helvetica Neue",
                font_size=64,
                line_spacing=1.0
            )
            title_height = title_measurement["optimal_height"] + 0.3  # title margin
            total_height += title_height
        
        # text content height with fixed line spacing
        text_content = section.get("text_content", [])
        if text_content:
            # join all bullet points with proper paragraph separation
            full_text = "\n\n".join(text_content)  # double newline between paragraphs
            
            text_padding = self.config["layout"]["text_padding"]["left_right"]  # consistent with layout positioning
            text_measurement = measure_text_height(
                text_content=full_text,
                width_inches=column_width - (2 * text_padding),  # account for padding
                font_name=self.body_text_font_family, 
                font_size=44,
                line_spacing=1.0
            )
            text_height = text_measurement["optimal_height"] + 0.2  # text margin
            total_height += text_height
        
        # visual assets height (fixed aspect ratio)
        visual_assets = section.get("visual_assets", [])
        for visual in visual_assets:
            visual_id = visual.get("visual_id", "")
            if visual_id:
                visual_width = self._get_visual_width_for_lane(column_width, state)
                final_visual_width, final_visual_height, scale_factor = self._calculate_visual_height(visual_id, visual_width, state, available_height)
                # use the already-scaled height for section sizing (no double scaling)
                total_height += final_visual_height + 0.3  # visual margin
        
        # section padding and margins
        section_padding = self.layout_constants["section_padding"]
        total_height += section_padding
        
        return total_height
    
    def _calculate_visual_height(self, visual_id: str, visual_width: float, state, available_height: float = None) -> tuple:
        """calculate proper visual width and height based on aspect ratio with auto-shrinking for large visuals
        
        returns: (final_width, final_height, scale_factor)
        """
        # visual width already accounts for padding (passed from caller)
        
        # get aspect ratio from state data
        visual_assets = state.get("visual_assets") or {}
        
        # better default aspect ratios based on visual type
        if visual_id.startswith("table_"):
            aspect_ratio = 1.5  # tables are often wider than tall
        elif visual_id.startswith("figure_"):
            aspect_ratio = 1.2  # figures vary but often slightly wider
        else:
            aspect_ratio = self.layout_constants["default_logo_aspect_ratio"]  # default square
        
        # handle both formats: "figure_1"/"table_1" and "1"/"2" etc.
        lookup_id = visual_id
        
        # if visual_id has prefix, extract the number
        if visual_id.startswith("figure_"):
            lookup_id = visual_id.replace("figure_", "")
        elif visual_id.startswith("table_"):
            lookup_id = visual_id.replace("table_", "")
        
        asset_data = visual_assets.get(visual_id)
        if asset_data:
            aspect_ratio = asset_data.get("aspect", aspect_ratio)
            log_agent_info(self.name, f"found visual {visual_id} in visual_assets, aspect={aspect_ratio:.2f}")
        else:
            log_agent_warning(self.name, f"visual {visual_id} (lookup: {lookup_id}) not found in state data, using fallback aspect={aspect_ratio:.2f}")
            log_agent_info(self.name, f"available visual assets: {list(visual_assets.keys())}")
        
        # calculate original height from aspect ratio
        original_height = visual_width / aspect_ratio
        
        # check if shrinking is needed
        scale_factor = 1.0
        if available_height:
            max_fraction = self._max_visual_height_fraction(visual_id, state)
            max_visual_height = available_height * max_fraction
            if original_height > max_visual_height:
                scale_factor = max_visual_height / max(original_height, 0.01)
                log_agent_info(
                    self.name,
                    f"visual {visual_id} capped ({original_height:.2f}\" > {max_fraction:.0%} of {available_height:.2f}\"), scale={scale_factor:.2f}",
                )
        
        # apply scaling to both width and height to maintain aspect ratio
        final_width = visual_width * scale_factor
        final_height = original_height * scale_factor
        lane = {
            "id": "",
            "w": visual_width,
            "h": available_height or final_height,
            "poster_orientation": "portrait"
            if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
            else "landscape",
        }
        final_width, final_height, footprint_report = enforce_visual_footprint(
            visual_id,
            final_width,
            final_height,
            visual_width,
            lane,
            state,
            self.config,
        )
        if footprint_report.get("adjusted"):
            scale_factor = final_width / max(visual_width, 0.01)
            log_agent_info(
                self.name,
                f"visual {visual_id} enlarged to footprint contract "
                f"({final_width:.2f}\"x{final_height:.2f}\")",
            )
        
        log_agent_info(self.name, f"visual {visual_id}: orig_w={visual_width:.2f}\", orig_h={original_height:.2f}\", scale={scale_factor:.1f}, final_w={final_width:.2f}\", final_h={final_height:.2f}\"")
        
        # return final width, height and scale factor for rendering
        return final_width, final_height, scale_factor

    def _max_visual_height_fraction(self, visual_id: str, state) -> float:
        if str(visual_id).startswith("generated_teaser"):
            teaser_config = self.config.get("generated_teaser") or {}
            template_layout = state.get("layout_template_metadata") or {}
            orientation = str(template_layout.get("orientation") or "").lower()
            if not orientation:
                orientation = (
                    "portrait"
                    if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0)
                    else "landscape"
                )
            if orientation == "portrait":
                return float(teaser_config.get("portrait_layout_max_height_fraction", teaser_config.get("layout_max_height_fraction", 0.76)))
            return float(teaser_config.get("layout_max_height_fraction", 0.76))
        fast_visual_policy = state.get("fast_visual_policy") or (self.config.get("template_fast_mode") or {}).get("visual_policy") or {}
        if state.get("template_fast_mode"):
            if str(visual_id).startswith("table_"):
                return float(fast_visual_policy.get("table_max_height_fraction", 0.62))
            if str(visual_id).startswith("figure_"):
                return float(fast_visual_policy.get("figure_max_height_fraction", 0.55))
            return float(fast_visual_policy.get("default_max_height_fraction", 0.42))
        return 0.40
    


def layout_agent_node(state: PosterState) -> Dict[str, Any]:
    result = LayoutAgent()(state)
    return {
        **state,
        "design_layout": result["design_layout"],
        "column_assignment": result.get("column_assignment"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }
