"""
Section title Designer
- Fixed style for current version
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error
from src.config.poster_config import load_config
from src.utils.style_options import resolve_poster_visual_style


class SectionTitleDesigner:
    def __init__(self):
        self.name = "section_title_designer"
        self.config = load_config()

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "generating section title styling (preset band word-art)")
        
        try:
            story_board = state.get("story_board")
            color_scheme = state.get("color_scheme")
            
            if not story_board:
                log_agent_error(self.name, "missing story_board")
                raise ValueError("missing story_board from curator")
            
            if not color_scheme:
                log_agent_error(self.name, "missing color_scheme")
                raise ValueError("missing color_scheme from color agent")
            
            title_design = self._generate_colorblock_design(story_board, color_scheme, state)
            
            state["section_title_design"] = title_design
            state["current_agent"] = self.name
            
            self._save_title_design(state)
            
            log_agent_success(self.name, "generated section title styling")
            log_agent_info(self.name, f"theme color: {color_scheme.get('theme', 'unknown')}")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _generate_colorblock_design(self, story_board: Dict, color_scheme: Dict, state: PosterState = None) -> Dict:
        """Generate colorblock design"""
        
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        
        # color mapping from config for navy_band_wordart template
        colors = self._map_rectangle_colors(color_scheme, state)
        
        # applications for all sections
        applications = self._generate_rectangle_applications(sections, colors)
        
        return {
            "section_title_design": {
                "selected_template": "navy_band_wordart"
                if colors["selected_preset"] == "navy_serif"
                else f"{colors['selected_preset']}_band_wordart",
                "design_rationale": "Code-generated full-width title band with preset word-art section text",
                "color_palette": colors,
                "spacing_specifications": {
                    "title_left_padding_inches": colors["horizontal_padding_inches"],
                    "bar_height_inches": colors["bar_height_inches"]
                },
                "section_applications": applications
            }
        }

    def _map_rectangle_colors(self, color_scheme: Dict, state: PosterState = None) -> Dict:
        """Map config to navy_band_wordart template colors"""
        
        visual_style = resolve_poster_visual_style(state, self.config)
        section_style = (visual_style or {}).get("section_title", {})
        theme_color = color_scheme.get("theme", "#1E3A8A")
        mono_light = color_scheme.get("mono_light", "#335f91")
        mono_dark = color_scheme.get("mono_dark", "#002c5e")
        
        return {
            "theme_color": theme_color,
            "mono_light": mono_light,
            "mono_dark": mono_dark,
            "title_text_color": section_style.get("font_color", "#FFFFFF"),
            "accent_rectangle_color": section_style.get("bar_fill_color", "#06134A"),
            "background_color": "#FFFFFF",
            "font_family": section_style.get("font_family", "Georgia"),
            "font_size": section_style.get("font_size", 48),
            "font_weight": section_style.get("font_weight", "bold"),
            "alignment": section_style.get("alignment", "center"),
            "bar_height_inches": section_style.get("bar_height_inches", 0.78),
            "horizontal_padding_inches": section_style.get("horizontal_padding_inches", 0.28),
            "shadow": section_style.get("shadow", {}),
            "selected_preset": visual_style.get("selected_preset", "navy_serif"),
        }

    def _generate_rectangle_applications(self, sections: List[Dict], colors: Dict) -> List[Dict]:
       
        applications = []
        
        for section in sections:
            application = {
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "title_styling": {
                    "font_family": colors["font_family"],
                    "font_size": colors["font_size"],
                    "font_weight": colors["font_weight"],
                    "color": colors["title_text_color"],
                    "alignment": colors["alignment"]
                },
                "accent_styling": {
                    "type": "full_width_bar",
                    "color": colors["accent_rectangle_color"],
                    "dimensions": {"width": "full_block_width", "height_inches": colors["bar_height_inches"]},
                    "position": "top_band",
                    "shadow": colors["shadow"],
                }
            }
            
            applications.append(application)
        
        return applications

    def _save_title_design(self, state: PosterState):
        """Save title design to json file"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "section_title_design.json", "w", encoding='utf-8') as f:
            json.dump(state.get("section_title_design", {}), f, indent=2)


def section_title_designer_node(state: PosterState) -> Dict[str, Any]:
    result = SectionTitleDesigner()(state)
    return {
        **state,
        "section_title_design": result["section_title_design"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }
