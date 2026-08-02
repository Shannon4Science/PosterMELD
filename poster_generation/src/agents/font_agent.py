"""
font styling and keyword highlighting
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config
from src.utils.style_options import resolve_poster_visual_style, resolve_typography_config
from src.utils.text_cleanup import normalize_text_for_poster
from jinja2 import Template


class FontAgent:
    """handles text styling and keyword highlighting"""
    
    def __init__(self):
        self.name = "font_agent"
        self.keyword_extraction_prompt = load_prompt("config/prompts/extract_keywords.txt")
        self.config = load_config()

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting font styling")
        
        try:
            design_layout = state.get("design_layout", [])
            color_scheme = state.get("color_scheme", {})
            story_board = state.get("story_board", {})
            
            if not design_layout:
                raise ValueError("missing design_layout from layout agent")
            if not color_scheme:
                raise ValueError("missing color_scheme from color agent")
            if not story_board:
                raise ValueError("missing story_board from story board curator")
            
            # identify keywords to highlight
            keywords = self._identify_keywords(story_board, state)
            keywords = self._normalize_keyword_section_ids(keywords, story_board)
            
            # apply styling to layout
            styled_layout = self._apply_styling(design_layout, color_scheme, keywords, state)
            styling_interfaces = self.get_styling_interfaces(state)
            
            state["styled_layout"] = styled_layout
            state["keywords"] = keywords
            state["styling_interfaces"] = styling_interfaces
            state["current_agent"] = self.name
            
            self._save_styled_layout(state)
            
            # count total keywords across all sections
            total_keywords = sum(
                len(keyword_list)
                for section_data in keywords.get("section_keywords", {}).values()
                for keyword_list in section_data.values()
                if isinstance(keyword_list, list)
            )
            
            log_agent_success(self.name, f"applied enhanced styling to {len(styled_layout)} elements")
            log_agent_success(self.name, f"identified {total_keywords} keywords for highlighting")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _identify_keywords(self, story_board: Dict, state: PosterState) -> Dict[str, Any]:
        """identify keywords using story board content and enhanced narrative"""
        
        narrative_content = state.get("narrative_content", {})
        
        # extract keywords using LLM with external prompt
        log_agent_info(self.name, "identifying keywords for highlighting")
        
        try:
            agent = LangGraphAgent("expert at identifying key terms for visual highlighting", state["text_model"], state, "font_agent")

            template_data = {
                "enhanced_narrative": json.dumps(narrative_content, indent=2),
                "curated_content": json.dumps(story_board, indent=2)
            }

            prompt = Template(self.keyword_extraction_prompt).render(**template_data)
            response = agent.step(prompt)
            result = extract_json(response.content)

            # add token usage
            state["tokens"].add_text(response.input_tokens, response.output_tokens)

            # Some models return a non-object payload (a bare string/list, or a dict
            # without section_keywords). Downstream styling assumes a dict, so validate
            # the shape here and fall back to heuristic keywords rather than crashing.
            if not isinstance(result, dict) or not isinstance(result.get("section_keywords"), dict):
                log_agent_warning(self.name, "keyword payload had unexpected shape; using heuristic fallback")
                return self._fallback_keywords(story_board)

            return result
        except Exception as exc:
            log_agent_warning(self.name, f"keyword extraction unavailable; using heuristic fallback: {exc}")
            return self._fallback_keywords(story_board)

    def _normalize_keyword_section_ids(self, keywords: Dict[str, Any], story_board: Dict) -> Dict[str, Any]:
        """Map LLM-friendly section labels back to exact layout section ids."""
        section_keywords = keywords.get("section_keywords") if isinstance(keywords, dict) else None
        if not isinstance(section_keywords, dict):
            return keywords

        sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
        alias_to_section: Dict[str, str] = {}
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            if not section_id:
                continue
            aliases = {
                section_id,
                section_id.removeprefix("sec_"),
                str(section.get("content_role") or "").strip(),
                str(section.get("slot_id") or "").strip(),
                str(section.get("column_assignment") or "").strip(),
            }
            for part in section_id.removeprefix("sec_").split("_"):
                if len(part) >= 4:
                    aliases.add(part)
            title_alias = re.sub(r"[^a-z0-9]+", "_", str(section.get("section_title") or "").lower()).strip("_")
            if title_alias:
                aliases.add(title_alias)
            for alias in aliases:
                alias_key = self._keyword_section_alias(alias)
                if alias_key:
                    alias_to_section.setdefault(alias_key, section_id)

        normalized: Dict[str, Dict[str, List[str]]] = {}
        for raw_key, value in section_keywords.items():
            raw_key_str = str(raw_key or "")
            exact_key = raw_key_str if raw_key_str in alias_to_section.values() else ""
            target_id = exact_key or alias_to_section.get(self._keyword_section_alias(raw_key_str))
            if not target_id:
                target_id = raw_key_str
            target_bucket = normalized.setdefault(target_id, {"bold_contrast": [], "bold": [], "italic": []})
            if isinstance(value, dict):
                for style_name in ("bold_contrast", "bold", "italic"):
                    existing = target_bucket.setdefault(style_name, [])
                    for item in value.get(style_name) or []:
                        text = str(item or "").strip()
                        if text and text not in existing:
                            existing.append(text)

        result = dict(keywords)
        result["section_keywords"] = normalized
        return result

    def _keyword_section_alias(self, value: str) -> str:
        value = str(value or "").strip().lower()
        if not value:
            return ""
        value = value.removeprefix("sec_")
        return re.sub(r"[^a-z0-9]+", "_", value).strip("_")

    def _fallback_keywords(self, story_board: Dict) -> Dict[str, Any]:
        section_keywords: Dict[str, Dict[str, List[str]]] = {}
        sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
        for section in sections:
            section_id = str(section.get("section_id") or "")
            if not section_id:
                continue
            text = " ".join([str(section.get("section_title") or ""), *[str(item) for item in section.get("text_content") or []]])
            acronyms = re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", text)
            terms = re.findall(r"\b[A-Za-z][A-Za-z-]{5,}\b", text)
            seen = set()
            bold_contrast = []
            for term in [*acronyms, *terms]:
                key = term.lower()
                if key in seen or key in {"section", "figure", "result", "results", "poster"}:
                    continue
                seen.add(key)
                bold_contrast.append(term)
                if len(bold_contrast) >= 4:
                    break
            section_keywords[section_id] = {
                "bold_contrast": bold_contrast,
                "bold": [],
                "italic": [],
            }
        return {
            "section_keywords": section_keywords,
            "formatting_summary": {"fallback": "heuristic_keywords"},
        }

    def _apply_styling(self, layout: List[Dict], colors: Dict, keywords: Dict, state: PosterState) -> List[Dict]:
        """apply styling with proper bullet point and bold formatting"""
        styled_layout = []
        section_keywords = keywords.get("section_keywords", {})
        
        # process all elements with enhanced styling
        for element in layout:
            styled_element = element.copy()
            
            # apply element-specific styling
            if element.get("type") == "title":
                self._apply_title_styling(styled_element, colors, state)
            
            elif element.get("type") in ["section_title", "title_accent_block", "title_accent_line"]:
                # these are handled by the section title designer
                pass
            
            elif element.get("type") == "section_container":
                self._apply_section_container_styling(styled_element, colors, state)
                
            elif element.get("type") in ["text", "visual", "mixed"]:
                self._apply_content_styling(styled_element, colors, section_keywords, state)
            
            elif element.get("type") in ["conf_logo", "aff_logo", "institution_logo", "logo_divider"]:
                # logos don't need text styling
                pass

            elif element.get("type") in ["template_background", "template_header_background", "template_footer_background"]:
                # extracted template style blocks already carry their own colors
                pass
            
            styled_layout.append(styled_element)
        
        # sort by priority for proper rendering order
        styled_layout.sort(key=lambda x: x.get("priority", 0.5))
        
        return styled_layout

    def _apply_title_styling(self, element: Dict, colors: Dict, state: PosterState = None):
        """apply styling to title elements"""
        if element.get("content"):
            element["content"] = normalize_text_for_poster(element["content"])
        typography = resolve_typography_config(state, self.config)
        visual_style = resolve_poster_visual_style(state, self.config)
        main_title_style = visual_style.get("main_title", {}) if visual_style.get("enabled", False) else {}
        title_style_override = self._portrait_header_main_title_override(element, state)
        if title_style_override:
            element["main_title_style_override"] = title_style_override
            main_title_style = {**main_title_style, **title_style_override}
        element["font_family"] = main_title_style.get("font_family") or typography.get("fonts", {}).get("title", "Helvetica Neue")
        template_meta = (state or {}).get("layout_template_metadata") or {}
        header_color = (template_meta.get("style_tokens") or {}).get("header_background")
        if main_title_style.get("font_color"):
            element["font_color"] = main_title_style["font_color"]
        elif template_meta.get("extracted_template") and header_color and self._is_dark_color(header_color):
            element["font_color"] = "#FFFFFF"
        else:
            element["font_color"] = colors.get("text_on_theme", "#FFFFFF")
        sizes = typography.get("sizes", {})
        if not element.get("lock_header_typography"):
            element["font_size"] = sizes.get("title", 100)
            element["author_font_size"] = sizes.get("authors", 72)
        else:
            element["font_size"] = element.get("font_size", sizes.get("title", 100))
            element["author_font_size"] = element.get("author_font_size", sizes.get("authors", 72))
            element["subtitle_font_size"] = element.get("subtitle_font_size", max(int(element["font_size"] * 0.58), 24))
            element["alignment"] = element.get("alignment", "left")
        element["font_weight"] = "bold"

    def _portrait_header_main_title_override(self, element: Dict, state: PosterState = None) -> Dict[str, Any]:
        if not state or int(state.get("poster_height") or 0) <= int(state.get("poster_width") or 0):
            return {}
        if element.get("header_route") != "split_logos":
            return {}
        style = (self.config.get("poster_visual_style") or {}).get("portrait_header_main_title") or {}
        if not style.get("enabled", False):
            return {}
        return {key: value for key, value in style.items() if key != "enabled"}

    def _is_dark_color(self, color: str) -> bool:
        color = (color or "").strip().lstrip("#")
        if len(color) != 6:
            return False
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
        except ValueError:
            return False
        brightness = (299 * r + 587 * g + 114 * b) / 1000
        return brightness < 128

    def _apply_section_container_styling(self, element: Dict, colors: Dict, state: PosterState):
        """apply styling to section container elements"""
        template_meta = state.get("layout_template_metadata") or {}
        is_extracted_template = bool(template_meta.get("extracted_template"))
        if is_extracted_template:
            element.setdefault("border_color", colors.get("mono_light", "#CCCCCC"))
            element.setdefault("border_width", 1)
            element.setdefault("fill_color", "#FFFFFF")

    def _apply_content_styling(self, element: Dict, colors: Dict, section_keywords: Dict, state: PosterState = None):
        """apply styling to content elements with keyword highlighting"""
        typography = resolve_typography_config(state, self.config)
        # determine parent section for keyword lookup
        parent_section = self._extract_parent_section(element)
        keywords_for_section = section_keywords.get(parent_section, {})
        
        # ensure proper bullet point formatting first (before keyword highlighting to preserve formatting)
        if element.get("content"):
            element["content"] = self._format_bullet_points(normalize_text_for_poster(element["content"]))
        
        # apply keyword highlighting to content (after bullet formatting)
        if keywords_for_section and element.get("content"):
            content = element["content"]
            original_content = content
            content = self._apply_keyword_highlighting(content, keywords_for_section, colors)
            element["content"] = content
            
            # debug logging
            if content != original_content:
                total_keywords = sum(len(kw_list) for kw_list in keywords_for_section.values() if isinstance(kw_list, list))
                log_agent_info(self.name, f"Applied highlighting to {parent_section}: found {total_keywords} keywords")
            elif keywords_for_section:
                total_keywords = sum(len(kw_list) for kw_list in keywords_for_section.values() if isinstance(kw_list, list))
                log_agent_warning(self.name, f"Keywords found for {parent_section} ({total_keywords} total) but no highlighting applied")
        
        # apply base text styling
        element["font_family"] = typography.get("fonts", {}).get("body_text", "Arial")
        element["font_color"] = colors.get("text", "#000000")
        element["font_size"] = typography.get("sizes", {}).get("body_text", 44)

    def _extract_parent_section(self, element: Dict) -> str:
        """extract parent section id from element"""
        element_id = element.get("id", "")
        
        # extract section id from element id
        if "_" in element_id and element_id.endswith("_text"):
            # remove the "_text" suffix to get the section ID
            return element_id[:-5]  # remove last 5 characters ("_text")
        elif "_" in element_id:
            # fallback: remove last part after underscore
            parts = element_id.split("_")
            if len(parts) > 1:
                return "_".join(parts[:-1])
        
        return ""

    def _apply_keyword_highlighting(self, content: str, keywords: Dict, colors: Dict) -> str:
        """apply semantic-based keyword highlighting with three distinct styles"""
        # use contrast color for highlighting
        highlight_color = colors.get("contrast", colors.get("theme", "#1E3A8A"))
        
        # define highlighting styles based on semantic categories
        style_functions = {
            "bold_contrast": lambda text: f"<color:{highlight_color}>{text}</color>",  # contrast color (bold applied automatically in renderer)
            "bold": lambda text: f"**{text}**",  # just bold
            "italic": lambda text: f"*{text}*"  # italic
        }
        
        # apply each style category
        for style_type, style_func in style_functions.items():
            keyword_list = keywords.get(style_type, [])
            for keyword in keyword_list:
                if not keyword.strip():
                    continue
                content = self._highlight_keyword_in_content(content, keyword, style_func)
        
        return content

    def _highlight_keyword_in_content(self, content: str, keyword: str, style_func) -> str:
        """highlight a specific keyword in content"""
        if f"<color:" in content and keyword.lower() in content.lower():
            return content
        
        escaped_keyword = re.escape(keyword.strip())
        
        # first try to match keyword with existing bold formatting
        bold_pattern = rf'\*\*([^*]*?{escaped_keyword}[^*]*?)\*\*'
        bold_match = re.search(bold_pattern, content, re.IGNORECASE)
        
        if bold_match:
            # extract the full bold text, replace only the keyword part
            full_bold_text = bold_match.group(1)
            keyword_in_bold = re.search(escaped_keyword, full_bold_text, re.IGNORECASE)
            if keyword_in_bold:
                # replace just the keyword within the bold text
                original_keyword = keyword_in_bold.group(0)
                new_keyword_formatted = style_func(original_keyword)
                
                # check if style_func returns color format
                if '<color:' in new_keyword_formatted:
                    # remove the outer ** since color already implies bold
                    new_bold_text = full_bold_text.replace(original_keyword, new_keyword_formatted, 1)
                    old_full_bold = bold_match.group(0)
                    return content.replace(old_full_bold, new_bold_text, 1)
                else:
                    # for regular bold/italic, keep the ** wrapper
                    new_keyword = new_keyword_formatted.replace('**', '').replace('**', '')  # remove any extra bold markers
                    new_bold_text = full_bold_text.replace(original_keyword, new_keyword, 1)
                    old_full_bold = bold_match.group(0)
                    new_full_bold = f'**{new_bold_text}**'
                    return content.replace(old_full_bold, new_full_bold, 1)
        
        # then match keyword with existing italic formatting  
        italic_pattern = rf'\*({escaped_keyword})\*'
        italic_match = re.search(italic_pattern, content, re.IGNORECASE)
        
        if italic_match:
            old_formatted = italic_match.group(0)
            new_formatted = style_func(keyword)
            return content.replace(old_formatted, new_formatted, 1)
        
        plain_pattern = rf'\b{escaped_keyword}\b'
        plain_match = re.search(plain_pattern, content, re.IGNORECASE)
        
        if plain_match:
            matched_text = plain_match.group(0)
            new_formatted = style_func(matched_text)
            return content.replace(matched_text, new_formatted, 1)
        
        return content

    def _format_bullet_points(self, content: str) -> str:
        """ensure proper bullet point formatting"""
        if not content:
            return content
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # ensure start with '•' or preserve existing '•'
            if line.startswith('• '):
                formatted_lines.append(line)
            elif line.startswith('- '):
                # dash -> bullet
                formatted_lines.append('• ' + line[2:])
            elif line.startswith('* '):
                # asterisk -> bullet
                formatted_lines.append('• ' + line[2:])
            elif not line.startswith('•'):
                # add bullet if missing (for content that should be bulleted)
                if any(line.lower().startswith(word) for word in ['the ', 'this ', 'our ', 'we ', 'new ', 'key ', 'main ']):
                    formatted_lines.append('• ' + line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def get_styling_interfaces(self, state: PosterState = None) -> Dict[str, Any]:
        """return interfaces for renderer to properly handle styled content"""
        font_params = resolve_typography_config(state, self.config)
        
        return {
            "bullet_point_marker": "•",
            "bold_start_tag": "**",
            "bold_end_tag": "**",
            "italic_start_tag": "*",
            "italic_end_tag": "*",
            "color_start_tag": "<color:",
            "color_end_tag": "</color>",
            "line_spacing": font_params["line_spacing"],  # from config
            "paragraph_spacing": font_params["paragraph_spacing"],
            "font_sizes": {
                "title": font_params["sizes"]["title"],
                "authors": font_params["sizes"]["authors"],
                "section_title": font_params["sizes"]["section_title"],
                "body_text": font_params["sizes"]["body_text"]
            }
        }

    def _save_styled_layout(self, state: PosterState):
        """save styled layout and keywords"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # styled layout
        with open(output_dir / "styled_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("styled_layout", []), f, indent=2)
        
        # keywords
        with open(output_dir / "keywords.json", "w", encoding='utf-8') as f:
            json.dump(state.get("keywords", {}), f, indent=2)
        
        # styling interfaces
        with open(output_dir / "styling_interfaces.json", "w", encoding='utf-8') as f:
            json.dump(state.get("styling_interfaces", self.get_styling_interfaces(state)), f, indent=2)


def font_agent_node(state: PosterState) -> Dict[str, Any]:
    result = FontAgent()(state)
    return {
        **state,
        "styled_layout": result["styled_layout"],
        "keywords": result.get("keywords"),
        "styling_interfaces": result.get("styling_interfaces"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }
