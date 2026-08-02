"""Header route planning for poster title, authors, and logos."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.tools.layout_api import LayoutTemplates
from src.utils.text_cleanup import normalize_text_for_poster, normalize_title_for_poster
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class HeaderPlanner:
    """Plan a single safe header composition route before layout rendering."""

    VALID_ROUTES = {"auto", "classic_left", "centered", "right_title", "split_logos"}
    VALID_SUBTITLE_POLICIES = {"auto", "off", "always"}
    VALID_TITLE_WRAP_POLICIES = {"auto", "single_line", "two_line"}

    def __init__(self):
        self.name = "header_planner"
        self.config = load_config()
        self.header_config = self.config.get("header_planner", {})

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "planning poster header route")

        try:
            if not self.header_config.get("enabled", True):
                state["header_plan"] = None
                state["current_agent"] = self.name
                return state

            template_layout = self._resolve_template_layout(state)
            title, authors = self._title_and_authors(state)
            aff_logos = self._collect_affiliation_logos(state, template_layout)
            has_conf = bool(state.get("logo_path") and Path(str(state["logo_path"])).exists())

            rng = self._rng(state, title)
            route = self._select_route(state, template_layout, has_conf, aff_logos, rng)
            subtitle_text = self._select_subtitle(state, title, rng)
            base_plan = self._build_plan(
                state=state,
                template_layout=template_layout,
                route=route,
                title=title,
                authors=authors,
                subtitle_text=subtitle_text,
                aff_logos=aff_logos,
                has_conf=has_conf,
                affiliation_logo_scale=1.0,
                conference_logo_scale=1.0,
                fallback=False,
            )
            plan = self._choose_checked_logo_plan(
                state=state,
                template_layout=template_layout,
                route=route,
                title=title,
                authors=authors,
                subtitle_text=subtitle_text,
                aff_logos=aff_logos,
                has_conf=has_conf,
                base_plan=base_plan,
            )

            if not plan["validation"]["passed"]:
                log_agent_warning(self.name, f"header route '{route}' failed validation: {plan['validation']['reason']}")
                fallback_plan = self._build_plan(
                    state=state,
                    template_layout=template_layout,
                    route="classic_left",
                    title=title,
                    authors=authors,
                    subtitle_text=subtitle_text,
                    aff_logos=aff_logos,
                    has_conf=has_conf,
                    affiliation_logo_scale=float(self.header_config.get("conservative_logo_scale", 0.82)),
                    conference_logo_scale=float(self.header_config.get("conservative_logo_scale", 0.82)),
                    fallback=True,
                )
                fallback_plan["logo_resize_decision"] = "fallback_conservative"
                fallback_plan["logo_resize_attempts"] = [
                    self._logo_attempt_summary(plan, "requested_route"),
                    self._logo_attempt_summary(fallback_plan, "fallback_conservative"),
                ]
                plan = fallback_plan

            state["header_plan"] = plan
            state["current_agent"] = self.name
            self._save_outputs(state, plan)

            log_agent_success(self.name, f"planned header route: {plan['route']}")
            return state
        except Exception as exc:
            log_agent_error(self.name, f"failed: {exc}")
            state["errors"].append(f"{self.name}: {exc}")
            return state

    def _resolve_template_layout(self, state: PosterState) -> Dict[str, Any]:
        template_layout = state.get("layout_template_metadata") or {}
        if template_layout.get("header"):
            return template_layout

        requested_template = (
            state.get("resolved_layout_template")
            or state.get("layout_template")
            or "three_column_postergen"
        )
        if requested_template == "auto":
            requested_template = "three_column_postergen"

        poster_margin = float(self.config["layout"]["poster_margin"])
        column_spacing = float(self.config["layout"]["column_spacing"])
        effective_height = float(state["poster_height"]) - 2 * poster_margin
        header_height = effective_height * float(self.config["layout"]["title_height_fraction"])

        template_layout = LayoutTemplates(
            float(state["poster_width"]),
            float(state["poster_height"]),
            margin=poster_margin,
            col_gap=column_spacing,
        ).get_template(
            str(requested_template),
            header_height=header_height,
            width_ratios=state.get("adaptive_lane_widths"),
        )
        state["resolved_layout_template"] = template_layout["template_name"]
        state["layout_template_metadata"] = template_layout
        return template_layout

    def _title_and_authors(self, state: PosterState) -> Tuple[str, str]:
        narrative = state.get("narrative_content") or {}
        meta = narrative.get("meta", {})
        title = (
            meta.get("poster_title")
            or meta.get("title")
            or (state.get("story_board") or {}).get("title")
            or state.get("poster_name")
            or "Title"
        )
        authors = meta.get("authors") or "Authors"
        return (
            normalize_title_for_poster(str(title)) or "Title",
            normalize_text_for_poster(str(authors)) or "Authors",
        )

    def _collect_affiliation_logos(self, state: PosterState, template_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        logos = [
            dict(logo)
            for logo in (state.get("affiliation_logos") or [])
            if logo.get("logo_path") and Path(str(logo["logo_path"])).exists()
        ]
        manual_logo = self._manual_affiliation_logo_entry(state)
        if manual_logo and self.header_config.get("manual_affiliation_logo_overrides_auto", True):
            max_logos = int(self.config.get("affiliation_logos", {}).get("max_logos", 4))
            return [manual_logo][:max_logos]
        if manual_logo and not any(
            Path(str(logo["logo_path"])).resolve() == Path(str(manual_logo["logo_path"])).resolve()
            for logo in logos
        ):
            logos.insert(0, manual_logo)
        max_logos = int(self.config.get("affiliation_logos", {}).get("max_logos", 4))
        if template_layout.get("orientation") == "portrait":
            max_logos = min(max_logos, int(self.header_config.get("portrait_max_affiliation_logos", 2)))
        return logos[:max_logos]

    def _manual_affiliation_logo_entry(self, state: PosterState) -> Optional[Dict[str, Any]]:
        logo_path = state.get("aff_logo_path")
        if not logo_path or not Path(str(logo_path)).exists():
            return None
        return {
            "institution": state.get("affiliation_logo_label") or "Affiliation",
            "logo_path": str(logo_path),
            "domain": None,
            "source": "manual",
            "aspect": self._get_image_aspect_ratio(str(logo_path)),
        }

    def _rng(self, state: PosterState, title: str) -> random.Random:
        seed = state.get("header_seed")
        if seed is not None:
            return random.Random(str(seed))
        return random.Random(f"{state.get('poster_name', '')}:{title}")

    def _select_route(
        self,
        state: PosterState,
        template_layout: Dict[str, Any],
        has_conf: bool,
        aff_logos: List[Dict[str, Any]],
        rng: random.Random,
    ) -> str:
        requested = str(
            state.get("header_route")
            or self.header_config.get("default_route")
            or "auto"
        ).strip()
        if requested not in self.VALID_ROUTES:
            requested = "auto"

        allowed = [
            route
            for route in self.header_config.get("allowed_routes", ["classic_left", "centered", "right_title", "split_logos"])
            if route in self.VALID_ROUTES and route != "auto"
        ]
        if not allowed:
            allowed = ["classic_left", "centered", "right_title", "split_logos"]

        eligible = []
        for route in allowed:
            if route == "split_logos" and not (has_conf and aff_logos):
                continue
            if route == "right_title" and not (has_conf or aff_logos):
                eligible.append(route)
                continue
            eligible.append(route)

        if (
            template_layout.get("orientation") == "portrait"
            and "split_logos" in eligible
            and not self.header_config.get("portrait_allow_split_logos", True)
        ):
            eligible.remove("split_logos")
        if not eligible:
            eligible = ["classic_left"]

        if requested != "auto":
            return requested if requested in eligible else "classic_left"
        if state.get("header_seed") is None:
            return "classic_left" if "classic_left" in eligible else eligible[0]
        return rng.choice(eligible)

    def _select_subtitle(self, state: PosterState, title: str, rng: random.Random) -> str:
        policy = str(
            state.get("header_subtitle_policy")
            or self.header_config.get("subtitle_policy")
            or "auto"
        ).strip()
        if policy not in self.VALID_SUBTITLE_POLICIES:
            policy = "auto"
        if policy == "off":
            return ""

        words = re.findall(r"[A-Za-z0-9]+", title)
        short_by_chars = len(title) <= int(self.header_config.get("short_title_max_chars", 82))
        short_by_words = len(words) <= int(self.header_config.get("short_title_max_words", 11))
        if not (short_by_chars or short_by_words):
            return ""
        if policy == "auto":
            auto_max_chars = int(
                self.header_config.get(
                    "auto_subtitle_max_chars",
                    min(int(self.header_config.get("short_title_max_chars", 82)), 52),
                )
            )
            auto_max_words = int(
                self.header_config.get(
                    "auto_subtitle_max_words",
                    min(int(self.header_config.get("short_title_max_words", 11)), 7),
                )
            )
            if len(title) > auto_max_chars or len(words) > auto_max_words:
                return ""
            if rng.random() > float(self.header_config.get("subtitle_probability", 0.5)):
                return ""
        return self._generate_subtitle(state, title)

    def _generate_subtitle(self, state: PosterState, title: str) -> str:
        max_chars = self._subtitle_max_chars(state)
        candidates = self._subtitle_candidates_from_state(state)
        for candidate in candidates:
            cleaned = self._clean_subtitle(candidate, max_chars)
            if 18 <= len(cleaned) <= max_chars:
                return cleaned

        if re.search(r"\sfor\s", title, flags=re.IGNORECASE):
            prefix, suffix = re.split(r"\sfor\s", title, maxsplit=1, flags=re.IGNORECASE)
            candidate = f"Visualizing {prefix.strip()} for {suffix.strip()}"
        else:
            keywords = [
                word
                for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", title)
                if word.lower() not in {"the", "and", "with", "from", "using", "toward", "towards"}
            ][:6]
            phrase = " ".join(keywords) if keywords else "the paper's main idea"
            candidate = f"Motivation, method, and evidence for {phrase}"
        return self._shorten_subtitle(candidate, max_chars)

    def _subtitle_max_chars(self, state: PosterState) -> int:
        default = int(self.header_config.get("subtitle_max_chars", 86))
        width = float(state.get("poster_width") or 0.0)
        height = float(state.get("poster_height") or 0.0)
        template_name = str(state.get("layout_template") or state.get("resolved_layout_template") or "")
        if height > width or template_name.endswith("_portrait"):
            return int(self.header_config.get("portrait_subtitle_max_chars", default))
        return default

    def _subtitle_candidates_from_state(self, state: PosterState) -> List[str]:
        candidates: List[str] = []
        story_board = state.get("story_board") or {}
        for section in story_board.get("spatial_content_plan", {}).get("sections", []):
            role = str(section.get("content_role") or section.get("content_type") or "").lower()
            if role and not any(token in role for token in ("overview", "foundation", "motivation", "intro")):
                continue
            for text in section.get("text_content") or []:
                candidates.append(str(text))

        for item in state.get("paper_poster_keypoints") or []:
            for key in ("key_point", "poster_text", "claim", "summary", "text"):
                if item.get(key):
                    candidates.append(str(item[key]))

        raw_text = str(state.get("raw_text") or "")
        abstract_match = re.search(r"(?is)\babstract\b\s*[:\-]?\s*(.{40,420}?)(?:\n\s*\b1\b|\n\s*introduction\b|\n\s*keywords\b)", raw_text)
        if abstract_match:
            candidates.append(abstract_match.group(1))
        return candidates

    def _clean_subtitle(self, text: str, max_chars: Optional[int] = None) -> str:
        text = normalize_text_for_poster(text)
        text = re.sub(r"^[\-\u2022\u25e6\*\s]+", "", text)
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\([^)]{0,24}\d{2,4}[^)]{0,24}\)", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .;:-")
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        limit = int(max_chars or self.header_config.get("subtitle_max_chars", 86))
        compact = self._compact_subtitle(first_sentence or text, limit)
        return self._shorten_subtitle(compact, limit)

    def _compact_subtitle(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip(" .;:-")
        if len(text) <= max_chars:
            return text
        if ":" in text:
            prefix = text.split(":", 1)[0].strip(" .;:-")
            if 18 <= len(prefix) <= max_chars:
                return prefix
        clause_patterns = [
            r"^(.{18,}?\b(?:is|are)\s+[^.;:]{4,42}?)\s+and\s+(?:must|can|will|should|requires?|needs?|uses?)\b",
            r"^(.{18,}?\b(?:faces|solves|addresses|studies)\s+[^.;:]{4,42}?)\s+(?:while|without|under|using|to)\b",
            r"^(.{18,}?\b(?:requires|uses|introduces)\s+[^.;:]{4,42}?)\s+(?:while|without|under|using|to|and)\b",
        ]
        for pattern in clause_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .;:-")
                if 18 <= len(candidate) <= max_chars:
                    return candidate
        return text

    def _shorten_subtitle(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        truncated = text[: max_chars + 1].rsplit(" ", 1)[0].strip()
        truncated = truncated.rstrip(".,;:")
        if ":" in truncated:
            prefix, suffix = truncated.rsplit(":", 1)
            if 18 <= len(prefix.strip()) and len(suffix.split()) <= 4:
                truncated = prefix.strip()
        dangling_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "by",
            "for",
            "from",
            "have",
            "in",
            "of",
            "or",
            "the",
            "to",
            "must",
            "which",
            "that",
            "using",
            "with",
        }
        while truncated and truncated.split()[-1].lower() in dangling_words:
            truncated = " ".join(truncated.split()[:-1]).rstrip(".,;:")
        return truncated if truncated else text[:max_chars].rstrip(".,;:")

    def _build_plan(
        self,
        *,
        state: PosterState,
        template_layout: Dict[str, Any],
        route: str,
        title: str,
        authors: str,
        subtitle_text: str,
        aff_logos: List[Dict[str, Any]],
        has_conf: bool,
        affiliation_logo_scale: float,
        conference_logo_scale: float,
        fallback: bool,
    ) -> Dict[str, Any]:
        header = template_layout["header"]
        title_font_size, subtitle_font_size, author_font_size = self._font_sizes(template_layout, bool(subtitle_text))
        conference_aspect = self._get_image_aspect_ratio(str(state.get("logo_path"))) if has_conf else 1.0
        title_box, logo_regions, alignment, physical_route = self._route_boxes(
            template_layout,
            route,
            has_conf,
            aff_logos,
            title=title,
            conference_aspect=conference_aspect,
        )
        layout_mode = "split" if {"aff", "conf"}.issubset(set(logo_regions)) else "combined"
        logo_elements = self._logo_elements(
            state=state,
            logo_regions=logo_regions,
            layout_mode=layout_mode,
            has_conf=has_conf,
            aff_logos=aff_logos,
            affiliation_logo_scale=affiliation_logo_scale,
            conference_logo_scale=conference_logo_scale,
        )
        title_box = self._expand_landscape_title_to_logo_bounds(
            template_layout,
            title_box,
            logo_elements,
        )
        compact_stack = self._uses_compact_portrait_header_stack(template_layout, physical_route)
        if compact_stack:
            text_band = self._portrait_split_text_band(title_box, logo_elements)
            title_box = {**title_box, "x": text_band["x"], "w": text_band["w"]}
        title_box = self._apply_title_vertical_offset(template_layout, title_box, physical_route)
        title_wrap_policy = self._resolve_title_wrap_policy(state, template_layout, route, physical_route)
        if title_wrap_policy == "auto":
            title_wrap_policy, title_font_size, display_title = self._resolve_auto_title_layout(
                title,
                title_box["w"],
                title_box["h"],
                title_font_size,
                author_font_size,
                template_layout,
            )
            if title_wrap_policy == "two_line":
                # A wrapped title dominates the header; keep the authors clearly smaller so
                # the title can be large and fill the width instead of leaving an empty band.
                author_font_size = min(author_font_size, self._wrapped_title_author_font_cap(title_box["h"]))
        else:
            display_title = self._display_title_text(title, title_wrap_policy)
            if title_wrap_policy == "single_line":
                title_font_size = self._fit_single_line_font_size(
                    title,
                    title_box["w"],
                    title_font_size,
                    template_layout,
                )
            else:
                title_font_size = self._fit_wrapped_title_font_size(
                    display_title,
                    title_box["w"],
                    title_font_size,
                    template_layout,
                    max_height_inches=self._available_title_height(title_box["h"], author_font_size),
                )
        if subtitle_text:
            subtitle_font_size = self._fit_single_line_font_size(
                subtitle_text,
                title_box["w"],
                subtitle_font_size,
                template_layout,
                min_key="subtitle_single_line_min_font_size",
            )
        author_box = self._author_box_for_plan(template_layout, title_box, logo_elements, physical_route)
        author_font_size, author_line_count = self._fit_author_font_size(
            authors,
            author_box["w"],
            author_font_size,
            template_layout,
            physical_route,
        )
        display_authors = self._display_author_text(authors, author_line_count)
        title_line_count = max(1, len([line for line in str(display_title or title).splitlines() if line.strip()]))
        title_metrics = self._title_metrics(
            title_box["h"],
            title_font_size,
            subtitle_font_size,
            author_font_size,
            bool(subtitle_text),
            title_line_count=title_line_count,
            author_line_count=author_line_count,
            compact_stack=compact_stack,
        )
        title_font_family = self.config["typography"]["fonts"].get("title", "Georgia")
        if compact_stack:
            portrait_title_style = (self.config.get("poster_visual_style") or {}).get("portrait_header_main_title") or {}
            if portrait_title_style.get("enabled", False):
                title_font_family = portrait_title_style.get("font_family", title_font_family)
        plan = {
            "selected_template": template_layout.get("template_name"),
            "route": route,
            "physical_route": physical_route,
            "fallback": fallback,
            "logo_scale": max(affiliation_logo_scale, conference_logo_scale),
            "affiliation_logo_scale": affiliation_logo_scale,
            "conference_logo_scale": conference_logo_scale,
            "title_box": title_box,
            "title": {
                "text": title,
                "display_text": display_title,
                "alignment": alignment,
                "font_size": title_font_size,
                "font_family": title_font_family,
                "box_height": title_metrics["title_box_height"],
                "single_line": title_wrap_policy == "single_line",
                "wrap_policy": title_wrap_policy,
            },
            "subtitle": {
                "text": subtitle_text,
                "font_size": subtitle_font_size,
                "box_height": title_metrics["subtitle_box_height"],
                "top_gap_inches": title_metrics["title_to_subtitle_gap_inches"],
                "single_line": bool(self.header_config.get("force_single_line_title", True)),
            },
            "authors": {
                "text": display_authors,
                "original_text": authors,
                "font_size": author_font_size,
                "x": author_box["x"],
                "w": author_box["w"],
                "word_wrap": author_line_count > 1,
                "line_count": author_line_count,
                "box_height": title_metrics["author_box_height"],
                "top_gap_inches": title_metrics["author_top_gap_inches"],
            },
            "logo_regions": logo_regions,
            "logo_elements": logo_elements,
            "validation": {"passed": True, "reason": "ok"},
        }
        plan["validation"] = self._validate_plan(plan, header)
        return plan

    def _display_author_text(self, authors: str, line_count: int) -> str:
        clean = re.sub(r"\s+", " ", str(authors or "")).strip()
        if line_count <= 1 or not clean or "," not in clean:
            return clean
        parts = [part.strip() for part in clean.split(",") if part.strip()]
        if len(parts) < 3:
            return clean
        target_lines = min(max(line_count, 1), 2)
        if target_lines != 2:
            return clean
        best_index = 1
        best_score = float("inf")
        for index in range(1, len(parts)):
            left = ", ".join(parts[:index])
            right = ", ".join(parts[index:])
            score = abs(len(left) - len(right))
            if len(right.split()) <= 2:
                score += 30
            if score < best_score:
                best_score = score
                best_index = index
        return f"{', '.join(parts[:best_index])},\n{', '.join(parts[best_index:])}"

    def _fit_single_line_font_size(
        self,
        text: str,
        width_inches: float,
        desired_size: int,
        template_layout: Dict[str, Any],
        *,
        min_key: str = "title_single_line_min_font_size",
    ) -> int:
        if not self.header_config.get("force_single_line_title", True):
            return int(desired_size)
        clean_text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean_text or width_inches <= 0:
            return int(desired_size)
        if min_key == "subtitle_single_line_min_font_size":
            avg_key = (
                "portrait_subtitle_fit_avg_char_width_em"
                if template_layout.get("orientation") == "portrait"
                else "subtitle_fit_avg_char_width_em"
            )
            avg_char_width = float(
                self.header_config.get(
                    avg_key,
                    self.header_config.get("subtitle_fit_avg_char_width_em", self.header_config.get("title_fit_avg_char_width_em", 0.56)),
                )
            )
        else:
            avg_key = (
                "portrait_title_fit_avg_char_width_em"
                if template_layout.get("orientation") == "portrait"
                else "title_fit_avg_char_width_em"
            )
            avg_char_width = float(
                self.header_config.get(
                    avg_key,
                    self.header_config.get("title_fit_avg_char_width_em", 0.56),
                )
            )
        width_safety_key = (
            "portrait_title_fit_width_safety"
            if template_layout.get("orientation") == "portrait"
            else "title_fit_width_safety"
        )
        width_safety = float(
            self.header_config.get(
                width_safety_key,
                self.header_config.get("title_fit_width_safety", 0.94),
            )
        )
        usable_width = max(width_inches * width_safety, 0.1)
        estimated_size = int((usable_width * 72) / max(len(clean_text) * avg_char_width, 1))
        if min_key == "title_single_line_min_font_size" and template_layout.get("orientation") == "portrait":
            min_size = int(
                self.header_config.get(
                    "portrait_title_single_line_min_font_size",
                    self.header_config.get(min_key, 50),
                )
            )
        else:
            min_size = int(self.header_config.get(min_key, 50))
        return max(min_size, min(int(desired_size), estimated_size))

    def _apply_title_vertical_offset(
        self,
        template_layout: Dict[str, Any],
        title_box: Dict[str, float],
        physical_route: str,
    ) -> Dict[str, float]:
        if template_layout.get("orientation") != "portrait":
            offset = float(self.header_config.get("title_vertical_offset_inches", 0.0) or 0.0)
        elif str(physical_route).startswith("portrait_split_logos"):
            offset = float(
                self.header_config.get(
                    "portrait_split_title_vertical_offset_inches",
                    self.header_config.get("portrait_title_vertical_offset_inches", 0.0),
                )
                or 0.0
            )
        else:
            offset = float(self.header_config.get("portrait_title_vertical_offset_inches", 0.0) or 0.0)

        if offset <= 0:
            return title_box

        header = template_layout.get("header") or {}
        header_bottom = float(header.get("y", 0.0) or 0.0) + float(header.get("h", 0.0) or 0.0)
        min_height = float(self.header_config.get("min_shifted_title_box_height_inches", 0.8) or 0.8)
        max_offset = max(float(title_box.get("h", 0.0) or 0.0) - min_height, 0.0)
        applied = min(offset, max_offset)
        shifted_y = min(float(title_box["y"]) + applied, max(header_bottom - min_height, float(title_box["y"])))
        shifted_h = max(
            min(float(title_box["h"]) - applied, header_bottom - shifted_y),
            min_height,
        )
        return {**title_box, "y": shifted_y, "h": shifted_h}

    def _resolve_title_wrap_policy(
        self,
        state: PosterState,
        template_layout: Dict[str, Any],
        route: str,
        physical_route: str,
    ) -> str:
        requested = str(
            state.get("header_title_wrap_policy")
            or self.header_config.get("title_wrap_policy")
            or "auto"
        ).strip()
        if requested not in self.VALID_TITLE_WRAP_POLICIES:
            requested = "auto"
        if requested == "single_line":
            return "single_line"
        if requested == "two_line":
            return "two_line"
        if (
            template_layout.get("orientation") == "portrait"
            and route == "split_logos"
            and str(physical_route).startswith("portrait_split_logos")
        ):
            return "two_line"
        # Landscape auto: defer the single-vs-two-line choice to _resolve_auto_title_layout,
        # which wraps long titles to a second line instead of shrinking them to an
        # unreadably small single line. Portrait routes keep their prior single-line
        # default (their vertical budget and logo strips are tuned for one line).
        if template_layout.get("orientation") == "portrait":
            return "single_line" if self.header_config.get("force_single_line_title", True) else "two_line"
        return "auto"

    def _wrapped_title_author_strip(self, box_height: float) -> float:
        """Height reserved for the author line under a wrapped (two-line) title. Kept
        compact so the title can grow large and fill the header width instead of leaving
        a big empty band; the author font is capped to match this strip."""
        frac = float(self.header_config.get("wrapped_title_author_strip_frac", 0.19))
        return max(box_height * frac, 0.55)

    def _wrapped_title_author_font_cap(self, box_height: float) -> int:
        """Author font that fits the compact strip reserved under a wrapped title."""
        strip = self._wrapped_title_author_strip(box_height)
        return max(int(strip / 1.12 * 72), 38)

    def _available_title_height(self, box_height: float, author_font_size: int, two_line: bool = False) -> float:
        """Vertical room left for the title text once the author line is placed, mirroring
        the landscape allocation in _title_metrics so the fitted font and the box agree.
        For a two-line title the author strip is compact so the title can be large."""
        author_gap = float(self.config["typography"].get("title_author_gap_points", 16)) / 72
        if two_line:
            author_box_h = self._wrapped_title_author_strip(box_height)
        else:
            author_box_h = min(
                max((author_font_size / 72) * 1.12, 0.45),
                max(box_height * 0.30, 0.45),
            )
        return max(box_height - author_gap - author_box_h, box_height * 0.38)

    def _resolve_auto_title_layout(
        self,
        title: str,
        width_inches: float,
        box_height: float,
        base_font_size: int,
        author_font_size: int,
        template_layout: Dict[str, Any],
    ) -> Tuple[str, int, str]:
        """Prefer a wide single-line title and wrap only below a readable font floor."""
        single_size = self._fit_single_line_font_size(title, width_inches, base_font_size, template_layout)
        gate_key = (
            "portrait_single_line_title_gate_min_font_size"
            if template_layout.get("orientation") == "portrait"
            else "single_line_title_gate_min_font_size"
        )
        single_line_gate = int(self.header_config.get(gate_key, 42))
        clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
        if single_size >= single_line_gate:
            return "single_line", single_size, clean_title

        wrapped_display = self._display_title_text(title, "two_line")
        if len([ln for ln in wrapped_display.splitlines() if ln.strip()]) < 2:
            return "single_line", single_size, clean_title
        two_size = self._fit_wrapped_title_font_size(
            wrapped_display,
            width_inches,
            base_font_size,
            template_layout,
            max_height_inches=self._available_title_height(box_height, author_font_size, two_line=True),
        )
        gain_threshold = int(self.header_config.get("title_auto_wrap_gain_threshold", 6))
        if two_size >= single_size + gain_threshold:
            return "two_line", two_size, wrapped_display
        return "single_line", single_size, clean_title

    def _expand_landscape_title_to_logo_bounds(
        self,
        template_layout: Dict[str, Any],
        title_box: Dict[str, float],
        logo_elements: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Reclaim unused logo-reservation space for the landscape title.

        Route planning reserves a safe region before logo aspect ratios are known. Once
        the logos have their actual boxes, the title can extend to their visible edge.
        """
        if template_layout.get("orientation") == "portrait":
            return title_box

        logos = [element for element in logo_elements if element.get("type") != "logo_divider"]
        if not logos:
            return title_box

        header = template_layout.get("header") or {}
        header_left = float(header.get("x", title_box["x"]))
        header_right = header_left + float(header.get("w", title_box["w"]))
        title_left = float(title_box["x"])
        title_right = title_left + float(title_box["w"])
        gap = max(
            float(self.header_config.get("title_logo_gap_inches", 0.28)),
            float(self.header_config.get("min_title_logo_gap_inches", 0.20)),
        )

        logo_boxes = [self._box_from_wh(element) for element in logos]
        left_logos = [box for box in logo_boxes if box[2] <= title_left + 0.08]
        right_logos = [box for box in logo_boxes if box[0] >= title_right - 0.08]

        if left_logos and not right_logos:
            new_left = max(box[2] for box in left_logos) + gap
            if new_left < header_right:
                return {**title_box, "x": new_left, "w": header_right - new_left}

        if right_logos and not left_logos:
            new_right = min(box[0] for box in right_logos) - gap
            if new_right > header_left:
                return {**title_box, "x": header_left, "w": new_right - header_left}

        if left_logos and right_logos:
            new_left = max(box[2] for box in left_logos) + gap
            new_right = min(box[0] for box in right_logos) - gap
            if new_right > new_left:
                return {**title_box, "x": new_left, "w": new_right - new_left}

        return title_box

    def _display_title_text(self, title: str, wrap_policy: str) -> str:
        clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
        if wrap_policy != "two_line" or not clean_title:
            return clean_title
        return "\n".join(self._split_title_two_lines(clean_title))

    def _split_title_two_lines(self, title: str) -> List[str]:
        words = title.split()
        if len(words) <= 2:
            return [title]
        best_index = 1
        best_score = float("inf")
        strong_break_words = {"for", "with", "via", "using", "under", "through", "toward", "towards"}
        for index in range(1, len(words)):
            left = " ".join(words[:index])
            right = " ".join(words[index:])
            balance_score = abs(len(left) - len(right))
            max_line_penalty = max(len(left), len(right)) * 0.08
            break_bonus = -7 if words[index].lower() in strong_break_words else 0
            score = balance_score + max_line_penalty + break_bonus
            if score < best_score:
                best_score = score
                best_index = index
        return [" ".join(words[:best_index]), " ".join(words[best_index:])]

    def _fit_wrapped_title_font_size(
        self,
        display_title: str,
        width_inches: float,
        desired_size: int,
        template_layout: Dict[str, Any],
        max_height_inches: float | None = None,
    ) -> int:
        clean_lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in str(display_title or "").splitlines()
            if re.sub(r"\s+", " ", line).strip()
        ]
        if not clean_lines or width_inches <= 0:
            return int(desired_size)
        if template_layout.get("orientation") == "portrait":
            avg_char_width = float(
                self.header_config.get(
                    "portrait_wrapped_title_fit_avg_char_width_em",
                    self.header_config.get("wrapped_title_fit_avg_char_width_em", 0.48),
                )
            )
        else:
            avg_char_width = float(self.header_config.get("wrapped_title_fit_avg_char_width_em", 0.48))
        width_safety = float(self.header_config.get("title_fit_width_safety", 0.94))
        usable_width = max(width_inches * width_safety, 0.1)
        max_line_len = max(len(line) for line in clean_lines)
        estimated_size = int((usable_width * 72) / max(max_line_len * avg_char_width, 1))
        if max_height_inches and max_height_inches > 0:
            line_height = float(self.header_config.get("wrapped_title_line_height_em", 1.15))
            height_cap = int((max_height_inches * 72) / max(len(clean_lines) * line_height, 1))
            estimated_size = min(estimated_size, height_cap)
        min_key = (
            "portrait_wrapped_title_min_font_size"
            if template_layout.get("orientation") == "portrait"
            else "title_single_line_min_font_size"
        )
        min_size = int(self.header_config.get(min_key, 44))
        return max(min_size, min(int(desired_size), estimated_size))

    def _choose_checked_logo_plan(
        self,
        *,
        state: PosterState,
        template_layout: Dict[str, Any],
        route: str,
        title: str,
        authors: str,
        subtitle_text: str,
        aff_logos: List[Dict[str, Any]],
        has_conf: bool,
        base_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Try one affiliation-logo enlargement, then keep the checked safe plan."""
        base_plan["logo_resize_decision"] = "base"
        base_plan["logo_resize_attempts"] = [self._logo_attempt_summary(base_plan, "base")]
        if not base_plan["validation"]["passed"]:
            base_plan["logo_resize_decision"] = "base_failed"
            return base_plan
        if not aff_logos or not self.header_config.get("logo_resize_check_enabled", True):
            return base_plan

        preferred_aff_scale = float(self.header_config.get("preferred_affiliation_logo_scale", 1.16))
        preferred_conf_scale = float(self.header_config.get("preferred_conference_logo_scale", 1.0))
        if preferred_aff_scale <= 1.0 and preferred_conf_scale <= 1.0:
            return base_plan

        boosted_plan = self._build_plan(
            state=state,
            template_layout=template_layout,
            route=route,
            title=title,
            authors=authors,
            subtitle_text=subtitle_text,
            aff_logos=aff_logos,
            has_conf=has_conf,
            affiliation_logo_scale=preferred_aff_scale,
            conference_logo_scale=preferred_conf_scale,
            fallback=False,
        )
        attempts = [
            self._logo_attempt_summary(base_plan, "base"),
            self._logo_attempt_summary(boosted_plan, "boosted_affiliation_logo"),
        ]
        if boosted_plan["validation"]["passed"]:
            base_area = self._logo_area(base_plan, "institution_logo")
            boosted_area = self._logo_area(boosted_plan, "institution_logo")
            area_ratio = boosted_area / base_area if base_area > 0 else 1.0
            boosted_plan["effective_affiliation_logo_area_ratio"] = round(area_ratio, 3)
            boosted_plan["logo_resize_decision"] = "boosted_affiliation_logo"
            if area_ratio < float(self.header_config.get("min_effective_logo_area_ratio", 1.08)):
                boosted_plan["logo_resize_decision"] = "boosted_affiliation_logo_cell_limited"
            boosted_plan["logo_resize_attempts"] = attempts
            return boosted_plan

        base_plan["logo_resize_decision"] = "base_after_boost_rejected"
        base_plan["logo_resize_attempts"] = attempts
        return base_plan

    def _logo_area(self, plan: Dict[str, Any], logo_type: str) -> float:
        return sum(
            float(element.get("width", 0.0) or 0.0) * float(element.get("height", 0.0) or 0.0)
            for element in plan.get("logo_elements") or []
            if element.get("type") == logo_type
        )

    def _logo_attempt_summary(self, plan: Dict[str, Any], label: str) -> Dict[str, Any]:
        logo_boxes = []
        for element in plan.get("logo_elements") or []:
            if element.get("type") == "logo_divider":
                continue
            logo_boxes.append({
                "type": element.get("type"),
                "width": round(float(element.get("width", 0.0)), 3),
                "height": round(float(element.get("height", 0.0)), 3),
            })
        return {
            "label": label,
            "affiliation_logo_scale": plan.get("affiliation_logo_scale", plan.get("logo_scale")),
            "conference_logo_scale": plan.get("conference_logo_scale", plan.get("logo_scale")),
            "passed": bool((plan.get("validation") or {}).get("passed")),
            "reason": (plan.get("validation") or {}).get("reason"),
            "logo_boxes": logo_boxes,
        }

    def _font_sizes(self, template_layout: Dict[str, Any], has_subtitle: bool) -> Tuple[int, int, int]:
        orientation = template_layout.get("orientation")
        if orientation == "portrait":
            title_size = int(self.header_config.get("portrait_title_font_size", 58))
            author_size = int(self.header_config.get("portrait_author_font_size", 34))
        else:
            title_size = int(self.header_config.get("landscape_title_font_size", 100))
            author_size = int(self.header_config.get("landscape_author_font_size", 72))
        if has_subtitle:
            title_size = int(title_size * float(self.header_config.get("subtitle_title_scale", 0.94)))
        subtitle_scale_key = "portrait_subtitle_font_scale" if orientation == "portrait" else "subtitle_font_scale"
        subtitle_size = max(
            int(title_size * float(self.header_config.get(subtitle_scale_key, self.header_config.get("subtitle_font_scale", 0.58)))),
            24,
        )
        return title_size, subtitle_size, author_size

    def _route_boxes(
        self,
        template_layout: Dict[str, Any],
        route: str,
        has_conf: bool,
        aff_logos: List[Dict[str, Any]],
        *,
        title: str = "",
        conference_aspect: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], str, str]:
        header = template_layout["header"]
        x0 = float(header["x"])
        y0 = float(header["y"])
        w = float(header["w"])
        h = float(header["h"])
        gap = float(self.header_config.get("title_logo_gap_inches", 0.46))
        vertical_pad = min(max(h * 0.10, 0.12), 0.35)
        title_h = max(h - 0.15, 0.8)
        has_logo = has_conf or bool(aff_logos)

        if template_layout.get("orientation") == "portrait":
            return self._portrait_route_boxes(
                template_layout,
                route,
                has_conf,
                aff_logos,
                title=title,
                conference_aspect=conference_aspect,
            )

        if not has_logo:
            return {"x": x0, "y": y0, "w": w, "h": title_h}, {}, self._alignment_for_route(route), route

        if route in {"centered", "split_logos"} and has_conf and aff_logos:
            side_w = min(max(w * 0.17, 2.8), w * 0.24)
            logo_y = y0 + vertical_pad
            logo_h = max(h - 2 * vertical_pad, 0.65)
            left = {"x": x0, "y": logo_y, "w": side_w, "h": logo_h}
            right = {"x": x0 + w - side_w, "y": logo_y, "w": side_w, "h": logo_h}
            title_x = left["x"] + left["w"] + gap
            title_w = max(right["x"] - title_x - gap, w * 0.42)
            return {"x": title_x, "y": y0, "w": title_w, "h": title_h}, {"aff": left, "conf": right}, "center", route

        explicit_logo = self._rightmost_logo_region(template_layout) if route != "right_title" else None
        reserve_frac = self._reserve_fraction(template_layout, has_conf, len(aff_logos))
        min_logo_w = 2.6 if template_layout.get("orientation") == "portrait" else 4.0
        max_logo_w = w * float(self.header_config.get("max_logo_zone_width_fraction", 0.38))
        logo_w = min(max(w * reserve_frac, min_logo_w), max_logo_w)
        logo_h = max(h - 2 * vertical_pad, 0.65)
        logo_y = y0 + vertical_pad

        if route == "right_title":
            logo_box = {"x": x0, "y": logo_y, "w": logo_w, "h": logo_h}
            title_x = logo_box["x"] + logo_box["w"] + gap
            title_w = max(x0 + w - title_x, w * 0.48)
            return {"x": title_x, "y": y0, "w": min(title_w, x0 + w - title_x), "h": title_h}, {"combined": logo_box}, "right", route

        if explicit_logo:
            logo_box = explicit_logo
            title_w = max(logo_box["x"] - x0 - gap, w * 0.45)
            return {"x": x0, "y": y0, "w": min(title_w, w), "h": title_h}, {"combined": logo_box}, self._alignment_for_route(route), route

        logo_box = {"x": x0 + w - logo_w, "y": logo_y, "w": logo_w, "h": logo_h}
        title_w = max(logo_box["x"] - x0 - gap, w * 0.50)
        return {"x": x0, "y": y0, "w": min(title_w, max(logo_box["x"] - x0 - gap, 0.1)), "h": title_h}, {"combined": logo_box}, self._alignment_for_route(route), route

    def _portrait_route_boxes(
        self,
        template_layout: Dict[str, Any],
        route: str,
        has_conf: bool,
        aff_logos: List[Dict[str, Any]],
        *,
        title: str = "",
        conference_aspect: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], str, str]:
        """Map shared header route intent onto stable portrait geometry.

        Portrait posters cannot afford the landscape side-logo routes for long
        paper titles. The intent still controls title alignment, but logos move
        into a compact strip so title/authors keep the full readable width.
        """
        header = template_layout["header"]
        x0 = float(header["x"])
        y0 = float(header["y"])
        w = float(header["w"])
        h = float(header["h"])
        alignment = self._alignment_for_route(route)
        has_logo = has_conf or bool(aff_logos)
        title_h = max(h - 0.15, 0.8)
        if not has_logo:
            return {"x": x0, "y": y0, "w": w, "h": title_h}, {}, alignment, "portrait_full_title"

        if route == "split_logos" and has_conf and aff_logos:
            side_fraction = float(self.header_config.get("portrait_split_logo_side_width_fraction", 0.15))
            if float(conference_aspect or 1.0) >= float(self.header_config.get("portrait_split_wide_conference_aspect_threshold", 3.0)):
                side_fraction = max(
                    side_fraction,
                    float(self.header_config.get("portrait_split_wide_conference_side_width_fraction", side_fraction)),
                )
            side_w = min(
                max(
                    w * side_fraction,
                    float(self.header_config.get("portrait_split_logo_min_side_width_inches", 3.2)),
                ),
                w * float(self.header_config.get("portrait_split_logo_max_side_width_fraction", 0.18)),
            )
            gap = float(self.header_config.get("portrait_split_title_logo_gap_inches", 0.34))
            vertical_pad = min(max(h * 0.12, 0.16), 0.34)
            logo_y = y0 + vertical_pad
            logo_h = max(h - 2 * vertical_pad, 0.65)
            left = {"x": x0, "y": logo_y, "w": side_w, "h": logo_h}
            right = {"x": x0 + w - side_w, "y": logo_y, "w": side_w, "h": logo_h}
            title_x = left["x"] + left["w"] + gap
            title_w = max(right["x"] - title_x - gap, w * 0.48)
            return (
                {"x": title_x, "y": y0, "w": min(title_w, right["x"] - title_x - gap), "h": title_h},
                {"aff": left, "conf": right},
                "center",
                "portrait_split_logos_title_center",
            )

        gap = float(self.header_config.get("portrait_logo_title_gap_inches", 0.14))
        min_title_h = float(self.header_config.get("portrait_min_title_height_inches", 3.75))
        desired_strip_h = max(
            h * float(self.header_config.get("portrait_logo_strip_height_fraction", 0.30)),
            float(self.header_config.get("portrait_logo_strip_min_height_inches", 1.25)),
        )
        max_strip_h = h * float(self.header_config.get("portrait_logo_strip_max_height_fraction", 0.40))
        strip_h = min(desired_strip_h, max_strip_h)
        if h - strip_h - gap < min_title_h:
            strip_h = max(h - min_title_h - gap, float(self.header_config.get("portrait_logo_strip_min_height_inches", 1.25)))
        strip_h = min(strip_h, max(h - gap - 0.8, 0.65))

        vertical_pad = min(max(strip_h * 0.10, 0.08), 0.18)
        logo_region = {
            "x": x0,
            "y": y0 + vertical_pad,
            "w": w,
            "h": max(strip_h - 2 * vertical_pad, 0.45),
            "portrait_logo_strip": True,
        }
        title_y = y0 + strip_h + gap
        title_box = {
            "x": x0,
            "y": title_y,
            "w": w,
            "h": max(y0 + h - title_y, 0.8),
        }
        return title_box, {"combined": logo_region}, alignment, "portrait_logo_strip_full_title"

    def _alignment_for_route(self, route: str) -> str:
        if route == "centered":
            return "center"
        if route == "right_title":
            return "right"
        return "left"

    def _rightmost_logo_region(self, template_layout: Dict[str, Any]) -> Optional[Dict[str, float]]:
        logo_regions = template_layout.get("logo_regions") or []
        if not logo_regions:
            return None
        region = max(logo_regions, key=lambda item: item.get("x", 0))
        return {
            "x": float(region["x"]),
            "y": float(region["y"]),
            "w": float(region["w"]),
            "h": float(region["h"]),
        }

    def _reserve_fraction(self, template_layout: Dict[str, Any], has_conf: bool, aff_count: int) -> float:
        if template_layout.get("orientation") == "portrait":
            if has_conf and aff_count:
                return 0.34
            if aff_count >= 3:
                return 0.30
            return 0.24
        if has_conf and aff_count:
            return float(self.header_config.get("landscape_combined_logo_zone_fraction", 0.27))
        if aff_count >= 3:
            return float(self.header_config.get("landscape_multi_affiliation_logo_zone_fraction", 0.30))
        return float(self.header_config.get("landscape_single_logo_zone_fraction", 0.24))

    def _logo_elements(
        self,
        *,
        state: PosterState,
        logo_regions: Dict[str, Dict[str, float]],
        layout_mode: str,
        has_conf: bool,
        aff_logos: List[Dict[str, Any]],
        affiliation_logo_scale: float,
        conference_logo_scale: float,
    ) -> List[Dict[str, Any]]:
        if not logo_regions:
            return []
        if layout_mode == "split":
            elements: List[Dict[str, Any]] = []
            elements.extend(self._layout_aff_grid(aff_logos, logo_regions["aff"], affiliation_logo_scale))
            if has_conf:
                elements.extend(self._layout_conf_only(str(state["logo_path"]), logo_regions["conf"], conference_logo_scale))
            return elements

        region = logo_regions.get("combined")
        if not region:
            return []
        if has_conf and aff_logos:
            return self._layout_combined(str(state["logo_path"]), aff_logos, region, affiliation_logo_scale, conference_logo_scale)
        if has_conf:
            return self._layout_conf_only(str(state["logo_path"]), region, conference_logo_scale)
        return self._layout_aff_grid(aff_logos, region, affiliation_logo_scale)

    def _layout_conf_only(self, conf_path: str, region: Dict[str, float], scale: float) -> List[Dict[str, Any]]:
        aspect = self._get_image_aspect_ratio(conf_path)
        max_frac = float(
            self.header_config.get("portrait_logo_strip_max_logo_fraction", 0.88)
            if region.get("portrait_logo_strip")
            else self.header_config.get("max_logo_header_fraction", 0.78)
        )
        logo_h = min(region["h"] * max_frac * scale, region["w"] / max(aspect, 0.1))
        logo_w = logo_h * aspect
        return [{
            "type": "conf_logo",
            "x": region["x"] + (region["w"] - logo_w) / 2,
            "y": region["y"] + (region["h"] - logo_h) / 2,
            "width": logo_w,
            "height": logo_h,
            "priority": 0.9,
            "header_planned": True,
        }]

    def _layout_combined(
        self,
        conf_path: str,
        aff_logos: List[Dict[str, Any]],
        region: Dict[str, float],
        aff_scale: float,
        conf_scale: float,
    ) -> List[Dict[str, Any]]:
        conf_cfg = self.config.get("conference_logos", {})
        divider_w = float(conf_cfg.get("divider_width", 0.04))
        gap = float(conf_cfg.get("divider_gap", 0.22))
        conf_frac = float(conf_cfg.get("conf_zone_fraction", 0.48))
        if region.get("portrait_logo_strip"):
            gap = float(self.header_config.get("portrait_logo_divider_gap_inches", gap))
            conf_frac = float(self.header_config.get("portrait_conf_zone_fraction", min(conf_frac, 0.34)))
        conf_zone_w = region["w"] * conf_frac
        aff_zone_w = max(region["w"] - conf_zone_w - divider_w - 2 * gap, 0.2)
        aff_region = {
            "x": region["x"],
            "y": region["y"],
            "w": aff_zone_w,
            "h": region["h"],
            "portrait_logo_strip": bool(region.get("portrait_logo_strip")),
        }
        conf_region = {
            "x": region["x"] + aff_zone_w + divider_w + 2 * gap,
            "y": region["y"],
            "w": conf_zone_w,
            "h": region["h"],
            "portrait_logo_strip": bool(region.get("portrait_logo_strip")),
        }
        elements = self._layout_aff_grid(aff_logos, aff_region, aff_scale)
        elements.append({
            "type": "logo_divider",
            "x": region["x"] + aff_zone_w + gap,
            "y": region["y"] + region["h"] * 0.08,
            "width": divider_w,
            "height": region["h"] * 0.84,
            "priority": 0.85,
            "header_planned": True,
        })
        elements.extend(self._layout_conf_only(conf_path, conf_region, conf_scale))
        return elements

    def _layout_aff_grid(
        self,
        aff_logos: List[Dict[str, Any]],
        region: Dict[str, float],
        scale: float,
    ) -> List[Dict[str, Any]]:
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

        logo_cfg = self.config.get("affiliation_logos", {})
        gap = float(logo_cfg.get("logo_box_gap", 0.24))
        cell_w = max((region["w"] - (cols - 1) * gap) / cols, 0.2)
        cell_h = max((region["h"] - (rows - 1) * gap) / rows, 0.2)
        configured_max_h = float(logo_cfg.get("max_logo_height", 1.55))
        if region.get("portrait_logo_strip"):
            configured_max_h = max(configured_max_h, float(self.header_config.get("portrait_max_logo_height", configured_max_h)))
        max_h = min(
            configured_max_h,
            region["h"] * float(
                self.header_config.get("portrait_logo_strip_max_logo_fraction", 0.88)
                if region.get("portrait_logo_strip")
                else self.header_config.get("max_logo_header_fraction", 0.78)
            ),
        ) * scale
        cell_h = min(cell_h, max_h)
        grid_h = rows * cell_h + (rows - 1) * gap
        grid_w = cols * cell_w + (cols - 1) * gap
        start_y = region["y"] + max((region["h"] - grid_h) / 2, 0)
        start_x = region["x"] + max((region["w"] - grid_w) / 2, 0)

        elements: List[Dict[str, Any]] = []
        for index, logo in enumerate(aff_logos):
            row, col = divmod(index, cols)
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
                "header_planned": True,
            })
        return elements

    def _uses_compact_portrait_header_stack(self, template_layout: Dict[str, Any], physical_route: str) -> bool:
        return (
            template_layout.get("orientation") == "portrait"
            and str(physical_route).startswith("portrait_split_logos")
        )

    def _author_box_for_plan(
        self,
        template_layout: Dict[str, Any],
        title_box: Dict[str, float],
        logo_elements: List[Dict[str, Any]],
        physical_route: str,
    ) -> Dict[str, float]:
        if not self._uses_compact_portrait_header_stack(template_layout, physical_route):
            return {"x": title_box["x"], "w": title_box["w"]}
        return self._portrait_split_text_band(title_box, logo_elements)

    def _portrait_split_text_band(
        self,
        title_box: Dict[str, float],
        logo_elements: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        non_divider_logos = [element for element in logo_elements if element.get("type") != "logo_divider"]
        if len(non_divider_logos) < 2:
            return {"x": title_box["x"], "w": title_box["w"]}

        left_logo = min(non_divider_logos, key=lambda item: float(item.get("x", 0.0) or 0.0))
        right_logo = max(non_divider_logos, key=lambda item: float(item.get("x", 0.0) or 0.0))
        gap = float(self.header_config.get("min_title_logo_gap_inches", 0.20))
        x = float(left_logo.get("x", 0.0) or 0.0) + float(left_logo.get("width", 0.0) or 0.0) + gap
        right = float(right_logo.get("x", title_box["x"] + title_box["w"]) or title_box["x"] + title_box["w"]) - gap
        if right - x < title_box["w"]:
            return {"x": title_box["x"], "w": title_box["w"]}
        return {"x": x, "w": right - x}

    def _fit_author_font_size(
        self,
        authors: str,
        width_inches: float,
        desired_size: int,
        template_layout: Dict[str, Any],
        physical_route: str,
    ) -> Tuple[int, int]:
        if not self._uses_compact_portrait_header_stack(template_layout, physical_route):
            return int(desired_size), 1

        clean_text = re.sub(r"\s+", " ", str(authors or "")).strip()
        if not clean_text or width_inches <= 0:
            return int(desired_size), 1

        max_lines = max(1, int(self.header_config.get("portrait_author_max_lines", 2)))
        min_size = int(self.header_config.get("portrait_author_min_font_size", 46))
        avg_char_width = float(self.header_config.get("author_fit_avg_char_width_em", 0.44))
        width_safety = float(self.header_config.get("author_fit_width_safety", 0.96))
        usable_width = max(width_inches * width_safety, 0.1)
        target_size = int(desired_size)
        for size in range(target_size, min_size - 1, -1):
            chars_per_line = max(int((usable_width * 72) / max(size * avg_char_width, 1)), 1)
            line_count = max(1, (len(clean_text) + chars_per_line - 1) // chars_per_line)
            if line_count <= max_lines:
                return size, line_count
        return min_size, max_lines

    def _title_metrics(
        self,
        box_height: float,
        title_font_size: int,
        subtitle_font_size: int,
        author_font_size: int,
        has_subtitle: bool,
        *,
        title_line_count: int = 1,
        author_line_count: int = 1,
        compact_stack: bool = False,
    ) -> Dict[str, float]:
        author_gap = float(self.config["typography"].get("title_author_gap_points", 16)) / 72
        if compact_stack:
            author_gap = float(self.header_config.get("portrait_title_author_gap_inches", author_gap))
        subtitle_gap = float(self.header_config.get("title_subtitle_gap_inches", 0.08)) if has_subtitle else 0.0
        author_line_height = float(
            self.header_config.get("portrait_author_line_height_em", 1.08) if compact_stack else 1.12
        )
        author_box_h = min(
            max((author_font_size / 72) * author_line_height * max(author_line_count, 1), 0.45),
            max(box_height * (0.42 if compact_stack else 0.30), 0.45),
        )
        subtitle_box_h = max((subtitle_font_size / 72) * 1.12, 0.36) if has_subtitle else 0.0
        if compact_stack:
            title_line_height = float(self.header_config.get("portrait_title_line_height_em", 1.05))
            desired_title_h = max(
                (title_font_size / 72) * title_line_height * max(title_line_count, 1),
                box_height * 0.28,
            )
            available_title_h = max(box_height - subtitle_gap - subtitle_box_h - author_gap - author_box_h, 0.45)
            title_box_h = min(max(desired_title_h, box_height * 0.28), available_title_h)
        else:
            title_line_height = float(self.header_config.get("wrapped_title_line_height_em", 1.15))
            desired_title_h = (title_font_size / 72) * title_line_height * max(title_line_count, 1)
            available_title_h = box_height - subtitle_gap - subtitle_box_h - author_gap - author_box_h
            title_box_h = min(max(desired_title_h, box_height * 0.46), max(available_title_h, box_height * 0.38))
        if title_box_h + subtitle_gap + subtitle_box_h + author_gap + author_box_h > box_height:
            title_box_h = max(box_height - subtitle_gap - subtitle_box_h - author_gap - author_box_h, box_height * 0.38)
        return {
            "title_box_height": title_box_h,
            "subtitle_box_height": subtitle_box_h,
            "title_to_subtitle_gap_inches": subtitle_gap,
            "author_top_gap_inches": author_gap,
            "author_box_height": author_box_h,
        }

    def _validate_plan(self, plan: Dict[str, Any], header: Dict[str, float]) -> Dict[str, Any]:
        title_box = self._box_from_wh(plan["title_box"])
        header_box = self._box_from_header(header)
        min_gap = float(self.header_config.get("min_title_logo_gap_inches", 0.20))
        if title_box[2] <= title_box[0] or title_box[3] <= title_box[1]:
            return {"passed": False, "reason": "invalid_title_box"}
        if not self._contains(header_box, title_box, tolerance=0.08):
            return {"passed": False, "reason": "title_outside_header"}
        min_title_w = float(header["w"]) * float(self.header_config.get("min_title_width_fraction", 0.42))
        if plan["title_box"]["w"] < min_title_w:
            return {"passed": False, "reason": "title_box_too_narrow"}

        padded_title = (title_box[0] - min_gap, title_box[1] - 0.02, title_box[2] + min_gap, title_box[3] + 0.02)
        max_logo_h = float(header["h"]) * float(self.header_config.get("hard_max_logo_header_fraction", 0.88))
        for element in plan.get("logo_elements") or []:
            if element.get("type") == "logo_divider":
                continue
            box = self._box_from_wh(element)
            if not self._contains(header_box, box, tolerance=0.10):
                return {"passed": False, "reason": f"{element.get('type')}_outside_header"}
            if self._intersects(padded_title, box):
                return {"passed": False, "reason": f"{element.get('type')}_overlaps_title"}
            if float(element.get("height", 0.0)) > max_logo_h:
                return {"passed": False, "reason": f"{element.get('type')}_too_tall"}
        return {"passed": True, "reason": "ok"}

    def _box_from_header(self, box: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (float(box["x"]), float(box["y"]), float(box["x"]) + float(box["w"]), float(box["y"]) + float(box["h"]))

    def _box_from_wh(self, box: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (
            float(box["x"]),
            float(box["y"]),
            float(box["x"]) + float(box.get("w", box.get("width", 0.0))),
            float(box["y"]) + float(box.get("h", box.get("height", 0.0))),
        )

    def _contains(
        self,
        outer: Tuple[float, float, float, float],
        inner: Tuple[float, float, float, float],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        return (
            inner[0] >= outer[0] - tolerance
            and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance
            and inner[3] <= outer[3] + tolerance
        )

    def _intersects(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _get_image_aspect_ratio(self, image_path: str | None) -> float:
        if not image_path or not Path(str(image_path)).exists():
            return float(self.config["layout_constants"]["default_logo_aspect_ratio"])
        from PIL import Image

        with Image.open(str(image_path)) as image:
            return image.size[0] / max(image.size[1], 1)

    def _save_outputs(self, state: PosterState, plan: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "header_plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def header_planner_node(state: PosterState) -> Dict[str, Any]:
    result = HeaderPlanner()(state)
    return {
        **state,
        "header_plan": result.get("header_plan"),
        "resolved_layout_template": result.get("resolved_layout_template"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors"),
    }
