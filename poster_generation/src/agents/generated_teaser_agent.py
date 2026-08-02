"""Generate a paper-specific teaser visual for motivation/introduction blocks."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.tools.image_api import ImageQuotaError, ImageTools
from src.tools.layout_api import LayoutTemplates
from src.utils.image_text_detector import detect_readable_text
from src.utils.text_cleanup import fit_complete_sentence_prefix, normalize_text_for_poster
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class GeneratedTeaserAgent:
    def __init__(self):
        self.name = "generated_teaser_agent"
        self.config = load_config()
        self.teaser_config = self.config.get("generated_teaser", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_generated_teaser", False):
            return state

        log_agent_info(self.name, "generating paper-specific teaser visual")
        try:
            story_board = state.get("story_board") or {}
            sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
            target = self._select_target_section(sections)
            if not target:
                report = self._report(False, "no eligible motivation/introduction section", state)
                state["generated_teaser_report"] = report
                self._save_report(state, report)
                log_agent_warning(self.name, report["reason"])
                return state

            output_dir = Path(state["output_dir"])
            asset_dir = output_dir / "assets" / "generated"
            asset_dir.mkdir(parents=True, exist_ok=True)
            asset_id = str(self.teaser_config.get("asset_id", "generated_teaser_1"))
            raw_path = asset_dir / f"raw_{asset_id}.png"
            final_path = asset_dir / f"{asset_id}.png"
            geometry = self._resolve_teaser_geometry(state, target)
            width = int(geometry["width_px"])
            height = int(geometry["height_px"])
            prompt = self._build_prompt(state, target, geometry)

            image_api_error = ""
            generation_attempts: List[Dict[str, Any]] = []
            postprocess_report: Dict[str, Any] = {
                "fallback_reason": "image_unavailable",
                "ocr_report": {"available": False, "rejected": False, "tokens": [], "reason": "image_unavailable"},
            }
            procedural_only = bool(self.teaser_config.get("procedural_only", False)) or os.getenv(
                "PAPER2POSTER_PROCEDURAL_TEASER"
            ) == "1"
            if procedural_only:
                self._procedural_teaser(width, height, state).save(final_path)
                used_fallback = True
                postprocess_report = {
                    "fallback_reason": "procedural_only",
                    "ocr_report": {"available": False, "rejected": False, "tokens": [], "reason": "procedural_only"},
                }
            else:
                used_fallback = False
                accepted = False
                max_attempts = max(1, int(self.teaser_config.get("validation_retry_attempts", 3) or 3))
                selected_raw_path = raw_path
                for attempt_number in range(1, max_attempts + 1):
                    attempt_raw_path = (
                        raw_path
                        if attempt_number == 1
                        else asset_dir / f"raw_{asset_id}_attempt_{attempt_number}.png"
                    )
                    attempt_raw_path.unlink(missing_ok=True)
                    attempt_prompt = self._validation_retry_prompt(prompt, attempt_number)
                    try:
                        generated_path = ImageTools().generate_image(
                            attempt_prompt,
                            width=width,
                            height=height,
                            output_path=str(attempt_raw_path),
                        )
                        if Path(generated_path).exists() and Path(generated_path) != attempt_raw_path:
                            shutil.copyfile(generated_path, attempt_raw_path)
                    except ImageQuotaError:
                        raise
                    except Exception as exc:
                        image_api_error = str(exc)
                        generation_attempts.append(
                            {"attempt": attempt_number, "accepted": False, "reason": "image_api_failed", "error": image_api_error}
                        )
                        log_agent_warning(self.name, f"image API failed; teaser marked for regeneration: {exc}")
                        break

                    accepted, postprocess_report = self._validate_teaser(
                        attempt_raw_path,
                        final_path,
                        width,
                        height,
                    )
                    generation_attempts.append(
                        {
                            "attempt": attempt_number,
                            "accepted": accepted,
                            "reason": postprocess_report.get("fallback_reason", ""),
                            "ocr_report": postprocess_report.get("ocr_report", {}),
                        }
                    )
                    selected_raw_path = attempt_raw_path
                    if accepted:
                        break
                    if attempt_number < max_attempts:
                        log_agent_warning(
                            self.name,
                            f"generated teaser rejected ({postprocess_report.get('fallback_reason')}); regenerating "
                            f"attempt {attempt_number + 1}/{max_attempts}",
                        )

                raw_path = selected_raw_path
                if not accepted:
                    rejection_reason = postprocess_report.get("fallback_reason") or "image_api_failed"
                    if self._allow_procedural_fallback():
                        self._procedural_teaser(width, height, state).save(final_path)
                        used_fallback = True
                        postprocess_report = {
                            **postprocess_report,
                            "fallback_reason": rejection_reason,
                        }
                    else:
                        final_path.unlink(missing_ok=True)
                        report = {
                            "enabled": True,
                            "source": self.name,
                            "asset_source": "none",
                            "degraded": True,
                            "applied": False,
                            "needs_regeneration": True,
                            "asset_id": asset_id,
                            "target_section_id": target.get("section_id"),
                            "target_section_title": target.get("section_title"),
                            "prompt": prompt,
                            "raw_path": str(raw_path) if raw_path.exists() else "",
                            "teaser_path": "",
                            "width_px": width,
                            "height_px": height,
                            "geometry": geometry,
                            "used_procedural_fallback": False,
                            "fallback_reason": rejection_reason,
                            "generation_attempt_count": len(generation_attempts),
                            "generation_attempts": generation_attempts,
                            "image_api_error": image_api_error,
                            "safety": {
                                "conceptual_only": True,
                                "no_readable_text": True,
                                "readable_text_rejected": rejection_reason == "readable_text_artifacts",
                                "no_fake_numeric_results": True,
                                "no_logos": True,
                            },
                        }
                        state.setdefault("degraded_quality_states", []).append(
                            {
                                "component": self.name,
                                "category": "generated_teaser",
                                "reason": rejection_reason,
                                "fallback": "disabled",
                                "needs_regeneration": True,
                            }
                        )
                        state["generated_teaser_report"] = report
                        state["current_agent"] = self.name
                        self._save_report(state, report)
                        log_agent_warning(self.name, f"teaser unavailable without fallback: {rejection_reason}")
                        return state

            self._inject_teaser_asset(state, target, asset_id, final_path, geometry)
            summary_text = self._compress_target_section_text(target, geometry)

            report = {
                "enabled": True,
                "source": self.name,
                "asset_source": "procedural" if used_fallback else "image_api",
                "degraded": bool(used_fallback and not procedural_only),
                "applied": True,
                "asset_id": asset_id,
                "target_section_id": target.get("section_id"),
                "target_section_title": target.get("section_title"),
                "prompt": prompt,
                "raw_path": str(raw_path) if raw_path.exists() else "",
                "teaser_path": str(final_path),
                "width_px": width,
                "height_px": height,
                "geometry": geometry,
                "summary_text": summary_text,
                "used_procedural_fallback": used_fallback,
                "fallback_reason": postprocess_report.get("fallback_reason", ""),
                "postprocess": postprocess_report,
                "needs_regeneration": False,
                "generation_attempt_count": len(generation_attempts) or 1,
                "generation_attempts": generation_attempts,
                "image_api_error": image_api_error,
                "safety": {
                    "conceptual_only": True,
                    "no_readable_text": True,
                    "readable_text_rejected": postprocess_report.get("fallback_reason") == "readable_text_artifacts",
                    "no_fake_numeric_results": True,
                    "no_logos": True,
                },
            }
            if report["degraded"]:
                state.setdefault("degraded_quality_states", []).append(
                    {
                        "component": self.name,
                        "category": "generated_teaser",
                        "reason": image_api_error or postprocess_report.get("fallback_reason") or "placeholder_or_unusable_image",
                        "fallback": "procedural",
                    }
                )
            state["generated_teaser_report"] = report
            state["current_agent"] = self.name
            self._save_report(state, report)
            log_agent_success(self.name, f"generated teaser asset: {final_path}")
        except ImageQuotaError:
            raise
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _select_target_section(self, sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        allow_existing_visual = bool(self.teaser_config.get("allow_existing_visual", False))
        allow_existing_visual = allow_existing_visual or os.getenv("PAPER2POSTER_TEASER_ALLOW_EXISTING_VISUAL") == "1"
        candidates = self._rank_target_sections(sections, allow_existing_visual=allow_existing_visual)

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _rank_target_sections(
        self,
        sections: List[Dict[str, Any]],
        allow_existing_visual: bool,
    ) -> List[tuple[int, int, Dict[str, Any]]]:
        candidates = []
        for index, section in enumerate(sections):
            if section.get("visual_assets") and not allow_existing_visual:
                continue
            title = str(section.get("section_title") or "").lower()
            section_id = str(section.get("section_id") or "").lower()
            role = str(section.get("content_role") or section.get("content_type") or "").lower()
            text = f"{title} {section_id} {role}"
            score = 0
            semantic_score = 0
            if role in {"foundation", "overview", "problem", "motivation", "background"}:
                score += 80
                semantic_score += 80
            if any(keyword in text for keyword in ("intro", "motivation", "problem", "why", "background", "challenge")):
                score += 60
                semantic_score += 60
            if str(section.get("preferred_slot_id") or section.get("slot_id") or section.get("column_assignment") or "") == "slot_1":
                score += 15
            if not section.get("visual_assets"):
                score += 10
            if semantic_score <= 0:
                continue
            candidates.append((score, -index, section))
        return [item for item in candidates if item[0] > 0]

    def _build_prompt(self, state: PosterState, section: Dict[str, Any], geometry: Dict[str, Any]) -> str:
        title = self._poster_title(state)
        section_title = str(section.get("section_title") or "Motivation")
        section_text = " ".join(str(item) for item in section.get("text_content") or [])[:900]
        keypoints = self._keypoint_context(state)
        colors = state.get("color_scheme") or {}
        theme = colors.get("theme", "#1E3A8A")
        style = str(self.teaser_config.get("prompt_style", "top-tier AI conference teaser figure"))
        return (
            f"Create a clean {style} for an academic poster. "
            "This is a conceptual teaser/motivation figure, not a quantitative results chart. "
            "Use no readable text, no letters, no numbers, no punctuation marks, no question marks, "
            "no text-like symbols, no fake axes, no fake tables, no logos, and no watermarks. "
            "Do not invent numeric results or dataset values. "
            "Avoid UI-style pictograms and icon rows; prefer unlabeled parcels, paths, regions, heat fields, "
            "abstract agents, and clean schematic geometry. "
            "Use abstract but paper-specific visual metaphors and a polished conference-poster aesthetic. "
            "The image should work as an extra-wide panoramic teaser banner placed under an introduction or motivation heading. "
            f"Design for a poster slot aperture of {geometry['target_width_inches']:.2f} inches wide by "
            f"{geometry['target_height_inches']:.2f} inches tall, aspect ratio {geometry['aspect']:.2f}:1, "
            f"rendered at {geometry['width_px']}x{geometry['height_px']} pixels. "
            "Use the full horizontal canvas; keep important visual content inside the central 88% width and 82% height. "
            f"Primary accent color: {theme}. "
            f"Paper title: {title}. "
            f"Target section: {section_title}. "
            f"Section facts: {section_text}. "
            f"Poster keypoints: {keypoints}."
        )

    def _poster_title(self, state: PosterState) -> str:
        story_board = state.get("story_board") or {}
        title = story_board.get("title") or story_board.get("poster_title")
        if title:
            return str(title)
        narrative = state.get("narrative_content") or {}
        meta = narrative.get("meta") if isinstance(narrative.get("meta"), dict) else {}
        return str(
            narrative.get("title")
            or narrative.get("poster_title")
            or meta.get("poster_title")
            or state.get("poster_name")
            or "research paper"
        )

    def _keypoint_context(self, state: PosterState) -> str:
        keypoints = state.get("paper_poster_keypoints") or []
        snippets = []
        for item in keypoints[:6]:
            if isinstance(item, dict):
                text = (
                    item.get("key_point")
                    or item.get("keypoint")
                    or item.get("claim")
                    or item.get("summary")
                    or item.get("text")
                )
                if text:
                    snippets.append(str(text))
            elif item:
                snippets.append(str(item))
        return " | ".join(snippets)[:1200]

    def _resolve_teaser_geometry(self, state: PosterState, section: Dict[str, Any]) -> Dict[str, Any]:
        template_name = str(
            state.get("resolved_layout_template")
            or state.get("layout_template")
            or self.config.get("templates", {}).get("default")
            or "three_column_postergen"
        )
        layout = self._load_template_layout(state, template_name)
        region = self._match_target_region(layout, section)

        fallback_aspect = float(self.teaser_config.get("aspect", 2.75) or 2.75)
        fallback_width_px = int(self.teaser_config.get("width_px", 1800) or 1800)
        if not region:
            fallback_height_px = max(384, int(round(fallback_width_px / max(fallback_aspect, 0.2))))
            return {
                "source": "config_fallback",
                "template_id": template_name,
                "slot_id": str(section.get("preferred_slot_id") or section.get("slot_id") or section.get("column_assignment") or ""),
                "slot_width_inches": 0.0,
                "slot_height_inches": 0.0,
                "target_width_inches": round(fallback_aspect * 2.0, 4),
                "target_height_inches": 2.0,
                "aspect": round(fallback_aspect, 4),
                "width_px": fallback_width_px,
                "height_px": fallback_height_px,
            }

        slot_w = max(float(region.get("w", region.get("width", 0.0)) or 0.0), 0.1)
        slot_h = max(float(region.get("h", region.get("height", 0.0)) or 0.0), 0.1)
        orientation = str(layout.get("orientation") or "").lower()
        if not orientation:
            orientation = "portrait" if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0) else "landscape"
        is_portrait = orientation == "portrait"
        layout_cfg = self.config.get("layout", {})
        padding = float((layout_cfg.get("text_padding") or {}).get("left_right", 0.24) or 0.24)
        title_gap = float(layout_cfg.get("title_to_content_spacing", 0.4) or 0.4)
        visual_gap = float((layout_cfg.get("visual_spacing") or {}).get("below_visual", 0.18) or 0.18)
        title_h = max(0.42, min(slot_h * 0.07, 0.72))
        text_h = float(self.teaser_config.get("summary_text_height_inches", 1.15) or 1.15)
        safety = float(
            self.teaser_config.get("portrait_geometry_safety_inches", 0.45)
            if is_portrait
            else self.teaser_config.get("geometry_safety_inches", 0.18)
        )
        target_w = max(slot_w - 2 * padding, 1.0)
        content_h = max(slot_h - title_h - title_gap - visual_gap - safety, 1.0)
        if is_portrait:
            target_fraction = float(
                self.teaser_config.get(
                    "portrait_target_block_height_fraction",
                    self.teaser_config.get("target_block_height_fraction", 0.72),
                )
                or 0.72
            )
            max_fraction = float(
                self.teaser_config.get(
                    "portrait_max_block_height_fraction",
                    self.teaser_config.get("max_block_height_fraction", 0.74),
                )
                or 0.74
            )
        else:
            target_fraction = float(self.teaser_config.get("target_block_height_fraction", 0.72) or 0.72)
            max_fraction = float(self.teaser_config.get("max_block_height_fraction", 0.74) or 0.74)
        available_h = max(content_h - text_h, 0.9)
        target_h = min(slot_h * target_fraction, slot_h * max_fraction, available_h)
        target_h = max(target_h, min(slot_h * 0.55, 3.0))
        min_aspect = float(
            self.teaser_config.get(
                "portrait_min_aspect" if is_portrait else "min_aspect",
                1.55 if is_portrait else 1.05,
            )
            or (1.55 if is_portrait else 1.05)
        )
        target_h = min(target_h, target_w / max(min_aspect, 0.2))
        target_h = max(target_h, 0.85)
        aspect = max(target_w / max(target_h, 0.1), min_aspect)

        max_px_w = int(self.teaser_config.get("width_px", 1800) or 1800)
        width_px = max(1024, min(max_px_w, int(round(target_w * 110))))
        height_px = int(round(width_px / aspect))
        height_px = max(360, min(int(self.teaser_config.get("height_px", 650) or 650), height_px))
        width_px = int(round(height_px * aspect))

        return {
            "source": "template_slot",
            "template_id": template_name,
            "orientation": orientation,
            "slot_id": str(region.get("region_id") or region.get("slot_id") or region.get("id") or ""),
            "slot_width_inches": round(slot_w, 4),
            "slot_height_inches": round(slot_h, 4),
            "target_width_inches": round(target_w, 4),
            "target_height_inches": round(target_h, 4),
            "reserved_text_height_inches": round(text_h, 4),
            "target_block_height_fraction": round(target_fraction, 4),
            "aspect": round(aspect, 4),
            "width_px": width_px,
            "height_px": height_px,
        }

    def _load_template_layout(self, state: PosterState, template_name: str) -> Dict[str, Any]:
        margin = float(self.config.get("layout", {}).get("poster_margin", 1.0) or 1.0)
        col_gap = float(self.config.get("layout", {}).get("column_spacing", 1.0) or 1.0)
        return LayoutTemplates(
            float(state.get("poster_width", 54.0) or 54.0),
            float(state.get("poster_height", 36.0) or 36.0),
            margin=margin,
            col_gap=col_gap,
        ).get_template(template_name)

    def _match_target_region(self, layout: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for key in ("regions", "content_slots", "lanes", "columns"):
            for item in layout.get(key) or []:
                if isinstance(item, dict) and (item.get("w") or item.get("width")) and (item.get("h") or item.get("height")):
                    regions.append(item)
        if not regions:
            return None

        preferred_ids = [
            section.get("preferred_slot_id"),
            section.get("slot_id"),
            section.get("region_id"),
            section.get("column_assignment"),
        ]
        preferred_ids = [str(item) for item in preferred_ids if item]
        for preferred in preferred_ids:
            for region in regions:
                region_ids = {
                    str(region.get("region_id") or ""),
                    str(region.get("slot_id") or ""),
                    str(region.get("id") or ""),
                }
                if preferred in region_ids:
                    return region

        semantic_lane = str(section.get("column_assignment") or "").lower()
        if semantic_lane in {"left", "middle", "right"}:
            for region in regions:
                if str(region.get("semantic_lane") or region.get("id") or "").lower() == semantic_lane:
                    return region

        anchor = str(layout.get("recommended_visual_anchor") or layout.get("hero_region_id") or "")
        if anchor:
            for region in regions:
                if anchor in {
                    str(region.get("region_id") or ""),
                    str(region.get("slot_id") or ""),
                    str(region.get("id") or ""),
                }:
                    return region

        return sorted(regions, key=lambda item: (float(item.get("y", 0.0) or 0.0), float(item.get("x", 0.0) or 0.0)))[0]

    def _estimate_text_reserve_inches(self, section: Dict[str, Any], width_inches: float) -> float:
        text = " ".join(str(item).strip() for item in section.get("text_content") or [] if str(item).strip())
        if not text:
            return 0.65
        chars_per_line = max(width_inches * 9.5, 28)
        line_count = max(2, min(7, int(len(text) / chars_per_line) + 1))
        return min(max(line_count * 0.42 + 0.25, 1.15), 3.2)

    def _compress_target_section_text(self, section: Dict[str, Any], geometry: Optional[Dict[str, Any]] = None) -> List[str]:
        is_portrait = str((geometry or {}).get("orientation") or "").lower() == "portrait"
        if is_portrait:
            max_items = max(1, int(self.teaser_config.get("portrait_summary_max_items", self.teaser_config.get("summary_max_items", 2)) or 1))
            max_chars = max(80, int(self.teaser_config.get("portrait_summary_max_chars", self.teaser_config.get("summary_max_chars", 260)) or 150))
        else:
            max_items = max(1, int(self.teaser_config.get("summary_max_items", 2) or 2))
            max_chars = max(80, int(self.teaser_config.get("summary_max_chars", 260) or 260))
        original_items = [str(item).strip() for item in section.get("text_content") or [] if str(item).strip()]
        candidates = self._summary_sentence_candidates(original_items)
        if not candidates:
            return []

        summary: List[str] = []
        used = set()
        for sentence in candidates:
            cleaned = normalize_text_for_poster(self._strip_markup(sentence))
            if cleaned and cleaned[-1] not in ".!?":
                cleaned = f"{cleaned}."
            key = cleaned.lower()
            if not cleaned or key in used:
                continue
            projected_chars = sum(len(item) for item in summary) + len(cleaned)
            if projected_chars > max_chars:
                continue
            summary.append(cleaned)
            used.add(key)
            if len(summary) >= max_items:
                break

        if not summary:
            # A complete source sentence is safer than a visually tidy fragment.
            cleaned = normalize_text_for_poster(self._strip_markup(candidates[0]))
            if cleaned:
                if cleaned[-1] not in ".!?":
                    cleaned = f"{cleaned}."
                summary.append(cleaned)

        if summary:
            section["text_content"] = summary
            section["generated_teaser_summary"] = True
            section["generated_teaser_original_text_count"] = len(original_items)
        return summary

    def _summary_sentence_candidates(self, items: List[str]) -> List[str]:
        text = " ".join(self._strip_markup(item) for item in items)
        fragments = [
            fragment.strip(" -;:,.")
            for fragment in re.split(r"(?<=[.!?])\s+|(?:\s+\*\*[A-Z][^*]{1,24}:\*\*\s+)", text)
            if fragment.strip(" -;:,.")
        ]
        cleaned = [self._strip_markup(fragment) for fragment in fragments]
        preferred = [
            item
            for item in cleaned
            if any(keyword in item.lower() for keyword in ("problem", "challenge", "goal", "budget", "limited", "must", "unknown", "search"))
        ]
        remaining = [item for item in cleaned if item not in preferred]
        return preferred + remaining

    def _strip_markup(self, value: str) -> str:
        value = re.sub(r"\*\*(.*?)\*\*", r"\1", str(value or ""))
        value = re.sub(r"\*(.*?)\*", r"\1", value)
        value = re.sub(r"<[^>]+>", "", value)
        return re.sub(r"\s+", " ", value).strip()

    def _truncate_on_word_boundary(self, value: str, limit: int) -> str:
        value = normalize_text_for_poster(self._strip_markup(value))
        return fit_complete_sentence_prefix(value, limit)

    def _validation_retry_prompt(self, prompt: str, attempt_number: int) -> str:
        if attempt_number <= 1:
            return prompt
        return (
            f"{prompt} "
            f"REGENERATION ATTEMPT {attempt_number}: the previous image was rejected because it contained "
            "readable text, text-like marks, or placeholder-like content. Create a genuinely new composition. "
            "Use only unlabeled visual forms and imagery; do not draw captions, labels, interface panels, axes, legends, or typography."
        )

    def _allow_procedural_fallback(self) -> bool:
        value = os.getenv("PAPER2POSTER_ALLOW_GENERATIVE_FALLBACK")
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(self.teaser_config.get("allow_procedural_fallback", False))

    def _validate_teaser(
        self,
        raw_path: Path,
        final_path: Path,
        width: int,
        height: int,
    ) -> tuple[bool, Dict[str, Any]]:
        rejection_reason = "image_unavailable"
        ocr_report: Dict[str, Any] = {
            "available": False,
            "rejected": False,
            "tokens": [],
            "reason": "image_unavailable",
        }
        if raw_path.exists():
            with Image.open(raw_path) as img:
                img = img.convert("RGB")
                rejected = self._is_placeholder_image(img)
                rejection_reason = "placeholder" if rejected else ""
                if not rejected:
                    ocr_report = detect_readable_text(
                        raw_path,
                        timeout_seconds=float(self.teaser_config.get("ocr_timeout_seconds", 15)),
                        min_confidence=float(self.teaser_config.get("ocr_min_confidence", 45)),
                    )
                    if ocr_report.get("rejected"):
                        rejected = True
                        rejection_reason = "readable_text_artifacts"
                if not rejected:
                    img = self._cover_resize(img, width, height)
                    img.save(final_path)
                    return True, {"fallback_reason": "", "ocr_report": ocr_report}

        return False, {"fallback_reason": rejection_reason, "ocr_report": ocr_report}

    def _cover_resize(self, img: Image.Image, width: int, height: int) -> Image.Image:
        ratio = max(width / img.width, height / img.height)
        new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        left = max((img.width - width) // 2, 0)
        top = max((img.height - height) // 2, 0)
        return img.crop((left, top, left + width, top + height))

    def _is_placeholder_image(self, img: Image.Image) -> bool:
        stat = ImageStat.Stat(img.resize((32, 32)).convert("RGB"))
        mean = sum(stat.mean) / 3
        variance = sum(stat.var) / 3
        return variance < 18 and 175 <= mean <= 230

    def _procedural_teaser(self, width: int, height: int, state: PosterState) -> Image.Image:
        colors = state.get("color_scheme") or {}
        theme = self._parse_hex(colors.get("theme", "#1E3A8A"))
        navy = self._parse_hex(colors.get("mono_dark", "#06134A"))
        rng = random.Random(str(state.get("poster_name") or "generated-teaser"))
        img = Image.new("RGB", (width, height), (248, 251, 255))
        base = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        grid_step_x = max(70, width // 14)
        grid_step_y = max(55, height // 10)
        for x0 in range(-width // 2, width + grid_step_x, grid_step_x):
            draw.line(
                [(x0, 0), (x0 + int(width * 0.28), height)],
                fill=(*theme, 24),
                width=max(2, width // 420),
            )
        for y0 in range(-height // 3, height + grid_step_y, grid_step_y):
            draw.line(
                [(0, y0), (width, y0 - int(height * 0.18))],
                fill=(*navy, 16),
                width=max(1, width // 620),
            )

        heat = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        heat_draw = ImageDraw.Draw(heat, "RGBA")
        for cx, cy, scale, color in (
            (0.35, 0.46, 0.22, theme),
            (0.63, 0.42, 0.18, theme),
            (0.52, 0.62, 0.15, navy),
        ):
            rx = int(width * scale)
            ry = int(height * scale * 0.72)
            x = int(width * cx)
            y = int(height * cy)
            heat_draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*color, 42))
        heat = heat.filter(ImageFilter.GaussianBlur(max(18, width // 36)))
        base = Image.alpha_composite(base, heat)
        draw = ImageDraw.Draw(base, "RGBA")

        parcel_origin_x = int(width * 0.11)
        parcel_origin_y = int(height * 0.18)
        parcel_w = int(width * 0.048)
        parcel_h = int(height * 0.062)
        for row in range(7):
            for col in range(14):
                if rng.random() < 0.14:
                    continue
                x = parcel_origin_x + col * int(parcel_w * 1.15) + row * int(parcel_w * 0.2)
                y = parcel_origin_y + row * int(parcel_h * 1.18)
                if x > width * 0.89 or y > height * 0.82:
                    continue
                active = rng.random()
                color = theme if active > 0.72 else navy if active < 0.2 else (125, 146, 170)
                alpha = 34 if color == (125, 146, 170) else 52 + int(active * 42)
                draw.rounded_rectangle(
                    [x, y, x + parcel_w, y + parcel_h],
                    radius=max(4, width // 420),
                    fill=(*color, alpha),
                    outline=(255, 255, 255, 105),
                    width=max(1, width // 900),
                )

        route = []
        for i in range(8):
            route.append(
                (
                    int(width * (0.16 + i * 0.095 + rng.uniform(-0.025, 0.025))),
                    int(height * (0.72 - (i % 3) * 0.16 + rng.uniform(-0.03, 0.03))),
                )
            )
        draw.line(route, fill=(*navy, 150), width=max(4, width // 230), joint="curve")
        for index, (x, y) in enumerate(route):
            radius = max(13, width // 95) if index in {0, len(route) - 1} else max(9, width // 135)
            fill = (*theme, 210) if index in {2, 5, len(route) - 1} else (255, 255, 255, 225)
            outline = (*navy, 185)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=outline, width=4)
            inner = max(4, radius // 3)
            draw.ellipse([x - inner, y - inner, x + inner, y + inner], fill=(*navy, 85))

        boundary = [
            (int(width * 0.68), int(height * 0.2)),
            (int(width * 0.9), int(height * 0.24)),
            (int(width * 0.86), int(height * 0.74)),
            (int(width * 0.66), int(height * 0.82)),
            (int(width * 0.58), int(height * 0.52)),
        ]
        draw.line(boundary + [boundary[0]], fill=(*theme, 92), width=max(3, width // 280))
        for x, y in boundary:
            draw.ellipse(
                [x - width // 150, y - width // 150, x + width // 150, y + width // 150],
                fill=(*theme, 150),
            )

        img = Image.alpha_composite(img.convert("RGBA"), base).filter(ImageFilter.GaussianBlur(0.25))
        return img.convert("RGB")

    def _parse_hex(self, value: str) -> tuple[int, int, int]:
        value = str(value or "#1E3A8A").lstrip("#")
        if len(value) != 6:
            return (30, 58, 138)
        try:
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return (30, 58, 138)

    def _inject_teaser_asset(
        self,
        state: PosterState,
        section: Dict[str, Any],
        asset_id: str,
        final_path: Path,
        geometry: Dict[str, Any],
    ) -> None:
        visual_assets = dict(state.get("visual_assets") or {})
        aspect = float(geometry.get("aspect") or self.teaser_config.get("aspect", 2.75) or 2.75)
        visual_assets[asset_id] = {
            "asset_id": asset_id,
            "asset_type": "figure",
            "source_path": str(final_path),
            "resolved_path": None,
            "caption": "Generated conceptual teaser visual for the motivation section.",
            "aspect": aspect,
            "provenance": "generated_teaser",
        }
        state["visual_assets"] = visual_assets
        section["visual_assets"] = [
            {
                "visual_id": asset_id,
                "purpose": "Generated paper-specific teaser for motivation/introduction",
                "importance": 1,
            }
        ]

    def _report(self, applied: bool, reason: str, state: PosterState) -> Dict[str, Any]:
        return {
            "enabled": True,
            "source": self.name,
            "applied": applied,
            "reason": reason,
            "paper": state.get("poster_name"),
        }

    def _save_report(self, state: PosterState, report: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "generated_teaser_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if state.get("visual_assets") is not None:
            with open(output_dir / "visual_assets.json", "w", encoding="utf-8") as f:
                json.dump(state.get("visual_assets") or {}, f, indent=2)
        if state.get("story_board") is not None:
            with open(output_dir / "story_board.json", "w", encoding="utf-8") as f:
                json.dump(state.get("story_board") or {}, f, indent=2)


def generated_teaser_agent_node(state: PosterState) -> Dict[str, Any]:
    result = GeneratedTeaserAgent()(state)
    return {
        **state,
        "story_board": result.get("story_board"),
        "visual_assets": result.get("visual_assets"),
        "generated_teaser_report": result.get("generated_teaser_report"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
