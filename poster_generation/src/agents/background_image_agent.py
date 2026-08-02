"""Generated poster background layer.

This agent only creates a decorative, low-contrast background image. It never
edits poster text, figures, tables, or logos; the renderer places the generated
image at the bottom of the slide.
"""

import json
import os
import random
import signal
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageStat

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from src.tools.image_api import ImageQuotaError, ImageTools
from src.utils.image_text_detector import detect_readable_text
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success

T = TypeVar("T")


class BackgroundImageAgent:
    def __init__(self):
        self.name = "background_image_agent"
        self.config = load_config()
        self.background_config = self.config.get("generated_background", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_generated_background", False):
            return state

        style_decision = self._background_style_decision(state)
        palette_name = self._palette_name(state, style_decision)
        log_agent_info(self.name, f"generating light academic background layer ({style_decision['resolved_style']}, {palette_name})")
        try:
            output_dir = Path(state["output_dir"])
            asset_dir = output_dir / "assets"
            asset_dir.mkdir(parents=True, exist_ok=True)

            filename = self.background_config.get("output_filename", "generated_background.png")
            raw_path = asset_dir / f"raw_{filename}"
            final_path = asset_dir / filename
            reference_path = self._reference_poster_path(state)
            prompt = self._build_prompt(state, style_decision, palette_name, conditioned=self._condition_on_poster())

            width, height = self._background_dimensions(state)
            procedural_only = bool(self.background_config.get("procedural_only", False)) or os.getenv(
                "PAPER2POSTER_PROCEDURAL_BACKGROUND"
            ) == "1"
            condition_on_poster = self._condition_on_poster()
            if procedural_only:
                self._save_procedural_fallback(final_path, width, height, state, style_decision, palette_name)
                used_fallback = True
                generation_mode = "procedural_only"
                raw_path_value = ""
                postprocess_report = {
                    "accepted": True,
                    "needs_regeneration": False,
                    "used_procedural_fallback": True,
                    "fallback_reason": "procedural_only",
                }
            else:
                used_fallback = False
                accepted = False
                image_api_error = ""
                generation_attempts: List[Dict[str, Any]] = []
                max_attempts = max(1, int(self.background_config.get("validation_retry_attempts", 3) or 3))
                postprocess_report = {
                    "accepted": False,
                    "needs_regeneration": True,
                    "used_procedural_fallback": False,
                    "fallback_reason": "image_unavailable",
                }
                selected_raw_path = raw_path
                for attempt_number in range(1, max_attempts + 1):
                    attempt_raw_path = (
                        raw_path
                        if attempt_number == 1
                        else asset_dir / f"raw_{final_path.stem}_attempt_{attempt_number}.png"
                    )
                    attempt_raw_path.unlink(missing_ok=True)
                    attempt_prompt = self._validation_retry_prompt(prompt, attempt_number)
                    try:
                        if reference_path and condition_on_poster:
                            generated_path = self._run_image_call_with_timeout(
                                lambda: ImageTools().edit_image(
                                    str(reference_path),
                                    attempt_prompt,
                                    output_path=str(attempt_raw_path),
                                )
                            )
                            if (
                                Path(generated_path).exists()
                                and Path(generated_path) != attempt_raw_path
                                and Path(generated_path) != reference_path
                            ):
                                shutil.copyfile(generated_path, attempt_raw_path)
                        else:
                            self._run_image_call_with_timeout(
                                lambda: ImageTools().generate_image(
                                    attempt_prompt,
                                    width=width,
                                    height=height,
                                    output_path=str(attempt_raw_path),
                                )
                            )
                    except ImageQuotaError:
                        raise
                    except Exception as exc:
                        image_api_error = str(exc)
                        generation_attempts.append(
                            {"attempt": attempt_number, "accepted": False, "reason": "image_api_failed", "error": image_api_error}
                        )
                        log_agent_error(self.name, f"image API failed; background marked for regeneration: {exc}")
                        break

                    selected_raw_path = attempt_raw_path
                    if attempt_raw_path.exists():
                        postprocess_report = self._postprocess_background(
                            attempt_raw_path,
                            final_path,
                            width,
                            height,
                            state,
                            style_decision,
                            palette_name,
                        )
                    else:
                        postprocess_report = {
                            "accepted": False,
                            "needs_regeneration": True,
                            "used_procedural_fallback": False,
                            "fallback_reason": "image_api_failed",
                        }
                    accepted = bool(postprocess_report.get("accepted"))
                    generation_attempts.append(
                        {
                            "attempt": attempt_number,
                            "accepted": accepted,
                            "reason": postprocess_report.get("fallback_reason", ""),
                        }
                    )
                    if accepted:
                        break
                    if attempt_number < max_attempts:
                        log_agent_info(
                            self.name,
                            f"generated background rejected ({postprocess_report.get('fallback_reason')}); regenerating "
                            f"attempt {attempt_number + 1}/{max_attempts}",
                        )

                raw_path = selected_raw_path
                raw_path_value = str(raw_path) if raw_path.exists() else ""
                base_mode = "poster_conditioned_image_api" if reference_path and condition_on_poster else "image_api"
                if accepted:
                    generation_mode = base_mode
                elif self._allow_procedural_fallback():
                    self._save_procedural_fallback(final_path, width, height, state, style_decision, palette_name)
                    used_fallback = True
                    generation_mode = f"{base_mode}_procedural_fallback"
                    postprocess_report = {
                        **postprocess_report,
                        "accepted": True,
                        "needs_regeneration": True,
                        "used_procedural_fallback": True,
                    }
                else:
                    final_path.unlink(missing_ok=True)
                    generation_mode = f"{base_mode}_rejected_no_fallback"
                    postprocess_report = {
                        **postprocess_report,
                        "accepted": False,
                        "needs_regeneration": True,
                        "used_procedural_fallback": False,
                    }
                postprocess_report["generation_attempt_count"] = len(generation_attempts)
                postprocess_report["generation_attempts"] = generation_attempts
                postprocess_report["image_api_error"] = image_api_error

            report = {
                "enabled": True,
                "source": self.name,
                "asset_source": "procedural" if used_fallback else ("image_api" if final_path.exists() else "none"),
                "degraded": bool((used_fallback and not procedural_only) or postprocess_report.get("needs_regeneration")),
                "applied": final_path.exists(),
                "needs_regeneration": bool(postprocess_report.get("needs_regeneration", False)),
                "generation_attempt_count": int(postprocess_report.get("generation_attempt_count", 0) or 0),
                "generation_attempts": postprocess_report.get("generation_attempts", []),
                "image_api_error": str(postprocess_report.get("image_api_error") or ""),
                "generation_mode": generation_mode,
                "prompt": prompt,
                "raw_path": raw_path_value,
                "reference_poster_path": str(reference_path) if reference_path else "",
                "background_image_path": str(final_path) if final_path.exists() else "",
                "width_px": width,
                "height_px": height,
                "requested_style": style_decision["requested_style"],
                "resolved_style": style_decision["resolved_style"],
                "auto_reason": style_decision["reason"],
                "requested_palette": self._requested_palette_name(state),
                "palette": palette_name,
                "resolved_palette": palette_name,
                "used_procedural_fallback": used_fallback,
                "postprocess": postprocess_report,
                "safety": {
                    "background_only": True,
                    "no_text": True,
                    "low_contrast_postprocess": True,
                    "layout_copy_artifacts_rejected": postprocess_report.get("fallback_reason") == "layout_copy_artifacts",
                    "readable_text_rejected": postprocess_report.get("fallback_reason") == "readable_text_artifacts",
                },
            }
            if report["degraded"]:
                state.setdefault("degraded_quality_states", []).append(
                    {
                        "component": self.name,
                        "category": "generated_background",
                        "reason": generation_mode,
                        "fallback": "procedural" if used_fallback else "disabled",
                        "needs_regeneration": report["needs_regeneration"],
                    }
                )
            state["background_image_path"] = str(final_path) if final_path.exists() else None
            state["background_image_report"] = report
            state["current_agent"] = self.name
            self._save_report(state, report)
            if final_path.exists():
                log_agent_success(self.name, f"generated background: {final_path}")
            else:
                log_agent_error(self.name, f"background unavailable without fallback: {postprocess_report.get('fallback_reason')}")
        except ImageQuotaError:
            raise
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _run_image_call_with_timeout(self, call: Callable[[], T]) -> T:
        timeout = self._api_timeout_seconds()
        if timeout <= 0 or not hasattr(signal, "SIGALRM"):
            return call()

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def handle_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError(f"background image API timed out after {timeout:g}s")

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return call()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer and previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    def _api_timeout_seconds(self) -> float:
        value = os.getenv("BACKGROUND_IMAGE_API_TIMEOUT_SECONDS", self.background_config.get("api_timeout_seconds", 75))
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 75.0

    def _validation_retry_prompt(self, prompt: str, attempt_number: int) -> str:
        if attempt_number <= 1:
            return prompt
        return (
            f"{prompt} "
            f"REGENERATION ATTEMPT {attempt_number}: the previous background was rejected because it contained "
            "readable text, copied layout artifacts, or placeholder-like content. Produce a genuinely new background "
            "with no typography, panels, charts, logos, or poster content."
        )

    def _allow_procedural_fallback(self) -> bool:
        value = os.getenv("PAPER2POSTER_ALLOW_GENERATIVE_FALLBACK")
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(self.background_config.get("allow_procedural_fallback", False))

    def _build_prompt(
        self,
        state: PosterState,
        style_decision: Dict[str, Any] | None = None,
        palette_name: str | None = None,
        conditioned: bool = False,
    ) -> str:
        style_decision = style_decision or self._background_style_decision(state)
        palette_name = palette_name or self._palette_name(state, style_decision)
        colors = state.get("color_scheme") or {}
        theme = colors.get("theme", self.config["colors"].get("fallback_theme", "#1E3A8A"))
        mono = colors.get("mono_light", "#E6EAEF")
        base_style = self.background_config.get("prompt_style", "premium academic background")
        style_spec = style_decision.get("spec") or {}
        style_prompt = style_spec.get("prompt") or base_style
        palette = self._palette_spec(state, palette_name)
        palette_label = palette.get("label", "pale blue-gray")
        poster_width = float(state.get("poster_width") or 0.0)
        poster_height = float(state.get("poster_height") or 0.0)
        orientation = "landscape" if poster_width >= poster_height else "portrait"

        reference_clause = (
            "Use the provided poster image only as a spatial and stylistic reference: infer where content blocks, "
            "title, figures, logos, and white spaces are, then create a clean background underneath them. "
            "Do not copy or redraw any visible poster text, logo, figure, table, chart, or number from the reference. "
            if conditioned else
            "No reference image is provided; generate the background purely from this description. "
            "It will sit behind a dense multi-column poster, so keep the entire canvas even, calm, and uncluttered. "
        )

        return (
            f"Create a {orientation} premium academic conference poster BACKGROUND ONLY, with no text, no letters, "
            "no numbers, no logos, no icons, no charts, no diagrams, and no readable symbols. "
            "The background must look intentionally designed, not plain white, while remaining light enough behind dense black poster text. "
            f"{reference_clause}"
            f"Use a refined top-tier AI conference poster palette based on {palette_label}. "
            "Keep text-heavy content regions calm, flat, and almost textureless. "
            "Do not create block frames, title bars, panel fills, headers, footers, or rectangular containers that compete with the real layout. "
            "Concentrate visible decoration in page margins, gutters, corners, and unused whitespace; keep central content apertures clean. "
            "Use subtle abstract scientific geometry appropriate to the paper domain, but do not reproduce any phrase from this instruction. "
            "Do not create AI faces, brains, robots, glowing orbs, or decorative blobs. "
            f"Base finish: {base_style}. Resolved background style: {style_decision['resolved_style']} ({style_spec.get('label', 'custom')}). "
            f"Style visual language: {style_prompt}. Selected background palette: {palette_name}. "
            f"Primary accent color reference: {theme}; pale neutral reference: {mono}."
        )

    def _condition_on_poster(self) -> bool:
        """Whether to condition background generation on the rendered draft poster.

        Disabled by default: conditioning feeds the fully-rendered poster (text and
        all) to an image-edit model, which frequently redraws that text into the
        "background", producing ghosted/duplicated text behind the real content.
        Text-to-image generation from the prompt alone cannot copy text it never
        sees, so it eliminates the ghosting deterministically. Re-enable only with a
        text-free layout reference.
        """
        if os.getenv("PAPER2POSTER_BACKGROUND_CONDITION_ON_POSTER") == "1":
            return True
        return bool(self.background_config.get("condition_on_poster", False))

    def _reference_poster_path(self, state: PosterState) -> Path | None:
        for key in ("poster_preview_path",):
            value = state.get(key)
            if value and Path(str(value)).exists():
                return Path(str(value))
        output_dir = Path(state.get("output_dir") or "")
        poster_name = str(state.get("poster_name") or "")
        for candidate in [
            output_dir / f"{poster_name}_draft.png",
            output_dir / f"{poster_name}.png",
        ]:
            if candidate.exists():
                return candidate
        return None

    def _poster_title(self, state: PosterState) -> str:
        story_board = state.get("story_board") or {}
        title = story_board.get("title") or story_board.get("poster_title")
        if title:
            return str(title)
        for element in state.get("styled_layout") or []:
            if element.get("type") == "title" and element.get("content"):
                return str(element["content"]).splitlines()[0]
        narrative = state.get("narrative_content") or {}
        return str(narrative.get("title") or state.get("poster_name") or "research poster")

    def _section_summaries(self, state: PosterState) -> List[str]:
        story_board = state.get("story_board") or {}
        sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
        summaries = []
        for section in sections:
            title = str(section.get("section_title") or section.get("section_id") or "").strip()
            role = str(section.get("content_role") or "").strip()
            if title and role:
                summaries.append(f"{title} ({role})")
            elif title:
                summaries.append(title)
        return summaries

    def _background_style_decision(self, state: PosterState) -> Dict[str, Any]:
        styles = self.background_config.get("styles") or {}
        requested = self._requested_style_name(state)
        if requested != "auto":
            spec = dict(styles.get(requested) or {})
            return {
                "requested_style": requested,
                "resolved_style": requested,
                "reason": "explicit background style requested",
                "spec": spec,
            }

        scores = {name: 0 for name in styles.keys()}
        context = self._paper_context_text(state)
        keyword_scores = {
            "cartographic": [
                "geospatial", "spatial", "map", "mapping", "region", "urban",
                "search", "route", "city",
            ],
            "tech_grid": [
                "llm", "neural", "multimodal", "agent", "model", "detection", "classifier",
                "network", "security", "phishing", "vision", "embedding", "learning",
            ],
            "blueprint": [
                "theorem", "proof", "optimization", "algorithm", "mechanism", "robustness",
                "certified", "architecture", "system", "pipeline", "framework", "bound",
            ],
            "flat_cartoon": [
                "policy", "social", "outreach", "education", "healthcare", "intervention",
                "field", "community", "human", "decision", "public",
            ],
        }
        for style_name, keywords in keyword_scores.items():
            for keyword in keywords:
                if keyword in context:
                    scores[style_name] = scores.get(style_name, 0) + 18

        section_count = len(((state.get("story_board") or {}).get("spatial_content_plan") or {}).get("sections") or [])
        visual_density = str(state.get("visual_density") or "").lower()
        poster_width = float(state.get("poster_width") or 0.0)
        poster_height = float(state.get("poster_height") or 0.0)
        if visual_density == "rich" or section_count >= 6:
            scores["academic_paper"] = scores.get("academic_paper", 0) + 16
            scores["minimal_solid"] = scores.get("minimal_solid", 0) + 8
        if poster_width and poster_height and poster_width < poster_height:
            scores["minimal_solid"] = scores.get("minimal_solid", 0) + 10
            scores["geometric_soft"] = scores.get("geometric_soft", 0) + 6
        if state.get("enable_generated_teaser"):
            scores["academic_paper"] = scores.get("academic_paper", 0) + 8
            scores["geometric_soft"] = scores.get("geometric_soft", 0) + 6

        if all(value <= 0 for value in scores.values()):
            resolved = "academic_paper" if "academic_paper" in styles else next(iter(styles), "academic_paper")
            reason = "auto default: no strong domain keywords detected"
        else:
            resolved = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
            if scores[resolved] < 20 and "academic_paper" in styles:
                resolved = "academic_paper"
                reason = "auto default: weak domain signal"
            else:
                reason = f"auto selected from paper/layout context; scores={scores}"

        return {
            "requested_style": "auto",
            "resolved_style": resolved,
            "reason": reason,
            "spec": dict(styles.get(resolved) or {}),
        }

    def _requested_style_name(self, state: PosterState) -> str:
        styles = self.background_config.get("styles") or {}
        requested = str(state.get("background_style") or self.background_config.get("style") or "auto").strip()
        if requested == "auto":
            return "auto"
        if requested in styles:
            return requested
        return "auto"

    def _paper_context_text(self, state: PosterState) -> str:
        parts = [self._poster_title(state)]
        parts.extend(self._section_summaries(state))
        for item in state.get("paper_poster_keypoints") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("key_point") or item.get("keypoint") or item.get("summary") or item.get("text") or ""))
            elif item:
                parts.append(str(item))
        story_board = state.get("story_board") or {}
        sections = (story_board.get("spatial_content_plan") or {}).get("sections") or []
        for section in sections:
            parts.extend(str(value) for value in section.get("text_content") or [])
        return " ".join(parts).lower()

    def _postprocess_background(
        self,
        raw_path: Path,
        final_path: Path,
        width: int,
        height: int,
        state: PosterState,
        style_decision: Dict[str, Any],
        palette_name: str,
    ) -> Dict[str, Any]:
        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            fallback_reason = ""
            copy_artifact_report = self._background_copy_artifact_report(img, state)
            ocr_report = detect_readable_text(
                raw_path,
                timeout_seconds=float(self.background_config.get("ocr_timeout_seconds", 15)),
                min_confidence=float(self.background_config.get("ocr_min_confidence", 45)),
            )
            used_fallback = self._is_placeholder_image(img)
            if used_fallback:
                fallback_reason = "placeholder"
            elif copy_artifact_report["rejected"]:
                used_fallback = True
                fallback_reason = "layout_copy_artifacts"
            elif ocr_report.get("rejected"):
                used_fallback = True
                fallback_reason = "readable_text_artifacts"
            if used_fallback:
                final_path.unlink(missing_ok=True)
                return {
                    "accepted": False,
                    "needs_regeneration": True,
                    "used_procedural_fallback": False,
                    "fallback_reason": fallback_reason,
                    "copy_artifact_report": copy_artifact_report,
                    "ocr_report": ocr_report,
                }
            self._save_light_background(img, final_path, width, height, state, style_decision)
            return {
                "accepted": True,
                "needs_regeneration": False,
                "used_procedural_fallback": False,
                "fallback_reason": fallback_reason,
                "copy_artifact_report": copy_artifact_report,
                "ocr_report": ocr_report,
            }

    def _save_procedural_fallback(
        self,
        final_path: Path,
        width: int,
        height: int,
        state: PosterState,
        style_decision: Dict[str, Any],
        palette_name: str,
    ) -> None:
        img = self._procedural_academic_background(width, height, state, style_decision, palette_name)
        self._save_light_background(img, final_path, width, height, state, style_decision)

    def _save_light_background(
        self,
        img: Image.Image,
        final_path: Path,
        width: int,
        height: int,
        state: PosterState,
        style_decision: Dict[str, Any],
    ) -> None:
        postprocess = self._style_postprocess(style_decision)
        img = img.convert("RGB")
        img = self._cover_resize(img, width, height)
        img = ImageEnhance.Color(img).enhance(float(postprocess.get("max_saturation", 0.28)))
        img = img.filter(ImageFilter.GaussianBlur(float(postprocess.get("blur_radius", 1.2))))

        alpha = float(postprocess.get("white_overlay_alpha", 0.76))
        overlay = Image.new("RGB", img.size, "white")
        img = Image.blend(img, overlay, min(max(alpha, 0.0), 1.0))
        img = self._add_layout_hierarchy_to_background(img, state)
        img = self._enforce_background_visibility(img, postprocess)
        img.save(final_path)

    def _style_postprocess(self, style_decision: Dict[str, Any]) -> Dict[str, Any]:
        values = {
            "white_overlay_alpha": self.background_config.get("white_overlay_alpha", 0.76),
            "blur_radius": self.background_config.get("blur_radius", 1.2),
            "max_saturation": self.background_config.get("max_saturation", 0.28),
        }
        values.update((style_decision.get("spec") or {}).get("postprocess") or {})
        return values

    def _enforce_background_visibility(self, img: Image.Image, postprocess: Dict[str, Any]) -> Image.Image:
        visibility = self.background_config.get("visibility_floor") or {}
        enabled = postprocess.get("visibility_floor_enabled", visibility.get("enabled", True))
        if not enabled:
            return img

        try:
            min_distance = float(
                postprocess.get(
                    "min_average_distance_from_white",
                    visibility.get("min_average_distance_from_white", 0),
                )
            )
            min_stddev = float(
                postprocess.get(
                    "min_channel_stddev",
                    visibility.get("min_channel_stddev", 0),
                )
            )
            max_boost = max(
                1.0,
                float(postprocess.get("max_boost_factor", visibility.get("max_boost_factor", 1.0))),
            )
        except (TypeError, ValueError):
            return img

        if min_distance <= 0 and min_stddev <= 0:
            return img

        metrics = self._background_visibility_metrics(img)
        distance = max(metrics["average_distance_from_white"], 0.01)
        stddev = max(metrics["channel_stddev"], 0.01)
        required_boost = 1.0
        if min_distance > 0 and distance < min_distance:
            required_boost = max(required_boost, min_distance / distance)
        if min_stddev > 0 and stddev < min_stddev:
            required_boost = max(required_boost, min_stddev / stddev)

        boost = min(max_boost, required_boost)
        if boost <= 1.02:
            return img

        return img.point(
            lambda value: max(0, min(255, int(round(255 - (255 - value) * boost))))
        )

    def _background_visibility_metrics(self, img: Image.Image) -> Dict[str, float]:
        sample = img.convert("RGB")
        if max(sample.size) > 256:
            scale = 256 / max(sample.size)
            sample = sample.resize(
                (max(1, int(sample.width * scale)), max(1, int(sample.height * scale))),
                Image.Resampling.BILINEAR,
            )
        stat = ImageStat.Stat(sample)
        mean = sum(stat.mean) / 3
        stddev = sum(stat.stddev) / 3
        return {
            "average_distance_from_white": 255 - mean,
            "channel_stddev": stddev,
        }

    def _background_dimensions(self, state: PosterState) -> tuple[int, int]:
        base_width = max(1, int(self.background_config.get("width_px", 1440)))
        base_height = max(1, int(self.background_config.get("height_px", 2035)))
        poster_width = float(state.get("poster_width") or 0.0)
        poster_height = float(state.get("poster_height") or 0.0)
        if poster_width <= 0 or poster_height <= 0:
            return base_width, base_height

        long_side = max(base_width, base_height)
        if poster_width >= poster_height:
            return long_side, max(1, int(round(long_side * poster_height / poster_width)))
        return max(1, int(round(long_side * poster_width / poster_height))), long_side

    def _add_layout_hierarchy_to_background(self, img: Image.Image, state: PosterState) -> Image.Image:
        """Bake only soft layout-aware washes into the bitmap background."""
        poster_width = float(state.get("poster_width") or 0.0)
        poster_height = float(state.get("poster_height") or 0.0)
        layout = state.get("styled_layout") or []
        if poster_width <= 0 or poster_height <= 0 or not layout:
            return img

        palette = self._palette_spec(state)
        edge = self._rgb_tuple(palette.get("edge"), (225, 236, 247))
        theme = self._parse_hex((state.get("color_scheme") or {}).get("theme", "#0057B8"))
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        def sx(value: float) -> int:
            return int(value / poster_width * img.width)

        def sy(value: float) -> int:
            return int(value / poster_height * img.height)

        title = next((item for item in layout if item.get("type") == "title"), None)
        if title:
            x = float(title.get("x", 0.0))
            y = float(title.get("y", 0.0))
            width = min(poster_width - x, max(float(title.get("width", 0.0)) * 1.35, poster_width * 0.35))
            height = max(float(title.get("height", 0.0)), poster_height * 0.06)
            if width > 1.0:
                draw.polygon(
                    [
                        (sx(max(0.0, x - 0.8)), sy(max(0.0, y - 0.4))),
                        (sx(min(poster_width, x + width)), sy(max(0.0, y - 0.2))),
                        (sx(min(poster_width, x + width - 1.2)), sy(min(poster_height, y + height + 0.5))),
                        (sx(max(0.0, x - 1.8)), sy(min(poster_height, y + height + 0.25))),
                    ],
                    fill=(*theme, 14),
                )

        containers = [
            item
            for item in layout
            if item.get("type") == "section_container"
            and float(item.get("x", 0.0) or 0.0) >= poster_width * 0.60
            and float(item.get("y", 0.0) or 0.0) >= poster_height * 0.40
        ]
        if len(containers) >= 2:
            left = min(float(item.get("x", 0.0) or 0.0) for item in containers) - 0.35
            top = min(float(item.get("y", 0.0) or 0.0) for item in containers) - 0.30
            right = max(float(item.get("x", 0.0) or 0.0) + float(item.get("width", 0.0) or 0.0) for item in containers) + 0.25
            bottom = max(float(item.get("y", 0.0) or 0.0) + float(item.get("height", 0.0) or 0.0) for item in containers) + 0.25
            draw.ellipse(
                [
                    sx(max(0.0, left)),
                    sy(max(0.0, top)),
                    sx(min(poster_width, right)),
                    sy(min(poster_height, bottom)),
                ],
                fill=(*edge, 38),
            )

        overlay = overlay.filter(ImageFilter.GaussianBlur(max(8, min(img.size) // 90)))
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def _is_placeholder_image(self, img: Image.Image) -> bool:
        stat = ImageStat.Stat(img.resize((32, 32)).convert("RGB"))
        mean = sum(stat.mean) / 3
        variance = sum(stat.var) / 3
        return variance < 18 and 175 <= mean <= 230

    def _background_copy_artifact_report(self, img: Image.Image, state: PosterState) -> Dict[str, Any]:
        """Detect generated backgrounds that copied poster text or title bars.

        Prompt-only safety is not enough for image-edit models: a conditioned
        background can redraw faint section headers and panels from the draft
        poster. The background layer must stay decorative, so obvious dark text
        in the header or repeated section-title bands forces procedural fallback.
        """
        poster_width = float(state.get("poster_width") or 0.0)
        poster_height = float(state.get("poster_height") or 0.0)
        if poster_width <= 0 or poster_height <= 0:
            return {
                "rejected": False,
                "reason": "",
                "header": {},
                "contaminated_title_regions": 0,
                "title_regions": [],
            }

        sample = img.convert("RGB")
        header_height = max(1, int(round(sample.height * 0.20)))
        header_metrics = self._crop_darkness_metrics(sample.crop((0, 0, sample.width, header_height)))
        layout = state.get("styled_layout") or []
        title_regions = []
        contaminated_regions = 0
        for element in layout:
            if element.get("type") not in {"section_title", "title_accent_block"}:
                continue
            crop = self._layout_element_crop(sample, element, poster_width, poster_height, expand_y=0.18)
            if crop is None:
                continue
            metrics = self._crop_darkness_metrics(crop)
            contaminated = metrics["dark_fraction"] >= 0.020 and metrics["stddev"] >= 12.0
            if contaminated:
                contaminated_regions += 1
            title_regions.append(
                {
                    "section_id": element.get("section_id"),
                    "type": element.get("type"),
                    "dark_fraction": round(metrics["dark_fraction"], 4),
                    "very_dark_fraction": round(metrics["very_dark_fraction"], 4),
                    "stddev": round(metrics["stddev"], 2),
                    "contaminated": contaminated,
                }
            )

        header_contaminated = (
            header_metrics["dark_fraction"] >= 0.030
            and header_metrics["very_dark_fraction"] >= 0.012
            and header_metrics["stddev"] >= 14.0
        )
        title_contaminated = contaminated_regions >= 2
        rejected = header_contaminated or title_contaminated
        reason = ""
        if header_contaminated and title_contaminated:
            reason = "header_text_and_section_title_copy"
        elif header_contaminated:
            reason = "header_text_copy"
        elif title_contaminated:
            reason = "section_title_copy"

        return {
            "rejected": rejected,
            "reason": reason,
            "header": {
                "dark_fraction": round(header_metrics["dark_fraction"], 4),
                "very_dark_fraction": round(header_metrics["very_dark_fraction"], 4),
                "stddev": round(header_metrics["stddev"], 2),
                "contaminated": header_contaminated,
            },
            "contaminated_title_regions": contaminated_regions,
            "title_region_count": len(title_regions),
            "title_regions": title_regions[:12],
        }

    def _layout_element_crop(
        self,
        img: Image.Image,
        element: Dict[str, Any],
        poster_width: float,
        poster_height: float,
        *,
        expand_y: float = 0.0,
    ) -> Image.Image | None:
        width = float(element.get("width", 0.0) or 0.0)
        height = float(element.get("height", 0.0) or 0.0)
        if width <= 0 or height <= 0:
            return None
        x = float(element.get("x", 0.0) or 0.0)
        y = float(element.get("y", 0.0) or 0.0)
        top = max(0, int(round((y - expand_y) / poster_height * img.height)))
        bottom = min(img.height, int(round((y + height + expand_y) / poster_height * img.height)))
        left = max(0, int(round(x / poster_width * img.width)))
        right = min(img.width, int(round((x + width) / poster_width * img.width)))
        if right <= left or bottom <= top:
            return None
        return img.crop((left, top, right, bottom))

    def _crop_darkness_metrics(self, crop: Image.Image) -> Dict[str, float]:
        gray = crop.convert("L")
        if max(gray.size) > 128:
            scale = 128 / max(gray.size)
            gray = gray.resize(
                (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
                Image.Resampling.BILINEAR,
            )
        stat = ImageStat.Stat(gray)
        hist = gray.histogram()
        total = max(1, sum(hist))
        return {
            "mean": float(stat.mean[0]),
            "stddev": float(stat.stddev[0]),
            "dark_fraction": sum(hist[:175]) / total,
            "very_dark_fraction": sum(hist[:120]) / total,
        }

    def _procedural_academic_background(
        self,
        width: int,
        height: int,
        state: PosterState,
        style_decision: Dict[str, Any] | None = None,
        palette_name: str | None = None,
    ) -> Image.Image:
        style_decision = style_decision or self._background_style_decision(state)
        style_name = str(style_decision.get("resolved_style") or "academic_paper")
        palette_name = palette_name or self._palette_name(state, style_decision)
        palette = self._palette_spec(state, palette_name)
        base_color = self._rgb_tuple(palette.get("base"), (250, 252, 255))
        edge_color = self._rgb_tuple(palette.get("edge"), (225, 236, 247))
        line_color = self._rgba_tuple(palette.get("line"), (0, 87, 184, 24))
        node_color = self._rgba_tuple(palette.get("node"), (0, 87, 184, 30))
        neutral_color = self._rgba_tuple(palette.get("neutral"), (116, 132, 150, 24))
        rng = random.Random(str(state.get("poster_name") or "poster-background"))
        img = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(img, "RGBA")

        # Subtle paper texture.
        pixels = img.load()
        edge_span = max(1.0, min(width, height) * 0.26)
        for y in range(height):
            for x in range(width):
                jitter = rng.randint(-2, 2)
                edge_distance = min(x, width - 1 - x, y, height - 1 - y)
                edge_weight = max(0.0, min(1.0, 1.0 - (edge_distance / edge_span))) * 0.22
                pixel = tuple(
                    max(0, min(255, int(base_color[channel] * (1.0 - edge_weight) + edge_color[channel] * edge_weight) + jitter))
                    for channel in range(3)
                )
                pixels[x, y] = pixel

        if style_name == "minimal_solid":
            draw.polygon(
                [
                    (int(width * 0.72), 0),
                    (width, 0),
                    (width, int(height * 0.22)),
                    (int(width * 0.83), int(height * 0.14)),
                ],
                fill=(*edge_color, 22),
            )
            draw.arc(
                (int(width * -0.08), int(height * 0.78), int(width * 0.22), int(height * 1.10)),
                210,
                350,
                fill=neutral_color,
                width=max(1, width // 900),
            )
            return img

        if style_name == "flat_cartoon":
            for cx, cy, rx, ry, color in [
                (0.08, 0.18, 0.13, 0.10, edge_color),
                (0.92, 0.82, 0.18, 0.13, self._rgb_tuple(palette.get("node"), edge_color)),
                (0.78, 0.10, 0.14, 0.08, self._rgb_tuple(palette.get("neutral"), edge_color)),
            ]:
                x = int(width * cx)
                y = int(height * cy)
                draw.ellipse(
                    [x - int(width * rx), y - int(height * ry), x + int(width * rx), y + int(height * ry)],
                    fill=(*color[:3], 48),
                )

        # Broad conference-poster washes; postprocessing keeps them behind text.
        band_alpha = max(24, min(62, int(line_color[3] * 1.55)))
        cyan_wash = (120, 205, 230, max(14, band_alpha - 10))
        indigo_wash = (105, 128, 210, max(10, band_alpha - 18))
        draw.polygon(
            [
                (int(width * 0.55), 0),
                (width, 0),
                (width, int(height * 0.28)),
                (int(width * 0.70), int(height * 0.19)),
            ],
            fill=(*edge_color, band_alpha),
        )
        draw.polygon(
            [
                (0, int(height * 0.74)),
                (int(width * 0.42), int(height * 0.82)),
                (int(width * 0.30), height),
                (0, height),
            ],
            fill=cyan_wash,
        )
        draw.polygon(
            [
                (int(width * 0.80), int(height * 0.58)),
                (width, int(height * 0.50)),
                (width, height),
                (int(width * 0.90), height),
            ],
            fill=indigo_wash,
        )

        if style_name == "cartographic":
            for idx in range(14):
                y0 = int(height * (0.08 + idx * 0.065))
                points = []
                for step in range(9):
                    x = int(width * (step / 8))
                    y = y0 + int(height * 0.015 * rng.uniform(-1.0, 1.0)) + int(16 * rng.uniform(-1.0, 1.0))
                    points.append((x, y))
                draw.line(points, fill=line_color, width=max(1, width // 900))
            for idx in range(18):
                x = int(width * rng.uniform(0.02, 0.98))
                y = int(height * rng.uniform(0.02, 0.98))
                box_w = int(width * rng.uniform(0.035, 0.08))
                box_h = int(height * rng.uniform(0.025, 0.06))
                draw.rectangle([x, y, x + box_w, y + box_h], outline=neutral_color, width=1)
            return img

        if style_name == "blueprint":
            step = max(44, min(width, height) // 12)
            for x in range(0, width, step):
                draw.line([(x, 0), (x, height)], fill=neutral_color, width=1)
            for y in range(0, height, step):
                draw.line([(0, y), (width, y)], fill=neutral_color, width=1)

        def point_near_edge() -> tuple[int, int]:
            side = rng.choice(["left", "right", "top", "bottom"])
            if side == "left":
                return rng.randint(0, int(width * 0.18)), rng.randint(0, height)
            if side == "right":
                return rng.randint(int(width * 0.78), width), rng.randint(0, height)
            if side == "top":
                return rng.randint(0, width), rng.randint(0, int(height * 0.18))
            return rng.randint(0, width), rng.randint(int(height * 0.82), height)

        nodes = [point_near_edge() for _ in range(34)]
        for idx, start in enumerate(nodes):
            nearest = sorted(nodes, key=lambda point: (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)[1:3]
            for end in nearest:
                if rng.random() < 0.56:
                    color = line_color if idx % 3 else neutral_color
                    draw.line([start, end], fill=color, width=1)

        for x, y in nodes:
            radius = rng.choice([2, 2, 3, 4])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=node_color)

        # Faint ranking-flow arcs along corners, away from dense content.
        for _ in range(8):
            margin_x = rng.choice([rng.randint(-120, 80), rng.randint(width - 80, width + 120)])
            margin_y = rng.randint(-80, height + 80)
            box_w = rng.randint(int(width * 0.22), int(width * 0.42))
            box_h = rng.randint(int(height * 0.10), int(height * 0.22))
            bbox = (margin_x, margin_y, margin_x + box_w, margin_y + box_h)
            draw.arc(bbox, rng.randint(0, 180), rng.randint(190, 360), fill=line_color, width=2)

        return img

    def _requested_palette_name(self, state: PosterState) -> str:
        requested = str(state.get("background_palette") or self.background_config.get("palette") or "auto").strip()
        return requested or "auto"

    def _palette_name(self, state: PosterState, style_decision: Dict[str, Any] | None = None) -> str:
        palettes = self.background_config.get("palettes") or {}
        requested = self._requested_palette_name(state)
        if requested == "auto":
            style_decision = style_decision or self._background_style_decision(state)
            style_spec = style_decision.get("spec") or {}
            style_palette = str(style_spec.get("default_palette") or "").strip()
            if style_palette in palettes:
                return style_palette
            return "light_blue" if "light_blue" in palettes else next(iter(palettes), "light_blue")
        if requested in palettes:
            return requested
        return "light_blue" if "light_blue" in palettes else next(iter(palettes), "light_blue")

    def _palette_spec(self, state: PosterState, palette_name: str | None = None) -> Dict[str, Any]:
        palettes = self.background_config.get("palettes") or {}
        return dict(palettes.get(palette_name or self._palette_name(state)) or {})

    def _rgb_tuple(self, value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, str):
            return self._parse_hex(value)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return tuple(max(0, min(255, int(value[idx]))) for idx in range(3))
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _rgba_tuple(self, value: Any, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if isinstance(value, str):
            rgb = self._parse_hex(value)
            return (*rgb, fallback[3])
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                red = max(0, min(255, int(value[0])))
                green = max(0, min(255, int(value[1])))
                blue = max(0, min(255, int(value[2])))
                alpha = max(0, min(255, int(value[3]))) if len(value) >= 4 else fallback[3]
                return (red, green, blue, alpha)
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _parse_hex(self, color: str) -> tuple[int, int, int]:
        value = str(color or "").strip().lstrip("#")
        if len(value) != 6:
            return (0, 87, 184)
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return (0, 87, 184)

    def _cover_resize(self, img: Image.Image, width: int, height: int) -> Image.Image:
        target_aspect = width / max(height, 1)
        image_aspect = img.width / max(img.height, 1)
        if image_aspect > target_aspect:
            new_width = int(img.height * target_aspect)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            new_height = int(img.width / target_aspect)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        return img.resize((width, height), Image.Resampling.LANCZOS)

    def _save_report(self, state: PosterState, report: Dict[str, Any]) -> None:
        content_dir = Path(state["output_dir"]) / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        with open(content_dir / "background_image_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def background_image_agent_node(state: PosterState) -> Dict[str, Any]:
    result = BackgroundImageAgent()(state)
    return {
        **state,
        "background_image_path": result.get("background_image_path"),
        "background_image_report": result.get("background_image_report"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"],
    }
