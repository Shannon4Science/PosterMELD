"""Header-block quality review and repair.

This pass is intentionally narrow: it only reviews the top title block after a
draft render, then applies small typography/spacing corrections before the
poster enters block-level content refinement.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.agents.vlm_layout_reviewer import VLMLayoutReviewer
from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success


class HeaderBlockReviewer:
    """Review and repair the title/subtitle/authors/logo header block."""

    def __init__(self):
        self.name = "header_block_reviewer"
        self.config = load_config()
        self.review_config = self.config.get("header_block_review", {})
        self.header_config = self.config.get("header_planner", {})
        self.vlm_client = VLMLayoutReviewer()

    def __call__(self, state: PosterState) -> PosterState:
        if not self.review_config.get("enabled", True):
            state["header_block_review"] = {"accepted": True, "source": "disabled"}
            state["header_block_patch_applied"] = False
            return state

        log_agent_info(self.name, "reviewing rendered header block")
        state["header_block_patch_applied"] = False
        try:
            layout = state.get("styled_layout") or []
            title = self._find_title_element(layout)
            report: Dict[str, Any] = {
                "source": "header_block_reviewer",
                "accepted": True,
                "issues": [],
                "patch": [],
                "vlm": {"status": "not_run"},
            }
            if not title:
                report["accepted"] = False
                report["issues"].append({"severity": "high", "category": "missing_header", "description": "No title element found."})
                state["header_block_review"] = report
                self._save_outputs(state)
                return state

            crop_path = self._crop_header_preview(state)
            if crop_path:
                report["crop_path"] = crop_path

            report["before"] = self._metrics(title)
            patches = self._apply_deterministic_patch(title)
            report["after_deterministic"] = self._metrics(title)
            report["issues"].extend(self._issues_from_metrics(report["after_deterministic"]))
            report["vlm"] = self._review_with_vlm(state, crop_path)
            vlm_patches = self._apply_vlm_patch(title, report["vlm"])
            if vlm_patches:
                patches.extend(vlm_patches)
                self._ensure_title_stack_fits(title, patches)
                self._refresh_title_content(title)
            report["patch"] = patches
            report["patch_applied"] = bool(patches)
            state["header_block_patch_applied"] = bool(patches)
            report["after"] = self._metrics(title)
            report["issues"] = self._issues_from_metrics(report["after"])
            if report["issues"]:
                report["accepted"] = not any(issue.get("severity") == "high" for issue in report["issues"])
            state["header_block_review"] = report
            self._save_outputs(state)
            if patches:
                log_agent_success(self.name, f"applied {len(patches)} header patch(es)")
            else:
                log_agent_success(self.name, "header block accepted without patch")
        except Exception as exc:
            log_agent_error(self.name, f"header review failed: {exc}")
            state.setdefault("errors", []).append(f"{self.name}: {exc}")
        return state

    def _find_title_element(self, layout: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for element in layout:
            if element.get("type") == "title":
                return element
        return None

    def _metrics(self, title: Dict[str, Any]) -> Dict[str, Any]:
        title_font = float(title.get("font_size") or 0.0)
        subtitle_font = float(title.get("subtitle_font_size") or 0.0)
        author_font = float(title.get("author_font_size") or 0.0)
        return {
            "x": round(float(title.get("x", 0.0) or 0.0), 3),
            "y": round(float(title.get("y", 0.0) or 0.0), 3),
            "width": round(float(title.get("width", 0.0) or 0.0), 3),
            "height": round(float(title.get("height", 0.0) or 0.0), 3),
            "title_font_size": round(title_font, 2),
            "subtitle_font_size": round(subtitle_font, 2),
            "author_font_size": round(author_font, 2),
            "subtitle_title_ratio": round(subtitle_font / title_font, 3) if title_font else 0.0,
            "author_top_gap_inches": round(float(title.get("author_top_gap_inches") or 0.0), 3),
            "title_to_subtitle_gap_inches": round(float(title.get("title_to_subtitle_gap_inches") or 0.0), 3),
            "subtitle_box_height": round(float(title.get("subtitle_box_height") or 0.0), 3),
            "author_box_height": round(float(title.get("author_box_height") or 0.0), 3),
        }

    def _apply_deterministic_patch(self, title: Dict[str, Any]) -> List[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []
        title_font = float(title.get("font_size") or 0.0)
        subtitle_text = str(title.get("subtitle_text") or "").strip()
        if title_font > 0 and subtitle_text:
            current_subtitle = float(title.get("subtitle_font_size") or 0.0)
            target_ratio = float(self.review_config.get("target_subtitle_title_ratio", 0.62))
            max_ratio = float(self.review_config.get("max_subtitle_title_ratio", 0.74))
            desired = min(title_font * max_ratio, max(current_subtitle, title_font * target_ratio))
            fitted = self._fit_subtitle_font_size(subtitle_text, float(title.get("width") or 0.0), desired)
            if fitted > current_subtitle + 0.5:
                title["subtitle_font_size"] = round(fitted, 2)
                patches.append({
                    "target": "subtitle",
                    "op": "increase_font_size",
                    "from": round(current_subtitle, 2),
                    "to": round(fitted, 2),
                    "reason": "subtitle was visually too small relative to the main title",
                })

            line_height = float(self.review_config.get("subtitle_box_line_height", 1.12))
            target_box = max((float(title.get("subtitle_font_size") or fitted) / 72.0) * line_height, 0.36)
            current_box = float(title.get("subtitle_box_height") or 0.0)
            if target_box > current_box + 0.02:
                title["subtitle_box_height"] = round(target_box, 4)
                patches.append({
                    "target": "subtitle",
                    "op": "increase_box_height",
                    "from": round(current_box, 3),
                    "to": round(target_box, 3),
                    "reason": "subtitle font increase requires matching line box height",
                })

        target_subtitle_gap = float(
            self.review_config.get(
                "title_subtitle_gap_inches",
                self.header_config.get("title_subtitle_gap_inches", 0.10),
            )
        )
        current_subtitle_gap = float(title.get("title_to_subtitle_gap_inches") or 0.0)
        if target_subtitle_gap > current_subtitle_gap + 0.01:
            title["title_to_subtitle_gap_inches"] = round(target_subtitle_gap, 4)
            patches.append({
                "target": "subtitle",
                "op": "increase_top_gap",
                "from": round(current_subtitle_gap, 3),
                "to": round(target_subtitle_gap, 3),
                "reason": "subtitle needs a clearer separation from the title",
            })

        target_author_gap = float(self.header_config.get("portrait_title_author_gap_inches", 0.30))
        current_author_gap = float(title.get("author_top_gap_inches") or 0.0)
        if target_author_gap > current_author_gap + 0.01:
            title["author_top_gap_inches"] = round(target_author_gap, 4)
            patches.append({
                "target": "authors",
                "op": "move_down",
                "from": round(current_author_gap, 3),
                "to": round(target_author_gap, 3),
                "reason": "author line was too close to the subtitle",
            })

        self._ensure_title_stack_fits(title, patches)
        if patches:
            self._refresh_title_content(title)
        return patches

    def _fit_subtitle_font_size(self, text: str, width_inches: float, desired_size: float) -> float:
        clean = " ".join(str(text or "").split())
        if not clean or width_inches <= 0:
            return desired_size
        avg_char_width = float(
            self.header_config.get(
                "portrait_subtitle_fit_avg_char_width_em",
                self.header_config.get("subtitle_fit_avg_char_width_em", self.header_config.get("title_fit_avg_char_width_em", 0.56)),
            )
        )
        width_safety = float(self.header_config.get("title_fit_width_safety", 0.94))
        usable_width = max(width_inches * width_safety, 0.1)
        estimated = (usable_width * 72.0) / max(len(clean) * avg_char_width, 1.0)
        return max(float(self.header_config.get("subtitle_single_line_min_font_size", 24)), min(float(desired_size), estimated))

    def _apply_vlm_patch(self, title: Dict[str, Any], vlm_review: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(vlm_review, dict) or vlm_review.get("status") != "ok":
            return []
        author_spacing_issue = any(
            str(issue.get("category") or "").lower() == "author_spacing"
            and str(issue.get("severity") or "low").lower() in {"medium", "high"}
            for issue in vlm_review.get("issues") or []
            if isinstance(issue, dict)
        )
        author_spacing_recommendation = any(
            str(rec.get("target") or "").lower() == "authors"
            and str(rec.get("action") or "").lower() in {"move_down", "increase_gap"}
            for rec in vlm_review.get("recommendations") or []
            if isinstance(rec, dict)
        )
        if not (author_spacing_issue or author_spacing_recommendation):
            return []

        current = float(title.get("author_top_gap_inches") or 0.0)
        increment = float(self.review_config.get("vlm_author_gap_increment_inches", 0.10) or 0.10)
        maximum = float(self.review_config.get("max_author_gap_inches", 0.50) or 0.50)
        target = min(max(current + increment, float(self.header_config.get("portrait_title_author_gap_inches", current))), maximum)
        if target <= current + 0.01:
            return []
        title["author_top_gap_inches"] = round(target, 4)
        return [{
            "target": "authors",
            "op": "move_down",
            "from": round(current, 3),
            "to": round(target, 3),
            "source": "vlm_header_review",
            "reason": "VLM flagged medium author-spacing issue in the header crop",
        }]

    def _ensure_title_stack_fits(self, title: Dict[str, Any], patches: List[Dict[str, Any]]) -> None:
        total_height = float(title.get("height") or 0.0)
        if total_height <= 0:
            return
        title_box = float(title.get("title_box_height") or 0.0)
        subtitle_box = float(title.get("subtitle_box_height") or 0.0)
        subtitle_gap = float(title.get("title_to_subtitle_gap_inches") or 0.0)
        author_gap = float(title.get("author_top_gap_inches") or 0.0)
        author_box = float(title.get("author_box_height") or 0.0)
        stack_height = title_box + subtitle_gap + subtitle_box + author_gap + author_box
        if stack_height <= total_height:
            return
        new_title_box = max(total_height - subtitle_gap - subtitle_box - author_gap - author_box, total_height * 0.30)
        if new_title_box < title_box:
            title["title_box_height"] = round(new_title_box, 4)
            patches.append({
                "target": "title",
                "op": "decrease_box_height",
                "from": round(title_box, 3),
                "to": round(new_title_box, 3),
                "reason": "keep title/subtitle/authors stack inside the header text box",
            })

    def _refresh_title_content(self, title: Dict[str, Any]) -> None:
        parts = [
            str(title.get("title_text") or "").strip(),
            str(title.get("subtitle_text") or "").strip(),
            str(title.get("authors_text") or "").strip(),
        ]
        title["content"] = "\n".join(part for part in parts if part)

    def _issues_from_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        min_ratio = float(self.review_config.get("min_subtitle_title_ratio", 0.54))
        if metrics.get("subtitle_font_size", 0.0) and metrics.get("subtitle_title_ratio", 0.0) < min_ratio:
            issues.append({
                "severity": "medium",
                "category": "hierarchy",
                "description": "Subtitle remains too small relative to the title.",
            })
        author_gap = float(metrics.get("author_top_gap_inches") or 0.0)
        target_author_gap = float(self.header_config.get("portrait_title_author_gap_inches", 0.30))
        if author_gap + 0.01 < target_author_gap:
            issues.append({
                "severity": "medium",
                "category": "spacing",
                "description": "Author line is still too close to the subtitle.",
            })
        return issues

    def _crop_header_preview(self, state: PosterState) -> Optional[str]:
        preview = state.get("poster_preview_path")
        if not preview or not Path(preview).exists():
            return None
        template = state.get("layout_template_metadata") or {}
        header = template.get("header") or {}
        if not header:
            return None
        try:
            with Image.open(preview) as image:
                width_px, height_px = image.size
                poster_w = float(state.get("poster_width") or 0.0)
                poster_h = float(state.get("poster_height") or 0.0)
                if poster_w <= 0 or poster_h <= 0:
                    return None
                margin = float(self.review_config.get("crop_margin_inches", 0.18))
                left = max((float(header.get("x", 0.0)) - margin) / poster_w * width_px, 0)
                top = max((float(header.get("y", 0.0)) - margin) / poster_h * height_px, 0)
                right = min((float(header.get("x", 0.0)) + float(header.get("w", poster_w)) + margin) / poster_w * width_px, width_px)
                bottom = min((float(header.get("y", 0.0)) + float(header.get("h", 0.0)) + margin) / poster_h * height_px, height_px)
                if right <= left or bottom <= top:
                    return None
                crop = image.crop((int(left), int(top), int(right), int(bottom)))
                output_dir = Path(state["output_dir"]) / "content"
                output_dir.mkdir(parents=True, exist_ok=True)
                crop_path = output_dir / "header_block_crop.png"
                crop.save(crop_path)
                return str(crop_path)
        except Exception:
            return None

    def _review_with_vlm(self, state: PosterState, crop_path: Optional[str]) -> Dict[str, Any]:
        if not self.review_config.get("vlm_enabled", True):
            return {"status": "disabled"}
        if not crop_path or not Path(crop_path).exists():
            return {"status": "skipped", "reason": "header crop unavailable"}
        base_url = os.getenv("VLM_BASE_URL")
        api_key = os.getenv("VLM_API_KEY")
        model = state.get("vlm_model") or os.getenv("VLM_MODEL")
        if not (base_url and api_key and model):
            return {"status": "skipped", "reason": "VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL are required"}
        try:
            image_data = self.vlm_client._encode_image(crop_path)
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = self.vlm_client._post_vlm_request(base_url, headers, model, self._vlm_prompt(), image_data)
            response.raise_for_status()
            content = self.vlm_client._extract_response_text(response)
            self.vlm_client._record_usage(state, self.name)
            review = self.vlm_client._parse_json(content)
            review["status"] = "ok"
            return review
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def _vlm_prompt(self) -> str:
        return """
You are reviewing only the title/header block crop of an academic poster.

Judge:
- title/subtitle/author visual hierarchy
- whether the subtitle is readable and not treated like a footnote
- whether the author line is aligned and comfortably separated from the subtitle
- whether institution and conference logos look balanced with the title group
- whether any text or logo overlaps, looks clipped, or feels uncentered

Return strict JSON only:
{
  "score": 0-100,
  "accept": true/false,
  "issues": [
    {
      "severity": "low|medium|high",
      "category": "subtitle_size|author_spacing|alignment|logo_balance|overlap|style",
      "description": "short visual diagnosis"
    }
  ],
  "recommendations": [
    {
      "target": "title|subtitle|authors|logos",
      "action": "increase_font|decrease_font|move_down|move_up|increase_gap|decrease_gap|keep",
      "reason": "short reason"
    }
  ]
}
""".strip()

    def _save_outputs(self, state: PosterState) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "header_block_review.json", "w", encoding="utf-8") as f:
            json.dump(state.get("header_block_review", {}), f, indent=2, ensure_ascii=False)
        with open(output_dir / "styled_layout.json", "w", encoding="utf-8") as f:
            json.dump(state.get("styled_layout", []), f, indent=2, ensure_ascii=False)


def header_block_reviewer_node(state: PosterState) -> Dict[str, Any]:
    result = HeaderBlockReviewer()(state)
    return {
        **state,
        "styled_layout": result.get("styled_layout"),
        "header_block_review": result.get("header_block_review"),
        "header_block_patch_applied": result.get("header_block_patch_applied", False),
        "current_agent": result.get("current_agent", "header_block_reviewer"),
        "errors": result.get("errors", []),
    }
