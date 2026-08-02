"""
VLM review for individual template blocks.

The reviewer crops each block, sends a labeled contact sheet to the VLM, and
asks only for qualitative status labels. It does not decide how much text to
add or remove; that stays with the deterministic occupancy report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from src.agents.vlm_layout_reviewer import VLMLayoutReviewer
from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class BlockVLMReviewer(VLMLayoutReviewer):
    VALID_STATUSES = {"empty", "ok", "crowded", "overflow", "visual_too_small", "underfilled"}

    def __init__(self):
        super().__init__()
        self.name = "block_vlm_reviewer"
        self.config = load_config()
        self.review_config = self.config.get("block_refinement", {})

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_block_vlm_review", False):
            return state

        log_agent_info(self.name, "reviewing block crops for fill, crowding, overflow, and visual legibility")

        try:
            review = self._review_or_fallback(state)
            if review.get("degraded"):
                state.setdefault("degraded_quality_states", []).append(
                    {
                        "component": self.name,
                        "category": "block_vlm_review",
                        "reason": "; ".join(str(item) for item in review.get("warnings", [])),
                        "fallback": review.get("fallback") or "occupancy_only_block_review",
                    }
                )
            state["block_vlm_review"] = review
            state["current_agent"] = self.name
            self._save_outputs(state, review)
            issue_count = sum(
                1
                for block in review.get("blocks", [])
                if block.get("status") not in {"ok"}
            )
            log_agent_success(self.name, f"reviewed {len(review.get('blocks', []))} block(s), flagged {issue_count}")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _review_or_fallback(self, state: PosterState) -> Dict[str, Any]:
        occupancy = state.get("block_occupancy_report") or {}
        self._last_occupancy = occupancy
        preview_path = state.get("poster_preview_path")
        if not occupancy.get("blocks"):
            return self._fallback_review("block occupancy report is unavailable")
        if not preview_path or not Path(preview_path).exists():
            return self._fallback_review("poster preview PNG is unavailable; using occupancy-only block review")

        crop_info = self._build_contact_sheet(state, occupancy, preview_path)
        base_url = os.getenv("VLM_BASE_URL")
        api_key = os.getenv("VLM_API_KEY")
        model = state.get("vlm_model") or os.getenv("VLM_MODEL")
        if not base_url or not api_key or not model:
            warning = "VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL are required for block VLM review"
            fallback = self._fallback_review(warning)
            fallback["contact_sheet_path"] = crop_info.get("contact_sheet_path")
            fallback["crop_paths"] = crop_info.get("crop_paths", {})
            return fallback

        prompt = self._build_prompt(occupancy)
        image_data = self._encode_image(crop_info["contact_sheet_path"])
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            content = self._request_vlm_text(base_url, headers, model, prompt, image_data)
            self._record_usage(state, self.name)
            parsed = self._parse_json(content)
            review = self._normalize_review(parsed, occupancy)
            review["source"] = "vlm"
            review["review_available"] = True
            review["degraded"] = False
            review.setdefault("warnings", [])
        except Exception as exc:
            review = self._fallback_from_occupancy(
                occupancy,
                f"block VLM request failed ({exc}); using occupancy-only block review",
            )
        review["contact_sheet_path"] = crop_info.get("contact_sheet_path")
        review["crop_paths"] = crop_info.get("crop_paths", {})
        return review

    def _build_contact_sheet(
        self,
        state: PosterState,
        occupancy: Dict[str, Any],
        preview_path: str,
    ) -> Dict[str, Any]:
        image = Image.open(preview_path).convert("RGB")
        crop_dir = Path(state["output_dir"]) / "block_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        scale_x = image.width / max(float(state.get("poster_width") or 1), 1.0)
        scale_y = image.height / max(float(state.get("poster_height") or 1), 1.0)
        crops: List[Dict[str, Any]] = []
        crop_paths: Dict[str, str] = {}

        for index, block in enumerate(occupancy.get("blocks", []), start=1):
            bbox = block.get("bbox") or block.get("container_bbox") or {}
            left = int(max(float(bbox.get("x", 0.0)) * scale_x, 0))
            top = int(max(float(bbox.get("y", 0.0)) * scale_y, 0))
            right = int(min((float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0))) * scale_x, image.width))
            bottom = int(min((float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0))) * scale_y, image.height))
            if right <= left or bottom <= top:
                continue
            label = str(block.get("slot_id") or block.get("section_id") or f"block_{index}")
            crop = image.crop((left, top, right, bottom))
            crop_path = crop_dir / f"{self._safe_filename(label)}.png"
            crop.save(crop_path)
            crop_paths[label] = str(crop_path)
            crops.append({"label": label, "image": crop})

        if not crops:
            raise ValueError("no valid block crops were produced")

        contact_sheet_path = crop_dir / "block_contact_sheet.png"
        self._save_contact_sheet(crops, contact_sheet_path)
        return {"contact_sheet_path": str(contact_sheet_path), "crop_paths": crop_paths}

    def _save_contact_sheet(self, crops: List[Dict[str, Any]], output_path: Path) -> None:
        label_height = 34
        cell_width = 620
        cell_height = 520
        columns = 2 if len(crops) > 1 else 1
        rows = (len(crops) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        font = self._load_font(size=20)

        for index, item in enumerate(crops):
            col = index % columns
            row = index // columns
            x0 = col * cell_width
            y0 = row * cell_height
            draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(180, 180, 180), width=2)
            draw.text((x0 + 12, y0 + 7), item["label"], fill=(0, 0, 0), font=font)
            crop = item["image"]
            max_w = cell_width - 24
            max_h = cell_height - label_height - 18
            scale = min(max_w / crop.width, max_h / crop.height, 1.0)
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            paste_x = x0 + (cell_width - resized.width) // 2
            paste_y = y0 + label_height + (max_h - resized.height) // 2
            sheet.paste(resized, (paste_x, paste_y))

        sheet.save(output_path)

    def _build_prompt(self, occupancy: Dict[str, Any]) -> str:
        block_summaries = [
            {
                "slot_id": block.get("slot_id"),
                "section_id": block.get("section_id"),
                "section_title": block.get("section_title"),
                "utilization": block.get("utilization"),
                "occupancy_action": block.get("action"),
                "target_extra_chars": block.get("target_extra_chars"),
                "visual_count": block.get("visual_count"),
            }
            for block in occupancy.get("blocks", [])
        ]
        return f"""
You are reviewing cropped blocks from an academic poster.

The image is a contact sheet. Each crop label matches slot_id below.

For every block, judge only what is visible:
- empty: block is mostly blank or missing meaningful content
- ok: content fills the block without feeling crowded
- crowded: content is too dense but still inside the block
- overflow: text, visuals, or labels are clipped or spill outside the block
- visual_too_small: a figure/table is present but its internal labels are too small to read
- underfilled: the block has excessive whitespace but is not empty

Do not propose text amounts, wording, geometry patches, or new science.

Return strict JSON only:
{{
  "blocks": [
    {{
      "slot_id": "must match one of the provided slot_id labels",
      "section_id": "section id if visible or provided",
      "status": "empty|underfilled|ok|crowded|overflow|visual_too_small",
      "severity": "low|medium|high",
      "description": "short visual diagnosis"
    }}
  ],
  "warnings": []
}}

Block metadata:
{json.dumps(block_summaries, ensure_ascii=False, indent=2)}
"""

    def _normalize_review(self, parsed: Dict[str, Any], occupancy: Dict[str, Any]) -> Dict[str, Any]:
        provided = parsed.get("blocks") if isinstance(parsed, dict) else []
        provided_by_slot = {
            str(item.get("slot_id")): item
            for item in provided
            if isinstance(item, dict) and item.get("slot_id")
        }
        blocks = []
        for block in occupancy.get("blocks", []):
            slot_id = str(block.get("slot_id") or "")
            item = provided_by_slot.get(slot_id, {})
            status = str(item.get("status") or self._fallback_status_for_block(block)).lower()
            if status not in self.VALID_STATUSES:
                status = self._fallback_status_for_block(block)
            severity = str(item.get("severity") or self._severity_for_status(status, block)).lower()
            if severity not in {"low", "medium", "high"}:
                severity = self._severity_for_status(status, block)
            blocks.append({
                "slot_id": slot_id,
                "section_id": item.get("section_id") or block.get("section_id"),
                "status": status,
                "severity": severity,
                "description": item.get("description") or block.get("reason") or "",
                "utilization": block.get("utilization"),
            })
        return {
            "blocks": blocks,
            "warnings": parsed.get("warnings", []) if isinstance(parsed, dict) else [],
        }

    def _fallback_review(self, warning: str) -> Dict[str, Any]:
        log_agent_warning(self.name, warning)
        occupancy = getattr(self, "_last_occupancy", None)
        if not occupancy:
            return {
                "source": "fallback",
                "review_available": False,
                "degraded": True,
                "fallback": "occupancy_only_block_review",
                "blocks": [],
                "warnings": [warning],
            }
        return self._fallback_from_occupancy(occupancy, warning)

    def _fallback_from_occupancy(self, occupancy: Dict[str, Any], warning: str) -> Dict[str, Any]:
        blocks = []
        for block in occupancy.get("blocks", []):
            status = self._fallback_status_for_block(block)
            blocks.append({
                "slot_id": block.get("slot_id"),
                "section_id": block.get("section_id"),
                "status": status,
                "severity": self._severity_for_status(status, block),
                "description": block.get("reason", ""),
                "utilization": block.get("utilization"),
            })
        return {
            "source": "fallback",
            "review_available": False,
            "degraded": True,
            "fallback": "occupancy_only_block_review",
            "blocks": blocks,
            "warnings": [warning],
        }

    def _fallback_status_for_block(self, block: Dict[str, Any]) -> str:
        action = block.get("action")
        if action == "expand":
            return "underfilled"
        if action == "reduce":
            return "crowded"
        return "ok"

    def _severity_for_status(self, status: str, block: Dict[str, Any]) -> str:
        utilization = float(block.get("utilization") or 0.0)
        if status in {"overflow", "visual_too_small"}:
            return "high"
        if status == "crowded":
            return "high" if utilization > float(self.review_config.get("hard_max", 0.98)) else "medium"
        if status in {"empty", "underfilled"}:
            return "medium" if utilization < float(self.review_config.get("acceptable_min", 0.90)) else "low"
        return "low"

    def _safe_filename(self, label: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
        return safe[:80] or "block"

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        for path in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def _save_outputs(self, state: PosterState, review: Dict[str, Any]) -> None:
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "block_vlm_review.json", "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)


def block_vlm_reviewer_node(state: PosterState) -> Dict[str, Any]:
    result = BlockVLMReviewer()(state)
    return {
        **state,
        "block_vlm_review": result.get("block_vlm_review"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
