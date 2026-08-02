from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TemplateProfile:
    template_id: str
    orientation: str
    header: Dict[str, float]
    lanes: List[Dict[str, Any]]
    panels: List[Dict[str, Any]]
    logo_regions: List[Dict[str, Any]]
    footer: Dict[str, float] | None = None


PROFILES: Dict[str, TemplateProfile] = {
    "poster(1).png": TemplateProfile(
        template_id="extracted_poster1_landscape_three_panel",
        orientation="landscape",
        header={"x": 0.035, "y": 0.025, "w": 0.93, "h": 0.17},
        lanes=[
            {"id": "left", "x": 0.035, "y": 0.235, "w": 0.285, "h": 0.665, "panel_ids": ["intro_panel"]},
            {"id": "middle", "x": 0.355, "y": 0.235, "w": 0.29, "h": 0.665, "panel_ids": ["method_panel"]},
            {"id": "right", "x": 0.68, "y": 0.235, "w": 0.285, "h": 0.665, "panel_ids": ["results_panel"]},
        ],
        panels=[
            {"id": "intro_panel", "lane_id": "left", "x": 0.035, "y": 0.235, "w": 0.285, "h": 0.665},
            {"id": "method_panel", "lane_id": "middle", "x": 0.355, "y": 0.235, "w": 0.29, "h": 0.665},
            {"id": "results_panel", "lane_id": "right", "x": 0.68, "y": 0.235, "w": 0.285, "h": 0.665},
        ],
        logo_regions=[{"id": "title_right_logo", "x": 0.78, "y": 0.04, "w": 0.18, "h": 0.12}],
        footer={"x": 0.035, "y": 0.915, "w": 0.93, "h": 0.055},
    ),
    "poster(2).png": TemplateProfile(
        template_id="extracted_poster2_landscape_multi_panel",
        orientation="landscape",
        header={"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.15},
        lanes=[
            {"id": "left", "x": 0.025, "y": 0.18, "w": 0.30, "h": 0.72, "panel_ids": ["motivation_panel", "task1_panel"]},
            {"id": "middle", "x": 0.345, "y": 0.18, "w": 0.31, "h": 0.72, "panel_ids": ["task2_panel", "task3_panel"]},
            {"id": "right", "x": 0.675, "y": 0.18, "w": 0.30, "h": 0.72, "panel_ids": ["results_panel"]},
        ],
        panels=[
            {"id": "motivation_panel", "lane_id": "left", "x": 0.025, "y": 0.18, "w": 0.14, "h": 0.72},
            {"id": "task1_panel", "lane_id": "left", "x": 0.175, "y": 0.18, "w": 0.15, "h": 0.72},
            {"id": "task2_panel", "lane_id": "middle", "x": 0.345, "y": 0.18, "w": 0.15, "h": 0.72},
            {"id": "task3_panel", "lane_id": "middle", "x": 0.505, "y": 0.18, "w": 0.15, "h": 0.72},
            {"id": "results_panel", "lane_id": "right", "x": 0.675, "y": 0.18, "w": 0.30, "h": 0.72},
        ],
        logo_regions=[{"id": "title_right_logo", "x": 0.82, "y": 0.02, "w": 0.15, "h": 0.11}],
        footer={"x": 0.0, "y": 0.925, "w": 1.0, "h": 0.075},
    ),
    "poster(3).png": TemplateProfile(
        template_id="extracted_poster3_portrait_section_band",
        orientation="portrait",
        header={"x": 0.025, "y": 0.025, "w": 0.95, "h": 0.115},
        lanes=[
            {"id": "left", "x": 0.025, "y": 0.175, "w": 0.95, "h": 0.22, "panel_ids": ["introduction_band"]},
            {"id": "middle", "x": 0.025, "y": 0.44, "w": 0.95, "h": 0.28, "panel_ids": ["methodology_band"]},
            {"id": "right", "x": 0.025, "y": 0.765, "w": 0.95, "h": 0.20, "panel_ids": ["results_band"]},
        ],
        panels=[
            {"id": "introduction_band", "lane_id": "left", "x": 0.025, "y": 0.175, "w": 0.95, "h": 0.22},
            {"id": "methodology_band", "lane_id": "middle", "x": 0.025, "y": 0.44, "w": 0.95, "h": 0.28},
            {"id": "results_band", "lane_id": "right", "x": 0.025, "y": 0.765, "w": 0.95, "h": 0.20},
        ],
        logo_regions=[
            {"id": "title_left_logo", "x": 0.025, "y": 0.03, "w": 0.13, "h": 0.10},
            {"id": "title_right_logo", "x": 0.82, "y": 0.03, "w": 0.15, "h": 0.10},
        ],
    ),
}


def normalize_box(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> Dict[str, float]:
    return {
        "x": round(x / image_w, 5),
        "y": round(y / image_h, 5),
        "w": round(w / image_w, 5),
        "h": round(h / image_h, 5),
    }


def run_tesseract_tsv(path: Path) -> List[Dict[str, Any]]:
    command = ["tesseract", str(path), "stdout", "--psm", "6", "tsv"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    rows = []
    reader = csv.DictReader(result.stdout.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        try:
            conf = float(row.get("conf", "-1"))
            left = int(float(row.get("left", "0")))
            top = int(float(row.get("top", "0")))
            width = int(float(row.get("width", "0")))
            height = int(float(row.get("height", "0")))
        except ValueError:
            continue
        if text and conf >= 30 and width > 2 and height > 2:
            rows.append({"text": text, "conf": conf, "left": left, "top": top, "width": width, "height": height})
    return rows


def detect_large_regions(image: Image.Image) -> List[Dict[str, Any]]:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_h, image_w = gray.shape
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(image_w * image_h)
        if area_ratio < 0.01 or w < image_w * 0.05 or h < image_h * 0.04:
            continue
        box = normalize_box(x, y, w, h, image_w, image_h)
        box["area_ratio"] = round(area_ratio, 5)
        regions.append(box)
    return sorted(regions, key=lambda item: item["area_ratio"], reverse=True)[:40]


def sample_style_tokens(image: Image.Image, profile: TemplateProfile) -> Dict[str, str]:
    rgb = image.convert("RGB")
    w, h = rgb.size

    def median_hex(box: Dict[str, float]) -> str:
        x0 = max(int(box["x"] * w), 0)
        y0 = max(int(box["y"] * h), 0)
        x1 = min(int((box["x"] + box["w"]) * w), w)
        y1 = min(int((box["y"] + box["h"]) * h), h)
        crop = np.array(rgb.crop((x0, y0, x1, y1)))
        if crop.size == 0:
            return "#FFFFFF"
        median = np.median(crop.reshape(-1, 3), axis=0).astype(int)
        return "#{:02X}{:02X}{:02X}".format(*median)

    panel_fill = median_hex(profile.panels[0]) if profile.panels else "#FFFFFF"
    footer_color = median_hex(profile.footer) if profile.footer else "#FFFFFF"
    return {
        "background": "#FFFFFF",
        "header_background": median_hex(profile.header),
        "section_bar_color": "#D0D0D0" if profile.orientation == "portrait" else panel_fill,
        "panel_fill_color": panel_fill,
        "panel_border_color": "#2F5F8F" if profile.template_id.endswith("three_panel") else "#CCCCCC",
        "footer_background": footer_color,
    }


def build_template(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image = Image.open(path)
    image_w, image_h = image.size
    profile = PROFILES.get(path.name)
    if profile is None:
        orientation = "landscape" if image_w >= image_h else "portrait"
        profile = TemplateProfile(
            template_id=f"extracted_{path.stem.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
            orientation=orientation,
            header={"x": 0.04, "y": 0.03, "w": 0.92, "h": 0.15},
            lanes=[
                {"id": "left", "x": 0.04, "y": 0.22, "w": 0.28, "h": 0.68},
                {"id": "middle", "x": 0.36, "y": 0.22, "w": 0.28, "h": 0.68},
                {"id": "right", "x": 0.68, "y": 0.22, "w": 0.28, "h": 0.68},
            ],
            panels=[],
            logo_regions=[{"id": "title_right_logo", "x": 0.78, "y": 0.04, "w": 0.18, "h": 0.10}],
        )

    ocr_rows = run_tesseract_tsv(path)
    ocr_blocks = [
        {
            "text": row["text"],
            "conf": row["conf"],
            **normalize_box(row["left"], row["top"], row["width"], row["height"], image_w, image_h),
        }
        for row in ocr_rows
    ]
    detected_regions = detect_large_regions(image)

    template = {
        "template_id": profile.template_id,
        "source_image": str(path),
        "orientation": profile.orientation,
        "preferred_orientation": profile.orientation,
        "geometry_policy": "soft",
        "style_strength": "medium",
        "aspect_ratio": round(image_w / image_h, 5),
        "normalized_canvas": {"w": 1.0, "h": 1.0},
        "header": profile.header,
        "logo_regions": profile.logo_regions,
        "lanes": profile.lanes,
        "source_lanes": profile.lanes,
        "panels": profile.panels,
        "source_panels": profile.panels,
        "footer": profile.footer,
        "style_tokens": sample_style_tokens(image, profile),
    }
    template["panel_style_tokens"] = {
        key: template["style_tokens"][key]
        for key in ["panel_fill_color", "panel_border_color", "section_bar_color"]
        if key in template["style_tokens"]
    }
    lane_boxes = profile.lanes or profile.panels
    if lane_boxes:
        x0 = min(box["x"] for box in lane_boxes)
        y0 = min(box["y"] for box in lane_boxes)
        x1 = max(box["x"] + box["w"] for box in lane_boxes)
        y1 = max(box["y"] + box["h"] for box in lane_boxes)
        template["preferred_body_frame"] = {
            "x": round(x0, 5),
            "y": round(y0, 5),
            "w": round(x1 - x0, 5),
            "h": round(y1 - y0, 5),
        }
    raw = {
        **template,
        "image_size": {"width": image_w, "height": image_h},
        "ocr_blocks": ocr_blocks,
        "detected_regions": detected_regions,
    }
    return template, raw


def draw_overlay(image_path: Path, template: Dict[str, Any], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    max_w = 1800
    scale = min(max_w / image.width, 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    draw = ImageDraw.Draw(image)
    w, h = image.size

    def rect(box: Dict[str, float], color: str, label: str) -> None:
        x0 = box["x"] * w
        y0 = box["y"] * h
        x1 = (box["x"] + box["w"]) * w
        y1 = (box["y"] + box["h"]) * h
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.text((x0 + 6, y0 + 6), label, fill=color)

    rect(template["header"], "#1D4ED8", "header")
    for lane in template.get("lanes", []):
        rect(lane, "#16A34A", f"lane:{lane.get('id', '')}")
    for panel in template.get("panels", []):
        rect(panel, "#EA580C", f"panel:{panel.get('id', '')}")
    for region in template.get("logo_regions", []):
        rect(region, "#9333EA", f"logo:{region.get('id', '')}")
    if template.get("footer"):
        rect(template["footer"], "#64748B", "footer")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_templates(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    templates = []
    for image_path in sorted(input_dir.glob("*.png")):
        template, raw = build_template(image_path)
        template_id = template["template_id"]
        write_json(output_dir / "templates" / f"{template_id}.json", template)
        write_json(output_dir / "raw" / f"{template_id}.extracted.json", raw)
        draw_overlay(image_path, template, output_dir / "overlays" / f"{template_id}.overlay.png")
        templates.append(template)
    return templates


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract reusable PosterMELD templates from poster PNGs.")
    parser.add_argument("--input_dir", type=Path, default=Path("template"))
    parser.add_argument("--output_dir", type=Path, default=Path("template_library"))
    args = parser.parse_args()

    templates = extract_templates(args.input_dir, args.output_dir)
    print(f"Extracted {len(templates)} templates to {args.output_dir}")
    for template in templates:
        print(f"- {template['template_id']} ({template['orientation']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
