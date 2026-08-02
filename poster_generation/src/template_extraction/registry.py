from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SEMANTIC_LANES = ["left", "middle", "right"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_template_dir() -> Path:
    return repo_root() / "template_library" / "templates"


def iter_template_files(template_dir: Optional[Path] = None) -> Iterable[Path]:
    root = template_dir or default_template_dir()
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def list_extracted_template_ids(template_dir: Optional[Path] = None) -> List[str]:
    ids: List[str] = []
    for path in iter_template_files(template_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        template_id = data.get("template_id")
        if template_id:
            ids.append(template_id)
    return sorted(set(ids))


def load_extracted_template(template_id: str, template_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    for path in iter_template_files(template_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("template_id") == template_id:
            data["_template_path"] = str(path)
            return data
    return None


def scale_template_to_canvas(
    template: Dict[str, Any],
    page_width: float,
    page_height: float,
    margin: float = 1.0,
    col_gap: float = 1.0,
    header_height: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert a normalized extracted template into the layout API contract.

    Extracted templates are treated as soft priors: OCR boxes preserve visual
    style and broad proportions, while the returned lanes are safe content
    regions sized for the current poster canvas.
    """

    def scale_box(box: Dict[str, Any], default_id: Optional[str] = None) -> Dict[str, Any]:
        scaled = {
            "x": float(box.get("x", 0.0)) * page_width,
            "y": float(box.get("y", 0.0)) * page_height,
            "w": float(box.get("w", 0.0)) * page_width,
            "h": float(box.get("h", 0.0)) * page_height,
        }
        if default_id is not None:
            scaled["id"] = box.get("id", default_id)
        for key, value in box.items():
            if key not in {"x", "y", "w", "h", "width", "height"} and key not in scaled:
                scaled[key] = value
        return scaled

    source_header = template.get("header") or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.16}
    header = scale_box(source_header)
    if header_height:
        header["h"] = max(header["h"], header_height * 0.75)

    usable_w = max(page_width - 2 * margin, 0.1)
    effective_header_h = header_height or header["h"]
    body_y = margin + effective_header_h
    body_h = max(page_height - margin - body_y, 0.1)
    orientation = template.get("orientation") or ("portrait" if page_height > page_width else "landscape")

    source_lanes = []
    for idx, lane in enumerate(template.get("lanes") or []):
        lane_id = lane.get("id") or SEMANTIC_LANES[min(idx, len(SEMANTIC_LANES) - 1)]
        scaled_lane = scale_box(lane, lane_id)
        scaled_lane["id"] = lane_id
        source_lanes.append(scaled_lane)

    lanes = _build_soft_lanes(
        template.get("lanes") or [],
        orientation=orientation,
        page_width=page_width,
        page_height=page_height,
        margin=margin,
        col_gap=col_gap,
        body_y=body_y,
        body_h=body_h,
        usable_w=usable_w,
    )

    panels = [scale_box(panel, panel.get("id")) for panel in template.get("panels", [])]
    logo_regions = [scale_box(region, region.get("id")) for region in template.get("logo_regions", [])]
    footer = scale_box(template["footer"]) if template.get("footer") else None

    layout = {
        "template_name": template["template_id"],
        "source_template_path": template.get("_template_path"),
        "source_image": template.get("source_image"),
        "orientation": orientation,
        "aspect_ratio": template.get("aspect_ratio"),
        "geometry_policy": template.get("geometry_policy", "soft"),
        "style_strength": template.get("style_strength", "medium"),
        "preferred_orientation": template.get("preferred_orientation", orientation),
        "preferred_body_frame": _normalized_body_frame(template.get("lanes") or template.get("panels") or []),
        "header": header,
        "lanes": lanes,
        "columns": lanes,
        "source_lanes": source_lanes,
        "panels": panels,
        "source_panels": panels,
        "logo_regions": logo_regions,
        "footer": footer,
        "style_tokens": template.get("style_tokens", {}),
        "panel_style_tokens": template.get("panel_style_tokens") or template.get("style_tokens", {}),
        "visual_width_cap": template.get("visual_width_cap"),
        "extracted_template": True,
    }
    return layout


def _build_soft_lanes(
    normalized_lanes: List[Dict[str, Any]],
    *,
    orientation: str,
    page_width: float,
    page_height: float,
    margin: float,
    col_gap: float,
    body_y: float,
    body_h: float,
    usable_w: float,
) -> List[Dict[str, Any]]:
    source = normalized_lanes or [
        {"id": lane_id, "w": 1.0, "h": 1.0}
        for lane_id in SEMANTIC_LANES
    ]

    if orientation == "portrait" or page_height > page_width:
        ratios = _positive_ratios([float(lane.get("h", 1.0)) for lane in source])
        vertical_gap = min(col_gap, max(page_height * 0.018, 0.35))
        available_h = max(body_h - vertical_gap * (len(SEMANTIC_LANES) - 1), 0.1)
        current_y = body_y
        lanes = []
        for idx, lane_id in enumerate(SEMANTIC_LANES):
            height = available_h * ratios[idx]
            lanes.append({
                "id": lane_id,
                "x": margin,
                "y": current_y,
                "w": usable_w,
                "h": height,
                "source_height_ratio": ratios[idx],
                "soft_template_lane": True,
            })
            current_y += height + vertical_gap
        return lanes

    ratios = _positive_ratios([float(lane.get("w", 1.0)) for lane in source])
    available_w = max(usable_w - col_gap * (len(SEMANTIC_LANES) - 1), 0.1)
    current_x = margin
    lanes = []
    for idx, lane_id in enumerate(SEMANTIC_LANES):
        width = available_w * ratios[idx]
        lanes.append({
            "id": lane_id,
            "x": current_x,
            "y": body_y,
            "w": width,
            "h": body_h,
            "source_width_ratio": ratios[idx],
            "soft_template_lane": True,
        })
        current_x += width + col_gap
    return lanes


def _positive_ratios(values: List[float]) -> List[float]:
    values = [max(value, 0.01) for value in values[: len(SEMANTIC_LANES)]]
    if len(values) < len(SEMANTIC_LANES):
        values.extend([1.0] * (len(SEMANTIC_LANES) - len(values)))
    total = sum(values) or 1.0
    return [value / total for value in values]


def _normalized_body_frame(boxes: List[Dict[str, Any]]) -> Dict[str, float]:
    if not boxes:
        return {"x": 0.0, "y": 0.2, "w": 1.0, "h": 0.75}
    x0 = min(float(box.get("x", 0.0)) for box in boxes)
    y0 = min(float(box.get("y", 0.0)) for box in boxes)
    x1 = max(float(box.get("x", 0.0)) + float(box.get("w", 0.0)) for box in boxes)
    y1 = max(float(box.get("y", 0.0)) + float(box.get("h", 0.0)) for box in boxes)
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(max(x1 - x0, 0.0), 5),
        "h": round(max(y1 - y0, 0.0), 5),
    }
