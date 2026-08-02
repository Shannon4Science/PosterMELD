"""Deterministic footprint checks for poster visuals.

The poster pipeline already optimizes block occupancy. This module adds the
missing complementary contract: a visual is only useful if its final rendered
size is large enough to read.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


def visual_footprint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config.get("visual_footprint") or {})
    if "enabled" not in cfg:
        cfg["enabled"] = True
    return cfg


def visual_kind(visual_id: Any, asset: Optional[Dict[str, Any]] = None) -> str:
    asset = asset or {}
    asset_type = str(asset.get("asset_type") or "").lower()
    visual_id = str(visual_id or "")
    if visual_id.startswith("table_") or asset_type == "table":
        return "table"
    return "figure"


def visual_aspect_ratio(visual_id: Any, asset: Optional[Dict[str, Any]] = None) -> float:
    asset = asset or {}
    fallback = 1.5 if visual_kind(visual_id, asset) == "table" else 1.2
    try:
        return max(float(asset.get("aspect") or fallback), 0.2)
    except (TypeError, ValueError):
        return fallback


def visual_requirements(
    visual_id: Any,
    asset: Optional[Dict[str, Any]],
    lane: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = visual_footprint_config(config)
    kind = visual_kind(visual_id, asset)
    aspect = visual_aspect_ratio(visual_id, asset)
    lane_w = max(float((lane or {}).get("w", 0.0) or 0.0), 0.0)
    lane_h = max(float((lane or {}).get("h", 0.0) or 0.0), 0.0)

    orientation = str((lane or {}).get("poster_orientation") or (lane or {}).get("orientation") or "").lower()
    prefix = "portrait_" if orientation == "portrait" else ""

    def cfg_float(key: str, default: float = 0.0) -> float:
        return float(cfg.get(f"{prefix}{key}", cfg.get(key, default)) or default)

    min_width = cfg_float(f"{kind}_min_width_inches")
    min_height = cfg_float(f"{kind}_min_height_inches")
    min_area = cfg_float(f"{kind}_min_area_inches")
    width_fraction = cfg_float(f"{kind}_min_slot_width_fraction")
    height_fraction = cfg_float(f"{kind}_min_slot_height_fraction")

    if lane_w > 0 and width_fraction > 0:
        min_width = max(min_width, lane_w * width_fraction)
    if lane_h > 0 and height_fraction > 0:
        min_height = max(min_height, lane_h * height_fraction)

    required_width = min_width
    if min_height > 0:
        required_width = max(required_width, min_height * aspect)
    if min_area > 0:
        required_width = max(required_width, math.sqrt(min_area * aspect))

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "visual_type": kind,
        "aspect_ratio": round(aspect, 4),
        "min_width": round(min_width, 4),
        "min_height": round(min_height, 4),
        "min_area": round(min_area, 4),
        "required_width": round(required_width, 4),
    }


def enforce_visual_footprint(
    visual_id: Any,
    width: float,
    height: float,
    max_width: float,
    lane: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[float, float, Dict[str, Any]]:
    assets = state.get("visual_assets") or {}
    asset = assets.get(str(visual_id)) or {}
    lane = _lane_with_orientation(lane, {}, state)
    requirements = visual_requirements(visual_id, asset, lane, config)
    if not requirements["enabled"]:
        return width, height, {"enabled": False}

    aspect = max(float(width) / max(float(height), 0.01), 0.2)
    required_width = float(requirements["required_width"])
    target_width = min(max(float(width), required_width), max(float(max_width), 0.1))
    target_height = target_width / aspect

    report = _evaluate_size(
        visual_id=visual_id,
        width=target_width,
        height=target_height,
        lane=_lane_with_orientation(lane, {}, state),
        requirements=requirements,
    )
    report["requested_width"] = round(float(width), 4)
    report["max_width"] = round(float(max_width), 4)
    report["adjusted"] = target_width > float(width) + 1e-6
    return target_width, target_height, report


def evaluate_visual_footprints(
    layout: Iterable[Dict[str, Any]],
    template_layout: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = visual_footprint_config(config)
    if not cfg.get("enabled", True):
        return {"enabled": False, "checks": [], "violations": []}

    lanes = template_layout.get("lanes") or []
    lane_map = {str(lane.get("id") or ""): lane for lane in lanes if lane.get("id")}
    assets = state.get("visual_assets") or {}
    checks: List[Dict[str, Any]] = []

    for element in layout:
        if element.get("type") != "visual":
            continue
        visual_id = str(element.get("visual_id") or element.get("id") or "")
        lane = _lane_for_element(element, lane_map)
        lane = _lane_with_orientation(lane, template_layout, state)
        requirements = visual_requirements(visual_id, assets.get(visual_id) or {}, lane, config)
        check = _evaluate_size(
            visual_id=visual_id,
            width=float(element.get("width", 0.0) or 0.0),
            height=float(element.get("height", 0.0) or 0.0),
            lane=lane,
            requirements=requirements,
        )
        check.update(
            {
                "element_id": element.get("id"),
                "section_id": element.get("section_id"),
                "slot_id": element.get("lane_id") or element.get("slot_id"),
            }
        )
        checks.append(check)

    violations = [check for check in checks if not check.get("ok", False)]
    return {
        "enabled": True,
        "checks": checks,
        "violations": violations,
        "violation_count": len(violations),
    }


def visual_slot_is_feasible(
    visual_id: Any,
    lane: Optional[Dict[str, Any]],
    visual_assets: Dict[str, Any],
    config: Dict[str, Any],
    *,
    max_width: Optional[float] = None,
) -> bool:
    if not lane:
        return False
    asset = visual_assets.get(str(visual_id)) or {}
    requirements = visual_requirements(visual_id, asset, lane, config)
    if not requirements["enabled"]:
        return True
    aspect = float(requirements["aspect_ratio"])
    width_limit = max_width if max_width is not None else float(lane.get("w", 0.0) or 0.0)
    required_width = float(requirements["required_width"])
    required_height = required_width / max(aspect, 0.2)
    cfg = visual_footprint_config(config)
    max_height_fraction = float(cfg.get("feasibility_max_slot_height_fraction", 0.78) or 0.78)
    return (
        required_width <= max(float(width_limit), 0.0) + 0.05
        and required_height <= max(float(lane.get("h", 0.0) or 0.0) * max_height_fraction, 0.0) + 0.05
    )


def _evaluate_size(
    *,
    visual_id: Any,
    width: float,
    height: float,
    lane: Optional[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    min_width = float(requirements.get("min_width") or 0.0)
    min_height = float(requirements.get("min_height") or 0.0)
    min_area = float(requirements.get("min_area") or 0.0)
    area = max(width, 0.0) * max(height, 0.0)
    tolerance = 0.01
    width_ok = width + tolerance >= min_width
    height_ok = height + tolerance >= min_height
    area_ok = area + tolerance >= min_area
    ok = width_ok and height_ok and area_ok
    return {
        "visual_id": str(visual_id or ""),
        "visual_type": requirements.get("visual_type"),
        "lane_id": str((lane or {}).get("id") or ""),
        "width": round(width, 4),
        "height": round(height, 4),
        "area": round(area, 4),
        "min_width": round(min_width, 4),
        "min_height": round(min_height, 4),
        "min_area": round(min_area, 4),
        "required_width": requirements.get("required_width"),
        "ok": ok,
        "failed_dimensions": [
            name
            for name, passed in {
                "width": width_ok,
                "height": height_ok,
                "area": area_ok,
            }.items()
            if not passed
        ],
    }


def _lane_for_element(
    element: Dict[str, Any],
    lane_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    lane_id = str(element.get("lane_id") or element.get("slot_id") or "")
    if lane_id in lane_map:
        return lane_map[lane_id]
    x = float(element.get("x", 0.0) or 0.0)
    y = float(element.get("y", 0.0) or 0.0)
    for lane in lane_map.values():
        lane_x = float(lane.get("x", 0.0) or 0.0)
        lane_y = float(lane.get("y", 0.0) or 0.0)
        if lane_x - 0.05 <= x <= lane_x + float(lane.get("w", 0.0) or 0.0) + 0.05:
            if lane_y - 0.05 <= y <= lane_y + float(lane.get("h", 0.0) or 0.0) + 0.05:
                return lane
    return None


def _lane_with_orientation(
    lane: Optional[Dict[str, Any]],
    template_layout: Dict[str, Any],
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not lane:
        return lane
    enriched = dict(lane)
    orientation = (
        template_layout.get("orientation")
        or ("portrait" if float(state.get("poster_height", 0.0) or 0.0) > float(state.get("poster_width", 0.0) or 0.0) else "landscape")
    )
    enriched.setdefault("poster_orientation", orientation)
    return enriched
