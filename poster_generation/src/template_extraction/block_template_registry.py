from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_PORTRAIT_WIDTH = 36.0
DEFAULT_LANDSCAPE_WIDTH = 54.0
DEFAULT_TEMPLATE_LIBRARY_DIRS = (
    ("landscape", "templates/landscape"),
    ("portrait", "templates/portrait"),
)
SOFT_GEOMETRY_TEMPLATES: set[str] = set()


@dataclass(frozen=True)
class TemplateRecord:
    template_id: str
    bare_id: str
    orientation_hint: Optional[str]
    path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_template_dir() -> Path:
    return repo_root() / DEFAULT_TEMPLATE_LIBRARY_DIRS[0][1]


def default_template_picture_dir() -> Path:
    return repo_root() / DEFAULT_TEMPLATE_LIBRARY_DIRS[0][1]


def default_template_dirs() -> List[Path]:
    return [repo_root() / dirname for _, dirname in DEFAULT_TEMPLATE_LIBRARY_DIRS]


def _default_template_records() -> List[TemplateRecord]:
    records: List[TemplateRecord] = []
    for orientation, dirname in DEFAULT_TEMPLATE_LIBRARY_DIRS:
        root = repo_root() / dirname
        if not root.exists():
            continue
        for path in sorted(root.glob("cluster_*_template.json")):
            bare_id = path.stem.replace("_template", "")
            records.append(
                TemplateRecord(
                    template_id=f"{bare_id}_{orientation}",
                    bare_id=bare_id,
                    orientation_hint=orientation,
                    path=path,
                )
            )
    return records


def _template_records(template_dir: Optional[Path] = None) -> List[TemplateRecord]:
    if template_dir is None:
        return _default_template_records()

    root = template_dir
    if not root.exists():
        return []
    return [
        TemplateRecord(
            template_id=path.stem.replace("_template", ""),
            bare_id=path.stem.replace("_template", ""),
            orientation_hint=None,
            path=path,
        )
        for path in sorted(root.glob("cluster_*_template.json"))
    ]


def iter_block_template_files(template_dir: Optional[Path] = None) -> Iterable[Path]:
    return [record.path for record in _template_records(template_dir)]


def list_block_template_ids(template_dir: Optional[Path] = None) -> List[str]:
    return [record.template_id for record in _template_records(template_dir)]


def is_block_template_id(template_id: str) -> bool:
    return _find_template_record(template_id) is not None


def load_block_template_raw(template_id: str, template_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    record = _find_template_record(template_id, template_dir=template_dir)
    if not record:
        return None
    try:
        data = json.loads(record.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    data["_template_path"] = str(record.path)
    data["_template_id"] = record.template_id
    data["_template_bare_id"] = record.bare_id
    data["_template_orientation_hint"] = record.orientation_hint
    image_path = _template_image_path(record.template_id, data)
    if image_path.exists():
        data["_template_image_path"] = str(image_path)
    return data


def _find_template_record(template_id: str, template_dir: Optional[Path] = None) -> Optional[TemplateRecord]:
    requested = str(template_id or "").strip()
    if not requested:
        return None

    records = _template_records(template_dir)
    for record in records:
        if record.template_id == requested:
            return record

    bare_matches = [record for record in records if record.bare_id == requested]
    if len(bare_matches) == 1:
        return bare_matches[0]
    return None
    return None


def get_block_template_info(template_id: str, template_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    raw = load_block_template_raw(template_id, template_dir=template_dir)
    if not raw:
        return None
    aspect_ratio = _template_aspect_ratio(raw, template_id)
    orientation = "portrait" if aspect_ratio < 1.0 else "landscape"
    if orientation == "portrait":
        width = DEFAULT_PORTRAIT_WIDTH
        height = width / max(aspect_ratio, 1e-6)
    else:
        width = DEFAULT_LANDSCAPE_WIDTH
        height = width / max(aspect_ratio, 1e-6)
    return {
        "template_id": raw.get("_template_id") or template_id,
        "source_template_id": raw.get("_template_bare_id") or template_id,
        "orientation": orientation,
        "aspect_ratio": aspect_ratio,
        "recommended_canvas_size": {
            "width": round(width, 3),
            "height": round(height, 3),
        },
        "slot_count": int(raw.get("num_slots") or len(raw.get("slots") or [])),
        "source_template_path": raw.get("_template_path"),
        "source_image_path": raw.get("_template_image_path"),
    }


def load_block_template_layout(
    template_id: str,
    page_width: float,
    page_height: float,
    *,
    margin: float = 1.0,
) -> Optional[Dict[str, Any]]:
    raw = load_block_template_raw(template_id)
    if not raw:
        return None
    return build_runtime_template(raw, raw.get("_template_id") or template_id, page_width, page_height, margin=margin)


def build_runtime_template(
    raw: Dict[str, Any],
    template_id: str,
    page_width: float,
    page_height: float,
    *,
    margin: float = 1.0,
) -> Dict[str, Any]:
    slots = raw.get("slots") or []
    source_slots = [_slot_from_raw(slot) for slot in slots]
    if not source_slots:
        raise ValueError(f"template '{template_id}' has no slots")

    template_aspect_ratio = _template_aspect_ratio(raw, template_id)
    template_orientation = "portrait" if template_aspect_ratio < 1.0 else "landscape"
    source_width, source_height = _normalized_coordinate_extent(source_slots)
    norm_slots = [_normalize_slot(slot, source_width, source_height) for slot in source_slots]

    header_slot = _identify_header_slot(norm_slots)
    content_slots = [slot for slot in norm_slots if slot["slot_id"] != header_slot["slot_id"]]
    ordered_content_slots = _order_content_slots(content_slots)
    semantic_slots = _decorate_content_slots(ordered_content_slots)

    inner_w = max(page_width - 2 * margin, 0.1)
    inner_h = max(page_height - 2 * margin, 0.1)

    scaled_header = _scale_box(header_slot, inner_w, inner_h, margin, margin)
    scaled_content_slots = [
        _scale_box(slot, inner_w, inner_h, margin, margin)
        for slot in semantic_slots
    ]
    scaled_content_slots, gap_absorption_report = _apply_soft_gap_absorption(
        template_id,
        scaled_content_slots,
        page_width,
        page_height,
        margin,
        header_slot=scaled_header,
    )

    adjacency_graph = _build_adjacency_graph(semantic_slots)
    slot_prominence = {
        slot["slot_id"]: slot["prominence_score"]
        for slot in semantic_slots
    }
    ordered_by_prominence = sorted(
        semantic_slots,
        key=lambda slot: (-float(slot.get("prominence_score", 0.0)), float(slot.get("y", 0.0)), float(slot.get("x", 0.0))),
    )
    hero_region_id = ordered_by_prominence[0]["slot_id"]
    primary_region_ids = [slot["slot_id"] for slot in ordered_by_prominence[: max(2, min(3, len(ordered_by_prominence)))]]
    secondary_region_ids = [slot["slot_id"] for slot in ordered_by_prominence if slot["slot_id"] not in primary_region_ids]
    density_profile = _template_density_profile(semantic_slots)
    regions = _build_regions(scaled_content_slots, hero_region_id, primary_region_ids)

    return {
        "template_name": template_id,
        "template_id": template_id,
        "layout_mode": "template_prior",
        "header_slot": scaled_header,
        "header_region": scaled_header,
        "header": {
            "x": scaled_header["x"],
            "y": scaled_header["y"],
            "w": scaled_header["w"],
            "h": scaled_header["h"],
        },
        "content_slots": scaled_content_slots,
        "slot_count": len(scaled_content_slots),
        "lanes": scaled_content_slots,
        "columns": scaled_content_slots,
        "slot_order": [slot["slot_id"] for slot in scaled_content_slots],
        "normalized_slots": semantic_slots,
        "adjacency_graph": adjacency_graph,
        "slot_prominence_score": slot_prominence,
        "orientation": template_orientation,
        "template_aspect_ratio": template_aspect_ratio,
        "recommended_canvas_size": _recommended_canvas_size(raw, template_id),
        "regions": regions,
        "hero_region_id": hero_region_id,
        "primary_regions": [region for region in regions if region["region_id"] in primary_region_ids],
        "secondary_regions": [region for region in regions if region["region_id"] in secondary_region_ids],
        "recommended_visual_anchor": hero_region_id,
        "template_density_profile": density_profile,
        "style_tokens": {
            "background": "#FFFFFF",
            "header_background": "#FFFFFF",
        },
        "panel_style_tokens": {},
        "logo_regions": [],
        "footer": None,
        "visual_width_cap": None,
        "raw_num_posters": raw.get("num_posters"),
        "occupancy_heatmap": raw.get("occupancy_heatmap"),
        "gap_absorption_report": gap_absorption_report,
        "source_template_path": raw.get("_template_path"),
        "source_template_id": raw.get("_template_bare_id") or template_id,
        "source_image_path": raw.get("_template_image_path"),
        "template_prior": True,
    }


def _recommended_canvas_size(raw: Dict[str, Any], template_id: str) -> Dict[str, float]:
    aspect_ratio = _template_aspect_ratio(raw, template_id)
    orientation = "portrait" if aspect_ratio < 1.0 else "landscape"
    if orientation == "portrait":
        width = DEFAULT_PORTRAIT_WIDTH
        height = width / max(aspect_ratio, 1e-6)
    else:
        width = DEFAULT_LANDSCAPE_WIDTH
        height = width / max(aspect_ratio, 1e-6)
    return {"width": round(width, 3), "height": round(height, 3)}


def _template_image_path(template_id: str, raw: Optional[Dict[str, Any]] = None) -> Path:
    if raw and raw.get("_template_path"):
        template_path = Path(str(raw["_template_path"]))
        bare_id = str(raw.get("_template_bare_id") or template_path.stem.replace("_template", ""))
        return template_path.with_name(f"{bare_id}_layout.png")

    record = _find_template_record(template_id)
    if record:
        return record.path.with_name(f"{record.bare_id}_layout.png")

    suffix = str(template_id or "").replace("cluster_", "")
    return default_template_picture_dir() / f"cluster_{suffix}_layout.png"


def _apply_soft_gap_absorption(
    template_id: str,
    slots: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    margin: float,
    *,
    header_slot: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    area_before = sum(float(slot.get("w", 0.0)) * float(slot.get("h", 0.0)) for slot in slots)
    report = {
        "enabled": False,
        "template_id": template_id,
        "mode": "equal_split",
        "rule": "adjacent content block gaps are split at the midpoint",
        "area_before": round(area_before, 4),
        "area_after": round(area_before, 4),
        "total_area_gain": 0.0,
        "absorptions": [],
        "edge_expansions": [],
    }
    if template_id not in SOFT_GEOMETRY_TEMPLATES and not _soft_geometry_enabled_for_template(template_id):
        return slots, report

    min_size = 0.25
    min_header_clearance = 0.35
    min_edge_gain = 0.05
    content_top_bound = margin
    if header_slot:
        header_bottom = float(header_slot.get("y", 0.0)) + float(header_slot.get("h", 0.0))
        content_top_bound = max(content_top_bound, header_bottom + min_header_clearance)
    max_right = page_width - margin
    max_bottom = page_height - margin
    overlap_threshold = 0.25
    originals = [dict(slot) for slot in slots]
    expanded = {str(slot.get("slot_id") or slot.get("id")): dict(slot) for slot in slots}

    def sid(slot: Dict[str, Any]) -> str:
        return str(slot.get("slot_id") or slot.get("id") or "")

    def horizontal_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        left = max(float(a["x"]), float(b["x"]))
        right = min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"]))
        return max(right - left, 0.0)

    def vertical_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        top = max(float(a["y"]), float(b["y"]))
        bottom = min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"]))
        return max(bottom - top, 0.0)

    def right_edge(slot: Dict[str, Any]) -> float:
        return float(slot["x"]) + float(slot["w"])

    def bottom_edge(slot: Dict[str, Any]) -> float:
        return float(slot["y"]) + float(slot["h"])

    def has_overlap(candidate_slots: List[Dict[str, Any]]) -> bool:
        for index, current in enumerate(candidate_slots):
            cx1, cy1 = float(current["x"]), float(current["y"])
            cx2, cy2 = right_edge(current), bottom_edge(current)
            for other in candidate_slots[index + 1:]:
                ox1, oy1 = float(other["x"]), float(other["y"])
                ox2, oy2 = right_edge(other), bottom_edge(other)
                if min(cx2, ox2) - max(cx1, ox1) > 0.01 and min(cy2, oy2) - max(cy1, oy1) > 0.01:
                    return True
        return False

    def record_absorption(
        first_id: str,
        second_id: str,
        orientation: str,
        gap: float,
        boundary: float,
        first_gain: float,
        second_gain: float,
    ) -> None:
        report["absorptions"].append({
            "affected_slot_ids": [first_id, second_id],
            "orientation": orientation,
            "original_gap_inches": round(gap, 4),
            "split_boundary": round(boundary, 4),
            "left_or_upper_gain": round(max(first_gain, 0.0), 4),
            "right_or_lower_gain": round(max(second_gain, 0.0), 4),
        })

    def record_edge_expansion(
        slot_id: str,
        edge: str,
        original_gap: float,
        absorbed: float,
        new_boundary: float,
    ) -> None:
        if absorbed <= min_edge_gain:
            return
        report["edge_expansions"].append({
            "slot_id": slot_id,
            "edge": edge,
            "original_gap_inches": round(original_gap, 4),
            "absorbed_inches": round(absorbed, 4),
            "new_boundary": round(new_boundary, 4),
        })

    horizontal_pairs = []
    vertical_pairs = []
    for first in originals:
        first_id = sid(first)
        if not first_id:
            continue
        for second in originals:
            second_id = sid(second)
            if not second_id or second_id == first_id:
                continue
            gap_x = float(second["x"]) - right_edge(first)
            if gap_x > 0.01:
                overlap = vertical_overlap(first, second)
                min_h = max(min(float(first["h"]), float(second["h"])), 0.01)
                if overlap / min_h >= overlap_threshold:
                    horizontal_pairs.append((gap_x, first_id, second_id, first, second))

            gap_y = float(second["y"]) - bottom_edge(first)
            if gap_y > 0.01:
                overlap = horizontal_overlap(first, second)
                min_w = max(min(float(first["w"]), float(second["w"])), 0.01)
                if overlap / min_w >= overlap_threshold:
                    vertical_pairs.append((gap_y, first_id, second_id, first, second))

    claimed_right_edges: set[str] = set()
    claimed_left_edges: set[str] = set()
    for gap, left_id, right_id, left, right in sorted(horizontal_pairs, key=lambda item: item[0]):
        if left_id in claimed_right_edges or right_id in claimed_left_edges:
            continue
        left_right = right_edge(left)
        right_left = float(right["x"])
        boundary = (left_right + right_left) / 2
        boundary = min(max(boundary, float(left["x"]) + min_size), right_edge(right) - min_size, max_right)
        if boundary <= left_right + 0.01 or boundary >= right_left - 0.01:
            continue

        trial = [dict(slot) for slot in expanded.values()]
        trial_by_id = {sid(slot): slot for slot in trial}
        left_trial = trial_by_id[left_id]
        right_trial = trial_by_id[right_id]
        old_right_edge = right_edge(right_trial)
        left_trial["w"] = max(boundary - float(left_trial["x"]), min_size)
        right_trial["x"] = boundary
        right_trial["w"] = max(old_right_edge - boundary, min_size)
        if has_overlap(trial):
            continue

        expanded[left_id] = left_trial
        expanded[right_id] = right_trial
        expanded[left_id]["gap_absorbed"] = True
        expanded[right_id]["gap_absorbed"] = True
        claimed_right_edges.add(left_id)
        claimed_left_edges.add(right_id)
        record_absorption(
            left_id,
            right_id,
            "vertical",
            gap,
            boundary,
            boundary - left_right,
            right_left - boundary,
        )

    claimed_bottom_edges: set[str] = set()
    claimed_top_edges: set[str] = set()
    for gap, upper_id, lower_id, upper, lower in sorted(vertical_pairs, key=lambda item: item[0]):
        if upper_id in claimed_bottom_edges or lower_id in claimed_top_edges:
            continue
        upper_bottom = bottom_edge(upper)
        lower_top = float(lower["y"])
        boundary = (upper_bottom + lower_top) / 2
        boundary = min(max(boundary, float(upper["y"]) + min_size), bottom_edge(lower) - min_size, max_bottom)
        if boundary <= upper_bottom + 0.01 or boundary >= lower_top - 0.01:
            continue

        trial = [dict(slot) for slot in expanded.values()]
        trial_by_id = {sid(slot): slot for slot in trial}
        upper_trial = trial_by_id[upper_id]
        lower_trial = trial_by_id[lower_id]
        old_lower_bottom = bottom_edge(lower_trial)
        upper_trial["h"] = max(boundary - float(upper_trial["y"]), min_size)
        lower_trial["y"] = boundary
        lower_trial["h"] = max(old_lower_bottom - boundary, min_size)
        if has_overlap(trial):
            continue

        expanded[upper_id] = upper_trial
        expanded[lower_id] = lower_trial
        expanded[upper_id]["gap_absorbed"] = True
        expanded[lower_id]["gap_absorbed"] = True
        claimed_bottom_edges.add(upper_id)
        claimed_top_edges.add(lower_id)
        record_absorption(
            upper_id,
            lower_id,
            "horizontal",
            gap,
            boundary,
            boundary - upper_bottom,
            lower_top - boundary,
        )

    def has_left_neighbor(slot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
        slot_left = float(slot["x"])
        for other in candidates:
            if sid(other) == sid(slot):
                continue
            gap = slot_left - right_edge(other)
            if gap < -0.01:
                continue
            overlap = vertical_overlap(slot, other)
            min_h = max(min(float(slot["h"]), float(other["h"])), 0.01)
            if overlap / min_h >= overlap_threshold:
                return True
        return False

    def has_right_neighbor(slot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
        slot_right = right_edge(slot)
        for other in candidates:
            if sid(other) == sid(slot):
                continue
            gap = float(other["x"]) - slot_right
            if gap < -0.01:
                continue
            overlap = vertical_overlap(slot, other)
            min_h = max(min(float(slot["h"]), float(other["h"])), 0.01)
            if overlap / min_h >= overlap_threshold:
                return True
        return False

    def has_upper_neighbor(slot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
        slot_top = float(slot["y"])
        for other in candidates:
            if sid(other) == sid(slot):
                continue
            gap = slot_top - bottom_edge(other)
            if gap < -0.01:
                continue
            overlap = horizontal_overlap(slot, other)
            min_w = max(min(float(slot["w"]), float(other["w"])), 0.01)
            if overlap / min_w >= overlap_threshold:
                return True
        return False

    def has_lower_neighbor(slot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
        slot_bottom = bottom_edge(slot)
        for other in candidates:
            if sid(other) == sid(slot):
                continue
            gap = float(other["y"]) - slot_bottom
            if gap < -0.01:
                continue
            overlap = horizontal_overlap(slot, other)
            min_w = max(min(float(slot["w"]), float(other["w"])), 0.01)
            if overlap / min_w >= overlap_threshold:
                return True
        return False

    edge_slots = [dict(slot) for slot in expanded.values()]
    for slot in edge_slots:
        slot_id = sid(slot)
        if not slot_id:
            continue
        target = expanded[slot_id]
        if not has_left_neighbor(slot, edge_slots):
            old_x = float(target["x"])
            new_x = min(old_x, max(margin, 0.0))
            gain = old_x - new_x
            if gain > min_edge_gain:
                target["x"] = new_x
                target["w"] = float(target["w"]) + gain
                target["gap_absorbed"] = True
                record_edge_expansion(slot_id, "left", old_x - margin, gain, new_x)
        if not has_right_neighbor(slot, edge_slots):
            old_right = right_edge(target)
            new_right = max(old_right, max_right)
            gain = new_right - old_right
            if gain > min_edge_gain:
                target["w"] = max(new_right - float(target["x"]), min_size)
                target["gap_absorbed"] = True
                record_edge_expansion(slot_id, "right", max_right - old_right, gain, new_right)
        if not has_upper_neighbor(slot, edge_slots):
            old_y = float(target["y"])
            new_y = min(old_y, content_top_bound)
            gain = old_y - new_y
            if gain > min_edge_gain:
                target["y"] = new_y
                target["h"] = float(target["h"]) + gain
                target["gap_absorbed"] = True
                record_edge_expansion(slot_id, "top", old_y - content_top_bound, gain, new_y)
        if not has_lower_neighbor(slot, edge_slots):
            old_bottom = bottom_edge(target)
            new_bottom = max(old_bottom, max_bottom)
            gain = new_bottom - old_bottom
            if gain > min_edge_gain:
                target["h"] = max(new_bottom - float(target["y"]), min_size)
                target["gap_absorbed"] = True
                record_edge_expansion(slot_id, "bottom", max_bottom - old_bottom, gain, new_bottom)

    report["enabled"] = True
    ordered_slots = [expanded[sid(slot)] for slot in slots]
    area_after = sum(float(slot.get("w", 0.0)) * float(slot.get("h", 0.0)) for slot in ordered_slots)
    report["area_after"] = round(area_after, 4)
    report["total_area_gain"] = round(area_after - area_before, 4)
    return ordered_slots, report


def _soft_geometry_enabled_for_template(template_id: str) -> bool:
    try:
        from src.config.poster_config import load_config

        soft_config = load_config().get("soft_geometry") or {}
    except Exception:
        soft_config = {}
    if soft_config:
        if not bool(soft_config.get("enabled", False)):
            return False
        configured_templates = {str(item) for item in (soft_config.get("templates") or [])}
        if configured_templates:
            return str(template_id) in configured_templates
    return str(template_id).endswith("_landscape") or str(template_id).endswith("_portrait")


def _template_aspect_ratio(raw: Dict[str, Any], template_id: str) -> float:
    try:
        ratio = float(raw.get("aspect_ratio") or 0)
        if ratio > 0:
            return ratio
    except (TypeError, ValueError):
        pass

    image_path = _template_image_path(template_id)
    if image_path.exists():
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
            if height > 0:
                return float(width) / float(height)
        except Exception:
            pass
    return 1.0


def _normalized_coordinate_extent(slots: List[Dict[str, Any]]) -> Tuple[float, float]:
    max_x = max(slot["x"] + slot["w"] for slot in slots)
    max_y = max(slot["y"] + slot["h"] for slot in slots)
    # The cluster template JSONs use a canonical 0..1000 coordinate frame,
    # independent from the real portrait image aspect ratio. The aspect ratio
    # must be applied by the PPT canvas size, not by stretching these bboxes.
    source_width = 1000.0 if max_x <= 1005.0 else max_x
    source_height = 1000.0 if max_y <= 1005.0 else max_y
    return source_width, source_height


def _slot_from_raw(slot: Dict[str, Any]) -> Dict[str, Any]:
    bbox = slot.get("bbox") or [0, 0, 1, 1]
    if isinstance(bbox, dict):
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("w", bbox.get("width", 1.0)))
        h = float(bbox.get("h", bbox.get("height", 1.0)))
    else:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return {
        "slot_id": f"slot_{slot.get('slot_id')}",
        "x": x,
        "y": y,
        "w": max(w, 1.0),
        "h": max(h, 1.0),
        "frequency": float(slot.get("frequency", 1.0)),
        "polygon": slot.get("polygon"),
    }


def _normalize_slot(slot: Dict[str, Any], source_width: float, source_height: float) -> Dict[str, Any]:
    return {
        **slot,
        "x": slot["x"] / max(source_width, 1.0),
        "y": slot["y"] / max(source_height, 1.0),
        "w": slot["w"] / max(source_width, 1.0),
        "h": slot["h"] / max(source_height, 1.0),
        "area_ratio": (slot["w"] * slot["h"]) / max(source_width * source_height, 1.0),
    }


def _identify_header_slot(slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_width = max(slot["w"] for slot in slots)

    def score(slot: Dict[str, Any]) -> Tuple[float, float, float]:
        width_score = slot["w"] / max(max_width, 1e-6)
        return (-slot["y"], width_score, -slot["h"])

    return max(slots, key=score)


def _order_content_slots(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(slots, key=lambda slot: (round(slot["y"], 4), round(slot["x"], 4), -slot["area_ratio"]))


def _decorate_content_slots(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    decorated: List[Dict[str, Any]] = []
    max_area = max((slot["area_ratio"] for slot in slots), default=1.0)
    for slot in slots:
        center_x = slot["x"] + slot["w"] / 2
        center_y = slot["y"] + slot["h"] / 2
        topness = 1.0 - min(center_y, 1.0)
        centrality = 1.0 - min(abs(center_x - 0.5) / 0.5, 1.0)
        area_weight = slot["area_ratio"] / max(max_area, 1e-6)
        prominence = round(area_weight * 0.55 + topness * 0.25 + centrality * 0.20, 4)
        semantic_lane = "left" if center_x < 0.34 else "middle" if center_x < 0.67 else "right"
        vertical_band = "top" if center_y < 0.34 else "middle" if center_y < 0.67 else "bottom"
        decorated.append({
            **slot,
            "prominence_score": prominence,
            "semantic_lane": semantic_lane,
            "vertical_band": vertical_band,
        })
    return decorated


def _scale_box(slot: Dict[str, Any], inner_w: float, inner_h: float, offset_x: float, offset_y: float) -> Dict[str, Any]:
    scaled = {
        "id": slot["slot_id"],
        "slot_id": slot["slot_id"],
        "x": offset_x + slot["x"] * inner_w,
        "y": offset_y + slot["y"] * inner_h,
        "w": slot["w"] * inner_w,
        "h": slot["h"] * inner_h,
        "area_ratio": slot.get("area_ratio", 0.0),
        "prominence_score": slot.get("prominence_score", 0.0),
        "semantic_lane": slot.get("semantic_lane"),
        "vertical_band": slot.get("vertical_band"),
        "frequency": slot.get("frequency", 1.0),
        "template_block_slot": True,
    }
    return scaled


def _build_adjacency_graph(slots: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adjacency: Dict[str, List[Dict[str, Any]]] = {slot["slot_id"]: [] for slot in slots}
    for idx, slot in enumerate(slots):
        for other in slots[idx + 1:]:
            relation = _shared_edge(slot, other)
            if not relation:
                continue
            adjacency[slot["slot_id"]].append({
                "slot_id": other["slot_id"],
                **relation,
            })
            adjacency[other["slot_id"]].append({
                "slot_id": slot["slot_id"],
                **relation,
            })
    return adjacency


def _shared_edge(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tol = 0.03
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    if abs(ax2 - bx1) <= tol or abs(bx2 - ax1) <= tol:
        overlap = min(ay2, by2) - max(ay1, by1)
        if overlap > tol:
            return {"orientation": "vertical", "shared_span": round(overlap, 4)}
    if abs(ay2 - by1) <= tol or abs(by2 - ay1) <= tol:
        overlap = min(ax2, bx2) - max(ax1, bx1)
        if overlap > tol:
            return {"orientation": "horizontal", "shared_span": round(overlap, 4)}
    return None


def _build_regions(
    scaled_slots: List[Dict[str, Any]],
    hero_region_id: str,
    primary_region_ids: List[str],
) -> List[Dict[str, Any]]:
    regions = []
    max_area = max((float(slot.get("area_ratio", 0.0) or 0.0) for slot in scaled_slots), default=0.0)
    for rank, slot in enumerate(
        sorted(
            scaled_slots,
            key=lambda region: (-float(region.get("prominence_score", 0.0)), float(region.get("y", 0.0)), float(region.get("x", 0.0))),
        ),
        start=1,
    ):
        area_ratio = float(slot.get("area_ratio", 0.0) or 0.0)
        slot_w = float(slot.get("w", 0.0) or 0.0)
        slot_h = float(slot.get("h", 0.0) or 0.0)
        relative_large = max_area > 0 and area_ratio >= max_area * 0.58
        can_host_visual = (
            slot_h >= 5.0
            and slot_w >= 10.0
            and (area_ratio >= 0.06 or relative_large or slot["slot_id"] in primary_region_ids)
        )
        regions.append({
            **slot,
            "region_id": slot["slot_id"],
            "region_rank": rank,
            "region_tier": "primary" if slot["slot_id"] in primary_region_ids else "secondary",
            "can_host_visual": can_host_visual,
            "text_density_limit": "high" if area_ratio >= 0.18 else "medium" if area_ratio >= 0.09 else "low",
            "is_hero_region": slot["slot_id"] == hero_region_id,
        })
    return sorted(regions, key=lambda region: (float(region.get("y", 0.0)), float(region.get("x", 0.0))))


def _template_density_profile(slots: List[Dict[str, Any]]) -> str:
    if not slots:
        return "balanced"
    ordered = sorted(slots, key=lambda slot: float(slot.get("area_ratio", 0.0)), reverse=True)
    largest = float(ordered[0].get("area_ratio", 0.0))
    if largest >= 0.20:
        return "hero_wide"
    if len(ordered) >= 2 and float(ordered[0].get("area_ratio", 0.0)) + float(ordered[1].get("area_ratio", 0.0)) >= 0.32:
        return "dual_primary"
    return "balanced"
